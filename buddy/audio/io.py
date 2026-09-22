"""Microphone capture and speaker playback (sounddevice / PortAudio).

Privacy: captured audio lives only in a small in-memory queue.  Nothing is ever
written to disk, and every buffer is overwritten after it has been consumed.
Muting *closes* the input stream, so the microphone is really off.
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from .dsp import resample, wipe

ASR_RATE = 16000


def _sd():
    import sounddevice as sd  # imported lazily: needs libportaudio2
    return sd


def _device(spec):
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    sd = _sd()
    for i, d in enumerate(sd.query_devices()):
        if str(spec).lower() in d["name"].lower():
            return i
    return None


class Microphone:
    def __init__(self, device=None, rate: Optional[int] = None, block_ms: int = 32):
        self.device = _device(device)
        self.rate = rate
        self.block_ms = block_ms
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self.stream = None

    def start(self) -> None:
        if self.stream is not None:
            return
        sd = _sd()
        rate = self.rate or int(sd.query_devices(self.device, "input")["default_samplerate"])
        self._rate = rate

        def cb(indata, frames, t, status):  # noqa: ARG001
            x = indata[:, 0].astype(np.float32).copy()
            try:
                self.q.put_nowait(x)
            except queue.Full:
                pass

        self.stream = sd.InputStream(device=self.device, channels=1, samplerate=rate, dtype="float32",
                                     blocksize=int(rate * self.block_ms / 1000), callback=cb)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        while not self.q.empty():  # drop and wipe anything still queued
            wipe(self.q.get_nowait())

    @property
    def active(self) -> bool:
        return self.stream is not None

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            x = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        y = resample(x, self._rate, ASR_RATE)
        wipe(x)
        return y


class Speaker:
    """Non-blocking player: queue audio, stop instantly, software volume."""

    def __init__(self, device=None, rate: int = 48000, volume: float = 0.7):
        self.device = _device(device)
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self.stream = None

    def start(self) -> None:
        sd = _sd()

        def cb(outdata, frames, t, status):  # noqa: ARG001
            with self._lock:
                chunk = self._buf[self._pos:self._pos + frames]
                self._pos += len(chunk)
                vol = self.volume
            out = np.zeros(frames, dtype=np.float32)
            out[:len(chunk)] = chunk * vol
            outdata[:, 0] = out

        self.stream = sd.OutputStream(device=self.device, channels=1, samplerate=self.rate, dtype="float32",
                                      callback=cb, blocksize=1024)
        self.stream.start()

    def play(self, x: np.ndarray) -> None:
        with self._lock:
            rest = self._buf[self._pos:]
            self._buf = np.concatenate([rest, x.astype(np.float32)])
            self._pos = 0

    def stop(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._pos = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pos < len(self._buf)

    def remaining(self) -> float:
        with self._lock:
            return max(0, len(self._buf) - self._pos) / self.rate

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
