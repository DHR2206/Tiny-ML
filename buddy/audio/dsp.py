"""Small numpy-only audio helpers (no scipy needed on the Pi)."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Band-limited resampling with a windowed-sinc FIR + linear interpolation.

    Good enough for speech; exact for integer up-sampling ratios after filtering.
    """
    x = np.asarray(x, dtype=np.float32)
    if src == dst or len(x) == 0:
        return x
    if dst < src:  # anti-alias low-pass before decimation
        cutoff = 0.5 * dst / src
        taps = 63
        n = np.arange(taps) - (taps - 1) / 2
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        h /= h.sum()
        x = np.convolve(x, h.astype(np.float32), mode="same")
    n_out = int(round(len(x) * dst / src))
    t = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def fade(x: np.ndarray, rate: int, ms: float = 8.0) -> np.ndarray:
    n = min(len(x) // 2, int(rate * ms / 1000))
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def earcon(kind: str, rate: int) -> np.ndarray:
    """Friendly synthetic chimes (no sample files needed)."""
    level = {"ready": 0.18}.get(kind, 0.35)  # "ready" = soft "I'm still listening" chime after a task
    notes = {
        "listen": [(880, 0.09), (1175, 0.14)],
        "correct": [(784, 0.08), (988, 0.08), (1319, 0.18)],
        "wrong": [(392, 0.15)],
        "mute": [(988, 0.1), (659, 0.16)],
        "unmute": [(659, 0.1), (988, 0.16)],
        "end": [(1175, 0.08), (880, 0.12)],
        "ready": [(988, 0.07), (1319, 0.11)],
    }.get(kind, [(880, 0.1)])
    out = []
    for f, d in notes:
        t = np.arange(int(rate * d)) / rate
        env = np.exp(-t * 18) * np.minimum(1, t * 400)
        tone = level * env * (np.sin(2 * math.pi * f * t) + 0.3 * np.sin(4 * math.pi * f * t))
        out.append(tone.astype(np.float32))
    return np.concatenate(out)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def write_wav(path: str | Path, x: np.ndarray, rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def wipe(buf: np.ndarray) -> None:
    """Overwrite an audio buffer in place (privacy: no child audio lingers in RAM)."""
    try:
        buf.fill(0)
    except (ValueError, AttributeError):
        pass
