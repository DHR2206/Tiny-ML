"""Pre-synthesised speech inventory.

Every sentence Buddy can say is rendered once (at setup) to
<cache>/<voice-key>/<sha1(text)>.wav at the output sample rate.  At runtime the
device only *plays* files; no text-to-speech model has to run while the child
waits, which cuts response latency and CPU use on the Raspberry Pi.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..audio.dsp import fade, read_wav, resample, write_wav


def key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:20]


class SpeechCache:
    def __init__(self, root: Path, voice_key: str, rate: int,
                 synth: Optional[Callable[[str], tuple[np.ndarray, int]]] = None):
        self.dir = Path(root) / voice_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.synth = synth  # runtime fallback (may be None)
        self._mem: dict[str, np.ndarray] = {}

    def path(self, text: str) -> Path:
        return self.dir / f"{key(text)}.wav"

    def has(self, text: str) -> bool:
        return self.path(text).exists()

    def render(self, text: str) -> Path:
        if self.synth is None:
            raise RuntimeError("no synthesizer available")
        x, sr = self.synth(text)
        x = fade(resample(x, sr, self.rate), self.rate, 5)
        p = self.path(text)
        write_wav(p, x, self.rate)
        return p

    def get(self, text: str) -> Optional[np.ndarray]:
        k = key(text)
        if k in self._mem:
            return self._mem[k]
        p = self.dir / f"{k}.wav"
        if not p.exists():
            if self.synth is None:
                return None
            self.render(text)
        x, sr = read_wav(p)
        if sr != self.rate:
            x = resample(x, sr, self.rate)
        if len(self._mem) < 64:  # keep short phrases hot in RAM
            self._mem[k] = x
        return x

    def missing(self, texts: Iterable[str]) -> list[str]:
        return [t for t in texts if not self.has(t)]
