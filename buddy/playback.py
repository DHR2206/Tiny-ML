"""Speech output: turns a Response into audio and plays it one item (sentence) at a time.

Playing sentence by sentence, instead of one long buffer, is what makes pause, continue
and repeat possible: the player always knows which sentence the child is hearing, so a
paused reply can later be resumed from the start of that sentence.  It also means Buddy
starts talking as soon as the first sentence is ready, even when later sentences still
have to be synthesised (runtime TTS fallback).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio.dsp import earcon, read_wav, resample, silence
from .core import Earcon, Item, Pause, Response, Say, Sound
from .speech.cache import SpeechCache

log = logging.getLogger("buddy")
EMPTY = np.zeros(0, dtype=np.float32)


class Renderer:
    """Turns Response items into audio at the output rate."""

    def __init__(self, cache: SpeechCache, rate: int, sounds_dir: Optional[str]):
        self.cache = cache
        self.rate = rate
        self.sounds = Path(sounds_dir).expanduser() if sounds_dir else None

    def render_item(self, it: Item) -> tuple[np.ndarray, int]:
        """(audio incl. the pause after it, number of samples that are actual sound)."""
        if isinstance(it, Say):
            x = self.cache.get(it.text)
            if x is None:
                log.warning("missing speech clip (run build-content): %s", it.text[:60])
                return EMPTY, 0
            return np.concatenate([x, silence(it.pause_after, self.rate)]), len(x)
        if isinstance(it, Pause):
            return silence(it.seconds, self.rate), 0
        if isinstance(it, Earcon):
            e = earcon(it.kind, self.rate)
            return np.concatenate([e, silence(0.1, self.rate)]), len(e)
        if isinstance(it, Sound) and self.sounds:
            p = self.sounds / f"{it.name}.wav"
            if p.exists():
                x, sr = read_wav(p)
                x = resample(x[: sr * 6], sr, self.rate)  # at most 6 s
                x = 0.8 * x / (np.abs(x).max() + 1e-6)
                return np.concatenate([x, silence(0.3, self.rate)]), len(x)
        return EMPTY, 0

    def render(self, r: Response) -> np.ndarray:
        """The whole Response as one buffer (used by tools and tests)."""
        parts = [a for a, _ in (self.render_item(it) for it in r.items) if len(a)]
        return np.concatenate(parts) if parts else EMPTY.copy()


class Player:
    """Feeds one Response to the speaker item by item and tracks what is being heard."""

    LOOKAHEAD = 0.5  # seconds of audio kept queued in the speaker (keeps playback gapless)

    def __init__(self, renderer: Renderer, speaker, on_audio: Optional[Callable[[float], None]] = None):
        self.renderer = renderer
        self.speaker = speaker
        self.on_audio = on_audio  # called with the seconds of audio queued (usage time)
        self._clear()

    def _clear(self) -> None:
        self.response: Optional[Response] = None
        self.items: list[Item] = []
        self.next = 0  # next item to queue
        self.spans: list[tuple[int, int, int]] = []  # (item index, end of its sound, end of its audio) in samples
        self.total = 0  # samples queued for this Response

    @property
    def active(self) -> bool:
        """True from start() until the Response has been completely heard (or stopped)."""
        return self.response is not None

    def start(self, response: Response, start_index: int = 0) -> None:
        self.stop()
        self.response = response
        self.items = list(response.items)
        self.next = max(0, start_index)

    def stop(self) -> None:
        if self.response is not None:
            self.speaker.stop()
        self._clear()

    def _queued_ahead(self) -> float:
        rem = getattr(self.speaker, "remaining", None)
        if rem is not None:
            return rem()
        return self.LOOKAHEAD if self.speaker.busy else 0.0

    def tick(self) -> Optional[Response]:
        """Keep the speaker fed.  Returns the Response once it has completely finished playing."""
        if self.response is None:
            return None
        while self.next < len(self.items) and self._queued_ahead() < self.LOOKAHEAD:
            idx = self.next
            self.next += 1
            audio, sound_len = self.renderer.render_item(self.items[idx])
            if not len(audio):
                continue
            self.spans.append((idx, self.total + sound_len, self.total + len(audio)))
            self.total += len(audio)
            self.speaker.play(audio)
            if self.on_audio:
                self.on_audio(len(audio) / self.speaker.rate)
        if self.next >= len(self.items) and not self.speaker.busy:
            done = self.response
            self._clear()
            return done
        return None

    def heard_samples(self) -> int:
        rem = getattr(self.speaker, "remaining", None)
        left = int(rem() * self.speaker.rate) if rem is not None else 0
        return max(0, self.total - left)

    def item_at(self, seconds_ago: float = 0.0) -> int:
        """Index of the item the child was hearing `seconds_ago` seconds ago.  If that item's
        sound had already finished (the pause after it), the index of the item after it."""
        pos = self.heard_samples() - int(max(0.0, seconds_ago) * self.speaker.rate)
        for idx, sound_end, _end in self.spans:
            if pos < sound_end:
                return idx
        return self.next
