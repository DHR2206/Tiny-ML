"""The real-time voice loop:  mic -> VAD -> ASR -> wake/brain -> cached speech -> speaker."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .audio.dsp import earcon, read_wav, resample, silence, wipe
from .brain import CONTROL, Brain
from .config import Config
from .core import Earcon, Pause, Response, Say, Sound
from .speech.cache import SpeechCache

log = logging.getLogger("buddy")


class Renderer:
    """Turns a Response into one audio buffer at the output rate."""

    def __init__(self, cache: SpeechCache, rate: int, sounds_dir: Optional[str]):
        self.cache = cache
        self.rate = rate
        self.sounds = Path(sounds_dir).expanduser() if sounds_dir else None

    def render(self, r: Response) -> np.ndarray:
        parts: list[np.ndarray] = []
        for it in r.items:
            if isinstance(it, Say):
                x = self.cache.get(it.text)
                if x is None:
                    log.warning("missing speech clip (run build-content): %s", it.text[:60])
                    continue
                parts += [x, silence(it.pause_after, self.rate)]
            elif isinstance(it, Pause):
                parts.append(silence(it.seconds, self.rate))
            elif isinstance(it, Earcon):
                parts += [earcon(it.kind, self.rate), silence(0.1, self.rate)]
            elif isinstance(it, Sound) and self.sounds:
                p = self.sounds / f"{it.name}.wav"
                if p.exists():
                    x, sr = read_wav(p)
                    x = resample(x[: sr * 6], sr, self.rate)  # at most 6 s
                    x = 0.8 * x / (np.abs(x).max() + 1e-6)
                    parts += [x, silence(0.3, self.rate)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class Engine:
    def __init__(self, cfg: Config, brain: Brain, mic, speaker, asr, vad, cache: SpeechCache, button=None):
        self.cfg = cfg
        self.brain = brain
        self.mic, self.speaker, self.asr, self.vad = mic, speaker, asr, vad
        self.renderer = Renderer(cache, speaker.rate, cfg["paths"].get("sounds"))
        self.button = button
        self.events: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.muted = False
        self.await_command_until = 0.0
        self.answer_deadline: Optional[float] = None
        self.was_busy = False
        self.running = True
        self.log_text = bool(cfg["privacy"].get("log_transcripts"))

    # ----------------------------------------------------------------- output
    def play(self, r: Response) -> None:
        a = self.cfg["audio"]
        if r.action == "stop":
            self.speaker.stop()
            self.brain.expect = None
            self.answer_deadline = None
        elif r.action in ("volume_up", "volume_down"):
            step = 0.12 if r.action == "volume_up" else -0.12
            new = min(a["max_volume"], max(a["min_volume"], self.speaker.volume + step))
            key = "louder" if r.action == "volume_up" else "softer"
            if new == self.speaker.volume:
                key = "max_volume" if step > 0 else "min_volume"
            self.speaker.volume = new
            r = Response([Say(self.brain.skills.p(key))], repeatable=False)
        audio = self.renderer.render(r)
        if len(audio):
            self.speaker.play(audio)
            self.brain.progress.add_usage(len(audio) / self.speaker.rate)
        self.answer_deadline = None  # (re)armed when playback finishes
        log.info("-> [%s] %s", r.intent, r.transcript()[:120])

    # ------------------------------------------------------------------ input
    def on_text(self, text: str) -> None:
        if not text:
            return
        woke, rest, score = self.brain.wake.split(text)
        if self.log_text:
            log.info("heard: %r (wake=%s %.2f)", text, woke, score)
        now = time.time()
        if self.speaker.busy:
            # barge-in: only "Hey Buddy, stop / louder / softer" while talking
            if woke and rest:
                intent = self.brain.matcher.parse(rest)
                if intent.name in CONTROL - {"repeat"}:
                    self.play(self.brain.skills.handle(intent))
            return
        if self.brain.expecting and not woke:
            self.play(self.brain.answer(text))
        elif woke and rest:
            self.await_command_until = 0
            self.play(self.brain.command(rest))
        elif woke:
            self.await_command_until = now + float(self.cfg["wake"]["listen_seconds"])
            self.play(Response([Earcon("listen")], repeatable=False, intent="listening"))
        elif now < self.await_command_until:
            self.await_command_until = 0
            self.play(self.brain.command(text))

    def toggle_mute(self) -> None:
        self.events.put(("mute", ""))

    def _do_mute(self) -> None:
        self.muted = not self.muted
        if self.muted:
            self.mic.stop()
            self.vad.reset()
            self.brain.expect = None
        else:
            self.mic.start()
        if self.button:
            self.button.show_muted(self.muted)
        key = "muted" if self.muted else "unmuted"
        self.play(Response([Earcon("mute" if self.muted else "unmute"), Say(self.brain.skills.p(key))],
                           repeatable=False, intent=key))

    def _timers(self) -> None:
        busy = self.speaker.busy
        now = time.time()
        if self.was_busy and not busy:
            self.vad.reset()  # drop echo of our own voice
            if self.brain.expecting:
                self.answer_deadline = now + float(self.cfg["nlu"]["answer_window"])
        self.was_busy = busy
        if not busy and self.answer_deadline and now > self.answer_deadline:
            self.answer_deadline = None
            self.play(self.brain.timeout())
        if self.await_command_until and now > self.await_command_until and not busy:
            self.await_command_until = 0
            self.play(Response([Earcon("end")], repeatable=False, intent="timeout"))

    # ------------------------------------------------------------------- loop
    def keyboard(self) -> None:
        """Keyboard helper: m = mute, t = talk (as if 'Hey Buddy'), q = quit, other text = typed utterance."""
        def run():
            for line in sys.stdin:
                cmd = line.strip()
                if cmd == "m":
                    self.toggle_mute()
                elif cmd == "t":
                    self.events.put(("text", "hey buddy"))
                elif cmd == "q":
                    self.events.put(("quit", ""))
                    return
                elif cmd:
                    self.events.put(("text", cmd if cmd.lower().startswith(("hey", "hi ")) else "hey buddy " + cmd))
        threading.Thread(target=run, daemon=True).start()

    def run(self) -> None:
        self.speaker.start()
        self.mic.start()
        self.play(Response([Earcon("unmute"), Say(self.brain.skills.p("hello"))], repeatable=False, intent="boot"))
        log.info("Buddy is ready. Say 'Hey Buddy'!")
        while self.running:
            try:
                while True:
                    kind, val = self.events.get_nowait()
                    if kind == "mute":
                        self._do_mute()
                    elif kind == "text":
                        self.on_text(val)
                    elif kind == "quit":
                        self.running = False
            except queue.Empty:
                pass
            if self.muted:
                time.sleep(0.05)
                self._timers()
                continue
            x = self.mic.read(timeout=0.05)
            if x is not None:
                if self.speaker.busy and not self.cfg["audio"]["barge_in"]:
                    wipe(x)
                else:
                    for seg in self.vad.accept(x):
                        text = self.asr.transcribe(seg)
                        wipe(seg)
                        self.on_text(text)
                    wipe(x)
            self._timers()
        self.mic.stop()
        self.speaker.close()
