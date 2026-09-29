"""The real-time voice loop:  mic -> VAD -> ASR -> wake/brain -> cached speech -> speaker.

Conversation states:
  asleep     Only "Hey Buddy ..." is answered.
  listening  After "Hey Buddy" and after every finished task (for [session] follow_up_seconds):
             the child just says the next request, no wake phrase needed.  Speech that is not
             a request (background talk) is ignored.  Silence -> a short goodbye line -> asleep.
  talking    Buddy is speaking.  Only speech that starts with the wake phrase is taken, because
             Buddy's own voice reaches the microphone too: "Hey Buddy, pause / stop / repeat / quit /
             louder / softer", or a clear new request ("Hey Buddy, tell me a joke"), which keeps the
             interrupted story paused.
  answering  A quiz question or riddle waits for an answer (no wake phrase needed).

"Pause" remembers the reply and the sentence the child was hearing; "continue" resumes from
the start of that sentence.  The paused reply is kept until a new long activity starts or
[session] pause_memory_seconds pass.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .audio.dsp import wipe
from .brain import Brain
from .config import Config
from .core import Earcon, Expect, Response, Say
from .nlu.normalize import normalize
from .playback import Player, Renderer  # noqa: F401  (Renderer re-exported for older imports)
from .speech.cache import SpeechCache

log = logging.getLogger("buddy")

BARGE_IN = {"stop", "pause", "repeat", "quit", "volume_up", "volume_down"}  # heard while Buddy talks
MORE_OK = {"joke", "riddle", "story", "poem", "song", "animal_sound", "animal_fact"}  # "Want another one?" fits


@dataclass
class Paused:
    response: Response
    index: int  # item to resume from
    at: float
    expect: Optional[Expect] = None  # the quiz question / riddle that was being asked, if any


class Engine:
    def __init__(self, cfg: Config, brain: Brain, mic, speaker, asr, vad, cache: SpeechCache, button=None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.brain = brain
        self.mic, self.speaker, self.asr, self.vad = mic, speaker, asr, vad
        self.renderer = Renderer(cache, speaker.rate, cfg["paths"].get("sounds"))
        self.player = Player(self.renderer, speaker, on_audio=brain.progress.add_usage)
        self.button = button
        self.clock = clock
        self.events: "queue.Queue[tuple[str, str]]" = queue.Queue()
        s = cfg["session"]
        self.follow_up_s = float(s.get("follow_up_seconds", 15.0))
        self.prompt_every = max(1, int(s.get("prompt_every", 2)))
        self.pause_memory_s = float(s.get("pause_memory_seconds", 180.0))
        self.barge_window_s = float(s.get("barge_in_window", 4.0))
        self.muted = False
        self.awake_until = 0.0  # speech without the wake phrase is taken until then
        self.window: Optional[str] = None  # "wake" (after a bare "Hey Buddy") | "follow" (after a task)
        self.answer_deadline: Optional[float] = None
        self.paused: Optional[Paused] = None
        self.barge_until = 0.0  # "Hey Buddy" heard while talking: next control word taken until then
        self.tasks_since_prompt = 0
        self.session_tasks = 0
        self.running = True
        self.log_text = bool(cfg["privacy"].get("log_transcripts"))

    # ----------------------------------------------------------------- output
    def respond(self, r: Response, heard_seconds: float = 0.0) -> None:
        """Carry out a reply from the brain: control actions first, then say it."""
        a = r.action
        if a == "pause":
            return self._pause(heard_seconds, r.meta.get("word", "pause"))
        if a == "continue":
            return self._continue()
        if a in ("stop", "quit"):
            self.player.stop()
            self.paused = None
            self.answer_deadline = None
        elif a in ("volume_up", "volume_down"):
            key = self._change_volume(a)
            if self.player.active:
                return  # while talking, the change itself is the answer
            r = Response([Say(self.brain.skills.p(key))], repeatable=False, intent=a, after="silent")
        self._say(r)

    play = respond  # name used by earlier versions

    def _say(self, r: Response, start_index: int = 0) -> None:
        if self.paused is not None and (self._is_long(r) or (self.paused.expect is not None
                                                            and self.paused.expect is not self.brain.expect)):
            self.paused = None  # a new activity replaces the paused one (or its quiz is over)
        self.answer_deadline = None
        self.awake_until = 0.0  # re-opened when this reply has finished
        self.player.start(r, start_index)
        log.info("-> [%s] %s", r.intent, r.transcript()[:120])

    @staticmethod
    def _is_long(r: Response) -> bool:
        return r.intent == "quiz" or sum(isinstance(i, Say) for i in r.items) >= 3

    def _change_volume(self, action: str) -> str:
        a = self.cfg["audio"]
        step = 0.12 if action == "volume_up" else -0.12
        new = min(a["max_volume"], max(a["min_volume"], self.speaker.volume + step))
        key = "louder" if step > 0 else "softer"
        if new == self.speaker.volume:
            key = "max_volume" if step > 0 else "min_volume"
        self.speaker.volume = new
        return key

    # ------------------------------------------------------ pause / continue
    def _hold_current(self, heard_seconds: float = 0.0) -> bool:
        """Stop the reply that is playing and remember where, for "continue".  False if nothing to keep."""
        cur = self.player.response
        if cur is None or cur.after != "task":  # not an activity (a short "Okay!", a prompt, a chime)
            return False
        # the child started speaking before we understood them: go back to that moment
        idx = self.player.item_at(heard_seconds + float(self.cfg["vad"]["min_silence"]) + 0.3)
        self.player.stop()
        if idx < len(cur.items):
            self.paused = Paused(cur, idx, self.clock(), self.brain.expect)
        return True

    def _pause(self, heard_seconds: float = 0.0, word: str = "pause") -> None:
        p = self.brain.skills.p
        cur = self.player.response
        if self._hold_current(heard_seconds):
            self._say(Response([Say(p("pause_ack"))], repeatable=False, intent="pause", after="silent"))
        elif cur is not None:  # only a short line or chime was playing: just stop it
            self.player.stop()
            self._open_window("follow")
        elif self.paused is not None:
            self._say(Response([Say(p("already_paused"))], repeatable=False, intent="pause", after="silent"))
        elif self.brain.expecting or word == "wait":  # "wait!" while thinking: more time
            self._say(Response([Say(p("wait_ack"))], repeatable=False, intent="wait", after="silent"))
        else:
            self._say(Response([Say(p("nothing_to_pause"))], repeatable=False, intent="pause", after="silent"))

    def _continue(self) -> None:
        if self.player.active:
            return  # already talking
        paused, self.paused = self.paused, None
        if paused is not None and self.clock() - paused.at <= self.pause_memory_s:
            if paused.expect is not None and self.brain.expect is None:
                self.brain.expect = paused.expect
            r = paused.response
            self._say(Response(r.items[paused.index:], expect=r.expect, repeatable=False, intent=r.intent,
                               meta=r.meta, after=r.after))
        else:
            self._say(Response([Say(self.brain.skills.p("nothing_to_continue"))], repeatable=False,
                               intent="continue", after="silent"))

    # ------------------------------------------------------------------ input
    def on_text(self, text: str, heard_seconds: float = 0.0) -> None:
        """One recognised utterance (heard_seconds = its length, used to find where to pause)."""
        if not text:
            return
        now = self.clock()
        if self.player.active:
            return self._while_talking(text, heard_seconds, now)
        woke, rest, score = self.brain.wake.split(text)
        if self.log_text:
            log.info("heard: %r (wake=%s %.2f)", text, woke, score)
        if self.brain.expecting and not woke and now < self.awake_until:
            self.respond(self.brain.answer(text), heard_seconds)
        elif woke and rest:
            self._wake(now)
            self.respond(self.brain.command(rest), heard_seconds)
        elif woke:
            self._wake(now)
            self.respond(Response([Earcon("listen")], repeatable=False, intent="listening", after="wake"))
        elif now < self.awake_until:
            if self.window == "wake":  # just after "Hey Buddy": always reply, even "I didn't catch that"
                self.respond(self.brain.command(text), heard_seconds)
            else:  # follow-up window: only real requests, background talk is ignored
                r = self.brain.follow_up(text)
                if r is not None:
                    self.respond(r, heard_seconds)
                elif self.log_text:
                    log.info("ignored (not a request): %r", text)

    def _while_talking(self, text: str, heard_seconds: float, now: float) -> None:
        """Barge-in: "Hey Buddy, <control or request>" while Buddy talks (its own voice reaches the mic too)."""
        woke, rest, score = self.brain.wake.find(text)
        if self.log_text:
            log.info("heard while talking: %r (wake=%s %.2f)", text, woke, score)
        if woke and not rest:
            self._duck(now)  # "Hey Buddy ..." then a pause: talk softer and wait for the command
            return
        if woke:
            if self._is_echo(rest):
                return
            words = rest
        elif now < self.barge_until:
            words = text
        else:
            return
        intent = self.brain.matcher.parse(words)
        if intent.name in BARGE_IN:
            self._unduck()
            self.respond(self.brain.command(words, intent), heard_seconds)
        elif intent.score >= 0.88 and intent.name not in ("continue", "curiosity"):
            # "Hey Buddy, tell me a joke" in the middle of a story: a clear new request interrupts.
            # The story is kept paused, so "continue" can bring it back after a short activity.
            self._unduck()
            self._hold_current(heard_seconds)
            self.respond(self.brain.command(words, intent), heard_seconds)

    def _is_echo(self, words: str) -> bool:
        """The recogniser heard Buddy itself saying "...Hey Buddy, <words>" (e.g. a goodbye hint)."""
        r = self.player.response
        said = " " + " ".join(normalize(" ".join(i.text for i in r.items if isinstance(i, Say)))) + " " if r else ""
        if not any(f" {p} " in said for p, _ in self.brain.wake.targets if " " in p):
            return False
        toks = normalize(words)
        return bool(toks) and sum(f" {t} " in said for t in toks) / len(toks) >= 0.6

    def _duck(self, now: float) -> None:
        self.barge_until = now + self.barge_window_s
        self.speaker.duck = 0.35

    def _unduck(self) -> None:
        self.barge_until = 0.0
        self.speaker.duck = 1.0

    # --------------------------------------------------------- session state
    def _wake(self, now: float) -> None:
        if now >= self.awake_until:  # a new conversation
            self.tasks_since_prompt = self.prompt_every - 1  # the first task gets a "what next?" prompt
            self.session_tasks = 0

    def _open_window(self, kind: str, seconds: Optional[float] = None) -> None:
        secs = self.follow_up_s if seconds is None else seconds
        if secs > 0:
            self.awake_until, self.window = self.clock() + secs, kind

    def _sleep(self) -> None:
        self.awake_until, self.window = 0.0, None
        self.brain.offer = None
        self._unduck()

    def _finished(self, r: Response) -> None:
        """A reply has been completely heard: decide what Buddy listens for next."""
        self.vad.reset()  # drop the echo of our own voice
        if r.after == "sleep":
            return self._sleep()
        paused_question = self.paused is not None and self.paused.expect is not None \
            and self.paused.expect is self.brain.expect
        if self.brain.expecting and not paused_question:
            self.answer_deadline = self.clock() + float(self.cfg["nlu"]["answer_window"])
            self.awake_until, self.window = self.answer_deadline, "follow"
            return
        if r.intent == "boot":
            return self._open_window("wake")  # the greeting asks "what shall we do?"
        if r.after == "task" and self.follow_up_s > 0:
            self.session_tasks += 1
            return self._say(self._follow_up(r))
        if r.after == "wake":
            return self._open_window("wake", float(self.cfg["wake"]["listen_seconds"]))
        self._open_window("follow")  # (follow_up_seconds = 0: wake phrase needed every time, as before)

    def _follow_up(self, r: Response) -> Response:
        """Soft chime after every task; a short varied prompt after every `prompt_every`-th task."""
        self.tasks_since_prompt += 1
        items = []
        if self.tasks_since_prompt >= self.prompt_every:
            self.tasks_since_prompt = 0
            more = r.intent in MORE_OK and self.brain.last_intent is not None
            items.append(Say(self.brain.skills.p("follow_up_more" if more else "follow_up"), 0.15))
            self.brain.offer = "more" if more else None
        items.append(Earcon("ready"))
        return Response(items, repeatable=False, intent="follow_up", after="silent")

    def _window_closed(self) -> None:
        window, self.awake_until, self.window = self.window, 0.0, None
        if window == "wake" and self.session_tasks == 0:
            r = Response([Earcon("end")], repeatable=False, intent="timeout", after="sleep")  # as before
        else:
            key = "sleep_paused" if self.paused is not None else "sleep"
            r = Response([Say(self.brain.skills.p(key)), Earcon("end")], repeatable=False, intent="sleep",
                         after="sleep")
        self._say(r)

    def toggle_mute(self) -> None:
        self.events.put(("mute", ""))

    def _do_mute(self) -> None:
        self.muted = not self.muted
        if self.muted:
            cur = self.player.response
            if cur is not None and not self.brain.expecting and self._is_long(cur):
                self.paused = Paused(cur, self.player.item_at(0.0), self.clock())  # "continue" after unmuting
            self.player.stop()
            self.mic.stop()
            self.vad.reset()
            self.brain.expect = None
            self.answer_deadline = None
        else:
            self.mic.start()
        if self.button:
            self.button.show_muted(self.muted)
        key = "muted" if self.muted else "unmuted"
        self._say(Response([Earcon("mute" if self.muted else "unmute"), Say(self.brain.skills.p(key))],
                           repeatable=False, intent=key, after="sleep" if self.muted else "silent"))

    def _timers(self) -> None:
        now = self.clock()
        done = self.player.tick()
        if done is not None:
            self._finished(done)
        if self.barge_until and now > self.barge_until:
            self._unduck()
        if self.paused is not None and now - self.paused.at > self.pause_memory_s:
            if self.paused.expect is not None and self.paused.expect is self.brain.expect:
                self.brain.expect = None  # the paused question is forgotten too
            self.paused = None
        if self.player.active:
            return
        if self.answer_deadline and now > self.answer_deadline:
            self.answer_deadline = None
            self.respond(self.brain.timeout())
        elif self.awake_until and now > self.awake_until:
            self._window_closed()

    # ------------------------------------------------------------------- loop
    def keyboard(self) -> None:
        """Keyboard helper: m = mute, t = talk (as if 'Hey Buddy'), p = pause/continue, r = repeat,
        x = end the conversation, q = quit the program, other text = typed utterance."""
        def run():
            for line in sys.stdin:
                cmd = line.strip()
                if cmd == "m":
                    self.toggle_mute()
                elif cmd == "t":
                    self.events.put(("text", "hey buddy"))
                elif cmd in ("p", "r", "x"):
                    self.events.put(("key", cmd))
                elif cmd == "q":
                    self.events.put(("quit", ""))
                    return
                elif cmd:
                    self.events.put(("text", cmd if cmd.lower().startswith(("hey", "hi ")) else "hey buddy " + cmd))
        threading.Thread(target=run, daemon=True).start()

    def _key(self, k: str) -> None:
        name = {"p": "pause" if self.player.active else "continue", "r": "repeat", "x": "quit"}[k]
        self._wake(self.clock())
        self.respond(self.brain.command(name))

    def run(self) -> None:
        self.speaker.start()
        self.mic.start()
        self._wake(self.clock())
        self._say(Response([Earcon("unmute"), Say(self.brain.skills.p("hello")), Earcon("ready")],
                           repeatable=False, intent="boot", after="silent"))
        log.info("Buddy is ready. Say 'Hey Buddy'!")
        while self.running:
            try:
                while True:
                    kind, val = self.events.get_nowait()
                    if kind == "mute":
                        self._do_mute()
                    elif kind == "text":
                        self.on_text(val)
                    elif kind == "key":
                        self._key(val)
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
                if self.player.active and not self.cfg["audio"]["barge_in"]:
                    wipe(x)
                else:
                    for seg in self.vad.accept(x):
                        text = self.asr.transcribe(seg)
                        heard = len(seg) / 16000
                        wipe(seg)
                        self.on_text(text, heard)
                    wipe(x)
            self._timers()
        self.player.stop()
        self.mic.stop()
        self.speaker.close()
