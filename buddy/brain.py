"""Dialogue manager: text in -> Response out.  Independent of audio (testable).

Three ways the child's words reach Buddy (the engine decides which one applies):
  command(text)    after the wake phrase ("Hey Buddy, tell me a story")
  follow_up(text)  in the listening window after a task, without the wake phrase;
                   returns None for speech that is not a request (background talk)
  answer(text)     while a quiz question or riddle is waiting for an answer
"""
from __future__ import annotations

import random
import time
from typing import Optional

from .config import Config
from .content_store import Content
from .core import Response, Say
from .learning.progress import Progress
from .nlu.fallback import load_fallback
from .nlu.matcher import Intent, Matcher
from .skills.handlers import Skills
from .speech.wake import WakeMatcher

CONTROL = {"stop", "volume_up", "volume_down", "repeat", "pause", "continue", "quit"}
# A clear request for one of these ends a quiz/riddle instead of counting as a wrong answer.
# (Not counting, letters, colours, sums ...: a child thinking aloud about the answer uses those words.)
SWITCH_TO = {"story", "poem", "song", "joke", "riddle", "quiz", "breathing", "manners", "feeling",
             "good_night", "goodbye"}
HELLO_AGAIN_WITHIN = 600.0  # seconds: "hello" soon after playing gets "Hello again!"


class Brain:
    def __init__(self, cfg: Config, content: Optional[Content] = None, rng: Optional[random.Random] = None,
                 progress: Optional[Progress] = None):
        self.cfg = cfg
        self.content = content or Content()
        self.rng = rng or random.Random()
        self.matcher = Matcher(self.content, age=cfg.child_age)
        self.matcher.fallback = load_fallback(cfg["nlu"].get("fallback"))
        self.wake = WakeMatcher(cfg["wake"]["phrases"], self.matcher.phon, cfg["wake"]["threshold"])
        self.wake.validator = lambda rest: bool(rest) and self.matcher.parse(rest).name not in (
            "unknown", "none", "curiosity")
        start_level = 1 if cfg.child_age <= 4 else 2 if cfg.child_age <= 6 else 3
        self.progress = progress or Progress(cfg.data_dir / "progress.db", start_level=start_level)
        self.skills = Skills(self.content, self.matcher, self.progress, self.rng)
        self.expect = None  # pending Expect
        self.last: Optional[Response] = None  # last meaningful reply, for "repeat"
        self.last_intent = None  # last task, for "another one"
        self.offer: Optional[str] = None  # Buddy just asked "Want another one?" -> "yes" means more
        self.last_task_at = 0.0

    # ------------------------------------------------------------ public API
    @property
    def expecting(self) -> bool:
        return self.expect is not None

    def gate(self) -> Optional[Response]:
        """Parental rules checked before handling a command."""
        qh = self.cfg["parental"].get("quiet_hours") or ""
        if qh and _in_window(qh):
            return Response([Say(self.skills.p("quiet_hours"))], repeatable=False, intent="quiet_hours",
                            after="sleep")
        limit = float(self.cfg["parental"].get("daily_limit_minutes") or 0)
        if limit and self.progress.usage_today() >= limit * 60:
            return Response([Say(self.skills.p("daily_limit"))], repeatable=False, intent="daily_limit",
                            after="sleep")
        return None

    def command(self, text: str, intent: Optional[Intent] = None) -> Response:
        """Handle a request (after the wake phrase, or in the listening window)."""
        intent = intent or self.matcher.parse(text)
        offer, self.offer = self.offer, None
        if offer and intent.name in ("unknown", "none"):
            if self.matcher.is_yes(text):  # "Want another one?" - "Yes!"
                intent = Intent("more")
            elif self.matcher.is_no(text):
                return Response([Say(self.skills.p("no_problem"))], repeatable=False, intent="no", after="silent")
        if self.expect is not None and intent.name in CONTROL:
            return self._control_during_question(intent)  # "Hey Buddy, repeat" keeps the quiz going
        self.expect = None
        if intent.name not in CONTROL:
            blocked = self.gate()
            if blocked:
                return blocked
        return self._run(intent)

    def follow_up(self, text: str) -> Optional[Response]:
        """Speech in the listening window after a task.  None = not meant for Buddy (ignored)."""
        intent = self.matcher.parse(text)
        if intent.name in ("unknown", "none") and not (self.offer and (self.matcher.is_yes(text)
                                                                       or self.matcher.is_no(text))):
            return None
        return self.command(text, intent)

    def answer(self, text: str) -> Response:
        """Handle speech while Buddy is waiting for an answer (no wake phrase needed)."""
        intent = self.matcher.parse(text)
        if intent.name in CONTROL:
            return self._control_during_question(intent)
        exp = self.expect
        if exp.check is not None and intent.name in SWITCH_TO and intent.score >= 0.85:
            # A clear new request ends the game ("tell me a story").  Answers are short, and the
            # answer check is forgiving (child speech), so a full request wins even if a word of it
            # sounds like the answer; a feeling ("I am sad") only if it isn't the answer itself.
            if (intent.name != "feeling" and len(intent.tokens) >= 3) or not exp.check(text):
                self.expect = None
                return self.command(text, intent)
        self.expect = None
        r = exp.handler(intent.tokens, text)
        self.expect = r.expect
        if r.expect is None and r.items:
            self.last = r  # the end of a game can be repeated too
        return r

    def timeout(self) -> Response:
        exp, self.expect = self.expect, None
        if exp and exp.on_timeout:
            r = exp.on_timeout()
            self.expect = r.expect
            return r
        return Response([], repeatable=False)

    def listening_prompt(self) -> Response:
        return Response([Say(self.skills.p("listening"))], repeatable=False, intent="listening")

    # ------------------------------------------------------------ internals
    def _control_during_question(self, intent: Intent) -> Response:
        exp = self.expect
        if intent.name == "repeat" and exp is not None and exp.prompt:
            return Response(list(exp.prompt), expect=exp, intent="repeat", repeatable=False)
        if intent.name in ("stop", "quit"):
            self.expect = None
            return self._run(intent)
        return self._run(intent, keep_expect=True)  # pause / continue / louder / softer

    def _run(self, intent: Intent, keep_expect: bool = False) -> Response:
        saved = self.expect
        then = intent.slots.get("then")  # "tell me the story again"
        if intent.name == "repeat" and then is not None:
            if self.last is None or self.last_intent is None or self.last_intent.name != then.name:
                intent = then  # nothing like that to repeat yet: just do it
        if intent.name == "hello" and time.monotonic() - self.last_task_at < HELLO_AGAIN_WITHIN and self.last_task_at:
            intent = Intent("hello", {"again": True}, intent.confidence, intent.tokens, intent.score)
        r = self.skills.handle(intent)
        if r.action == "repeat":
            if self.last is not None:
                r = Response(list(self.last.items), expect=self.last.expect, intent="repeat", repeatable=False)
            else:
                r = Response([Say(self.skills.p("nothing_to_repeat"))], repeatable=False, after="silent")
        elif r.action == "more":
            if self.last_intent is not None:
                again = type(self.last_intent)(self.last_intent.name, {k: v for k, v in self.last_intent.slots.items()
                                                                      if k not in ("id", "animal", "letter")})
                r = self.skills.handle(again)
            else:
                r = self.skills.handle(type(intent)("help"))
        if r.repeatable and r.items:
            self.last = r
            self.last_intent = intent
        if intent.name not in CONTROL:
            self.last_task_at = time.monotonic()
        self.expect = r.expect if r.expect else (saved if keep_expect else None)
        return r


def _in_window(spec: str, now: Optional[time.struct_time] = None) -> bool:
    """'21:00-06:30' -> True if the local time is inside the window (wraps midnight)."""
    try:
        a, b = spec.split("-")
        ah, am = map(int, a.split(":"))
        bh, bm = map(int, b.split(":"))
    except ValueError:
        return False
    t = now or time.localtime()
    cur = t.tm_hour * 60 + t.tm_min
    start, end = ah * 60 + am, bh * 60 + bm
    return start <= cur < end if start <= end else cur >= start or cur < end
