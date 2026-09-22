"""Dialogue manager: text in -> Response out.  Independent of audio (testable)."""
from __future__ import annotations

import random
import time
from typing import Optional

from .config import Config
from .content_store import Content
from .core import Response, Say
from .learning.progress import Progress
from .nlu.matcher import Matcher
from .skills.handlers import Skills
from .speech.wake import WakeMatcher

CONTROL = {"stop", "volume_up", "volume_down", "repeat"}


class Brain:
    def __init__(self, cfg: Config, content: Optional[Content] = None, rng: Optional[random.Random] = None,
                 progress: Optional[Progress] = None):
        self.cfg = cfg
        self.content = content or Content()
        self.rng = rng or random.Random()
        self.matcher = Matcher(self.content, age=cfg.child_age)
        self.wake = WakeMatcher(cfg["wake"]["phrases"], self.matcher.phon, cfg["wake"]["threshold"])
        self.wake.validator = lambda rest: bool(rest) and self.matcher.parse(rest).name not in (
            "unknown", "none", "curiosity")
        start_level = 1 if cfg.child_age <= 4 else 2 if cfg.child_age <= 6 else 3
        self.progress = progress or Progress(cfg.data_dir / "progress.db", start_level=start_level)
        self.skills = Skills(self.content, self.matcher, self.progress, self.rng)
        self.expect = None  # pending Expect
        self.last: Optional[Response] = None
        self.last_intent = None

    # ------------------------------------------------------------ public API
    @property
    def expecting(self) -> bool:
        return self.expect is not None

    def gate(self) -> Optional[Response]:
        """Parental rules checked before handling a command."""
        qh = self.cfg["parental"].get("quiet_hours") or ""
        if qh and _in_window(qh):
            return Response([Say(self.skills.p("quiet_hours"))], repeatable=False, intent="quiet_hours")
        limit = float(self.cfg["parental"].get("daily_limit_minutes") or 0)
        if limit and self.progress.usage_today() >= limit * 60:
            return Response([Say(self.skills.p("daily_limit"))], repeatable=False, intent="daily_limit")
        return None

    def command(self, text: str) -> Response:
        """Handle a command spoken after the wake phrase."""
        self.expect = None
        intent = self.matcher.parse(text)
        if intent.name not in CONTROL:
            blocked = self.gate()
            if blocked:
                return blocked
        return self._run(intent)

    def answer(self, text: str) -> Response:
        """Handle speech while Buddy is waiting for an answer (no wake phrase needed)."""
        intent = self.matcher.parse(text)
        if intent.name in CONTROL:
            if intent.name == "stop":
                self.expect = None
            return self._run(intent, keep_expect=intent.name != "stop")
        exp, self.expect = self.expect, None
        r = exp.handler(intent.tokens, text)
        self.expect = r.expect
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
    def _run(self, intent, keep_expect: bool = False) -> Response:
        saved = self.expect
        r = self.skills.handle(intent)
        if r.action == "repeat":
            if self.last is not None:
                r = Response(list(self.last.items), expect=self.last.expect, intent="repeat", repeatable=False)
            else:
                r = Response([Say(self.skills.p("nothing_to_repeat"))], repeatable=False)
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
