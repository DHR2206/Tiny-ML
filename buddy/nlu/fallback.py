"""Optional last-resort intent classifier (a hook - nothing is plugged in today).

The rule-based matcher (nlu/matcher.py) handles everything on its own.  When a request
is still not understood after both matching passes, Matcher.parse() asks the fallback
configured here, if any.  This is the one place where a future offline classifier (for
example a small open-source language model) could be added without touching the rest of
Buddy.  It may only return one of Buddy's own intents: it never writes what Buddy says,
so every spoken sentence stays in the reviewed, pre-rendered inventory.

Select it with   [nlu] fallback = "none"   in buddy.toml ("none" is the only option today).
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol

log = logging.getLogger(__name__)


class IntentFallback(Protocol):
    def classify(self, text: str, tokens: list[str]) -> Optional[tuple[str, dict]]:
        """Return (intent_name, slots) for text the rules did not understand, or None."""


def load_fallback(name: str | None) -> Optional[IntentFallback]:
    if not name or str(name).lower() == "none":
        return None
    # A typo here used to raise out of Brain.__init__, so Buddy would not start at all.  The
    # rules understand every command on their own, so warn and carry on without a fallback.
    log.warning("Unknown [nlu] fallback %r - ignored. Only 'none' is available in this version.", name)
    return None
