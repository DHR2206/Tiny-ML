"""Response data model shared by skills, the engine and the content builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


@dataclass
class Say:
    """A sentence spoken by Buddy (pre-rendered into the speech cache)."""
    text: str
    pause_after: float = 0.3


@dataclass
class Sound:
    """An optional recorded sound (e.g. CC0 animal recording). Skipped if missing."""
    name: str


@dataclass
class Pause:
    seconds: float


@dataclass
class Earcon:
    """Short synthetic tone: 'listen', 'mute', 'unmute', 'correct', 'wrong'."""
    kind: str


Item = Union[Say, Sound, Pause, Earcon]


@dataclass
class Expect:
    """Buddy asked something and waits for an answer without the wake word.

    handler(tokens, raw_text) -> Response  is called with the child's reply.
    on_timeout() -> Response  is called if nothing is said within the answer window.
    prompt  the question itself, said again when the child asks Buddy to repeat.
    check(raw_text) -> bool  True if the reply answers the question correctly; lets a clear
            new request ("tell me a story") end the game instead of counting as a wrong answer.
    """
    handler: Callable[[list[str], str], "Response"]
    on_timeout: Optional[Callable[[], "Response"]] = None
    data: dict = field(default_factory=dict)
    prompt: list = field(default_factory=list)
    check: Optional[Callable[[str], bool]] = None


# What the engine does once a Response has finished playing (see engine.py):
#   task   - a task is complete: soft chime (sometimes a short "what next?" prompt), keep listening
#   silent - keep listening for the next command, no chime or prompt
#   wake   - short listening window after a bare "Hey Buddy"
#   sleep  - the conversation is over; wait for "Hey Buddy" again
AFTER_KINDS = ("task", "silent", "wake", "sleep")


@dataclass
class Response:
    items: list[Item] = field(default_factory=list)
    expect: Optional[Expect] = None
    action: Optional[str] = None  # stop | volume_up | volume_down | repeat | more | pause | continue | quit
    repeatable: bool = True  # store as "last response" for "say it again"
    intent: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    after: str = "task"  # one of AFTER_KINDS

    def say(self, text: str, pause: float = 0.3) -> "Response":
        self.items.append(Say(text, pause))
        return self

    def texts(self) -> list[str]:
        return [i.text for i in self.items if isinstance(i, Say)]

    def transcript(self) -> str:
        out = []
        for i in self.items:
            if isinstance(i, Say):
                out.append(i.text)
            elif isinstance(i, Sound):
                out.append(f"[sound:{i.name}]")
            elif isinstance(i, Earcon):
                out.append(f"[{i.kind}]")
        return " ".join(out)
