"""Parameter extraction shared by the skills: numbers, ranges, steps, direction, sums.

The intent matcher decides *what* the child wants ("count", "math" ...); this module works
out the *parameters* from whatever words the child (or the recogniser) used, instead of
expecting one fixed phrasing.  All of these give the same counting slots:

    count from 1 to 10 · count 1 to 10 · can you count from one to ten · count 1-10
    count one through ten · count between 1 and 10 · count from one two ten ("to" heard as "two")

Tokens come from nlu/normalize.py, so numbers are digits or number words, and a range
written as "1-10" arrives as "1 dash 10".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .normalize import WORD_VALUES, parse_numbers

RANGE_JOINERS = {"to", "till", "til", "until", "through", "thru", "upto", "dash", "minus"}
STEP_PLURALS = {"ones": 1, "twos": 2, "threes": 3, "fours": 4, "fives": 5, "sixes": 6, "sevens": 7,
                "eights": 8, "nines": 9, "tens": 10}
BACKWARD_CUES = {"backwards", "backward", "countdown", "reverse"}
TWO_LIKE = {"two", "too", "2"}  # "to" is often written as "two"/"2" between two numbers


@dataclass
class CountSlots:
    start: Optional[int] = None
    end: Optional[int] = None
    step: Optional[int] = None
    backwards: bool = False
    has_range: bool = False  # an explicit "from X to Y" / "X-Y" / "between X and Y" was said

    def as_dict(self) -> dict:
        d = {}
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        if self.step is not None:
            d["step"] = self.step
        if self.backwards:
            d["backwards"] = True
        return d


def _num_at(tokens: list[str], i: int) -> Optional[int]:
    t = tokens[i] if 0 <= i < len(tokens) else ""
    if t.isdigit():
        return int(t)
    return WORD_VALUES.get(t)


def count_slots(tokens: list[str]) -> CountSlots:
    """Start, end, step and direction of a counting request, in any phrasing."""
    toks = list(tokens)
    out = CountSlots()
    used: set[int] = set()
    # ---- step: "by 2", "by twos", "in fives", "count in 5s" (normalised to "fives"), "skip count by 3"
    for i, t in enumerate(toks):
        if t in STEP_PLURALS:
            out.step = STEP_PLURALS[t]
            used.add(i)
        elif t in ("by", "in") and i + 1 < len(toks):
            v = _num_at(toks, i + 1)
            if v is not None and 1 <= v <= 100:
                out.step = v
                used.update({i, i + 1})
    # ---- direction
    joined = " " + " ".join(toks) + " "
    if set(toks) & BACKWARD_CUES or any(p in joined for p in (" count down ", " counting down ", " down from ",
                                                                " down to ", " counts down ")):
        out.backwards = True
    # ---- numbers (ignoring the step number)
    nums = [(v, a, b) for v, a, b in parse_numbers(toks) if not ({a, b - 1} & used)]
    if len(nums) == 3 and nums[1][0] == 2 and nums[1][2] - nums[1][1] == 1 and toks[nums[1][1]] in TWO_LIKE:
        first, last = nums[0][0], nums[2][0]
        if first != last and (last > 2 or out.backwards):  # "from one two ten" = "from one to ten"
            out.start, out.end, out.has_range = first, last, True
            return out
    if len(nums) >= 2:
        (a, ai, aj), (b, bi, _bj) = nums[0], nums[1]
        between = set(toks[aj:bi])
        before = set(toks[max(0, ai - 2):ai])
        if len(nums) > 2 and not (between & RANGE_JOINERS):
            b = nums[-1][0]  # "count 1 2 3 4 5" -> 1 to 5
        out.start, out.end = a, b
        out.has_range = bool(between & RANGE_JOINERS) or (between == {"and"} and "between" in before) or not between
        return out
    if len(nums) == 1:
        v, i, _j = nums[0]
        prev = toks[i - 1] if i > 0 else ""
        prev2 = toks[i - 2] if i > 1 else ""
        if prev in ("from", "at", "with") or (prev == "number" and prev2 == "from"):
            out.start = v
        elif prev in ("to", "till", "til", "until", "upto", "through", "thru") or prev2 == "up":
            out.end = v
        elif out.backwards:
            out.start = v  # "count backwards 20"
        else:
            out.end = v  # "count 10"
    return out


def number_range(tokens: list[str]) -> Optional[tuple[int, int]]:
    """An explicit range like "from 5 to 15", "1-10" or "between 3 and 8", else None."""
    s = count_slots(tokens)
    if s.has_range and s.start is not None and s.end is not None:
        return s.start, s.end
    return None


# ------------------------------------------------------------------ sums
PLUS_WORDS = {"plus", "add", "added", "adding", "altogether", "sum"}
MINUS_WORDS = {"minus", "subtract", "subtracted", "takeaway"}
TIMES_WORDS = {"times", "multiply", "multiplied", "multiplication"}


def math_slots(tokens: list[str]) -> Optional[dict]:
    """Two operands and an operation: "what is 3 plus 4", "take away 2 from 5", "add 2 and 3",
    "5 - 3", "how much is 5 and 2", "multiply 3 by 4".  None if this is not a sum."""
    toks = list(tokens)
    nums = parse_numbers(toks)
    if len(nums) < 2:
        return None
    s = set(toks)
    joined = " " + " ".join(toks) + " "
    (a, ai, aj), (b, bi, _bj) = nums[0], nums[1]
    between = toks[aj:bi]
    op = None
    if s & TIMES_WORDS or "x" in between:
        op = "times"
    elif s & MINUS_WORDS or " take away " in joined or (between == ["dash"] and not s & {"count", "counting"}):
        op = "minus"
    elif s & PLUS_WORDS or (between == ["and"] and (" how much " in joined or " what is " in joined
                                                   or " makes " in joined or " make " in joined)):
        op = "plus"
    if op is None:
        return None
    # "take away 2 from 5" / "subtract 2 from 5" -> 5 - 2
    if op == "minus" and "from" in between and any(w in toks[:ai] for w in ("take", "away", "subtract", "minus")):
        a, b = b, a
    return {"a": a, "b": b, "op": op}
