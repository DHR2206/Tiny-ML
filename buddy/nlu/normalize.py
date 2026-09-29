"""Text normalisation and number handling for recogniser output."""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " and " + number_to_words(r))
    th, r = divmod(n, 1000)
    return number_to_words(th) + " thousand" + ("" if r == 0 else " " + number_to_words(r))


_PLURALS = {1: "ones", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes", 7: "sevens", 8: "eights",
            9: "nines", 10: "tens"}

WORD_VALUES = {w: i for i, w in enumerate(_ONES)}
WORD_VALUES.update({w: i * 10 for i, w in enumerate(_TENS) if w})
WORD_VALUES.update({"hundred": 100, "fourty": 40, "twelf": 12})
# Child / recogniser variants that are only trusted when a number is *expected*
# (e.g. answering "what comes after two?")
LENIENT_VALUES = dict(WORD_VALUES)
LENIENT_VALUES.update({"won": 1, "ate": 8, "tree": 3, "free": 3, "sree": 3, "fwee": 3, "sick": 6,
                       "sicks": 6, "nein": 9, "tin": 10, "fie": 5, "fibe": 5, "fife": 5,
                       "sewen": 7, "seben": 7, "tirty": 30, "firty": 30, "fifdy": 50,
                       "to": 2, "too": 2, "for": 4, "fore": 4, "tree's": 3, "nothing": 0,
                       "none": 0, "tan": 10, "then": 10, "hate": 8, "wan": 1})

FILLERS = {
    "um", "umm", "uh", "uhh", "hmm", "mm", "er", "erm", "ah", "oh", "like", "please", "pls",
    "plz", "just", "now", "okay", "ok", "so", "well", "yeah", "hey", "hi",
}

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "let's": "let us", "lets": "let us", "i'm": "i am",
    "it's": "it is", "don't": "do not", "dont": "do not", "can't": "can not", "cant": "can not",
    "who's": "who is", "where's": "where is", "that's": "that is", "how's": "how is",
    "you're": "you are", "i'd": "i would", "wanna": "want to", "gonna": "going to",
    "gimme": "give me", "lemme": "let me", "doesn't": "does not", "isn't": "is not",
    "didn't": "did not", "won't": "will not", "i've": "i have", "there's": "there is",
    "abc's": "abcs", "a.b.c.": "abc", "a b c": "abc", "a, b, c": "abc", "a-b-c": "abc",
}


def normalize(text: str) -> list[str]:
    """Lower-case, strip punctuation, expand contractions; returns tokens."""
    t = text.lower().strip()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"\b([a-z])4\b", r"\1 for", t)  # recogniser writes "b for" as "B4"
    t = re.sub(r"\b([a-z])-([a-z]{2,})\b", r"\1\2", t)  # "P-Way" -> "pway"
    t = re.sub(r"(\d)\s*[x×\*]\s*(\d)", r"\1 times \2", t)
    t = re.sub(r"(\d)\s*\+\s*(\d)", r"\1 plus \2", t)
    # Recognisers write ranges as "1-10" / "1–10" but subtraction as "5 - 3".  Keep the
    # difference as a neutral "dash" token; nlu/slots.py decides range vs. minus from context.
    t = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1 dash \2", t)
    t = re.sub(r"(\d)-(\d)", r"\1 dash \2", t)
    t = re.sub(r"(\d)\s*-\s*(\d)", r"\1 minus \2", t)
    t = re.sub(r"\b(10|[1-9])'?s\b", lambda m: _PLURALS[int(m.group(1))], t)  # "count in 5s" -> "fives"
    for k in sorted(CONTRACTIONS, key=len, reverse=True):
        if " " in k or "." in k or "," in k or "-" in k:
            t = t.replace(k, CONTRACTIONS[k])
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    toks = []
    for w in t.split():
        w = CONTRACTIONS.get(w, w)
        toks.extend(w.split())
    return [w.strip("'") for w in toks if w.strip("'")]


def strip_fillers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FILLERS]


def parse_numbers(tokens: list[str], lenient: bool = False) -> list[tuple[int, int, int]]:
    """Find numbers in tokens.  Returns (value, start_index, end_index_exclusive).

    Handles digits ("27"), words ("twenty seven", "one hundred") and mixed.
    ``lenient`` also accepts common child / recogniser variants ("free" -> 3).
    """
    WORD_VALUES = LENIENT_VALUES if lenient else globals()["WORD_VALUES"]
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.isdigit():
            out.append((int(tok), i, i + 1))
            i += 1
            continue
        if tok in WORD_VALUES:
            val = WORD_VALUES[tok]
            j = i + 1
            if val == 100 and i > 0 and tokens[i - 1] == "a":
                pass
            # "one hundred", "hundred and five"
            if j < n and tokens[j] == "hundred" and val < 10:
                val *= 100
                j += 1
                if j < n and tokens[j] == "and":
                    j += 1
            if val >= 20 and val % 10 == 0 and val < 100 and j < n and tokens[j] in WORD_VALUES \
                    and 0 < WORD_VALUES[tokens[j]] < 10:
                val += WORD_VALUES[tokens[j]]
                j += 1
            elif val % 100 == 0 and val >= 100 and j < n and tokens[j] in WORD_VALUES \
                    and WORD_VALUES[tokens[j]] < 100:
                sub = WORD_VALUES[tokens[j]]
                j += 1
                if sub >= 20 and j < n and tokens[j] in WORD_VALUES and 0 < WORD_VALUES[tokens[j]] < 10:
                    sub += WORD_VALUES[tokens[j]]
                    j += 1
                val += sub
            out.append((val, i, j))
            i = j
            continue
        i += 1
    return out
