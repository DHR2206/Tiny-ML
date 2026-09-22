"""Child-phonology-aware phonetic matching.

Young children systematically simplify adult speech (ASHA "phonological patterns"):

    fronting            cat  -> "tat"      (velar k/g -> alveolar t/d)      gone by ~4 y
    stopping            fun  -> "pun"      (fricative -> stop)              gone by ~3-5 y
    gliding             rabbit -> "wabbit" (r/l -> w/y)                     gone by ~6-7 y
    deaffrication       chip -> "ship"                                      gone by ~4 y
    th-fronting         three -> "free"                                     gone by ~6-7 y
    cluster reduction   stop -> "top"                                       gone by ~4-5 y
    final-consonant del bus -> "bu"                                         gone by ~3 y
    weak-syllable del   banana -> "nana"                                    gone by ~4 y

A speech recogniser trained on adult speech turns these into *different English words*
("tat" -> "that", "wion" -> "we on").  Instead of retraining the recogniser, Buddy
compares the recognised text with its closed vocabulary in a phoneme space using a
weighted edit distance whose substitution / deletion costs are LOW for exactly those
sound changes that a child of the configured age is still expected to make, and
normal for everything else.  The cost schedule is therefore a function of the
child's age (developmental norms), which is the key idea of this module.
"""
from __future__ import annotations

import functools
import re
from typing import Sequence

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

# ---------------------------------------------------------------------------
# 1. A small rule-based grapheme-to-phoneme converter (ARPAbet-like symbols).
#    Both the vocabulary and the recogniser output go through the same rules, so
#    consistency matters more than dictionary accuracy.  Irregular words that
#    commonly appear in children's commands are listed explicitly.
# ---------------------------------------------------------------------------
LEXICON: dict[str, str] = {
    "a": "AH", "the": "DH AH", "of": "AH V", "to": "T UW", "too": "T UW", "two": "T UW",
    "do": "D UW", "does": "D AH Z", "you": "Y UW", "your": "Y AO R", "who": "HH UW",
    "what": "W AH T", "where": "W EH R", "there": "DH EH R", "their": "DH EH R",
    "they": "DH EY", "said": "S EH D", "says": "S EH Z", "some": "S AH M", "come": "K AH M",
    "one": "W AH N", "won": "W AH N", "once": "W AH N S", "four": "F AO R", "for": "F AO R",
    "eight": "EY T", "ate": "EY T", "eighty": "EY T IY", "eighteen": "EY T IY N",
    "zero": "Z IH R OW", "twelve": "T W EH L V", "twenty": "T W EH N T IY", "thirty": "TH ER T IY",
    "forty": "F AO R T IY", "hundred": "HH AH N D R AH D", "eleven": "IH L EH V AH N",
    "water": "W AO T ER", "color": "K AH L ER", "colour": "K AH L ER", "colors": "K AH L ER Z",
    "colours": "K AH L ER Z", "orange": "AO R IH N JH", "purple": "P ER P AH L", "blue": "B L UW",
    "eye": "AY", "eyes": "AY Z", "heart": "HH AA R T", "ear": "IY R", "ears": "IY R Z",
    "lion": "L AY AH N", "tiger": "T AY G ER", "elephant": "EH L AH F AH N T", "monkey": "M AH NG K IY",
    "giraffe": "JH IH R AE F", "owl": "AW L", "cow": "K AW", "horse": "HH AO R S", "mouse": "M AW S",
    "donkey": "D AA NG K IY", "rooster": "R UW S T ER", "sheep": "SH IY P", "goat": "G OW T",
    "buddy": "B AH D IY", "body": "B AA D IY", "hey": "HH EY", "hi": "HH AY", "okay": "OW K EY",
    "ok": "OW K EY", "alphabet": "AE L F AH B EH T", "alphabets": "AE L F AH B EH T S",
    "abc": "EY B IY S IY", "abcs": "EY B IY S IY Z", "story": "S T AO R IY", "stories": "S T AO R IY Z",
    "poem": "P OW AH M", "poems": "P OW AH M Z", "rhyme": "R AY M", "rhymes": "R AY M Z",
    "song": "S AO NG", "songs": "S AO NG Z", "sing": "S IH NG", "again": "AH G EH N",
    "animal": "AE N AH M AH L", "animals": "AE N AH M AH L Z", "people": "P IY P AH L",
    "one's": "W AH N Z", "friend": "F R EH N D", "quiz": "K W IH Z", "because": "B IH K AH Z",
    "busy": "B IH Z IY", "move": "M UW V", "love": "L AH V", "give": "G IH V", "live": "L IH V",
    "have": "HH AE V", "gone": "G AO N", "done": "D AH N", "none": "N AH N", "shoe": "SH UW",
    "shoes": "SH UW Z", "knee": "N IY", "knees": "N IY Z", "toes": "T OW Z", "nose": "N OW Z",
    "mouth": "M AW TH", "tooth": "T UW TH", "teeth": "T IY TH", "brush": "B R AH SH",
    "moon": "M UW N", "sky": "S K AY", "why": "W AY", "my": "M AY", "by": "B AY", "fly": "F L AY",
    "cry": "K R AY", "dry": "D R AY", "try": "T R AY", "shy": "SH AY", "spy": "S P AY",
    "table": "T EY B AH L", "tables": "T EY B AH L Z", "circle": "S ER K AH L", "triangle": "T R AY AE NG G AH L",
    "square": "S K W EH R", "rectangle": "R EH K T AE NG G AH L", "oval": "OW V AH L",
    "diamond": "D AY M AH N D", "star": "S T AA R", "heart's": "HH AA R T S",
    "monday": "M AH N D EY", "tuesday": "T UW Z D EY", "wednesday": "W EH N Z D EY",
    "thursday": "TH ER Z D EY", "friday": "F R AY D EY", "saturday": "S AE T ER D EY",
    "sunday": "S AH N D EY", "january": "JH AE N Y UW EH R IY", "february": "F EH B R UW EH R IY",
    "months": "M AH N TH S", "month": "M AH N TH", "week": "W IY K", "weeks": "W IY K S",
    "i": "AY", "i'm": "AY M", "im": "AY M", "i've": "AY V", "don't": "D OW N T", "dont": "D OW N T",
    "know": "N OW", "knows": "N OW Z", "sure": "SH UH R", "please": "P L IY Z",
    "goodnight": "G UH D N AY T", "night": "N AY T", "light": "L AY T", "right": "R AY T",
    "bright": "B R AY T", "tonight": "T AH N AY T", "high": "HH AY", "sigh": "S AY",
    "laugh": "L AE F", "enough": "IH N AH F", "cough": "K AO F", "through": "TH R UW",
    "jokes": "JH OW K S", "joke": "JH OW K", "riddle": "R IH D AH L", "riddles": "R IH D AH L Z",
    "puzzle": "P AH Z AH L", "phonics": "F AA N IH K S", "counting": "K AW N T IH NG", "count": "K AW N T",
    "backward": "B AE K W ER D", "backwards": "B AE K W ER D Z", "times": "T AY M Z", "plus": "P L AH S",
    "minus": "M AY N AH S", "equals": "IY K W AH L Z", "breathe": "B R IY DH", "breathing": "B R IY DH IH NG",
    "tiny": "T AY N IY", "baby": "B EY B IY", "lady": "L EY D IY", "crocodile": "K R AA K AH D AY L",
    "tortoise": "T AO R T AH S", "hare": "HH EH R", "wolf": "W UH L F", "bear": "B EH R",
    "cheetah": "CH IY T AH", "zebra": "Z IY B R AH", "kangaroo": "K AE NG G ER UW", "camel": "K AE M AH L",
    "peacock": "P IY K AA K", "parrot": "P EH R AH T", "crow": "K R OW", "snake": "S N EY K",
    "bee": "B IY", "frog": "F R AA G", "duck": "D AH K", "dog": "D AO G", "doggy": "D AO G IY",
    "puppy": "P AH P IY", "kitty": "K IH T IY", "cat": "K AE T", "pig": "P IH G", "hen": "HH EH N",
    "chicken": "CH IH K AH N", "rabbit": "R AE B IH T", "bunny": "B AH N IY", "whale": "W EY L",
    "dolphin": "D AA L F IH N", "bird": "B ER D", "birds": "B ER D Z", "owl's": "AW L Z",
    "macdonald": "M AH K D AA N AH L D", "mcdonald": "M AH K D AA N AH L D",
    "humpty": "HH AH M P T IY", "dumpty": "D AH M P T IY", "twinkle": "T W IH NG K AH L",
    "little": "L IH T AH L", "itsy": "IH T S IY", "bitsy": "B IH T S IY", "spider": "S P AY D ER",
    "tune": "T UW N", "music": "M Y UW Z IH K", "lullaby": "L AH L AH B AY",
    "morning": "M AO R N IH NG", "hello": "HH AH L OW", "hullo": "HH AH L OW", "bye": "B AY",
    "goodbye": "G UH D B AY", "thanks": "TH AE NG K S", "thank": "TH AE NG K",
}

_DIGRAPHS = [
    ("tch", "CH"), ("dge", "JH"), ("igh", "AY"), ("sch", "S K"), ("ch", "CH"), ("sh", "SH"),
    ("th", "TH"), ("ph", "F"), ("wh", "W"), ("ck", "K"), ("ng", "NG"), ("qu", "K W"),
    ("gh", ""), ("ee", "IY"), ("ea", "IY"), ("oo", "UW"), ("ou", "AW"), ("ow", "OW"),
    ("oa", "OW"), ("ai", "EY"), ("ay", "EY"), ("oi", "OY"), ("oy", "OY"), ("au", "AO"),
    ("aw", "AO"), ("ey", "IY"), ("ie", "IY"), ("ue", "UW"), ("ew", "UW"), ("ar", "AA R"),
    ("er", "ER"), ("ir", "ER"), ("ur", "ER"), ("or", "AO R"), ("ll", "L"), ("ss", "S"),
    ("ff", "F"), ("zz", "Z"), ("tt", "T"), ("pp", "P"), ("bb", "B"), ("dd", "D"), ("gg", "G"),
    ("mm", "M"), ("nn", "N"), ("rr", "R"),
]
_SINGLE = {
    "b": "B", "d": "D", "f": "F", "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N",
    "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "v": "V", "w": "W", "x": "K S", "z": "Z",
}
_SHORT = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH", "y": "IH"}
_LONG = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW", "y": "AY"}


def _rule_g2p(word: str) -> list[str]:
    w = word
    out: list[str] = []
    if w.startswith("kn"):
        w = w[1:]
    elif w.startswith("wr"):
        w = w[1:]
    if len(w) > 4 and (w.endswith("our") or w.endswith("ours")):  # colour, flavours
        w = w.replace("our", "er")
    # magic-e:  V C e  at the end  -> long vowel, e silent
    magic = re.search(r"([aeiouy])([^aeiouy])e$", w) and len(w) > 3
    i = 0
    n = len(w)
    while i < n:
        if magic and i == n - 1 and w[i] == "e":
            break
        matched = False
        for g, p in _DIGRAPHS:
            if w.startswith(g, i):
                if p:
                    out.extend(p.split())
                i += len(g)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        if ch == "c":
            out.append("S" if nxt in ("e", "i", "y") else "K")
        elif ch == "g":
            out.append("JH" if nxt in ("e", "i", "y") and i > 0 else "G")
        elif ch == "y":
            if i == 0:
                out.append("Y")
            elif i == n - 1:
                out.append("IY")
            else:
                out.append("IH")
        elif ch in "aeiou":
            if magic and i == n - 3:
                out.append(_LONG[ch])
            elif ch == "e" and i == n - 1 and n > 2:
                pass  # silent final e
            else:
                out.append(_SHORT[ch])
        elif ch in _SINGLE:
            out.extend(_SINGLE[ch].split())
        i += 1
    # collapse accidental duplicates (e.g. "S S")
    dedup: list[str] = []
    for p in out:
        if not dedup or dedup[-1] != p or p in VOWELS:
            dedup.append(p)
    return dedup


@functools.lru_cache(maxsize=20000)
def g2p(word: str) -> tuple[str, ...]:
    word = word.lower().strip("'")
    if not word:
        return ()
    if word in LEXICON:
        return tuple(LEXICON[word].split())
    if word.isdigit():
        from .normalize import number_to_words

        return tuple(p for w in number_to_words(int(word)).split() for p in g2p(w))
    if word.endswith("'s") and word[:-2] in LEXICON:
        return tuple(LEXICON[word[:-2]].split()) + ("Z",)
    if re.fullmatch(r"[a-z]", word):  # spelled letters
        return tuple(LETTER_NAMES[word].split())
    if word.endswith("s") and len(word) > 3 and word[:-1] in LEXICON:
        return tuple(LEXICON[word[:-1]].split()) + ("Z",)
    return tuple(_rule_g2p(word))


LETTER_NAMES = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY", "f": "EH F", "g": "JH IY",
    "h": "EY CH", "i": "AY", "j": "JH EY", "k": "K EY", "l": "EH L", "m": "EH M", "n": "EH N",
    "o": "OW", "p": "P IY", "q": "K Y UW", "r": "AA R", "s": "EH S", "t": "T IY", "u": "Y UW",
    "v": "V IY", "w": "D AH B AH L Y UW", "x": "EH K S", "y": "W AY", "z": "Z IY",
}


def phrase_phonemes(words: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for w in words:
        out.extend(g2p(w))
    return tuple(out)


# ---------------------------------------------------------------------------
# 2. Age-scheduled cost model.
#    Each child phonological process has an (approximate) age by which typically
#    developing children stop using it (ASHA practice portal).  While the child is
#    younger than that age (+1 year of slack), the related sound change is cheap.
# ---------------------------------------------------------------------------
PROCESS_ELIMINATION_AGE = {
    "fronting": 4.0,
    "stopping": 5.0,
    "gliding": 7.0,
    "deaffrication": 4.0,
    "th_fronting": 7.0,
    "cluster_reduction": 5.0,
    "final_consonant_deletion": 3.5,
    "weak_syllable_deletion": 4.0,
    "voicing": 6.0,  # also a very common recogniser confusion
}

_PAIRS = {
    "fronting": [("K", "T"), ("G", "D"), ("NG", "N"), ("G", "T"), ("K", "D")],
    "stopping": [("F", "P"), ("V", "B"), ("S", "T"), ("Z", "D"), ("SH", "T"), ("CH", "T"),
                 ("JH", "D"), ("TH", "T"), ("DH", "D")],
    "gliding": [("R", "W"), ("L", "W"), ("L", "Y"), ("R", "L")],
    "deaffrication": [("CH", "SH"), ("JH", "ZH"), ("SH", "S"), ("ZH", "Z"), ("CH", "S")],
    "th_fronting": [("TH", "F"), ("DH", "V"), ("DH", "D"), ("TH", "S")],
    "voicing": [("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"), ("CH", "JH")],
}

ACTIVE_COST = 0.25  # cost of a process-related change the child is still expected to make
RESIDUAL_COST = 0.55  # same change after the typical elimination age (still recogniser noise)
DEFAULT_SUB = 1.0
VOWEL_SUB = 0.45
CV_SUB = 1.3
INS_DEL_CONS = 0.85
INS_DEL_VOWEL = 0.6


class ChildPhonology:
    """Weighted phoneme edit distance, parameterised by the child's age."""

    def __init__(self, age: float = 5.0):
        self.age = float(age)
        self.sub_cost: dict[tuple[str, str], float] = {}
        for proc, pairs in _PAIRS.items():
            c = self.process_cost(proc)
            for a, b in pairs:
                for key in ((a, b), (b, a)):
                    self.sub_cost[key] = min(self.sub_cost.get(key, DEFAULT_SUB), c)
        self.final_del = self.process_cost("final_consonant_deletion") + 0.1
        self.cluster_del = self.process_cost("cluster_reduction") + 0.1
        self.weak_syl_del = self.process_cost("weak_syllable_deletion") + 0.15
        self._cache: dict = {}

    def process_cost(self, process: str) -> float:
        return ACTIVE_COST if self.age < PROCESS_ELIMINATION_AGE[process] + 1.0 else RESIDUAL_COST

    # -- costs ---------------------------------------------------------------
    def sub(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        av, bv = a in VOWELS, b in VOWELS
        if av and bv:
            return VOWEL_SUB
        if av != bv:
            return CV_SUB
        return self.sub_cost.get((a, b), DEFAULT_SUB)

    def dele(self, seq: Sequence[str], i: int) -> float:
        """Cost of the child *omitting* seq[i] (target phoneme missing from what was heard)."""
        p = seq[i]
        if p in VOWELS:
            # an unstressed-looking vowel near the start (ba-NA-na -> nana)
            return self.weak_syl_del if i <= 1 and len(seq) > 4 else INS_DEL_VOWEL
        if p == "HH":
            return 0.2
        if p in ("R", "L") and 0 < i < len(seq) - 1:
            return self.process_cost("gliding") + 0.25  # liquid glided/vocalised away ("stowy")
        if i == len(seq) - 1:
            return self.final_del
        prev_c = i > 0 and seq[i - 1] not in VOWELS
        next_c = i + 1 < len(seq) and seq[i + 1] not in VOWELS
        if prev_c or next_c:
            return self.cluster_del
        if i <= 1 and len(seq) > 4 and seq[i + 1:i + 2] and seq[i + 1] in VOWELS:
            return self.weak_syl_del  # consonant of a dropped weak syllable
        return INS_DEL_CONS

    @staticmethod
    def ins(p: str) -> float:
        if p == "HH":
            return 0.2
        return INS_DEL_VOWEL if p in VOWELS else INS_DEL_CONS

    # -- distance --------------------------------------------------------------
    def distance(self, heard: Sequence[str], target: Sequence[str]) -> float:
        key = (tuple(heard), tuple(target))
        if key in self._cache:
            return self._cache[key]
        n, m = len(heard), len(target)
        prev = [0.0] * (m + 1)
        for j in range(1, m + 1):
            prev[j] = prev[j - 1] + self.dele(target, j - 1)
        for i in range(1, n + 1):
            cur = [prev[0] + self.ins(heard[i - 1])] + [0.0] * m
            hi = heard[i - 1]
            for j in range(1, m + 1):
                cur[j] = min(
                    prev[j - 1] + self.sub(hi, target[j - 1]),
                    prev[j] + self.ins(hi),
                    cur[j - 1] + self.dele(target, j - 1),
                )
            prev = cur
        d = prev[m]
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[key] = d
        return d

    def similarity(self, heard: Sequence[str], target: Sequence[str]) -> float:
        """1.0 = identical, 0.0 = unrelated."""
        if not heard or not target:
            return 0.0
        if tuple(heard) == tuple(target):
            return 1.0
        d = self.distance(heard, target)
        norm = max(len(target), len(heard))
        return max(0.0, 1.0 - d / norm)

    def word_similarity(self, heard_word: str, target_word: str) -> float:
        if heard_word == target_word:
            return 1.0
        return self.similarity(g2p(heard_word), g2p(target_word))
