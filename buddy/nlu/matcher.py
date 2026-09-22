"""Closed-vocabulary intent recognition for children's commands.

Two stages:
 1. *Canonicalisation* - every recognised token that is not an ordinary function
    word is snapped to the closest word of Buddy's vocabulary using the
    age-scheduled child-phonology distance (nlu/phonetics.py).  Adjacent tokens are
    also tried merged ("ele phant" -> "elephant").  "wion" -> "lion", "tount" -> "count".
 2. *Rules* - deterministic, prioritised rules on the canonical tokens decide the
    intent and fill slots (animal, number, letter, story/poem title, quiz domain ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..content_store import Content
from .normalize import WORD_VALUES, normalize, parse_numbers
from .phonetics import ChildPhonology, g2p

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "i", "me", "my", "you", "your", "we", "us", "it", "to",
    "of", "and", "in", "on", "at", "for", "with", "can", "could", "would", "will", "do", "does",
    "did", "what", "who", "how", "why", "where", "when", "which", "that", "this", "these", "those",
    "be", "was", "were", "have", "has", "had", "want", "wanna", "let", "please", "some", "any",
    "go", "goes", "say", "says", "tell", "play", "give", "make", "hear", "listen", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten", "so", "no", "yes", "not",
    "yeah", "ok", "okay", "oh", "up", "down", "there", "here", "about", "from", "till", "until",
    "by", "all", "more", "again", "like", "know", "now", "then", "too", "very", "our", "they",
    "he", "she", "him", "her", "his", "them", "if", "or", "but", "as", "much", "many", "time",
    "new", "big", "little", "good", "bad", "day", "way", "out", "off", "get", "got", "see", "look",
    "come", "put", "take", "help", "love", "feel", "am", "sure", "really", "just", "done",
    "ask", "something", "thing", "things", "fly", "live", "long", "into", "need", "try", "show",
    "everyone", "mummy", "mommy", "daddy", "papa", "mama", "dinner", "lunch", "water", "home",
}

# Words that trigger intents (canonical spellings).  Order does not matter.
KEYWORDS = {
    "stop", "quiet", "enough", "shush", "shh", "pause", "cancel", "louder", "softer", "quieter",
    "volume", "loud", "soft", "repeat", "another", "next", "story", "stories", "tale", "poem",
    "poems", "rhyme", "rhymes", "poetry", "nursery", "song", "songs", "sing", "music", "alphabet",
    "alphabets", "abc", "abcs", "letter", "letters", "phonics", "sound", "sounds", "noise", "count",
    "counting", "numbers", "number", "backwards", "backward", "twos", "fives", "tens", "animal",
    "animals", "color", "colors", "colour", "colours", "shape", "shapes", "body", "parts", "days",
    "week", "months", "year", "opposite", "opposites", "table", "tables", "times", "multiply",
    "multiplication", "plus", "minus", "add", "subtract", "quiz", "game", "games", "questions",
    "question", "test", "joke", "jokes", "funny", "riddle", "riddles", "puzzle", "breathe",
    "breathing", "calm", "relax", "manners", "magic", "words", "sad", "angry", "mad", "scared",
    "afraid", "happy", "bored", "hello", "hi", "morning", "night", "goodnight", "bye", "goodbye",
    "thank", "thanks", "name", "old", "sorry", "learn", "teach", "bedtime", "hungry", "tired",
    "sleepy", "lonely", "cry", "crying", "upset", "frightened", "worried", "excited", "glad",
}

LETTER_WORDS = {
    "a": "a", "ay": "a", "hey": None, "b": "b", "bee": "b", "be": "b", "c": "c", "see": "c", "sea": "c",
    "d": "d", "dee": "d", "e": "e", "ee": "e", "f": "f", "eff": "f", "ef": "f", "g": "g", "gee": "g",
    "h": "h", "aitch": "h", "age": "h", "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k", "kay": "k",
    "okay": None, "l": "l", "el": "l", "elle": "l", "m": "m", "em": "m", "n": "n", "en": "n", "o": "o",
    "oh": "o", "owe": "o", "p": "p", "pee": "p", "pea": "p", "q": "q", "queue": "q", "cue": "q",
    "r": "r", "are": "r", "ar": "r", "s": "s", "es": "s", "ess": "s", "t": "t", "tea": "t", "tee": "t",
    "u": "u", "you": "u", "v": "v", "vee": "v", "w": "w", "x": "x", "ex": "x", "y": "y", "why": "y",
    "z": "z", "zee": "z", "zed": "z",
}
LETTER_WORDS = {k: v for k, v in LETTER_WORDS.items() if v}

NUMBER_WORDS = set(WORD_VALUES) | {"hundred", "thousand"}
CONTROL_WORDS = {"stop", "enough", "shush", "shh", "pause", "cancel", "louder", "softer", "quieter", "repeat"}
QUESTION_WORDS = {"why", "what", "how", "where", "who", "when", "which", "is", "are", "can", "do", "does", "will"}
TITLE_STOP = {"the", "a", "and", "of", "who", "her", "his", "story", "poem"}


@dataclass
class Intent:
    name: str
    slots: dict = field(default_factory=dict)
    confidence: float = 1.0
    tokens: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # compact for logs/tests
        return f"Intent({self.name}, {self.slots}, {self.confidence:.2f})"


class Matcher:
    def __init__(self, content: Content, age: float = 5.0, word_threshold: float | None = None):
        self.content = content
        self.phon = ChildPhonology(age)
        # younger children -> more lenient snapping
        self.word_threshold = word_threshold or (0.62 if age < 5 else 0.68)
        self.animal_by_name: dict[str, str] = {}
        for a in content.animals:
            for n in a["names"] + [a["id"]]:
                self.animal_by_name[n] = a["id"]
        self.color_names = {c["name"] for c in content.lessons["colors"]}
        self.shape_names = {s["name"] for s in content.lessons["shapes"]}
        vocab = set(KEYWORDS) | {w for n in self.animal_by_name for w in n.split()}
        vocab |= self.color_names | self.shape_names
        vocab |= {w for t in self._titles() for w in t[2].split() if w not in TITLE_STOP and len(w) > 2}
        vocab |= {"elephant", "alphabet"}
        vocab |= {f for f, _ in content.lessons["fruit_colors"]}
        self.vocab = sorted(vocab)
        self._vocab_ph = {w: g2p(w) for w in self.vocab}
        # snapping targets also include ordinary words, so "pway" -> "play" rather than a keyword
        self._target_ph = dict(self._vocab_ph)
        self._target_ph.update({w: g2p(w) for w in STOPWORDS if len(w) > 1})
        self._snap_cache: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------ helpers
    def _titles(self):
        for p in self.content.poems:
            for a in p["aliases"] + [p["title"].lower()]:
                yield ("poem", p["id"], a.lower())
        for s in self.content.stories:
            for a in s["aliases"] + [s["title"].lower()]:
                yield ("story", s["id"], a.lower())

    def snap(self, tok: str) -> tuple[str, float]:
        """Map one token onto the vocabulary (or keep it)."""
        if tok in self._snap_cache:
            return self._snap_cache[tok]
        if tok in self._vocab_ph or tok in STOPWORDS or tok in NUMBER_WORDS or tok.isdigit() or len(tok) < 2:
            res = (tok, 1.0)
        elif len(g2p(tok)) < 2:
            res = (tok, 1.0)  # too little evidence to snap ("zzz", "qqq")
        else:
            ph = g2p(tok)
            best, best_s = tok, 0.0
            vbest, vbest_s = tok, 0.0
            for w, wph in self._target_ph.items():
                if abs(len(wph) - len(ph)) > 3:
                    continue
                s = self.phon.similarity(ph, wph)
                if s > best_s:
                    best, best_s = w, s
                if w in self._vocab_ph and s > vbest_s:
                    vbest, vbest_s = w, s
            # prefer a content word when it is almost as close as a function word
            if best not in self._vocab_ph and vbest_s >= best_s - 0.08:
                best, best_s = vbest, vbest_s
            need = 0.9 if best in CONTROL_WORDS else self.word_threshold  # never stop by accident
            res = (best, best_s) if best_s >= need else (tok, 1.0)
        self._snap_cache[tok] = res
        return res

    def canonicalize(self, tokens: list[str]) -> tuple[list[str], float]:
        out: list[str] = []
        conf = 1.0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            # try merging two tokens ("ele phant", "alpha bet")
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt and not ({t, nxt} & STOPWORDS) and t not in self._vocab_ph and nxt not in self._vocab_ph:
                merged = t + nxt
                m, s = self.snap(merged)
                if (m in self._vocab_ph and s >= 0.85) or merged in self._vocab_ph:
                    out.append(m)
                    conf = min(conf, s)
                    i += 2
                    continue
            m, s = self.snap(t)
            out.append(m)
            conf = min(conf, s)
            i += 1
        return out, conf

    def best_title(self, toks: list[str], kinds=("poem", "story")) -> tuple[str | None, str | None, float]:
        """Soft match of multi-word titles/aliases against the utterance."""
        content = [t for t in toks if t not in STOPWORDS or t in ("little", "big", "one", "ten")]
        if not content:
            return None, None, 0.0
        best = (None, None, 0.0)
        for kind, tid, alias in self._titles():
            if kind not in kinds:
                continue
            words = [w for w in alias.split() if w not in TITLE_STOP]
            if not words:
                continue
            total = 0.0
            for w in words:
                total += max((1.0 if w == t else self.phon.word_similarity(t, w)) for t in content)
            score = total / len(words)
            # prefer longer (more specific) aliases slightly
            score += 0.01 * len(words)
            if score > best[2]:
                best = (kind, tid, score)
        return best

    def find_animal(self, toks: list[str]) -> str | None:
        for t in toks:
            if t in self.animal_by_name:
                return self.animal_by_name[t]
        for i in range(len(toks) - 1):
            bi = toks[i] + " " + toks[i + 1]
            if bi in self.animal_by_name:
                return self.animal_by_name[bi]
        return None

    def find_animal_soft(self, raw: list[str]) -> str | None:
        """Slot-directed search: the child asked for a *sound*, so an animal name is expected.
        Tries 1-2 token windows (the recogniser often splits a mispronounced name: 'wion' -> 'we on')."""
        cues = {"sound", "sounds", "noise", "say", "says", "go", "goes"}
        cands = [t for t in raw if t not in cues and t not in ("the", "a", "an", "does", "what", "how", "do")]
        best, best_s = None, 0.0
        windows = [(t,) for t in cands if t not in STOPWORDS] + [tuple(cands[i:i + 2]) for i in range(len(cands) - 1)]
        for win in windows:
            ph = tuple(p for w in win for p in g2p(w))
            if len(ph) < 2:
                continue
            for name, aid in self.animal_by_name.items():
                s = self.phon.similarity(ph, g2p(name.replace(" ", "")))
                if s > best_s:
                    best, best_s = aid, s
        return best if best_s >= 0.7 else None

    @staticmethod
    def find_letter(raw_tokens: list[str], toks: list[str]) -> str | None:
        """Letter slot, only trusted next to cue words (letter / for / sound)."""
        cues = {"letter", "for", "sound", "sounds", "phonics", "starts", "start", "makes", "make", "say"}
        for i, t in enumerate(raw_tokens):
            if t in LETTER_WORDS:
                prev = raw_tokens[i - 1] if i > 0 else ""
                nxt = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else ""
                if prev in ("letter", "about") or nxt in ("for", "sound", "sounds", "says", "say", "make", "makes") \
                        or (len(raw_tokens) <= 2 and any(c in raw_tokens for c in cues)):
                    return LETTER_WORDS[t]
        return None

    # --------------------------------------------------------------------- main
    def parse(self, text: str) -> Intent:
        raw = normalize(text)
        raw = [t for t in raw if t not in ("um", "umm", "uh", "hmm", "please", "buddy")]
        toks, conf = self.canonicalize(raw)
        s = set(toks)
        joined = " " + " ".join(toks) + " "
        nums = [n for n, _, _ in parse_numbers(toks)]
        I = lambda name, **slots: Intent(name, slots, conf, toks)  # noqa: E731

        if not toks:
            return Intent("none", {}, 0.0, toks)

        # ---- controls (highest priority, usually short)
        if s & {"stop", "enough", "shush", "shh", "cancel", "pause"} or re.search(r" be quiet ", joined) \
                or joined.strip() in ("quiet", "that is enough"):
            return I("stop")
        if "louder" in s or re.search(r" (volume|turn it|speak) up ", joined) or joined.strip() == "loud":
            return I("volume_up")
        if s & {"softer", "quieter"} or re.search(r" (volume|turn it) down | too loud ", joined):
            return I("volume_down")
        content_words = {"story", "poem", "joke", "riddle", "quiz", "count", "counting", "animal", "animals"}
        if ("repeat" in s or re.search(r" (say|do|tell) (it|that|this) again | one more time |^ again ", joined)) \
                and not (s & content_words) and len(toks) <= 5:
            return I("repeat")
        if (s & {"another", "next"} or re.search(r" one more ", joined)) and len(toks) <= 4 \
                and not (s & content_words):
            return I("more")
        if re.search(r" what can you do | what do you do | help me | what can i say ", joined) or joined.strip() == "help":
            return I("help")

        # ---- maths: "what is 3 plus 4", "5 times 6"
        if len(nums) >= 2 and s & {"plus", "add", "minus", "subtract", "times", "multiply"} \
                and not (s & {"table", "tables"}):
            op = "plus" if s & {"plus", "add"} else "minus" if s & {"minus", "subtract"} else "times"
            return I("math", a=nums[0], b=nums[1], op=op)

        # ---- quiz / game
        if s & {"quiz", "game", "games", "test"} or re.search(r" ask me | question for me ", joined):
            dom = None
            if s & {"animal", "animals"} or self.find_animal(toks):
                dom = "animals"
            elif s & {"letter", "letters", "alphabet", "abc", "phonics"}:
                dom = "letters"
            elif s & {"number", "numbers", "count", "counting", "math", "plus"}:
                dom = "numbers"
            elif s & {"color", "colors", "colour", "colours"}:
                dom = "colors"
            elif s & {"shape", "shapes"}:
                dom = "shapes"
            elif s & {"opposite", "opposites"}:
                dom = "opposites"
            return I("quiz", domain=dom)
        if s & {"riddle", "riddles", "puzzle"}:
            return I("riddle")
        if s & {"joke", "jokes", "funny"}:
            return I("joke")

        # ---- times tables
        if s & {"table", "tables", "multiplication"} and not (s & {"color", "colors"}):
            n = nums[0] if nums else None
            return I("table", n=n)

        # ---- counting
        if s & {"count", "counting"} or (s & {"numbers", "number"} and not s & {"quiz"}):
            slots = {}
            if "backwards" in s or "backward" in s or "down" in raw:
                slots["backwards"] = True
            for step_word, step in (("twos", 2), ("fives", 5), ("tens", 10)):
                if step_word in s:
                    slots["step"] = step
            m = re.search(r" by (\d+|two|five|ten) ", " " + " ".join(raw) + " ")
            if m and "step" not in slots:
                slots["step"] = {"two": 2, "five": 5, "ten": 10}.get(m.group(1), None) or int(m.group(1))
            if len(nums) >= 2:
                slots["start"], slots["end"] = nums[0], nums[1]
            elif len(nums) == 1:
                if re.search(r" from ", joined) and slots.get("backwards"):
                    slots["start"] = nums[0]
                elif slots.get("step") and nums[0] == slots["step"] and "to" not in raw:
                    pass
                else:
                    slots["end"] = nums[0]
            return I("count", **slots)

        # ---- phonics / letters
        letter = self.find_letter(raw, toks)
        if "phonics" in s or re.search(r" letter sounds? ", joined) or letter:
            return I("phonics", letter=letter)
        if s & {"alphabet", "alphabets", "abc", "abcs"} or re.search(r" (the )?letters ", joined):
            return I("alphabet")

        # ---- lessons
        if s & {"color", "colors", "colour", "colours"}:
            thing = next((t for t in toks if t in {f for f, _ in self.content.lessons["fruit_colors"]}), None)
            if thing and ("what" in raw or "which" in raw):
                return I("color_of", thing=thing)
            return I("colors")
        if s & {"shape", "shapes"} or (s & self.shape_names and s & {"what", "teach", "learn", "tell"}):
            return I("shapes", shape=next((t for t in toks if t in self.shape_names), None))
        if "body" in s or re.search(r" body parts? ", joined):
            return I("body")
        if s & {"days", "week"}:
            return I("days")
        if s & {"months", "month"} or re.search(r" months of the year ", joined):
            return I("months")
        if s & {"opposite", "opposites"}:
            return I("opposites")

        # ---- animals
        animal = self.find_animal(toks)
        if not animal and s & {"sound", "sounds", "noise"} and not s & {"letter", "letters", "phonics"}:
            animal = self.find_animal_soft(raw)
            if animal:
                return I("animal_sound", animal=animal)
        if animal and re.search(r" (tell me )?about | facts? ", joined) and not s & {"sound", "sounds", "noise"}:
            return I("animal_fact", animal=animal)
        if s & {"animal", "animals"} or (animal and (s & {"sound", "sounds", "noise"} or re.search(
                r" (what|how) (does|do) (the |a )?\w+ (say|go) | says? ", joined))) or (animal and len(toks) <= 3):
            return I("animal_sound", animal=animal)

        # ---- wellbeing, manners
        if s & {"breathe", "breathing", "calm", "relax"}:
            return I("breathing")
        if "manners" in s or re.search(r" magic words ", joined):
            return I("manners")
        feel = self._feeling(raw, s)
        if feel:
            return I("feeling", mood=feel)

        # ---- small talk
        if re.search(r" (what is|tell me) your name | who are you ", joined):
            return I("name")
        if re.search(r" how old are you ", joined):
            return I("age")
        if re.search(r" how are you ", joined):
            return I("how_are_you")
        if re.search(r" i love you ", joined):
            return I("love")
        if s & {"thank", "thanks"}:
            return I("thanks")
        if "sorry" in s:
            return I("sorry")
        if "morning" in s and "good" in raw:
            return I("good_morning")
        if s & {"goodnight", "night"} and ("good" in raw or "goodnight" in s):
            return I("good_night")
        if s & {"bye", "goodbye"}:
            return I("goodbye")
        if s & {"hello", "hi"} and len(toks) <= 3:
            return I("hello")

        # ---- open "why / how / where / when" questions -> small FAQ or curiosity response
        is_question = raw[0] in ("why", "how", "where", "when") or "wonder" in raw or \
            (raw[0] in QUESTION_WORDS and not s & {"story", "poem", "rhyme", "song", "sing"})
        if is_question:
            faq = self.match_faq(raw)
            if faq is not None:
                return I("faq", index=faq)
            if raw[0] in ("why", "how", "where", "when") or "wonder" in raw:
                return I("curiosity")

        # ---- stories / poems / songs
        kind, tid, score = self.best_title(toks)
        if s & {"story", "stories", "tale", "bedtime"}:
            sid = tid if kind == "story" and score >= 0.8 else None
            return I("story", id=sid)
        if s & {"poem", "poems", "rhyme", "rhymes", "poetry", "nursery"}:
            pid = tid if kind == "poem" and score >= 0.8 else None
            return I("poem", id=pid)
        if s & {"song", "songs", "sing", "music"}:
            pid = tid if kind == "poem" and score >= 0.8 else None
            return I("song", id=pid)
        if kind and score >= 0.9 and not is_question:
            return I(kind, id=tid)
        if s & {"teach", "learn"}:
            return I("help")
        if is_question:
            return I("curiosity")
        return Intent("unknown", {}, conf, toks)

    @staticmethod
    def _feeling(raw: list[str], s: set[str]) -> str | None:
        text = " " + " ".join(raw) + " "
        if not re.search(r" (i am|i feel|i m|feeling|am feeling|i get) ", text) and len(raw) > 2:
            return None
        if s & {"sad", "cry", "crying", "upset", "lonely"}:
            return "sad"
        if s & {"angry", "mad"}:
            return "angry"
        if s & {"scared", "afraid", "frightened", "worried"}:
            return "scared"
        if s & {"happy", "glad", "excited"}:
            return "happy"
        if "bored" in s:
            return "bored"
        return None

    def match_faq(self, raw: list[str]) -> int | None:
        content = [t for t in raw if t not in STOPWORDS or t in ("why", "how", "where")]
        best, best_s = None, 0.0
        for i, item in enumerate(self.content.talk["faq"]):
            for q in item["q"]:
                qw = [w for w in normalize(q) if w not in STOPWORDS or w in ("why", "how", "where")]
                if not qw:
                    continue
                hit = sum(max((1.0 if w == t else self.phon.word_similarity(t, w)) for t in content or [""])
                          for w in qw) / len(qw)
                extra = max(0, len(content) - len(qw)) * 0.05
                sc = hit - extra
                if sc > best_s:
                    best, best_s = i, sc
        return best if best_s >= 0.85 else None

    # ----------------------------------------------------------- answer matching
    def matches_answer(self, text: str, answers: list[str | int]) -> bool:
        raw = normalize(text)
        nums = [n for n, _, _ in parse_numbers(raw, lenient=True)]
        for a in answers:
            if isinstance(a, int):
                if a in nums:
                    return True
                continue
            aw = normalize(str(a))
            if all(any(t == w or self.phon.word_similarity(t, w) >= self.word_threshold for t in raw) for w in aw):
                return True
        return False

    @staticmethod
    def is_dont_know(text: str) -> bool:
        t = " " + " ".join(normalize(text)) + " "
        return bool(re.search(r" (do not|dunno|don t) know | no idea | tell me | give up | not sure | help ", t))

    @staticmethod
    def is_yes(text: str) -> bool:
        return bool(set(normalize(text)) & {"yes", "yeah", "yep", "ok", "okay", "sure", "ya", "haan"})

    @staticmethod
    def is_no(text: str) -> bool:
        return bool(set(normalize(text)) & {"no", "nope", "nah", "stop", "enough"})
