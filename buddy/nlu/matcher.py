"""Closed-vocabulary intent recognition for children's commands.

    speech text -> canonical words -> intent candidates (scored) -> best intent + parameters

 1. *Canonicalisation* - words are snapped to the closest word of Buddy's vocabulary using
    the age-scheduled child-phonology distance (nlu/phonetics.py).  Adjacent tokens are also
    tried merged ("ele phant" -> "elephant").  "wion" -> "lion", "tount" -> "count".
    This runs in two passes.  The first pass leaves ordinary English words alone (see
    nlu/common_words.py), so "through", "hot" or "quit" are not turned into "three",
    "heart" or "quiet"; only non-words are snapped.  If that pass understands nothing
    (or only a weak guess), the second pass snaps every word, as the original design did.
 2. *Intent scoring* - every intent detector that fires adds a candidate with a score
    (explicit keyword > weaker cue > parameter-only guess); the best score wins, with a
    fixed priority order breaking ties.  Parameters (numbers, ranges, steps, sums) come
    from one shared extractor (nlu/slots.py), not from per-command phrase patterns.
 3. *Fallback hook* - if still nothing is understood, an optional classifier
    (nlu/fallback.py) may be asked.  None is configured today.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..content_store import Content
from .common_words import is_common_word
from .normalize import WORD_VALUES, normalize, parse_numbers
from .phonetics import ChildPhonology, g2p
from .slots import count_slots, math_slots, number_range

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
    "saying", "said", "going", "doing", "playing", "telling", "making", "talking",
    "everyone", "mummy", "mommy", "daddy", "papa", "mama", "dinner", "lunch", "water", "home",
    "dash", "never", "through", "thru",
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
    # controls added for pause / continue / quit
    "continue", "resume", "wait", "quit", "exit", "countdown",
    # "count in threes" etc. (so the step words are never snapped to other words)
    "ones", "threes", "fours", "sixes", "sevens", "eights", "nines", "step", "steps",
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
# Letter names that are also ordinary words ("a", "you", "see", "bee" ...): they only count as a
# letter right after "letter" or right before "for" ("a for apple"), never inside a sentence.
AMBIGUOUS_LETTER_WORDS = {"a", "i", "you", "be", "see", "sea", "are", "ar", "why", "oh", "owe", "eye", "tea",
                          "tee", "pea", "pee", "bee", "queue", "cue", "age", "ex", "gee", "jay", "kay", "el",
                          "en", "em", "es", "ess", "ay", "ee"}

# The *second* matching pass snaps every word, including ordinary English the first pass kept on
# purpose.  That last resort is what turned "call mummy" into breathing ("call"->"calm", 0.84) and
# "show me my ears" into a horse ("ears"->"horses", 0.80).  It may still overrule an everyday word
# when the match is strong enough to really be that keyword, but not on a loose resemblance.
# Measured on tests/test_nlu.py and the QA probe (see docs/FIXES.md): every value from 0.80 to
# "never" gives exactly the same results on all 353 cases, i.e. nothing Buddy understands today
# relies on the loose rewrite.  0.9 matches the bar already used for the control words below.
VOCAB_NEED = 0.90
# An ordinary English word recognised this clearly is taken as said, and is never replaced by a
# keyword that happens to be a little closer ("set" is "said", not the quiz keyword "test").
HEARD_CLEARLY = 0.90
# No word Buddy knows is anywhere near this long, so a longer token cannot match one.  Guards the
# quadratic phoneme comparison against an unbounded input (finding M8).
MAX_SNAP_LEN = 32

# Body parts: content/lessons.json names them in the plural ("eyes"), but a child asks "where is
# my nose" or "my tummy".  Singulars map to the lesson's own entry; the rest have no line of their
# own, so they open the body lesson, exactly as the words "body parts" already do (finding M5).
BODY_SINGULAR = {"eye": "eyes", "ear": "ears", "hand": "hands", "leg": "legs", "foot": "feet",
                 "tooth": "teeth"}
BODY_OTHER = {"tummy", "belly", "face", "hair", "finger", "fingers", "toe", "toes", "knee",
              "knees", "arm", "arms", "elbow", "shoulder", "shoulders", "chin", "cheek", "cheeks",
              "thumb"}

NUMBER_WORDS = set(WORD_VALUES) | {"hundred", "thousand"}
CONTROL_WORDS = {"stop", "enough", "shush", "shh", "pause", "cancel", "louder", "softer", "quieter", "repeat",
                 "continue", "resume", "wait", "quit", "exit"}
QUESTION_WORDS = {"why", "what", "how", "where", "who", "when", "which", "is", "are", "can", "do", "does", "will"}
TITLE_STOP = {"the", "a", "and", "of", "who", "her", "his", "story", "poem"}
NEGATIONS = {"not", "never"}  # "don't" is normalised to "do not"
# words naming an activity: "say the story again" repeats a story, "another story" is a new story
CONTENT_WORDS = {"story", "stories", "poem", "poems", "rhyme", "rhymes", "song", "joke", "jokes", "riddle", "riddles",
                 "quiz", "game", "count", "counting", "animal", "animals", "alphabet", "table", "tables", "colors",
                 "colours", "shapes", "phonics", "opposites", "days", "months"}
STORY_WORDS = {"story", "stories", "tale", "bedtime"}
POEM_WORDS = {"poem", "poems", "rhyme", "rhymes", "poetry", "nursery"}
SONG_WORDS = {"song", "songs", "sing", "music"}
FILLER = {"um", "umm", "uh", "hmm", "please", "buddy"}

# Tie-break order when two intents get the same score (earlier wins).
PRIORITY = ["stop", "pause", "continue", "quit", "volume_up", "volume_down", "repeat", "more", "help",
            "math", "quiz", "riddle", "joke", "table", "count", "phonics", "alphabet", "color_of", "colors",
            "shapes", "body", "days", "months", "opposite_of", "opposites", "animal_fact", "animal_sound",
            "breathing", "manners", "feeling", "name", "age", "how_are_you", "love", "thanks", "sorry",
            "good_morning", "good_night", "goodbye", "hello", "faq", "story", "poem", "song", "curiosity"]
_PRIO = {n: i for i, n in enumerate(PRIORITY)}
# Intents that are only ever started by a word of Buddy's own vocabulary.  Everything else
# (the controls, small talk, the FAQ, a sum, a letter) is matched on the words as they were
# actually said, so a word the snapper guessed at is never the only reason one of those fires.
KEYWORD_INTENTS = {"story", "poem", "song", "count", "table", "alphabet", "colors", "color_of",
                   "shapes", "body", "days", "months", "opposites", "opposite_of", "animal_sound",
                   "animal_fact", "quiz", "riddle", "joke", "breathing", "manners"}


@dataclass
class Intent:
    name: str
    slots: dict = field(default_factory=dict)
    confidence: float = 1.0  # how exactly the words were heard (phonetic snapping)
    tokens: list[str] = field(default_factory=list)
    score: float = 1.0  # how strongly the words point to this intent (see Matcher._decide)

    def __repr__(self) -> str:  # compact for logs/tests
        return f"Intent({self.name}, {self.slots}, {self.confidence:.2f})"


class Matcher:
    def __init__(self, content: Content, age: float = 5.0, word_threshold: float | None = None,
                 min_confidence: float = 0.0):
        self.content = content
        # [nlu] min_confidence: how clearly the words of a request must have been heard before
        # Buddy will start an activity on them.  0 = off (every guess is acted on).
        self.min_confidence = float(min_confidence or 0.0)
        self.phon = ChildPhonology(age)
        # younger children -> more lenient snapping
        self.word_threshold = word_threshold or (0.62 if age < 5 else 0.68)
        self.vocab_threshold = VOCAB_NEED
        self.heard_clearly = HEARD_CLEARLY
        self.animal_by_name: dict[str, str] = {}
        for a in content.animals:
            for n in a["names"] + [a["id"]]:
                self.animal_by_name[n] = a["id"]
        # "how does a dog bark", "make the lion roar": each animal's own sound verb
        self.sound_verbs: set[str] = set()
        for a in content.animals:
            v = a.get("verb", "")
            base = v[:-2] if v.endswith(("sses", "zzes", "shes", "ches")) else v[:-1] if v.endswith("s") else v
            if base and base not in ("sing", "call", "crow"):
                self.sound_verbs |= {v, base}
        self.color_names = {c["name"] for c in content.lessons["colors"]}
        self.body_names = {b["name"] for b in content.lessons["body_parts"]}
        self.body_words = self.body_names | set(BODY_SINGULAR) | BODY_OTHER
        self.shape_names = {s["name"] for s in content.lessons["shapes"]}
        self.opposite_of: dict[str, str] = {}
        for a, b in content.lessons["opposites"]:
            self.opposite_of.setdefault(a, b)
            self.opposite_of.setdefault(b, a)
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
        self.fallback = None  # optional nlu.fallback.IntentFallback (none by default)

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
        elif len(tok) > MAX_SNAP_LEN:
            res = (tok, 1.0)  # far longer than any word Buddy knows: nothing to compare it with
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
            # Prefer a content word when it is almost as close as a function word - unless the
            # ordinary word was itself heard clearly, in which case that is simply what was said.
            if best not in self._vocab_ph and vbest_s >= best_s - 0.08 and best_s < self.heard_clearly:
                best, best_s = vbest, vbest_s
            need = 0.9 if best in CONTROL_WORDS else self.word_threshold  # never stop by accident
            res = (best, best_s) if best_s >= need else (tok, 1.0)
        self._snap_cache[tok] = res
        return res

    def is_known(self, tok: str) -> bool:
        """A word Buddy should take as heard: its own vocabulary, numbers or everyday English."""
        return (tok in self._vocab_ph or tok in STOPWORDS or tok in NUMBER_WORDS or tok.isdigit()
                or is_common_word(tok))

    def canonicalize(self, tokens: list[str], aggressive: bool = True) -> tuple[list[str], float]:
        """Snap tokens onto the vocabulary.  aggressive=False keeps every known word as it is."""
        out, conf, _ = self.canonicalize_scored(tokens, aggressive)
        return out, conf

    def canonicalize_scored(self, tokens: list[str], aggressive: bool = True) \
            -> tuple[list[str], float, dict[str, float]]:
        """As canonicalize(), plus how clearly each resulting word was heard (1.0 = exactly as
        said).  A word Buddy only *guessed* at should not outrank one the child really said."""
        out: list[str] = []
        heard: dict[str, float] = {}
        conf = 1.0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if not aggressive and self.is_known(t):
                out.append(t)
                i += 1
                continue
            # try merging two tokens ("ele phant", "alpha bet")
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt and not ({t, nxt} & STOPWORDS) and t not in self._vocab_ph and nxt not in self._vocab_ph \
                    and (aggressive or not self.is_known(nxt)):
                merged = t + nxt
                m, s = self.snap(merged)
                if (m in self._vocab_ph and s >= 0.85) or merged in self._vocab_ph:
                    out.append(m)
                    heard[m] = min(heard.get(m, 1.0), s)
                    conf = min(conf, s)
                    i += 2
                    continue
            m, s = self.snap(t)
            if aggressive and m != t and m in self._vocab_ph and s < self.vocab_threshold \
                    and self.is_known(t):
                m, s = t, 1.0  # everyday English: only a clear match may turn it into a keyword
            out.append(m)
            heard[m] = min(heard.get(m, 1.0), s)
            conf = min(conf, s)
            i += 1
        return out, conf, heard

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
    def find_letter(raw_tokens: list[str], toks: list[str] | None = None) -> str | None:
        """Letter slot, only trusted next to cue words ("letter m", "b for ball", "z sound")."""
        cues = {"letter", "for", "sound", "sounds", "phonics", "starts", "start", "makes", "make", "say"}
        for i, t in enumerate(raw_tokens):
            if t not in LETTER_WORDS:
                continue
            prev = raw_tokens[i - 1] if i > 0 else ""
            nxt = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else ""
            if prev == "letter":
                return LETTER_WORDS[t]
            if t in AMBIGUOUS_LETTER_WORDS:
                # "a for apple", "what is a for": an ordinary word only counts right before "for"
                if nxt == "for" and (i == 0 or prev in ("is", "what", "about")):
                    return LETTER_WORDS[t]
                continue
            if prev == "about" or nxt in ("for", "sound", "sounds", "says", "say", "make", "makes") \
                    or (len(raw_tokens) <= 2 and any(c in raw_tokens for c in cues)):
                return LETTER_WORDS[t]
        return None

    @staticmethod
    def _negated(toks: list[str]) -> set[str]:
        """Control words preceded by "not"/"never" ("don't stop", "do not pause")."""
        return {t for i, t in enumerate(toks) if t in CONTROL_WORDS and set(toks[max(0, i - 3):i]) & NEGATIONS}

    @staticmethod
    def _topic(toks: list[str]) -> list[str]:
        """What a story/poem should be about: the words after "about" ("a story about a lion")."""
        if "about" not in toks:
            return []
        rest = toks[toks.index("about") + 1:]
        return [t for t in rest if t not in STOPWORDS and t not in CONTENT_WORDS and t not in STORY_WORDS
                and t not in POEM_WORDS and len(t) > 2]

    def _quiz_domain(self, s: set[str], toks: list[str]) -> str | None:
        if s & {"animal", "animals"} or self.find_animal(toks):
            return "animals"
        if s & {"letter", "letters", "alphabet", "abc", "phonics"}:
            return "letters"
        if s & {"number", "numbers", "count", "counting", "math", "plus"}:
            return "numbers"
        if s & {"color", "colors", "colour", "colours"}:
            return "colors"
        if s & {"shape", "shapes"}:
            return "shapes"
        if s & {"opposite", "opposites"}:
            return "opposites"
        return None

    # --------------------------------------------------------------------- main
    def parse(self, text: str) -> Intent:
        raw = normalize(text)
        raw = [t for t in raw if t not in FILLER]
        if not raw:
            # Something was said but no word came out of it (punctuation, another script, only
            # fillers): that is speech Buddy could not understand, not silence, so it gets the
            # "I didn't catch that" reply.  True silence never reaches here (see engine.on_text).
            return Intent("unknown" if text and text.strip() else "none", {}, 0.0, [], 0.0)
        toks, conf, heard = self.canonicalize_scored(raw, aggressive=False)
        best = self._decide(raw, toks, conf, heard)
        if best.score < 0.6:  # nothing (or only a weak guess): snap every word, as before
            toks2, conf2, heard2 = self.canonicalize_scored(raw, aggressive=True)
            if toks2 != toks:
                alt = self._decide(raw, toks2, conf2, heard2)
                if alt.score * 0.97 > best.score:
                    best = alt
        # Too unsure to start an activity: say so instead of guessing one (finding M4).
        # Controls, small talk, sums and letters are matched on the words as said, so they are
        # not affected; only the keyword-driven activities are.
        if best.name in KEYWORD_INTENTS and best.confidence < self.min_confidence:
            best = Intent("unknown", {}, best.confidence, best.tokens, 0.0)
        if best.name == "unknown" and self.fallback is not None:
            res = self.fallback.classify(text, raw)
            if res:
                best = Intent(res[0], dict(res[1]), 0.5, raw, 0.5)
        return best

    def _decide(self, raw: list[str], toks: list[str], conf: float,
                heard: dict[str, float] | None = None) -> Intent:
        """Score every intent the words point to; return the best one with its parameters."""
        if not toks:
            return Intent("none", {}, 0.0, toks, 0.0)
        heard = heard or {}

        def clearly(words) -> float:
            """How clearly the word that fires an intent was heard (1.0 = said, not guessed).
            Keeps a word Buddy only guessed at from outranking one the child really said:
            in "teww me a poem", "teww" resembles "tale", but "poem" was said outright."""
            hits = [heard.get(t, 1.0) for t in toks if t in words]
            return max(hits) if hits else 1.0
        s = set(toks)
        joined = " " + " ".join(toks) + " "
        n = len(toks)
        short = n <= 4
        # Buddy's vocabulary is small, so the snapper will always map *some* nonsense onto a
        # keyword.  That guess may start an activity when it is the child's whole request
        # ("tolours" is plainly "colours"), but not when the rest of the utterance made no sense
        # either: then the keyword is an artefact of the snapper, not something that was said.
        # "florp bimble wug" used to recite the days of the week (finding M4).
        _known = set(self._vocab_ph) | self.body_words | NUMBER_WORDS
        _kw = [t for t in toks if t in self._vocab_ph]
        _unresolved = [t for t in toks if len(t) > 1 and t not in _known and t not in STOPWORDS
                       and not t.isdigit()]
        # Only when part of the utterance made no sense at all does the keyword have to have been
        # heard clearly; on its own a guess is still trusted ("tolours" is plainly "colours").
        guessed_only = (bool(_kw) and bool(_unresolved)
                        and max(heard.get(t, 1.0) for t in _kw) < self.vocab_threshold)

        def pick(cs):
            if guessed_only:
                cs = [c for c in cs if c[2].name not in KEYWORD_INTENTS]
            if not cs:
                return Intent("unknown", {}, conf, toks, 0.0)
            return max(cs, key=lambda c: (c[0], c[1]))[2]
        nums = [v for v, _, _ in parse_numbers(toks)]
        neg = self._negated(toks)
        # "what story do you know" is a request, not a question - but only when the story word was
        # really heard: a guessed one stopped "who is the prime minister" from being a question at
        # all ("minister" resembles "months", "prime" resembles "rhyme").  Finding M4.
        _told = {"story", "poem", "rhyme", "song", "sing"}
        is_question = raw[0] in ("why", "how", "where", "when") or "wonder" in raw or \
            (raw[0] in QUESTION_WORDS and not (s & _told and clearly(_told) >= 0.99))
        cands: list[tuple[float, int, Intent]] = []

        def add(score: float, name: str, **slots) -> None:
            slots = {k: v for k, v in slots.items() if v is not None and v != []}
            cands.append((score, -_PRIO.get(name, len(PRIORITY)), Intent(name, slots, conf, toks, score)))

        # ---- controls (short commands; they win over everything else)
        stop_words = (s & {"stop", "enough", "shush", "shh", "cancel"}) - neg
        if re.search(r" (stop|wait) (for )?(a|one) (minute|moment|second|sec|bit) | take a (little )?break ", joined):
            add(1.0, "pause")
        elif stop_words or re.search(r" be quiet ", joined) or joined.strip() in ("quiet", "that is enough"):
            add(1.0, "stop")
        if "stop" in neg:
            add(0.9, "continue")  # "don't stop!"
        if (s & {"pause", "wait"}) - neg or re.search(r" (hold|hang) on | (just )?(a|one) (minute|moment|second|sec) ",
                                                      joined):
            add(1.0 if short or "pause" in s else 0.8, "pause")
        if (s & {"continue", "resume", "unpause"}) - neg:
            add(0.97, "continue")
        elif re.search(r" (go|keep|carry) (on|going) | go ahead | what happens next | what happened next | where were we "
                       r"| and then what ", joined):
            add(1.0 if short else 0.8, "continue")
        if (s & {"quit", "exit"}) - neg:
            add(1.0, "quit")
        elif re.search(r" go to sleep | sleep now | that is all | i am done | we are done | all done | i am finished "
                       r"| we are finished | turn (yourself )?off | switch (yourself )?off | shut down | power off ", joined):
            add(1.0 if n <= 5 else 0.8, "quit")
        if "louder" in s or joined.strip() == "loud" or re.search(
                r" (volume|turn it|turn the volume|speak|talk) up | too (quiet|soft|low) | can not hear | make it louder ",
                joined):
            add(1.0 if short else 0.85, "volume_up")
        if s & {"softer", "quieter"} or re.search(
                r" (volume|turn it|turn the volume) down | too loud | not so loud | make it (softer|quieter) ", joined):
            add(1.0 if short else 0.85, "volume_down")
        if "repeat" in s or re.search(r" (say|do|tell|read) (it|that|this) again | one more time |^ again | again $"
                                      r"| what did you say | what was that | pardon | come again | did not hear "
                                      r"| say again ", joined):
            then = None
            if s & CONTENT_WORDS:  # "tell me the story again": repeat it if it was the last one, else tell one
                rest = [t for t in toks if t not in ("repeat", "again")]
                then = self._decide(raw, rest, conf) if rest else None
            if then is not None and then.name not in ("unknown", "none", "repeat", "curiosity"):
                add(0.95, "repeat", then=then)
            elif n <= 6:
                add(1.0, "repeat")
        if (s & {"another", "next"} or re.search(r" one more ", joined)) and n <= 4 and not (s & CONTENT_WORDS) \
                and " one more time " not in joined:
            add(0.95, "more")
        if re.search(r" what can you do | what do you do | help me | what can i say | what games ", joined) \
                or joined.strip() == "help":
            add(0.95 if short else 0.7, "help")  # "help me count to ten" is counting

        # ---- maths: "what is 3 plus 4", "take away 2 from 5", "5 - 3" (never while counting)
        counting = s & {"count", "counting", "countdown"}
        ms = None if counting else math_slots(toks)
        if ms:
            add(0.95, "math", **ms)

        # ---- quiz / game
        if s & {"quiz", "game", "games", "test"} or re.search(r" ask me | question for me | quiz me ", joined):
            add(0.92, "quiz", domain=self._quiz_domain(s, toks))
        elif re.search(r"^ (let us|can we|shall we|i want to|want to) play( with me| something| a game)? $"
                       r"|^ play with me $", joined):
            add(0.85, "quiz")  # "let's play!"
        if s & {"riddle", "riddles", "puzzle"}:
            add(0.9, "riddle")
        if s & {"joke", "jokes"}:
            add(0.9, "joke")
        elif "funny" in s or re.search(r" make me laugh | something silly ", joined):
            add(0.85, "joke")

        # ---- times tables
        if s & {"table", "tables", "multiplication"} and not (s & {"color", "colors", "colour", "colours"}):
            add(0.9, "table", n=nums[0] if nums else None)

        # ---- counting: any range, step and direction (nlu/slots.py)
        if counting or (s & {"numbers", "number"} and "quiz" not in s):
            add(0.93 if counting else 0.75, "count", **count_slots(toks).as_dict())
        elif not ms and number_range(toks):
            add(0.6, "count", **count_slots(toks).as_dict())  # "...from one to ten" even if "count" was misheard

        # ---- phonics / letters
        letter = self.find_letter(raw, toks)
        if "phonics" in s or re.search(r" letter sounds? ", joined):
            add(0.9, "phonics", letter=letter)
        elif letter:
            add(0.88, "phonics", letter=letter)
        if s & {"alphabet", "alphabets", "abc", "abcs"} or re.search(r" (the )?letters ", joined):
            add(0.9, "alphabet")

        # ---- lessons
        if s & {"color", "colors", "colour", "colours"}:
            thing = next((t for t in toks if t in {f for f, _ in self.content.lessons["fruit_colors"]}), None)
            if thing and ("what" in raw or "which" in raw):
                add(0.92, "color_of", thing=thing)
            else:
                add(0.88, "colors")
        if s & {"shape", "shapes"} or (s & self.shape_names and s & {"what", "teach", "learn", "tell"}):
            add(0.88, "shapes", shape=next((t for t in toks if t in self.shape_names), None))
        if "body" in s or re.search(r" body parts? ", joined):
            add(0.88, "body")
        elif s & self.body_words and clearly(self.body_words) >= 0.99 \
                and not [t for t in toks if t not in STOPWORDS and t not in self.body_words]:
            # "where is my nose", "show me my ears", "my tummy" - but not a passing mention in a
            # sentence about something else ("set a brush teeth timer", "i brush my teeth").
            part = next((t for t in toks if t in self.body_names), None) \
                or next((BODY_SINGULAR[t] for t in toks if t in BODY_SINGULAR), None)
            add(0.88, "body", part=part)
        if s & {"days", "week"}:
            add(0.88, "days")
        if s & {"months", "month"} or re.search(r" months of the year ", joined):
            add(0.88, "months")
        if s & {"opposite", "opposites"}:
            word = next((t for t in toks if t in self.opposite_of and t not in ("in", "up", "out", "down")), None) \
                or next((t for t in reversed(toks) if t in self.opposite_of), None)
            if word and "opposite" in s and "of" in toks:
                add(0.93, "opposite_of", word=word)
            else:
                add(0.88, "opposites")

        # ---- animals
        animal = self.find_animal(toks)
        sound_cue = s & {"sound", "sounds", "noise"}
        if not animal and sound_cue and not s & {"letter", "letters", "phonics", "animal", "animals"}:
            soft = self.find_animal_soft(raw)
            if soft:
                add(0.85, "animal_sound", animal=soft)
        if animal and re.search(r" (tell me )?about | facts? ", joined) and not sound_cue and not s & CONTENT_WORDS:
            add(0.9, "animal_fact", animal=animal)  # (not "a story about a lion")
        if animal and (sound_cue or re.search(r" (what|how) (does|do|is) (the |a )?\w+ (say|saying|go|going) | says? ",
                                              joined)):
            add(0.93, "animal_sound", animal=animal)
        elif animal and raw[0] != "why" and (set(raw) & self.sound_verbs):
            add(0.9, "animal_sound", animal=animal)  # "how does a dog bark", "make the lion roar"
        elif s & {"animal", "animals"}:
            add(0.85, "animal_sound", animal=animal)
        elif animal and n <= 3 and clearly(self.animal_by_name) >= 0.99:
            # just "monkey!" (beats the poem/story title match).  Only when the name was said as
            # it is: a guessed-at name turned "order a pizza" into a horse and "abra ka dabra"
            # into a snake (finding M4).
            add(0.82, "animal_sound", animal=animal)

        # ---- wellbeing, manners
        if s & {"breathe", "breathing", "calm", "relax"}:
            add(0.88, "breathing")
        if "manners" in s or re.search(r" magic words ", joined):
            add(0.88, "manners")
        feel = self._feeling(raw, s)
        if feel:
            add(0.88, "feeling", mood=feel)

        # ---- small talk
        if re.search(r" (what is|tell me) your name | who are you ", joined):
            add(0.9, "name")
        if re.search(r" how old are you ", joined):
            add(0.9, "age")
        if re.search(r" how are you ", joined):
            add(0.9, "how_are_you")
        if re.search(r" i love you ", joined):
            add(0.9, "love")
        if s & {"thank", "thanks"}:
            add(0.85, "thanks")
        if "sorry" in s:
            add(0.8, "sorry")
        if "morning" in s and "good" in raw:
            add(0.85, "good_morning")
        if s & {"goodnight", "night"} and ("good" in raw or "goodnight" in s):
            add(0.85, "good_night")
        if s & {"bye", "goodbye"}:
            add(0.85, "goodbye")
        if s & {"hello", "hi"} and n <= 3:
            add(0.7, "hello")

        # ---- open "why / how / where / when" questions -> small FAQ or curiosity response
        if is_question:
            faq = self.match_faq(raw)
            if faq is not None:
                add(0.87, "faq", index=faq)  # lessons (e.g. "what colour is the sky") answer first
            elif raw[0] in ("why", "how", "where", "when") or "wonder" in raw:
                add(0.4, "curiosity")

        # ---- stories / poems / songs (a title, a topic after "about", or a random one)
        # (a "why/how/where/when" question is answered above and never turns into a story)
        if raw[0] in ("why", "how", "where", "when") or "wonder" in raw:
            return pick(cands)
        kind, tid, score = self.best_title(toks)
        topic = self._topic(toks)
        if s & STORY_WORDS:
            sid = tid if kind == "story" and score >= 0.8 else None
            add(0.9 * clearly(STORY_WORDS), "story", id=sid, topic=None if sid else topic)
        if s & POEM_WORDS:
            pid = tid if kind == "poem" and score >= 0.8 else None
            add(0.9 * clearly(POEM_WORDS), "poem", id=pid, topic=None if pid else topic)
        if s & SONG_WORDS:
            pid = tid if kind == "poem" and score >= 0.8 else None
            add(0.9 * clearly(SONG_WORDS), "song", id=pid, topic=None if pid else topic)
        if kind and score >= 0.9 and not is_question:
            add(0.8, kind, id=tid)
        if s & {"teach", "learn"}:
            add(0.5, "help")
        if is_question:
            add(0.35, "curiosity")

        return pick(cands)

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
        return bool(set(normalize(text)) & {"yes", "yeah", "yep", "ok", "okay", "sure", "ya", "haan", "yup"})

    @staticmethod
    def is_no(text: str) -> bool:
        return bool(set(normalize(text)) & {"no", "nope", "nah", "stop", "enough"})
