"""Wake-phrase detection on recognised text ("Hey Buddy, play a story").

Instead of a separately trained wake-word network (the popular open ones ship with
non-commercial model licences), Buddy runs its on-device recogniser on each
voice-activity segment and checks, in phoneme space with the child-phonology
distance, whether the segment *starts* with the wake phrase.  The same utterance
may already contain the command, so children can say it in one breath.
"""
from __future__ import annotations

from ..nlu.normalize import normalize
from ..nlu.phonetics import ChildPhonology, phrase_phonemes


class WakeMatcher:
    def __init__(self, phrases: list[str], phon: ChildPhonology, threshold: float = 0.8):
        self.phon = phon
        self.threshold = threshold
        self.targets = [(p, phrase_phonemes(p.split())) for p in phrases]
        # Called with the rest of the utterance when only the weak one-word form ("Buddy ...")
        # was heard; it must look like a real command (e.g. not "Buddy the dog is asleep").
        self.validator = None

    def split(self, text: str) -> tuple[bool, str, float]:
        """Returns (woke, remaining_command_text, score)."""
        toks = normalize(text)
        if not toks:
            return False, "", 0.0
        best = (0.0, 0, "")
        for k in (1, 2, 3):
            if k > len(toks):
                break
            ph = phrase_phonemes(toks[:k])
            for phrase, target in self.targets:
                # a one-word wake phrase must match a one-token prefix
                if len(phrase.split()) == 1 and k != 1:
                    continue
                s = self.phon.similarity(ph, target)
                if phrase == "buddy":
                    s -= 0.05  # bare "buddy" is less specific; require a closer match
                if s > best[0]:
                    best = (s, k, phrase)
        # recogniser sometimes glues the phrase into one token ("Heybody.")
        glued = phrase_phonemes(toks[:1])
        for phrase, target in self.targets:
            if len(phrase.split()) > 1 and len(glued) >= len(target) - 1:
                s = self.phon.similarity(glued, target) - 0.03
                if s > best[0]:
                    best = (s, 1, phrase)
        score, k, phrase = best
        if score >= self.threshold:
            rest = " ".join(toks[k:])
            weak = len(phrase.split()) == 1 or score < 0.9
            if weak and self.validator is not None and not self.validator(rest):
                return False, " ".join(toks), score
            return True, rest, score
        return False, " ".join(toks), score

    def find(self, text: str) -> tuple[bool, str, float]:
        """Wake phrase *anywhere* in the text -> (woke, words after it, score).

        Used while Buddy is talking: the microphone then also hears Buddy, so the recognised
        text may start with Buddy's own words ("...drank the water hey buddy pause").
        Only the two-word phrases count here; a bare "buddy" inside a sentence is not a wake-up.
        """
        toks = normalize(text)
        best = (0.0, 0, 0)
        for i in range(len(toks)):
            for k in (1, 2):
                if i + k > len(toks):
                    break
                ph = phrase_phonemes(toks[i:i + k])
                for phrase, target in self.targets:
                    if len(phrase.split()) == 1 or (k == 1 and len(ph) < len(target) - 1):
                        continue
                    s = self.phon.similarity(ph, target) - (0.03 if k == 1 else 0.0)  # glued "heybody"
                    if s > best[0]:
                        best = (s, i, k)
        score, i, k = best
        if score >= max(self.threshold, 0.85):
            return True, " ".join(toks[i + k:]), score
        return False, " ".join(toks), score
