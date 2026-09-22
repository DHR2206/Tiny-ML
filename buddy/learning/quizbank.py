"""Question banks for the adaptive quiz, generated from the content packs.

Every question has a fixed text so it can be pre-rendered to audio.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ..content_store import Content
from ..nlu.normalize import number_to_words
from .progress import Progress

# the word a child is most likely to say for each animal sound
ANIMAL_ANSWERS = {
    "cow": ["moo"], "dog": ["woof", "bark", "bow wow"], "cat": ["meow", "miaow", "mew"],
    "lion": ["roar"], "tiger": ["roar", "grr"], "elephant": ["pawoo", "trumpet", "toot"],
    "monkey": ["ooh ooh", "aah", "ooh"], "duck": ["quack"], "sheep": ["baa", "ba"],
    "goat": ["maa", "meh", "baa"], "horse": ["neigh", "nay"], "pig": ["oink"],
    "hen": ["cluck", "bawk"], "rooster": ["cock a doodle doo", "doodle"], "donkey": ["hee haw", "haw"],
    "frog": ["ribbit", "croak"], "snake": ["hiss", "sss"], "bee": ["buzz", "bzz"], "owl": ["hoo", "who"],
    "crow": ["caw", "kaw"], "parrot": ["squawk", "hello"], "bird": ["tweet", "chirp"],
    "mouse": ["squeak"], "wolf": ["howl", "aooo"],
}

LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class Question:
    domain: str
    id: str
    level: int
    text: str
    answers: list  # str or int
    say_answer: str  # "The answer is <say_answer>."
    hint: str = ""
    extra: dict = field(default_factory=dict)


def build_bank(content: Content) -> dict[str, list[Question]]:
    bank: dict[str, list[Question]] = {"animals": [], "numbers": [], "letters": [], "colors": [],
                                       "shapes": [], "opposites": []}
    # ---- animals
    for a in content.animals:
        if a["id"] not in ANIMAL_ANSWERS:
            continue
        nm = a["names"][0]
        lvl = 1 if a["id"] in ("cow", "dog", "cat", "duck", "sheep", "lion") else 2
        bank["animals"].append(Question("animals", f"say_{a['id']}", lvl, f"What does the {nm} say?",
                                        ANIMAL_ANSWERS[a["id"]], a["sound"], hint=f"Think about the {nm} on a farm or in a forest. What sound does it make?"))
        first = ANIMAL_ANSWERS[a["id"]][0]
        bank["animals"].append(Question("animals", f"who_{a['id']}", lvl + 1, f"Which animal says {first}?",
                                        list(a["names"]), f"the {nm}", hint=f"Listen: {a['sound']} Who makes that sound?"))
    # ---- numbers
    for n in range(0, 100):
        lvl = 1 if n < 10 else 2 if n < 20 else 3 if n < 50 else 4
        w = number_to_words(n)
        bank["numbers"].append(Question("numbers", f"after_{n}", lvl, f"What number comes after {w}?", [n + 1],
                                        number_to_words(n + 1), hint=f"Count with me: {w}, and then?"))
    for n in range(1, 21):
        w = number_to_words(n)
        bank["numbers"].append(Question("numbers", f"before_{n}", 2 if n <= 10 else 3,
                                        f"What number comes before {w}?", [n - 1], number_to_words(n - 1),
                                        hint="Try counting backwards."))
    for a in range(0, 11):
        for b in range(0, 11):
            if a + b <= 10:
                lvl = 3 if a + b <= 5 else 4
                bank["numbers"].append(Question("numbers", f"add_{a}_{b}", lvl,
                                                f"What is {number_to_words(a)} plus {number_to_words(b)}?",
                                                [a + b], number_to_words(a + b),
                                                hint=f"Hold up {number_to_words(a)} fingers, then {number_to_words(b)} more. How many?"))
            if a >= b and a <= 10:
                bank["numbers"].append(Question("numbers", f"sub_{a}_{b}", 5,
                                                f"What is {number_to_words(a)} minus {number_to_words(b)}?",
                                                [a - b], number_to_words(a - b),
                                                hint=f"Start with {number_to_words(a)} and take away {number_to_words(b)}."))
    # ---- letters
    for i, l in enumerate(LETTERS[:-1]):
        nxt = LETTERS[i + 1]
        bank["letters"].append(Question("letters", f"after_{l}", 2 if i < 12 else 3,
                                        f"Which letter comes after {l.upper()}?", [f"letter:{nxt}"], nxt.upper(),
                                        hint="Let's sing the alphabet in your head. What comes next?"))
    for item in content.lessons["letters"]:
        if item["l"] == "x":
            continue
        bank["letters"].append(Question("letters", f"starts_{item['word']}", 1 if item["l"] in "abcdmst" else 2,
                                        f"What letter does {item['word']} start with?", [f"letter:{item['l']}"],
                                        item["l"].upper(),
                                        hint=f"Listen to the first sound. {item['sound']}, {item['sound']}, {item['word']}."))
    # ---- colours
    for thing, col in content.lessons["fruit_colors"]:
        bank["colors"].append(Question("colors", f"color_{thing}", 1, f"What colour is {_article(thing)}?", [col], col,
                                       hint=f"Close your eyes and picture {_article(thing)}."))
    # ---- shapes
    for s in content.lessons["shapes"]:
        if s["name"] in ("triangle", "square", "rectangle"):
            bank["shapes"].append(Question("shapes", f"sides_{s['name']}", 2,
                                           f"How many sides does a {s['name']} have?", [s["sides"]],
                                           number_to_words(s["sides"]), hint=s["desc"]))
    bank["shapes"].append(Question("shapes", "round", 1, "Which shape is round, like a ball?", ["circle"], "a circle",
                                   hint="It has no corners at all!"))
    bank["shapes"].append(Question("shapes", "three_sides", 2, "Which shape has three sides?", ["triangle"],
                                   "a triangle", hint="Think of a slice of pizza!"))
    # ---- opposites
    for a, b in content.lessons["opposites"]:
        bank["opposites"].append(Question("opposites", f"opp_{a}", 2 if a in ("big", "hot", "up", "day", "happy") else 3,
                                          f"What is the opposite of {a}?", [b], b,
                                          hint=f"If something is not {a}, it is..."))
    return bank


def _article(thing: str) -> str:
    if thing in ("grass", "milk", "sky", "chocolate", "grapes"):
        return "the " + thing if thing in ("sky", "grass") else thing
    return ("an " if thing[0] in "aeiou" else "a ") + thing


class QuizPicker:
    """Chooses the next question: right level, weakest items first, no repeats in a session."""

    def __init__(self, bank: dict[str, list[Question]], progress: Progress, rng: random.Random | None = None):
        self.bank = bank
        self.progress = progress
        self.rng = rng or random.Random()

    def domains(self) -> list[str]:
        return [d for d, qs in self.bank.items() if qs]

    def pick(self, domain: str, exclude: set[str]) -> Question:
        lvl = self.progress.level(domain)
        boxes = self.progress.boxes(domain)
        cands = [q for q in self.bank[domain] if q.level <= lvl and q.id not in exclude]
        if not cands:
            cands = [q for q in self.bank[domain] if q.id not in exclude] or self.bank[domain]
        now = time.time()

        def weight(q: Question) -> float:
            box, seen = boxes.get(q.id, (0, 0.0))
            w = 1.0 / (1 + box) ** 2
            if q.level == lvl:
                w *= 2.0  # focus on the current level
            if now - seen < 120:
                w *= 0.2
            return w

        weights = [weight(q) for q in cands]
        return self.rng.choices(cands, weights=weights, k=1)[0]
