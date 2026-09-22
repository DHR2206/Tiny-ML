"""All of Buddy's skills.  Each handler turns an Intent into a Response.

`all_phrases()` enumerates every sentence any handler can ever say, so the whole
speech inventory can be synthesised once at setup time (closed-domain design).
"""
from __future__ import annotations

import random
from typing import Callable

from ..content_store import Content
from ..core import Earcon, Expect, Pause, Response, Say, Sound
from ..learning.progress import Progress
from ..learning.quizbank import LETTERS, Question, QuizPicker, build_bank
from ..nlu.matcher import LETTER_WORDS, Intent, Matcher
from ..nlu.normalize import normalize, number_to_words

QUIZ_LEN = 5
MAX_COUNT = 100


def dot(s: str) -> str:
    return s if s.rstrip()[-1:] in ".!?" else s + "."


def num(n: int) -> str:
    return number_to_words(n).capitalize() + "."


class Skills:
    def __init__(self, content: Content, matcher: Matcher, progress: Progress, rng: random.Random | None = None):
        self.c = content
        self.m = matcher
        self.progress = progress
        self.rng = rng or random.Random()
        self.bank = build_bank(content)
        self.picker = QuizPicker(self.bank, progress, self.rng)
        self.unknown_streak = 0
        self.handlers: dict[str, Callable[[Intent], Response]] = {
            name[3:]: getattr(self, name) for name in dir(self) if name.startswith("do_")
        }

    # ------------------------------------------------------------- utilities
    def p(self, key: str) -> str:
        return self.rng.choice(self.c.phrases[key])

    def handle(self, intent: Intent) -> Response:
        fn = self.handlers.get(intent.name, self.do_unknown)
        r = fn(intent)
        r.intent = r.intent or intent.name
        if intent.name != "unknown":
            self.unknown_streak = 0
        return r

    # ------------------------------------------------------------- controls
    def do_stop(self, i: Intent) -> Response:
        return Response([Say(self.p("stop"))], action="stop", repeatable=False)

    def do_volume_up(self, i: Intent) -> Response:
        return Response([], action="volume_up", repeatable=False)

    def do_volume_down(self, i: Intent) -> Response:
        return Response([], action="volume_down", repeatable=False)

    def do_repeat(self, i: Intent) -> Response:
        return Response([], action="repeat", repeatable=False)

    def do_more(self, i: Intent) -> Response:
        return Response([], action="more", repeatable=False)

    def do_help(self, i: Intent) -> Response:
        return Response([Say(self.p("help"))])

    # ------------------------------------------------------------- stories & poems
    def do_story(self, i: Intent) -> Response:
        sid = i.slots.get("id")
        story = next((s for s in self.c.stories if s["id"] == sid), None) or self.rng.choice(self.c.stories)
        r = Response([Say(f"{self.p('story_intro')} {story['title']}.", 0.6)], meta={"id": story["id"]})
        for para in story["paragraphs"]:
            r.items.append(Say(para, 0.55))
        return r

    def _poem(self, poem: dict, intro_key: str) -> Response:
        r = Response([Say(f"{self.p(intro_key)} {poem['title']}.", 0.6)], meta={"id": poem["id"]})
        for line in poem["lines"]:
            r.items.append(Say(line, 0.35))
        return r

    def do_poem(self, i: Intent) -> Response:
        pid = i.slots.get("id")
        poem = next((p for p in self.c.poems if p["id"] == pid), None) or self.rng.choice(self.c.poems)
        return self._poem(poem, "poem_intro")

    def do_song(self, i: Intent) -> Response:
        # songs were removed from the prototype: recite a poem instead
        pid = i.slots.get("id")
        poem = next((p for p in self.c.poems if p["id"] == pid), None) or self.rng.choice(self.c.poems)
        r = self._poem(poem, "song_redirect")
        r.intent = "poem"
        return r

    # ------------------------------------------------------------- letters
    def do_alphabet(self, i: Intent) -> Response:
        r = Response([Say(self.p("alphabet_intro"), 0.5)])
        for chunk in ("A, B, C, D, E, F, G.", "H, I, J, K, L, M, N, O, P.", "Q, R, S, T, U, V.",
                      "W, X, Y, and Z."):
            r.items.append(Say(chunk, 0.45))
        r.items.append(Say(self.p("alphabet_outro")))
        return r

    def _letter_line(self, item: dict) -> str:
        L = item["l"].upper()
        if item["l"] == "x":
            return f"{L} says {item['sound']}, like the end of {item['word']}."
        return f"{L} is for {item['word']}. {L} says {item['sound']}, {item['sound']}, {item['word']}!"

    def do_phonics(self, i: Intent) -> Response:
        letter = i.slots.get("letter")
        items = self.c.lessons["letters"]
        if letter:
            item = next(x for x in items if x["l"] == letter)
            return Response([Say(self._letter_line(item))], meta={"letter": letter})
        r = Response([Say(self.p("phonics_intro"), 0.5)])
        for item in items:
            r.items.append(Say(self._letter_line(item), 0.4))
        return r

    # ------------------------------------------------------------- numbers
    def do_count(self, i: Intent) -> Response:
        step = int(i.slots.get("step") or 1)
        step = step if step in (1, 2, 5, 10) else 1
        back = bool(i.slots.get("backwards"))
        start = i.slots.get("start")
        end = i.slots.get("end")
        r = Response([Say(self.p("counting_intro"), 0.4)])
        if back:
            start = start if start is not None else (end if end is not None else 10)
            start = min(int(start), MAX_COUNT)
            seq = list(range(start, -1, -step))
        else:
            first = int(start) if start is not None else (step if step > 1 else 1)
            last = int(end) if end is not None else (100 if step > 1 else 10 if end is None and start is None else first + 10)
            if last > MAX_COUNT:
                r.items.append(Say("I can count up to one hundred. Let's go!", 0.4))
                last = MAX_COUNT
            if last < first:
                seq = list(range(first, last - 1, -step))
            else:
                seq = list(range(first, last + 1, step))
        for k, n in enumerate(seq):
            r.items.append(Say(num(n), 0.12))
            if not back and step == 1 and n % 10 == 0 and n not in (0, seq[-1]) and len(seq) > 20:
                r.items.append(Pause(0.3))
        if back and seq and seq[-1] == 0:
            r.items.append(Say("Blast off!", 0.3))
        r.items.append(Say(self.p("counting_outro")))
        return r

    def do_table(self, i: Intent) -> Response:
        n = i.slots.get("n")
        if not n or not 1 <= n <= 10:
            n = self.rng.randint(2, 5)
        r = Response([Say(f"The table of {number_to_words(n)}.", 0.5)])
        for k in range(1, 11):
            r.items.append(Say(self.times_line(n, k), 0.25))
        r.items.append(Say(self.p("counting_outro")))
        return r

    @staticmethod
    def times_line(a: int, b: int) -> str:
        return f"{number_to_words(a).capitalize()} times {number_to_words(b)} is {number_to_words(a * b)}."

    def do_math(self, i: Intent) -> Response:
        a, b, op = i.slots["a"], i.slots["b"], i.slots["op"]
        if not (0 <= a <= 10 and 0 <= b <= 10):
            return Response([Say("Wow, those are big numbers! I can do sums with numbers up to ten.")])
        if op == "plus":
            return Response([Say(f"{number_to_words(a).capitalize()} plus {number_to_words(b)} is {number_to_words(a + b)}!")])
        if op == "minus":
            if b > a:
                return Response([Say("Hmm, that would go below zero! Let's try a smaller number to take away.")])
            return Response([Say(f"{number_to_words(a).capitalize()} minus {number_to_words(b)} is {number_to_words(a - b)}!")])
        return Response([Say(self.times_line(a, b))])

    # ------------------------------------------------------------- animals
    def _animal_items(self, a: dict, with_fact: bool = True) -> list:
        nm = a["names"][0]
        items = [Say(f"The {nm} says... {a['sound']}", 0.3), Sound(a["id"])]
        if with_fact:
            items.append(Say(a["fact"], 0.5))
        return items

    def do_animal_sound(self, i: Intent) -> Response:
        aid = i.slots.get("animal")
        if aid:
            return Response(self._animal_items(self.c.animal(aid)), meta={"animal": aid})
        r = Response([Say(self.p("animal_intro"), 0.4)])
        for a in self.rng.sample(self.c.animals, 5):
            r.items.extend(self._animal_items(a, with_fact=False))
            r.items.append(Pause(0.4))
        return r

    def do_animal_fact(self, i: Intent) -> Response:
        a = self.c.animal(i.slots["animal"])
        return Response([Say(a["fact"], 0.3)] + self._animal_items(a, with_fact=False))

    # ------------------------------------------------------------- lessons
    def do_colors(self, i: Intent) -> Response:
        r = Response([Say(self.p("colors_intro"), 0.4)])
        for col in self.c.lessons["colors"]:
            r.items.append(Say(self.color_line(col), 0.45))
        return r

    @staticmethod
    def color_line(col: dict) -> str:
        t = col["things"]
        return f"{col['name'].capitalize()}! {t[0].capitalize()} is {col['name']}, and {t[1]} is {col['name']} too."

    def do_color_of(self, i: Intent) -> Response:
        thing = i.slots["thing"]
        col = dict(self.c.lessons["fruit_colors"])[thing]
        return Response([Say(self.color_fact(thing, col))])

    @staticmethod
    def color_fact(thing: str, col: str) -> str:
        from ..learning.quizbank import _article
        return f"{_article(thing).capitalize()} is {col}!"

    def do_shapes(self, i: Intent) -> Response:
        shape = i.slots.get("shape")
        shapes = self.c.lessons["shapes"]
        if shape:
            return Response([Say(next(s for s in shapes if s["name"] == shape)["desc"])])
        r = Response([Say(self.p("shapes_intro"), 0.4)])
        for s in shapes:
            r.items.append(Say(s["desc"], 0.45))
        return r

    def do_body(self, i: Intent) -> Response:
        r = Response([Say(self.p("body_intro"), 0.4)])
        for b in self.c.lessons["body_parts"]:
            r.items.append(Say(b["use"], 0.4))
        return r

    def do_days(self, i: Intent) -> Response:
        return Response([Say(self.p("days_intro"), 0.4), Say(", ".join(self.c.lessons["days"]) + ".")])

    def do_months(self, i: Intent) -> Response:
        m = self.c.lessons["months"]
        return Response([Say(self.p("months_intro"), 0.4), Say(", ".join(m[:6]) + ",", 0.2),
                         Say(", ".join(m[6:]) + ".")])

    def do_opposites(self, i: Intent) -> Response:
        r = Response([Say(self.p("opposites_intro"), 0.4)])
        for a, b in self.c.lessons["opposites"]:
            r.items.append(Say(f"{a.capitalize()}, and {b}.", 0.35))
        return r

    # ------------------------------------------------------------- fun
    def do_joke(self, i: Intent) -> Response:
        return Response([Say(self.p("joke_intro"), 0.3), Say(self.rng.choice(self.c.talk["jokes"]))])

    def do_riddle(self, i: Intent) -> Response:
        rd = self.rng.choice(self.c.talk["riddles"])

        def reveal(prefix: str) -> Response:
            return Response([Say(f"{prefix} {rd['say']}")], repeatable=False)

        def handler(tokens: list[str], text: str) -> Response:
            if self.m.matches_answer(text, rd["a"]):
                return Response([Earcon("correct"), Say(self.p("praise"), 0.2), Say(rd["say"])], repeatable=False)
            return reveal(self.p("reveal"))

        return Response([Say(self.p("riddle_intro"), 0.3), Say(rd["q"])],
                        expect=Expect(handler, on_timeout=lambda: reveal("The answer is")))

    # ------------------------------------------------------------- quiz (adaptive)
    def do_quiz(self, i: Intent) -> Response:
        domain = i.slots.get("domain") or self.rng.choice(self.picker.domains())
        state = {"domain": domain, "asked": set(), "right": 0, "n": 0}
        return self._ask(state, intro=self.p("quiz_intro"))

    def _ask(self, state: dict, intro: str | None = None) -> Response:
        q = self.picker.pick(state["domain"], state["asked"])
        state["asked"].add(q.id)
        state["n"] += 1
        state["tries"] = 0
        items = ([Say(intro, 0.4)] if intro else []) + [Say(q.text)]
        return Response(items, expect=self._quiz_expect(state, q), repeatable=False, intent="quiz")

    def _quiz_expect(self, state: dict, q: Question) -> Expect:
        def next_or_end(r: Response) -> Response:
            if state["n"] >= QUIZ_LEN:
                key = "quiz_end" if state["right"] else "quiz_end_zero"
                r.items.append(Say(self.p(key).replace("{n}", number_to_words(state["right"]))))
                return r
            nxt = self._ask(state, intro=self.p("quiz_next"))
            r.items.extend(nxt.items)
            r.expect = nxt.expect
            return r

        def handler(tokens: list[str], text: str) -> Response:
            if self.check_answer(text, q.answers):
                first = state["tries"] == 0
                state["right"] += 1 if first else 0
                self.progress.record(q.domain, q.id, True, first_try=first)
                return next_or_end(Response([Earcon("correct"), Say(self.p("praise"), 0.4)], repeatable=False))
            if self.m.is_no(text) and len(normalize(text)) <= 2 and "stop" in normalize(text):
                return Response([Say(self.p("stop"))], repeatable=False)
            state["tries"] += 1
            if state["tries"] == 1 and not self.m.is_dont_know(text):
                r = Response([Say(self.p("try_again"), 0.3), Say(q.hint, 0.3), Say(q.text)], repeatable=False)
                r.expect = Expect(handler, on_timeout=timeout)
                return r
            self.progress.record(q.domain, q.id, False)
            return next_or_end(Response([Say(f"{self.p('reveal')} {dot(q.say_answer)}", 0.5)], repeatable=False))

        def timeout() -> Response:
            return Response([Say(f"The answer is {dot(q.say_answer)}", 0.4), Say(self.p("no_command"))], repeatable=False)

        return Expect(handler, on_timeout=timeout, data={"question": q.id})

    def check_answer(self, text: str, answers: list) -> bool:
        letters = [a.split(":", 1)[1] for a in answers if isinstance(a, str) and a.startswith("letter:")]
        if letters:
            toks = normalize(text)
            heard = {LETTER_WORDS.get(t) for t in toks} | {t for t in toks if len(t) == 1}
            return any(l in heard for l in letters)
        return self.m.matches_answer(text, answers)

    # ------------------------------------------------------------- wellbeing
    def do_breathing(self, i: Intent) -> Response:
        r = Response([])
        for line in self.c.phrases["breathing"]:
            r.items.append(Say(line, 2.0))
        return r

    def do_manners(self, i: Intent) -> Response:
        return Response([Say(self.p("manners"))])

    def do_feeling(self, i: Intent) -> Response:
        return Response([Say(self.p("feel_" + i.slots["mood"]))])

    # ------------------------------------------------------------- small talk
    def _simple(self, key: str) -> Callable[[Intent], Response]:
        return lambda i: Response([Say(self.p(key))])

    def do_hello(self, i): return Response([Say(self.p("hello"))])
    def do_good_morning(self, i): return Response([Say(self.p("good_morning"))])
    def do_good_night(self, i): return Response([Say(self.p("good_night"))])
    def do_goodbye(self, i): return Response([Say(self.p("goodbye"))])
    def do_thanks(self, i): return Response([Say(self.p("thanks"))])
    def do_love(self, i): return Response([Say(self.p("love"))])
    def do_name(self, i): return Response([Say(self.p("name"))])
    def do_age(self, i): return Response([Say(self.p("age"))])
    def do_how_are_you(self, i): return Response([Say(self.p("how_are_you"))])
    def do_sorry(self, i): return Response([Say(self.p("sorry"))])

    # ------------------------------------------------------------- questions
    def do_faq(self, i: Intent) -> Response:
        return Response([Say(self.c.talk["faq"][i.slots["index"]]["a"], 0.4), Say("Isn't that amazing?")])

    def do_curiosity(self, i: Intent) -> Response:
        return Response([Say(self.p("curiosity"))])

    def do_unknown(self, i: Intent) -> Response:
        self.unknown_streak += 1
        key = "not_understood_help" if self.unknown_streak >= 2 else "not_understood"
        return Response([Say(self.p(key))], repeatable=False)

    def do_none(self, i: Intent) -> Response:
        return Response([], repeatable=False)

    # ------------------------------------------------------------- inventory
    def all_phrases(self) -> set[str]:
        """Every sentence Buddy can say (used to pre-render the speech cache)."""
        c, out = self.c, set()
        for key, vals in c.phrases.items():
            if key in ("story_intro", "poem_intro", "song_redirect"):
                continue
            for v in vals:
                if "{n}" in v:
                    out.update(v.replace("{n}", number_to_words(k)) for k in range(0, QUIZ_LEN + 1))
                else:
                    out.add(v)
        for s in c.stories:
            out.update(f"{x} {s['title']}." for x in c.phrases["story_intro"])
            out.update(s["paragraphs"])
        for p in c.poems:
            out.update(f"{x} {p['title']}." for x in c.phrases["poem_intro"] + c.phrases["song_redirect"])
            out.update(p["lines"])
        out.update(["A, B, C, D, E, F, G.", "H, I, J, K, L, M, N, O, P.", "Q, R, S, T, U, V.", "W, X, Y, and Z."])
        out.update(self._letter_line(x) for x in c.lessons["letters"])
        out.update(num(n) for n in range(0, MAX_COUNT + 1))
        out.update(["I can count up to one hundred. Let's go!", "Blast off!"])
        for n in range(1, 11):
            out.add(f"The table of {number_to_words(n)}.")
        for a in range(0, 11):
            for b in range(0, 11):
                out.add(self.times_line(a, b))
                out.add(f"{number_to_words(a).capitalize()} plus {number_to_words(b)} is {number_to_words(a + b)}!")
                if a >= b:
                    out.add(f"{number_to_words(a).capitalize()} minus {number_to_words(b)} is {number_to_words(a - b)}!")
        out.update(["Wow, those are big numbers! I can do sums with numbers up to ten.",
                    "Hmm, that would go below zero! Let's try a smaller number to take away."])
        for a in c.animals:
            out.add(f"The {a['names'][0]} says... {a['sound']}")
            out.add(a["fact"])
        out.update(self.color_line(col) for col in c.lessons["colors"])
        out.update(self.color_fact(t, col) for t, col in c.lessons["fruit_colors"])
        out.update(s["desc"] for s in c.lessons["shapes"])
        out.update(b["use"] for b in c.lessons["body_parts"])
        out.add(", ".join(c.lessons["days"]) + ".")
        m = c.lessons["months"]
        out.update([", ".join(m[:6]) + ",", ", ".join(m[6:]) + "."])
        out.update(f"{a.capitalize()}, and {b}." for a, b in c.lessons["opposites"])
        out.update(c.talk["jokes"])
        for rd in c.talk["riddles"]:
            out.add(rd["q"])
            out.add(rd["say"])
            out.update(f"{x} {rd['say']}" for x in c.phrases["reveal"] + ["The answer is"])
        for f in c.talk["faq"]:
            out.add(f["a"])
        out.add("Isn't that amazing?")
        for qs in self.bank.values():
            for q in qs:
                out.update([q.text, q.hint, f"The answer is {dot(q.say_answer)}"])
                out.update(f"{x} {dot(q.say_answer)}" for x in c.phrases["reveal"])
        return {t for t in out if t}
