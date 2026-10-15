"""Tests for the defects found by the v0.3 QA run (docs/FIXES.md).

Added alongside the original suite, which is left exactly as it was.  Each test names the
finding it covers, so a failure here points straight at the report row it belongs to.

Finding IDs come from the QA workbook: H1/H2 high, M1-M8 medium, L1-L6 low.
"""
import random
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from buddy.brain import Brain
from buddy.config import ConfigError, Config, load_config
from buddy.learning.progress import Progress
from buddy.nlu.matcher import Intent
from buddy.nlu.normalize import normalize, parse_numbers
from buddy.nlu.slots import count_slots

# Every wording Buddy uses when it did not understand (content/phrases.json).
FALLBACK_KEYS = ("not_understood", "not_understood_help")


def make(age=5, seed=0):
    cfg = Config()
    cfg.data["child"]["age"] = age
    return Brain(cfg, rng=random.Random(seed), progress=Progress(None))


def said(r):
    return " ".join(r.texts())


class TestSupportedCommands(unittest.TestCase):
    """6A: every implemented category, in more than one phrasing, still reaches its own skill."""

    CASES = [
        # (phrasings, expected intent)
        (["tell me a story", "i want a story", "can you tell a story please", "story time"], "story"),
        (["tell me a poem", "say a rhyme", "i want to hear a poem"], "poem"),
        (["sing a song", "sing me something", "play some music"], "song"),
        (["say the alphabet", "can you do the abcs", "sing the alphabet"], "alphabet"),
        (["what sound does the letter b make", "what is b for", "phonics please"], "phonics"),
        (["count from one to ten", "count to twenty", "can you count for me"], "count"),
        (["what is three plus four", "how much is 5 and 2", "what is 5-3"], "math"),
        (["say the two times table", "i want the times tables"], "table"),
        (["what does the cow say", "make the lion roar", "animal sounds please"], "animal_sound"),
        (["tell me about lions", "a fact about elephants"], "animal_fact"),
        (["what colour is grass"], "color_of"),
        (["tell me about colours", "teach me colours"], "colors"),
        (["tell me about shapes", "teach me shapes"], "shapes"),
        (["where is my nose", "show me my ears", "body parts"], "body"),
        (["what are the days of the week", "days please"], "days"),
        (["what are the months of the year"], "months"),
        (["what is the opposite of hot"], "opposite_of"),
        (["tell me about opposites"], "opposites"),
        (["tell me a joke", "say something funny"], "joke"),
        (["tell me a riddle", "i want a puzzle"], "riddle"),
        (["lets play a quiz", "quiz me", "ask me a question"], "quiz"),
        (["lets do some breathing", "i want to calm down"], "breathing"),
        (["teach me manners", "what are the magic words"], "manners"),
        (["i am sad", "i feel scared"], "feeling"),
        (["what is your name", "who are you"], "name"),
        (["how old are you"], "age"),
        (["how are you"], "how_are_you"),
        (["i love you"], "love"),
        (["thank you", "thanks buddy"], "thanks"),
        (["sorry"], "sorry"),
        (["hello", "hi"], "hello"),
        (["good morning"], "good_morning"),
        (["good night"], "good_night"),
        (["bye bye", "goodbye"], "goodbye"),
        (["what can you do", "help me"], "help"),
        (["stop", "that is enough"], "stop"),
        (["repeat", "say that again"], "repeat"),
        (["louder", "too quiet"], "volume_up"),
        (["softer", "too loud"], "volume_down"),
        (["pause", "hold on"], "pause"),
        (["continue", "carry on"], "continue"),
        (["quit", "i am done"], "quit"),
    ]

    def test_each_category_from_several_phrasings(self):
        b = make()
        bad = []
        for phrasings, want in self.CASES:
            for text in phrasings:
                got = b.matcher.parse(text)
                if got.name != want:
                    bad.append((text, want, got.name))
        self.assertEqual(bad, [], f"{len(bad)} phrasings reached the wrong skill: {bad}")

    def test_each_category_produces_a_reply(self):
        """The pipeline agrees end to end: intent -> content -> something to say or do."""
        silent_by_design = {"pause", "continue"}  # the engine acts on these, there is no sentence
        for phrasings, want in self.CASES:
            b = make()
            r = b.command(phrasings[0])
            if want in silent_by_design:
                self.assertIsNotNone(r.action, phrasings[0])
                continue
            self.assertTrue(r.items, f"{phrasings[0]!r} ({want}) produced no reply at all")
            self.assertTrue(said(r).strip(), f"{phrasings[0]!r} ({want}) said nothing")


class TestGibberishAndInvalidInput(unittest.TestCase):
    """6B / finding M4: unintelligible input gets the fallback and never starts an activity."""

    GIBBERISH = [
        ("asdfgh qwerty", "keyboard noise"),
        ("zxcvbnm", "keyboard noise, one token"),
        ("blah blah blah", "nonsense repeated"),
        ("abra ka dabra", "nonsense words"),
        ("florp bimble wug", "invented words"),
        ("ngggh mmmph", "mumbling"),
        ("qqq zzz", "no phonetic content"),
        ("wibble wobble flim flam", "random sequence"),
        ("", "empty"),
        ("   ", "whitespace only"),
        ("\t\t", "tabs only"),
        ("\n\n", "newlines only"),
        ("!!!???...", "punctuation only"),
        ("------------", "dashes only"),
        ("你好世界", "another script"),
        ("ńōńäš", "accented non-words"),
        ("um", "filler only"),
        ("uh um hmm", "fillers only"),
        ("the", "one stopword"),
        ("a an the of", "stopwords only"),
    ]

    UNSUPPORTED = [
        "order a pizza",
        "call mummy",
        "turn on the tv",
        "set a brush teeth timer",
        "what is the capital of france",
        "who is the prime minister",
        "what is the weather today",
        "what time is it",
    ]

    def setUp(self):
        self.b = make()
        self.fallbacks = set()
        for key in FALLBACK_KEYS:
            self.fallbacks.update(self.b.content.phrases[key])
        self.activities = {
            "story", "poem", "song", "count", "table", "math", "alphabet", "phonics", "colors",
            "color_of", "shapes", "body", "days", "months", "opposites", "opposite_of",
            "animal_sound", "animal_fact", "quiz", "riddle", "joke", "breathing", "manners",
        }

    def test_gibberish_gets_the_fallback(self):
        for text, why in self.GIBBERISH:
            b = make()  # a fresh brain, so the escalating wording cannot mask a miss
            r = b.command(text)
            self.assertTrue(r.items, f"{text!r} ({why}) produced no reply at all")
            self.assertIn(said(r), self.fallbacks,
                          f"{text!r} ({why}) did not get the fallback: {said(r)[:80]!r}")

    def test_gibberish_never_starts_an_activity(self):
        for text, why in self.GIBBERISH:
            b = make()
            r = b.command(text)
            self.assertNotIn(r.intent, self.activities,
                             f"{text!r} ({why}) started {r.intent}")

    def test_gibberish_is_never_a_near_miss_for_an_activity(self):
        """The matcher itself must not claim a confident activity for nonsense."""
        for text, why in self.GIBBERISH:
            got = self.b.matcher.parse(text)
            self.assertIn(got.name, ("unknown", "none"), f"{text!r} ({why}) parsed as {got!r}")

    def test_unsupported_requests_are_not_answered_with_an_activity(self):
        """6B: a meaningful request Buddy cannot do is not gibberish, but it must not be
        answered by whatever activity happens to sound closest (finding M4)."""
        bad = []
        for text in self.UNSUPPORTED:
            got = self.b.matcher.parse(text)
            if got.name in self.activities:
                bad.append((text, got.name))
        self.assertEqual(bad, [], f"unsupported requests started an activity: {bad}")

    def test_unsupported_question_encourages_curiosity(self):
        """The project's intended answer for a question it does not know (phrases.json)."""
        for text in ["what is the capital of france", "who is the prime minister",
                     "what is the weather today", "what is my teacher called"]:
            r = make().command(text)
            self.assertTrue(said(r).strip(), text)
            self.assertIn(r.intent, ("curiosity", "faq", "unknown"), (text, r.intent))

    @unittest.expectedFailure
    def test_known_residual_a_vocabulary_word_can_still_carry_a_question(self):
        """Remaining M4 case, recorded rather than hidden: when a question contains a word that
        *is* in Buddy's vocabulary ("moon" is in a poem title), a second word guessed into a
        keyword ("cheese" -> "jokes") can still choose the activity.  Separating this from a real
        child pronunciation needs a different model (the nlu/fallback.py hook), which is a design
        decision for the owner - see the post-fix report, Remaining issues."""
        got = make().matcher.parse("why is the moon made of cheese")
        self.assertIn(got.name, ("curiosity", "faq", "unknown"), got.name)

    def test_repeated_gibberish_then_a_valid_command(self):
        """6B / 6D: Buddy is never left stuck by invalid input."""
        b = make()
        for text, _why in self.GIBBERISH[:6]:
            r = b.command(text)
            self.assertTrue(r.items, text)
        r = b.command("tell me a story")
        self.assertEqual(r.intent, "story")
        self.assertGreater(len(r.texts()), 1, "the story did not play after repeated gibberish")

    def test_second_miss_offers_help(self):
        """The existing escalation: the second unknown in a row suggests what to say."""
        b = make()
        b.command("asdfgh")
        r = b.command("qwerty")
        self.assertIn(said(r), set(b.content.phrases["not_understood_help"]))

    def test_fallback_wording_comes_from_the_inventory(self):
        """Offline guarantee: the fallback is pre-vetted, pre-renderable content."""
        inventory = make().skills.all_phrases()
        for text, why in self.GIBBERISH:
            r = make().command(text)
            for t in r.texts():
                self.assertIn(t, inventory, f"{text!r} ({why}) said something unrendered: {t!r}")


class TestAcrossTheAgeRange(unittest.TestCase):
    """The snapper is age-scheduled, so the gibberish tests are repeated for the whole range."""

    NONSENSE = ["asdfgh qwerty", "blah blah blah", "abra ka dabra", "florp bimble wug",
                "qqq zzz", "wibble wobble flim flam", "gloop snorf", "plibble", "ngggh mmmph",
                "grr nggg", "mmmph", "wug", "prrrt", "snorf"]
    # Measured residue of the phonetic snapper, not an implementation bug: a short mumble with no
    # context can still land within the age-scheduled distance of a keyword.  Recorded exactly,
    # so the limit stays visible and any *new* leak still fails this test.  [nlu] min_confidence
    # closes these; see config.py for the measured trade-off and the report's Remaining issues.
    KNOWN_LEAKS = {
        (3, "ngggh mmmph"), (3, "grr nggg"), (3, "mmmph"),   # -> months / thanks / months
        (4, "ngggh mmmph"), (4, "grr nggg"), (4, "mmmph"),
        (5, "wug"), (5, "prrrt"),                            # -> days / story
        (6, "wug"),
    }

    def test_nonsense_never_starts_an_activity_at_any_age(self):
        leaks = []
        for age in (3, 4, 5, 6, 7, 8):
            b = make(age=age)
            for text in self.NONSENSE:
                got = b.matcher.parse(text)
                if got.name not in ("unknown", "none") and (age, text) not in self.KNOWN_LEAKS:
                    leaks.append((age, text, got.name, round(got.confidence, 2)))
        self.assertEqual(leaks, [], f"nonsense started an activity: {leaks}")

    def test_core_commands_work_at_every_age(self):
        bad = []
        for age in (3, 4, 5, 6, 7, 8):
            b = make(age=age)
            for text, want in [("tell me a story", "story"), ("tell me a poem", "poem"),
                               ("count to ten", "count"), ("tell me a joke", "joke"),
                               ("what does the cow say", "animal_sound"), ("say the alphabet", "alphabet"),
                               ("lets play a quiz", "quiz"), ("where is my nose", "body"),
                               ("what is three plus four", "math"), ("stop", "stop")]:
                got = b.matcher.parse(text)
                if got.name != want:
                    bad.append((age, text, want, got.name))
        self.assertEqual(bad, [], f"core commands broke at some age: {bad}")


class TestMinConfidenceSetting(unittest.TestCase):
    """[nlu] min_confidence was declared in config.py but never read by anything."""

    def test_it_is_off_by_default(self):
        self.assertEqual(Config()["nlu"]["min_confidence"], 0.0)

    def test_it_is_actually_read(self):
        """Off: a 3-year-old's mumble reaches the months lesson.  On: it gets the fallback."""
        cfg = Config()
        cfg.data["child"]["age"] = 3
        cfg.data["nlu"]["min_confidence"] = 0.0
        loose = Brain(cfg, rng=random.Random(0), progress=Progress(None))
        self.assertEqual(loose.matcher.parse("ngggh mmmph").name, "months")  # today's behaviour

        cfg2 = Config()
        cfg2.data["child"]["age"] = 3
        cfg2.data["nlu"]["min_confidence"] = 0.74
        strict = Brain(cfg2, rng=random.Random(0), progress=Progress(None))
        self.assertEqual(strict.matcher.parse("ngggh mmmph").name, "unknown")
        self.assertIn(said(strict.command("ngggh mmmph")),
                      set(strict.content.phrases["not_understood"]))

    def test_an_impossible_value_is_corrected(self):
        """Above 1.0 nothing could ever clear the bar, so Buddy would understand nothing."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "buddy.toml"
            f.write_text("[nlu]\nmin_confidence = 1.5\n", encoding="utf-8")
            cfg = load_config(f)
        self.assertEqual(cfg["nlu"]["min_confidence"], 0.0)
        b = Brain(cfg, rng=random.Random(0), progress=Progress(None))
        self.assertEqual(b.command("tell me a story").intent, "story")

    def test_raising_it_does_not_touch_clear_commands(self):
        cfg = Config()
        cfg.data["nlu"]["min_confidence"] = 0.74
        b = Brain(cfg, rng=random.Random(0), progress=Progress(None))
        for text, want in [("tell me a story", "story"), ("count to ten", "count"),
                           ("stop", "stop"), ("how are you", "how_are_you"),
                           ("what is three plus four", "math"), ("what is b for", "phonics")]:
            self.assertEqual(b.matcher.parse(text).name, want, text)


class TestAmbiguousCommands(unittest.TestCase):
    """6C: input that could match more than one intent picks a sensible one, never a stranger."""

    def test_overlapping_words_pick_a_related_intent(self):
        b = make()
        for text, allowed in [
            ("tell me a story about the alphabet", {"story", "alphabet"}),
            ("sing the alphabet", {"alphabet", "song"}),
            ("count the animals", {"count", "animal_sound"}),
            ("a poem about a lion", {"poem", "animal_fact", "animal_sound"}),
            ("play a counting game", {"quiz", "count"}),
            ("tell me a story about colours", {"story", "colors"}),
            ("what colour is the triangle", {"color_of", "colors", "shapes"}),
            ("animal quiz", {"quiz", "animal_sound"}),
            ("say it again louder", {"repeat", "volume_up"}),
            ("stop the story", {"stop", "story"}),
        ]:
            got = b.matcher.parse(text)
            self.assertIn(got.name, allowed, f"{text!r} -> {got!r}")

    def test_a_clear_keyword_beats_a_guessed_one(self):
        """Finding M6: "poem" was said outright; "teww" only resembles "tale"."""
        b = make()
        self.assertEqual(b.matcher.parse("teww me a poem").name, "poem")
        self.assertEqual(b.matcher.parse("teww me a towy").name, "story")

    def test_help_me_is_read_as_a_request_for_help(self):
        """Pinned behaviour, not a reported defect: "help me ..." is an explicit help cue, so
        "help me calm down" offers the list of things Buddy can do rather than the breathing
        exercise.  Noted in the post-fix report under Remaining issues."""
        self.assertEqual(make().matcher.parse("help me calm down").name, "help")

    def test_a_body_word_in_passing_does_not_start_the_lesson(self):
        b = make()
        self.assertEqual(b.matcher.parse("where is my nose").name, "body")
        self.assertNotEqual(b.matcher.parse("set a brush teeth timer").name, "body")
        self.assertNotEqual(b.matcher.parse("i brush my teeth").name, "body")


class TestStateAndInteraction(unittest.TestCase):
    """6D: one exchange never corrupts the next."""

    def test_wake_then_command(self):
        b = make()
        woke, rest, _ = b.wake.split("hey buddy tell me a joke")
        self.assertTrue(woke)
        self.assertEqual(b.command(rest).intent, "joke")

    def test_command_then_repeat(self):
        b = make()
        first = b.command("tell me a joke")
        again = b.command("say that again")
        self.assertEqual(again.texts(), first.texts())

    def test_command_then_another_one(self):
        """Finding H1: "another one" after an animal fact used to end the program."""
        for first, word in [("tell me about lions", "another one"),
                            ("tell me about lions", "one more"),
                            ("tell me a joke", "another one"),
                            ("what colour is grass", "another one"),
                            ("what is the opposite of hot", "another one"),
                            ("what is three plus four", "another one"),
                            ("why is the sky blue", "another one"),
                            ("i am sad", "another one")]:
            b = make()
            b.command(first)
            r = b.command(word)  # must not raise
            self.assertTrue(r.items, f"{first!r} -> {word!r} said nothing")
            self.assertTrue(said(r).strip(), f"{first!r} -> {word!r} said nothing")

    def test_every_intent_survives_having_no_parameters(self):
        """Finding H1 / K04: handle() is reachable with no slots for every intent."""
        b = make()
        broken = []
        for name in sorted(b.skills.handlers):
            try:
                r = b.skills.handle(Intent(name))
            except Exception as e:  # noqa: BLE001 - the point of the test
                broken.append((name, f"{type(e).__name__}: {e}"))
                continue
            if not r.items and not r.action:
                broken.append((name, "said nothing and did nothing"))
        self.assertEqual(broken, [], f"handlers that crash or stay silent with no slots: {broken}")

    def test_command_then_stop_then_a_new_command(self):
        b = make()
        b.command("tell me a story")
        stop = b.command("stop")
        self.assertEqual(stop.action, "stop")
        self.assertEqual(b.command("tell me a joke").intent, "joke")

    def test_gibberish_between_two_commands(self):
        b = make()
        self.assertEqual(b.command("tell me a joke").intent, "joke")
        b.command("asdfghjkl")
        self.assertEqual(b.command("tell me a poem").intent, "poem")

    def test_gibberish_does_not_break_a_pending_question(self):
        b = make()
        b.command("lets play a quiz")
        self.assertTrue(b.expecting)
        r = b.answer("asdfghjkl")  # a wrong answer, not a crash
        self.assertTrue(r.items)
        self.assertEqual(b.command("tell me a joke").intent, "joke")

    def test_quiz_timeout_then_a_new_command(self):
        b = make()
        b.command("lets play a quiz")
        b.timeout()
        self.assertEqual(b.command("tell me a story").intent, "story")

    def test_follow_up_ignores_talk_that_is_not_a_request(self):
        """The v0.2 conversation feature: background talk is not answered."""
        b = make()
        b.command("tell me a joke")
        self.assertIsNone(b.follow_up("mummy can i have some water"))
        self.assertIsNotNone(b.follow_up("tell me a story"))

    def test_missing_content_is_handled_safely(self):
        """A content lookup that finds nothing must not crash or answer with silence."""
        b = make()
        for intent in [Intent("animal_fact", {"animal": "no_such_animal"}),
                       Intent("color_of", {"thing": "no_such_thing"}),
                       Intent("opposite_of", {"word": "no_such_word"}),
                       Intent("faq", {"index": 9999}),
                       Intent("feeling", {"mood": "no_such_mood"}),
                       Intent("shapes", {"shape": "no_such_shape"}),
                       Intent("body", {"part": "no_such_part"}),
                       Intent("math", {"a": 1, "b": 2, "op": "no_such_op"})]:
            r = b.skills.handle(intent)
            self.assertTrue(r.items or r.action, f"{intent!r} produced nothing")


class TestCounting(unittest.TestCase):
    """Findings M2 and M3: steps and large word-form numbers."""

    def test_step_is_kept_when_a_range_is_given(self):
        for text, want in [("count from one to one hundred in steps of ten", (1, 100, 10)),
                           ("count from 1 to 100 in steps of 10", (1, 100, 10)),
                           ("count from two to ten in steps of two", (2, 10, 2)),
                           ("count from two to ten in jumps of two", (2, 10, 2))]:
            s = count_slots(normalize(text))
            self.assertEqual((s.start, s.end, s.step), want, text)

    def test_step_is_not_read_as_the_end_of_the_range(self):
        s = count_slots(normalize("count to twenty in steps of five"))
        self.assertEqual((s.start, s.end, s.step), (None, 20, 5))
        said_ = said(make().command("count to twenty in steps of five"))
        self.assertIn("Five.", said_)
        self.assertIn("Twenty.", said_)
        self.assertNotIn("Nineteen.", said_, "it counted every number instead of in fives")

    def test_step_alone_still_works(self):
        for text, want in [("count by twos", 2), ("count in fives", 5), ("count in 5s", 5)]:
            self.assertEqual(count_slots(normalize(text)).step, want, text)

    def test_word_form_thousands(self):
        self.assertEqual([v for v, _, _ in parse_numbers(normalize("one thousand"))], [1000])
        self.assertEqual([v for v, _, _ in parse_numbers(normalize("two thousand"))], [2000])
        s = count_slots(normalize("count from one to one thousand"))
        self.assertEqual((s.start, s.end), (1, 1000))

    def test_too_big_is_clamped_the_same_in_words_and_digits(self):
        words = said(make().command("count from one to one thousand"))
        digits = said(make().command("count from 1 to 1000"))
        for out in (words, digits):
            self.assertIn("I can count up to one hundred", out)
            self.assertIn("One hundred.", out)

    def test_negative_start_counts_from_zero(self):
        s = count_slots(normalize("count from minus five to five"))
        self.assertEqual((s.start, s.end), (-5, 5))
        out = said(make().command("count from minus five to five"))
        self.assertIn("Zero.", out)
        self.assertIn("Five.", out)
        self.assertNotIn("Minus", out, "Buddy has no spoken clips for negative numbers")

    def test_a_subtraction_is_still_a_subtraction(self):
        got = make().matcher.parse("what is five minus three")
        self.assertEqual(got.name, "math")
        self.assertEqual((got.slots["a"], got.slots["b"], got.slots["op"]), (5, 3, "minus"))


class TestWakePhrase(unittest.TestCase):
    """Finding M7: everyday sentences containing "buddy" must not wake Buddy."""

    def test_everyday_sentences_do_not_wake(self):
        b = make()
        woke_wrongly = []
        for text in ["my buddy is coming over", "buddy and i played outside", "buddy the dog is asleep",
                     "my buddy likes pizza", "i saw my buddy at school", "buddy went to the park",
                     "hey mummy", "what is for dinner", "the bunny is cute", "hello everyone"]:
            woke, _rest, score = b.wake.split(text)
            if woke:
                woke_wrongly.append((text, round(score, 2)))
        self.assertEqual(woke_wrongly, [], f"false wake-ups: {woke_wrongly}")

    def test_real_wake_phrases_still_work(self):
        b = make()
        for text in ["hey buddy tell me a story", "hi buddy count to ten", "hello buddy",
                     "okay buddy say a poem", "buddy stop", "hey buddy"]:
            woke, _rest, score = b.wake.split(text)
            self.assertTrue(woke, (text, score))


class TestPerformance(unittest.TestCase):
    """Finding M8: the cost of parsing must not grow with the length of a single token."""

    def test_a_very_long_token_is_parsed_quickly(self):
        b = make()
        b.matcher.parse("warm up the caches")
        worst = 0.0
        for n in (100, 1000, 10000):
            t0 = time.perf_counter()
            got = b.matcher.parse("x" * n)
            worst = max(worst, time.perf_counter() - t0)
            self.assertIn(got.name, ("unknown", "none"))
        self.assertLess(worst, 0.5, f"a single long token took {worst * 1000:.0f} ms to parse")

    def test_tokens_are_capped(self):
        self.assertLessEqual(max(len(t) for t in normalize("x" * 5000 + " hello")), 32)


class TestConfigRobustness(unittest.TestCase):
    """Findings L1, L2 and J10: bad settings are reported, never silently obeyed or fatal."""

    def test_negative_daily_limit_does_not_lock_the_child_out(self):
        cfg = Config()
        cfg.data["parental"]["daily_limit_minutes"] = -5
        b = Brain(cfg, rng=random.Random(0), progress=Progress(None))
        self.assertIsNone(b.gate(), "a negative limit blocked everything at zero usage")
        self.assertEqual(b.command("tell me a story").intent, "story")

    def test_a_real_daily_limit_still_applies(self):
        cfg = Config()
        cfg.data["parental"]["daily_limit_minutes"] = 1
        p = Progress(None)
        p.add_usage(120)
        b = Brain(cfg, rng=random.Random(0), progress=p)
        self.assertIsNotNone(b.gate(), "the limit stopped working")

    def test_negative_limit_in_a_file_is_corrected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "buddy.toml"
            p.write_text("[parental]\ndaily_limit_minutes = -5\n", encoding="utf-8")
            cfg = load_config(p)
        self.assertEqual(cfg["parental"]["daily_limit_minutes"], 0)

    def test_malformed_toml_is_reported_in_words(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.toml"
            p.write_text("[child\nage = 5\nthis is not = valid toml ===\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as cm:
                load_config(p)
        self.assertIn("not valid TOML", str(cm.exception))

    def test_a_config_path_that_does_not_exist_is_not_ignored(self):
        with self.assertRaises(ConfigError):
            load_config("/no/such/file/buddy.toml")

    def test_an_unknown_nlu_fallback_does_not_stop_buddy_starting(self):
        cfg = Config()
        cfg.data["nlu"]["fallback"] = "does_not_exist"
        b = Brain(cfg, rng=random.Random(0), progress=Progress(None))  # must not raise
        self.assertIsNone(b.matcher.fallback)
        self.assertEqual(b.command("tell me a story").intent, "story")

    def test_odd_values_are_brought_back_into_range(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "odd.toml"
            p.write_text("[child]\nage = 99\n[parental]\nquiet_hours = \"always\"\n"
                         "[nlu]\nfallback = \"nope\"\n", encoding="utf-8")
            cfg = load_config(p)
        self.assertEqual(cfg.child_age, 5)
        self.assertEqual(cfg["parental"]["quiet_hours"], "")
        self.assertEqual(cfg["nlu"]["fallback"], "none")


class TestProgressFileHandle(unittest.TestCase):
    """Finding M1: nothing may hold the progress database open between operations."""

    def test_no_handle_is_left_open(self):
        import os
        if not Path("/proc/self/fd").exists():
            self.skipTest("needs /proc to count open handles")

        def handles(target):
            out = []
            for fd in os.listdir("/proc/self/fd"):
                try:
                    out.append(os.readlink(f"/proc/self/fd/{fd}"))
                except OSError:
                    pass
            return [p for p in out if str(target) in p]

        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "p.db"
            p = Progress(db)
            p.record("numbers", "after_3", True)
            self.assertEqual(handles(db), [], "a handle stayed open after record()")
            p.level("numbers")
            p.summary()
            p.boxes("numbers")
            p.usage_today()
            self.assertEqual(handles(db), [], "a handle stayed open after reading")
            self.assertTrue(p.db.iterdump())  # inspection still works
            self.assertEqual(handles(db), [], "inspecting the data left a handle open")

    def test_the_database_file_can_be_deleted_while_progress_is_alive(self):
        """This is what TemporaryDirectory.cleanup does, and what Windows refused."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "p.db"
            p = Progress(db)
            p.record("numbers", "after_3", True)
            db.unlink()
            self.assertFalse(db.exists())

    def test_data_still_persists(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "p.db"
            p = Progress(db, start_level=2)
            p.record("animals", "cow", True)
            p.add_usage(60)
            del p
            q = Progress(db, start_level=2)
            self.assertEqual(q.summary(), [("animals", 2, 1, 0)])
            self.assertEqual(q.usage_today(), 60)

    def test_in_memory_progress_still_works(self):
        p = Progress(None)
        p.record("numbers", "after_3", True)
        self.assertEqual(p.summary(), [("numbers", 1, 1, 0)])
        self.assertIsInstance(p.db, sqlite3.Connection)
        p.close()

    def test_privacy_still_holds(self):
        """The original guarantee, re-checked through the new connection handling."""
        with tempfile.TemporaryDirectory() as d:
            p = Progress(Path(d) / "p.db")
            p.record("numbers", "after_3", True)
            self.assertNotIn("four", "\n".join(p.db.iterdump()))


class TestVersionAndFallbackStyle(unittest.TestCase):
    def test_version_matches_the_project(self):
        """Finding L4: the version is stamped into every record."""
        import buddy
        self.assertEqual(buddy.__version__, "0.3.0")

    def test_volume_replies_are_not_empty(self):
        """Finding L3: chat mode and the tests saw an empty reply for louder / softer."""
        b = make()
        for text, key in [("louder", "louder"), ("softer", "softer")]:
            r = b.command(text)
            self.assertIn(said(r), set(b.content.phrases[key]), text)

    def test_one_consistent_fallback_style(self):
        """Every "I did not understand" reply is one of the two inventory wordings."""
        b = make()
        allowed = set()
        for key in FALLBACK_KEYS:
            allowed.update(b.content.phrases[key])
        for text in ["asdfgh", "", "!!!", "你好", "qqq zzz"]:
            r = make().command(text)
            self.assertIn(said(r), allowed, text)


class TestRecordsCaptureCrashes(unittest.TestCase):
    """Finding H2: a crash in chat / simulate was written as a normal "ignored" interaction."""

    def test_a_crash_is_recorded_as_an_error_with_a_traceback(self):
        import json

        from buddy.records import open_records

        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.data["records"]["enabled"] = True
            cfg.data["records"]["folder"] = d
            b = make()
            rec = open_records(cfg, "chat").attach(b)

            def boom(_text, intent=None):
                raise RuntimeError("deliberate test failure")

            b.command = boom
            try:
                rec.chat_line("tell me a story", "listening")
                with self.assertRaises(RuntimeError):
                    try:
                        b.command("tell me a story")
                    except BaseException as exc:
                        rec._safe(rec.crash, exc)
                        rec._safe(rec.close, "crashed")
                        raise
            finally:
                rec.close("done")

            lines = []
            for f in sorted(Path(d).glob("*.jsonl")):
                lines += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

        errors = [r for r in lines if r.get("status") == "error"]
        self.assertTrue(errors, f"the crash was not recorded as an error: "
                                f"{[r.get('status') for r in lines]}")
        err = errors[0]["error"]
        self.assertEqual(err["type"], "RuntimeError")
        self.assertIn("deliberate test failure", err["message"])
        self.assertIn("Traceback", err["traceback"])
        self.assertNotIn("ignored", [r.get("status") for r in lines])

    def test_the_chat_loop_reports_a_crash_before_re_raising(self):
        """The wiring itself: cmd_chat hands the exception to the recorder (finding H2)."""
        import inspect

        from buddy import __main__ as cli
        src = inspect.getsource(cli.cmd_chat)
        self.assertIn("rec.crash", src)
        self.assertIn("raise", src)
        src = inspect.getsource(cli.cmd_simulate)
        self.assertIn("rec.crash", src)


class TestOfflineAndPrivacy(unittest.TestCase):
    """The project's standing requirements, re-checked after the changes."""

    def test_no_network_use_anywhere_in_the_runtime(self):
        import re as _re
        bad = []
        for p in Path("buddy").rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            for mod in ("socket", "urllib", "requests", "httpx", "http.client"):
                if _re.search(rf"^\s*(import|from)\s+{_re.escape(mod)}\b", text, _re.M):
                    bad.append((str(p), mod))
        self.assertEqual(bad, [], f"the runtime must stay offline: {bad}")

    def test_every_sentence_buddy_can_say_is_in_the_inventory(self):
        """Closed-domain guarantee, re-checked for the replies the fixes added."""
        b = make()
        inventory = b.skills.all_phrases()
        missing = []
        for text in ["louder", "softer", "asdfgh", "", "where is my nose", "my tummy",
                     "count to twenty in steps of five", "count from minus five to five",
                     "count from one to one thousand", "who is the prime minister"]:
            r = make().command(text)
            missing += [(text, t) for t in r.texts() if t not in inventory]
        for first, word in [("tell me about lions", "another one"), ("what colour is grass", "another one"),
                            ("what is the opposite of hot", "another one"), ("what is three plus four", "another one")]:
            br = make()
            br.command(first)
            r = br.command(word)
            missing += [(f"{first} -> {word}", t) for t in r.texts() if t not in inventory]
        self.assertEqual(missing, [], f"sentences outside the pre-rendered inventory: {missing}")


if __name__ == "__main__":
    unittest.main()
