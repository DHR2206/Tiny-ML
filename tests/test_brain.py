import random
import tempfile
import unittest
from pathlib import Path

from buddy.brain import Brain, _in_window
from buddy.config import Config
from buddy.core import Say
from buddy.learning.progress import Progress
from buddy.speech.wake import WakeMatcher


def make(age=5, seed=0):
    cfg = Config()
    cfg.data["child"]["age"] = age
    return Brain(cfg, rng=random.Random(seed), progress=Progress(None))


class TestWake(unittest.TestCase):
    def setUp(self):
        self.b = make()

    def test_wake_variants(self):
        for t in ["Hey Buddy, play a story.", "hey buddy", "Hi buddy count to ten", "Hey, Buddy!",
                  "hey body tell me a joke", "a buddy play a poem", "Buddy, stop."]:
            woke, rest, s = self.b.wake.split(t)
            self.assertTrue(woke, (t, s))
        woke, rest, _ = self.b.wake.split("Hey Buddy, tell me a story")
        self.assertEqual(rest, "tell me a story")

    def test_non_wake(self):
        for t in ["what's for dinner", "hey mummy", "the bunny is cute", "play a story", "hello everyone",
                  "they said hello", "the body is here"]:
            woke, _, s = self.b.wake.split(t)
            self.assertFalse(woke, (t, s))


class TestBrain(unittest.TestCase):
    def test_every_spoken_sentence_is_prerenderable(self):
        """Closed-domain guarantee: every Say produced at runtime is in the phrase inventory."""
        b = make(seed=3)
        inventory = b.skills.all_phrases()
        cmds = ["tell me a story", "play a poem", "sing a song", "alphabet", "phonics", "what is z for",
                "count to 100", "count backwards from 20", "count by fives", "count from 5 to 15", "animal sounds",
                "what does the owl say", "tell me about camels", "colours", "what color is grass", "shapes",
                "body parts", "days of the week", "months", "opposites", "table of 7", "what is 4 plus 5",
                "what is 9 minus 3", "3 times 8", "joke", "riddle", "breathe", "magic words", "i am sad",
                "i am scared", "hello", "good morning", "good night", "bye", "thank you", "i love you",
                "what is your name", "how old are you", "why is the sky blue", "why do fish swim", "blorp",
                "flibber", "what can you do", "say it again", "stop", "count to 500", "what is 50 plus 60"]
        for _ in range(3):
            for c in cmds:
                r = b.command(c)
                for t in r.texts():
                    self.assertIn(t, inventory, (c, t))
        # quiz dialogues in every domain, with right, wrong and silent answers
        for dom in ["animal quiz", "number quiz", "letter quiz", "colour quiz", "shape quiz", "opposite quiz"]:
            for replies in (["zzz", "zzz"] * 6, ["i don't know"] * 6, [None] * 2):
                r = b.command(dom)
                for rep in replies:
                    for t in r.texts():
                        self.assertIn(t, inventory, (dom, t))
                    if not b.expecting:
                        break
                    r = b.timeout() if rep is None else b.answer(rep)
                for t in r.texts():
                    self.assertIn(t, inventory, (dom, t))
        print(f"phrase inventory: {len(inventory)} sentences")

    def test_quiz_flow_and_adaptation(self):
        b = make(age=4)
        r = b.command("number quiz")
        self.assertTrue(b.expecting)
        correct = 0
        for _ in range(5):
            q = next(qq for qq in b.skills.bank["numbers"] if qq.text in r.texts())
            r = b.answer(str(q.answers[0]))
            correct += 1
        self.assertFalse(b.expecting)
        self.assertIn("five", " ".join(r.texts()))
        self.assertGreaterEqual(b.progress.level("numbers"), 2)  # levelled up after a perfect run

    def test_quiz_hint_then_reveal(self):
        b = make()
        b.command("animal quiz")
        r = b.answer("banana")  # wrong -> hint and same question
        self.assertTrue(b.expecting)
        self.assertTrue(any("Try" in t or "try" in t or "Almost" in t for t in r.texts()))
        r = b.answer("banana")  # wrong again -> reveal + next question
        self.assertTrue(any("answer is" in t for t in r.texts()))

    def test_stop_during_quiz(self):
        b = make()
        b.command("quiz")
        r = b.answer("stop")
        self.assertEqual(r.action, "stop")
        self.assertFalse(b.expecting)

    def test_repeat_and_more(self):
        b = make()
        r1 = b.command("tell me a joke")
        r2 = b.command("say it again")
        self.assertEqual(r1.texts(), r2.texts())
        r3 = b.command("another one")
        self.assertEqual(r3.intent, "joke")

    def test_unknown_then_help(self):
        b = make()
        self.assertEqual(b.command("blorp").intent, "unknown")
        r = b.command("zzzz qqq")
        self.assertTrue(any("You can say" in t for t in r.texts()))

    def test_curiosity_is_encouraging(self):
        b = make()
        t = " ".join(b.command("why do dinosaurs have long tails").texts())
        self.assertTrue("question" in t.lower())
        self.assertTrue("grown-up" in t or "teacher" in t)

    def test_parental_quiet_hours(self):
        self.assertTrue(_in_window("21:00-06:30", _t(22, 0)))
        self.assertTrue(_in_window("21:00-06:30", _t(5, 0)))
        self.assertFalse(_in_window("21:00-06:30", _t(12, 0)))
        b = make()
        b.cfg.data["parental"]["quiet_hours"] = "00:00-23:59"
        self.assertEqual(b.command("tell me a story").intent, "quiet_hours")
        self.assertEqual(b.command("stop").intent, "stop")  # controls still work

    def test_privacy_store_has_no_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(Path(d) / "p.db")
            p.record("numbers", "after_3", True)
            dump = "\n".join(p.db.iterdump())
            self.assertNotIn("four", dump)  # answers / transcripts are never stored


def _t(h, m):
    import time
    return time.struct_time((2026, 1, 1, h, m, 0, 0, 1, -1))


if __name__ == "__main__":
    unittest.main()
