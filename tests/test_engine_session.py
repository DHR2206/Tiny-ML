"""Conversation flow of the real-time engine, with fake audio devices and a fake clock.

Covers: follow-up listening without "Hey Buddy", ignoring background talk, going back to
sleep, pause / continue (from the interrupted sentence), repeat, quit, barge-in while
Buddy talks, keyboard keys, mute, and the old wake-phrase-every-time mode.
No models or sound card needed.
"""
import random
import unittest

import numpy as np

from buddy.brain import Brain
from buddy.config import Config
from buddy.core import Earcon, Pause, Response, Say
from buddy.engine import Engine
from buddy.learning.progress import Progress

RATE = 16000
CHAR_SECONDS = 0.05  # fake speech: 20 characters per second


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeSpeaker:
    """Plays in fake time: audio queued with play() is 'heard' as the clock advances."""

    def __init__(self, clock):
        self.clock, self.rate, self.volume, self.duck = clock, RATE, 0.7, 1.0
        self.end = 0.0

    def play(self, x):
        self.end = max(self.end, self.clock()) + len(x) / RATE

    def stop(self):
        self.end = self.clock()

    def remaining(self):
        return max(0.0, self.end - self.clock())

    @property
    def busy(self):
        return self.clock() < self.end

    def start(self):
        pass

    def close(self):
        pass


class FakeRenderer:
    """Records what Buddy says and when (instead of synthesising audio)."""

    def __init__(self, clock, speaker):
        self.clock, self.speaker, self.log = clock, speaker, []

    def render_item(self, it):
        if isinstance(it, Say):
            n = int(len(it.text) * CHAR_SECONDS * RATE)
            self.log.append((max(self.clock(), self.speaker.end), it.text))
            return np.zeros(n + int(it.pause_after * RATE), dtype=np.float32), n
        if isinstance(it, Earcon):
            self.log.append((max(self.clock(), self.speaker.end), f"[{it.kind}]"))
            return np.zeros(int(0.3 * RATE), dtype=np.float32), int(0.2 * RATE)
        if isinstance(it, Pause):
            return np.zeros(int(it.seconds * RATE), dtype=np.float32), 0
        return np.zeros(0, dtype=np.float32), 0


class Dummy:
    def __getattr__(self, name):
        return lambda *a, **k: None


def make(follow_up=15.0, seed=1):
    cfg = Config()
    cfg.data["session"]["follow_up_seconds"] = follow_up
    brain = Brain(cfg, rng=random.Random(seed), progress=Progress(None))
    clock = Clock()
    spk = FakeSpeaker(clock)
    eng = Engine(cfg, brain, Dummy(), spk, Dummy(), Dummy(), Dummy(), clock=clock)
    eng.player.renderer = FakeRenderer(clock, spk)
    return eng, clock


def run(eng, clock, seconds, step=0.05):
    for _ in range(int(seconds / step)):
        clock.t += step
        eng._timers()


def said(eng, since=0):
    return [t for _, t in eng.player.renderer.log[since:]]


def run_until_quiet(eng, clock, limit=600):
    for _ in range(int(limit / 0.05)):
        clock.t += 0.05
        eng._timers()
        if not eng.player.active:
            return
    raise AssertionError("Buddy never stopped talking")


class TestConversation(unittest.TestCase):
    def test_no_wake_phrase_needed_after_a_task(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a joke")
        run_until_quiet(eng, clock)  # joke, then the follow-up prompt and chime
        self.assertIn("[ready]", said(eng))
        n = len(said(eng))
        eng.on_text("tell me a story")  # no "Hey Buddy"
        run(eng, clock, 1)
        self.assertTrue(any("story" in t for t in said(eng, n)), said(eng, n))

    def test_background_talk_is_ignored_and_buddy_sleeps(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a joke")
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text("mummy where are my shoes")
        run(eng, clock, 1)
        self.assertEqual(said(eng, n), [])  # nothing said
        run(eng, clock, 22)  # silence: goodbye line + end chime, then asleep
        self.assertIn("[end]", said(eng, n))
        self.assertEqual(eng.awake_until, 0.0)
        m = len(said(eng))
        eng.on_text("tell me a story")  # asleep again: the wake phrase is needed
        run(eng, clock, 1)
        self.assertEqual(said(eng, m), [])

    def test_prompt_is_varied_and_not_after_every_task(self):
        eng, clock = make()
        ph = eng.brain.content.phrases
        prompt_lines = set(ph["follow_up"] + ph["follow_up_more"])
        eng.on_text("hey buddy tell me a joke")
        prompts = []
        for _ in range(6):
            n = len(said(eng))
            run_until_quiet(eng, clock)
            prompts.append([t for t in said(eng, n) if t in prompt_lines])
            self.assertEqual(said(eng)[-1], "[ready]")  # a soft chime after every task
            eng.on_text("what does the cow say")
        spoken = [p[0] for p in prompts if p]
        self.assertEqual(len(spoken), 3, prompts)  # after every other task
        self.assertTrue(all(a != b for a, b in zip(spoken, spoken[1:])), spoken)  # never the same twice in a row

    def test_answer_yes_to_want_another_one(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a joke")
        run_until_quiet(eng, clock)
        self.assertEqual(eng.brain.offer, "more")  # "Want another one?" was asked after the first task
        n = len(said(eng))
        eng.on_text("yes")
        run(eng, clock, 1)
        self.assertTrue(said(eng, n) and said(eng, n)[0].startswith("Here's a joke"), said(eng, n))

    def test_greeting_only_at_start_up(self):
        eng, clock = make()
        eng._wake(clock())
        eng._say(Response([Earcon("unmute"), Say(eng.brain.skills.p("hello")), Earcon("ready")],
                          repeatable=False, intent="boot", after="silent"))
        run_until_quiet(eng, clock)
        self.assertGreater(eng.awake_until, clock())  # "What shall we do today?" - listening
        eng.on_text("tell me a joke")
        run_until_quiet(eng, clock)
        eng.on_text("tell me another joke")
        run_until_quiet(eng, clock)
        greetings = set(eng.brain.content.phrases["hello"])
        self.assertEqual(sum(t in greetings for t in said(eng)), 1)  # no greeting before each task

    def test_wait_during_a_quiz_gives_more_time(self):
        eng, clock = make()
        eng.on_text("hey buddy animal quiz")
        run_until_quiet(eng, clock)
        run(eng, clock, 7)
        n = len(said(eng))
        eng.on_text("wait")
        run_until_quiet(eng, clock)
        self.assertIn(said(eng, n)[0], eng.brain.content.phrases["wait_ack"])
        run(eng, clock, 7)  # more than the original 9 s in total, still waiting
        self.assertTrue(eng.brain.expecting)
        self.assertFalse(any(t.startswith("The answer is") for t in said(eng, n)))

    def test_wake_phrase_every_time_when_follow_up_is_off(self):
        eng, clock = make(follow_up=0)
        eng.on_text("hey buddy tell me a joke")
        run_until_quiet(eng, clock)
        self.assertNotIn("[ready]", said(eng))
        n = len(said(eng))
        eng.on_text("tell me a story")
        run(eng, clock, 1)
        self.assertEqual(said(eng, n), [])


class TestPauseContinue(unittest.TestCase):
    def start_story(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me the thirsty crow story")
        run(eng, clock, 12)  # a few sentences in
        return eng, clock

    def test_pause_then_continue_resumes_the_interrupted_sentence(self):
        eng, clock = self.start_story()
        story = eng.player.response
        hearing = eng.player.item_at(1.0 + 0.55 + 0.3)  # what was playing when the child started talking
        eng.on_text("hey buddy pause", heard_seconds=1.0)
        self.assertIsNotNone(eng.paused)
        self.assertEqual(eng.paused.index, hearing)
        run_until_quiet(eng, clock)  # "Okay, I'll wait..."
        n = len(said(eng))
        eng.on_text("continue")  # no wake phrase needed, Buddy is listening
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], story.items[hearing].text)

    def test_continue_after_buddy_went_to_sleep(self):
        eng, clock = self.start_story()
        eng.on_text("hey buddy pause")
        resume_at = eng.paused.response.items[eng.paused.index].text
        run(eng, clock, 40)  # window closes: "I'll keep our place..."
        self.assertTrue(any("keep our place" in t for t in said(eng)))
        self.assertEqual(eng.awake_until, 0.0)
        n = len(said(eng))
        eng.on_text("hey buddy continue")
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], resume_at)
        self.assertIsNone(eng.paused)

    def test_pause_is_forgotten_after_three_minutes(self):
        eng, clock = self.start_story()
        eng.on_text("hey buddy pause")
        run(eng, clock, 181)
        self.assertIsNone(eng.paused)
        n = len(said(eng))
        eng.on_text("hey buddy continue")
        run(eng, clock, 0.2)
        self.assertIn("nothing to continue", said(eng, n)[0])

    def test_short_task_keeps_pause_long_task_replaces_it(self):
        eng, clock = self.start_story()
        eng.on_text("hey buddy pause")
        run_until_quiet(eng, clock)
        eng.on_text("tell me a joke")  # short: the story stays paused
        run_until_quiet(eng, clock)
        self.assertIsNotNone(eng.paused)
        eng.on_text("tell me a poem")  # a new long activity replaces it
        run(eng, clock, 0.5)
        self.assertIsNone(eng.paused)

    def test_nothing_to_pause(self):
        eng, clock = make()
        eng.on_text("hey buddy pause")
        run(eng, clock, 0.2)
        self.assertIn("nothing playing", said(eng)[0])

    def test_plain_pause_while_talking_is_ignored_but_barge_in_window_works(self):
        eng, clock = self.start_story()
        eng.on_text("pause")  # no wake phrase while Buddy talks: could be Buddy's own voice
        self.assertIsNone(eng.paused)
        eng.on_text("hey buddy")  # the child pauses after the wake phrase
        self.assertLess(eng.speaker.duck, 1.0)  # Buddy talks softer and listens
        eng.on_text("pause")
        self.assertIsNotNone(eng.paused)
        self.assertEqual(eng.speaker.duck, 1.0)

    def test_new_request_while_talking_interrupts_and_story_can_continue(self):
        eng, clock = self.start_story()
        eng.on_text("hey buddy tell me a joke", heard_seconds=1.5)
        self.assertEqual(eng.player.response.intent, "joke")
        self.assertIsNotNone(eng.paused)  # the story is kept
        resume_at = eng.paused.response.items[eng.paused.index].text
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text("continue")
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], resume_at)

    def test_buddys_own_words_do_not_start_a_new_request(self):
        eng, clock = self.start_story()
        story = eng.player.response
        eng.on_text("hey buddy")  # the child starts, then hesitates
        eng.on_text("once upon a time a crow was flying he was so thirsty")  # only Buddy's echo is heard
        self.assertIs(eng.player.response, story)

    def test_keyboard_p_pauses_and_continues(self):
        eng, clock = self.start_story()
        eng._key("p")
        self.assertIsNotNone(eng.paused)
        run_until_quiet(eng, clock)
        eng._key("p")
        run(eng, clock, 0.2)
        self.assertIsNone(eng.paused)
        self.assertTrue(eng.player.active)

    def test_mute_pauses_the_story(self):
        eng, clock = self.start_story()
        eng._do_mute()
        self.assertIsNotNone(eng.paused)
        run_until_quiet(eng, clock)
        eng._do_mute()
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text("continue")
        run(eng, clock, 0.2)
        self.assertTrue(eng.player.active)
        self.assertNotIn("nothing to continue", " ".join(said(eng, n)))


class TestRepeatQuitStop(unittest.TestCase):
    def test_repeat_plays_the_same_story_again(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a story")
        first = [i.text for i in eng.player.response.items if isinstance(i, Say)]
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text("repeat")
        run_until_quiet(eng, clock)
        again = said(eng, n)
        self.assertEqual(again[:len(first)], first)

    def test_repeat_while_talking_restarts_the_reply(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a story")
        first = eng.player.response.items[0].text
        run(eng, clock, 10)
        n = len(said(eng))
        eng.on_text("hey buddy repeat")
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], first)

    def test_repeat_in_a_quiz_repeats_the_question(self):
        eng, clock = make()
        eng.on_text("hey buddy animal quiz")
        question = eng.brain.expect.prompt[0].text
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text("repeat")
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], question)
        self.assertTrue(eng.brain.expecting)  # still waiting for the answer

    def test_quit_ends_the_conversation_but_not_the_program(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a story")
        run(eng, clock, 5)
        eng.on_text("hey buddy pause")
        run_until_quiet(eng, clock)
        eng.on_text("quit")
        run_until_quiet(eng, clock)
        self.assertIsNone(eng.paused)
        self.assertEqual(eng.awake_until, 0.0)
        self.assertTrue(eng.running)
        n = len(said(eng))
        eng.on_text("tell me a joke")  # asleep now
        run(eng, clock, 1)
        self.assertEqual(said(eng, n), [])

    def test_stop_while_talking_keeps_listening(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a story")
        run(eng, clock, 5)
        eng.on_text("hey buddy stop")
        run_until_quiet(eng, clock)
        self.assertGreater(eng.awake_until, clock())
        n = len(said(eng))
        eng.on_text("tell me a joke")
        run(eng, clock, 1)
        self.assertTrue(said(eng, n))

    def test_louder_while_talking_does_not_interrupt(self):
        eng, clock = make()
        eng.on_text("hey buddy tell me a story")
        run(eng, clock, 5)
        story = eng.player.response
        v = eng.speaker.volume
        eng.on_text("hey buddy louder")
        self.assertGreater(eng.speaker.volume, v)
        self.assertIs(eng.player.response, story)

    def test_quiz_answer_and_timeout(self):
        eng, clock = make()
        eng.on_text("hey buddy number quiz")
        q = next(qq for qq in eng.brain.skills.bank["numbers"] if qq.text == eng.brain.expect.prompt[0].text)
        run_until_quiet(eng, clock)
        n = len(said(eng))
        eng.on_text(str(q.answers[0]))
        run(eng, clock, 0.2)
        self.assertEqual(said(eng, n)[0], "[correct]")
        run_until_quiet(eng, clock)
        run(eng, clock, 10)  # no answer: Buddy tells the answer
        self.assertTrue(any(t.startswith("The answer is") for t in said(eng, n)))


class TestEcho(unittest.TestCase):
    def test_buddy_saying_hey_buddy_is_not_a_command(self):
        eng, clock = make()
        eng._say(Response([Say("I'll keep our place. Say Hey Buddy, continue, when you want to hear the rest!")]))
        self.assertTrue(eng._is_echo("continue when you want to hear the rest"))
        eng._say(Response([Say("Once upon a time, a big lion said stop!")]))
        self.assertFalse(eng._is_echo("stop"))  # Buddy never said "Hey Buddy": a real command


if __name__ == "__main__":
    unittest.main()
