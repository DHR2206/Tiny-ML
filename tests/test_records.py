"""Interaction records - RECORDS (temporary analysis feature, see docs/RECORDS.md).

Checks what is recorded in the main situations, that the voice engine behaves exactly the same
with records switched on, that records are appended across restarts, that talk not meant for
Buddy is not stored, and that a recording problem never affects Buddy.
No models or sound card needed.
"""
import io
import json
import logging
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

from buddy.brain import Brain
from buddy.config import Config
from buddy.core import Say
from buddy.engine import Engine
from buddy.learning.progress import Progress
from buddy.records import DUCKED, NO_WAKE_WHILE_TALKING, Recorder, RecordingEngine
from tests.test_engine_session import Clock, Dummy, FakeRenderer, FakeSpeaker, run, run_until_quiet, said


def make(folder, record=True, seed=1, near_miss=0.7):
    cfg = Config()
    brain = Brain(cfg, rng=random.Random(seed), progress=Progress(None))
    clock = Clock()
    spk = FakeSpeaker(clock)
    rec = None
    if record:
        rec = Recorder(folder, "voice", near_miss)
        eng = RecordingEngine(cfg, brain, Dummy(), spk, Dummy(), Dummy(), Dummy(), clock=clock, recorder=rec)
    else:
        eng = Engine(cfg, brain, Dummy(), spk, Dummy(), Dummy(), Dummy(), clock=clock)
    eng.player.renderer = FakeRenderer(clock, spk)
    return eng, clock, rec


def read_all(folder):
    out = []
    for p in sorted(Path(folder).glob("*.jsonl")):
        out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    return out


def closed(rec, reason="test"):
    rec.close(reason)
    return read_all(rec.dir)


def inputs(rs):
    return [r for r in rs if r["kind"] == "interaction"]


def by_input(rs, text):
    found = [r for r in rs if r.get("input") == text]
    assert found, f"no record for {text!r}"
    return found[0]


class TestVoiceRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_follow_up_and_ignored_talk(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("hey buddy tell me a joke", 1.5)
        run_until_quiet(eng, clock)
        eng.on_text("count from one to ten", 1.2)  # listening window: no "Hey Buddy"
        run_until_quiet(eng, clock)
        eng.on_text("mummy look", 0.8)  # background talk in the listening window
        run(eng, clock, 20)  # the window closes: Buddy's goodbye line
        rs = closed(rec)

        joke = by_input(rs, "hey buddy tell me a joke")
        self.assertEqual((joke["state"], joke["path"], joke["status"]), ("asleep", "command", "ok"))
        self.assertEqual(joke["understood"]["intent"], "joke")
        self.assertTrue(joke["wake"]["heard"])
        self.assertEqual(joke["delivery"], "complete")
        self.assertEqual(joke["input_seconds"], 1.5)
        self.assertEqual(joke["input_via"], "microphone")
        self.assertEqual(joke["conversation"], 1)
        joke_lines = [t for t in said(eng) if t in joke["response"]]
        self.assertTrue(joke_lines)  # the exact sentences Buddy said

        count = by_input(rs, "count from one to ten")
        self.assertEqual((count["state"], count["path"], count["status"]), ("listening", "follow_up", "ok"))
        self.assertEqual(count["understood"]["slots"], {"start": 1, "end": 10})
        self.assertIn("One", count["response"])

        talk = by_input(rs, "mummy look")
        self.assertEqual((talk["status"], talk["path"], talk["response"]), ("ignored", "follow_up", None))
        self.assertEqual(talk["understood"]["intent"], "unknown")

        events = [r["event"] for r in rs if r["kind"] == "event"]
        self.assertIn("conversation_end", events)
        self.assertEqual(events[-1], "program_end")
        end = [r for r in rs if r.get("event") == "conversation_end"][0]
        self.assertTrue(end["response"])

    def test_pause_cuts_the_story_and_is_recorded(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("hey buddy tell me a story", 1.0)
        run(eng, clock, 8)
        eng.on_text("hey buddy pause", 1.0)
        run_until_quiet(eng, clock)
        eng.on_text("continue", 0.6)
        run(eng, clock, 2)
        rs = closed(rec)
        story = by_input(rs, "hey buddy tell me a story")
        self.assertEqual(story["delivery"], "cut_off")
        cut = story["cut_off_at"]
        self.assertGreaterEqual(cut["heard_sentences"], 1)
        self.assertIn(cut["stopped_during"], story["response"])
        pause = by_input(rs, "hey buddy pause")
        self.assertEqual((pause["state"], pause["path"], pause["action"]), ("talking", "barge_in", "pause"))
        self.assertEqual(pause["delivery"], "complete")
        cont = by_input(rs, "continue")
        self.assertEqual(cont["action"], "continue")
        self.assertIn(cut["stopped_during"], cont["response"])  # resumed from that sentence
        self.assertEqual(cont["delivery"], "cut_off")  # the program ended while it was playing
        self.assertEqual(len({r["conversation"] for r in inputs(rs)}), 1)

    def test_quiz_answer_and_timeout(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("hey buddy riddle", 1.0)
        run_until_quiet(eng, clock)
        eng.on_text("a cat", 0.7)
        run_until_quiet(eng, clock)
        eng.on_text("another riddle", 0.9)  # listening window
        run_until_quiet(eng, clock)
        run(eng, clock, 30)  # no answer: Buddy tells it
        rs = closed(rec)
        riddle = by_input(rs, "hey buddy riddle")
        ans = by_input(rs, "a cat")
        self.assertEqual((ans["state"], ans["path"]), ("answering", "answer"))
        self.assertTrue(ans["question"].endswith("?"))
        self.assertIn(ans["question"], riddle["response"])  # the question Buddy had asked
        second = by_input(rs, "another riddle")
        timeouts = [r for r in rs if r.get("event") == "answer_timeout"]
        self.assertEqual(len(timeouts), 1)
        self.assertIn(timeouts[0]["question"], second["response"])
        self.assertIn("The answer is", timeouts[0]["response"])
        self.assertEqual(timeouts[0]["delivery"], "complete")

    def test_only_speech_meant_for_buddy_is_kept(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("what a lovely day", 1.0)  # asleep, ordinary talk: never stored
        eng.on_text("i am going outside now", 1.0)
        eng.on_text("hey bobby tell me a joke", 1.5)  # almost "Hey Buddy" + a request
        eng.on_text("buddy the dog is asleep", 1.2)  # "buddy" heard, but not a request
        rs = closed(rec)
        texts = [r["input"] for r in inputs(rs)]
        self.assertNotIn("what a lovely day", texts)
        self.assertNotIn("i am going outside now", texts)
        miss = by_input(rs, "hey bobby tell me a joke")
        self.assertEqual((miss["status"], miss["conversation"], miss["response"]), ("near_miss", None, None))
        self.assertEqual(miss["understood"]["intent"], "joke")
        self.assertEqual(by_input(rs, "buddy the dog is asleep")["status"], "near_miss")
        self.assertEqual(said(eng), [])  # and Buddy itself did nothing, as before

    def test_while_talking(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("hey buddy tell me a story", 1.0)
        run(eng, clock, 4)
        story_words = eng.player.response.texts()[0]
        eng.on_text(story_words, 2.0)  # the microphone hears Buddy itself: not stored
        eng.on_text("stop", 0.4)  # no "Hey Buddy" while Buddy talks: ignored by Buddy, stored
        eng.on_text("hey buddy", 0.6)  # Buddy talks softer and waits
        eng.on_text("pause", 0.4)
        run_until_quiet(eng, clock)
        rs = closed(rec)
        self.assertNotIn(story_words, [r["input"] for r in inputs(rs)])
        stop = by_input(rs, "stop")
        self.assertEqual((stop["status"], stop["note"]), ("ignored", NO_WAKE_WHILE_TALKING))
        self.assertEqual(stop["understood"]["intent"], "stop")
        duck = by_input(rs, "hey buddy")
        self.assertEqual((duck["path"], duck["status"], duck["note"]), ("wake_only", "ok", DUCKED))
        pause = by_input(rs, "pause")
        self.assertEqual((pause["state"], pause["path"], pause["action"]), ("talking", "barge_in", "pause"))

    def test_keys_mute_and_problems(self):
        eng, clock, rec = make(self.dir)
        eng.on_text("hey buddy tell me a story", 1.0)
        run(eng, clock, 3)
        eng._do_mute()  # the mute button
        run_until_quiet(eng, clock)
        eng._do_mute()
        run_until_quiet(eng, clock)
        eng._key("p")  # nothing playing: "continue"
        run(eng, clock, 2)
        eng._key("x")
        run_until_quiet(eng, clock)
        rs = closed(rec)
        self.assertEqual(by_input(rs, "hey buddy tell me a story")["delivery"], "cut_off")
        events = [r["event"] for r in rs if r["kind"] == "event"]
        self.assertIn("muted", events)
        self.assertIn("unmuted", events)
        keys = [r for r in inputs(rs) if r["input"].startswith("[key")]
        self.assertEqual([k["input"] for k in keys], ["[key p]", "[key x]"])
        self.assertEqual([k["path"] for k in keys], ["key", "key"])
        self.assertEqual([k["input_via"] for k in keys], ["keyboard", "keyboard"])
        self.assertEqual(keys[1]["understood"]["intent"], "quit")

    def test_missing_speech_clip_is_reported(self):
        eng, clock, rec = make(self.dir)
        render = eng.player.renderer.render_item

        def flaky(it):  # like the real Renderer when a sentence is not in the speech cache
            if isinstance(it, Say) and "?" in it.text:
                logging.getLogger("buddy").warning("missing speech clip (run build-content): %s", it.text[:60])
                return np.zeros(0, dtype=np.float32), 0
            return render(it)

        eng.player.renderer.render_item = flaky
        eng.on_text("hey buddy tell me a riddle", 1.0)
        run(eng, clock, 10)
        rs = closed(rec)
        r = by_input(rs, "hey buddy tell me a riddle")
        self.assertEqual(r["status"], "incomplete")
        self.assertTrue(any("missing speech clip" in p for p in r["problems"]))

    def test_errors_are_recorded_and_raised_as_before(self):
        eng, clock, rec = make(self.dir)
        with mock.patch.object(eng.brain.skills, "handle", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                eng.on_text("hey buddy tell me a joke", 1.0)
        rs = closed(rec)
        r = by_input(rs, "hey buddy tell me a joke")
        self.assertEqual(r["status"], "error")
        self.assertEqual((r["error"]["type"], r["error"]["message"]), ("RuntimeError", "boom"))
        self.assertIn("boom", r["error"]["traceback"])

    def test_a_recording_problem_never_reaches_buddy(self):
        not_a_folder = Path(self.dir) / "file.txt"
        not_a_folder.write_text("x")
        eng, clock, rec = make(not_a_folder)  # every write fails
        with mock.patch("buddy.records.Recorder.end", side_effect=ValueError("bug")), \
                self.assertLogs("buddy.records", "WARNING"):
            eng.on_text("hey buddy tell me a joke", 1.0)
            run_until_quiet(eng, clock)
        self.assertTrue(any("Why" in t or "?" in t for t in said(eng)))  # Buddy still told the joke
        with self.assertLogs("buddy.records", "WARNING") as logs:
            rec.close()  # the folder can't be written: one warning, no exception
        self.assertIn("could not write", logs.output[0])

    def test_same_behaviour_with_and_without_records(self):
        script = [("hey buddy tell me a joke", 1.0, "quiet"), ("tell me a story", 1.0, 5),
                  ("hey buddy pause", 0.8, "quiet"), ("continue", 0.6, 3), ("hey buddy stop", 0.7, "quiet"),
                  ("hey buddy riddle", 1.0, "quiet"), ("a cat", 0.5, "quiet"), (None, 0, 30),
                  ("what a lovely day", 1.0, 1), ("hey buddy", 0.5, "quiet"), ("count from 1 to 10", 1.0, "quiet"),
                  ("key r", 0, "quiet"), ("louder", 0.5, "quiet"), ("mute", 0, "quiet"), ("mute", 0, "quiet"),
                  ("hey buddy what is two plus two", 1.0, "quiet"), ("blah blah", 0.5, 2),
                  ("hey buddy goodbye", 0.8, "quiet"), (None, 0, 30)]

        def play(record):
            eng, clock, rec = make(self.dir, record=record, seed=7)
            for text, secs, then in script:
                if text == "key r":
                    eng._key("r")
                elif text == "mute":
                    eng._do_mute()
                elif text:
                    eng.on_text(text, secs)
                if then == "quiet":
                    run_until_quiet(eng, clock)
                else:
                    run(eng, clock, then)
            if rec:
                rec.close()
            return eng.player.renderer.log, (eng.awake_until, eng.paused, eng.speaker.volume, eng.muted)

        plain, state_plain = play(False)
        recorded, state_recorded = play(True)
        self.assertGreater(len(plain), 20)
        self.assertEqual(plain, recorded)  # every line, at the same moment
        self.assertEqual(state_plain, state_recorded)
        rs = read_all(self.dir)
        self.assertGreater(len(inputs(rs)), 12)


class TestRecordFile(unittest.TestCase):
    def test_lines_fields_ids_and_append_after_restart(self):
        with tempfile.TemporaryDirectory() as d:
            for _ in range(2):  # two program runs
                eng, clock, rec = make(d)
                eng.on_text("hey buddy tell me a joke", 1.0)
                run_until_quiet(eng, clock)
                rec.close()
            files = list(Path(d).glob("*.jsonl"))
            self.assertEqual([f.name for f in files], [f"{datetime.now():%Y-%m-%d}_voice.jsonl"])
            rs = read_all(d)
            self.assertEqual(len(inputs(rs)), 2)  # appended, not overwritten
            self.assertEqual(len({r["id"] for r in rs}), len(rs))  # unique ids
            for r in rs:
                for key in ("id", "time", "source", "kind", "status"):
                    self.assertIn(key, r)
                datetime.fromisoformat(r["time"])  # ISO 8601 with the UTC offset
                self.assertIsNotNone(datetime.fromisoformat(r["time"]).utcoffset())
            self.assertNotEqual(rs[0]["id"].rsplit("-", 1)[0], rs[-1]["id"].rsplit("-", 1)[0])  # run ids differ

    def test_nothing_personal_or_audio(self):
        with tempfile.TemporaryDirectory() as d:
            eng, clock, rec = make(d)
            eng.on_text("hey buddy what does the cow say", 1.0)
            run_until_quiet(eng, clock)
            rs = closed(rec)
            blob = json.dumps(rs)
            self.assertNotIn(str(Path.home()), blob)
            self.assertNotIn("samples", blob)


class TestChatAndSimulateRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = Config()
        cfg.data["records"]["folder"] = self.tmp.name + "/records"
        cfg.data["paths"]["data"] = self.tmp.name + "/data"  # (never the real progress database)
        self.cfg = cfg

    def tearDown(self):
        self.tmp.cleanup()

    def test_chat(self):
        from buddy.__main__ import cmd_chat

        typed = "hey buddy tell me a joke\nwhat is two plus two\nblah blah\nhello there\n\n\nhey buddy riddle\n\n"
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(typed)), redirect_stdout(out):
            cmd_chat(self.cfg, None)
        rs = read_all(self.tmp.name + "/records")
        self.assertTrue(all(r["source"] == "chat" for r in rs))
        self.assertEqual(rs[0]["event"], "program_start")
        self.assertEqual(rs[0]["settings"]["age_setting"], 5)
        joke = by_input(rs, "hey buddy tell me a joke")
        self.assertEqual((joke["path"], joke["status"], joke["delivery"]), ("command", "ok", "printed"))
        self.assertIn(joke["response"], out.getvalue())  # exactly what was printed
        self.assertEqual(by_input(rs, "what is two plus two")["understood"]["slots"], {"a": 2, "b": 2, "op": "plus"})
        self.assertEqual(by_input(rs, "blah blah")["status"], "ignored")
        end = [r for r in rs if r.get("event") == "conversation_end"]
        self.assertEqual(len(end), 1)
        self.assertIn(end[0]["response"], out.getvalue())
        timeout = [r for r in rs if r.get("event") == "answer_timeout"]
        self.assertEqual(len(timeout), 1)
        self.assertIn(timeout[0]["question"], by_input(rs, "hey buddy riddle")["response"])
        self.assertIn(timeout[0]["response"], out.getvalue())
        self.assertEqual([r["conversation"] for r in inputs(rs) if r["input"].startswith("hey buddy")], [1, 2])
        self.assertEqual(rs[-1]["event"], "program_end")

    def test_simulate(self):
        from buddy import __main__ as cli

        heard = iter(["hey buddy count to five", "what a lovely day", "hey bobby tell me a joke"])

        class FakeAsr:
            def __init__(self, *a, **k):
                pass

            def transcribe(self, seg):
                return next(heard)

        class FakeVad:
            def __init__(self, *a, **k):
                pass

            def accept(self, x):
                return [np.zeros(16000, dtype=np.float32)] * 3

            def flush(self):
                return []

        with mock.patch("buddy.speech.models.Recognizer", FakeAsr), mock.patch("buddy.speech.models.Vad", FakeVad), \
                mock.patch("buddy.audio.dsp.read_wav", return_value=(np.zeros(16000, dtype=np.float32), 16000)), \
                redirect_stdout(io.StringIO()):
            cli.cmd_simulate(self.cfg, type("A", (), {"files": ["C:/Users/someone/tests/count.wav"]})())
        rs = read_all(self.tmp.name + "/records")
        count = by_input(rs, "hey buddy count to five")
        self.assertEqual((count["source"], count["input_via"], count["file"]), ("simulate", "wav_file", "count.wav"))
        self.assertEqual(count["input_seconds"], 1.0)
        self.assertIn("asr", count["timing_ms"])
        self.assertNotIn("what a lovely day", [r.get("input") for r in rs])
        self.assertEqual(by_input(rs, "hey bobby tell me a joke")["status"], "near_miss")


if __name__ == "__main__":
    unittest.main()
