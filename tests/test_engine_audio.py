"""End-to-end test of the real-time engine with real models but fake audio devices.

Feeds a recorded command ("Hey Buddy, what does the cow say?") into the engine in
32 ms blocks, as a microphone would, and checks that Buddy answers with audio.
Skipped automatically when the models are not installed.
"""
import importlib.util
import random
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from buddy.config import Config
from buddy.learning.progress import Progress

MODELS = Path.home() / "buddy-data" / "models"
# The models are loaded *through* sherpa-onnx, so having the files is not enough: without the
# package this test used to run and die with SystemExit instead of skipping (finding M1).
HAVE_RUNTIME = importlib.util.find_spec("sherpa_onnx") is not None
HAVE = (HAVE_RUNTIME and (MODELS / "silero_vad.onnx").exists()
        and (MODELS / "vits-piper-en_US-ljspeech-medium").exists())
ASR = "moonshine-base" if (MODELS / "sherpa-onnx-moonshine-base-en-quantized-2026-02-27").exists() else "moonshine-tiny"


class FakeMic:
    def __init__(self, audio16k):
        self.x = audio16k
        self.pos = 0
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def read(self, timeout=0.05):
        if self.pos >= len(self.x):
            time.sleep(0.005)
            return np.zeros(512, dtype=np.float32)
        blk = self.x[self.pos:self.pos + 512]
        self.pos += 512
        return blk.copy()


class FakeSpeaker:
    def __init__(self, rate=48000):
        self.rate, self.volume, self.out, self._busy_until = rate, 0.7, [], 0

    def start(self):
        pass

    def play(self, x):
        self.out.append(x)
        self._busy_until = time.time() + 0.05  # pretend playback is quick

    def stop(self):
        self._busy_until = 0

    @property
    def busy(self):
        return time.time() < self._busy_until

    def close(self):
        pass


@unittest.skipUnless(HAVE, "speech models not installed")
class TestEngineAudio(unittest.TestCase):
    def test_voice_command_round_trip(self):
        from buddy.audio.dsp import resample
        from buddy.brain import Brain
        from buddy.engine import Engine
        from buddy.speech.cache import SpeechCache
        from buddy.speech.models import Recognizer, Synthesizer, Vad

        cfg = Config()
        cfg.data["paths"]["models"] = str(MODELS)
        tts = Synthesizer(MODELS, "piper", 2)
        # the "child" speaks
        x, sr = tts.synth("Hey Buddy, what does the cow say?", 0, 1.0)
        mic_audio = np.concatenate([np.zeros(8000), resample(x, sr, 16000), np.zeros(24000)]).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            cache = SpeechCache(Path(d), "piper-test", 48000, lambda t: tts.synth(t, 0, 1.0))
            brain = Brain(cfg, rng=random.Random(0), progress=Progress(None))
            spk = FakeSpeaker()
            eng = Engine(cfg, brain, FakeMic(mic_audio), spk, Recognizer(MODELS, ASR, 2),
                         Vad(MODELS), cache)
            heard = []
            orig = eng.on_text
            eng.on_text = lambda t: (heard.append(t), orig(t))
            eng.mic.start()
            t0 = time.time()
            while eng.mic.pos < len(mic_audio) and time.time() - t0 < 60:
                blk = eng.mic.read()
                for seg in eng.vad.accept(blk):
                    eng.on_text(eng.asr.transcribe(seg))
                eng._timers()
            while eng.player.active and time.time() - t0 < 60:  # Buddy speaks sentence by sentence
                eng._timers()
                time.sleep(0.01)
            self.assertTrue(heard, "nothing recognised")
            self.assertIn("cow", heard[0].lower())
            self.assertTrue(spk.out, "Buddy did not answer")
            answer_seconds = sum(len(a) for a in spk.out) / 48000
            self.assertGreater(answer_seconds, 2.0)
            # the answer was rendered into the cache and is replayed from it next time
            self.assertTrue(cache.has("The cow says... Moooo! Moooo!"))
            print(f"heard {heard[0]!r}; Buddy answered with {answer_seconds:.1f} s of audio")


if __name__ == "__main__":
    unittest.main()
