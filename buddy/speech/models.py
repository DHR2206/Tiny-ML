"""Thin wrappers around sherpa-onnx (Apache-2.0) for VAD, ASR and TTS.

Models (all downloaded by scripts/download_models.sh):
  * ASR  Moonshine v2 English (MIT)             sherpa-onnx-moonshine-{base,tiny}-en-quantized-2026-02-27
  * VAD  Silero VAD (MIT)                        silero_vad.onnx
  * TTS  Kokoro-82M int8 English (Apache-2.0)    kokoro-int8-en-v0_19
         or Piper LJSpeech voice (public-domain data) vits-piper-en_US-ljspeech-medium
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIRS = {
    "moonshine-base": "sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
    "kokoro": "kokoro-int8-en-v0_19",
    "piper": "vits-piper-en_US-ljspeech-medium",
    "vad": "silero_vad.onnx",
}


def _sherpa():
    try:
        import sherpa_onnx  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise SystemExit("sherpa-onnx is not installed. Run: pip install sherpa-onnx") from e
    return sherpa_onnx


def _need(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}\nRun: bash scripts/download_models.sh")
    return str(p)


class Recognizer:
    """Offline (utterance-level) speech recognition with Moonshine."""

    def __init__(self, models_dir: Path, name: str = "moonshine-base", threads: int = 3):
        so = _sherpa()
        d = Path(models_dir) / MODEL_DIRS[name]
        self.rec = so.OfflineRecognizer.from_moonshine_v2(
            encoder=_need(d / "encoder_model.ort"),
            decoder=_need(d / "decoder_model_merged.ort"),
            tokens=_need(d / "tokens.txt"),
            num_threads=threads,
        )

    def transcribe(self, samples: np.ndarray, rate: int = 16000) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(rate, np.ascontiguousarray(samples, dtype=np.float32))
        self.rec.decode_stream(s)
        return s.result.text.strip()


class Vad:
    """Silero voice-activity detector; yields complete speech segments."""

    def __init__(self, models_dir: Path, threshold: float = 0.5, min_silence: float = 0.55,
                 min_speech: float = 0.2, max_speech: float = 10.0):
        so = _sherpa()
        c = so.VadModelConfig()
        c.silero_vad.model = _need(Path(models_dir) / MODEL_DIRS["vad"])
        c.silero_vad.threshold = threshold
        c.silero_vad.min_silence_duration = min_silence
        c.silero_vad.min_speech_duration = min_speech
        c.silero_vad.max_speech_duration = max_speech
        c.sample_rate = 16000
        self.window = c.silero_vad.window_size
        self.vad = so.VoiceActivityDetector(c, buffer_size_in_seconds=30)
        self._buf = np.zeros(0, dtype=np.float32)

    def accept(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        while len(self._buf) >= self.window:
            self.vad.accept_waveform(self._buf[: self.window])
            self._buf = self._buf[self.window:]
        segs = []
        while not self.vad.empty():
            segs.append(np.array(self.vad.front.samples, dtype=np.float32))
            self.vad.pop()
        return segs

    def speaking(self) -> bool:
        return bool(self.vad.is_speech_detected())

    def flush(self) -> list[np.ndarray]:
        """End of input: return any pending segment."""
        self.vad.flush()
        return self.accept(np.zeros(0, dtype=np.float32))

    def reset(self) -> None:
        self.vad.reset()
        self._buf = np.zeros(0, dtype=np.float32)


class Synthesizer:
    """Text-to-speech (used at setup time to pre-render; optional at runtime)."""

    def __init__(self, models_dir: Path, engine: str = "kokoro", threads: int = 3):
        so = _sherpa()
        d = Path(models_dir) / MODEL_DIRS[engine]
        if engine == "kokoro":
            model = so.OfflineTtsModelConfig(
                kokoro=so.OfflineTtsKokoroModelConfig(
                    model=_need(d / "model.int8.onnx"), voices=_need(d / "voices.bin"),
                    tokens=_need(d / "tokens.txt"), data_dir=_need(d / "espeak-ng-data")),
                num_threads=threads)
        else:
            model = so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(
                    model=_need(d / "en_US-ljspeech-medium.onnx"), tokens=_need(d / "tokens.txt"),
                    data_dir=_need(d / "espeak-ng-data")),
                num_threads=threads)
        self.tts = so.OfflineTts(so.OfflineTtsConfig(model=model))
        self.engine = engine

    def synth(self, text: str, voice: int = 1, speed: float = 0.9) -> tuple[np.ndarray, int]:
        sid = voice if self.engine == "kokoro" else 0
        a = self.tts.generate(text, sid=sid, speed=speed)
        return np.array(a.samples, dtype=np.float32), a.sample_rate
