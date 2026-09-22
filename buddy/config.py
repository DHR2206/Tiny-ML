"""Configuration loading (TOML, standard library only)."""
from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("BUDDY_HOME", Path.home() / "buddy-data"))

DEFAULTS: dict[str, Any] = {
    "child": {
        # Age drives how tolerant speech matching is (see nlu/phonetics.py) and quiz start level.
        "age": 5,
        "name": "",  # optional; not used in speech unless pre-rendered
    },
    "paths": {
        "models": str(HOME / "models"),
        "cache": str(HOME / "cache"),
        "data": str(HOME / "data"),
        "sounds": "",  # optional folder with CC0 animal sound WAVs (<animal>.wav)
    },
    "audio": {
        "input_device": None,  # None = system default; or name substring / index
        "output_device": None,
        "input_rate": None,  # None = device default; resampled to 16 kHz internally
        "output_rate": 48000,
        "volume": 0.7,  # 0..1 software gain
        "min_volume": 0.2,
        "max_volume": 0.85,  # parental volume cap
        "barge_in": True,  # allow "Hey Buddy, stop!" while speaking
    },
    "wake": {
        "phrases": ["hey buddy", "hi buddy", "hello buddy", "okay buddy", "buddy"],
        "threshold": 0.8,
        "listen_seconds": 7.0,  # how long to wait for a command after "Hey Buddy"
    },
    "asr": {
        "model": "moonshine-base",  # moonshine-base | moonshine-tiny
        "threads": 3,
    },
    "vad": {
        "threshold": 0.5,
        "min_silence": 0.55,  # seconds of silence that end an utterance (kids pause a lot)
        "min_speech": 0.20,
        "max_utterance": 10.0,
    },
    "tts": {
        "engine": "kokoro",  # kokoro | piper
        "voice": 1,  # kokoro v0.19 speaker id (1 = af_bella)
        "speed": 0.9,  # a little slower for young listeners
        "threads": 3,
        "runtime_fallback": True,  # synthesise missing phrases on the fly
    },
    "nlu": {
        "min_confidence": 0.55,
        "answer_window": 9.0,  # seconds to answer a quiz question without the wake word
    },
    "parental": {
        "quiet_hours": "",  # e.g. "21:00-06:30" (needs a correct clock / RTC battery)
        "daily_limit_minutes": 0,  # 0 = unlimited
    },
    "hardware": {
        "mute_button_gpio": 17,  # BCM pin; button to GND. None disables
        "led_gpio": 27,  # optional LED (+330 ohm) BCM pin; None disables
        "keyboard": True,  # 'm' = mute toggle, 't' = talk (simulated wake), 'q' = quit
    },
    "privacy": {
        "log_transcripts": False,  # never log what the child says unless debugging
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    def __getitem__(self, section: str) -> dict:
        return self.data[section]

    @property
    def models_dir(self) -> Path:
        return Path(os.path.expanduser(self.data["paths"]["models"]))

    @property
    def cache_dir(self) -> Path:
        return Path(os.path.expanduser(self.data["paths"]["cache"]))

    @property
    def data_dir(self) -> Path:
        return Path(os.path.expanduser(self.data["paths"]["data"]))

    @property
    def child_age(self) -> int:
        return int(self.data["child"]["age"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load defaults, then overlay the TOML file (if any)."""
    data = copy.deepcopy(DEFAULTS)
    candidates = [path] if path else [os.environ.get("BUDDY_CONFIG"), HOME / "buddy.toml", "buddy.toml"]
    for c in candidates:
        if c and Path(c).is_file():
            with open(c, "rb") as fh:
                data = _merge(data, tomllib.load(fh))
            break
    # TOML has no null; allow "none"/-1 for optional pins/devices
    for sec, key in [("hardware", "mute_button_gpio"), ("hardware", "led_gpio")]:
        if data[sec][key] in ("none", "", -1):
            data[sec][key] = None
    return Config(data)
