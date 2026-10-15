"""Configuration loading (TOML, standard library only)."""
from __future__ import annotations

import copy
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HOME = Path(os.environ.get("BUDDY_HOME", Path.home() / "buddy-data"))


class ConfigError(Exception):
    """A settings file that cannot be used, reported in words a parent can act on."""

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
    "session": {
        # Conversation mode: after each task Buddy keeps listening, so "Hey Buddy" is only needed to wake it.
        "follow_up_seconds": 15.0,  # listening window after each task (0 = old behaviour: wake phrase every time)
        "prompt_every": 2,  # short "What next?" line after every Nth task (a soft chime after every task)
        "pause_memory_seconds": 180.0,  # "continue" works this long after "pause"
        "barge_in_window": 4.0,  # "Hey Buddy" heard while Buddy talks: seconds to say pause / stop / repeat / quit
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
        # How clearly the words of a request must have been heard before Buddy will start an
        # activity on them.  0 = off, which is the behaviour Buddy has always had (before v0.4
        # this setting was declared here but never read by anything).
        #
        # It trades unclear speech against nonsense, and the useful range depends on [child] age,
        # because the phoneme distance is age-scheduled.  Measured over the child-speech sets in
        # tests/test_nlu.py and the QA probe (19 mispronounced commands, 16 nonsense utterances):
        #
        #   value   age 3-4                       age 5 (default)               age 8
        #   0.00    3 nonsense get an activity    2 get an activity             0
        #   0.70    2 nonsense                    2 nonsense                    8 commands lost
        #   0.74    1 nonsense                    2 nonsense, 1 command lost    10 commands lost
        #   0.78    1 nonsense, 6 commands lost   13 commands lost              16 commands lost
        #
        # So there is no single value that is right for every age: raising it makes Buddy answer
        # "I didn't catch that" more often, which is safer but less forgiving of a 4-year-old.
        # Left off by default; see docs/FIXES.md before changing it.
        "min_confidence": 0.0,
        "answer_window": 9.0,  # seconds to answer a quiz question without the wake word
        "fallback": "none",  # optional extra intent classifier (nlu/fallback.py); none available yet
    },
    "parental": {
        "quiet_hours": "",  # e.g. "21:00-06:30" (needs a correct clock / RTC battery)
        "daily_limit_minutes": 0,  # 0 = unlimited
    },
    "hardware": {
        "mute_button_gpio": 17,  # BCM pin; button to GND. None disables
        "led_gpio": 27,  # optional LED (+330 ohm) BCM pin; None disables
        # keyboard helper: m = mute, t = talk (as "Hey Buddy"), p = pause/continue, r = repeat,
        # x = end the conversation, q = quit the program
        "keyboard": True,
    },
    "privacy": {
        "log_transcripts": False,  # never log what the child says unless debugging
    },
    # RECORDS (temporary): interaction records for analysis - remove before the final release (docs/RECORDS.md)
    "records": {
        "enabled": True,  # what Buddy heard and said, one JSON line per interaction (text only, no audio)
        "folder": str(HOME / "records"),  # outside the project folder, so records never go to git
        "near_miss_score": 0.7,  # while asleep, keep only speech that sounded at least this much like "Hey Buddy"
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
    """Load defaults, then overlay the TOML file (if any).

    Raises ConfigError for a file that was asked for but cannot be used, so a parent is told
    rather than silently getting the built-in defaults (finding L2).
    """
    data = copy.deepcopy(DEFAULTS)
    if path and not Path(path).is_file():
        raise ConfigError(f"no settings file at {path}")
    candidates = [path] if path else [os.environ.get("BUDDY_CONFIG"), HOME / "buddy.toml", "buddy.toml"]
    for c in candidates:
        if c and Path(c).is_file():
            try:
                with open(c, "rb") as fh:
                    data = _merge(data, tomllib.load(fh))
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"{c} is not valid TOML: {e}") from e
            except OSError as e:
                raise ConfigError(f"cannot read {c}: {e}") from e
            break
    # TOML has no null; allow "none"/-1 for optional pins/devices
    for sec, key in [("hardware", "mute_button_gpio"), ("hardware", "led_gpio")]:
        if data[sec][key] in ("none", "", -1):
            data[sec][key] = None
    _check(data)
    return Config(data)


def _check(data: dict) -> None:
    """Put obviously impossible settings back to something safe and say so.  A wrong number in a
    settings file must never leave the child with a toy that does nothing (finding L1)."""
    limit = data["parental"].get("daily_limit_minutes")
    if isinstance(limit, (int, float)) and limit < 0:
        log.warning("[parental] daily_limit_minutes = %s is negative; treating it as 0 (no limit).", limit)
        data["parental"]["daily_limit_minutes"] = 0
    qh = data["parental"].get("quiet_hours") or ""
    if qh and not re.fullmatch(r"\d{1,2}:\d{2}-\d{1,2}:\d{2}", str(qh).strip()):
        log.warning("[parental] quiet_hours = %r is not like '21:00-06:30'; ignoring it.", qh)
        data["parental"]["quiet_hours"] = ""
    age = data["child"].get("age")
    if not isinstance(age, (int, float)) or not 2 <= age <= 12:
        log.warning("[child] age = %r is outside 2-12; using 5.", age)
        data["child"]["age"] = 5
    mc = data["nlu"].get("min_confidence")
    if not isinstance(mc, (int, float)) or not 0.0 <= mc <= 1.0:
        # Above 1.0 nothing can ever clear the bar, so Buddy would understand nothing at all.
        log.warning("[nlu] min_confidence = %r is outside 0.0-1.0; using 0 (off).", mc)
        data["nlu"]["min_confidence"] = 0.0
    fb = data["nlu"].get("fallback")
    if fb and str(fb).lower() != "none":
        log.warning("[nlu] fallback = %r is not available; using none.", fb)
        data["nlu"]["fallback"] = "none"
