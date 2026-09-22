"""Loads the JSON content packs shipped in buddy/content (and user additions)."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "content"


class Content:
    def __init__(self, content_dir: Path = CONTENT_DIR):
        self.dir = Path(content_dir)

    def _load(self, name: str) -> dict:
        p = self.dir / name
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    @cached_property
    def animals(self) -> list[dict]:
        return self._load("animals.json")["animals"]

    @cached_property
    def poems(self) -> list[dict]:
        base = self._load("poems.json").get("poems", [])
        user = self._load("rhymes_user.json").get("poems", [])
        return base + [p for p in user if p.get("lines")]

    @cached_property
    def stories(self) -> list[dict]:
        return self._load("stories.json")["stories"]

    @cached_property
    def lessons(self) -> dict:
        return self._load("lessons.json")

    @cached_property
    def talk(self) -> dict:
        return self._load("talk.json")

    @cached_property
    def phrases(self) -> dict[str, list[str]]:
        return self._load("phrases.json")

    def animal(self, aid: str) -> dict:
        return next(a for a in self.animals if a["id"] == aid)
