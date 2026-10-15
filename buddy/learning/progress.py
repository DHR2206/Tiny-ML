"""On-device learning progress (adaptive quiz engine).

Privacy: only item identifiers and right/wrong counts are stored - never audio,
never transcripts, never names.  The database lives on the device's SD card and
can be wiped with `python -m buddy reset-progress`.

Adaptation:
  * Leitner boxes per item (box 0 = new/weak ... box 4 = mastered). Correct -> box+1,
    wrong -> box 0.  Items in low boxes are asked more often.
  * A level (1..5) per domain.  The level goes up when the child answers >= 75 % of
    the last 4 questions correctly and down when <= 25 %, keeping the child in the
    "just challenging enough" zone.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

WINDOW = 4
UP, DOWN = 0.75, 0.25
MAX_LEVEL = 5


SCHEMA = """
    CREATE TABLE IF NOT EXISTS item (domain TEXT, item TEXT, box INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0, wrong INTEGER DEFAULT 0, last_seen REAL DEFAULT 0,
        PRIMARY KEY (domain, item));
    CREATE TABLE IF NOT EXISTS domain (domain TEXT PRIMARY KEY, level INTEGER, recent TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, seconds REAL DEFAULT 0);
"""


class Progress:
    """The database file is opened for each operation and closed again.  Keeping one connection
    open for the life of the object leaked a handle, and on Windows an open handle stops the file
    being deleted at all, which is what broke three tests in tearDown (finding M1).  An in-memory
    database only exists while a connection is open, so that one is held."""

    def __init__(self, path: str | Path | None, start_level: int = 1):
        self.path = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(self.path, check_same_thread=False) if not path else None
        self.start_level = start_level
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.commit()

    @contextmanager
    def _connect(self):
        if self._mem is not None:
            yield self._mem
            return
        db = sqlite3.connect(self.path, check_same_thread=False)
        try:
            yield db
        finally:
            db.close()

    @property
    def db(self) -> sqlite3.Connection:
        """A connection for inspecting the data (e.g. iterdump).  For a file-backed database this
        is an in-memory *copy*, so that reading it never holds the file open; use the methods
        above to change anything, as writes to the copy are not kept."""
        if self._mem is not None:
            return self._mem
        src = sqlite3.connect(self.path)
        try:
            snapshot = sqlite3.connect(":memory:", check_same_thread=False)
            src.backup(snapshot)
            return snapshot
        finally:
            src.close()

    def close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- domains
    def level(self, domain: str) -> int:
        with self._connect() as db:
            row = db.execute("SELECT level FROM domain WHERE domain=?", (domain,)).fetchone()
        return row[0] if row else self.start_level

    def _recent(self, domain: str) -> str:
        with self._connect() as db:
            row = db.execute("SELECT recent FROM domain WHERE domain=?", (domain,)).fetchone()
        return row[0] if row else ""

    def record(self, domain: str, item: str, correct: bool, first_try: bool = True) -> int:
        """Record an answer; returns the (possibly new) domain level."""
        now = time.time()
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO item(domain, item) VALUES (?,?)", (domain, item))
            if correct and first_try:
                db.execute("UPDATE item SET box=MIN(box+1,4), correct=correct+1, last_seen=? "
                           "WHERE domain=? AND item=?", (now, domain, item))
            else:
                db.execute("UPDATE item SET box=0, wrong=wrong+1, last_seen=? WHERE domain=? AND item=?",
                           (now, domain, item))
            db.commit()
        recent = (self._recent(domain) + ("1" if correct and first_try else "0"))[-WINDOW:]
        lvl = self.level(domain)
        if len(recent) >= WINDOW:
            rate = recent.count("1") / len(recent)
            if rate >= UP and lvl < MAX_LEVEL:
                lvl, recent = lvl + 1, ""
            elif rate <= DOWN and lvl > 1:
                lvl, recent = lvl - 1, ""
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO domain(domain, level, recent) VALUES (?,?,?)",
                       (domain, lvl, recent))
            db.commit()
        return lvl

    def boxes(self, domain: str) -> dict[str, tuple[int, float]]:
        with self._connect() as db:
            rows = db.execute("SELECT item, box, last_seen FROM item WHERE domain=?", (domain,)).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def summary(self) -> list[tuple]:
        with self._connect() as db:
            return db.execute(
                "SELECT d.domain, d.level, COALESCE(SUM(i.correct),0), COALESCE(SUM(i.wrong),0) "
                "FROM domain d LEFT JOIN item i ON i.domain=d.domain GROUP BY d.domain").fetchall()

    # --------------------------------------------------------------- usage time
    def add_usage(self, seconds: float) -> None:
        day = time.strftime("%Y-%m-%d")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO usage(day) VALUES (?)", (day,))
            db.execute("UPDATE usage SET seconds=seconds+? WHERE day=?", (seconds, day))
            db.commit()

    def usage_today(self) -> float:
        with self._connect() as db:
            row = db.execute("SELECT seconds FROM usage WHERE day=?",
                             (time.strftime("%Y-%m-%d"),)).fetchone()
        return row[0] if row else 0.0

    def reset(self) -> None:
        with self._connect() as db:
            db.executescript("DELETE FROM item; DELETE FROM domain; DELETE FROM usage;")
            db.commit()
