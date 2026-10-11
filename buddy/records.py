"""Interaction records: what Buddy heard, what it understood and what it said.

RECORDS (temporary): for analysing how Buddy does with real children.  Remove this feature
before the final release - docs/RECORDS.md lists exactly what to delete.

Each record is one JSON object on its own line (JSON Lines), appended to
    <records folder>/<YYYY-MM-DD>_<source>.jsonl        source = voice | chat | simulate
The folder is outside the project (default ~/buddy-data/records), so records never end up in git.

Recorded
  * everything said after "Hey Buddy" and during a conversation, including speech Buddy ignored
  * near misses: speech that sounded like "Hey Buddy" but did not wake Buddy
  * what Buddy said back (exact text) and whether it was heard to the end
  * Buddy's own lines without a request (greeting, prompts, timeouts, mute), program start / end
  * errors (with traceback) and warnings such as a missing speech clip
Never recorded
  * audio, other talk while Buddy sleeps, Buddy's own voice (echo), names, user or device ids, file paths

Recording never changes what Buddy does:
  * RecordingEngine is a subclass of Engine; every override runs the original method unchanged.
  * Recorder.attach(brain) wraps a few Brain / WakeMatcher methods only to observe their
    arguments and results, which pass through untouched.
  * A background thread writes the files; any recording problem is logged once and ignored.
  * Real errors are recorded and then re-raised exactly as before.
"""
from __future__ import annotations

import atexit
import itertools
import json
import logging
import os
import queue
import secrets
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from .core import Response, Say
from .engine import Engine
from .nlu.normalize import normalize

log = logging.getLogger("buddy.records")

NOT_A_REQUEST = ("unknown", "none", "curiosity")
# Buddy's lines said without a child's request (Response.intent) -> event name in the record
EVENTS = {"boot": "greeting", "follow_up": "follow_up_prompt", "sleep": "conversation_end",
          "timeout": "conversation_end", "muted": "muted", "unmuted": "unmuted"}
ECHO_SHARE = 0.6  # speech heard while Buddy talks that is mostly Buddy's own words = its echo (not kept)

DUCKED = "'Hey Buddy' while Buddy was talking: Buddy talked softer and waited for the command"
ECHO = "Buddy's own voice (echo of its reply)"
UNCLEAR = "not a clear command while Buddy was talking"
NO_WAKE_WHILE_TALKING = "no 'Hey Buddy' while Buddy was talking (then Buddy only listens for 'Hey Buddy ...')"

# order of the fields in each line (easier to read; unknown fields go last)
FIELDS = ["id", "time", "source", "kind", "event", "conversation", "state", "input", "input_via", "input_seconds",
          "file", "wake", "path", "question", "understood", "response", "response_intent", "action", "after",
          "delivery", "cut_off_at", "status", "note", "problems", "error", "timing_ms", "settings"]


class Entry:
    """A record being built.  Written once the input has been handled and Buddy's reply has been
    heard to the end (or was cut off)."""

    def __init__(self, data: dict[str, Any], day: str):
        self.data = data
        self.day = day
        self.started = time.perf_counter()
        self.wake: Optional[tuple] = None  # (woke, words, score) of the first wake-phrase check
        self.path: Optional[str] = None  # how it was handled: the Brain method, or key / wake_only ...
        self.words: Optional[str] = None  # the words the Brain was given
        self.intent = None  # the Intent the engine passed to the Brain, if it parsed it itself
        self.reply: Optional[Response] = None  # what the Brain returned
        self.reply_ms: Optional[float] = None
        self.said: list[Response] = []  # replies Buddy started saying for this input
        self.in_brain = False
        self.barge = False  # "Hey Buddy" had just been heard while Buddy talked: next words taken
        self.echo = False  # (talking, no wake phrase) mostly Buddy's own words
        self.ended = False  # the input has been handled
        self.delivered = False  # the playback outcome is known (or there is nothing to play)
        self.written = False


class Recorder:
    """Builds the records and appends them to the day's file (background thread)."""

    def __init__(self, folder, source: str, near_miss: float = 0.7):
        self.dir = Path(os.path.expanduser(str(folder)))
        self.source = source
        self.near_miss = float(near_miss)
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)  # this program run
        self._ids = itertools.count(1)
        self.conversation = 0
        self.count_conversations = True  # chat / simulate; the voice engine counts them itself
        self.events_from_engine = False  # voice: the engine records Buddy's lines when they are played
        self.brain = None
        self.turn: Optional[Entry] = None  # the input being handled right now
        self.pending: list[Entry] = []  # built, not yet written
        self.timeout_question: Optional[str] = None
        self.timeout_reply: Optional[Response] = None
        self.closed = False
        self._warned: set[str] = set()
        self._q: "queue.Queue[Optional[tuple[str, str]]]" = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, name="buddy-records", daemon=True)
        self._writer.start()
        self._handler = _Problems(self)
        logging.getLogger("buddy").addHandler(self._handler)
        atexit.register(self.close, "program exit")

    # ------------------------------------------------------------------ watching the brain
    def attach(self, brain) -> "Recorder":
        """Observe which Brain method handles each input (with which words, and its reply) and the
        wake-phrase check.  Arguments and results pass through unchanged."""
        if self.brain is brain:
            return self
        self.brain = brain
        for name in ("command", "follow_up", "answer", "timeout"):
            setattr(brain, name, self._watch_brain(name, getattr(brain, name)))
        brain.wake.split = self._watch_wake(brain.wake.split)
        brain.wake.find = self._watch_wake(brain.wake.find)
        return self

    def _watch_brain(self, name: str, method: Callable) -> Callable:
        def watched(*args, **kwargs):
            e = self.turn
            outer = e is not None and not e.in_brain  # (follow_up and answer call command themselves)
            if outer:
                e.in_brain = True
                self._safe(self._brain_called, e, name, args, kwargs)
            elif e is None and name == "timeout":
                self._safe(self._timeout_called)
            try:
                result = method(*args, **kwargs)
            finally:
                if outer:
                    e.in_brain = False
            if outer:
                self._safe(self._brain_returned, e, result)
            elif e is None and name == "timeout":
                self._safe(self._timeout_returned, result)
            return result

        watched.__wrapped__ = method
        return watched

    def _watch_wake(self, method: Callable) -> Callable:
        def watched(text, *args, **kwargs):
            result = method(text, *args, **kwargs)
            e = self.turn
            if e is not None and e.wake is None:
                e.wake = result
            return result

        watched.__wrapped__ = method
        return watched

    def _brain_called(self, e: Entry, name: str, args: tuple, kwargs: dict) -> None:
        e.path = e.path or name
        e.words = args[0] if args else kwargs.get("text")
        e.intent = args[1] if len(args) > 1 else kwargs.get("intent")
        exp = self.brain.expect if self.brain is not None else None
        if name == "answer" and exp is not None and exp.prompt:
            e.data["question"] = _transcript(exp.prompt)

    def _brain_returned(self, e: Entry, result) -> None:
        e.reply = result
        e.reply_ms = (time.perf_counter() - e.started) * 1000

    def _timeout_called(self) -> None:
        exp = self.brain.expect
        self.timeout_question = _transcript(exp.prompt) if exp is not None and exp.prompt else None

    def _timeout_returned(self, r) -> None:
        self.timeout_reply = r
        if not self.events_from_engine and isinstance(r, Response) and r.items:  # chat: nobody plays it
            self.finish(self.event("answer_timeout", r, question=self.timeout_question), "printed")

    # ------------------------------------------------------------------ building records
    def begin(self, text: str, state: str, via: str, heard_seconds: Optional[float] = None,
              asr_ms: Optional[float] = None, file: Optional[str] = None) -> Entry:
        """An utterance (or typed line, or key) arrived while Buddy was asleep / listening /
        answering / talking."""
        if self.turn is not None and not self.turn.ended:
            self.end(self.turn)
        e = self._new("interaction", state=state, input=text, input_via=via)
        if heard_seconds:
            e.data["input_seconds"] = round(float(heard_seconds), 2)
        if file:
            e.data["file"] = Path(file).name  # the WAV's name only, never its folder
        if asr_ms is not None:
            e.data["timing_ms"] = {"asr": round(float(asr_ms))}
        self.turn = e
        return e

    def chat_line(self, text: str, state: str) -> None:
        """chat mode: a typed line ('' = silence).  Finishes the previous line's record."""
        if self.turn is not None and not self.turn.ended:
            self.end(self.turn)
        if text:
            self.begin(text, state, "keyboard")

    def end(self, e: Optional[Entry], note: Optional[str] = None, playing: bool = False) -> None:
        """The input has been handled.  Keep it (written now, or when Buddy has finished saying the
        reply if `playing`) or drop it: talk that was not meant for Buddy is never stored."""
        if e is None or e.ended:
            return
        if self.turn is e:
            self.turn = None
        e.ended = True
        verdict = self._verdict(e, note)
        if verdict is None:
            self._drop(e)
            return
        status, why = verdict
        d = e.data
        self._fill(e)
        if self.count_conversations and d["state"] == "asleep" and status == "ok" and e.path:
            self.conversation += 1  # chat / simulate: Buddy woke up
        d["conversation"] = self.conversation if (status != "near_miss" and self.conversation) else None
        d["status"] = status
        if why:
            d["note"] = why
        if not playing and not e.delivered:
            e.delivered = True
            d["delivery"] = "none" if self.source == "voice" else "printed"
        self._maybe_write(e)

    def event(self, name: str, response: Optional[Response] = None, **fields) -> Entry:
        """Something Buddy did without a request (greeting, prompt, timeout, mute, start / end)."""
        e = self._new("event", event=name, conversation=self.conversation or None)
        e.data.update({k: v for k, v in fields.items() if v is not None})
        if response is not None:
            e.data["response"] = response.transcript()
            e.data["response_intent"] = response.intent or None
            e.data["after"] = response.after
        e.data.setdefault("status", "ok")
        e.ended = True
        return e

    def event_now(self, name: str, text: Optional[str] = None, **fields) -> None:
        """An event that is complete right away (text = what Buddy printed or said, if anything)."""
        e = self.event(name, **fields)
        if text is not None:
            e.data["response"] = text
        self.finish(e, None if text is None else ("printed" if self.source != "voice" else "complete"))

    def finish(self, e: Optional[Entry], delivery: Optional[str], response: Optional[Response] = None,
               at: Optional[int] = None) -> None:
        """How the reply was delivered: complete | cut_off | none | printed (None = not applicable).
        For cut_off, `at` is the index of the item that was playing (None = unknown)."""
        if e is None or e.delivered:
            return
        e.delivered = True
        if delivery is not None:
            e.data["delivery"] = delivery
        if delivery == "cut_off" and response is not None:
            e.data["cut_off_at"] = _cut_at(response, at)
        self._maybe_write(e)

    def fail(self, e: Optional[Entry], exc: BaseException) -> None:
        """An exception escaped while this input was handled: keep the details (it is re-raised)."""
        if e is None or e.written or getattr(exc, "_buddy_recorded", False):
            return
        _mark(exc)
        if self.turn is e:
            self.turn = None
        e.ended = e.delivered = True
        try:
            self._fill(e)
        except Exception:  # noqa: BLE001  (the error may come from the same step; keep the record anyway)
            pass
        e.data["conversation"] = self.conversation or None
        e.data["status"] = "error"
        e.data["error"] = _error(exc)
        self._maybe_write(e)

    def crash(self, exc: BaseException) -> None:
        """Buddy stopped with an error: attach it to what was going on, or record it on its own."""
        if getattr(exc, "_buddy_recorded", False):
            return
        if self.turn is not None and not self.turn.ended:
            return self.fail(self.turn, exc)
        _mark(exc)
        playing = next((p for p in reversed(self.pending) if p.ended and not p.delivered), None)
        if playing is not None:
            playing.data["status"] = "error"
            playing.data["error"] = _error(exc)
            self.finish(playing, "cut_off")
        else:
            self.finish(self.event("crash", status="error", error=_error(exc)), None)

    def problem(self, message: str) -> None:
        """A warning or error logged by Buddy (e.g. a missing speech clip): add it to the record of
        the input being handled, or of the reply being played; else record it on its own."""
        target = self.turn
        if target is None:
            target = next((p for p in reversed(list(self.pending)) if p.ended and not p.delivered), None)
        if target is not None:
            target.data.setdefault("problems", []).append(message)
        else:
            self.finish(self.event("warning", note=message), None)

    def close(self, reason: Optional[str] = None) -> None:
        """Write what is still open (a reply still playing counts as cut off) and stop the writer."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.turn is not None and not self.turn.ended:
                self.end(self.turn)
            for e in list(self.pending):
                if e.ended and not e.delivered:
                    e.data.setdefault("note", "the program ended while Buddy was saying this")
                    self.finish(e, "cut_off")
            self.finish(self.event("program_end", note=reason), None)
        finally:
            logging.getLogger("buddy").removeHandler(self._handler)
            self._q.put(None)
            self._writer.join(timeout=5)

    def flush(self) -> None:
        """Wait until everything recorded so far is in the file (used by the tests)."""
        self._q.join()

    # ------------------------------------------------------------------ decisions
    def _verdict(self, e: Entry, note: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
        """(status, note) of a handled input, or None when it must not be kept."""
        d = e.data
        state = d["state"]
        woke, words, score = e.wake if e.wake else (False, "", None)
        if e.path is None and woke and not words:  # just "Hey Buddy"
            e.path = "wake_only"
            if note is None and self.source != "voice":
                note = "Buddy played its listening chime"
        if e.path is not None:
            if e.reply is None and not e.said and e.path not in ("key", "wake_only"):
                return "ignored", note or "not taken as a request (in the listening window only requests are answered)"
            return "ok", note
        if e.said or e.reply is not None:
            return "ok", note
        if woke:
            return "ignored", note or "the wake phrase was heard, but nothing was done"
        if state in ("listening", "answering") or e.barge:
            return "ignored", note or "not taken as a request"
        # asleep, or Buddy was talking and no "Hey Buddy" was heard
        why = self._near_miss(e, score)
        if why:
            return "near_miss", why
        if state == "talking" and not e.echo:
            return "ignored", note or NO_WAKE_WHILE_TALKING
        return None

    def _near_miss(self, e: Entry, score: Optional[float]) -> Optional[str]:
        """Why this speech looks like a failed "Hey Buddy", or None (other talk: not kept)."""
        if score is None or score < self.near_miss or self.brain is None:
            return None
        w, m = self.brain.wake, self.brain.matcher
        if e.data["state"] == "talking":  # (while talking the wake phrase must be clearer)
            return "sounded like 'Hey Buddy', but not clearly enough while Buddy was talking" \
                if score >= w.threshold else None
        if score >= w.threshold:
            return "sounded like 'Hey Buddy', but what followed was not understood as a request"
        toks = normalize(e.data["input"])
        for k in (1, 2, 3):
            rest = " ".join(toks[k:])
            if rest and m.parse(rest).name not in NOT_A_REQUEST:
                e.words = rest
                return "almost 'Hey Buddy', followed by a request"
        return None

    def _fill(self, e: Entry) -> None:
        """Wake check, understanding, Buddy's reply and timings of an input."""
        d = e.data
        if e.wake is not None:
            woke, _words, score = e.wake
            d["wake"] = {"heard": bool(woke), "score": round(float(score), 2)}
        if e.path:
            d["path"] = "barge_in" if (e.path == "command" and d["state"] == "talking") else e.path
        understood = self._understood(e)
        if understood:
            d["understood"] = understood
        if e.said:
            d["response"] = " ".join(t for t in (r.transcript() for r in e.said) if t)
        elif e.reply is not None:
            d["response"] = e.reply.transcript()
        else:
            d["response"] = None
        last = e.said[-1] if e.said else e.reply
        if last is not None:
            d["response_intent"] = last.intent or None
            d["after"] = last.after
        if e.reply is not None and e.reply.action:
            d["action"] = e.reply.action
        if e.reply_ms is not None:
            d.setdefault("timing_ms", {})["reply"] = round(e.reply_ms, 1)

    def _understood(self, e: Entry) -> Optional[dict]:
        """What the language-understanding step makes of the words (same matcher, same result)."""
        if self.brain is None:
            return None
        words = e.words
        if words is None and e.data["state"] != "asleep":
            woke, rest, _ = e.wake if e.wake else (False, "", None)
            words = rest if woke else e.data.get("input")
        if not words:
            return None
        i = e.intent if e.intent is not None else self.brain.matcher.parse(words)
        out = {"intent": i.name, "score": round(float(i.score), 2), "confidence": round(float(i.confidence), 2),
               "words": " ".join(i.tokens)}
        if i.slots:
            out["slots"] = i.slots
        return out

    # ------------------------------------------------------------------ writing
    def _new(self, kind: str, **fields) -> Entry:
        now = datetime.now().astimezone()
        data = {"id": f"{self.run_id}-{next(self._ids):04d}", "time": now.isoformat(timespec="milliseconds"),
                "source": self.source, "kind": kind}
        data.update(fields)
        e = Entry(data, now.strftime("%Y-%m-%d"))
        self.pending.append(e)
        return e

    def _drop(self, e: Entry) -> None:
        e.written = True
        if e in self.pending:
            self.pending.remove(e)

    def _maybe_write(self, e: Entry) -> None:
        if e.written or not (e.ended and e.delivered):
            return
        e.written = True
        if e in self.pending:
            self.pending.remove(e)
        d = e.data
        if d.get("status") == "ok" and d.get("problems"):
            d["status"] = "incomplete"
        try:
            line = json.dumps(_ordered(d), ensure_ascii=False, default=_plain)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("json", f"interaction records: could not encode a record: {exc!r}")
            return
        self._q.put((e.day, line))

    def _write_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                day, line = item
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.dir / f"{day}_{self.source}.jsonl", "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())  # survives a power cut right after
            except Exception as exc:  # noqa: BLE001
                self._warn_once("write", f"interaction records: could not write to {self.dir}: {exc}")
            finally:
                self._q.task_done()

    def _safe(self, fn: Callable, *args, **kwargs):
        """Run one recording step.  A problem here is logged once and never reaches Buddy."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            name = getattr(fn, "__name__", "step")
            self._warn_once(name, f"interaction records: {name} failed ({exc!r}); Buddy is not affected")
            return None

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message)


class _Problems(logging.Handler):
    """Copies Buddy's own warnings and errors (e.g. "missing speech clip") into the records."""

    def __init__(self, recorder: Recorder):
        super().__init__(logging.WARNING)
        self.recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(log.name):  # the recorder's own messages
            return
        try:
            self.recorder.problem(_scrub(f"{record.levelname}: {record.getMessage()}"))
        except Exception:  # noqa: BLE001
            pass


class _TimedAsr:
    """The speech recogniser, timed for timing_ms.asr.  Results pass through unchanged."""

    def __init__(self, asr):
        self._asr = asr
        self._ms: Optional[float] = None

    def transcribe(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._asr.transcribe(*args, **kwargs)
        finally:
            self._ms = (time.perf_counter() - t0) * 1000

    def take_ms(self) -> Optional[float]:
        ms, self._ms = self._ms, None
        return ms

    def __getattr__(self, name):
        return getattr(self._asr, name)


class RecordingEngine(Engine):
    """The voice loop of engine.py, plus records.  Each override runs the original Engine method
    unchanged; the added lines only look at what happened."""

    def __init__(self, *args, recorder: Recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec = recorder.attach(self.brain)
        self.rec.events_from_engine = True
        self.rec.count_conversations = False  # _wake() below counts them, exactly like Engine does
        self.asr = _TimedAsr(self.asr)
        self._playing: Optional[tuple[Response, Entry]] = None  # reply in the player and its record
        self._spot: Optional[tuple[Response, int]] = None  # (reply, item) being heard when the input came

    # ------------------------------------------------------------------ input
    def on_text(self, text: str, heard_seconds: float = 0.0) -> None:
        if not text:
            return super().on_text(text, heard_seconds)
        e = self.rec._safe(self._begin, text, heard_seconds)
        try:
            super().on_text(text, heard_seconds)
        except BaseException as exc:
            self.rec._safe(self.rec.fail, e, exc)
            raise
        self.rec._safe(self._end, e)

    def _key(self, k: str) -> None:
        e = self.rec._safe(self._begin, f"[key {k}]", 0.0, "key")
        try:
            super()._key(k)
        except BaseException as exc:
            self.rec._safe(self.rec.fail, e, exc)
            raise
        self.rec._safe(self._end, e)

    def _do_mute(self) -> None:
        self.rec._safe(self._remember_spot)
        super()._do_mute()
        self.rec._safe(self._check_cut)

    def _wake(self, now: float) -> None:
        if now >= self.awake_until:  # the test Engine._wake uses for "a new conversation"
            self.rec.conversation += 1
        super()._wake(now)

    # ------------------------------------------------------------------ output
    def _say(self, r: Response, start_index: int = 0) -> None:
        self.rec._safe(self._starting, r)
        super()._say(r, start_index)

    def _finished(self, r: Response) -> None:
        self.rec._safe(self._done, r)
        super()._finished(r)

    def run(self) -> None:
        reason = "quit"
        try:
            super().run()
        except KeyboardInterrupt:
            reason = "stopped (Ctrl+C)"
            raise
        except BaseException as exc:
            reason = "crashed"
            self.rec._safe(self.rec.crash, exc)
            raise
        finally:
            self.rec._safe(self.rec.close, reason)

    # ------------------------------------------------------------------ helpers (observe only)
    def _state(self, now: float) -> str:
        if self.player.active:
            return "talking"
        if now < self.awake_until:
            return "answering" if self.brain.expecting else "listening"
        return "asleep"

    def _remember_spot(self) -> None:
        p = self.player
        self._spot = (p.response, p.item_at(0.0)) if p.active else None

    def _begin(self, text: str, heard_seconds: float, path: Optional[str] = None) -> Entry:
        now = self.clock()
        self._remember_spot()
        typed = not heard_seconds
        e = self.rec.begin(text, self._state(now), "keyboard" if typed else "microphone",
                           heard_seconds=heard_seconds, asr_ms=None if typed else self.asr.take_ms())
        e.path = path
        e.barge = now < self.barge_until
        return e

    def _end(self, e: Optional[Entry]) -> None:
        self._check_cut()
        if e is None:
            return
        note = None
        if e.data["state"] == "talking" and e.path is None and not e.said:
            woke, words, _ = e.wake if e.wake else (False, "", None)
            if woke and not words:
                note = DUCKED
            elif woke and self._is_echo(words):
                note = ECHO
            elif woke or e.barge:
                note = UNCLEAR
            else:
                e.echo = self._sounds_like_buddy(e.data["input"])
        self.rec.end(e, note, playing=self._playing is not None and self._playing[1] is e)

    def _sounds_like_buddy(self, text: str) -> bool:
        r = self._spot[0] if self._spot else None
        if r is None:
            return False
        said = set(normalize(" ".join(r.texts())))
        toks = normalize(text)
        return len(toks) >= 3 and sum(t in said for t in toks) / len(toks) >= ECHO_SHARE

    def _starting(self, r: Response) -> None:
        """Buddy is about to say r: whatever was playing before is cut off by it."""
        if self._playing is not None and self._playing[0] is not r:
            old, entry = self._playing
            at = self.player.item_at(0.0) if self.player.response is old else self._spot_of(old)
            self.rec.finish(entry, "cut_off", old, at)
        self._playing = None
        turn = self.rec.turn
        if turn is not None:
            turn.said.append(r)
            self._playing = (r, turn)
            return
        name = "answer_timeout" if r is self.rec.timeout_reply else EVENTS.get(r.intent, r.intent or "buddy_spoke")
        if name == "follow_up_prompt" and not r.texts():
            return  # only the soft "ready" chime after a task
        extra = {"question": self.rec.timeout_question} if name == "answer_timeout" else {}
        self._playing = (r, self.rec.event(name, r, **extra))

    def _done(self, r: Response) -> None:
        if self._playing is not None and self._playing[0] is r:
            _, entry = self._playing
            self._playing = None
            self.rec.finish(entry, "complete")

    def _check_cut(self) -> None:
        """The reply that was playing was stopped (stop, quit, pause, mute) without a new one."""
        if self._playing is not None and self.player.response is not self._playing[0]:
            old, entry = self._playing
            self._playing = None
            self.rec.finish(entry, "cut_off", old, self._spot_of(old))

    def _spot_of(self, r: Response) -> Optional[int]:
        return self._spot[1] if self._spot is not None and self._spot[0] is r else None


def settings(cfg) -> dict:
    """Settings that change how Buddy hears and answers (nothing personal)."""
    t = cfg["tts"]
    return {"buddy_version": __version__, "age_setting": cfg.child_age, "asr_model": cfg["asr"]["model"],
            "tts": f"{t['engine']} voice {t['voice']} speed {t['speed']}",
            "wake_phrases": list(cfg["wake"]["phrases"]), "wake_threshold": cfg["wake"]["threshold"],
            "follow_up_seconds": cfg["session"]["follow_up_seconds"],
            "near_miss_score": cfg["records"].get("near_miss_score", 0.7)}


def open_records(cfg, source: str) -> Optional[Recorder]:
    """The recorder for this run, or None when [records] enabled = false."""
    r = cfg["records"]
    if not r.get("enabled"):
        return None
    rec = Recorder(r.get("folder") or Path.home() / "buddy-data" / "records", source,
                   float(r.get("near_miss_score", 0.7)))
    rec.event_now("program_start", settings=settings(cfg))
    print(f"Recording interactions in {rec.dir} (temporary analysis feature - remove before the final "
          f"release, see docs/RECORDS.md)")
    return rec


# ---------------------------------------------------------------------- small helpers
def _transcript(items) -> str:
    return Response(list(items)).transcript()


def _cut_at(r: Response, at: Optional[int]) -> dict:
    """Where a reply was cut off, in sentences."""
    total = len(r.texts())
    if at is None:
        return {"total_sentences": total}
    out = {"heard_sentences": sum(isinstance(i, Say) for i in r.items[:at]), "total_sentences": total}
    if 0 <= at < len(r.items) and isinstance(r.items[at], Say):
        out["stopped_during"] = r.items[at].text
    return out


def _error(exc: BaseException) -> dict:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {"type": type(exc).__name__, "message": _scrub(str(exc)), "traceback": _scrub(tb)[-6000:]}


def _scrub(text: str) -> str:
    """Hide the user's home folder (it contains their user name) in error texts and warnings."""
    home = str(Path.home())
    return text.replace(home, "~").replace(home.replace("\\", "/"), "~") if len(home) > 1 else text


def _mark(exc: BaseException) -> None:
    try:
        exc._buddy_recorded = True  # recorded once, even though it passes through several levels
    except Exception:  # noqa: BLE001
        pass


def _ordered(d: dict) -> dict:
    rank = {k: i for i, k in enumerate(FIELDS)}
    return {k: d[k] for k in sorted(d, key=lambda k: rank.get(k, len(FIELDS)))}


def _plain(o: Any) -> Any:
    """JSON for values that are not plain data (Intent objects in slots, numpy numbers, ...)."""
    if is_dataclass(o) and not isinstance(o, type):
        try:
            return asdict(o)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return repr(o)
