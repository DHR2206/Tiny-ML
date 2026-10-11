"""Command line:  python -m buddy <command>

  run             start the voice assistant (microphone + speaker)
  chat            type to Buddy (no audio) - great for testing
  build-content   pre-render every sentence to the speech cache (run once after install)
  simulate WAV..  run WAV files through VAD -> ASR -> Buddy (no microphone needed)
  doctor          check models, audio devices and cache
  progress        show the child's learning levels
  reset-progress  wipe the learning database
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config


def _cache(cfg, with_synth: bool):
    from .speech.cache import SpeechCache

    t = cfg["tts"]
    synth = None
    if with_synth:
        from .speech.models import Synthesizer

        s = Synthesizer(cfg.models_dir, t["engine"], int(t["threads"]))
        synth = lambda text: s.synth(text, int(t["voice"]), float(t["speed"]))  # noqa: E731
    vkey = f"{t['engine']}-{t['voice']}-{t['speed']}"
    return SpeechCache(cfg.cache_dir, vkey, int(cfg["audio"]["output_rate"]), synth)


def _records(cfg, source: str, brain=None):  # RECORDS (temporary): remove before the final release
    """Interaction records for analysis (docs/RECORDS.md), or None when [records] enabled = false."""
    if not cfg["records"].get("enabled"):
        return None
    from .records import open_records

    rec = open_records(cfg, source)
    return rec.attach(brain) if brain is not None else rec


def cmd_chat(cfg, args) -> None:
    """Type as the child.  Works like the voice mode's conversation: after "Hey Buddy", Buddy
    keeps listening, so the next requests need no wake phrase.  An empty line = silence."""
    from .brain import Brain

    b = Brain(cfg)
    rec = _records(cfg, "chat", b)  # RECORDS (temporary)
    print("Type what the child says, e.g. 'hey buddy tell me a story', then 'tell me a joke' (no 'hey buddy' needed "
          "while Buddy is listening).")
    print("Empty line = silence (ends a quiz question, then the conversation). Ctrl+D (Mac/Linux) or Ctrl+Z Enter "
          "(Windows) to quit.")
    awake, window = False, None
    for line in sys.stdin:
        text = line.strip()
        if rec:  # RECORDS (temporary)
            rec.chat_line(text, "answering" if b.expecting and awake else "listening" if awake else "asleep")
        if not text:
            if b.expecting:
                r = b.timeout()
                print("  (silence) Buddy:", r.transcript())
            elif awake:
                awake = False
                bye = b.skills.p("sleep")
                print("  (silence) Buddy:", bye, "[Buddy goes to sleep]")
                if rec:  # RECORDS (temporary)
                    rec.event_now("conversation_end", text=bye)
            continue
        woke, rest, _ = b.wake.split(text)
        if b.expecting and not woke and awake:
            r = b.answer(text)
        elif woke and rest:
            r = b.command(rest)
        elif woke:
            awake, window = True, "wake"
            print("  Buddy: [listen chime]")
            continue
        elif awake:
            r = b.command(text) if window == "wake" else b.follow_up(text)
            if r is None:
                print("  (Buddy ignores this: not a request)")
                continue
        else:
            print("  (Buddy is asleep: start with 'Hey Buddy')")
            continue
        awake, window = r.after != "sleep", "follow"
        if r.action in ("pause", "continue"):
            print(f"  Buddy [{r.action}]: (pause and continue act on Buddy's speech in voice mode: python -m buddy run)")
            continue
        print(f"  Buddy [{r.intent}{' / ' + r.action if r.action else ''}]:", r.transcript())
        if not awake:
            print("  [Buddy goes to sleep - say 'Hey Buddy' to wake it]")
    if rec:  # RECORDS (temporary)
        rec.close("end of typed input")


def cmd_build(cfg, args) -> None:
    from .brain import Brain

    b = Brain(cfg)
    phrases = sorted(b.skills.all_phrases(), key=len)
    cache = _cache(cfg, with_synth=True)
    todo = cache.missing(phrases)
    print(f"{len(phrases)} phrases, {len(todo)} to render with {cfg['tts']['engine']} ...")
    t0 = time.time()
    for i, text in enumerate(todo, 1):
        cache.render(text)
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  elapsed {el / 60:.1f} min, about {eta / 60:.1f} min left", flush=True)
    print("Speech cache ready:", cache.dir)


def cmd_run(cfg, args) -> None:
    from .audio.io import Microphone, Speaker
    from .brain import Brain
    from .engine import Engine
    from .hardware.button import MuteButton
    from .speech.models import Recognizer, Vad

    a, v = cfg["audio"], cfg["vad"]
    brain = Brain(cfg)
    cache = _cache(cfg, with_synth=bool(cfg["tts"]["runtime_fallback"]))
    missing = cache.missing(list(brain.skills.all_phrases())[:50])
    if missing and not cfg["tts"]["runtime_fallback"]:
        print("Speech cache is empty. Run: python -m buddy build-content")
    asr = Recognizer(cfg.models_dir, cfg["asr"]["model"], int(cfg["asr"]["threads"]))
    vad = Vad(cfg.models_dir, v["threshold"], v["min_silence"], v["min_speech"], v["max_utterance"])
    mic = Microphone(a["input_device"], a["input_rate"])
    spk = Speaker(a["output_device"], int(a["output_rate"]), float(a["volume"]))
    rec = _records(cfg, "voice")  # RECORDS (temporary): the same engine, plus records
    if rec:
        from .records import RecordingEngine

        engine = RecordingEngine(cfg, brain, mic, spk, asr, vad, cache, recorder=rec)
    else:
        engine = Engine(cfg, brain, mic, spk, asr, vad, cache)
    h = cfg["hardware"]
    engine.button = MuteButton(h.get("mute_button_gpio"), h.get("led_gpio"), engine.toggle_mute)
    if h.get("keyboard") and sys.stdin.isatty():
        engine.keyboard()
        print("Keys: m=mute/unmute  t=talk  p=pause/continue  r=repeat  x=end conversation  q=quit  "
              "(or type a command)")
    try:
        engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        engine.button.close()


def cmd_simulate(cfg, args) -> None:
    import numpy as np

    from .audio.dsp import read_wav, resample
    from .brain import Brain
    from .speech.models import Recognizer, Vad

    b = Brain(cfg)
    rec = _records(cfg, "simulate", b)  # RECORDS (temporary)
    asr = Recognizer(cfg.models_dir, cfg["asr"]["model"], int(cfg["asr"]["threads"]))
    v = cfg["vad"]
    for f in args.files:
        x, sr = read_wav(f)
        x = resample(x, sr, 16000)
        vad = Vad(cfg.models_dir, v["threshold"], v["min_silence"], v["min_speech"], v["max_utterance"])
        segs = vad.accept(np.concatenate([x, np.zeros(16000, dtype=np.float32)])) + vad.flush()
        for seg in segs:
            t0 = time.time()
            text = asr.transcribe(seg)
            asr_ms = (time.time() - t0) * 1000
            if rec:  # RECORDS (temporary)
                rec.begin(text, "answering" if b.expecting else "asleep", "wav_file", len(seg) / 16000, asr_ms, f)
            woke, rest, score = b.wake.split(text)
            if b.expecting and not woke:
                r = b.answer(text)
            elif woke and rest:
                r = b.command(rest)
            else:
                r = None
            print(f"{f}: heard {text!r} ({len(seg) / 16000:.1f}s audio, ASR {asr_ms:.0f} ms) wake={woke}({score:.2f})")
            if r:
                print(f"   -> [{r.intent}] {r.transcript()[:150]}")
            if rec:  # RECORDS (temporary)
                rec.end(rec.turn)
    if rec:  # RECORDS (temporary)
        rec.close("all files done")


def cmd_doctor(cfg, args) -> None:
    from .speech.models import MODEL_DIRS

    ok = True
    print("Models folder:", cfg.models_dir)
    for k, d in MODEL_DIRS.items():
        p = cfg.models_dir / d
        need = k in ("vad", cfg["asr"]["model"], cfg["tts"]["engine"])
        mark = "ok " if p.exists() else ("MISSING" if need else "optional (not downloaded)")
        ok &= p.exists() or not need
        print(f"  {k:15s} {mark}  {d}")
    try:
        import sherpa_onnx  # noqa: F401
        print("sherpa-onnx: ok")
    except ImportError:
        ok = False
        print("sherpa-onnx: MISSING (pip install sherpa-onnx)")
    try:
        import sounddevice as sd
        print("Audio devices:\n", sd.query_devices())
    except Exception as e:  # noqa: BLE001
        ok = False
        print("sounddevice/PortAudio problem:", e, "(sudo apt install libportaudio2)")
    from .brain import Brain

    cache = _cache(cfg, with_synth=False)
    phrases = Brain(cfg).skills.all_phrases()
    miss = cache.missing(phrases)
    print(f"Speech cache: {len(phrases) - len(miss)}/{len(phrases)} phrases rendered ({cache.dir})")
    print("All good!" if ok and not miss else "Some items need attention (see above).")


def cmd_progress(cfg, args) -> None:
    from .learning.progress import Progress

    p = Progress(cfg.data_dir / "progress.db")
    rows = p.summary()
    if not rows:
        print("No quiz answers recorded yet.")
    for d, lvl, c, w in rows:
        print(f"{d:10s} level {lvl}/5   correct {c}   wrong {w}")
    print(f"Time with Buddy today: {p.usage_today() / 60:.1f} min")
    if args.reset:
        p.reset()
        print("Progress wiped.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="buddy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="path to buddy.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "chat", "build-content", "doctor"):
        sub.add_parser(name)
    s = sub.add_parser("simulate")
    s.add_argument("files", nargs="+")
    sub.add_parser("progress").set_defaults(reset=False)
    sub.add_parser("reset-progress").set_defaults(reset=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    {"run": cmd_run, "chat": cmd_chat, "build-content": cmd_build, "simulate": cmd_simulate,
     "doctor": cmd_doctor, "progress": cmd_progress, "reset-progress": cmd_progress}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
