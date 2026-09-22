"""Evaluate the full audio pipeline (VAD -> Moonshine ASR -> wake -> intent) on labelled WAV files.

Compares three text-matching strategies to measure the effect of age-scheduled
child-phonology matching:
  exact     - keywords must be recognised exactly (typical rule-based assistant)
  uniform   - phonetic fuzzy matching with fixed, age-independent costs
  child     - Buddy's age-scheduled child-phonology costs (default)

usage: python scripts/evaluate_audio.py /path/labels.json [--asr moonshine-base] [--age 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from buddy.audio.dsp import read_wav, resample  # noqa: E402
from buddy.config import load_config  # noqa: E402
from buddy.content_store import Content  # noqa: E402
from buddy.nlu.matcher import Matcher  # noqa: E402
from buddy.speech.models import Recognizer, Vad  # noqa: E402
from buddy.speech.wake import WakeMatcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--asr", default="moonshine-base")
ap.add_argument("--age", type=float, default=4)
args = ap.parse_args()
cfg = load_config()
content = Content()
labels = json.load(open(args.labels))
asr = Recognizer(cfg.models_dir, args.asr, 2)
v = cfg["vad"]

# transcribe once
rows = []
for lab in labels:
    x, sr = read_wav(lab["file"])
    x = resample(x, sr, 16000)
    vad = Vad(cfg.models_dir, v["threshold"], v["min_silence"], v["min_speech"], v["max_utterance"])
    segs = vad.accept(np.concatenate([x, np.zeros(8000, np.float32)])) + vad.flush()
    seg = np.concatenate(segs) if segs else x
    t0 = time.time()
    text = asr.transcribe(seg)
    ms = (time.time() - t0) * 1000
    rows.append((lab, text, ms, len(seg) / 16000))

strategies = {
    "exact": Matcher(content, age=99, word_threshold=1.01),
    "uniform": Matcher(content, age=99),
    "child": Matcher(content, age=args.age),
}
print(f"ASR model: {args.asr}   files: {len(rows)}   mean ASR time: {np.mean([r[2] for r in rows]):.0f} ms "
      f"for {np.mean([r[3] for r in rows]):.1f} s of speech")
res = {k: [0, 0, 0, 0] for k in strategies}  # clean_ok, clean_n, child_ok, child_n
wake_ok = 0
for i, (lab, text, ms, dur) in enumerate(rows):
    is_child = i >= 20
    line = f"{Path(lab['file']).name}: {lab['text']!r:42} heard {text!r:42}"
    for name, m in strategies.items():
        w = WakeMatcher(cfg["wake"]["phrases"], m.phon if name != "exact" else Matcher(content, age=99).phon,
                        cfg["wake"]["threshold"])
        woke, rest, _ = w.split(text)
        intent = m.parse(rest) if woke else None
        got = intent.name if intent else "(no wake)"
        ok = got == lab["intent"] or (lab["intent"] == "poem" and got == "song")
        if ok and lab.get("animal"):  # the slot must be right too
            ok = intent.slots.get("animal") == lab["animal"]
            got += f"({intent.slots.get('animal')})"
        res[name][2 if is_child else 0] += ok
        res[name][3 if is_child else 1] += 1
        if name == "child":
            wake_ok += woke
            line += f" -> {got}{'' if ok else '  <-- MISS'}"
    print(line)
print(f"\nwake phrase detected: {wake_ok}/{len(rows)}")
print(f"{'strategy':10s} {'clean commands':>16s} {'child pronunciations':>22s}")
for k, (co, cn, ko, kn) in res.items():
    print(f"{k:10s} {co:>7d}/{cn:<3d} {co / cn:5.0%} {ko:>11d}/{kn:<3d} {ko / kn:5.0%}")
