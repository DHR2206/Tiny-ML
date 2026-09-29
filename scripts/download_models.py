"""Cross-platform model downloader (Windows, macOS, Linux, Raspberry Pi).

Same models as download_models.sh, but needs only Python, so it also works on Windows.
Usage:   python scripts/download_models.py            (all models, ~400 MB)
         python scripts/download_models.py --small    (Moonshine tiny + Piper only, ~150 MB, fastest to try)
"""
import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
DEST = Path(os.environ.get("BUDDY_HOME", Path.home() / "buddy-data")) / "models"

FULL = [
    ("asr-models", "sherpa-onnx-moonshine-base-en-quantized-2026-02-27", True),
    ("asr-models", "silero_vad.onnx", False),
    ("tts-models", "kokoro-int8-en-v0_19", True),
    ("tts-models", "vits-piper-en_US-ljspeech-medium", True),
]
SMALL = [
    ("asr-models", "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27", True),
    ("asr-models", "silero_vad.onnx", False),
    ("tts-models", "vits-piper-en_US-ljspeech-medium", True),
]


def progress(block, size, total):
    if total > 0:
        done = min(100, block * size * 100 // total)
        sys.stdout.write(f"\r    {done:3d}% of {total / 1e6:.0f} MB")
        sys.stdout.flush()


def fetch(tag: str, name: str, is_archive: bool) -> None:
    target = DEST / name
    if target.exists():
        print(f"ok   {name}")
        return
    fname = f"{name}.tar.bz2" if is_archive else name
    url = f"{BASE}/{tag}/{fname}"
    tmp = DEST / (fname + ".part")
    print(f"get  {name}")
    urllib.request.urlretrieve(url, tmp, progress)
    print()
    if is_archive:
        with tarfile.open(tmp, "r:bz2") as tf:
            tf.extractall(DEST)
        tmp.unlink()
    else:
        tmp.rename(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="smaller/faster models for a quick try")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for tag, name, arch in (SMALL if args.small else FULL):
        fetch(tag, name, arch)
    print(f"Models ready in {DEST}")


if __name__ == "__main__":
    main()
