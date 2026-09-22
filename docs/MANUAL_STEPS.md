# Manual steps (things to do yourself)

The workspace used to build this project could not reach some websites, and some tasks need a person.
Those items were skipped there. **None of them need an API key or an account.**

## A. On the Raspberry Pi (one time, with internet)

| # | What | Why it was skipped | What to do |
|---|---|---|---|
| 1 | Python packages `sherpa-onnx`, `numpy`, `sounddevice` | pypi.org was blocked in the build workspace (`403 Host not in allowlist`). The code was tested with sherpa-onnx built from source instead. | `bash scripts/install_pi.sh` does it (`pip install -r requirements.txt`) |
| 2 | System audio + GPIO libraries | not needed in the workspace | done by `install_pi.sh` (`libportaudio2`, `python3-gpiozero`, `python3-lgpio`) |
| 3 | Speech models (~400 MB) | these downloaded fine; they just need downloading again on the Pi | done by `install_pi.sh` → `scripts/download_models.sh` |
| 4 | Pre-render the voice | takes 2–3 h with Kokoro on the Pi | `python -m buddy build-content` (overnight), or use `engine = "piper"` (~15 min) |
| 5 | Real hardware test with mic, speaker and button | the workspace has no audio hardware (tests used simulated mic/speaker) | follow docs/SETUP.md §7 and §10 |

## B. Content you may want to add

| # | What | Why | What to do |
|---|---|---|---|
| 6 | Classic nursery rhymes (e.g. traditional public-domain rhymes) | Writing out famous lyrics word for word was blocked by the AI's content filter, so Buddy ships with 14 **original** poems instead | paste public-domain rhymes into `buddy/content/rhymes_user.json` (format shown inside), then run `python -m buddy build-content` |
| 7 | Real animal sounds | need a person to pick recordings with a free licence (CC0 / public domain) | save as `cow.wav`, `dog.wav`, … (names = animal ids in `animals.json`), set `[paths] sounds = "~/buddy-data/sounds"` |
| 8 | Check the phonics clips | TTS voices sometimes swallow sounds like "buh". In testing, "B says buh, buh, ball" came back from speech recognition as "B says ball". | listen to `Hey Buddy, phonics`. If a sound is unclear, change its spelling in `lessons.json` (e.g. "buh" → "buh!") and rebuild. |

## C. Before filing the patent

| # | What | Why |
|---|---|---|
| 9 | Record 20–50 real children (3–8 years) saying commands, with written parental consent | the accuracy figures so far come from **synthetic** child-like audio; real data makes the patent's technical-effect claims credible |
| 10 | Run `python scripts/evaluate_audio.py labels.json` on those recordings | produces the exact / uniform / child-phonology comparison table |
| 11 | Patent attorney: freedom-to-operate check against **US10885918B2 (Microsoft)**, plus a licence review (espeak-ng GPL-3.0 inside sherpa-onnx) | see the patent disclosure draft |
