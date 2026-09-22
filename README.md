# Buddy: an offline voice learning companion for children (3–8 years)

An Alexa-style assistant built for small children. It runs **100 % offline** on a **Raspberry Pi 5**,
uses **only open-source models**, and keeps **no recordings**.

```
"Hey Buddy, what does the cow say?"
  mic → Silero VAD → Moonshine speech-to-text → wake phrase + child-speech-tolerant intent matching
      → skill (stories, poems, alphabet, phonics, counting, animals, quizzes …) → pre-rendered voice → speaker
```

- **Setup & build guide:** [docs/SETUP.md](docs/SETUP.md)
- **Things to do yourself on the Pi:** [docs/MANUAL_STEPS.md](docs/MANUAL_STEPS.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Open-source components

| Part | Model / library | Licence |
|---|---|---|
| Runtime | sherpa-onnx | Apache-2.0 |
| Speech-to-text | Moonshine v2 base/tiny, English | MIT |
| Voice activity detection | Silero VAD | MIT |
| Voice | Kokoro-82M (int8) | Apache-2.0 |
| Alternative voice | Piper `en_US-ljspeech-medium` | MIT model, public-domain dataset |
| Buddy code and content | this repository | project owner's |

Kokoro and Piper use the espeak-ng phonemiser (GPL-3.0) inside sherpa-onnx. Buddy uses text-to-speech mainly
**at setup time**, to pre-render its fixed set of sentences. Have your patent attorney review how this affects distribution.

## Quick start (on the Pi)

```bash
bash scripts/install_pi.sh
python -m buddy build-content     # one time
python -m buddy run
```

## Project layout

```
buddy/
  nlu/phonetics.py   age-scheduled child-phonology phonetic distance   (core invention 1)
  nlu/matcher.py     closed-vocabulary intent + slot recognition
  speech/wake.py     transcript-level wake phrase detection with command validation
  speech/cache.py    pre-synthesised closed-domain speech inventory
  learning/          adaptive quiz: Leitner boxes + per-topic level        (core invention 2)
  brain.py           dialogue manager (answers without wake word, repeat, parental rules)
  engine.py          real-time loop, barge-in, echo guard, mute             (core invention 3: privacy)
  skills/handlers.py all activities
  content/*.json     stories, poems, animals, lessons, jokes, FAQ, phrases (editable)
tests/               unit + end-to-end tests
scripts/             installer, model download, systemd service, audio evaluation
```

## Measured results (build workspace, 2 vCPU x86 VM; synthetic child-like test audio)

| Test | Result |
|---|---|
| Wake phrase detected in commands | 36 / 36 |
| False wake-ups on 19 everyday sentences (with "Buddy the dog…", "Hey Bobby", "my best buddy") | 0 |
| Clean commands, end to end (audio → intent) | 20 / 20 |
| Child-pronunciation commands ("pway a towy", "wion sound", "tount to ten" …) | **62 %** with age-scheduled matching vs 38 % uniform phonetic vs 25 % exact keywords |
| Speech-to-text time | about 110 ms for a 1.4 s command (Moonshine base) |
| Unit tests (99 clean + 16 child-variant text commands) | 100 % |

The child-like audio was synthesised (higher pitch, mispronounced spellings, background noise).
**Re-measure with real children's recordings, with parental consent, before filing.**
