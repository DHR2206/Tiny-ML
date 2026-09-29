# Buddy: an offline voice learning companion for children (3–8 years)

An Alexa-style assistant built for small children. It runs **100 % offline** on a **Raspberry Pi 5**,
uses **only open-source models**, and keeps **no recordings**.

```
"Hey Buddy, what does the cow say?"  …  "Now count from 5 to 15!"  (no "Hey Buddy" needed while Buddy listens)
  mic → Silero VAD → Moonshine speech-to-text → wake phrase / conversation window
      → child-speech-tolerant intent matching + parameter extraction (numbers, ranges, topics …)
      → skill (stories, poems, alphabet, phonics, counting, animals, quizzes …)
      → pre-rendered voice, played sentence by sentence (pause / continue / repeat) → speaker
```

**Talking to Buddy:** say "Hey Buddy" once. After each answer a soft chime means Buddy is still listening for
15 seconds, so the child just says the next thing. While Buddy talks: "Hey Buddy, pause / stop / repeat / quit".
"Continue" resumes a paused story from the interrupted sentence (up to 3 minutes later).

- **Try it on your own PC first (Windows / Mac / Linux):** [docs/RUN_ON_PC.md](docs/RUN_ON_PC.md)
- **Setup & build guide (Raspberry Pi):** [docs/SETUP.md](docs/SETUP.md)
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
  nlu/matcher.py     closed-vocabulary intent recognition: two-pass word snapping + scored intents
  nlu/slots.py       shared parameter extraction: numbers, ranges, steps, direction, sums
  nlu/common_words.py everyday words that are never snapped onto Buddy's keywords
  nlu/fallback.py    optional hook for a future intent classifier (none configured)
  speech/wake.py     transcript-level wake phrase detection with command validation
  speech/cache.py    pre-synthesised closed-domain speech inventory
  learning/          adaptive quiz: Leitner boxes + per-topic level        (core invention 2)
  brain.py           dialogue manager (follow-up requests, quiz answers, repeat, parental rules)
  playback.py        sentence-by-sentence player (pause / continue / repeat position)
  engine.py          real-time loop, conversation session, barge-in, echo guard, mute (core invention 3: privacy)
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
| Unit tests (99 clean + 16 child-variant + 85 natural-variation text commands) | 100 % |
| Conversation-flow tests (follow-up, pause/continue, repeat, quit, barge-in; fake audio devices) | 23 / 23 |

The child-like audio was synthesised (higher pitch, mispronounced spellings, background noise).
**Re-measure with real children's recordings, with parental consent, before filing.**
