# Buddy architecture

```
                 ┌──────────────── Raspberry Pi 5 (all on-device, no network) ────────────────┐
 USB mic ──► [Capture 16 kHz, RAM ring buffer, wiped after use] ──► [Silero VAD] ──► speech segment
                                                                                        │
                                                                     [Moonshine ASR (sherpa-onnx)]
                                                                                        │ text
                                        ┌──────────── [Wake matcher: "Hey Buddy …" in phoneme space,
                                        │              weak matches must look like a real command]
                                        ▼
       [Dialogue manager] ◄── expecting an answer? (quiz/riddle: no wake phrase needed for 9 s)
             │  parental gate (quiet hours, daily limit)      controls: stop / louder / softer / repeat
             ▼
       [Intent matcher]  1) snap words to the closed vocabulary with an AGE-SCHEDULED
                            child-phonology edit distance (fronting, gliding, stopping …)
                         2) prioritised rules → intent + slots (animal, number, letter, title…)
             ▼
       [Skills] stories · poems · alphabet · phonics · counting · animals · colours · shapes ·
                body · days/months · opposites · tables · sums · quiz · riddles · jokes ·
                feelings · breathing · manners · FAQ · curiosity reply
             │                      ▲
             │           [Adaptive learning store: SQLite, item ids + right/wrong only,
             │            Leitner boxes, per-topic level 1–5 kept at ~75 % success]
             ▼
       [Renderer] ──► pre-synthesised speech cache (every sentence rendered once at setup
                      with Kokoro/Piper TTS) + chimes + optional CC0 sounds ──► USB speaker
                                                                                        │
  Mute button (GPIO17) ──► closes mic stream, LED (GPIO27) on          barge-in: "Hey Buddy, stop"
                 └────────────────────────────────────────────────────────────────────────────┘
```

## Why these choices

| Decision | Reason |
|---|---|
| **Transcribe, then match the wake phrase** (no trained wake-word network) | Freely usable wake-word models carry non-commercial licences. The recogniser is already needed, and children can say "Hey Buddy, play a story" in one breath. Weak matches ("Buddy the dog…") are accepted only if the rest is a real command, which gave 0 false wake-ups in testing. |
| **Closed vocabulary + phonetic snapping** instead of an LLM | Predictable, safe answers for 3-year-olds. Uses under 5 ms of CPU. Fully offline. No hallucinations. |
| **Age-scheduled costs** | Children's pronunciation errors are systematic and depend on age (ASHA norms). Matching tolerates exactly the errors expected at the configured age. |
| **Pre-rendered speech** | The whole inventory (~2,450 sentences) is known in advance, so it's synthesised once. At runtime there's no TTS latency and no neural TTS load, so the CPU stays free for listening. |
| **Moonshine base** | MIT licence, English. About 110 ms for a 1.4 s command on 2 x86 cores. Fast on a Pi 5. |
| **sherpa-onnx** | One Apache-2.0 runtime for VAD, ASR and TTS, with aarch64 wheels for the Pi. |

## Memory and CPU budget (Pi 5, 8 GB)

| Component | RAM (approx.) |
|---|---|
| Moonshine base (quantised) + runtime | ~250 MB |
| Silero VAD | ~10 MB |
| Python app + phrase index | ~120 MB |
| Kokoro TTS (only if runtime fallback is used) | ~350 MB |
| **Total** | **under 1 GB**, leaving plenty of headroom on an 8 GB Pi |

CPU: VAD runs continuously (under 5 % of one core). ASR runs only on speech segments.
