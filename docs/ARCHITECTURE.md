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
       [Conversation session] after "Hey Buddy" and after every answer: 15 s listening window,
             │                no wake phrase needed; background talk ignored; silence → sleep
             ▼
       [Dialogue manager] ◄── expecting an answer? (quiz/riddle: no wake phrase needed for 9 s)
             │  parental gate (quiet hours, daily limit)
             │  controls: stop / pause / continue / repeat / quit / louder / softer
             ▼
       [Intent matcher]  1) snap words to the closed vocabulary with an AGE-SCHEDULED
                            child-phonology edit distance (fronting, gliding, stopping …);
                            pass 1 keeps everyday English words, pass 2 snaps everything
                         2) scored intent candidates → best intent (ties: fixed priority)
                         3) shared slot extraction (nlu/slots.py): numbers, ranges, steps,
                            direction, sums; titles, topics, animal, letter
             ▼
       [Skills] stories · poems · alphabet · phonics · counting · animals · colours · shapes ·
                body · days/months · opposites · tables · sums · quiz · riddles · jokes ·
                feelings · breathing · manners · FAQ · curiosity reply
             │                      ▲
             │           [Adaptive learning store: SQLite, item ids + right/wrong only,
             │            Leitner boxes, per-topic level 1–5 kept at ~75 % success]
             ▼
       [Player] ──► plays the reply sentence by sentence from the pre-synthesised speech cache
                    (every sentence rendered once at setup with Kokoro/Piper TTS) + chimes
                    + optional CC0 sounds ──► USB speaker.  Knows which sentence is playing:
                    pause saves it, continue resumes from the start of that sentence.
                                                                                        │
  Mute button (GPIO17) ──► closes mic stream, LED (GPIO27) on          barge-in: "Hey Buddy, stop"
                 └────────────────────────────────────────────────────────────────────────────┘
```

## Conversation, interruptions and controls

| State | What Buddy listens for |
|---|---|
| asleep | only "Hey Buddy …" |
| listening (15 s after "Hey Buddy" and after every answer; `[session] follow_up_seconds`) | any request, no wake phrase. Speech that is not a request is ignored. Silence → short goodbye line → asleep |
| talking | only "Hey Buddy, pause / stop / repeat / quit / louder / softer", or a clear new request ("Hey Buddy, tell me a joke": the story is kept paused for "continue"). The mic also hears Buddy, so the wake phrase is required. The wake phrase may come after Buddy's own words in the same segment; "Hey Buddy" + a pause makes Buddy talk softer for 4 s and wait for the command. An echo guard ignores Buddy saying "Hey Buddy …" itself |
| answering (quiz/riddle, 9 s) | the answer; "repeat" says the question again; "wait" gives more time; a clear new request ("tell me a story") ends the game |

- **Pause** stores the reply and the sentence the child was hearing when they *started* to speak (the recogniser
  needs a moment, so Buddy looks back by the length of the utterance). **Continue** plays from the start of that
  sentence. The paused reply is kept until a new long activity starts or 3 minutes pass (`pause_memory_seconds`).
  Muting also pauses a story.
- **Repeat** replays the last *meaningful* reply as it was said (the same story, not a new random one). Control
  replies ("Okay!", volume, prompts) never replace it. During a quiz it repeats the current question.
- **Quit / bye / good night** end the conversation (Buddy goes back to sleep). The program keeps running, so a child
  cannot switch the toy off by voice. Keyboard `q` still exits the program on a PC.

## How a request is understood

```
"can you count from one through ten?"
  normalise ──► can you count from one through ten          ("1-10" → "1 dash 10", "5s" → "fives")
  pass 1    ──► everyday words kept, only non-words snapped ("tount" → count, "through" stays)
  intents   ──► count 0.93 (keyword)   … (every detector that fires adds a scored candidate)
  slots     ──► start 1, end 10         (nlu/slots.py: to / till / through / "-" / between … and …)
  skill     ──► One. Two. … Ten.
```

If pass 1 finds nothing or only a weak guess, pass 2 snaps every word (the original behaviour), so
children's pronunciations are still understood. A bare range ("… from one to ten") is enough to count,
even if "count" itself was misheard. Limits: counting 0–100 (steps 1–10, forwards or backwards),
sums and take-aways up to 20, times tables 1–12. `nlu/fallback.py` is an empty hook where an offline
intent classifier could be added later; it may only choose among Buddy's intents, never write speech.

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
