# v0.4 — fixes for the v0.3 QA findings

The QA run of 13 October 2026 (`~/buddy-data/qa/Buddy_v0.3_QA_report.xlsx`) recorded 19 findings
over 393 exchanges. This file explains what was changed and why, and records the measurements the
decisions rest on. The post-fix report is `Buddy_v0.4_QA_report_postfix.xlsx`.

Nothing was changed in the original test suite except one skip condition (see M1); the original
report, its probe script and its transcripts are untouched.

## What each finding turned into

| Finding | Root cause | Changed |
|---|---|---|
| **H1** crash on "another one" after an animal fact | `brain._run` drops `id`/`animal`/`letter` when it repeats a task, and six handlers read their slot unguarded | `skills/handlers.py`: every handler now does the same activity with a fresh item when its parameter is missing |
| **H2** crashes recorded as "ignored" | `Recorder.crash()` was wired only into `RecordingEngine.run()`, so an exception in chat/simulate escaped unrecorded | `__main__.py`: `cmd_chat` and `cmd_simulate` hand the exception to the recorder, then re-raise |
| **H3** voice stack not installable here | environment, not code | not a code change; see the report |
| **M1** 4 unit-test errors on Windows | `Progress` held one sqlite connection open for its whole life, so Windows refused to delete the file; `test_engine_audio`'s skip guard checked for model *files* but not for the package that loads them | `learning/progress.py` opens the file per operation and closes it again; `tests/test_engine_audio.py` guard also requires `sherpa_onnx` (no assertion changed) |
| **M2** "in steps of N" ignored, step read as the range end | `count_slots` only understood "by N"/"in Ns", so "steps" was not a step and its number became the end of the range | `nlu/slots.py`: `STEP_NOUNS` ("steps of ten", "jumps of two"); `matcher.py` keeps "step"/"steps" from being snapped |
| **M3** word-form big numbers collapsed | `parse_numbers` handled "hundred" but not "thousand"; a leading "minus" was read as a range joiner | `nlu/normalize.py` handles thousands; `nlu/slots.py` reads "minus five" as a sign; `do_count` clamps a negative start to zero |
| **M4** out-of-scope requests got a confident wrong activity | four separate leaks, below | `nlu/matcher.py`, `nlu/common_words.py` |
| **M5** body parts reachable only as "body parts" | the `body` intent fired only on the word "body"; the part names in `lessons.json` were not matched at all | `matcher.py` matches the lesson's own part names plus singular and nursery forms; `do_body` answers about one part, the way `do_shapes` already did |
| **M6** a lisped word outranked a word that was said | "teww" resembles "tale", so story and poem both scored 0.9 and the fixed priority order picked story | `matcher.py`: story/poem/song are weighted by how clearly their keyword was heard |
| **M7** two everyday sentences woke Buddy | the wake validator asks the matcher whether the rest is a command, and the matcher was fooled by the same loose snapping ("coming"→"counting", "played"→a poem title) | fixed by the M4 work; no change to `speech/wake.py` |
| **M8** a 10,000-character token took 4.8 s to parse | the token was compared character by character against every title and FAQ question | `nlu/normalize.py` caps a token at 32 characters, which bounds every later step |
| **L1** `daily_limit_minutes = -5` locked the child out | `if limit and usage >= limit*60` is true at zero usage for a negative limit | `brain.gate()` requires `limit > 0`; `config.py` corrects a negative value and says so |
| **L2** raw traceback for bad TOML; a missing `--config` silently ignored | no error handling around `tomllib.load`, and a path that does not exist just fell through to the defaults | `config.py` raises `ConfigError` with a readable message; `__main__.py` prints it |
| **L3** "louder"/"softer" printed an empty reply | the acknowledgement was produced inside the audio engine, so every other surface got nothing | `handlers.py` puts it in the reply; the engine still replaces it with the "as loud as I can go" wording at the cap |
| **L4** version reported 0.1.0 | stale string, stamped into every record | `buddy/__init__.py` → 0.3.0 |
| **L5** quiz hints are per domain, not per item | content authoring | not fixed; see the report |
| **L6** non-ASCII crashed the Windows console | cp1252 stdout | `__main__.py` sets `errors="backslashreplace"` on stdout/stderr |
| **J10** an unknown `[nlu] fallback` stopped Buddy starting | `load_fallback` raised out of `Brain.__init__` | warns and carries on with no fallback; `config.py` corrects the value |

Two further crashes of the same shape as H1 were found by the new tests and fixed:
`do_shapes` and `do_phonics` raised `StopIteration` for an item that is not in the lesson.

## M4 in detail: four leaks, not one threshold

M4 ("out-of-scope requests get a confident wrong activity instead of the curiosity reply") and M7
(false wake-ups) are the same bug seen twice, because the wake validator asks the matcher whether
the words after "buddy" look like a command.

The obvious fix — require a higher phonetic similarity before a word may become a keyword — was
tried first and **rejected on measurement**. Real child pronunciations of a keyword and words
that simply are not in Buddy's world occupy the same range:

```
must snap (tests/test_nlu.py, QA probe)      must not snap (QA probe section D)
  stowy  -> story    0.810                     blah   -> blue     0.838
  cowour -> colour   0.762                     call   -> calm     0.838
  tolours-> colours  0.750                     dabra  -> cobra    0.800
  animaw -> animal   0.742                     ears   -> horses   0.800
  towy   -> story    0.740                     abra   -> cobra    0.770
```

A bar anywhere in that range costs as many good commands as it saves. So the leaks were fixed
one mechanism at a time instead:

1. **The second pass threw away the first pass's protection.** `common_words.py` deliberately
   keeps ordinary English as it is, but when nothing was understood the second pass snapped
   everything, including those words. That is what turned "call mummy" into a breathing exercise
   and "show me my ears" into a horse. The second pass may now only overrule an everyday word on
   a strong match (`VOCAB_NEED`, 0.90). Swept over 0.80…never: **every value gives identical
   results on all 353 cases**, i.e. nothing Buddy understands relies on the loose rewrite.

2. **Gaps in `common_words.py`.** "order", "set", "routine", "prime", "minister", "france",
   "capital", "weather", "twinkle", "drop", "alert" were absent, so they were snapped in the
   *first* pass ("order"→"horses", "minister"→"months"). Added, along with the body-part words.
   This is the list's documented purpose and its intended extension point.

3. **A bare animal name was trusted even when guessed.** "just say monkey" is a supported
   shortcut, but it accepted a name the snapper had invented, which is why "order a pizza"
   produced a horse and "abra ka dabra" a snake. It now requires the name to have been heard as
   it is; "wion sound" still works, because that goes through the sound-cue branch.

4. **An invented keyword could win with nothing to support it.** A guess is still trusted when it
   accounts for the whole request ("tolours" is plainly "colours"), but not when part of the
   utterance made no sense either *and* the keyword was not heard clearly — "florp bimble wug"
   used to recite the days of the week.

## What is left, and the knob for it

`[nlu] min_confidence` was declared in `config.py` and **never read by anything**. It now works:
it is how clearly the words of a request must have been heard before Buddy will start an activity
on them. It is left at **0 (off)**, which is the behaviour Buddy has always had, because the
useful value depends on `[child] age` — the phoneme distance is age-scheduled — and there is no
single setting that is right across 3 to 8. Measured over 19 mispronounced commands and 16
nonsense utterances:

```
value   age 3-4                        age 5 (default)                age 8
0.00    3 nonsense get an activity     2 get an activity              0
0.70    2 nonsense                     2 nonsense                     8 commands lost
0.74    1 nonsense                     2 nonsense, 1 command lost     10 commands lost
0.78    1 nonsense, 6 commands lost    13 commands lost               16 commands lost
```

With it off, nine short mumbles out of 84 (14 utterances × 6 ages) still reach an activity;
`tests/test_fixes.py::TestAcrossTheAgeRange` lists them exactly, so the limit stays visible and
any new leak fails the test. Closing them properly is a model question, not a threshold: the
`nlu/fallback.py` hook exists for exactly this, and the choice is the owner's.

## Where the gibberish fallback appears

One wording, from `content/phrases.json` (`not_understood`, then `not_understood_help` on the
second miss in a row) — already the project's style, now actually reached:

* after the wake phrase, and in the short window just after a bare "Hey Buddy", **any**
  unintelligible input gets the fallback, including empty, whitespace, punctuation-only and
  non-Latin text, which used to produce *silence* (`do_none` returned an empty reply);
* in the 15 s follow-up window after a task, speech that is not a request is still **ignored**,
  which is the v0.2 "background talk" feature and is deliberately unchanged. If Buddy should
  instead answer "I didn't catch that" there too, that is a one-line change and a product
  decision — it is the one open question in the post-fix report.
