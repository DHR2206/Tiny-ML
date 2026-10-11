# Interaction records (temporary)

> **Temporary, for analysis while testing.** Remove this feature before the final release.
> The steps are under [Removing the feature](#removing-the-feature). Until it is removed, Buddy prints a reminder
> every time it starts.

While you test, Buddy writes down every interaction:

- what it heard;
- what it understood;
- what it said back.

Later you can give these files to an AI assistant, or read them yourself, to find:

- misunderstood commands;
- wrong or incomplete answers;
- missed requests;
- problems that keep coming back.

## Where the records are

| How you run Buddy | File |
|---|---|
| `python -m buddy run` (microphone) | `~/buddy-data/records/YYYY-MM-DD_voice.jsonl` |
| `python -m buddy chat` (typing) | `~/buddy-data/records/YYYY-MM-DD_chat.jsonl` |
| `python -m buddy simulate …` (WAV files) | `~/buddy-data/records/YYYY-MM-DD_simulate.jsonl` |

On Windows, `~` is your user folder, for example `C:\Users\<you>\buddy-data\records\2026-10-10_voice.jsonl`.

- There is one file per day for each of these modes.
- New records are always added at the end. Nothing is overwritten, and the files stay after Buddy restarts.
- The folder is outside the project folder, so records can never end up on GitHub.

Each line is one complete JSON object (the JSON Lines format). Open a file in any text editor, or load it in
Python with `pandas.read_json(path, lines=True)`.

## What is recorded, and what is not

**Recorded:**

- everything said after "Hey Buddy" and during a conversation, **including speech Buddy ignored**. This covers
  the listening window after an answer, quiz answers, and speech while Buddy is talking;
- **near misses**: speech that sounded like "Hey Buddy" but did not wake Buddy. These show the times "nothing
  happened". While Buddy is asleep, a sentence is kept only if:
  - it scored at least `near_miss_score` against the wake phrase and a request followed it ("hey bobby, tell me
    a joke"); or
  - it scored above the wake threshold, but what followed was not a request;
- Buddy's exact reply, and whether it was heard to the end or cut off (and at which sentence);
- Buddy's own lines that answer no request: the greeting, "what next?" prompts, quiz time-outs, the
  end-of-conversation line, and mute / unmute;
- the program's start (with the settings that affect understanding) and its end;
- errors, with their traceback;
- warnings, such as a sentence missing from the speech cache.

**Never recorded:**

- audio. Buddy still never writes sound to disk;
- talk while Buddy is asleep that does not sound like "Hey Buddy";
- Buddy's own voice picked up by the microphone (echo);
- names, user or device identifiers, and folder paths. In error texts your home folder is replaced by `~`, and a
  simulated WAV keeps only its file name.

The child's words are stored exactly as recognised, which is the point of the records. So if a child tells Buddy
their name, the name will be in the record. Delete the records folder when you no longer need it.

## Record format

An example record, shown formatted. In the file each record is a single line:

```json
{
  "id": "20261010-143201-7f3a-0003",
  "time": "2026-10-10T14:32:07.123+05:30",
  "source": "voice",
  "kind": "interaction",
  "conversation": 1,
  "state": "listening",
  "input": "count from one to ten",
  "input_via": "microphone",
  "input_seconds": 1.6,
  "wake": {"heard": false, "score": 0.41},
  "path": "follow_up",
  "understood": {"intent": "count", "score": 0.93, "confidence": 1.0,
                 "words": "count from one to ten", "slots": {"start": 1, "end": 10}},
  "response": "Let's count together! One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Wow, we did it! Amazing counting!",
  "response_intent": "count",
  "after": "task",
  "delivery": "complete",
  "status": "ok",
  "timing_ms": {"asr": 140, "reply": 6.2}
}
```

| Field | Meaning |
|---|---|
| `id` | Unique id: `<program start time>-<random>-<number>`. |
| `time` | When the input arrived (for an event: when it happened). Local time, with the UTC offset. |
| `source` | `voice`, `chat` or `simulate`. |
| `kind` | `interaction` (the child said or typed something) or `event` (Buddy did something by itself). |
| `event` | Events only: `program_start`, `greeting`, `follow_up_prompt`, `answer_timeout`, `conversation_end`, `muted`, `unmuted`, `warning`, `crash`, `program_end`. |
| `conversation` | Number of the conversation in this run, from waking up to going back to sleep. `null` means none. |
| `state` | Buddy's state when the speech arrived: `asleep`, `listening` (after "Hey Buddy" or after an answer), `answering` (a quiz question or riddle was waiting), or `talking`. |
| `input` | **Exactly** what the speech recogniser returned, or what was typed. Keyboard keys appear as `[key p]`. |
| `input_via` | `microphone`, `keyboard` (typed commands and keys) or `wav_file`. |
| `input_seconds` | Length of the spoken utterance. |
| `file` | Simulate only: the WAV file's name. |
| `wake` | `heard`: the wake phrase was detected. `score`: how close the start sounded to a wake phrase, from 0 to 1 (a score of `wake_threshold` or more wakes Buddy). |
| `path` | How Buddy handled it: `command` (with the wake phrase, or just after it), `follow_up` (in the listening window, no wake phrase), `answer` (answer to a question), `barge_in` (while Buddy talked), `wake_only` (just "Hey Buddy"), or `key`. Missing means not handled. |
| `question` | For `answer` and `answer_timeout`: the question Buddy had asked. |
| `understood` | What the language-understanding step made of the words. See the list below the table. |
| `response` | **Exactly** what Buddy said, all sentences. `[ready]`, `[listen]`, `[correct]` and the like are chimes; `[sound:cow]` is a recorded sound. `null` means nothing. |
| `response_intent` | The activity of the reply: `joke`, `count`, `pause` and so on. |
| `action` | A control action: `pause`, `continue`, `stop`, `quit`, `repeat`, `more`, `volume_up` or `volume_down`. |
| `after` | What Buddy did next: `task` (kept listening, with a chime), `silent` (kept listening), `wake` (short listening window) or `sleep` (conversation over). |
| `delivery` | `complete` (heard to the end), `cut_off` (stopped early: by pause, stop, a new request, mute or the program ending), `none` (nothing to say) or `printed` (chat and simulate). |
| `cut_off_at` | For `cut_off`: `heard_sentences`, `total_sentences` and `stopped_during` (the sentence playing at that moment). |
| `status` | `ok`, `ignored` (heard during a conversation but not taken as a request), `near_miss` (see above), `incomplete` (a problem while answering, see `problems`) or `error`. |
| `note` | Why something was ignored or counted as a near miss, and other explanations. |
| `problems` | Warnings logged while Buddy handled or said it, for example `WARNING: missing speech clip …`. |
| `error` | `type`, `message` and `traceback` of an error. Buddy then stops, exactly as it would without records. |
| `timing_ms` | `asr`: speech-to-text time. `reply`: time to decide the answer. |
| `settings` | `program_start` only: version, age setting, models, wake phrases and threshold, listening window, and near-miss score. |

The `understood` field contains:

- `intent` and its `score` (0 to 1);
- `confidence`: how exactly the words were heard;
- `words`: the words after Buddy's sound-alike correction;
- `slots`: numbers, names, topics and so on.

For `answer` records, `understood` shows how the words would read as a command. Buddy uses this to notice a new
request during a quiz.

A record is written when its interaction is complete, that is, after Buddy's reply has finished or was cut off.
So the file is almost, but not always, in time order. Sort by `time` or `id` when the exact order matters.

## Switching it off or changing it

In `buddy.toml`:

```toml
[records]
enabled = false            # stop recording
# folder = "D:/buddy-records"
# near_miss_score = 0.7    # higher = fewer near misses kept
```

`[privacy] log_transcripts` is a separate setting: it only shows what Buddy heard in the console window.

## Using the records later

The analysis is not part of Buddy. When you want it, give one or more `.jsonl` files to an AI assistant together
with this page, which explains every field. Useful things to look at:

- records whose `status` is not `ok`;
- `understood.intent` = `unknown`;
- a reply that was `cut_off` after its first sentence (the child stopped a wrong answer);
- the same request repeated within a few seconds;
- `near_miss` records (the wake phrase was missed).

## Removing the feature

Before the final release:

1. Delete `buddy/records.py`, `tests/test_records.py` and this file (`docs/RECORDS.md`).
2. Remove everything marked `RECORDS (temporary)`. To find it, run `git grep -n "RECORDS (temporary)"`.
   - `buddy/__main__.py`: delete the `_records()` function and the marked lines in `cmd_chat`, `cmd_run` and
     `cmd_simulate`. In `cmd_run`, keep only `engine = Engine(cfg, brain, mic, spk, asr, vad, cache)`.
   - `buddy/config.py`: delete the `"records"` section.
   - `buddy.example.toml`: delete the `[records]` section.
   - `README.md` and `docs/RUN_ON_PC.md`: delete the notes about records.
3. Run the tests: `python -m unittest discover -s tests -t .`
4. Delete the records folder (`~/buddy-data/records`), or move it somewhere safe, away from the product.

Nothing else in Buddy depends on the records, so removing them does not change how Buddy works.
