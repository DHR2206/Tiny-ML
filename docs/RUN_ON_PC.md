# Run Buddy on your own PC (no Raspberry Pi needed)

Buddy is ordinary Python, so it runs the same way on a Windows, Mac or Linux computer. Everything works
except the physical mute button: on a PC you press **`m` + Enter** instead.
Your laptop's built-in microphone and speakers are fine. Headphones work best, because Buddy can't hear itself through them.

**What you need:** a PC with 8 GB RAM, about 2 GB of free disk space, and internet for the one-time setup.

---

## Step 1: Install Python 3.11 or 3.12

Buddy needs **Python 3.11 or newer**. 3.11 or 3.12 are the safest choices, because the speech library has ready-made installers for them.

**Windows**
1. Go to https://www.python.org/downloads/windows/ and download **Python 3.12.x (64-bit)**.
2. Run the installer and **tick "Add python.exe to PATH"** at the bottom of the first screen, then click *Install Now*.
3. Open **PowerShell** (Start menu → type `PowerShell`) and check:
   ```powershell
   py --version
   ```
   It should print `Python 3.12.x`.

**macOS**: install from https://www.python.org/downloads/macos/ and check with `python3 --version`.

**Ubuntu / Linux**: `sudo apt install python3 python3-venv python3-pip libportaudio2`, then `python3 --version`.

## Step 2: Unzip the project

1. Unzip `buddy_prototype.zip`, for example to `C:\Users\<you>\buddy` (Windows) or `~/buddy` (Mac/Linux).
   The folder must directly contain `requirements.txt`, `buddy.example.toml` and the `buddy`, `scripts`, `docs` and `tests` folders.
   If unzipping created `buddy\buddy\buddy`, use the outer `buddy` folder that holds `requirements.txt`.
2. Open a terminal **in that folder**:
   - Windows: open the folder in File Explorer, click the address bar, type `powershell` and press Enter.
   - Mac/Linux: `cd ~/buddy`

## Step 3: Create a private Python environment

This keeps Buddy's libraries separate from the rest of your computer.

| Windows (PowerShell) | Mac / Linux |
|---|---|
| `py -3.12 -m venv .venv` | `python3 -m venv .venv` |
| `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

After activating, the prompt starts with `(.venv)`. **Do this activation step every time you open a new terminal.**

On Windows, if you see *"running scripts is disabled on this system"*, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Step 4: Install the libraries (one command)

```bash
pip install -r requirements.txt
```

This installs `sherpa-onnx` (the speech engine), `numpy` and `sounddevice` (microphone and speaker access).
On Windows and Mac, `sounddevice` brings its own audio driver, so nothing else is needed.

## Step 5: Download the speech models

For a **quick first try** (~150 MB, fastest):
```bash
python scripts/download_models.py --small
```
For the **full quality** set (~400 MB: better recogniser plus the natural Kokoro voice):
```bash
python scripts/download_models.py
```
The models go into `buddy-data\models` inside your user folder (`C:\Users\<you>\buddy-data\models`).
You can run both commands; the second only adds what's missing.

## Step 6: Create your settings file

Copy the example settings into the project folder as `buddy.toml`:

| Windows | Mac / Linux |
|---|---|
| `copy buddy.example.toml buddy.toml` | `cp buddy.example.toml buddy.toml` |

If you downloaded the **small** models in Step 5, open `buddy.toml` in Notepad (or any editor) and change these two lines:
```toml
model = "moonshine-tiny"   # in the [asr] section
engine = "piper"           # in the [tts] section
```
Also set `mute_button_gpio = "none"` and `led_gpio = "none"` in `[hardware]`. This is optional, but it hides a harmless warning, since a PC has no GPIO pins.

## Step 7: Test the "brain" by typing (no microphone yet)

```bash
python -m buddy chat
```
Type lines like these and press Enter after each. Only the first line needs "hey buddy": after that Buddy is
listening, just like in voice mode. An empty line means silence (Buddy goes back to sleep).
```
hey buddy tell me a story
what does the cow say
count from 5 to 15
animal quiz
moo
repeat
tell me a story about a lion
why is the sky blue
quit
```
Buddy prints what it would say. Press **Ctrl+Z then Enter** (Windows) or **Ctrl+D** (Mac/Linux) to quit.

## Step 8: Check the microphone and speakers

```bash
python -m buddy doctor
```
- Each model should show `ok`.
- Under **Audio devices**, the `>` marks your default microphone and `<` your default speaker.
  To use a different one, add part of its name to `buddy.toml` under `[audio]`, e.g. `input_device = "Headset"`.
- "Speech cache: 0/3054" is normal at this point (see Step 9).

**Windows only:** allow microphone access under *Settings → Privacy & security → Microphone →*
"Let desktop apps access your microphone" = **On**.
**Mac only:** the first time, macOS asks whether Terminal may use the microphone. Click **Allow**.

## Step 9: Pre-render Buddy's voice (recommended, one time)

```bash
python -m buddy build-content
```
This speaks all ~3,050 sentences once and saves them, so Buddy answers instantly afterwards.
On a typical laptop it takes about **5–20 minutes with Piper** or **35–110 minutes with Kokoro**.
You can stop it with Ctrl+C and run it again later; it continues where it stopped.
**After updating Buddy**, run it again: it only renders the new sentences (about 600 in this update).

*Want to try right away?* You can skip this step. Buddy then creates each sentence the first time it needs it,
so the first answers take a second or two longer.

## Step 10: Talk to Buddy

```bash
python -m buddy run
```
Wait for the chime and "Hello, my friend!", then speak clearly from about half a metre.
You only need "Hey Buddy" to start: after each answer a soft chime means Buddy is still listening
(for 15 seconds), so just say the next thing.

- "**Hey Buddy, tell me a story.**" … when it ends: "**Tell me a joke.**" … "**Another one.**"
- "**What does the lion say?**" · "**Count from 5 to 15.**" · "**What's 12 plus 5?**"
- "**Let's play a quiz.**" Then answer the questions. "**Repeat**" says the question again.
- While Buddy is talking, start with the wake phrase: "**Hey Buddy, pause.**" Later: "**Continue.**"
  (Buddy continues from the sentence you interrupted, up to 3 minutes later.)
  "**Hey Buddy, stop!**", "**Hey Buddy, repeat**" and "**Hey Buddy, quit**" work the same way.
- "**Quit**" or "**Bye**" ends the conversation; Buddy then waits for "Hey Buddy" again.

Keyboard helpers in the same window (type, then press Enter):
`m` = mute/unmute (like the Pi's button), `t` = talk (acts like "Hey Buddy"), `p` = pause/continue,
`r` = repeat, `x` = end the conversation, `q` = quit the program, or type any command, e.g. `tell me a joke`.

To see what Buddy heard while you test, set `log_transcripts = true` under `[privacy]` in `buddy.toml`.
Set it back to `false` afterwards.

<!-- RECORDS (temporary): remove this note with the feature -->
**Interaction records (temporary, for testing):** every run also writes down what Buddy heard, understood and said,
as text only (no audio). The voice mode writes to `C:\Users\<you>\buddy-data\records\<date>_voice.jsonl`, and typing
in Step 7 writes to `<date>_chat.jsonl`. Later you can give these files to an AI assistant to find where Buddy
misunderstands. To switch it off, add `[records]` with `enabled = false` to `buddy.toml`. Details: [RECORDS.md](RECORDS.md).

## Step 11 (optional): Run the automated tests

```bash
python -m unittest discover -s tests -t .
```
All tests should end with `OK`. The audio test is skipped if the Piper voice isn't downloaded.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `py` / `python` is not recognised | Reinstall Python and tick "Add python.exe to PATH", then open a new PowerShell window |
| `No module named 'tomllib'` | Your Python is older than 3.11. Install 3.12 and recreate `.venv` (Step 3). |
| `pip install` fails on `sherpa-onnx` | Use Python 3.11 or 3.12. Very new Python versions may not have a ready-made package yet. |
| `No module named buddy` | You're not in the project folder. `cd` into the folder that contains `requirements.txt`. |
| `Missing model file …` | Run Step 5 again, or match `buddy.toml` to the models you downloaded (Step 6) |
| Buddy never reacts | Check the microphone permission (Step 8), turn on `log_transcripts` to see what it hears, and speak a little louder |
| Buddy reacts to its own voice | Use headphones or lower the speaker volume |
| Buddy reacts to people talking after an answer | In `buddy.toml` under `[session]`, lower `follow_up_seconds` (e.g. 8), or set it to 0 to need "Hey Buddy" every time |
| "Hey Buddy, pause" is not heard while Buddy talks | Laptop speakers are loud in the mic: use headphones, lower the volume, or press `p` + Enter |
| Wakes up by mistake / misses "Hey Buddy" | In `buddy.toml` under `[wake]`, set `threshold` to 0.85 (stricter) or 0.75 (easier) |
| `PortAudio library not found` (Linux) | `sudo apt install libportaudio2` |
| "missing speech clip" in the log | Run Step 9, or leave it, since Buddy creates missing clips itself |

## Moving to the Raspberry Pi later

Nothing in the code changes. On the Pi, follow `docs/SETUP.md`. You can even copy the `buddy-data\cache` folder
from your PC to the Pi (`~/buddy-data/cache`) to skip the long voice build there. Use the same voice settings on both.
