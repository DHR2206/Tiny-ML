# Buddy: setup and build guide (Raspberry Pi 5, 8 GB)

Buddy is an offline voice learning companion for children aged 3–8. You say **"Hey Buddy"**, then a command.
After that Buddy keeps listening, so the next request needs no "Hey Buddy" (see §8).
Everything runs on the Pi. Internet is needed **only once**, during setup.

---

## 1. Parts list

| Part | Notes | Approx. cost (INR) |
|---|---|---|
| Raspberry Pi 5, 8 GB | 4 GB also works, but 8 GB leaves headroom | 8,000 |
| Official 27 W USB-C power supply | under-powered supplies cause random resets | 1,100 |
| microSD card, 32 GB+, A2 class | holds the OS, models and voice cache | 500 |
| Active cooler for Pi 5 | keeps speech models fast (no thermal throttling) | 500 |
| USB microphone | any USB mic; a small omni-directional "conference" mic works best for kids | 500–1,500 |
| USB speaker (or 3.5 mm speaker via a USB sound card) | Pi 5 has no 3.5 mm jack | 500–1,500 |
| 1 push button (tactile, 12 mm) + 2 jumper wires | **mute / unmute** button | 20 |
| *(optional)* 1 red LED + 330 Ω resistor | lights up when muted | 10 |
| *(optional)* Pi 5 RTC battery | keeps the clock right offline (needed only for "quiet hours") | 400 |

No display, microphone HAT or AI accelerator is needed.

## 2. Wiring the mute button (and optional LED)

```
Raspberry Pi 5 40-pin header (BCM numbering)

  pin 9  (GND)    ───────┐
                         [ push button ]
  pin 11 (GPIO17) ───────┘                 ← mute/unmute (internal pull-up, no resistor needed)

  pin 13 (GPIO27) ── 330 Ω ──►|── pin 14 (GND)   ← optional LED (long leg towards the resistor)
```

Press once to **mute** (the microphone stream is closed and the LED turns on). Press again to **unmute**.
A muted Buddy cannot hear anything, so unmuting only works with the button. This is on purpose.

## 3. Install Raspberry Pi OS

1. Use **Raspberry Pi Imager** to flash **Raspberry Pi OS (64-bit)** (Bookworm or newer) to the SD card.
   In the Imager settings, set a username/password and Wi-Fi (needed only for setup), and enable SSH if you like.
2. Boot the Pi, plug in the USB mic and speaker, then open a terminal.
3. Update: `sudo apt update && sudo apt full-upgrade -y && sudo reboot`

## 4. Install Buddy

Copy the `buddy` folder to the Pi (USB stick, `scp`, or git), then:

```bash
cd ~/buddy
bash scripts/install_pi.sh
```

This installs system packages (`libportaudio2`, `gpiozero`), creates a Python virtual environment,
installs `sherpa-onnx`, `numpy` and `sounddevice`, downloads the models (~400 MB) and runs `python -m buddy doctor`.

## 5. Pre-render Buddy's voice (one time)

Every sentence Buddy can say (about 3,050 sentences, roughly 2 hours of speech) is synthesised once into a local cache.
After that the Pi only **plays** audio clips, so answers start instantly and the CPU stays free for listening.

```bash
cd ~/buddy && . .venv/bin/activate
python -m buddy build-content
```

| Voice (`[tts] engine`) | Quality | Time on a Pi 5 (estimate) |
|---|---|---|
| `kokoro` (default) | most natural, warm voice | about 2½–3½ hours, so run it overnight |
| `piper` | clear, slightly robotic | about 20 minutes |

The build is **resumable**: if it stops, run the same command again and it continues where it left off.
After updating Buddy, run it again: it renders only the new sentences.
To build faster, you can run the same command on a laptop (Linux, Mac or Windows with Python and
`pip install sherpa-onnx numpy`). Then copy `~/buddy-data/cache/` to the Pi.

## 6. Configure

Edit `~/buddy-data/buddy.toml` (created from `buddy.example.toml`). The most useful settings:

- `[child] age = 5`: from 3 to 8. Younger means more forgiving of baby-talk pronunciation, and an easier quiz start.
- `[audio] max_volume`: the parental volume cap.
- `[wake] threshold`: raise it if Buddy wakes up by mistake, lower it if it misses "Hey Buddy".
- `[parental] quiet_hours = "21:00-06:30"`, `daily_limit_minutes = 60`: both off by default.
- `[session] follow_up_seconds = 15`: how long Buddy keeps listening after each task without "Hey Buddy".
  `0` restores the old behaviour (wake phrase before every request). `pause_memory_seconds = 180`: how long
  "continue" works after "pause".

## 7. Run

```bash
python -m buddy run
```

Say **"Hey Buddy, tell me a story!"**. You can also say "Hey Buddy", wait for the chime, and then give the command.
When the story ends, a soft chime means Buddy is still listening: just say the next thing ("tell me a joke").
Keyboard helpers while testing: `m` = mute/unmute, `t` = talk (acts like "Hey Buddy"), `p` = pause/continue,
`r` = repeat, `x` = end the conversation, `q` = quit the program, or type a command.

**Start automatically on boot:**

```bash
sudo cp scripts/buddy.service /etc/systemd/system/
sudo sed -i "s#@USER@#$USER#g; s#@APP@#$HOME/buddy#g" /etc/systemd/system/buddy.service
sudo systemctl enable --now buddy
journalctl -u buddy -f      # view logs
```

After setup you can **turn Wi-Fi off** (`sudo nmcli radio wifi off`). Buddy never needs it.

## 8. What children can say

| Say "Hey Buddy, …" | Buddy does |
|---|---|
| tell me a story / the thirsty crow story | one of 11 short stories (original or retold) |
| say a poem / the mango poem / sing a song | one of 14 original poems ("sing" gets a poem, because songs were left out) |
| alphabet / ABC | recites A to Z |
| phonics / what is B for | letter sounds: "B is for ball…" |
| count to 100 / count backwards from 20 / count by twos | counting |
| animal sounds / what does the cow say / lion sound | animal sound plus a fun fact (30 animals) |
| colours / shapes / body parts / days of the week / months / opposites | short lessons |
| table of 5 / what is 3 plus 4 | times tables and small sums |
| let's play a quiz / animal quiz / number game | adaptive 5-question quiz (answer without saying "Hey Buddy") |
| tell me a joke / riddle | jokes, and riddles that wait for an answer |
| I'm sad / I'm scared / breathe with me / magic words | feelings support, calm breathing, manners |
| why is the sky blue? | 20 simple science answers; other "why" questions get an encouraging curiosity reply |
| stop / louder / softer / say it again / another one / what can you do | controls |
| pause / wait · continue / go on · repeat · quit / go to sleep / bye | pause a story or poem and continue it later (from the sentence that was interrupted, for up to 3 minutes), repeat the last thing (or the quiz question), end the conversation |

**Conversation mode.** After "Hey Buddy" and after every answer, Buddy keeps listening for 15 seconds, so a child can
go straight on: "tell me a story" … "another one" … "count from 5 to 15". Speech that isn't a request (people talking
in the room) is ignored. After 15 seconds of silence Buddy says a short goodbye line and waits for "Hey Buddy" again.
**While Buddy is talking**, start with the wake phrase: "Hey Buddy, pause" / "stop" / "repeat" / "quit", or a new
request such as "Hey Buddy, tell me a joke" (the story waits, and "continue" brings it back). The microphone also
hears Buddy's own voice, so plain words are not used then. If the child pauses after
"Hey Buddy", Buddy talks more quietly for a few seconds and waits for the command.

Requests can be phrased freely: "count from 1 to 10", "can you count 1 to 10?", "count one through ten",
"count by threes", "countdown from 5", "what's 12 plus 5?", "take away 2 from 5", "tell me a story about a lion",
"what is the opposite of hot?". Counting works for any numbers from 0 to 100, sums and take-aways up to 20,
times tables 1 to 12.

## 9. Testing without a microphone

```bash
python -m buddy chat                       # type as the child
python -m buddy simulate my_recording.wav  # run a WAV through the whole speech pipeline
python -m unittest discover -s tests       # automated tests
python scripts/evaluate_audio.py labels.json   # accuracy on your own labelled recordings
python -m buddy progress                   # quiz levels per topic
```

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `PortAudio library not found` | `sudo apt install libportaudio2` |
| Buddy doesn't hear anything | `python -m buddy doctor` lists the devices. Set `input_device = "USB"` in the config. Test with `arecord -d 3 t.wav && aplay t.wav`. |
| No sound | set `output_device = "USB"`, check `alsamixer` |
| Wakes up by mistake | raise `[wake] threshold` to 0.85 |
| Reacts to people talking after a task | lower `[session] follow_up_seconds` (e.g. 8), or 0 to need "Hey Buddy" every time |
| "Hey Buddy, pause" not heard while Buddy talks | use headphones or a mic further from the speaker, lower the volume, or press `p` |
| Misses "Hey Buddy" | lower the threshold to 0.75; speak within 1–2 m of the mic |
| "missing speech clip" in the log | run `python -m buddy build-content` again |
| Slow answers | use `asr.model = "moonshine-tiny"`; make sure the cooler is fitted |

## 11. Privacy by design

- No internet connection is used at runtime, and there are no accounts, cloud or API keys.
- Microphone audio exists only in RAM. Buffers are overwritten after use and **never written to disk**.
- What the child says is not logged (`log_transcripts = false`).
- The learning database stores only item IDs and right/wrong counts (e.g. `numbers/after_3: 2 correct`).
  Wipe it with `python -m buddy reset-progress`.
- The mute button **closes** the microphone stream, and the LED shows it.
