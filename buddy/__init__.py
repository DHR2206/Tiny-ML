"""Buddy - an offline, privacy-first voice learning companion for children aged 3-8.

Pipeline:  microphone -> VAD -> on-device ASR -> wake/intent matching (child-phonology aware)
           -> skill -> pre-synthesised speech/music -> speaker

Everything runs locally on a Raspberry Pi 5. No audio is ever written to disk or sent anywhere.
"""

__version__ = "0.3.0"
