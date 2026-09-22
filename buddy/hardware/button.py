"""Physical mute button + status LED on the Raspberry Pi GPIO header (gpiozero).

Wiring (BCM numbering, defaults in config):
    button: one leg -> GPIO17 (pin 11), other leg -> GND (pin 9). Internal pull-up is used.
    LED:    GPIO27 (pin 13) -> 330 ohm resistor -> LED long leg; LED short leg -> GND (pin 14).
LED on = microphone muted (like the red ring on a smart speaker).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger("buddy.hw")


class MuteButton:
    def __init__(self, pin: Optional[int], led_pin: Optional[int], on_press: Callable[[], None]):
        self.button = None
        self.led = None
        if pin is None:
            return
        try:
            from gpiozero import LED, Button  # available on Raspberry Pi OS

            self.button = Button(pin, pull_up=True, bounce_time=0.08)
            self.button.when_pressed = lambda: on_press()
            if led_pin is not None:
                self.led = LED(led_pin)
            log.info("mute button on GPIO%s ready", pin)
        except Exception as e:  # not on a Pi, or gpiozero missing
            log.warning("GPIO mute button disabled (%s). Use the keyboard: press m + Enter.", e)
            self.button = None

    def show_muted(self, muted: bool) -> None:
        if self.led is not None:
            self.led.on() if muted else self.led.off()

    def close(self) -> None:
        for dev in (self.button, self.led):
            if dev is not None:
                dev.close()
