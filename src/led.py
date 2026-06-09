#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""GRANT/DENY indicator LED driver for the Jetson 40-pin header.

Drives two LEDs (green = access granted, red = access denied) via
Jetson.GPIO BOARD-numbered pins.  All timing constants are module-level
so callers can override them without subclassing.
"""

import time

import Jetson.GPIO as GPIO

GREEN_LED_PIN: int = 7
RED_LED_PIN: int = 11
GREEN_LED_HOLD_S: float = 3.0
RED_LED_HOLD_S: float = 2.0


class LED:
    """GRANT/DENY indicator LEDs on the Jetson 40-pin header."""

    def __init__(
        self, green_pin: int = GREEN_LED_PIN, red_pin: int = RED_LED_PIN
    ) -> None:
        """Initialise GPIO pins for green and red LEDs.

        Parameters
        ----------
        green_pin:
            BOARD pin number for the green (grant) LED. Defaults to ``GREEN_LED_PIN`` (7).
        red_pin:
            BOARD pin number for the red (deny) LED. Defaults to ``RED_LED_PIN`` (11).
        """
        self.green_pin = green_pin
        self.red_pin = red_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.green_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.red_pin, GPIO.OUT, initial=GPIO.LOW)

    def indicate(self, success: bool, duration: float | None = None) -> None:
        """Light the appropriate LED for *duration* seconds then turn it off.

        Parameters
        ----------
        success:
            ``True`` → green LED (access granted); ``False`` → red LED (denied).
        duration:
            How long to keep the LED on in seconds.  Defaults to
            ``GREEN_LED_HOLD_S`` on success or ``RED_LED_HOLD_S`` on failure.
        """
        if duration is None:
            duration = GREEN_LED_HOLD_S if success else RED_LED_HOLD_S
        pin = self.green_pin if success else self.red_pin
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)

    def cleanup(self) -> None:
        """Drive both LEDs LOW and release their GPIO resources."""
        GPIO.output(self.green_pin, GPIO.LOW)
        GPIO.output(self.red_pin, GPIO.LOW)
        GPIO.cleanup([self.green_pin, self.red_pin])
