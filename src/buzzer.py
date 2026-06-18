#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題
"""Piezo buzzer driver for the Jetson 40-pin header.

Drives a single piezo buzzer via Jetson.GPIO BOARD-numbered pin.
Provides a high-level indicate() façade as well as the low-level
_beep() primitive used by ActuatorController for multi-beep patterns.
"""

import time

import Jetson.GPIO as GPIO

BUZZER_PIN: int = 29
BEEP_ON_S: float = 0.2
BEEP_OFF_S: float = 0.1
LONG_BEEP_S: float = 0.6


class Buzzer:
    """Piezo buzzer on Jetson pin 29."""

    def __init__(self, pin: int = BUZZER_PIN) -> None:
        """Initialise the GPIO pin for the buzzer.

        Parameters
        ----------
        pin:
            BOARD pin number for the buzzer signal wire.
            Defaults to ``BUZZER_PIN`` (29).
        """
        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def _beep(self, duration: float) -> None:
        GPIO.output(self.pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.pin, GPIO.LOW)

    def indicate(self, success: bool) -> None:
        """Emit a short beep on success or a short-then-long beep on failure.

        Parameters
        ----------
        success:
            ``True`` → single short beep (``BEEP_ON_S``).
            ``False`` → short beep, gap, then long beep (``LONG_BEEP_S``).
        """
        if success:
            self._beep(BEEP_ON_S)
        else:
            self._beep(BEEP_ON_S)
            time.sleep(BEEP_OFF_S)
            self._beep(LONG_BEEP_S)

    def cleanup(self) -> None:
        """Drive the buzzer pin LOW and release its GPIO resource."""
        GPIO.output(self.pin, GPIO.LOW)
        GPIO.cleanup([self.pin])
