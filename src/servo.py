#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""SG90 servo motor driver for the Jetson 40-pin header.

Controls the door latch via 50 Hz PWM on Jetson BOARD pin 33 (PWM5).
Duty cycle 5 % → locked (0°); duty cycle 10 % → unlocked (90°).
``unlock_then_relock()`` is the primary entry point used by
ActuatorController; it blocks for ``UNLOCK_HOLD_S`` seconds before
returning the servo to the locked position.
"""

import time

import Jetson.GPIO as GPIO

SERVO_PIN: int = 33
SERVO_FREQ_HZ: int = 50
DUTY_LOCKED: float = 5.0
DUTY_UNLOCKED: float = 10.0
UNLOCK_HOLD_S: float = 3.0


class Servo:
    """SG90 servo motor controlling the door latch (Jetson pin 33 / PWM5)."""

    def __init__(self, pin: int = SERVO_PIN) -> None:
        """Initialise PWM on *pin* and hold the servo in the locked position.

        Parameters
        ----------
        pin:
            BOARD pin number for the PWM signal wire.
            Defaults to ``SERVO_PIN`` (33).
        """
        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._pwm = GPIO.PWM(self.pin, SERVO_FREQ_HZ)
        self._pwm.start(DUTY_LOCKED)

    def set_lock(self, locked: bool) -> None:
        """Move the servo to the locked or unlocked position immediately.

        Parameters
        ----------
        locked:
            ``True`` → duty cycle ``DUTY_LOCKED`` (0°, latch closed).
            ``False`` → duty cycle ``DUTY_UNLOCKED`` (90°, latch open).
        """
        self._pwm.ChangeDutyCycle(DUTY_LOCKED if locked else DUTY_UNLOCKED)

    def set_angle(self, duty_cycle: float) -> None:
        """Set an arbitrary PWM duty cycle for manual angle control.

        Parameters
        ----------
        duty_cycle:
            Duty cycle in percent (0.0 – 100.0).  Use ``DUTY_LOCKED``
            (5 %) or ``DUTY_UNLOCKED`` (10 %) for the standard positions.
        """
        self._pwm.ChangeDutyCycle(duty_cycle)

    def unlock_then_relock(self) -> None:
        """Rotate to the unlocked position, hold for ``UNLOCK_HOLD_S`` s, then relock.

        This is the primary method called by ActuatorController on a GRANT
        decision.  The call blocks for ``UNLOCK_HOLD_S`` seconds so the
        caller must run it in a daemon thread to keep the main pipeline live.
        """
        self._pwm.ChangeDutyCycle(DUTY_UNLOCKED)
        time.sleep(UNLOCK_HOLD_S)
        self._pwm.ChangeDutyCycle(DUTY_LOCKED)

    def cleanup(self) -> None:
        """Return servo to locked position, stop PWM, and release GPIO."""
        self._pwm.ChangeDutyCycle(DUTY_LOCKED)
        time.sleep(0.5)
        self._pwm.stop()
        GPIO.cleanup([self.pin])
