#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""HC-SR04 ultrasonic distance sensor driver for the Jetson 40-pin header.

Polls the sensor at ``POLL_INTERVAL_S`` intervals and applies a 2-of-3
majority vote to confirm proximity, filtering out single-sample noise.
The class serves as the sense gate for the AI pipeline: the pipeline
only activates when someone is within ``APPROACH_THRESHOLD_CM`` cm.

Stuck-ECHO recovery
-------------------
When the AI pipeline (TRT inference) heavily loads the CPU, the ECHO
timeout can fire while the ECHO pin is still HIGH.  The HC-SR04 then
keeps ECHO HIGH indefinitely until it is power-cycled.  ``_measure_distance``
detects this condition at the start of every measurement and holds TRIG LOW
for ``_STUCK_RECOVERY_S`` seconds to force the sensor back to idle.
"""

import contextlib
import logging
import time

import Jetson.GPIO as GPIO

logger = logging.getLogger(__name__)

TRIG_PIN: int = 31
ECHO_PIN: int = 15
SPEED_OF_SOUND_CM_PER_S: float = 34300.0
TRIGGER_PULSE_S: float = 10e-6
ECHO_TIMEOUT_S: float = 0.05
POLL_INTERVAL_S: float = 0.10
APPROACH_THRESHOLD_CM: float = 60.0

_CONFIRM_SAMPLES: int = 3  # total samples per majority-vote window
_CONFIRM_HITS: int = 2  # minimum hits required to confirm proximity
_STUCK_RECOVERY_S: float = 0.15  # TRIG-LOW hold time to un-stuck ECHO


class HcSr04:
    """HC-SR04 ultrasonic distance sensor — sense gate for the AI pipeline."""

    def __init__(
        self,
        trigger_pin: int = TRIG_PIN,
        echo_pin: int = ECHO_PIN,
        threshold_cm: float = APPROACH_THRESHOLD_CM,
    ) -> None:
        """Initialise GPIO pins and proximity threshold.

        Parameters
        ----------
        trigger_pin:
            BOARD pin number connected to the sensor TRIG line.
            Defaults to ``TRIG_PIN`` (31).
        echo_pin:
            BOARD pin number connected to the sensor ECHO line (via
            voltage divider: 1 kΩ + 2 kΩ to bring 5 V → 3.3 V).
            Defaults to ``ECHO_PIN`` (15).
        threshold_cm:
            Distance in centimetres below which a person is considered
            present.  Defaults to ``APPROACH_THRESHOLD_CM`` (60.0 cm).
        """
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.threshold_cm = threshold_cm
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        # Defensive cleanup: clear any state left by a previous process
        with contextlib.suppress(Exception):
            GPIO.cleanup([self.trigger_pin, self.echo_pin])
        GPIO.setup(self.trigger_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.echo_pin, GPIO.IN)
        # Allow sensor to settle after power-on / pin reconfiguration
        time.sleep(0.05)

    def _recover_stuck_echo(self) -> bool:
        """Attempt to un-stuck ECHO by holding TRIG LOW for _STUCK_RECOVERY_S.

        Returns ``True`` if ECHO returned LOW (sensor recovered),
        ``False`` if ECHO is still HIGH after recovery attempt.
        """
        logger.warning("HC-SR04: ECHO stuck HIGH — attempting recovery")
        GPIO.output(self.trigger_pin, GPIO.LOW)
        time.sleep(_STUCK_RECOVERY_S)
        recovered = GPIO.input(self.echo_pin) == GPIO.LOW
        if recovered:
            logger.info("HC-SR04: ECHO recovered")
        else:
            logger.error(
                "HC-SR04: ECHO still HIGH after recovery — power-cycle required"
            )
        return recovered

    def _measure_distance(self) -> float:
        """Send one ultrasonic pulse and return the measured distance in cm.

        Returns ``float('inf')`` on timeout or when the sensor is stuck.
        Automatically attempts stuck-ECHO recovery before each measurement.
        """
        # ── Stuck-ECHO guard ──────────────────────────────────────────────
        if GPIO.input(self.echo_pin) == GPIO.HIGH and not self._recover_stuck_echo():
            return float("inf")

        # ── Trigger pulse ─────────────────────────────────────────────────
        GPIO.output(self.trigger_pin, GPIO.LOW)
        time.sleep(2e-6)
        GPIO.output(self.trigger_pin, GPIO.HIGH)
        time.sleep(TRIGGER_PULSE_S)
        GPIO.output(self.trigger_pin, GPIO.LOW)

        # ── Wait for ECHO HIGH (pulse start) ──────────────────────────────
        deadline = time.monotonic() + ECHO_TIMEOUT_S
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            if time.monotonic() > deadline:
                return float("inf")
        t_start = time.monotonic()

        # ── Wait for ECHO LOW (pulse end) ─────────────────────────────────
        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            if time.monotonic() > deadline:
                # ECHO is stuck HIGH — mark for recovery on next call
                logger.debug("HC-SR04: timeout while ECHO=HIGH")
                return float("inf")

        return (time.monotonic() - t_start) * SPEED_OF_SOUND_CM_PER_S / 2.0

    def _confirmed_near(self) -> bool:
        hits = sum(
            1
            for _ in range(_CONFIRM_SAMPLES)
            if self._measure_distance() <= self.threshold_cm
        )
        return hits >= _CONFIRM_HITS

    def measure_distance(self) -> float | None:
        """Return distance in cm, or None on timeout.

        Public wrapper used by the orchestrator gate check.
        """
        d = self._measure_distance()
        return None if d == float("inf") else d

    def is_someone_near(self) -> bool:
        """Return True if a person is within threshold_cm on this sample.

        Takes ``_CONFIRM_SAMPLES`` distance readings and returns ``True``
        when at least ``_CONFIRM_HITS`` of them are within
        ``self.threshold_cm`` (2-of-3 majority vote).
        """
        return self._confirmed_near()

    def wait_for_person(self) -> None:
        """Block until a person is confirmed within threshold_cm.

        Polls ``_confirmed_near()`` every ``POLL_INTERVAL_S`` seconds.
        Returns as soon as the majority-vote condition is satisfied.
        """
        while not self._confirmed_near():
            time.sleep(POLL_INTERVAL_S)

    def cleanup(self) -> None:
        """Drive TRIG LOW, then release GPIO resources for both pins."""
        with contextlib.suppress(Exception):
            GPIO.output(self.trigger_pin, GPIO.LOW)
        GPIO.cleanup([self.trigger_pin, self.echo_pin])


# Alias for backward compatibility with orchestrator.py
HCSR04 = HcSr04
