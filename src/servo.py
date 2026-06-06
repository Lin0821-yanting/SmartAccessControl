#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""SG90 servo motor driver for the Jetson 40-pin header.

Controls the door latch via 50 Hz software PWM on BOARD pin 33
(SoC PH.00, gpiochip0 line 43).

Jetson.GPIO hardware PWM is not available on the Yahboom carrier board
for this pin, so this module implements software PWM using gpiod,
consistent with the pattern established in capstone_pins.py.

Duty cycle:  5 % (1.0 ms pulse) → locked  (0°)
            10 % (2.0 ms pulse) → unlocked (90°)
``unlock_then_relock()`` is the primary entry point used by
ActuatorController; it blocks for ``UNLOCK_HOLD_S`` seconds before
returning the servo to the locked position.
"""

import threading
import time

import gpiod

CHIP_NAME: str = "gpiochip0"
SERVO_LINE: int = 43          # PH.00 — BOARD pin 33
CONSUMER: str = "capstone"

SERVO_FREQ_HZ: int = 50
PERIOD_S: float = 1.0 / SERVO_FREQ_HZ   # 20 ms

# SG90: 1.0 ms pulse = 0° (locked), 2.0 ms pulse = 90° (unlocked)
PULSE_LOCKED_MS: float = 1.0
PULSE_UNLOCKED_MS: float = 2.0
UNLOCK_HOLD_S: float = 3.0


class _SoftPWM:  # pragma: no cover
    """Background-thread software PWM for a single gpiod line."""

    def __init__(self, line: gpiod.Line) -> None:
        """Initialise with an already-requested gpiod output line."""
        self._line = line
        self._pulse_s: float = PULSE_LOCKED_MS / 1000.0
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, pulse_ms: float) -> None:
        """Start the PWM loop with *pulse_ms* millisecond ON time."""
        with self._lock:
            self._pulse_s = pulse_ms / 1000.0
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_pulse(self, pulse_ms: float) -> None:
        """Change pulse width while the PWM loop is running."""
        with self._lock:
            self._pulse_s = pulse_ms / 1000.0

    def stop(self) -> None:
        """Stop the PWM loop and drive the line LOW."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        self._line.set_value(0)

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
                pulse_s = self._pulse_s
            self._line.set_value(1)
            time.sleep(pulse_s)
            self._line.set_value(0)
            time.sleep(PERIOD_S - pulse_s)


class Servo:
    """SG90 servo motor controlling the door latch (BOARD pin 33 / gpiochip0 line 43).

    Uses gpiod software PWM because Jetson.GPIO hardware PWM is not
    available on the Yahboom carrier board for this pin.
    """

    def __init__(
        self,
        chip_name: str = CHIP_NAME,
        line_num: int = SERVO_LINE,
    ) -> None:
        """Initialise gpiod line and start PWM in the locked position.

        Parameters
        ----------
        chip_name:
            gpiod chip name. Defaults to ``"gpiochip0"``.
        line_num:
            gpiod line number for the servo signal wire.
            Defaults to ``SERVO_LINE`` (43, BOARD pin 33).
        """
        self._chip = gpiod.Chip(chip_name)
        self._line = self._chip.get_line(line_num)
        self._line.request(
            consumer=CONSUMER,
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[0],
        )
        self._pwm = _SoftPWM(self._line)
        self._pwm.start(PULSE_LOCKED_MS)

    def set_lock(self, locked: bool) -> None:
        """Move the servo to the locked or unlocked position immediately.

        Parameters
        ----------
        locked:
            ``True`` → 1.0 ms pulse (0°, latch closed).
            ``False`` → 2.0 ms pulse (90°, latch open).
        """
        pulse = PULSE_LOCKED_MS if locked else PULSE_UNLOCKED_MS
        self._pwm.set_pulse(pulse)

    def set_angle(self, pulse_ms: float) -> None:
        """Set an arbitrary pulse width for manual angle control.

        Parameters
        ----------
        pulse_ms:
            Pulse width in milliseconds (1.0 – 2.0 for this servo).
        """
        self._pwm.set_pulse(pulse_ms)

    def unlock_then_relock(self) -> None:
        """Rotate to the unlocked position, hold for ``UNLOCK_HOLD_S`` s, then relock.

        This is the primary method called by ActuatorController on a GRANT
        decision.  The call blocks for ``UNLOCK_HOLD_S`` seconds so the
        caller must run it in a daemon thread to keep the main pipeline live.
        """
        self._pwm.set_pulse(PULSE_UNLOCKED_MS)
        time.sleep(UNLOCK_HOLD_S)
        self._pwm.set_pulse(PULSE_LOCKED_MS)

    def cleanup(self) -> None:
        """Return servo to locked position, stop PWM, and release GPIO."""
        self._pwm.set_pulse(PULSE_LOCKED_MS)
        time.sleep(0.5)
        self._pwm.stop()
        self._line.release()
        self._chip.close()
