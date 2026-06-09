#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""SG90 servo motor driver for the Jetson 40-pin header.

Controls the door latch via synchronous burst-pulse PWM on BOARD pin 33
(SoC PH.00, gpiochip0 line 43).

Jetson.GPIO hardware PWM is not available on the Yahboom carrier board
for this pin, so this module implements software PWM using gpiod.

Unlike a continuous-background-thread approach, this driver sends a
finite burst of pulses to reach the target position and then silences
the signal.  This eliminates two failure modes:

1. **Jitter at rest**: No background timer is running between commands,
   so there is no timer-inaccuracy noise to cause hunting.
2. **GIL starvation**: The burst runs synchronously in the calling
   thread; the GIL cannot starve it mid-burst the way it can starve a
   background daemon thread.

Duty cycle:  5 % (1.0 ms pulse) → locked  (0°)
            10 % (2.0 ms pulse) → unlocked (90°)
``unlock_then_relock()`` is the primary entry point used by
ActuatorController; it blocks for ``UNLOCK_HOLD_S`` seconds before
returning the servo to the locked position.
"""

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

# How long to send pulses when driving to a new position.
# 30 cycles × 20 ms = 600 ms — more than enough for SG90 (~300 ms travel).
SETTLE_S: float = 0.6


class Servo:
    """SG90 servo motor controlling the door latch (BOARD pin 33 / gpiochip0 line 43).

    Uses gpiod synchronous burst-pulse PWM.  A background thread is NOT
    used; instead, ``_goto()`` sends a fixed-duration burst and then
    silences the signal.  The SG90 holds its last commanded position
    without a continuous signal, so this approach is both quieter and
    more reliable under CPU load.
    """

    def __init__(
        self,
        chip_name: str = CHIP_NAME,
        line_num: int = SERVO_LINE,
    ) -> None:
        """Initialise gpiod line and drive servo to the locked position.

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
        # Drive to locked position immediately, then go silent.
        self._goto(PULSE_LOCKED_MS)

    # ------------------------------------------------------------------
    # Core burst-pulse primitive
    # ------------------------------------------------------------------

    def _goto(self, pulse_ms: float, settle_s: float = SETTLE_S) -> None:  # pragma: no cover
        """Send PWM pulses synchronously for *settle_s* seconds then silence the output.

        Runs in the calling thread so the GIL cannot starve it.
        After the burst the line is driven LOW; the SG90 holds its
        last commanded position without a continuous signal.

        Parameters
        ----------
        pulse_ms:
            Pulse width in milliseconds (1.0 = 0°, 2.0 = 90°).
        settle_s:
            Duration of the pulse burst in seconds.
            Default ``SETTLE_S`` (0.6 s = 30 cycles).
        """
        pulse_s = pulse_ms / 1000.0
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            self._line.set_value(1)
            time.sleep(pulse_s)
            self._line.set_value(0)
            time.sleep(PERIOD_S - pulse_s)
        # Silence the signal — eliminates jitter when idle.
        self._line.set_value(0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_lock(self, locked: bool) -> None:
        """Move the servo to the locked or unlocked position immediately.

        Parameters
        ----------
        locked:
            ``True`` → 1.0 ms pulse (0°, latch closed).
            ``False`` → 2.0 ms pulse (90°, latch open).
        """
        pulse = PULSE_LOCKED_MS if locked else PULSE_UNLOCKED_MS
        self._goto(pulse)

    def set_angle(self, pulse_ms: float) -> None:
        """Set an arbitrary pulse width for manual angle control.

        Parameters
        ----------
        pulse_ms:
            Pulse width in milliseconds (1.0 – 2.0 for this servo).
        """
        self._goto(pulse_ms)

    def unlock_then_relock(self) -> None:
        """Rotate to the unlocked position, hold for ``UNLOCK_HOLD_S`` s, then relock.

        Sequence
        --------
        1. Burst to 90° (``SETTLE_S`` s — servo reaches position).
        2. Signal silenced; servo holds 90° for ``UNLOCK_HOLD_S`` s.
        3. Burst to 0° (``SETTLE_S`` s — servo relocks).

        The call blocks for ``UNLOCK_HOLD_S + 2 × SETTLE_S`` seconds
        (≈ 4.2 s with defaults).  Run it in a worker thread if you need
        the main pipeline to stay live during the unlock period.
        """
        self._goto(PULSE_UNLOCKED_MS)        # burst → servo reaches 90°
        time.sleep(UNLOCK_HOLD_S)            # hold (no signal = no jitter)
        self._goto(PULSE_LOCKED_MS)          # burst → servo returns to 0°

    def cleanup(self) -> None:
        """Return servo to locked position and release GPIO."""
        self._goto(PULSE_LOCKED_MS)
        self._line.set_value(0)
        self._line.release()
        self._chip.close()
