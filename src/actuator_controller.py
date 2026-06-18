#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題
"""High-level actuator façade for the door-access system.

Decision engine calls one of four semantic methods; this module owns
all hardware timing constants and drives LED / Buzzer / Servo through
their respective low-level classes.

Decision → ActuatorController method mapping (from proposal §4.5):
  GRANT   → grant_access()   green LED 3 s + servo 90° (auto-relock)
  DENY    → deny_access()    red LED 2 s
  UNKNOWN → alert_unknown()  red LED 2 s + buzzer 3 beeps
  SPOOF   → alert_spoof()    red LED 2 s + buzzer 3 beeps  (same HW, diff MQTT)
  IGNORE  → (no call)

Drop-on-busy design
-------------------
All public methods use ``self._lock.acquire(blocking=False)``.  If the
actuator is already executing a previous decision the new call is
silently dropped.  This prevents thread-queue backlog when the camera
pipeline produces decisions faster than hardware can execute them
(e.g. UNKNOWN at 18 FPS would queue 270 beeps without this guard).
"""

import logging
import threading

from src.buzzer import Buzzer
from src.led import LED
from src.servo import Servo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants — all durations in seconds
# Change these constants to tune behaviour; never scatter magic numbers in code
# ---------------------------------------------------------------------------
_GRANT_LED_S: float = 3.0  # green LED hold on GRANT
_DENY_LED_S: float = 2.0  # red LED hold on DENY / UNKNOWN / SPOOF
_ALERT_BEEPS: int = 3  # number of beeps on UNKNOWN / SPOOF
_BEEP_ON_S: float = 0.20  # buzzer ON duration per beep
_BEEP_OFF_S: float = 0.15  # buzzer OFF gap between beeps


class ActuatorController:
    """Semantic façade over LED, Buzzer, and Servo.

    Each public method maps 1-to-1 with a decision outcome and fully
    encapsulates the required hardware sequence.  Callers must NOT
    manipulate LED / Buzzer / Servo directly after constructing this class.

    **Drop-on-busy**: every public method calls
    ``self._lock.acquire(blocking=False)``.  If the lock is already held
    by another thread the call returns immediately without queuing.  This
    is intentional — stale decisions accumulating in a queue produce
    confusing LED / buzzer behaviour (e.g. buzzer beeping for 90 seconds
    after a person walks away).

    Thread safety: the reentrant lock still serialises concurrent callers
    that slip through the non-blocking guard (e.g. two calls on the same
    thread).

    Usage::

        controller = ActuatorController()
        try:
            controller.grant_access()   # unlocks door, green LED, auto-relocks
            controller.deny_access()    # red LED only
            controller.alert_unknown()  # red LED + 3 beeps
            controller.alert_spoof()    # same HW as unknown; MQTT differs upstream
        finally:
            controller.cleanup()
    """

    def __init__(
        self,
        led: LED | None = None,
        buzzer: Buzzer | None = None,
        servo: Servo | None = None,
    ) -> None:
        """Initialise with optional injected hardware instances.

        Parameters
        ----------
        led, buzzer, servo:
            Inject pre-constructed instances (useful in unit tests with mocks).
            Pass None (default) to let ActuatorController instantiate them.
        """
        self._led: LED = led if led is not None else LED()
        self._buzzer: Buzzer = buzzer if buzzer is not None else Buzzer()
        self._servo: Servo = servo if servo is not None else Servo()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public semantic methods
    # ------------------------------------------------------------------

    def grant_access(self) -> None:
        """Unlock the door latch and illuminate the green LED.

        Sequence:
          1. Servo rotates to 90° (latch opens) — burst-pulse, then silent hold.
          2. Green LED illuminates for _GRANT_LED_S seconds (concurrent).
          3. Servo auto-returns to 0° (latch closes) — burst-pulse.

        Dropped silently if the actuator is already busy (drop-on-busy).
        Call from a worker thread; total blocking time ≈ 4.2 s.
        """
        if not self._lock.acquire(blocking=False):
            logger.debug("ACT GRANT — skipped (actuator busy)")
            return
        try:
            logger.info("ACT GRANT — servo unlock + green LED")
            servo_thread = threading.Thread(target=self._servo.unlock_then_relock, daemon=False)
            servo_thread.start()
            self._led.indicate(success=True, duration=_GRANT_LED_S)
            servo_thread.join()
        finally:
            self._lock.release()

    def deny_access(self) -> None:
        """Illuminate the red LED to signal a denied access attempt.

        Triggered when the face is in the DB but similarity < 0.85.
        Red LED ON for _DENY_LED_S seconds. No buzzer, no servo action.
        Dropped silently if the actuator is already busy (drop-on-busy).
        """
        if not self._lock.acquire(blocking=False):
            logger.debug("ACT DENY — skipped (actuator busy)")
            return
        try:
            logger.info("ACT DENY — red LED")
            self._led.indicate(success=False, duration=_DENY_LED_S)
        finally:
            self._lock.release()

    def alert_unknown(self) -> None:
        """Alert with red LED and buzzer for an unrecognised face.

        Triggered when a face is detected but is not enrolled in the DB.
        Red LED ON for _DENY_LED_S seconds, plus _ALERT_BEEPS buzzer pulses.
        Dropped silently if the actuator is already busy (drop-on-busy).
        """
        if not self._lock.acquire(blocking=False):
            logger.debug("ACT UNKNOWN — skipped (actuator busy)")
            return
        try:
            logger.info("ACT UNKNOWN — red LED + %d beeps", _ALERT_BEEPS)
            led_thread = threading.Thread(
                target=self._led.indicate,
                kwargs={"success": False, "duration": _DENY_LED_S},
                daemon=False,
            )
            led_thread.start()
            # self._multi_beep(_ALERT_BEEPS)
            led_thread.join()
        finally:
            self._lock.release()

    def alert_spoof(self) -> None:
        """Alert with red LED and buzzer when a liveness check fails.

        Identical hardware sequence to alert_unknown().
        The distinction is handled upstream via the MQTT payload.
        Dropped silently if the actuator is already busy (drop-on-busy).
        """
        if not self._lock.acquire(blocking=False):
            logger.debug("ACT SPOOF — skipped (actuator busy)")
            return
        try:
            logger.info("ACT SPOOF — red LED + %d beeps", _ALERT_BEEPS)
            led_thread = threading.Thread(
                target=self._led.indicate,
                kwargs={"success": False, "duration": _DENY_LED_S},
                daemon=False,
            )
            led_thread.start()
            self._multi_beep(_ALERT_BEEPS)
            led_thread.join()
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all GPIO resources. Call in a finally block or atexit."""
        logger.info("ActuatorController cleanup")
        self._led.cleanup()
        self._buzzer.cleanup()
        self._servo.cleanup()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _multi_beep(self, count: int) -> None:
        """Emit *count* short beeps with gaps between them."""
        import time

        for i in range(count):
            self._buzzer._beep(_BEEP_ON_S)
            if i < count - 1:
                time.sleep(_BEEP_OFF_S)
