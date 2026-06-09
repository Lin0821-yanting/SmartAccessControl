#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
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

    Thread safety: each method acquires a reentrant lock so overlapping
    decisions (e.g. rapid-fire deny during an ongoing grant) do not race.

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
          1. Servo rotates to 90° (latch opens).
          2. Green LED illuminates for _GRANT_LED_S seconds.
          3. Servo auto-returns to 0° (latch closes).

        The servo relock happens inside unlock_then_relock(), which blocks
        for UNLOCK_HOLD_S seconds, so this call is synchronous.
        Call from a worker thread if you need the main pipeline to stay live.
        """
        with self._lock:
            logger.info("ACT GRANT — servo unlock + green LED")
            # Start servo unlock (blocks for UNLOCK_HOLD_S internally)
            # We run it in a thread so green LED can be concurrent
            servo_thread = threading.Thread(
                target=self._servo.unlock_then_relock, daemon=True
            )
            servo_thread.start()
            # Green LED stays on for grant duration
            self._led.indicate(success=True, duration=_GRANT_LED_S)
            servo_thread.join()  # ensure servo relocks before method returns

    def deny_access(self) -> None:
        """Illuminate the red LED to signal a denied access attempt.

        Triggered when the face is in the DB but similarity < 0.85.
        Red LED ON for _DENY_LED_S seconds. No buzzer, no servo action.
        """
        with self._lock:
            logger.info("ACT DENY — red LED")
            self._led.indicate(success=False, duration=_DENY_LED_S)

    def alert_unknown(self) -> None:
        """Alert with red LED and buzzer for an unrecognised face.

        Triggered when a face is detected but is not enrolled in the DB.
        Red LED ON for _DENY_LED_S seconds, plus _ALERT_BEEPS buzzer pulses.
        """
        with self._lock:
            logger.info("ACT UNKNOWN — red LED + %d beeps", _ALERT_BEEPS)
            # Run LED and buzzer concurrently (LED duration ≥ buzzer duration)
            led_thread = threading.Thread(
                target=self._led.indicate,
                kwargs={"success": False, "duration": _DENY_LED_S},
                daemon=True,
            )
            led_thread.start()
            self._multi_beep(_ALERT_BEEPS)
            led_thread.join()

    def alert_spoof(self) -> None:
        """Alert with red LED and buzzer when a liveness check fails.

        Identical hardware sequence to alert_unknown().
        The distinction is handled upstream via the MQTT payload.
        """
        with self._lock:
            logger.info("ACT SPOOF — red LED + %d beeps", _ALERT_BEEPS)
            led_thread = threading.Thread(
                target=self._led.indicate,
                kwargs={"success": False, "duration": _DENY_LED_S},
                daemon=True,
            )
            led_thread.start()
            self._multi_beep(_ALERT_BEEPS)
            led_thread.join()

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
            # Directly drive buzzer pin via its private _beep if available,
            # otherwise fall back to indicate(success=False) once.
            # Using internal _beep gives us finer control over count and gap.
            self._buzzer._beep(_BEEP_ON_S)
            if i < count - 1:
                time.sleep(_BEEP_OFF_S)
