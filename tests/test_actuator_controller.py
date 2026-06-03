#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""tests/test_actuator_controller.py — unit tests for ActuatorController.

All LED / Buzzer / Servo dependencies are injected as MagicMock objects so
these tests run on any x86 CI runner without a real Jetson or GPIO library.

Coverage target: ≥ 90% of src/actuator_controller.py (enforced by CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

# Safe to import after the stub is in place
from src.actuator_controller import (  # noqa: E402
    ActuatorController,
    _ALERT_BEEPS,
    _BEEP_ON_S,
    _DENY_LED_S,
    _GRANT_LED_S,
)


# ---------------------------------------------------------------------------
# Shared fixture — fresh mocks + controller for every test
# ---------------------------------------------------------------------------

@pytest.fixture()
def mocks():
    """Return (mock_led, mock_buzzer, mock_servo, controller)."""
    mock_led    = MagicMock()
    mock_buzzer = MagicMock()
    mock_servo  = MagicMock()
    ctrl = ActuatorController(
        led=mock_led,
        buzzer=mock_buzzer,
        servo=mock_servo,
    )
    return mock_led, mock_buzzer, mock_servo, ctrl


# ===========================================================================
# grant_access
# ===========================================================================

class TestGrantAccess:
    """grant_access() → green LED + servo unlock; buzzer must stay silent."""

    def test_calls_led_indicate_with_green_and_correct_duration(self, mocks) -> None:
        """LED must be told success=True and duration=_GRANT_LED_S."""
        mock_led, _, _, ctrl = mocks
        ctrl.grant_access()
        mock_led.indicate.assert_called_once_with(
            success=True, duration=_GRANT_LED_S
        )

    def test_calls_servo_unlock_then_relock(self, mocks) -> None:
        """Servo must be unlocked-then-relocked exactly once."""
        _, _, mock_servo, ctrl = mocks
        ctrl.grant_access()
        mock_servo.unlock_then_relock.assert_called_once()

    def test_buzzer_is_never_called(self, mocks) -> None:
        """Buzzer must remain completely silent on GRANT."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.grant_access()
        mock_buzzer._beep.assert_not_called()
        mock_buzzer.indicate.assert_not_called()

    def test_servo_is_not_set_lock_directly(self, mocks) -> None:
        """grant_access() must use unlock_then_relock, not set_lock directly."""
        _, _, mock_servo, ctrl = mocks
        ctrl.grant_access()
        mock_servo.set_lock.assert_not_called()


# ===========================================================================
# deny_access
# ===========================================================================

class TestDenyAccess:
    """deny_access() → red LED only; servo and buzzer must stay silent."""

    def test_calls_led_indicate_with_red_and_correct_duration(self, mocks) -> None:
        """LED must be told success=False and duration=_DENY_LED_S."""
        mock_led, _, _, ctrl = mocks
        ctrl.deny_access()
        mock_led.indicate.assert_called_once_with(
            success=False, duration=_DENY_LED_S
        )

    def test_buzzer_is_never_called(self, mocks) -> None:
        """Buzzer must stay silent on DENY — this distinguishes DENY from UNKNOWN."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.deny_access()
        mock_buzzer._beep.assert_not_called()
        mock_buzzer.indicate.assert_not_called()

    def test_servo_is_never_called(self, mocks) -> None:
        """Servo must not move on DENY."""
        _, _, mock_servo, ctrl = mocks
        ctrl.deny_access()
        mock_servo.unlock_then_relock.assert_not_called()
        mock_servo.set_lock.assert_not_called()


# ===========================================================================
# alert_unknown
# ===========================================================================

class TestAlertUnknown:
    """alert_unknown() → red LED + _ALERT_BEEPS buzzer pulses; no servo."""

    def test_calls_led_indicate_with_red(self, mocks) -> None:
        """LED must be told success=False with the deny duration."""
        mock_led, _, _, ctrl = mocks
        ctrl.alert_unknown()
        mock_led.indicate.assert_called_once_with(
            success=False, duration=_DENY_LED_S
        )

    def test_buzzer_beeps_correct_number_of_times(self, mocks) -> None:
        """_beep must be called exactly _ALERT_BEEPS times."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_unknown()
        assert mock_buzzer._beep.call_count == _ALERT_BEEPS

    def test_buzzer_beep_duration_is_correct(self, mocks) -> None:
        """Every _beep call must use _BEEP_ON_S as the duration."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_unknown()
        for c in mock_buzzer._beep.call_args_list:
            assert c == call(_BEEP_ON_S)

    def test_servo_is_never_called(self, mocks) -> None:
        """Servo must not move on UNKNOWN."""
        _, _, mock_servo, ctrl = mocks
        ctrl.alert_unknown()
        mock_servo.unlock_then_relock.assert_not_called()
        mock_servo.set_lock.assert_not_called()


# ===========================================================================
# alert_spoof
# ===========================================================================

class TestAlertSpoof:
    """alert_spoof() → identical hardware sequence to alert_unknown().

    The two methods share the same actuator behaviour; the distinction lives
    in the MQTT payload (handled upstream).  Both code paths must be covered.
    """

    def test_calls_led_indicate_with_red(self, mocks) -> None:
        """LED must be told success=False with the deny duration."""
        mock_led, _, _, ctrl = mocks
        ctrl.alert_spoof()
        mock_led.indicate.assert_called_once_with(
            success=False, duration=_DENY_LED_S
        )

    def test_buzzer_beeps_correct_number_of_times(self, mocks) -> None:
        """_beep must be called exactly _ALERT_BEEPS times."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_spoof()
        assert mock_buzzer._beep.call_count == _ALERT_BEEPS

    def test_servo_is_never_called(self, mocks) -> None:
        """Servo must not move on SPOOF."""
        _, _, mock_servo, ctrl = mocks
        ctrl.alert_spoof()
        mock_servo.unlock_then_relock.assert_not_called()
        mock_servo.set_lock.assert_not_called()


# ===========================================================================
# cleanup
# ===========================================================================

class TestCleanup:
    """cleanup() must release all three hardware dependencies."""

    def test_led_cleanup_is_called(self, mocks) -> None:
        """LED cleanup must be invoked exactly once."""
        mock_led, _, _, ctrl = mocks
        ctrl.cleanup()
        mock_led.cleanup.assert_called_once()

    def test_buzzer_cleanup_is_called(self, mocks) -> None:
        """Buzzer cleanup must be invoked exactly once."""
        _, mock_buzzer, _, ctrl = mocks
        ctrl.cleanup()
        mock_buzzer.cleanup.assert_called_once()

    def test_servo_cleanup_is_called(self, mocks) -> None:
        """Servo cleanup must be invoked exactly once."""
        _, _, mock_servo, ctrl = mocks
        ctrl.cleanup()
        mock_servo.cleanup.assert_called_once()
