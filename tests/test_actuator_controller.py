#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""tests/test_actuator_controller.py — unit tests for ActuatorController."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from src.actuator_controller import (
    _ALERT_BEEPS,
    _BEEP_ON_S,
    _DENY_LED_S,
    _GRANT_LED_S,
    ActuatorController,
)


@pytest.fixture()
def mocks():
    mock_led = MagicMock()
    mock_buzzer = MagicMock()
    mock_servo = MagicMock()
    ctrl = ActuatorController(led=mock_led, buzzer=mock_buzzer, servo=mock_servo)
    return mock_led, mock_buzzer, mock_servo, ctrl


class TestGrantAccess:
    def test_calls_led_indicate_with_green_and_correct_duration(self, mocks):
        mock_led, _, _, ctrl = mocks
        ctrl.grant_access()
        mock_led.indicate.assert_called_once_with(success=True, duration=_GRANT_LED_S)

    def test_calls_servo_unlock_then_relock(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.grant_access()
        mock_servo.unlock_then_relock.assert_called_once()

    def test_buzzer_is_never_called(self, mocks):
        _, mock_buzzer, _, ctrl = mocks
        ctrl.grant_access()
        mock_buzzer._beep.assert_not_called()
        mock_buzzer.indicate.assert_not_called()

    def test_servo_is_not_set_lock_directly(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.grant_access()
        mock_servo.set_lock.assert_not_called()


class TestDenyAccess:
    def test_calls_led_indicate_with_red_and_correct_duration(self, mocks):
        mock_led, _, _, ctrl = mocks
        ctrl.deny_access()
        mock_led.indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_buzzer_is_never_called(self, mocks):
        _, mock_buzzer, _, ctrl = mocks
        ctrl.deny_access()
        mock_buzzer._beep.assert_not_called()
        mock_buzzer.indicate.assert_not_called()

    def test_servo_is_never_called(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.deny_access()
        mock_servo.unlock_then_relock.assert_not_called()
        mock_servo.set_lock.assert_not_called()


class TestAlertUnknown:
    def test_calls_led_indicate_with_red(self, mocks):
        mock_led, _, _, ctrl = mocks
        ctrl.alert_unknown()
        mock_led.indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_buzzer_is_silent_on_unknown(self, mocks):
        # UNKNOWN intentionally does NOT beep (only SPOOF alerts with the buzzer)
        # — see README.md problem-3 rationale: avoid nuisance beeping at people.
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_unknown()
        assert mock_buzzer._beep.call_count == 0

    def test_buzzer_beep_duration_is_correct(self, mocks):
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_unknown()
        for c in mock_buzzer._beep.call_args_list:
            assert c == call(_BEEP_ON_S)

    def test_servo_is_never_called(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.alert_unknown()
        mock_servo.unlock_then_relock.assert_not_called()


class TestAlertSpoof:
    def test_calls_led_indicate_with_red(self, mocks):
        mock_led, _, _, ctrl = mocks
        ctrl.alert_spoof()
        mock_led.indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_buzzer_beeps_correct_number_of_times(self, mocks):
        _, mock_buzzer, _, ctrl = mocks
        ctrl.alert_spoof()
        assert mock_buzzer._beep.call_count == _ALERT_BEEPS

    def test_servo_is_never_called(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.alert_spoof()
        mock_servo.unlock_then_relock.assert_not_called()


class TestCleanup:
    def test_led_cleanup_is_called(self, mocks):
        mock_led, _, _, ctrl = mocks
        ctrl.cleanup()
        mock_led.cleanup.assert_called_once()

    def test_buzzer_cleanup_is_called(self, mocks):
        _, mock_buzzer, _, ctrl = mocks
        ctrl.cleanup()
        mock_buzzer.cleanup.assert_called_once()

    def test_servo_cleanup_is_called(self, mocks):
        _, _, mock_servo, ctrl = mocks
        ctrl.cleanup()
        mock_servo.cleanup.assert_called_once()
