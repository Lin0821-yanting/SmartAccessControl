#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題
"""tests/test_sensors.py — unit tests for LED, Servo, Buzzer, and HC_SR04.

All GPIO calls are mocked so these tests run on any x86 CI runner
without a real Jetson — no hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch
from tests.conftest import _gpio_mock

import pytest

# Now safe to import the modules under test
from src.led import GREEN_LED_HOLD_S, LED, RED_LED_HOLD_S  # noqa: E402
from src.buzzer import BEEP_OFF_S, BEEP_ON_S, LONG_BEEP_S, Buzzer  # noqa: E402
from src.servo import DUTY_LOCKED, DUTY_UNLOCKED, UNLOCK_HOLD_S, Servo  # noqa: E402
from src.hc_sr04 import (  # noqa: E402
    APPROACH_THRESHOLD_CM,
    HC_SR04,
    POLL_INTERVAL_S,
    SPEED_OF_SOUND_CM_PER_S,
)


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_gpio():
    """Reset the GPIO mock between every test to avoid state bleed."""
    _gpio_mock.reset_mock()
    yield


# ===========================================================================
# LED tests  (≥ 6)
# ===========================================================================


class TestLED:
    """Tests for the LED indicator class."""

    def test_init_sets_both_pins_as_output(self) -> None:
        """Constructor must configure both green and red pins as OUT."""
        led = LED(green_pin=7, red_pin=11)
        setup_calls = [c for c in _gpio_mock.setup.call_args_list]
        pins_set = {c.args[0] for c in setup_calls}
        assert 7 in pins_set
        assert 11 in pins_set
        assert led.green_pin == 7
        assert led.red_pin == 11

    def test_indicate_success_lights_green_pin(self) -> None:
        """indicate(True) must drive the green pin HIGH then LOW."""
        led = LED(green_pin=7, red_pin=11)
        _gpio_mock.reset_mock()
        with patch("time.sleep"):
            led.indicate(success=True)
        high_call = call(7, _gpio_mock.HIGH)
        low_call = call(7, _gpio_mock.LOW)
        output_calls = _gpio_mock.output.call_args_list
        assert high_call in output_calls
        assert low_call in output_calls

    def test_indicate_failure_lights_red_pin(self) -> None:
        """indicate(False) must drive the red pin HIGH then LOW."""
        led = LED(green_pin=7, red_pin=11)
        _gpio_mock.reset_mock()
        with patch("time.sleep"):
            led.indicate(success=False)
        high_call = call(11, _gpio_mock.HIGH)
        low_call = call(11, _gpio_mock.LOW)
        output_calls = _gpio_mock.output.call_args_list
        assert high_call in output_calls
        assert low_call in output_calls

    @pytest.mark.parametrize("success,expected_duration", [
        (True, GREEN_LED_HOLD_S),
        (False, RED_LED_HOLD_S),
    ])
    def test_indicate_default_duration(self, success: bool, expected_duration: float) -> None:
        """indicate() without explicit duration uses the module-level constant."""
        led = LED(green_pin=7, red_pin=11)
        with patch("time.sleep") as mock_sleep:
            led.indicate(success=success)
        mock_sleep.assert_called_once_with(expected_duration)

    def test_indicate_custom_duration_overrides_default(self) -> None:
        """A caller-supplied duration must be forwarded to time.sleep."""
        led = LED(green_pin=7, red_pin=11)
        custom = 9.9
        with patch("time.sleep") as mock_sleep:
            led.indicate(success=True, duration=custom)
        mock_sleep.assert_called_once_with(custom)

    def test_cleanup_drives_both_pins_low_then_releases(self) -> None:
        """cleanup() must set both pins LOW and call GPIO.cleanup."""
        led = LED(green_pin=7, red_pin=11)
        _gpio_mock.reset_mock()
        led.cleanup()
        output_calls = _gpio_mock.output.call_args_list
        assert call(7, _gpio_mock.LOW) in output_calls
        assert call(11, _gpio_mock.LOW) in output_calls
        _gpio_mock.cleanup.assert_called_once()

    def test_indicate_does_not_touch_opposite_pin(self) -> None:
        """indicate(True) must never toggle the red pin and vice-versa."""
        led = LED(green_pin=7, red_pin=11)
        _gpio_mock.reset_mock()
        with patch("time.sleep"):
            led.indicate(success=True)
        for c in _gpio_mock.output.call_args_list:
            assert c.args[0] != 11, "Red pin must not be touched on success"


# ===========================================================================
# Buzzer tests  (≥ 6)
# ===========================================================================


class TestBuzzer:
    """Tests for the Buzzer piezo class."""

    def test_init_configures_pin_as_output(self) -> None:
        """Constructor must call GPIO.setup with OUT on the buzzer pin."""
        Buzzer(pin=29)
        setup_pins = {c.args[0] for c in _gpio_mock.setup.call_args_list}
        assert 29 in setup_pins

    def test_indicate_success_produces_single_beep(self) -> None:
        """indicate(True) must produce exactly one HIGH→LOW transition."""
        buz = Buzzer(pin=29)
        _gpio_mock.reset_mock()
        with patch("time.sleep"):
            buz.indicate(success=True)
        output_calls = _gpio_mock.output.call_args_list
        high_count = sum(1 for c in output_calls if c.args[1] == _gpio_mock.HIGH)
        assert high_count == 1

    def test_indicate_failure_produces_two_beeps(self) -> None:
        """indicate(False) must produce exactly two HIGH→LOW transitions."""
        buz = Buzzer(pin=29)
        _gpio_mock.reset_mock()
        with patch("time.sleep"):
            buz.indicate(success=False)
        output_calls = _gpio_mock.output.call_args_list
        high_count = sum(1 for c in output_calls if c.args[1] == _gpio_mock.HIGH)
        assert high_count == 2

    def test_indicate_failure_sleep_sequence(self) -> None:
        """indicate(False) must sleep: BEEP_ON_S → BEEP_OFF_S → LONG_BEEP_S."""
        buz = Buzzer(pin=29)
        with patch("time.sleep") as mock_sleep:
            buz.indicate(success=False)
        durations = [c.args[0] for c in mock_sleep.call_args_list]
        assert durations == [BEEP_ON_S, BEEP_OFF_S, LONG_BEEP_S]

    def test_indicate_success_sleep_duration(self) -> None:
        """indicate(True) must sleep for BEEP_ON_S only."""
        buz = Buzzer(pin=29)
        with patch("time.sleep") as mock_sleep:
            buz.indicate(success=True)
        assert mock_sleep.call_args_list == [call(BEEP_ON_S)]

    def test_cleanup_drives_pin_low_and_releases(self) -> None:
        """cleanup() must drive pin LOW then call GPIO.cleanup."""
        buz = Buzzer(pin=29)
        _gpio_mock.reset_mock()
        buz.cleanup()
        assert call(29, _gpio_mock.LOW) in _gpio_mock.output.call_args_list
        _gpio_mock.cleanup.assert_called_once()

    def test_custom_pin_forwarded_to_gpio(self) -> None:
        """A non-default pin must be passed through to every GPIO call."""
        Buzzer(pin=36)
        setup_pins = {c.args[0] for c in _gpio_mock.setup.call_args_list}
        assert 36 in setup_pins


# ===========================================================================
# Servo tests  (≥ 6)
# ===========================================================================


class TestServo:
    """Tests for the Servo SG90 class."""

    def test_init_starts_pwm_at_locked_duty(self) -> None:
        """Constructor must start PWM and set the locked duty cycle."""
        _gpio_mock.PWM.return_value = MagicMock()
        Servo(pin=33)
        _gpio_mock.PWM.assert_called_once_with(33, 50)
        _gpio_mock.PWM.return_value.start.assert_called_once_with(DUTY_LOCKED)

    def test_set_lock_true_applies_locked_duty(self) -> None:
        """set_lock(True) must call ChangeDutyCycle with DUTY_LOCKED."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        servo.set_lock(locked=True)
        mock_pwm.ChangeDutyCycle.assert_called_with(DUTY_LOCKED)

    def test_set_lock_false_applies_unlocked_duty(self) -> None:
        """set_lock(False) must call ChangeDutyCycle with DUTY_UNLOCKED."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        servo.set_lock(locked=False)
        mock_pwm.ChangeDutyCycle.assert_called_with(DUTY_UNLOCKED)

    @pytest.mark.parametrize("duty", [2.5, 5.0, 7.5, 10.0, 12.5])
    def test_set_angle_passes_duty_cycle_through(self, duty: float) -> None:
        """set_angle() must forward any duty_cycle value to ChangeDutyCycle."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        mock_pwm.reset_mock()
        servo.set_angle(duty)
        mock_pwm.ChangeDutyCycle.assert_called_once_with(duty)

    def test_unlock_then_relock_sequence(self) -> None:
        """unlock_then_relock() must unlock → sleep → relock in order."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        mock_pwm.reset_mock()
        with patch("time.sleep") as mock_sleep:
            servo.unlock_then_relock()
        calls = mock_pwm.ChangeDutyCycle.call_args_list
        assert calls[0] == call(DUTY_UNLOCKED)
        assert calls[1] == call(DUTY_LOCKED)
        mock_sleep.assert_called_once_with(UNLOCK_HOLD_S)

    def test_cleanup_stops_pwm_and_releases_gpio(self) -> None:
        """cleanup() must stop PWM, then call GPIO.cleanup on the pin."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        _gpio_mock.reset_mock()
        mock_pwm.reset_mock()
        with patch("time.sleep"):
            servo.cleanup()
        mock_pwm.stop.assert_called_once()
        _gpio_mock.cleanup.assert_called_once()

    def test_cleanup_relocks_before_stop(self) -> None:
        """cleanup() must restore DUTY_LOCKED before stopping the PWM."""
        mock_pwm = MagicMock()
        _gpio_mock.PWM.return_value = mock_pwm
        servo = Servo(pin=33)
        mock_pwm.reset_mock()
        with patch("time.sleep"):
            servo.cleanup()
        # First call on the mock after reset must be locking, then stop
        first_duty_call = mock_pwm.ChangeDutyCycle.call_args_list[0]
        assert first_duty_call == call(DUTY_LOCKED)


# ===========================================================================
# HC_SR04 tests  (≥ 6)
# ===========================================================================


class TestHCSR04:
    """Tests for the HC-SR04 ultrasonic distance sensor."""

    def test_init_configures_trigger_as_out_and_echo_as_in(self) -> None:
        """Constructor must set trigger=OUT and echo=IN."""
        HC_SR04(trigger_pin=31, echo_pin=15)
        calls = {c.args[0]: c.args[1] for c in _gpio_mock.setup.call_args_list}
        assert calls.get(31) == _gpio_mock.OUT
        assert calls.get(15) == _gpio_mock.IN

    def test_measure_distance_returns_inf_on_echo_timeout(self) -> None:
        """_measure_distance() must return inf when echo never goes HIGH."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15)
        # Echo pin 一直是 LOW → while LOW loop 一直跑 → timeout
        # monotonic 呼叫順序：
        #   [1] deadline = monotonic() + ECHO_TIMEOUT_S  → 0.0
        #   [2] if monotonic() > deadline (loop 第一次)  → 0.0，不 timeout
        #   [3] if monotonic() > deadline (loop 第二次)  → 99.0，timeout → return inf
        _gpio_mock.input.return_value = _gpio_mock.LOW
        with patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]):
            dist = sensor._measure_distance()
        assert dist == float("inf")

    def test_measure_distance_calculation(self) -> None:
        """_measure_distance() must compute distance using the speed-of-sound formula."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15)
        echo_duration = 0.001  # 1 ms → 17.15 cm
        expected = echo_duration * SPEED_OF_SOUND_CM_PER_S / 2.0

        # 對照實際 code 的執行順序：
        #
        # deadline = monotonic()          → [1] 0.0  (deadline = 0.0 + 0.05)
        # while input(echo) == LOW:       → input[0] = LOW，進入 loop
        #   if monotonic() > deadline     → [2] 0.0，不 timeout
        # while input(echo) == LOW:       → input[1] = HIGH，跳出 loop
        # t_start = monotonic()           → [3] 0.1
        # while input(echo) == HIGH:      → input[2] = HIGH，進入 loop
        #   if monotonic() > deadline     → [4] 0.0，不 timeout
        # while input(echo) == HIGH:      → input[3] = LOW，跳出 loop
        # return (monotonic() - t_start)  → [5] 0.101，算出 0.001 * 34300 / 2 = 17.15

        t_start = 0.1
        t_end = t_start + echo_duration  # 0.101

        _gpio_mock.input.side_effect = [
            _gpio_mock.LOW,   # input[0]: while LOW → 還在等，繼續 loop
            _gpio_mock.HIGH,  # input[1]: while LOW → pin 變 HIGH，跳出
            _gpio_mock.HIGH,  # input[2]: while HIGH → pin 還是 HIGH，繼續 loop
            _gpio_mock.LOW,   # input[3]: while HIGH → pin 變 LOW，跳出
        ]
        with patch("time.monotonic", side_effect=[
            0.0,     # [1] deadline 設定
            0.0,     # [2] while LOW loop 的 timeout check（不 timeout）
            t_start, # [3] t_start 記錄
            0.0,     # [4] while HIGH loop 的 timeout check（不 timeout）
            t_end,   # [5] 最後計算用的 monotonic()
        ]):
            dist = sensor._measure_distance()
        assert abs(dist - expected) < 0.01

    def test_confirmed_near_requires_majority_reads(self) -> None:
        """_confirmed_near() must return True only when ≥2 of 3 reads are within threshold."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15, threshold_cm=60.0)
        near = 30.0
        far = 100.0
        # 2 near, 1 far → majority → True
        with patch.object(sensor, "_measure_distance", side_effect=[near, near, far]):
            assert sensor._confirmed_near() is True

    def test_confirmed_near_false_when_minority_readings(self) -> None:
        """_confirmed_near() must return False when only 1 of 3 reads is near."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15, threshold_cm=60.0)
        with patch.object(sensor, "_measure_distance", side_effect=[30.0, 100.0, 100.0]):
            assert sensor._confirmed_near() is False

    def test_is_someone_near_delegates_to_confirmed_near(self) -> None:
        """is_someone_near() is the public API for _confirmed_near()."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15)
        with patch.object(sensor, "_confirmed_near", return_value=True) as mock_cn:
            result = sensor.is_someone_near()
        mock_cn.assert_called_once()
        assert result is True

    def test_wait_for_person_polls_until_near(self) -> None:
        """wait_for_person() must loop until _confirmed_near() returns True."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15)
        # False twice, then True on third call
        with patch.object(sensor, "_confirmed_near", side_effect=[False, False, True]):
            with patch("time.sleep") as mock_sleep:
                sensor.wait_for_person()
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(POLL_INTERVAL_S)

    def test_custom_threshold_passed_into_instance(self) -> None:
        """A non-default threshold_cm must be stored and honoured."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15, threshold_cm=30.0)
        assert sensor.threshold_cm == 30.0
        with patch.object(sensor, "_measure_distance", side_effect=[29.0, 29.0, 29.0]):
            assert sensor._confirmed_near() is True

    def test_cleanup_releases_both_pins(self) -> None:
        """cleanup() must call GPIO.cleanup with both trigger and echo pins."""
        sensor = HC_SR04(trigger_pin=31, echo_pin=15)
        _gpio_mock.reset_mock()
        sensor.cleanup()
        _gpio_mock.cleanup.assert_called_once()
        released = set(_gpio_mock.cleanup.call_args.args[0])
        assert 31 in released
        assert 15 in released

    def test_default_threshold_constant(self) -> None:
        """Default threshold_cm must equal the module-level APPROACH_THRESHOLD_CM."""
        sensor = HC_SR04()
        assert sensor.threshold_cm == APPROACH_THRESHOLD_CM
