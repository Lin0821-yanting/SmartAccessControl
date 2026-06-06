#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""tests/test_sensors.py — unit tests for LED, Servo, Buzzer, and HcSr04.

All GPIO calls are mocked so these tests run on any x86 CI runner
without a real Jetson — no hardware required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# GPIO stubs — use whatever is already in sys.modules (set by conftest.py
# or by this module if loaded first). This avoids double-mock issues when
# test_orchestrator.py is collected before test_sensors.py.
# ---------------------------------------------------------------------------
if "Jetson.GPIO" not in sys.modules:
    _gpio_mock = MagicMock()
    _gpio_mock.BOARD = "BOARD"
    _gpio_mock.OUT = "OUT"
    _gpio_mock.IN = "IN"
    _gpio_mock.HIGH = 1
    _gpio_mock.LOW = 0
    _jetson_pkg = types.ModuleType("Jetson")
    _jetson_pkg.GPIO = _gpio_mock
    sys.modules["Jetson"] = _jetson_pkg
    sys.modules["Jetson.GPIO"] = _gpio_mock
else:
    _gpio_mock = sys.modules["Jetson.GPIO"]
    # Ensure required attributes exist
    _gpio_mock.BOARD = "BOARD"
    _gpio_mock.OUT = "OUT"
    _gpio_mock.IN = "IN"
    _gpio_mock.HIGH = 1
    _gpio_mock.LOW = 0

if "gpiod" not in sys.modules:
    _gpiod_mock = MagicMock()
    _gpiod_mock.LINE_REQ_DIR_OUT = 1
    _gpiod_mock.LINE_REQ_DIR_IN = 2
    sys.modules["gpiod"] = _gpiod_mock
else:
    _gpiod_mock = sys.modules["gpiod"]
    _gpiod_mock.LINE_REQ_DIR_OUT = 1
    _gpiod_mock.LINE_REQ_DIR_IN = 2

# Now safe to import the modules under test
from src.led import GREEN_LED_HOLD_S, LED, RED_LED_HOLD_S  # noqa: E402
from src.buzzer import BEEP_OFF_S, BEEP_ON_S, LONG_BEEP_S, Buzzer  # noqa: E402
from src.servo import (  # noqa: E402
    PULSE_LOCKED_MS,
    PULSE_UNLOCKED_MS,
    UNLOCK_HOLD_S,
    Servo,
)
from src.hc_sr04 import (  # noqa: E402
    APPROACH_THRESHOLD_CM,
    HcSr04,
    POLL_INTERVAL_S,
    SPEED_OF_SOUND_CM_PER_S,
)


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_gpio():
    """Reset the GPIO mock between every test to avoid state bleed."""
    # reset_mock(return_value=True, side_effect=True) clears EVERYTHING
    _gpio_mock.reset_mock(return_value=True, side_effect=True)
    _gpiod_mock.reset_mock(return_value=True, side_effect=True)
    # Restore required attributes cleared by full reset
    _gpio_mock.BOARD = "BOARD"
    _gpio_mock.OUT = "OUT"
    _gpio_mock.IN = "IN"
    _gpio_mock.HIGH = 1
    _gpio_mock.LOW = 0
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
# Servo tests  (≥ 6)  — gpiod software PWM
# ===========================================================================


class TestServo:
    """Tests for the Servo SG90 class (gpiod software PWM on Yahboom board)."""

    @pytest.fixture(autouse=True)
    def mock_gpiod_and_pwm(self):
        """Mock gpiod chip/line and _SoftPWM to avoid real hardware and threads."""
        mock_chip = MagicMock()
        mock_line = MagicMock()
        mock_chip.get_line.return_value = mock_line
        mock_pwm_instance = MagicMock()

        with patch("src.servo.gpiod") as mock_gpiod_mod, \
             patch("src.servo._SoftPWM", return_value=mock_pwm_instance):
            mock_gpiod_mod.Chip.return_value = mock_chip
            mock_gpiod_mod.LINE_REQ_DIR_OUT = 1
            self.mock_chip = mock_chip
            self.mock_line = mock_line
            self.mock_pwm = mock_pwm_instance
            yield

    def test_init_starts_pwm_at_locked_pulse(self) -> None:
        """Constructor must start PWM at PULSE_LOCKED_MS (locked position)."""
        Servo()
        self.mock_pwm.start.assert_called_once_with(PULSE_LOCKED_MS)

    def test_init_requests_line_as_output(self) -> None:
        """Constructor must request the gpiod line as DIR_OUT."""
        Servo()
        self.mock_line.request.assert_called_once()

    def test_set_lock_true_applies_locked_pulse(self) -> None:
        """set_lock(True) must call set_pulse with PULSE_LOCKED_MS."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        servo.set_lock(locked=True)
        self.mock_pwm.set_pulse.assert_called_once_with(PULSE_LOCKED_MS)

    def test_set_lock_false_applies_unlocked_pulse(self) -> None:
        """set_lock(False) must call set_pulse with PULSE_UNLOCKED_MS."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        servo.set_lock(locked=False)
        self.mock_pwm.set_pulse.assert_called_once_with(PULSE_UNLOCKED_MS)

    @pytest.mark.parametrize("pulse_ms", [1.0, 1.25, 1.5, 1.75, 2.0])
    def test_set_angle_passes_pulse_through(self, pulse_ms: float) -> None:
        """set_angle() must forward pulse_ms value to _SoftPWM.set_pulse."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        servo.set_angle(pulse_ms)
        self.mock_pwm.set_pulse.assert_called_once_with(pulse_ms)

    def test_unlock_then_relock_sequence(self) -> None:
        """unlock_then_relock() must unlock → sleep → relock in order."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        with patch("time.sleep") as mock_sleep:
            servo.unlock_then_relock()
        calls = self.mock_pwm.set_pulse.call_args_list
        assert calls[0] == call(PULSE_UNLOCKED_MS)
        assert calls[1] == call(PULSE_LOCKED_MS)
        mock_sleep.assert_called_once_with(UNLOCK_HOLD_S)

    def test_cleanup_stops_pwm_and_releases_line(self) -> None:
        """cleanup() must stop PWM, release the gpiod line, and close the chip."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        with patch("time.sleep"):
            servo.cleanup()
        self.mock_pwm.stop.assert_called_once()
        self.mock_line.release.assert_called_once()
        self.mock_chip.close.assert_called_once()

    def test_cleanup_relocks_before_stop(self) -> None:
        """cleanup() must set PULSE_LOCKED_MS before stopping the PWM."""
        servo = Servo()
        self.mock_pwm.reset_mock()
        with patch("time.sleep"):
            servo.cleanup()
        first_pulse_call = self.mock_pwm.set_pulse.call_args_list[0]
        assert first_pulse_call == call(PULSE_LOCKED_MS)


# ===========================================================================
# HcSr04 tests  (≥ 9)
# ===========================================================================


class TestHCSR04:
    """Tests for the HC-SR04 ultrasonic distance sensor."""

    def test_init_configures_trigger_as_out_and_echo_as_in(self) -> None:
        """Constructor must set trigger=OUT and echo=IN."""
        with patch("time.sleep"):
            HcSr04(trigger_pin=31, echo_pin=15)
        calls = {c.args[0]: c.args[1] for c in _gpio_mock.setup.call_args_list}
        assert calls.get(31) == _gpio_mock.OUT
        assert calls.get(15) == _gpio_mock.IN

    def test_measure_distance_returns_inf_on_echo_timeout(self) -> None:
        """_measure_distance() must return inf when echo never goes HIGH.

        Call order with stuck-ECHO guard:
          input[0] = LOW  → guard: not stuck
          input[1] = LOW  → while LOW loop: enters (stays LOW until timeout)
          monotonic[0]    → deadline
          monotonic[1]    → timeout check 1 (not expired)
          input[2] = LOW  → while LOW loop: still LOW
          monotonic[2]    → timeout check 2 (expired) → return inf
        """
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        _gpio_mock.input.return_value = _gpio_mock.LOW
        with patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]):
            dist = sensor._measure_distance()
        assert dist == float("inf")

    def test_measure_distance_calculation(self) -> None:
        """_measure_distance() must compute distance using the speed-of-sound formula.

        Call order with stuck-ECHO guard:
          input[0] = LOW   → guard: not stuck
          deadline = monotonic[0] = 0.0  (deadline = 0.05)
          input[1] = LOW   → while LOW: enters loop
          monotonic[1] = 0.0  → timeout check: not expired
          input[2] = HIGH  → while LOW: exits loop
          t_start = monotonic[2] = 0.1
          input[3] = HIGH  → while HIGH: enters loop
          monotonic[3] = 0.0  → timeout check: not expired
          input[4] = LOW   → while HIGH: exits loop
          return (monotonic[4] - t_start) * factor
        """
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)

        echo_duration = 0.001   # 1 ms → 17.15 cm
        expected = echo_duration * SPEED_OF_SOUND_CM_PER_S / 2.0
        t_start = 0.1
        t_end = t_start + echo_duration  # 0.101

        _gpio_mock.input.side_effect = [
            _gpio_mock.LOW,   # input[0]: stuck guard → not stuck
            _gpio_mock.LOW,   # input[1]: while LOW → enters loop
            _gpio_mock.HIGH,  # input[2]: while LOW → exits loop
            _gpio_mock.HIGH,  # input[3]: while HIGH → enters loop
            _gpio_mock.LOW,   # input[4]: while HIGH → exits loop
        ]
        with patch("time.monotonic", side_effect=[
            0.0,      # [0] deadline = 0.0 + ECHO_TIMEOUT_S
            0.0,      # [1] while LOW timeout check (not expired)
            t_start,  # [2] t_start = 0.1
            0.0,      # [3] while HIGH timeout check (not expired)
            t_end,    # [4] final distance calculation
        ]):
            dist = sensor._measure_distance()

        assert abs(dist - expected) < 0.01

    def test_measure_distance_stuck_echo_triggers_recovery(self) -> None:
        """When ECHO is HIGH at measurement start, recovery must be attempted."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)

        # Patch GPIO.input to return HIGH (stuck), and mock _recover_stuck_echo
        with patch.object(sensor, "_recover_stuck_echo", return_value=False) as mock_rec,              patch("src.hc_sr04.GPIO") as mock_gpio:
            mock_gpio.HIGH = 1
            mock_gpio.LOW = 0
            mock_gpio.input.return_value = mock_gpio.HIGH  # ECHO stuck HIGH
            dist = sensor._measure_distance()

        mock_rec.assert_called_once()
        assert dist == float("inf")

    def test_recover_stuck_echo_returns_true_when_echo_clears(self) -> None:
        """_recover_stuck_echo() must return True when ECHO drops LOW after hold."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)

        # After TRIG-LOW hold, ECHO returns LOW → recovered
        with patch("src.hc_sr04.GPIO") as mock_gpio, patch("time.sleep"):
            mock_gpio.HIGH = 1
            mock_gpio.LOW = 0
            mock_gpio.input.return_value = mock_gpio.LOW
            result = sensor._recover_stuck_echo()

        assert result is True

    def test_recover_stuck_echo_returns_false_when_echo_stays_high(self) -> None:
        """_recover_stuck_echo() must return False when ECHO stays HIGH."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)

        # After TRIG-LOW hold, ECHO still HIGH → power-cycle needed
        with patch("src.hc_sr04.GPIO") as mock_gpio, patch("time.sleep"):
            mock_gpio.HIGH = 1
            mock_gpio.LOW = 0
            mock_gpio.input.return_value = mock_gpio.HIGH
            result = sensor._recover_stuck_echo()

        assert result is False

    def test_confirmed_near_requires_majority_reads(self) -> None:
        """_confirmed_near() must return True only when ≥2 of 3 reads are within threshold."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15, threshold_cm=60.0)
        with patch.object(sensor, "_measure_distance", side_effect=[30.0, 30.0, 100.0]):
            assert sensor._confirmed_near() is True

    def test_confirmed_near_false_when_minority_readings(self) -> None:
        """_confirmed_near() must return False when only 1 of 3 reads is near."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15, threshold_cm=60.0)
        with patch.object(sensor, "_measure_distance", side_effect=[30.0, 100.0, 100.0]):
            assert sensor._confirmed_near() is False

    def test_is_someone_near_delegates_to_confirmed_near(self) -> None:
        """is_someone_near() is the public API for _confirmed_near()."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        with patch.object(sensor, "_confirmed_near", return_value=True) as mock_cn:
            result = sensor.is_someone_near()
        mock_cn.assert_called_once()
        assert result is True

    def test_wait_for_person_polls_until_near(self) -> None:
        """wait_for_person() must loop until _confirmed_near() returns True."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        with patch.object(sensor, "_confirmed_near", side_effect=[False, False, True]):
            with patch("time.sleep") as mock_sleep:
                sensor.wait_for_person()
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(POLL_INTERVAL_S)

    def test_custom_threshold_passed_into_instance(self) -> None:
        """A non-default threshold_cm must be stored and honoured."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15, threshold_cm=30.0)
        assert sensor.threshold_cm == 30.0
        with patch.object(sensor, "_measure_distance", side_effect=[29.0, 29.0, 29.0]):
            assert sensor._confirmed_near() is True

    def test_cleanup_releases_both_pins(self) -> None:
        """cleanup() must call GPIO.cleanup with both trigger and echo pins."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        _gpio_mock.reset_mock()
        sensor.cleanup()
        _gpio_mock.cleanup.assert_called_once()
        released = set(_gpio_mock.cleanup.call_args.args[0])
        assert 31 in released
        assert 15 in released

    def test_default_threshold_constant(self) -> None:
        """Default threshold_cm must equal the module-level APPROACH_THRESHOLD_CM."""
        with patch("time.sleep"):
            sensor = HcSr04()
        assert sensor.threshold_cm == APPROACH_THRESHOLD_CM

    def test_measure_distance_public_returns_none_on_inf(self) -> None:
        """measure_distance() must return None when _measure_distance returns inf."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        with patch.object(sensor, "_measure_distance", return_value=float("inf")):
            assert sensor.measure_distance() is None

    def test_measure_distance_public_returns_value_on_success(self) -> None:
        """measure_distance() must return the distance value when measurement succeeds."""
        with patch("time.sleep"):
            sensor = HcSr04(trigger_pin=31, echo_pin=15)
        with patch.object(sensor, "_measure_distance", return_value=30.0):
            assert sensor.measure_distance() == pytest.approx(30.0)