import time
import Jetson.GPIO as GPIO

TRIG_PIN: int = 11
ECHO_PIN: int = 13
SPEED_OF_SOUND_CM_PER_S: float = 34300.0
TRIGGER_PULSE_S: float = 10e-6
ECHO_TIMEOUT_S: float = 0.05
POLL_INTERVAL_S: float = 0.10
APPROACH_THRESHOLD_CM: float = 60.0


class HC_SR04:
    """HC-SR04 ultrasonic distance sensor — sense gate for the AI pipeline."""

    def __init__(self, trigger_pin: int = TRIG_PIN, echo_pin: int = ECHO_PIN,
                 threshold_cm: float = APPROACH_THRESHOLD_CM) -> None:
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.threshold_cm = threshold_cm
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.trigger_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.echo_pin, GPIO.IN)

    def _measure_distance(self) -> float:
        GPIO.output(self.trigger_pin, GPIO.LOW)
        time.sleep(2e-6)
        GPIO.output(self.trigger_pin, GPIO.HIGH)
        time.sleep(TRIGGER_PULSE_S)
        GPIO.output(self.trigger_pin, GPIO.LOW)

        deadline = time.monotonic() + ECHO_TIMEOUT_S
        while GPIO.input(self.echo_pin) == GPIO.LOW:
            if time.monotonic() > deadline:
                return float("inf")
        t_start = time.monotonic()

        while GPIO.input(self.echo_pin) == GPIO.HIGH:
            if time.monotonic() > deadline:
                return float("inf")
        return (time.monotonic() - t_start) * SPEED_OF_SOUND_CM_PER_S / 2.0

    def _confirmed_near(self) -> bool:
        hits = sum(1 for _ in range(3) if self._measure_distance() <= self.threshold_cm)
        return hits >= 2

    def is_someone_near(self) -> bool:
        return self._confirmed_near()

    def wait_for_person(self) -> None:
        while not self._confirmed_near():
            time.sleep(POLL_INTERVAL_S)

    def cleanup(self) -> None:
        GPIO.cleanup([self.trigger_pin, self.echo_pin])