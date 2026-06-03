import time
import Jetson.GPIO as GPIO

GREEN_LED_PIN: int = 7
RED_LED_PIN: int = 11
GREEN_LED_HOLD_S: float = 3.0
RED_LED_HOLD_S: float = 2.0


class LED:
    """GRANT/DENY indicator LEDs on the Jetson 40-pin header."""

    def __init__(self, green_pin: int = GREEN_LED_PIN, red_pin: int = RED_LED_PIN) -> None:
        self.green_pin = green_pin
        self.red_pin = red_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.green_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.red_pin, GPIO.OUT, initial=GPIO.LOW)

    def indicate(self, success: bool, duration: float | None = None) -> None:
        if duration is None:
            duration = GREEN_LED_HOLD_S if success else RED_LED_HOLD_S
        pin = self.green_pin if success else self.red_pin
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)

    def cleanup(self) -> None:
        GPIO.output(self.green_pin, GPIO.LOW)
        GPIO.output(self.red_pin, GPIO.LOW)
        GPIO.cleanup([self.green_pin, self.red_pin])