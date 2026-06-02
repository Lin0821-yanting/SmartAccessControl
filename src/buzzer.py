import time
import Jetson.GPIO as GPIO

BUZZER_PIN: int = 22
BEEP_ON_S: float = 0.2
BEEP_OFF_S: float = 0.1
LONG_BEEP_S: float = 0.6


class Buzzer:
    """Piezo buzzer on Jetson pin 22."""

    def __init__(self, pin: int = BUZZER_PIN) -> None:
        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def _beep(self, duration: float) -> None:
        GPIO.output(self.pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.pin, GPIO.LOW)

    def indicate(self, success: bool) -> None:
        if success:
            self._beep(BEEP_ON_S)
        else:
            self._beep(BEEP_ON_S)
            time.sleep(BEEP_OFF_S)
            self._beep(LONG_BEEP_S)

    def cleanup(self) -> None:
        GPIO.output(self.pin, GPIO.LOW)
        GPIO.cleanup([self.pin])