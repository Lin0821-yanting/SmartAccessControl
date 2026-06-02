import time
import Jetson.GPIO as GPIO

SERVO_PIN: int = 33
SERVO_FREQ_HZ: int = 50
DUTY_LOCKED: float = 5.0
DUTY_UNLOCKED: float = 10.0
UNLOCK_HOLD_S: float = 3.0


class Servo:
    """SG90 servo motor controlling the door latch (Jetson pin 33 / PWM5)."""

    def __init__(self, pin: int = SERVO_PIN) -> None:
        self.pin = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._pwm = GPIO.PWM(self.pin, SERVO_FREQ_HZ)
        self._pwm.start(DUTY_LOCKED)

    def set_lock(self, locked: bool) -> None:
        self._pwm.ChangeDutyCycle(DUTY_LOCKED if locked else DUTY_UNLOCKED)

    def set_angle(self, duty_cycle: float) -> None:
        self._pwm.ChangeDutyCycle(duty_cycle)

    def unlock_then_relock(self) -> None:
        self._pwm.ChangeDutyCycle(DUTY_UNLOCKED)
        time.sleep(UNLOCK_HOLD_S)
        self._pwm.ChangeDutyCycle(DUTY_LOCKED)

    def cleanup(self) -> None:
        self._pwm.ChangeDutyCycle(DUTY_LOCKED)
        time.sleep(0.5)
        self._pwm.stop()
        GPIO.cleanup([self.pin])