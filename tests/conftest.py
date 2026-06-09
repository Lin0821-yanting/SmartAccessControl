"""tests/conftest.py — CI stubs for modules not available on x86.

Registers stubs for:
  - M1 AI modules (cv2, yaml, ultralytics, src.detection, etc.)
  - Jetson.GPIO (used by actuator_controller imported by orchestrator tests)
  - gpiod (used by servo imported by orchestrator tests)

Note: test_sensors.py maintains its OWN _gpio_mock and _gpiod_mock
to avoid cross-test state pollution.
"""

import sys
import types
from unittest.mock import MagicMock

# ── Jetson.GPIO (for orchestrator test imports) ────────────────────────────
_gpio_mock = MagicMock()
_gpio_mock.BOARD = "BOARD"
_gpio_mock.OUT = "OUT"
_gpio_mock.IN = "IN"
_gpio_mock.HIGH = 1
_gpio_mock.LOW = 0

_jetson_pkg = types.ModuleType("Jetson")
_jetson_pkg.GPIO = _gpio_mock
# setdefault: test_sensors.py's own stub takes precedence if loaded first
sys.modules.setdefault("Jetson", _jetson_pkg)
sys.modules.setdefault("Jetson.GPIO", _gpio_mock)

# ── gpiod ──────────────────────────────────────────────────────────────────
_gpiod_mock = MagicMock()
_gpiod_mock.LINE_REQ_DIR_OUT = 1
sys.modules.setdefault("gpiod", _gpiod_mock)

# ── M1 AI modules ──────────────────────────────────────────────────────────
for _mod in [
    "cv2",
    "yaml",
    "ultralytics",
    "src.detection",
    "src.detection.detector",
    "src.recognition",
    "src.recognition.recognizer",
    "src.antispoof",
    "src.antispoof.antispoof",
]:
    sys.modules.setdefault(_mod, MagicMock())
