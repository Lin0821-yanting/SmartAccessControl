#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""tests/conftest.py — inject Jetson.GPIO stub before any src module is imported."""

import sys
import types
from unittest.mock import MagicMock

import pytest

# 建立 mock 並注入 sys.modules（強制覆蓋）
_gpio_mock = MagicMock()
_gpio_mock.BOARD = "BOARD"
_gpio_mock.OUT   = "OUT"
_gpio_mock.IN    = "IN"
_gpio_mock.HIGH  = 1
_gpio_mock.LOW   = 0

_jetson_pkg = types.ModuleType("Jetson")
_jetson_pkg.GPIO = _gpio_mock
sys.modules["Jetson"]      = _jetson_pkg
sys.modules["Jetson.GPIO"] = _gpio_mock


@pytest.fixture(autouse=True)
def reset_gpio():
    """每個測試前後重置 GPIO mock，避免狀態殘留。"""
    _gpio_mock.reset_mock()
    yield
    _gpio_mock.reset_mock()


@pytest.fixture()
def gpio_mock():
    """把 _gpio_mock 暴露給需要直接 assert GPIO call 的測試。"""
    return _gpio_mock