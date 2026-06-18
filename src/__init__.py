#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>, <henrytsai>
# Tatung University — I4210 AI實務專題
"""Smart Access Control — hardware driver package.

Exposes the four sensor/actuator modules used by the access-control
pipeline:

    LED        — green/red indicator LEDs (BOARD pins 7, 11)
    Buzzer     — piezo alert buzzer (BOARD pin 29)
    Servo      — SG90 door-latch servo (BOARD pin 33, PWM5)
    HcSr04     — HC-SR04 ultrasonic distance sensor (BOARD pins 31, 15)
"""
