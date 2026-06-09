# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt* ./
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    "opencv-python-headless==4.10.0.84" \
    "paho-mqtt==2.1.0" \
    gpiod \
    onnxruntime \
    pyyaml \
    numpy

COPY . .

CMD ["python", "-m", "pytest", "tests/", "--ignore=tests/integration", "-v"]
