FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MBAL_PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY mbal_*.py ./
COPY mbal_pipeline ./mbal_pipeline
COPY mbal_experiments ./mbal_experiments
COPY ops ./ops
COPY release_gate ./release_gate
COPY research ./research
COPY docs ./docs
COPY signals ./signals
COPY tests ./tests

ARG EXTRAS=test
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[${EXTRAS}]"

CMD ["python", "ops/run_tests.py"]
