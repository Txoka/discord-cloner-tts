FROM python:3.12-slim AS builder

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu129 -r requirements.txt

FROM python:3.12-slim AS runtime

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"
ENV NUMBA_CACHE_DIR="/app/.numba_cache"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

RUN groupadd -g "${GROUP_ID}" appuser \
    && useradd -u "${USER_ID}" -g "${GROUP_ID}" -m -s /bin/bash appuser \
    && mkdir -p /app/.numba_cache /app/model /app/voices \
    && chown -R appuser:appuser /app

USER appuser

COPY app ./app

CMD ["python", "-m", "app.main"]
