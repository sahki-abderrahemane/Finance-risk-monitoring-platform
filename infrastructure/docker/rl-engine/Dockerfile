# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system sentinel \
    && useradd --system \
       --gid sentinel \
       --create-home \
       --home-dir /home/sentinel \
       sentinel

RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

COPY services/rl-engine/pyproject.toml /app/service/pyproject.toml
COPY services/rl-engine/app /app/service/app

WORKDIR /app/service

RUN python -m pip install --no-cache-dir .

RUN find /root/.cache -type f -delete 2>/dev/null || true

RUN chown -R sentinel:sentinel /app /home/sentinel

USER sentinel

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
