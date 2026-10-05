FROM python:3.13-slim AS base

RUN groupadd --gid 1000 sentinel && \
    useradd --uid 1000 --gid sentinel --create-home sentinel

WORKDIR /app

COPY --chown=sentinel:sentinel app.py .

USER sentinel

EXPOSE 5001

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
