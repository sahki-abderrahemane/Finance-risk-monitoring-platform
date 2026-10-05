FROM python:3.13-slim AS base

RUN groupadd --gid 1000 sentinel && \
    useradd --uid 1000 --gid sentinel --create-home sentinel

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "pydantic>=2.10,<3.0" \
    "numpy>=2.1,<3.0" \
    "torch>=2.1,<3.0" \
    "torchvision>=0.20,<1.0" \
    "Pillow>=10.0,<11.0"

COPY vision-engine/src/ /app/vision_engine/
COPY vision-engine-api/ /app/

USER sentinel

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
