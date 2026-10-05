"""Sentinel-AI Risk Engine API."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from dependencies import load_models, models_loaded
from routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("risk-engine-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading risk engine models...")
    load_models()
    logger.info("Risk engine models loaded. models_loaded=%s", models_loaded)
    yield
    logger.info("Shutting down risk engine API")


app = FastAPI(
    title="Sentinel-AI Risk Engine API",
    version="0.1.0",
    description="HTTP API wrapping the Sentinel-AI risk scoring engine.",
    lifespan=lifespan,
)

app.include_router(router)
