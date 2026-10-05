"""Sentinel-AI Vision Engine API."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from dependencies import load_models, model_loaded
from routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("vision-engine-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading vision CNN model...")
    load_models()
    logger.info("Vision model loaded. model_loaded=%s", model_loaded)
    yield
    logger.info("Shutting down vision engine API")


app = FastAPI(
    title="Sentinel-AI Vision Engine API",
    version="0.1.0",
    description="HTTP API wrapping the Sentinel-AI chart classification vision engine.",
    lifespan=lifespan,
)

app.include_router(router)
