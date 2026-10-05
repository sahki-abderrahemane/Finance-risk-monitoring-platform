"""Sentinel-AI NLP Engine API."""

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
logger = logging.getLogger("nlp-engine-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading NLP sentiment model...")
    load_models()
    logger.info("NLP model loaded. model_loaded=%s", model_loaded)
    yield
    logger.info("Shutting down NLP engine API")


app = FastAPI(
    title="Sentinel-AI NLP Engine API",
    version="0.1.0",
    description="HTTP API wrapping the Sentinel-AI NLP sentiment engine.",
    lifespan=lifespan,
)

app.include_router(router)
