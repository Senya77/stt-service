"""FastAPI-приложение с OpenAI-compatible API для распознавания."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.gigaam_ctc.api.http.routes import health_router, router
from src.gigaam_ctc.core.recognition_service import RecognitionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет жизненным циклом приложения: загрузка модели при старте.

    Создаёт RecognitionService и сохраняет его
    в app.state.recognition_service.

    Args:
        app: Экземпляр FastAPI, для которого настраивается lifespan.

    Yields:
        None
    """
    logger.info("Loading recognition model for OpenAI API...")
    app.state.recognition_service = RecognitionService()
    logger.info("OpenAI API ready")
    yield


app = FastAPI(title="STT OpenAI-compatible API", lifespan=lifespan)
app.include_router(health_router)
app.include_router(router)
