"""Точка входа для OpenAI-compatible HTTP API.

Принимает произвольные аудиофайлы, отдаёт JSON/text как OpenAI Whisper API.
"""

import logging.config

import uvicorn

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.logger import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)


def main() -> None:
    """Запускает HTTP-сервер OpenAI-compatible API."""
    uvicorn.run(
        "src.gigaam_ctc.api.http.app:app",
        host="0.0.0.0",
        port=settings.HTTP_SERVER_PORT,
    )


if __name__ == "__main__":
    main()
