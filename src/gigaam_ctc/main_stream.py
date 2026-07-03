"""Точка входа для потокового распознавания речи.

Отдельный entrypoint: только StreamingRecognize, VAD и partial results.
"""

import asyncio
import logging.config

from src.gigaam_ctc.api.grpc.registry import register_stream_servicer
from src.gigaam_ctc.api.grpc.server import run_grpc_server
from src.gigaam_ctc.api.grpc.stream_servicer import StreamSpeechToTextServicer
from src.gigaam_ctc.logger import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)


async def serve():
    """Запускает stream gRPC-сервер."""
    await run_grpc_server(
        servicer_factory=StreamSpeechToTextServicer,
        register_fn=register_stream_servicer,
        service_label="stream",
    )


if __name__ == "__main__":
    asyncio.run(serve())
