"""Точка входа для синхронного распознавания речи.

Отдельный entrypoint для деплоя только синхронного-инстанса без потокового RPC.
"""

import asyncio
import logging.config

from src.gigaam_ctc.api.grpc.registry import register_servicer
from src.gigaam_ctc.api.grpc.server import run_grpc_server
from src.gigaam_ctc.api.grpc.servicer import SpeechToTextServicer
from src.gigaam_ctc.logger import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)


async def serve():
    """Запускает синхронный gRPC-сервер."""
    await run_grpc_server(
        servicer_factory=SpeechToTextServicer,
        register_fn=register_servicer,
        service_label="batch",
    )


if __name__ == "__main__":
    asyncio.run(serve())
