"""Общий запуск gRPC-сервера STT."""

import asyncio
import logging
import signal
from collections.abc import Callable
from typing import Any

import grpc

from src.gigaam_ctc.config import settings

logger = logging.getLogger(__name__)

SERVER_OPTIONS = [
    ("grpc.max_receive_message_length", 150 * 1024 * 1024),
    ("grpc.max_send_message_length", 150 * 1024 * 1024),
]


async def run_grpc_server(
    *,
    servicer_factory: Callable[[], Any],
    register_fn: Callable[[grpc.aio.Server, Any], None],
    service_label: str,
) -> None:
    """Создаёт gRPC-сервер, регистрирует servicer и ждёт сигнал остановки.

    Сервер слушает порт на всех интерфейсах. По SIGINT или SIGTERM
    выполняется graceful shutdown с таймаутом 5 секунд.

    Args:
        servicer_factory: Фабрика, создающая экземпляр servicer при старте.
        register_fn: Функция регистрации RPC на сервере.
        service_label: Метка режима для логов.

    Returns:
        None. Функция завершается после остановки сервера.
    """
    server = grpc.aio.server(options=SERVER_OPTIONS)
    servicer = servicer_factory()
    register_fn(server, servicer)

    port = settings.GRPC_SERVER_PORT
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Starting STT service mode={service_label} on port={port}")
    await server.start()
    logger.info(f"gRPC server started (mode={service_label}, port={port})")

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    try:
        await shutdown_event.wait()
        logger.info(f"Shutting down gRPC server (mode={service_label})...")
        await server.stop(grace=5.0)
        if hasattr(servicer, "shutdown"):
            servicer.shutdown()
        logger.info(f"Server stopped gracefully (mode={service_label}).")
    except Exception as e:
        logger.error(f"Error during shutdown (mode={service_label}): {e}")
        await server.stop(grace=0)
