"""Асинхронная загрузка аудиозаписей из объектного хранилища S3."""

import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from os import getenv

from aioboto3 import Session
from botocore.client import Config

logger = logging.getLogger(__name__)


class AudioS3Storage:
    """Клиент для чтения аудиофайлов из S3 bucket.

    Параметры подключения берутся из переменных окружения.
    """

    def __init__(self):
        """Создаёт aioboto3-сессию и фабрику async context manager для bucket."""
        self._session = Session()

        @asynccontextmanager
        async def _get_s3_bucket():
            """Открывает асинхронное подключение к S3 bucket из окружения.

            Yields:
                Объект aioboto3 Bucket, готовый к загрузке аудио.
            """
            async with self._session.resource(
                service_name="s3",
                endpoint_url=getenv("URL"),
                aws_access_key_id=getenv("ACCESS_KEY"),
                aws_secret_access_key=getenv("SECRET_ACCESS"),
                config=Config(
                    connect_timeout=getenv("CONNECTION_TIMEOUT", 1),
                    read_timeout=getenv("READ_TIMEOUT", 5),
                    retries={"max_attempts": getenv("RETRY_MAX", 3)},
                ),
            ) as s3:
                yield await s3.Bucket(getenv("BUCKET"))

        self._get_bucket = _get_s3_bucket

    async def receive(self, audio_key: str) -> bytes:
        """Загружает аудио из S3 по ключу и возвращает его содержимое.

        Args:
            audio_key: Путь объекта в bucket.

        Returns:
            Байты загруженного аудиофайла.
        """
        buffer = BytesIO()
        start_time = time.perf_counter()

        async with self._get_bucket() as bucket:
            await bucket.download_fileobj(Fileobj=buffer, Key=audio_key)

        elapsed_time = time.perf_counter() - start_time
        file_size = buffer.tell()

        logger.info(
            f"Файл '{audio_key}' успешно загружен. Размер: {file_size} байт, время: {elapsed_time:.3f} с."
        )

        buffer.seek(0)
        return buffer.read()
