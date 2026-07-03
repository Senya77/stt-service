"""Сервис пакетного распознавания речи для gRPC и HTTP API."""

import logging

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.dtos import TranscriptionResult
from src.gigaam_ctc.stt_model.audio_utils import pcm_int16_duration_s
from src.gigaam_ctc.stt_model.factory import get_gigaam_model

logger = logging.getLogger(__name__)


class RecognitionService:
    """Обертка над моделью для синхронного распознавания PCM-аудио.

    Выбирает короткий или длинный путь распознавания по порогу
    LONG_AUDIO_THRESHOLD_S: короткие записи обрабатываются целиком,
    длинные — по чанкам через transcribe_long_audio.
    """

    def __init__(self) -> None:
        """Загружает модель GigaAM через фабрику get_gigaam_model."""
        self._model = get_gigaam_model()
        logger.info("Recognition model loaded")

    async def recognize(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        channels: int,
    ) -> TranscriptionResult:
        """Распознаёт PCM int16 и возвращает текст с метаданными.

        При пустом audio_bytes сразу возвращает пустой результат без вызова
        модели. Длительность аудио определяет выбор короткого или длинного пути.

        Args:
            audio_bytes: Сырые PCM int16 байты аудиозаписи.
            sample_rate: Частота дискретизации входного аудио.
            channels: Число каналов входного PCM.

        Returns:
            TranscriptionResult с текстом, пословной разметкой и длительностью
            обработки; пустой результат при отсутствии аудио.
        """
        if not audio_bytes:
            return TranscriptionResult(text="", words=[], duration=0.0)

        audio_duration = pcm_int16_duration_s(audio_bytes, sample_rate, channels)

        if audio_duration > settings.LONG_AUDIO_THRESHOLD_S:
            logger.info(
                f"Using long-audio path: {audio_duration:.1f} > {settings.LONG_AUDIO_THRESHOLD_S} threshold"
            )
            return await self._model.transcribe_long_audio(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                channels=channels,
            )

        logger.info(f"Using short-audio path: {audio_duration:.1f}")
        return await self._model.transcribe_audio(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
