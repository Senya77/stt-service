"""Сервис потокового распознавания речи для gRPC stream API."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.dtos import TranscriptionResult
from src.gigaam_ctc.core.streaming_session import StreamingSession
from src.gigaam_ctc.core.vad.factory import VADFactory
from src.gigaam_ctc.stt_model.factory import get_gigaam_model

logger = logging.getLogger(__name__)


class StreamingRecognitionService:
    """Оркестрация потокового распознавания: feed, VAD, partial и final decode.

    Аналог RecognitionService для batch/HTTP, но для gRPC StreamingRecognize.
    """

    def __init__(self) -> None:
        """Загружает модель GigaAM и инициализирует VAD."""
        VADFactory.initialize()
        self._model = get_gigaam_model()
        self._vad_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vad")
        logger.info("Streaming recognition model loaded")

    def shutdown(self) -> None:
        """Останавливает пул потоков VAD."""
        self._vad_executor.shutdown(wait=False)

    async def feed(self, session: StreamingSession, audio_bytes: bytes) -> bool:
        """Подаёт PCM-чанк в инкрементальный STT.

        Args:
            session: Потоковая сессия с stt_state.
            audio_bytes: Очередной фрагмент PCM int16.

        Returns:
            True, если добавлены новые кадры логитов.
        """
        return await self._model.streaming_feed(session, audio_bytes)

    async def run_vad(self, session: StreamingSession, audio_bytes: bytes) -> bool:
        """Запускает VAD в отдельном потоке для переданного аудио-чанка.

        Args:
            session: Потоковая сессия с экземпляром VAD.
            audio_bytes: Очередной фрагмент PCM для анализа тишины.

        Returns:
            True, если VAD зафиксировал конец фразы.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._vad_executor,
            session.vad.process_bytes,
            audio_bytes,
            session.sample_rate,
            session.channels,
        )

    async def process_audio_chunk(
        self,
        session: StreamingSession,
        audio_bytes: bytes,
        *,
        with_vad: bool,
    ) -> tuple[bool, bool]:
        """Обрабатывает аудио-чанк: feed STT и опционально VAD.

        Args:
            session: Потоковая сессия.
            audio_bytes: Фрагмент PCM int16.
            with_vad: True — параллельно запускать VAD.

        Returns:
            Кортеж (feed_updated, utterance_ended_by_vad).

        Raises:
            Exception: Пробрасывает ошибки feed или VAD без подавления.
        """
        if with_vad:
            feed_updated, utterance_ended_by_vad = await asyncio.gather(
                self.feed(session, audio_bytes),
                self.run_vad(session, audio_bytes),
            )
            return feed_updated, utterance_ended_by_vad

        feed_updated = await self.feed(session, audio_bytes)
        return feed_updated, False

    async def partial(self, session: StreamingSession) -> TranscriptionResult:
        """Декодирует промежуточный текст из накопленных логитов.

        Args:
            session: Потоковая сессия с накопленным stt_state.

        Returns:
            TranscriptionResult с промежуточным текстом.
        """
        return await self._model.streaming_partial(session)

    async def finalize(self, session: StreamingSession) -> TranscriptionResult:
        """Финализирует фразу: flush хвоста и final decode.

        Args:
            session: Потоковая сессия с накопленным аудио и логитами.

        Returns:
            TranscriptionResult с финальным текстом и таймкодами слов.
        """
        return await self._model.streaming_finalize(session)

    @staticmethod
    def has_accumulated_logits(session: StreamingSession) -> bool:
        """Проверяет, есть ли накопленные логиты для partial-decode.

        Args:
            session: Потоковая сессия.

        Returns:
            True, если accumulated_logits содержит хотя бы один кадр.
        """
        logits = session.stt_state.accumulated_logits
        return logits is not None and logits.shape[0] > 0

    @staticmethod
    def should_run_vad(session: StreamingSession) -> bool:
        """Определяет, достаточно ли аудио для запуска VAD.

        Args:
            session: Потоковая сессия.

        Returns:
            True, если длительность буфера >= VAD_MIN_BUFFER_DURATION_S.
        """
        return session.buffer_duration >= settings.VAD_MIN_BUFFER_DURATION_S
