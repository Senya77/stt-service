"""gRPC servicer для потокового распознавания речи."""

import asyncio
import logging
import time

import grpc

from src.gigaam_ctc.api.grpc import response_builder
from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.streaming_recognition_service import StreamingRecognitionService
from src.gigaam_ctc.core.streaming_session import StreamingSession
from src.gigaam_ctc.grpc_stub import stt_pb2

logger = logging.getLogger(__name__)


class StreamSpeechToTextServicer:
    """Обработчик потокового RPC StreamingRecognize.

    Жизненный цикл одного gRPC-стрима:
    1. Клиент отправляет streaming_config — создаётся StreamingSession.
    2. Клиент шлёт PCM int16 чанками; модель инкрементально накапливает логиты.
    3. Периодически отдаются partial-ответы; при конце фразы — final.
    4. Конец фразы определяется VAD, флагом is_final клиента или лимитом длины.
    """

    def __init__(self):
        """Инициализирует сервис потокового распознавания."""
        self._service = StreamingRecognitionService()
        self._version = settings.SERVICE_VERSION

    def shutdown(self) -> None:
        """Останавливает фоновые ресурсы сервиса."""
        self._service.shutdown()

    async def _finalize_utterance(
        self,
        session: StreamingSession,
        reason: str,
        request_id: str,
    ) -> stt_pb2.RecognizeResponse:
        """Завершает текущую фразу и возвращает финальный результат.

        Сбрасывает хвост буфера в модель, выполняет финальный decode
        и очищает буфер сессии для следующей фразы.

        Args:
            session: Активная потоковая сессия с накопленным аудио.
            reason: Причина завершения фразы для логирования.
            request_id: Идентификатор запроса клиента.

        Returns:
            RecognizeResponse с is_final=True.
        """
        logger.info(
            f"Utterance ended ({reason}), buffer={session.buffer_duration:.2f}, req_id={request_id}"
        )
        transcription = await self._service.finalize(session)
        session.clear_buffer()
        return response_builder.build_response(transcription, self._version, is_final=True)

    async def StreamingRecognize(self, request_iterator, context):
        """Обрабатывает поток запросов StreamingRecognize.

        Принимает конфигурацию, затем аудио-чанки; отдаёт partial и final
        RecognizeResponse по мере распознавания. При закрытии стрима клиентом
        финализирует остаток буфера, если в нём есть аудио.

        Args:
            request_iterator: Асинхронный итератор StreamingRecognizeRequest.
            context: gRPC-контекст вызова.

        Yields:
            RecognizeResponse с промежуточным или финальным текстом распознавания.

        Raises:
            grpc.RpcError: Пробрасывается без изменений.
            asyncio.CancelledError: При отмене задачи на стороне сервера.
            Exception: Любая необработанная ошибка; контекст получает INTERNAL.
        """
        session: StreamingSession | None = None
        last_intermediate_time = 0.0
        request_id = ""

        try:
            async for request in request_iterator:
                if context.cancelled():
                    return

                if request.request_id:
                    request_id = request.request_id

                if request.HasField("streaming_config"):
                    cfg = request.streaming_config.recognition_config
                    session = StreamingSession(
                        sample_rate=cfg.sample_rate_hertz,
                        channels=cfg.channels or 1,
                    )
                    last_intermediate_time = time.monotonic()
                    logger.info(f"Stream started: sr={cfg.sample_rate_hertz}, req_id={request_id}")
                    continue

                if session is None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details("Streaming config must be sent before audio chunks")
                    return

                if request.HasField("audio"):
                    feed_updated = False
                    utterance_ended_by_vad = False
                    audio_content = request.audio.audio_content

                    if audio_content:
                        session.append_audio(audio_content)
                        (
                            feed_updated,
                            utterance_ended_by_vad,
                        ) = await self._service.process_audio_chunk(
                            session,
                            audio_content,
                            with_vad=self._service.should_run_vad(session),
                        )

                    max_duration_exceeded = (
                        session.buffer_duration >= settings.MAX_UTTERANCE_DURATION_S
                    )

                    if utterance_ended_by_vad or request.audio.is_final or max_duration_exceeded:
                        if max_duration_exceeded:
                            reason = f"max duration ({settings.MAX_UTTERANCE_DURATION_S}s)"
                        elif utterance_ended_by_vad:
                            reason = "VAD (silence detected)"
                        else:
                            reason = "Client is_final=True"

                        response = await self._finalize_utterance(session, reason, request_id)
                        yield response
                        last_intermediate_time = time.monotonic()
                        continue

                    now = time.monotonic()
                    time_since_last = now - last_intermediate_time
                    has_logits = self._service.has_accumulated_logits(session)
                    can_partial = (
                        (feed_updated or has_logits)
                        and time_since_last >= settings.STREAM_PARTIAL_INTERVAL_S
                        and session.buffer_duration >= settings.STREAM_MIN_AUDIO_FOR_PARTIAL_S
                    )

                    if can_partial:
                        async with session.partial_lock:
                            transcription = await self._service.partial(session)
                            last_intermediate_time = time.monotonic()
                            if transcription.text:
                                yield response_builder.build_response(
                                    transcription, self._version, is_final=False
                                )

            if session and session.has_audio:
                logger.info(
                    f"Stream closed by client, recognizing remaining buffer, req_id={request_id}"
                )
                response = await self._finalize_utterance(session, "stream closed", request_id)
                yield response

        except grpc.RpcError:
            raise
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled, req_id={request_id}")
            raise
        except Exception as e:
            logger.error(f"StreamingRecognize error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"StreamingRecognize error: {str(e)}")
            raise
