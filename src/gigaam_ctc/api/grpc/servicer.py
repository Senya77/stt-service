"""gRPC servicer для синхронного распознавания речи."""

import logging

import grpc

from src.gigaam_ctc.api.grpc import response_builder
from src.gigaam_ctc.api.grpc.s3_storage import AudioS3Storage
from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.recognition_service import RecognitionService
from src.gigaam_ctc.grpc_stub import stt_pb2

logger = logging.getLogger(__name__)


class SpeechToTextServicer:
    """Обработчик RPC Recognize и TruthfullyRecognize.

    Поддерживает распознавание аудио, переданного в запросе,
    а также загрузку длинных записей из S3 по URI.
    """

    def __init__(self) -> None:
        """Инициализирует зависимости servicer и загружает модель распознавания."""
        self._s3 = AudioS3Storage()
        self._recognition = RecognitionService()
        self._version = settings.SERVICE_VERSION
        logger.info("Speech recognition model loaded")

    def _grpc_internal_error(
        self, context, error: Exception, label: str
    ) -> stt_pb2.RecognizeResponse:
        """Формирует пустой ответ и выставляет INTERNAL в gRPC-контексте.

        Args:
            context: gRPC-контекст текущего вызова.
            error: Исключение, ставшее причиной ошибки.
            label: Метка операции для лога.

        Returns:
            Пустой RecognizeResponse; клиент должен проверить status code контекста.
        """
        logger.error(f"{label} error: {error}", exc_info=True)
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(f"{label} error: {error}")
        return stt_pb2.RecognizeResponse()

    async def _recognize_pcm(
        self,
        config,
        audio_bytes: bytes,
        context,
        *,
        missing_audio_detail: str = "Audio content is missing",
    ) -> stt_pb2.RecognizeResponse:
        """Выполняет распознавание PCM-аудио и собирает ответ.

        Args:
            config: RecognitionConfig из запроса (sample rate, channels).
            audio_bytes: Сырые PCM-байты аудиозаписи.
            context: gRPC-контекст для установки INVALID_ARGUMENT при пустом аудио.
            missing_audio_detail: Текст ошибки, если audio_bytes пустой.

        Returns:
            RecognizeResponse с результатом распознавания или пустой ответ
            при ошибке валидации входных данных.
        """
        if not audio_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(missing_audio_detail)
            return stt_pb2.RecognizeResponse()

        transcription = await self._recognition.recognize(
            audio_bytes=audio_bytes,
            sample_rate=config.sample_rate_hertz,
            channels=config.channels,
        )
        return response_builder.build_response(transcription, self._version)

    async def Recognize(self, request, context):
        """Распознаёт речь из inline PCM в поле audio_content запроса.

        Args:
            request: RecognizeRequest с recognition_config и audio.
            context: gRPC-контекст вызова.

        Returns:
            RecognizeResponse с текстом, таймкодами слов и метаданными сервиса.
        """
        try:
            return await self._recognize_pcm(
                request.recognition_config,
                request.audio.audio_content,
                context,
            )
        except Exception as e:
            return self._grpc_internal_error(context, e, "Recognition")

    async def TruthfullyRecognize(self, request, context):
        """Распознаёт речь из S3 URI или inline PCM.

        Аналог Recognize, но допускает передачу ключа объекта S3 в поле uri
        вместо байтов — для длинных аудиозаписей.

        Args:
            request: RecognizeRequest с recognition_config и uri или audio.
            context: gRPC-контекст вызова.

        Returns:
            RecognizeResponse с результатом распознавания.
        """
        try:
            if request.HasField("uri"):
                task_name = request.uri
                logger.info(f"Loading audio from S3: {task_name}")
                raw_audio = await self._s3.receive(request.uri)
            elif request.HasField("audio") and request.audio.audio_content:
                raw_audio = request.audio.audio_content
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Audio content or URI is missing")
                return stt_pb2.RecognizeResponse()

            return await self._recognize_pcm(
                request.recognition_config,
                raw_audio,
                context,
                missing_audio_detail="Audio content or URI is missing",
            )
        except Exception as e:
            return self._grpc_internal_error(context, e, "TruthfullyRecognize")
