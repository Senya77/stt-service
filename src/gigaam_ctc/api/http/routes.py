"""HTTP-маршруты OpenAI-compatible API для распознавания речи."""

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

from src.gigaam_ctc.api.http import audio_decoder
from src.gigaam_ctc.api.http.schemas import (
    HealthResponse,
    ModelListResponse,
    ModelObject,
    TranscriptionResponse,
)
from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.recognition_service import RecognitionService
from src.gigaam_ctc.stt_model.audio_utils import pcm_int16_duration_s

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")
health_router = APIRouter()


def _get_recognition_service(request: Request) -> RecognitionService:
    """Возвращает сервис распознавания из состояния приложения.

    Args:
        request: Текущий HTTP-запрос с доступом к app.state.

    Returns:
        Экземпляр RecognitionService, созданный при старте приложения.
    """
    return request.app.state.recognition_service


@health_router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Проверяет готовность сервиса к обработке запросов.

    Args:
        request: HTTP-запрос для проверки наличия recognition_service в state.

    Returns:
        HealthResponse со status ok, если модель загружена.

    Raises:
        HTTPException: 503, если recognition_service ещё не инициализирован.
    """
    if not hasattr(request.app.state, "recognition_service"):
        raise HTTPException(status_code=503, detail="Service not ready")
    return HealthResponse()


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """Возвращает список доступных моделей в формате OpenAI API.

    Returns:
        ModelListResponse с одной локальной моделью из настроек OPENAI_MODEL_ID.
    """
    return ModelListResponse(
        data=[ModelObject(id=settings.OPENAI_MODEL_ID)],
    )


@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form(default=settings.OPENAI_MODEL_ID),
    response_format: str = Form(default="json"),
) -> TranscriptionResponse | PlainTextResponse:
    """Распознаёт речь из загруженного аудиофайла.

    Принимает произвольный аудиофайл, декодирует в PCM mono,
    выполняет распознавание и возвращает текст в формате json или text.

    Args:
        request: HTTP-запрос для доступа к RecognitionService.
        file: Загруженный аудиофайл (multipart/form-data).
        model: Идентификатор модели. Принимается для совместимости с OpenAI API.
        response_format: Формат ответа: json (TranscriptionResponse) или text
            (plain text).

    Returns:
        TranscriptionResponse при response_format=json или PlainTextResponse
        с текстом распознавания при response_format=text.

    Raises:
        HTTPException: 400 при неподдерживаемом формате, пустом файле или ошибке
            декодирования; 500 при ошибке распознавания.
    """
    if response_format not in ("json", "text"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format!r}, expected 'json' or 'text'",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    logger.info(
        f"Received upload {file.filename} ({len(file_bytes) / (1024 * 1024):.2f} MB), decoding audio...",
    )

    try:
        pcm_bytes, sample_rate, channels = await audio_decoder.decode_to_pcm(file_bytes)
    except audio_decoder.AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    audio_duration_s = pcm_int16_duration_s(pcm_bytes, sample_rate, channels)
    logger.info(
        f"Decode done: {audio_duration_s:.1fs} audio, {sample_rate} Hz, {channels} channel(s), PCM {len(pcm_bytes) / (1024 * 1024):.2f} MB — starting recognition...",
    )

    recognition = _get_recognition_service(request)
    try:
        transcription = await recognition.recognize(
            audio_bytes=pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
    except Exception as e:
        logger.error(f"Recognition error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition error: {e}") from e

    logger.info(
        f"Recognition done in {transcription.duration:.1fs}, text length {len(transcription.text)} chars"
    )

    if response_format == "text":
        return PlainTextResponse(content=transcription.text)

    return TranscriptionResponse(text=transcription.text)
