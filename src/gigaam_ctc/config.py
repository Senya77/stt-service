"""Настройки сервиса из переменных окружения.

Все параметры читаются один раз при импорте модуля.
Группы настроек:
- инференс модели (FP16, воркеры, KenLM);
- сеть (порты gRPC/HTTP, режим batch/stream);
- длинное аудио (чанкование со stride);
- стриминг (интервалы partial, шаг inference, beam width);
- VAD (пороги Silero для детекции конца фразы).
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Конфигурация сервиса"""

    # --- Модель и пулы потоков ---
    USE_FP16: bool = os.getenv("USE_FP16", "True").lower() == "true"
    FAST_WORKERS: int = int(os.getenv("FAST_WORKERS"))  # короткое аудио, стриминг
    SLOW_WORKERS: int = int(os.getenv("SLOW_WORKERS"))  # длинное аудио (тяжёлый forward)

    USE_KENLM: bool = os.getenv("USE_KENLM").lower() == "true"
    MODEL_VERSION: str = os.getenv("MODEL_VERSION")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION")

    # --- Сеть и режим деплоя ---
    GRPC_SERVER_PORT: int = int(os.getenv("GRPC_SERVER_PORT"))
    HTTP_SERVER_PORT: int = int(os.getenv("HTTP_SERVER_PORT"))
    OPENAI_MODEL_ID: str = os.getenv("OPENAI_MODEL_ID")
    STT_SERVICE_MODE: str = os.getenv("STT_SERVICE_MODE").lower()  # batch | stream
    SAMPLE_RATE: int = 16000  # целевая частота дискретизации для модели

    # --- Длинное аудио: sliding-window с overlap (stride < chunk) ---
    LONG_AUDIO_THRESHOLD_S: float = float(os.getenv("LONG_AUDIO_THRESHOLD_S"))
    LONG_AUDIO_CHUNK_LENGTH_S: int = int(os.getenv("LONG_AUDIO_CHUNK_LENGTH_S"))
    LONG_AUDIO_STRIDE_S: int = int(os.getenv("LONG_AUDIO_STRIDE_S"))
    LONG_AUDIO_MICRO_BATCH_SIZE: int = int(os.getenv("LONG_AUDIO_MICRO_BATCH_SIZE"))

    # --- Стриминг: частота partial-ответов и параметры инкрементального inference ---
    STREAM_PARTIAL_INTERVAL_S: float = float(os.getenv("STREAM_PARTIAL_INTERVAL_S"))
    STREAM_MIN_AUDIO_FOR_PARTIAL_S: float = float(os.getenv("STREAM_MIN_AUDIO_FOR_PARTIAL_S"))
    MAX_UTTERANCE_DURATION_S: float = float(os.getenv("MAX_UTTERANCE_DURATION_S"))

    STREAM_INFERENCE_STEP_S: float = float(os.getenv("STREAM_INFERENCE_STEP_S"))
    STREAM_INFERENCE_CONTEXT_S: float = float(os.getenv("STREAM_INFERENCE_CONTEXT_S"))
    STREAM_PARTIAL_BEAM_WIDTH: int = int(os.getenv("STREAM_PARTIAL_BEAM_WIDTH"))
    STREAM_FINAL_BEAM_WIDTH: int = int(os.getenv("STREAM_FINAL_BEAM_WIDTH"))

    # --- VAD (Silero): детекция конца фразы по паузе после речи ---
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD"))
    VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS"))
    VAD_MIN_SPEECH_MS: int = int(os.getenv("VAD_MIN_SPEECH_MS"))
    VAD_MIN_BUFFER_DURATION_S: float = float(os.getenv("VAD_MIN_BUFFER_DURATION_S"))


settings = Settings()
