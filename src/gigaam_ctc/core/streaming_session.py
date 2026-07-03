"""Состояние одного gRPC-стрима на время распознавания одной или нескольких фраз подряд."""

import asyncio
from dataclasses import dataclass, field

from src.gigaam_ctc.core.dtos import TranscriptionResult
from src.gigaam_ctc.core.vad.streaming_vad import StreamingVAD
from src.gigaam_ctc.stt_model.audio_utils import pcm_int16_duration_s
from src.gigaam_ctc.stt_model.streaming_inference import StreamingSTTState


@dataclass
class StreamingSession:
    """Состояние потоковой сессии распознавания в одном gRPC-стриме.

    Накапливает PCM-буфер, VAD и STT-состояние для инкрементального decode.
    После finalize буфер и состояния сбрасываются — сессия переиспользуется
    для следующей фразы в том же соединении.

    Attributes:
        sample_rate: Частота дискретизации аудио из streaming_config.
        channels: Число каналов PCM.
        audio_buffer: Накопленные PCM int16 байты текущей фразы.
        vad: Детектор конца фразы для текущего стрима.
        stt_state: Состояние инкрементального STT (логиты, offset и т.д.).
        partial_lock: Блокировка для безопасной выдачи partial-результатов.
        last_partial: Последний промежуточный результат распознавания или None.
    """

    sample_rate: int
    channels: int = 1
    audio_buffer: bytearray = field(default_factory=bytearray)
    vad: StreamingVAD = field(default_factory=StreamingVAD)
    stt_state: StreamingSTTState = field(default_factory=StreamingSTTState)
    partial_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_partial: TranscriptionResult | None = None

    def append_audio(self, audio_bytes: bytes) -> None:
        """Добавляет очередной фрагмент PCM в буфер текущей фразы.

        Args:
            audio_bytes: Сырые PCM int16 байты из StreamingRecognizeRequest.

        Returns:
            None.
        """
        self.audio_buffer.extend(audio_bytes)

    @property
    def has_audio(self) -> bool:
        """Проверяет, есть ли накопленное аудио в буфере.

        Returns:
            True, если audio_buffer не пустой.
        """
        return len(self.audio_buffer) > 0

    @property
    def buffer_duration(self) -> float:
        """Возвращает длительность накопленного аудио в секундах.

        Returns:
            Длительность буфера с учётом sample_rate и channels.
        """
        return pcm_int16_duration_s(bytes(self.audio_buffer), self.sample_rate, self.channels)

    @property
    def buffer_bytes(self) -> bytes:
        """Возвращает копию накопленного PCM-буфера.

        Returns:
            Неизменяемые bytes содержимого audio_buffer.
        """
        return bytes(self.audio_buffer)

    def clear_buffer(self) -> None:
        """Сбрасывает состояние сессии после финализации фразы.

        Очищает буфер аудио, последний partial, состояние VAD и STT
        для начала распознавания следующей фразы в том же стриме.

        Returns:
            None.
        """
        self.audio_buffer.clear()
        self.last_partial = None
        self.vad.reset()
        self.stt_state.reset()
