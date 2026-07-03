"""Потоковый детектор конца фразы на базе Silero VAD."""

from collections import deque

import torch

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.vad.factory import VADFactory
from src.gigaam_ctc.stt_model.audio_utils import (
    get_chunks,
    pcm_int16_bytes_to_mono_float32,
)


class StreamingVAD:
    """Детектор конца фразы для потокового распознавания речи.

    Оборачивает Silero VAD и отслеживает переходы речь/тишина по чанкам
    фиксированного размера CHUNK_SIZE. Фраза считается завершённой, когда
    после достаточно длинной речи накопилась тишина не короче min_silence_ms.

    Attributes:
        CHUNK_SIZE: Размер одного VAD-чанка в семплах.
        is_speaking: True, если в текущий момент детектируется речь.
    """

    CHUNK_SIZE = 512

    def __init__(
        self,
        threshold: float | None = None,
        min_silence_ms: int | None = None,
        min_speech_ms: int | None = None,
    ):
        """Создаёт экземпляр VAD с параметрами из аргументов или settings.

        Args:
            threshold: Порог вероятности речи Silero. None — settings.VAD_THRESHOLD.
            min_silence_ms: Минимальная длительность тишины для конца фразы.
                None — settings.VAD_MIN_SILENCE_MS.
            min_speech_ms: Минимальная длительность речи перед детектом тишины.
                None — settings.VAD_MIN_SPEECH_MS.
        """
        self._target_sr = settings.SAMPLE_RATE
        self.model = VADFactory.create_vad_model()
        self.threshold = threshold if threshold is not None else settings.VAD_THRESHOLD
        self.min_silence_ms = (
            min_silence_ms if min_silence_ms is not None else settings.VAD_MIN_SILENCE_MS
        )
        self.min_speech_ms = (
            min_speech_ms if min_speech_ms is not None else settings.VAD_MIN_SPEECH_MS
        )

        self.is_speaking = False
        self.silence_chunks = 0
        self.speech_chunks = 0
        self._buffer: deque = deque()
        self._buffer_len = 0

    def process_bytes(self, pcm_bytes: bytes, input_sample_rate: int, channels: int = 1) -> bool:
        """Обрабатывает очередной фрагмент PCM int16 и проверяет конец фразы.

        Args:
            pcm_bytes: Сырые PCM int16 байты входного фрагмента.
            input_sample_rate: Частота дискретизации входного аудио.
            channels: Число каналов входного PCM (по умолчанию 1).

        Returns:
            True, если зафиксирован конец фразы (тишина после достаточной речи);
            False во всех остальных случаях, включая пустой или неполный вход.
        """
        if not pcm_bytes:
            return False
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return False

        audio = pcm_int16_bytes_to_mono_float32(
            pcm_bytes, input_sample_rate, channels, self._target_sr
        )

        self._buffer.append(audio)
        self._buffer_len += len(audio)

        ready_chunks, self._buffer_len = get_chunks(self._buffer, self._buffer_len, self.CHUNK_SIZE)

        if not ready_chunks:
            return False
        batch_tensor = torch.stack(ready_chunks, dim=0)

        with torch.no_grad():
            speech_probs = self.model(batch_tensor, self._target_sr)

        if speech_probs.dim() > 1:
            speech_probs = speech_probs.squeeze(-1)
        probs = speech_probs.cpu().numpy()

        utterance_ended = False
        for prob in probs:
            if prob >= self.threshold:
                self.is_speaking = True
                self.speech_chunks += 1
                self.silence_chunks = 0
            elif self.is_speaking:
                self.silence_chunks += 1
                silence_ms = (self.silence_chunks * self.CHUNK_SIZE) / self._target_sr * 1000
                speech_ms = (self.speech_chunks * self.CHUNK_SIZE) / self._target_sr * 1000
                if silence_ms >= self.min_silence_ms and speech_ms >= self.min_speech_ms:
                    self.is_speaking = False
                    self.silence_chunks = 0
                    self.speech_chunks = 0
                    utterance_ended = True
        return utterance_ended

    def reset(self) -> None:
        """Сбрасывает состояние VAD после финализации фразы.

        Очищает состояние Silero, внутренний буфер семплов и счётчики
        речи/тишины для начала новой фразы в том же стриме.

        Returns:
            None.
        """
        self.model.reset_states()
        self._buffer.clear()
        self._buffer_len = 0
        self.is_speaking = False
        self.silence_chunks = 0
        self.speech_chunks = 0
