"""Декодирование загруженных аудиофайлов в PCM для FastAPI приложения."""

import asyncio
import io
import logging

import av
import numpy as np

from src.gigaam_ctc.config import settings

logger = logging.getLogger(__name__)


class AudioDecodeError(Exception):
    """Ошибка декодирования или валидации входного аудиофайла."""


class ASRAudioPreprocessor:
    """Декодирует произвольный тип аудиофайла в mono s16 PCM заданной частоты.

    Использует PyAV для чтения потока и ресемплинга в target_sr.
    """

    def __init__(self, target_sr: int | None = None):
        """Задаёт целевую частоту дискретизации после ресемплинга.

        Args:
            target_sr: Частота. Если None, берётся settings.SAMPLE_RATE.
        """
        self.target_sr = target_sr or settings.SAMPLE_RATE

    def process_bytes(self, file_bytes: bytes) -> bytes:
        """Декодирует байты файла в PCM int16 mono bytes.

        Args:
            file_bytes: Содержимое аудиофайла в любом поддерживаемом PyAV формате.

        Returns:
            Сырые байты PCM int16 для передачи в RecognitionService.

        Raises:
            AudioDecodeError: При пустом файле, отсутствии аудиопотока, пустом
                декоде или ошибке PyAV.
        """
        if not file_bytes:
            raise AudioDecodeError("Empty audio file")

        try:
            input_buffer = io.BytesIO(file_bytes)
            container = av.open(input_buffer)

            if not container.streams.audio:
                raise AudioDecodeError("No audio stream found")

            stream = container.streams.audio[0]

            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.target_sr,
            )

            chunks = []

            for frame in container.decode(stream):
                for r_frame in resampler.resample(frame):
                    chunks.append(r_frame.to_ndarray()[0])

            for r_frame in resampler.resample(None):
                chunks.append(r_frame.to_ndarray()[0])

            if not chunks:
                raise AudioDecodeError("Decoded audio is empty")

            audio_np = np.concatenate(chunks)
            return audio_np.tobytes()

        except av.AVError as e:
            raise AudioDecodeError(f"Corrupted audio file: {str(e)}") from e
        except Exception as e:
            raise AudioDecodeError(f"Unexpected error: {str(e)}") from e


preprocessor = ASRAudioPreprocessor()


async def decode_to_pcm(file_bytes: bytes) -> tuple[bytes, int, int]:
    """Асинхронно декодирует файл в PCM int16 mono на target_sr.

    Args:
        file_bytes: Содержимое загруженного аудиофайла.

    Returns:
        Кортеж (pcm_bytes, sample_rate, channels): PCM int16, частота
        дискретизации и число каналов.

    Raises:
        AudioDecodeError
    """
    pcm_bytes = await asyncio.to_thread(preprocessor.process_bytes, file_bytes)
    return pcm_bytes, preprocessor.target_sr, 1
