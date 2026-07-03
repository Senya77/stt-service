"""Утилиты предобработки PCM-аудио перед подачей в модель STT."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import torch
import torchaudio


@lru_cache(maxsize=8)
def get_resampler(orig_freq: int, target_freq: int) -> torchaudio.transforms.Resample:
    """Возвращает кэшированный ресэмплер для пары частот.

    Создание Resample дорого; типичные пары orig_freq/target_freq повторяются
    между запросами, поэтому экземпляры переиспользуются.

    Args:
        orig_freq: Исходная частота дискретизации в Гц.
        target_freq: Целевая частота дискретизации в Гц.

    Returns:
        Экземпляр torchaudio.transforms.Resample.
    """
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_freq)


def pcm_int16_duration_s(audio_bytes: bytes, sample_rate: int, channels: int) -> float:
    """Вычисляет длительность PCM int16 аудио в секундах.

    Args:
        audio_bytes: Сырые PCM int16 байты.
        sample_rate: Частота дискретизации в Гц.
        channels: Число каналов PCM.

    Returns:
        Длительность в секундах; 0.0 при пустых байтах или некорректных параметрах.
    """
    if not audio_bytes or sample_rate <= 0 or channels <= 0:
        return 0.0
    return len(audio_bytes) / (2 * channels * sample_rate)


def _parse_pcm_bytes(audio_bytes: bytes, channels: int) -> torch.Tensor:
    """Конвертирует PCM int16 bytes в float32 тензор формы (frames, channels).

    Неполные фреймы в конце буфера отбрасываются автоматически.

    Args:
        audio_bytes: Сырые PCM int16 байты.
        channels: Число каналов входного PCM.

    Returns:
        Тензор float32 с нормализованными значениями.
        Пустой тензор при отсутствии данных.
    """
    if not audio_bytes:
        return torch.empty((0, channels), dtype=torch.float32)

    samples = torch.frombuffer(audio_bytes, dtype=torch.int16)

    valid_len = (samples.shape[0] // channels) * channels
    samples = samples[:valid_len]

    return samples.float().div_(32768.0).view(-1, channels)


def _resample(waveform: torch.Tensor, sample_rate: int, target_sr: int) -> torch.Tensor:
    """Применяет ресэмплинг waveform, если частоты не совпадают.

    Args:
        waveform: Входной тензор аудио.
        sample_rate: Текущая частота дискретизации waveform.
        target_sr: Целевая частота дискретизации.

    Returns:
        Waveform без изменений или после ресэмплинга до target_sr.
    """
    if sample_rate == target_sr:
        return waveform
    resampler = get_resampler(sample_rate, target_sr)
    return resampler(waveform)


def pcm_int16_bytes_to_mono_float32(
    audio_bytes: bytes, sample_rate: int, channels: int, target_sr: int
) -> torch.Tensor:
    """Конвертирует PCM int16 в mono float32 с ресэмплингом.

    Args:
        audio_bytes: Сырые PCM int16 байты.
        sample_rate: Частота дискретизации входного аудио.
        channels: Число каналов входного PCM.
        target_sr: Целевая частота после ресэмплинга.

    Returns:
        Одномерный тензор float32 формы (time,) с mono-сигналом.
    """
    tensor = _parse_pcm_bytes(audio_bytes, channels)
    if tensor.shape[0] == 0:
        return torch.tensor([], dtype=torch.float32)

    mono_waveform = tensor.mean(dim=1)

    resampled = _resample(mono_waveform.unsqueeze(0), sample_rate, target_sr)

    return resampled.squeeze(0)


def pcm_int16_bytes_to_multichannel_float32(
    audio_bytes: bytes, sample_rate: int, channels: int, target_sr: int
) -> torch.Tensor:
    """Конвертирует PCM int16 в многоканальный float32 с ресэмплингом.

    Args:
        audio_bytes: Сырые PCM int16 байты.
        sample_rate: Частота дискретизации входного аудио.
        channels: Число каналов входного PCM.
        target_sr: Целевая частота после ресэмплинга.

    Returns:
        Тензор float32 формы (time, channels) с сохранением всех каналов.
    """
    tensor = _parse_pcm_bytes(audio_bytes, channels)
    if tensor.shape[0] == 0:
        return torch.empty((0, channels), dtype=torch.float32)

    waveform = tensor.T
    resampled = _resample(waveform, sample_rate, target_sr)

    return resampled.T


def get_chunks(
    buffer: deque,
    buffer_len: int,
    chunk_size: int,
) -> tuple[list[torch.Tensor], int]:
    """Собирает фиксированные чанки из deque, уменьшая buffer_len.

    Извлекает из буфера столько полных чанков chunk_size, сколько возможно.
    Остаток остаётся в deque для следующего вызова.

    Args:
        buffer: Очередь фрагментов torch.Tensor с накопленными семплами.
        buffer_len: Общее число семплов во всех элементах deque.
        chunk_size: Размер одного выходного чанка в семплах.

    Returns:
        Кортеж (chunks, new_buffer_len): список собранных чанков и обновлённая
        длина оставшегося буфера.
    """
    chunks: list[torch.Tensor] = []
    while buffer_len >= chunk_size:
        chunk = torch.empty(chunk_size, dtype=torch.float32)
        idx = 0
        while idx < chunk_size:
            take = buffer[0]
            needed = chunk_size - idx
            if len(take) <= needed:
                chunk[idx : idx + len(take)] = take
                idx += len(take)
                buffer.popleft()
                buffer_len -= len(take)
            else:
                chunk[idx : idx + needed] = take[:needed]
                buffer[0] = take[needed:]
                buffer_len -= needed
                idx += needed
        chunks.append(chunk)
    return chunks, buffer_len
