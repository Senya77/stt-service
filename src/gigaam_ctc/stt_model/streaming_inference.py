"""Инкрементальный CTC-стриминг: forward только на новых сэмплах, накопление логитов."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.dtos import TranscriptionResult
from src.gigaam_ctc.stt_model.audio_utils import (
    get_chunks,
    pcm_int16_bytes_to_mono_float32,
)
from src.gigaam_ctc.stt_model.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from src.gigaam_ctc.stt_model.model import GigaAM

logger = logging.getLogger(__name__)


@dataclass
class StreamingSTTState:
    """Состояние инкрементального распознавания в рамках одной фразы.

    Накапливает логиты по мере поступления аудио-чанков без повторного
    forward всего буфера. Сбрасывается через reset после finalize фразы.

    Attributes:
        accumulated_logits: Накопленные логиты формы (time, vocab) или None.
        tail_samples: Overlap-контекст с прошлого шага inference.
        pending_samples: Сэмплы, ещё не дошедшие до размера шага inference.
        pending_samples_len: Общее число семплов в pending_samples.
        time_per_frame: Секунд на один кадр логитов для расчёта таймкодов.
        channel_tag: Метка канала для WordInfoResult при финальном decode.
    """

    accumulated_logits: np.ndarray | None = None
    tail_samples: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=torch.float32)
    )
    pending_samples: deque = field(default_factory=deque)
    pending_samples_len: int = 0
    time_per_frame: float = 0.0
    channel_tag: int = 1

    def reset(self) -> None:
        """Сбрасывает все поля состояния для новой фразы.

        Returns:
            None.
        """
        self.accumulated_logits = None
        self.tail_samples = torch.tensor([], dtype=torch.float32)
        if isinstance(self.pending_samples, deque):
            self.pending_samples.clear()
        else:
            self.pending_samples = deque()
        self.pending_samples_len = 0
        self.time_per_frame = 0.0


class IncrementalCTCStreamer:
    """Инкрементальный стример CTC: новые чанки аудио → новые логиты → decode.

    Алгоритм feed():
    - Накапливает pending_samples до step_samples (STREAM_INFERENCE_STEP_S).
    - К каждому шагу подмешивает tail_samples (overlap STREAM_INFERENCE_CONTEXT_S).
    - Forward только на context + step; из логитов отрезается overlap.
    - Новые кадры дописываются в accumulated_logits; tail обновляется.
    """

    def __init__(self, model: GigaAM):
        """Создаёт стример, привязанный к экземпляру GigaAM.

        Args:
            model: Загруженная модель GigaAM с методом _forward.
        """
        self._model = model
        self._sample_rate = model._SAMPLE_RATE
        self._step_samples = int(settings.STREAM_INFERENCE_STEP_S * self._sample_rate)
        self._context_samples = int(settings.STREAM_INFERENCE_CONTEXT_S * self._sample_rate)

    def feed(
        self,
        state: StreamingSTTState,
        audio_bytes: bytes,
        sample_rate: int,
        channels: int,
    ) -> bool:
        """Подает новый PCM-фрагмент и выполняет forward при достижении шага.

        Args:
            state: Изменяемое состояние инкрементального STT.
            audio_bytes: Очередной фрагмент PCM int16.
            sample_rate: Частота дискретизации входного аудио.
            channels: Число каналов входного PCM.

        Returns:
            True, если в этом вызове были добавлены новые кадры логитов;
            False при пустом входе или недостаточном накоплении семплов.
        """
        new_samples = pcm_int16_bytes_to_mono_float32(
            audio_bytes, sample_rate, channels, self._sample_rate
        )
        if new_samples.numel() == 0:
            return False

        updated = False
        state.pending_samples.append(new_samples)
        state.pending_samples_len += len(new_samples)

        while state.pending_samples_len >= self._step_samples:
            step_chunks, state.pending_samples_len = get_chunks(
                state.pending_samples, state.pending_samples_len, self._step_samples
            )
            step = step_chunks[0]

            context = (
                state.tail_samples[-self._context_samples :]
                if len(state.tail_samples) >= self._context_samples
                else state.tail_samples
            )
            chunk_input = torch.cat([context, step]) if len(context) > 0 else step

            logits_np = self._model._forward(chunk_input)

            new_logits = self._slice_new_logits(logits_np, len(context), len(chunk_input))
            subsampling = len(chunk_input) / logits_np.shape[0]

            if new_logits.shape[0] > 0:
                if state.accumulated_logits is None:
                    state.accumulated_logits = new_logits
                else:
                    state.accumulated_logits = np.concatenate(
                        [state.accumulated_logits, new_logits], axis=0
                    )
                if state.time_per_frame == 0.0:
                    state.time_per_frame = subsampling / self._sample_rate
                updated = True

            if len(chunk_input) >= self._context_samples:
                state.tail_samples = chunk_input[-self._context_samples :].clone()
            else:
                state.tail_samples = chunk_input.clone()

        return updated

    def flush(self, state: StreamingSTTState) -> bool:
        """Дообрабатывает остаток pending_samples в конце фразы.

        Args:
            state: Состояние STT с необработанными семплами в pending_samples.

        Returns:
            True, если были добавлены новые кадры логитов; False при пустом остатке.
        """
        if isinstance(state.pending_samples, np.ndarray):
            state.pending_samples = deque([state.pending_samples])
            state.pending_samples_len = len(state.pending_samples[0])

        if len(state.pending_samples) == 0:
            return False

        context = (
            state.tail_samples[-self._context_samples :]
            if len(state.tail_samples) >= self._context_samples
            else state.tail_samples
        )
        if state.pending_samples_len == 0:
            return False

        step = torch.empty(state.pending_samples_len, dtype=torch.float32)
        idx = 0
        while state.pending_samples:
            take = state.pending_samples.popleft()
            step[idx : idx + len(take)] = take
            idx += len(take)
        state.pending_samples_len = 0

        chunk_input = torch.cat([context, step]) if len(context) > 0 else step
        logits_np = self._model._forward(chunk_input)
        new_logits = self._slice_new_logits(logits_np, len(context), len(chunk_input))
        subsampling = len(chunk_input) / logits_np.shape[0]

        if new_logits.shape[0] == 0:
            return False

        if state.accumulated_logits is None:
            state.accumulated_logits = new_logits
        else:
            state.accumulated_logits = np.concatenate(
                [state.accumulated_logits, new_logits], axis=0
            )
        if state.time_per_frame == 0.0:
            state.time_per_frame = subsampling / self._sample_rate
        state.tail_samples = torch.tensor([], dtype=torch.float32)
        return True

    def _decode_logits(self, logits_np: np.ndarray, *, is_final: bool = False) -> str:
        """Декодирует накопленные логиты с KenLM или greedy fallback.

        Args:
            logits_np: NumPy-массив логитов формы (time, vocab).
            is_final: True — широкий beam для финала; False — узкий для partial.

        Returns:
            Распознанный текст без обрезки пробелов по краям.
        """
        logits_np = LogitsProcessor.ensure_logits_2d(logits_np)

        if settings.USE_KENLM:
            beam_width = (
                settings.STREAM_FINAL_BEAM_WIDTH if is_final else settings.STREAM_PARTIAL_BEAM_WIDTH
            )
            try:
                text = self._model.decoder.decode(logits_np, beam_width=beam_width)
                if text.strip():
                    return text
            except ValueError:
                logger.debug(
                    f"KenLM decode failed ({logits_np.shape[0]} frames), falling back to greedy"
                )

        return self._model.logits_processor.decode_greedy_from_logits(logits_np)

    def decode(self, state: StreamingSTTState, *, is_final: bool = False) -> TranscriptionResult:
        """Декодирует accumulated_logits в TranscriptionResult.

        Args:
            state: Состояние STT с накопленными логитами.
            is_final: True — извлекать таймкоды слов; False — только текст partial.

        Returns:
            TranscriptionResult с текстом, словами (при is_final) и duration decode.
        """
        start = time.time()

        if state.accumulated_logits is None or state.accumulated_logits.shape[0] == 0:
            return TranscriptionResult(text="", words=[], duration=time.time() - start)

        logits_np = LogitsProcessor.ensure_logits_2d(state.accumulated_logits)
        time_per_frame = state.time_per_frame
        text = self._decode_logits(logits_np, is_final=is_final)

        text = text.strip()
        words = []
        if is_final:
            words = self._model.logits_processor.safe_extract_words(
                logits_np,
                time_per_frame,
                offset_seconds=0.0,
                channel_tag=state.channel_tag,
                log_context="Failed to extract word timestamps in streaming decode",
            )

        duration = time.time() - start
        return TranscriptionResult(text=text, words=words, duration=duration)

    @staticmethod
    def _slice_new_logits(
        logits_np: np.ndarray,
        context_samples: int,
        chunk_input_samples: int,
    ) -> np.ndarray:
        """Отрезает логиты overlap-контекста; гарантирует хотя бы один новый кадр.

        Args:
            logits_np: Полные логиты forward на context + step.
            context_samples: Число семплов overlap-контекста в chunk_input.
            chunk_input_samples: Общее число семплов на входе forward.

        Returns:
            Срез логитов, соответствующий только новому step без overlap.
        """
        logits_np = LogitsProcessor.ensure_logits_2d(logits_np)
        num_frames = logits_np.shape[0]
        if num_frames == 0 or context_samples <= 0:
            return logits_np

        subsampling = chunk_input_samples / num_frames
        context_frames = int(context_samples / subsampling)

        safety_margin_frames = int(0.03 / (subsampling / settings.SAMPLE_RATE))
        context_frames = max(0, min(context_frames + safety_margin_frames, num_frames - 1))
        return logits_np[context_frames:]
