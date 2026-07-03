"""Инференс модели: batch, long-audio и инкрементальный стриминг.

Архитектура:
- загрузка весов из ClearMLl;
- два ThreadPoolExecutor (fast/slow);
- short path: один forward на всё аудио;
- long path: sliding window + merge logits по stride;
- streaming: делегирует в IncrementalCTCStreamer (накопление логитов по чанкам).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from clearml import Model
from safetensors import safe_open
from transformers import AutoModel, AutoProcessor

from src.gigaam_ctc.config import settings
from src.gigaam_ctc.core.dtos import TranscriptionResult, WordInfoResult
from src.gigaam_ctc.stt_model.audio_utils import (
    pcm_int16_bytes_to_mono_float32,
    pcm_int16_bytes_to_multichannel_float32,
)
from src.gigaam_ctc.stt_model.logits_processor import LogitsProcessor
from src.gigaam_ctc.stt_model.streaming_inference import IncrementalCTCStreamer

if TYPE_CHECKING:
    from src.gigaam_ctc.core.streaming_session import StreamingSession

logger = logging.getLogger(__name__)


class GigaAM:
    """Модель распознавания речи.

    Поддерживает короткое и длинное синхронное-распознавание, а также
    инкрементальное потоковое через IncrementalCTCStreamer.
    """

    def __init__(self):
        """Загружает processor, веса из ClearML и инициализирует inference-компоненты.

        Raises:
            RuntimeError: При ошибке загрузки модели, processor, весов или KenLM decoder.
        """
        self._SAMPLE_RATE = settings.SAMPLE_RATE
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._use_fp16 = settings.USE_FP16 and self._device == "cuda"

        self._fast_executor = ThreadPoolExecutor(
            max_workers=settings.FAST_WORKERS,
            thread_name_prefix="stt-fast",
        )
        self._slow_executor = ThreadPoolExecutor(
            max_workers=settings.SLOW_WORKERS,
            thread_name_prefix="stt-slow",
        )
        self._version = settings.MODEL_VERSION

        try:
            model = Model(settings.HF_MODEL_ID)
            model_path = Path(model.get_local_copy())
            target: Path = model_path.parent / model.name
            if target.exists():
                shutil.rmtree(target)
            model_path = model_path.rename(target)
        except Exception as e:
            raise RuntimeError(f"Failed to download model from ClearML: {e}") from e

        try:
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            if settings.USE_KENLM:
                if hasattr(self.processor, "decoder"):
                    self.decoder = self.processor.decoder
                else:
                    raise RuntimeError("Processor does not contain a KenLM decoder")
            logger.info("Processor successfully loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load processor: {e}") from e

        try:
            logger.info("Loading model...")
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            logger.info("Model successfully loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

        try:
            logger.info("Loading model weights from ClearML...")
            checkpoint_path = (
                Model(settings.CLEARML_MODEL_ID).get_weights_package(return_path=True)
                + "/model.safetensors"
            )
            logger.info("Weights successfully loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}") from e

        try:
            logger.info("Applying custom model weights to pretrained model...")
            with safe_open(checkpoint_path, framework="pt", device=self._device) as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            self.model.load_state_dict(state_dict)

            self.model.to(torch.device(self._device))
            if self._use_fp16:
                self.model = self.model.half()
                logger.info("Model converted to FP16 for faster inference")
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to apply custom model weights: {e}") from e

        self.logits_processor = LogitsProcessor(
            processor=self.processor,
            use_kenlm=settings.USE_KENLM,
            decoder=getattr(self, "decoder", None),
            model_config=self.model.config,
        )
        self._incremental_streamer = IncrementalCTCStreamer(self)

    @staticmethod
    def _to_fp16_features(features: dict, use_fp16: bool) -> dict:
        """Конвертирует float-тензоры features в FP16 при необходимости.

        Args:
            features: Словарь входных тензоров processor.
            use_fp16: True — привести floating point тензоры к half.

        Returns:
            Словарь features без изменений или с FP16 тензорами.
        """
        if not use_fp16:
            return features
        return {k: v.half() if torch.is_floating_point(v) else v for k, v in features.items()}

    def _forward(self, waveform_1d: torch.Tensor) -> np.ndarray:
        """Выполняет один forward pass модели на mono waveform.

        Args:
            waveform_1d: Одномерный тензор float32 семплов на SAMPLE_RATE.

        Returns:
            NumPy-массив логитов формы (time, vocab) в float32.
        """
        input_features = self.processor(
            waveform_1d, sampling_rate=self._SAMPLE_RATE, return_tensors="pt"
        ).to(self._device)
        input_features = self._to_fp16_features(input_features, self._use_fp16)

        with torch.no_grad():
            logits = self.model(**input_features).logits

        return self.logits_processor.ensure_logits_2d(logits[0])

    @staticmethod
    def _reraise_cuda_oom(exc: torch.cuda.OutOfMemoryError, message: str) -> None:
        """Очищает CUDA cache и пробрасывает RuntimeError при OOM.

        Args:
            exc: Исходное исключение OutOfMemoryError.
            message: Сообщение для лога перед пробросом.

        Raises:
            RuntimeError
        """
        logger.error("%s Cleaning cache and rejecting request.", message)
        torch.cuda.empty_cache()
        raise RuntimeError("Audio chunk too large for GPU memory. Split the audio.") from exc

    def _forward_batch_chunks(self, batch_chunks: list) -> list[np.ndarray]:
        """Forward pass для списка waveform-чанков одним батчем.

        Args:
            batch_chunks: Список одномерных waveform-тензоров или массивов.

        Returns:
            Список NumPy-массивов логитов по одному на каждый чанк.
        """
        if not batch_chunks:
            return []

        per_chunk_features = [
            self.processor(c, sampling_rate=self._SAMPLE_RATE, return_tensors="pt")
            for c in batch_chunks
        ]
        batched_features = {
            k: torch.cat([f[k] for f in per_chunk_features], dim=0).to(self._device)
            for k in per_chunk_features[0].keys()
        }

        batched_features = self._to_fp16_features(batched_features, self._use_fp16)

        with torch.no_grad():
            batched_logits = self.model(**batched_features).logits

        return [
            self.logits_processor.ensure_logits_2d(batched_logits[i])
            for i in range(len(batch_chunks))
        ]

    def _forward_and_merge_chunks(
        self,
        padded_samples: np.ndarray,
        chunk_starts: list[int],
        chunk_length_frames: int,
        stride_frames: int,
        total_frames: int,
        *,
        progress_label: str = "",
    ) -> np.ndarray:
        """Forward по чанкам с инкрементальным merge logits в один массив.

        Не держит все logits чанков в памяти одновременно — merge выполняется
        по мере обработки micro-batch.

        Args:
            padded_samples: Mono семплы с padding для последнего чанка.
            chunk_starts: Список начальных индексов каждого чанка.
            chunk_length_frames: Длина одного чанка в семплах.
            stride_frames: Шаг sliding window в семплах.
            total_frames: Общее число семплов без padding.
            progress_label: Префикс для логов прогресса inference.

        Returns:
            Объединённый NumPy-массив логитов формы (total_logits, vocab).
        """
        if not chunk_starts:
            return np.array([])

        merged_logits: np.ndarray | None = None
        logits_per_stride: int | None = None
        total_original_logits: int | None = None
        micro_batch = settings.LONG_AUDIO_MICRO_BATCH_SIZE
        total_chunks = len(chunk_starts)
        processed_chunks = 0
        next_progress_pct = 0
        label = f"{progress_label} " if progress_label else ""

        logger.info(
            f"{label} Starting inference: {total_chunks} chunks (micro-batch={micro_batch})"
        )

        for batch_start in range(0, len(chunk_starts), micro_batch):
            batch_starts = chunk_starts[batch_start : batch_start + micro_batch]
            batch_chunks = [padded_samples[s : s + chunk_length_frames] for s in batch_starts]
            batch_logits = self._forward_batch_chunks(batch_chunks)

            for local_i, logits in enumerate(batch_logits):
                global_i = batch_start + local_i

                if merged_logits is None:
                    subsampling_rate = chunk_length_frames / logits.shape[0]
                    logits_per_stride = int(stride_frames / subsampling_rate)
                    total_original_logits = int(total_frames / subsampling_rate)
                    merged_logits = np.zeros(
                        (total_original_logits, logits.shape[1]),
                        dtype=np.float32,
                    )

                chunk_start_logits = global_i * logits_per_stride
                remaining = total_original_logits - chunk_start_logits
                if remaining <= 0:
                    continue

                take_logits = min(logits_per_stride, remaining)
                merged_logits[chunk_start_logits : chunk_start_logits + take_logits] = logits[
                    :take_logits
                ]

            processed_chunks += len(batch_logits)
            progress_pct = processed_chunks * 100 // total_chunks
            if progress_pct >= next_progress_pct:
                logger.info(
                    f"{label} Inference progress: {progress_pct}% ({processed_chunks}/{total_chunks} chunks)"
                )
                next_progress_pct = min(100, (progress_pct // 10 + 1) * 10)

        if merged_logits is not None:
            logger.info(f"{label}Inference done: merged {merged_logits.shape[0]} logit frames")

        return merged_logits if merged_logits is not None else np.array([])

    async def transcribe_audio(
        self, audio_bytes: bytes, sample_rate: int, channels: int = 1
    ) -> TranscriptionResult:
        """Асинхронно распознаёт короткое аудио одним forward pass.

        Args:
            audio_bytes: Сырые PCM int16 байты.
            sample_rate: Частота дискретизации входного аудио.
            channels: Число каналов PCM.

        Returns:
            TranscriptionResult с текстом, таймкодами слов и duration.

        Raises:
            ValueError: При пустых audio_bytes.
            RuntimeError: При ошибке inference или CUDA OOM.
        """
        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty")

        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._fast_executor,
                self._sync_transcribe,
                audio_bytes,
                sample_rate,
                channels,
            )
        except torch.cuda.OutOfMemoryError as e:
            self._reraise_cuda_oom(e, "CUDA out of memory on request.")
        except Exception as e:
            raise RuntimeError(f"Transcription error: {e}") from e

    async def transcribe_long_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        chunk_length_s: int | None = None,
        stride_s: int | None = None,
        channels: int = 1,
    ) -> TranscriptionResult:
        """Асинхронно распознаёт длинное аудио sliding window с merge logits.

        Args:
            audio_bytes: Сырые PCM int16 байты.
            sample_rate: Частота дискретизации входного аудио.
            chunk_length_s: Длина чанка в секундах; None — из settings.
            stride_s: Шаг окна в секундах; None — из settings.
            channels: Число каналов; при channels > 1 формируется диалог спикеров.

        Returns:
            TranscriptionResult с полным текстом и таймкодами слов.

        Raises:
            ValueError: При пустых audio_bytes.
            RuntimeError: При ошибке inference или CUDA OOM.
        """
        chunk_length_s = chunk_length_s or settings.LONG_AUDIO_CHUNK_LENGTH_S
        stride_s = stride_s or settings.LONG_AUDIO_STRIDE_S
        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty")

        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._slow_executor,
                self._sync_transcribe_long,
                audio_bytes,
                sample_rate,
                chunk_length_s,
                stride_s,
                channels,
            )
        except torch.cuda.OutOfMemoryError as e:
            self._reraise_cuda_oom(e, "CUDA OOM during long audio transcription.")
        except Exception as e:
            raise RuntimeError(f"Long transcription error: {e}") from e

    async def streaming_feed(self, session: StreamingSession, audio_bytes: bytes) -> bool:
        """Асинхронно подаёт PCM-чанк в инкрементальный стриминг.

        Args:
            session: Потоковая сессия с stt_state для накопления логитов.
            audio_bytes: Очередной фрагмент PCM int16.

        Returns:
            True, если добавлены новые кадры логитов; False при пустом входе.
        """
        if not audio_bytes:
            return False
        return await asyncio.get_running_loop().run_in_executor(
            self._fast_executor,
            self._sync_streaming_feed,
            session,
            audio_bytes,
        )

    def _sync_streaming_feed(self, session: StreamingSession, audio_bytes: bytes) -> bool:
        """Синхронная обёртка streaming_feed для thread pool.

        Args:
            session: Потоковая сессия распознавания.
            audio_bytes: Фрагмент PCM int16.

        Returns:
            True, если incremental streamer обновил accumulated_logits.
        """
        return self._incremental_streamer.feed(
            session.stt_state,
            audio_bytes,
            session.sample_rate,
            session.channels,
        )

    async def streaming_partial(self, session: StreamingSession) -> TranscriptionResult:
        """Асинхронно декодирует partial-текст из накопленных логитов.

        Args:
            session: Потоковая сессия с накопленным stt_state.

        Returns:
            TranscriptionResult с промежуточным текстом без таймкодов слов.
        """
        return await asyncio.get_running_loop().run_in_executor(
            self._fast_executor,
            self._sync_streaming_partial,
            session,
        )

    def _sync_streaming_partial(self, session: StreamingSession) -> TranscriptionResult:
        """Синхронно декодирует partial и сохраняет last_partial при непустом тексте.

        Args:
            session: Потоковая сессия распознавания.

        Returns:
            TranscriptionResult с промежуточным текстом.
        """
        result = self._incremental_streamer.decode(session.stt_state, is_final=False)
        if result.text:
            session.last_partial = result
        return result

    async def streaming_finalize(self, session: StreamingSession) -> TranscriptionResult:
        """Асинхронно финализирует фразу: flush хвоста и final decode.

        Args:
            session: Потоковая сессия с накопленным аудио и логитами.

        Returns:
            TranscriptionResult с финальным текстом и таймкодами слов.
        """
        return await asyncio.get_running_loop().run_in_executor(
            self._fast_executor,
            self._sync_streaming_finalize,
            session,
        )

    def _sync_streaming_finalize(self, session: StreamingSession) -> TranscriptionResult:
        """Финализирует фразу с fallback на partial и batch transcribe.

        Сначала flush + incremental final decode. При пустом тексте
        переиспользует last_partial; в крайнем случае — полный batch pass
        по audio_buffer.

        Args:
            session: Потоковая сессия с буфером аудио и stt_state.

        Returns:
            TranscriptionResult с финальным текстом или пустой результат.
        """
        if not session.has_audio:
            return TranscriptionResult(text="", words=[], duration=0.0)

        self._incremental_streamer.flush(session.stt_state)
        result = self._incremental_streamer.decode(session.stt_state, is_final=True)

        if result.text.strip():
            return result

        if session.last_partial is not None and session.last_partial.text.strip():
            logger.info("Incremental final decode empty, reusing last partial")
            return session.last_partial

        logger.warning(
            "Incremental final decode empty (%.2fs audio), falling back to batch transcribe",
            session.buffer_duration,
        )
        if session.buffer_duration > settings.LONG_AUDIO_THRESHOLD_S:
            return self._sync_transcribe_long(
                session.buffer_bytes,
                session.sample_rate,
                settings.LONG_AUDIO_CHUNK_LENGTH_S,
                settings.LONG_AUDIO_STRIDE_S,
                session.channels,
            )
        return self._sync_transcribe(
            session.buffer_bytes,
            session.sample_rate,
            session.channels,
        )

    def _sync_transcribe(
        self, audio_bytes: bytes, sample_rate: int, channels: int = 1
    ) -> TranscriptionResult:
        """Синхронно распознаёт короткое mono аудио одним forward pass.

        Args:
            audio_bytes: Сырые PCM int16 байты.
            sample_rate: Частота дискретизации входного аудио.
            channels: Число каналов; усредняется в mono перед forward.

        Returns:
            TranscriptionResult с текстом, словами и duration inference.

        Raises:
            ValueError: При пустых audio_bytes.
        """
        start = time.time()

        if not audio_bytes:
            raise ValueError("Empty audio bytes")

        if sample_rate != self._SAMPLE_RATE:
            logger.info(f"Resampling {sample_rate} Hz -> {self._SAMPLE_RATE} Hz")

        samples = pcm_int16_bytes_to_mono_float32(
            audio_bytes, sample_rate, channels, self._SAMPLE_RATE
        )
        num_samples = len(samples)
        logits_np = self._forward(samples)

        subsampling_rate = num_samples / logits_np.shape[0]
        time_per_frame = subsampling_rate / self._SAMPLE_RATE

        text = self.logits_processor.decode(logits_np, is_final=True).strip()
        words = self.logits_processor.safe_extract_words(
            logits_np, time_per_frame, offset_seconds=0.0, channel_tag=1
        )

        duration = time.time() - start
        logger.info(f"Recognized text: {text}. Time: {duration:.3f}s")

        return TranscriptionResult(text=text, words=words, duration=duration)

    def _sync_transcribe_long(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        chunk_length_s: int,
        stride_s: int,
        channels: int,
    ) -> TranscriptionResult:
        """Синхронно распознаёт длинное аудио sliding window по каналам.

        При channels == 1 возвращает сплошной текст. При нескольких каналах
        формирует диалог вида «Спикер N: ...» по utterances каждого канала.

        Args:
            audio_bytes: Сырые PCM int16 байты.
            sample_rate: Частота дискретизации входного аудио.
            chunk_length_s: Длина inference-чанка в секундах.
            stride_s: Шаг sliding window в секундах.
            channels: Число каналов PCM.

        Returns:
            TranscriptionResult с текстом, словами и общим duration.

        Raises:
            ValueError: При пустых audio_bytes.
        """
        start = time.time()

        if not audio_bytes:
            raise ValueError("Empty audio bytes")

        if sample_rate != self._SAMPLE_RATE:
            logger.info(f"Resampling {sample_rate} Hz -> {self._SAMPLE_RATE} Hz")

        samples = pcm_int16_bytes_to_multichannel_float32(
            audio_bytes, sample_rate, channels, self._SAMPLE_RATE
        )

        chunk_length_frames = int(chunk_length_s * self._SAMPLE_RATE)
        stride_frames = int(stride_s * self._SAMPLE_RATE)

        all_words: list[WordInfoResult] = []
        all_utterances: list[dict] = []
        single_channel_text = ""

        blank_id = self.logits_processor.get_blank_id()

        for ch_idx in range(channels):
            mono_samples = samples[:, ch_idx]
            if isinstance(mono_samples, torch.Tensor):
                mono_samples = mono_samples.cpu().numpy()
            total_frames = len(mono_samples)

            padded_samples = np.pad(mono_samples, (0, chunk_length_frames), mode="constant")

            chunk_starts = list(range(0, total_frames, stride_frames))
            if not chunk_starts:
                continue

            audio_duration_s = total_frames / self._SAMPLE_RATE
            logger.info(
                "Long audio channel %d: %.1fs, %d chunks",
                ch_idx + 1,
                audio_duration_s,
                len(chunk_starts),
            )

            progress_label = f"ch{ch_idx + 1}/{channels}"
            final_logits = self._forward_and_merge_chunks(
                padded_samples,
                chunk_starts,
                chunk_length_frames,
                stride_frames,
                total_frames,
                progress_label=progress_label,
            )

            if channels == 1:
                single_channel_text = self.logits_processor.decode_long(
                    final_logits,
                    progress_label=progress_label,
                )
            elif final_logits.shape[0] > 0:
                time_per_frame = total_frames / final_logits.shape[0] / self._SAMPLE_RATE
                utterances = self.logits_processor.extract_utterances(
                    final_logits, time_per_frame, blank_id
                )
                for utt in utterances:
                    utt["speaker"] = ch_idx + 1
                all_utterances.extend(utterances)

            if final_logits.shape[0] > 0 and final_logits.shape[0] <= 10_000:
                time_per_frame = total_frames / final_logits.shape[0] / self._SAMPLE_RATE
                channel_words = self.logits_processor.safe_extract_words(
                    final_logits,
                    time_per_frame,
                    offset_seconds=0.0,
                    channel_tag=ch_idx + 1,
                    log_context=f"Failed to extract word timestamps for channel {ch_idx + 1}",
                )
                all_words.extend(channel_words)

        if channels == 1:
            final_text = single_channel_text.strip()
        else:
            all_utterances.sort(key=lambda x: x["start"])
            dialogue = [f"Спикер {utt['speaker']}: {utt['text']}" for utt in all_utterances]
            final_text = "\n".join(dialogue)

        duration = time.time() - start
        logger.info(
            f"Long audio complete: {channels} channel(s), {len(final_text)} chars, {duration} total"
        )

        return TranscriptionResult(text=final_text, words=all_words, duration=duration)

    def __del__(self):
        """Останавливает thread pool executors при уничтожении объекта.

        shutdown выполняется с wait=False, чтобы не блокировать завершение процесса.
        """
        try:
            self._fast_executor.shutdown(wait=False)
            self._slow_executor.shutdown(wait=False)
        except Exception:
            pass
