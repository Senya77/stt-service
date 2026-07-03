"""Постобработка CTC-логитов: greedy-декод, таймкоды слов, сегментация реплик.

CTC выдаёт последовательность токенов с blank-вставками; здесь:
- collapse повторов и blank (greedy);
- извлечение границ слов по SentencePiece-префиксу ▁ или паузам между токенами;
- разбиение длинного аудио на куски по паузам > min_pause.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from src.gigaam_ctc.core.dtos import WordInfoResult

logger = logging.getLogger(__name__)


class LogitsProcessor:
    """Декодирование CTC-логитов и извлечение таймкодов слов.

    Оборачивает HuggingFace processor/tokenizer и опционально KenLM decoder
    для beam search на финальных результатах.
    """

    def __init__(self, processor, use_kenlm: bool = False, decoder=None, model_config=None):
        """Инициализирует процессор с tokenizer и опциональным KenLM.

        Args:
            processor: HuggingFace AutoProcessor модели STT.
            use_kenlm: True — использовать KenLM beam search для финального decode.
            decoder: decoder из processor; обязателен при use_kenlm=True.
            model_config: Конфиг модели для определения blank/pad token id.
        """
        self.processor = processor
        self._USE_KENLM = use_kenlm
        self.decoder = decoder
        self._model_config = model_config

    @staticmethod
    def to_numpy(logits: np.ndarray | torch.Tensor) -> np.ndarray:
        """Приводит логиты к NumPy float32 на CPU.

        Args:
            logits: NumPy-массив или torch.Tensor логитов CTC.

        Returns:
            NumPy-массив логитов на CPU в float32.
        """
        if isinstance(logits, np.ndarray):
            return logits
        return logits.detach().float().cpu().numpy()

    @staticmethod
    def ensure_logits_2d(logits: np.ndarray | torch.Tensor) -> np.ndarray:
        """Приводит логиты к двумерной форме (time, vocab) на CPU.

        Args:
            logits: NumPy-массив или torch.Tensor логитов CTC.

        Returns:
            NumPy-массив формы (time, vocab); одномерный вход reshape в (1, vocab).
        """
        arr = LogitsProcessor.to_numpy(logits)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def argmax_token_ids(logits: np.ndarray) -> np.ndarray:
        """Возвращает greedy token id по последней оси логитов.

        Args:
            logits: NumPy-массив логитов формы (time, vocab).

        Returns:
            Одномерный массив token id длиной time.
        """
        return np.asarray(logits.argmax(axis=-1), dtype=np.int64)

    def get_blank_id(self) -> int:
        """Возвращает id blank-токена CTC.

        Берёт pad_token_id из tokenizer или model_config; fallback — 0.

        Returns:
            Целочисленный id blank-токена.
        """
        blank_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if blank_id is None and self._model_config is not None:
            blank_id = getattr(self._model_config, "pad_token_id", 0)
        return blank_id or 0

    def decode(
        self,
        logits: np.ndarray,
        *,
        is_final: bool = True,
        beam_width: int = 64,
    ) -> str:
        """Декодирует логиты в текст: KenLM для final, greedy для partial и fallback.

        Args:
            logits: NumPy-массив логитов формы (time, vocab).
            is_final: True — разрешить KenLM beam search; False — только greedy.
            beam_width: Ширина луча для KenLM decode.

        Returns:
            Распознанный текст; пустая строка при отсутствии токенов.
        """
        logits = self.ensure_logits_2d(logits)

        if self._USE_KENLM and is_final and self.decoder is not None:
            try:
                text = self.decoder.decode(logits, beam_width=beam_width)
                if text.strip():
                    return text
            except ValueError:
                logger.debug(
                    f"KenLM decode failed ({logits.shape[0]} frames), falling back to greedy"
                )

        return self.decode_greedy_from_logits(logits)

    def decode_long(
        self,
        logits: np.ndarray,
        *,
        beam_width: int = 64,
        progress_label: str = "",
    ) -> str:
        """Декодирует merged logits длинного аудио сегментами или целиком.

        При включённом KenLM разбивает логиты на перекрывающиеся сегменты
        для экономии памяти; иначе выполняет один greedy decode.

        Args:
            logits: Объединённые логиты длинной записи формы (time, vocab).
            beam_width: Ширина луча KenLM для каждого сегмента.
            progress_label: Префикс для логов прогресса decode.

        Returns:
            Полный текст распознавания; пустая строка при logits.size == 0.
        """
        if logits.size == 0:
            return ""

        label = f"{progress_label} " if progress_label else ""
        logger.info("%sDecoding %d logit frames...", label, logits.shape[0])
        decode_start = time.time()

        if self._USE_KENLM and self.decoder is not None:
            segment_frames = 3000
            overlap_frames = 200
            step = max(1, segment_frames - overlap_frames)
            total_segments = (logits.shape[0] + step - 1) // step
            parts: list[str] = []
            for seg_idx, start in enumerate(range(0, logits.shape[0], step), start=1):
                segment = logits[start : start + segment_frames]
                text = self.decoder.decode(segment, beam_width=beam_width).strip()
                if text:
                    parts.append(text)
                if seg_idx == 1 or seg_idx == total_segments or seg_idx % 10 == 0:
                    logger.info(
                        "%sKenLM decode progress: segment %d/%d",
                        label,
                        seg_idx,
                        total_segments,
                    )
            result = " ".join(parts)
        else:
            result = self.decode_greedy_from_logits(logits)

        logger.info(
            "%sDecode done in %.1fs, %d chars",
            label,
            time.time() - decode_start,
            len(result),
        )
        return result

    def decode_greedy_from_logits(self, logits: np.ndarray) -> str:
        """Выполняет CTC greedy decode через tokenizer.

        Схлопывает blank и повторяющиеся токены, затем декодирует id в текст.
        Используется как fallback при ошибке KenLM и для partial-результатов.

        Args:
            logits: NumPy-массив логитов формы (time, vocab).

        Returns:
            Текст после CTC collapse и SentencePiece-декодирования.
        """
        logits = self.ensure_logits_2d(logits)
        blank_id = self.get_blank_id()
        greedy_ids = self.argmax_token_ids(logits)

        collapsed: list[int] = []
        prev: int | None = None
        for token_id in greedy_ids:
            token_id = int(token_id)
            if token_id == blank_id:
                prev = None
                continue
            if token_id != prev:
                collapsed.append(token_id)
            prev = token_id

        if not collapsed:
            return ""

        return self.processor.tokenizer.decode(collapsed).replace("▁", " ").strip()

    def safe_extract_words(
        self,
        logits: np.ndarray,
        time_per_frame: float,
        *,
        offset_seconds: float = 0.0,
        channel_tag: int = 0,
        log_context: str = "",
    ) -> list[WordInfoResult]:
        """Извлекает таймкоды слов с перехватом исключений.

        Args:
            logits: NumPy-массив логитов формы (time, vocab).
            time_per_frame: Длительность одного кадра логитов в секундах.
            offset_seconds: Смещение времени для абсолютных таймкодов.
            channel_tag: Метка канала для WordInfoResult.
            log_context: Контекст для warning при ошибке извлечения.

        Returns:
            Список WordInfoResult или пустой список при ошибке.
        """
        try:
            return self.extract_words_with_timestamps(
                logits, time_per_frame, offset_seconds=offset_seconds, channel_tag=channel_tag
            )
        except Exception as e:
            if log_context:
                logger.warning("%s: %s", log_context, e)
            else:
                logger.warning("Failed to extract word timestamps: %s", e)
            return []

    def extract_words_with_timestamps(
        self,
        logits: np.ndarray,
        time_per_frame: float,
        offset_seconds: float = 0.0,
        channel_tag: int = 0,
    ) -> list[WordInfoResult]:
        """Извлекает слова с таймкодами через processor или ручной CTC-парсер.

        Сначала пробует processor.decode(output_word_offsets=True);
        при неудаче вызывает extract_words_ctc_manual.

        Args:
            logits: NumPy-массив логитов формы (time, vocab).
            time_per_frame: Длительность одного кадра логитов в секундах.
            offset_seconds: Смещение начала фразы в секундах.
            channel_tag: Метка аудиоканала.

        Returns:
            Список WordInfoResult с текстом слова и start/end time.
        """
        logits = self.ensure_logits_2d(logits)
        greedy_ids = self.argmax_token_ids(logits)

        output = None
        try:
            output = self.processor.decode(
                torch.from_numpy(logits.copy()), output_word_offsets=True
            )
        except Exception:
            try:
                output = self.processor.decode(greedy_ids, output_word_offsets=True)
            except Exception:
                pass

        if output is not None:
            words: list[WordInfoResult] = []

            word_offsets = getattr(output, "word_offsets", None)
            if word_offsets is None and isinstance(output, dict):
                word_offsets = output.get("word_offsets")

            if word_offsets is not None and len(word_offsets) > 0:
                pad_token = getattr(self.processor.tokenizer, "pad_token", None)
                pad_token_str = str(pad_token) if pad_token is not None else None

                for w in word_offsets:
                    word_text = (
                        str(w["word"]).strip()
                        if isinstance(w, dict)
                        else str(getattr(w, "word", "")).strip()
                    )

                    start_off = float(
                        w["start_offset"] if isinstance(w, dict) else getattr(w, "start_offset", 0)
                    )
                    end_off = float(
                        w["end_offset"] if isinstance(w, dict) else getattr(w, "end_offset", 0)
                    )

                    is_pad = pad_token_str is not None and word_text == pad_token_str

                    if word_text and not is_pad:
                        words.append(
                            WordInfoResult(
                                word=word_text,
                                start_time=round(offset_seconds + start_off * time_per_frame, 2),
                                end_time=round(offset_seconds + end_off * time_per_frame, 2),
                                channel_tag=channel_tag,
                            )
                        )
                if words:
                    return words

        return self.extract_words_ctc_manual(
            greedy_ids, time_per_frame, offset_seconds, channel_tag
        )

    def extract_words_ctc_manual(
        self,
        greedy_ids: np.ndarray,
        time_per_frame: float,
        offset_seconds: float = 0.0,
        channel_tag: int = 0,
    ) -> list[WordInfoResult]:
        """Векторизованная экстракция слов из collapsed CTC-вывода.

        Определяет границы слов по SentencePiece-префиксу ▁, пробелам
        и паузам между токенами.

        Args:
            greedy_ids: Массив greedy token id по кадрам логитов.
            time_per_frame: Длительность одного кадра в секундах.
            offset_seconds: Смещение начала фразы в секундах.
            channel_tag: Метка аудиоканала.

        Returns:
            Список WordInfoResult с текстом и таймкодами каждого слова.
        """
        blank_id = self.get_blank_id()
        ids = np.asarray(greedy_ids, dtype=np.int64)

        if ids.size == 0:
            return []

        non_blank_mask = ids != blank_id
        non_blank_positions = np.where(non_blank_mask)[0]

        if non_blank_positions.size == 0:
            return []

        nb_tokens = ids[non_blank_positions]

        changes = np.where(np.diff(nb_tokens) != 0)[0] + 1
        group_starts = np.concatenate([[0], changes])
        group_ends = np.concatenate([changes, [len(nb_tokens)]])

        token_ids = nb_tokens[group_starts]
        first_frames = non_blank_positions[group_starts]
        last_frames = non_blank_positions[group_ends - 1]

        blanks_before = np.empty(len(token_ids), dtype=np.int64)
        blanks_before[0] = first_frames[0]
        if len(token_ids) > 1:
            blanks_before[1:] = first_frames[1:] - last_frames[:-1] - 1

        unique_tokens = np.unique(token_ids)
        token_to_char = {
            int(tid): self.processor.tokenizer.decode([int(tid)]) for tid in unique_tokens
        }
        chars = np.array([token_to_char[int(t)] for t in token_ids])

        min_blank_frames = max(2, int(0.15 / time_per_frame))

        is_start = np.zeros(len(token_ids), dtype=bool)
        is_start[0] = True
        for i in range(1, len(token_ids)):
            c = chars[i]
            if c.startswith("▁"):
                is_start[i] = True
            elif c.strip() == " " and i > 0:
                is_start[i] = True
            elif blanks_before[i] >= min_blank_frames:
                is_start[i] = True

        word_start_indices = np.where(is_start)[0]
        word_end_indices = np.concatenate([word_start_indices[1:], [len(token_ids)]]) - 1

        words: list[WordInfoResult] = []
        decode = self.processor.tokenizer.decode
        for ws, we in zip(word_start_indices, word_end_indices):
            group_token_ids = token_ids[ws : we + 1].tolist()
            group_chars = chars[ws : we + 1]

            while len(group_chars) > 0 and str(group_chars[-1]).strip() == "":
                group_token_ids.pop()
                group_chars = group_chars[:-1]
            if not group_token_ids:
                continue

            word_text = decode(group_token_ids).replace("▁", " ").strip()
            if not word_text:
                continue

            start_frame = int(first_frames[ws])
            end_frame = int(last_frames[we])

            words.append(
                WordInfoResult(
                    word=word_text,
                    start_time=round(offset_seconds + start_frame * time_per_frame, 2),
                    end_time=round(offset_seconds + (end_frame + 1) * time_per_frame, 2),
                    channel_tag=channel_tag,
                )
            )

        return words

    def extract_utterances(
        self,
        logits: np.ndarray,
        time_per_frame: float,
        blank_id: int | None = None,
        min_pause: float = 0.5,
        min_speech: float = 0.2,
    ) -> list[dict]:
        """Разбивает логиты многоканального аудио на реплики по паузам.

        Args:
            logits: NumPy-массив логитов одного канала формы (time, vocab).
            time_per_frame: Длительность одного кадра логитов в секундах.
            blank_id: Id blank-токена; None — определяется через get_blank_id.
            min_pause: Минимальная пауза между репликами в секундах.
            min_speech: Минимальная длительность реплики в секундах.

        Returns:
            Список словарей с ключами start (float) и text (str) для каждой реплики.
        """
        if blank_id is None:
            blank_id = self.get_blank_id()

        logits = self.ensure_logits_2d(logits)
        greedy_ids = self.argmax_token_ids(logits)

        speech_indices = np.where(greedy_ids != blank_id)[0]
        if len(speech_indices) == 0:
            return []

        gaps_frames = np.diff(speech_indices)
        gap_seconds = gaps_frames * time_per_frame
        split_points = np.where(gap_seconds > min_pause)[0] + 1

        segment_starts = np.concatenate([[0], split_points])
        segment_ends = np.concatenate([split_points, [len(speech_indices)]])

        utterances = []
        for s, e in zip(segment_starts, segment_ends):
            start_frame = int(speech_indices[s])
            end_frame = int(speech_indices[e - 1])

            start_time = start_frame * time_per_frame
            end_time = (end_frame + 1) * time_per_frame

            if end_time - start_time < min_speech:
                continue

            seg_logits = logits[start_frame : end_frame + 1]
            text = self.decode(seg_logits, is_final=True).strip()
            if text:
                utterances.append({"start": start_time, "text": text})

        return utterances
