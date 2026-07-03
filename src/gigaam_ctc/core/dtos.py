"""DTO результата распознавания."""

from dataclasses import dataclass, field


@dataclass
class WordInfoResult:
    """Одно слово с таймкодами; channel_tag — номер канала при многоканальном аудио."""

    word: str
    start_time: float  # секунды от начала фразы
    end_time: float
    channel_tag: int = 0


@dataclass
class TranscriptionResult:
    """Итог распознавания: текст, опциональные word-level таймкоды, время инференса."""

    text: str
    words: list[WordInfoResult] = field(default_factory=list)
    duration: float = 0.0  # время обработки, не длительность аудио
