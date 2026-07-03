"""Коллаторы для батчинга аудио при обучении."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class DataCollatorCTCWithPadding:
    """Собирает батч с паддингом признаков и меток.

    Attributes:
        processor: Процессор Hugging Face с feature extractor и токенизатором.
        padding: Стратегия паддинга для батчинга.
        max_length: Максимальная длина аудио-признаков.
        max_length_tokens: Максимальная длина последовательности токенов.
    """

    processor: Any
    padding: str = "longest"
    max_length: int | None = 3001
    max_length_tokens: int | None = 1000

    def __call__(
        self, features: list[dict[str, list[int] | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Собирает и паддит батч из списка аудио.

        Args:
            features: Список словарей с полями input_features, input_lengths и labels.

        Returns:
            Батч с тензорами input_features, input_lengths, labels и опционально
            attention_mask.
        """
        input_features = [
            {"input_features": np.asarray(feature["input_features"]).T} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, max_length=self.max_length, return_tensors="pt"
        )
        batch["input_features"] = batch["input_features"].transpose(1, 2)

        input_lengths = [feature["input_lengths"] for feature in features]

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_tokens,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["input_lengths"] = torch.LongTensor(input_lengths)
        batch["labels"] = labels

        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def create_collators(
    processor: Any,
) -> tuple[DataCollatorCTCWithPadding, DataCollatorCTCWithPadding]:
    """Создаёт коллаторы для обучения и валидации.

    Args:
        processor: Процессор Hugging Face с feature extractor и токенизатором.

    Returns:
        Кортеж (data_collator, val_data_collator с разными стратегиями паддинга.
    """
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
    val_data_collator = DataCollatorCTCWithPadding(processor=processor, padding="max_length")
    return data_collator, val_data_collator
