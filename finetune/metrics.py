"""Метрики оценки качества распознавания речи."""

import logging

import evaluate
import numpy as np


def create_compute_metrics(tokenizer, metric_name: str = "wer"):
    """Создаёт callback для вычисления метрик во время валидации.

    Args:
        tokenizer: Токенизатор для декодирования предсказаний и меток.
        metric_name: Имя метрики из библиотеки evaluate.

    Returns:
        Функция compute_metrics, совместимая с Trainer.
    """
    metric = evaluate.load(metric_name)

    def compute_metrics(pred):
        """Вычисляет WER по предсказаниям модели на валидационном батче.

        Args:
            pred: Объект EvalPrediction с полями predictions и label_ids.

        Returns:
            Словарь с ключом wer — значение метрики в процентах.
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(label_ids)

        logging.info("REF:", label_str[0])
        logging.info("HYP:", pred_str[0])

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics
