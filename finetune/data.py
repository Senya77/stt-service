"""Загрузка и предобработка аудиодатасета для CTC fine-tuning."""

import os
from functools import partial
from pathlib import Path

import numpy as np
from datasets import Audio, DatasetDict, Value, load_dataset


def prepare_dataset(batch, feature_extractor, tokenizer, text_column: str):
    """Извлекает лог-мел признаки и токенизирует текст.

    Args:
        batch: Словарь с полями audio и текстовой колонкой.
        feature_extractor: Экстрактор аудио-признаков.
        tokenizer: Токенизатор для преобразования текста.
        text_column: Имя колонки с транскрипцией.

    Returns:
        Словарь с полями input_features, input_lengths и labels.
    """
    audio = batch["audio"]

    feats = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding="longest"
    )
    batch["input_features"] = feats.input_features[0]
    batch["input_lengths"] = feats.input_lengths[0]
    batch["labels"] = tokenizer(batch[text_column]).input_ids
    return batch


def _add_duration_column(dataset):
    """Добавляет колонку duration с длительностью аудио в секундах.

    Args:
        dataset: Датасет с колонкой audio.

    Returns:
        Датасет с дополнительной колонкой duration.
    """
    durations = np.array([len(x["array"]) / x["sampling_rate"] for x in dataset["audio"]])
    return dataset.add_column("duration", durations)


def _resolve_audio_paths(dataset, csv_path: str, audio_column: str):
    """Преобразует относительные пути к аудио в абсолютные.

    Args:
        dataset: Датасет с колонкой путей к аудиофайлам.
        csv_path: Путь к CSV-файлу, относительно которого разрешаются пути.
        audio_column: Имя колонки с путями к аудио.

    Returns:
        Датасет с абсолютными путями в колонке audio_column.
    """
    base_dir = Path(csv_path).resolve().parent

    def resolve(example):
        path = example[audio_column]
        if not os.path.isabs(path):
            example[audio_column] = str(base_dir / path)
        return example

    return dataset.map(resolve)


def _load_csv_split(csv_path: str, audio_column: str, sampling_rate: int):
    """Загружает один сплит датасета из CSV-файла.

    Args:
        csv_path: Путь к CSV-файлу со сплитом.
        audio_column: Имя колонки с путями к аудиофайлам.
        sampling_rate: Частота дискретизации для загрузки аудио.

    Returns:
        Датасет с колонкой audio в формате Hugging Face Audio.
    """
    dataset = load_dataset("csv", data_files=csv_path, split="train")
    dataset = _resolve_audio_paths(dataset, csv_path, audio_column)
    dataset = dataset.cast_column(audio_column, Value("string"))
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    return dataset.rename_column(audio_column, "audio")


def _prepare_split(
    dataset,
    feature_extractor,
    tokenizer,
    text_column: str,
    max_duration: float | int,
    shuffle: bool = False,
):
    """Фильтрует и предобрабатывает один сплит датасета.

    Args:
        dataset: Сырой датасет со сплитом.
        feature_extractor: Экстрактор аудио-признаков.
        tokenizer: Токенизатор для преобразования текста.
        text_column: Имя колонки с транскрипцией.
        max_duration: Максимальная допустимая длительность аудио в секундах.
        shuffle: Перемешивать ли датасет перед обработкой.

    Returns:
        Предобработанный датасет, готовый к обучению.
    """
    if shuffle:
        dataset = dataset.shuffle()

    dataset = _add_duration_column(dataset)
    dataset = dataset.filter(lambda x: x["duration"] < max_duration)
    return dataset.map(
        partial(
            prepare_dataset,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            text_column=text_column,
        ),
        remove_columns=dataset.column_names,
        num_proc=1,
    )


def load_and_prepare_dataset(
    feature_extractor,
    tokenizer,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    max_duration: float | int,
    text_column: str,
    audio_column: str,
    sampling_rate: int,
) -> DatasetDict:
    """Загружает и предобрабатывает train/validation/test выборки из CSV.

    Args:
        feature_extractor: Экстрактор аудио-признаков.
        tokenizer: Токенизатор для преобразования текста.
        train_csv: Путь к CSV-файлу обучающей выборки.
        val_csv: Путь к CSV-файлу валидационной выборки.
        test_csv: Путь к CSV-файлу тестовой выборки.
        max_duration: Максимальная допустимая длительность аудио в секундах.
        text_column: Имя колонки с транскрипцией.
        audio_column: Имя колонки с путями к аудиофайлам.
        sampling_rate: Частота дискретизации для загрузки аудио.

    Returns:
        DatasetDict с ключами train, validation и test.

    Raises:
        FileNotFoundError: Если отсутствует train или validation CSV-файл.
    """
    splits = {
        "train": (train_csv, True),
        "validation": (val_csv, False),
        "test": (test_csv, False),
    }

    result = {}
    for split_name, (csv_path, shuffle) in splits.items():
        if not os.path.exists(csv_path):
            if split_name == "test":
                continue
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        raw = _load_csv_split(csv_path, audio_column=audio_column, sampling_rate=sampling_rate)
        result[split_name] = _prepare_split(
            raw,
            feature_extractor,
            tokenizer,
            text_column=text_column,
            max_duration=max_duration,
            shuffle=shuffle,
        )

    return DatasetDict(result)
