"""Загрузка и парсиг YAML-конфига для обучения."""

from pathlib import Path
from typing import Any

import yaml


def load_train_config(path: Path | str | None = None) -> dict[str, Any]:
    """Загружает конфиг для обучения из YAML-файла.

    Args:
        path: Путь к YAML-файлу.

    Returns:
        Словарь с секциями конфигурации: model, experiment, data, training.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """Собирает абсолютные пути к CSV-файлам датасета.

    Args:
        cfg: Загруженная конфигурация обучения.

    Returns:
        Словарь с ключами train_csv, val_csv и test_csv.
    """
    data_cfg = cfg["data"]
    data_dir = Path(data_cfg["data_dir"])
    return {
        "train_csv": str(data_dir / data_cfg["train_csv"]),
        "val_csv": str(data_dir / data_cfg["val_csv"]),
        "test_csv": str(data_dir / data_cfg["test_csv"]),
    }
