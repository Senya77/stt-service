"""Настройка Hugging Face Trainer для обучения."""

import os
from pathlib import Path

import datasets
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.utils import is_datasets_available

from finetune.collator import DataCollatorCTCWithPadding
from finetune.config import load_train_config


class TrainerDifCollators(Trainer):
    """Trainer с отдельным коллатором для валидации.

    Позволяет использовать разные стратегии паддинга при обучении
    longest и валидации max_length.
    """

    def __init__(self, val_data_collator=None, *args, **kwargs):
        """Инициализирует trainer с опциональным валидационным коллатором.

        Args:
            val_data_collator: Коллатор для валидационного даталоадера.
                Если не указан, используется data_collator.
            *args: Позиционные аргументы для Trainer.
            **kwargs: Именованные аргументы для Trainer.
        """
        super().__init__(*args, **kwargs)
        self.val_data_collator = val_data_collator

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """Создаёт DataLoader для валидации с отдельным коллатором.

        Args:
            eval_dataset: Валидационный датасет или имя сплита в eval_dataset.

        Returns:
            Подготовленный DataLoader для валидации.

        Raises:
            ValueError: Если валидационный датасет не задан.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.val_data_collator if self.val_data_collator else self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)


def create_training_args(
    config_path: Path | str | None = None,
    experiment_name: str | None = None,
    seed: int | None = None,
) -> TrainingArguments:
    """Создаёт TrainingArguments из YAML-конфига.

    Args:
        config_path: Путь к YAML-конфигу.
        experiment_name: Имя эксперимента. Переопределяет значение из конфига.
        seed: Seed для воспроизводимости. Переопределяет значение из конфига.

    Returns:
        Объект TrainingArguments для Trainer.
    """
    cfg = load_train_config(config_path)

    exp = cfg["experiment"]
    experiment_name = experiment_name or exp["name"]
    seed = seed if seed is not None else exp["seed"]

    os.environ.setdefault("CLEARML_PROJECT", exp["clearml_project"])
    os.environ["CLEARML_TASK"] = experiment_name

    training_kwargs = dict(cfg["training"])
    training_kwargs.update(
        output_dir=exp["output_dir"].format(experiment_name=experiment_name),
        seed=seed,
        run_name=experiment_name,
    )

    return TrainingArguments(**training_kwargs)


def create_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    data_collator: DataCollatorCTCWithPadding,
    val_data_collator: DataCollatorCTCWithPadding,
    compute_metrics,
    training_args: TrainingArguments | None = None,
) -> TrainerDifCollators:
    """Создаёт TrainerDifCollators со всеми компонентами обучения.

    Args:
        model: Модель для обучения.
        processor: Процессор Hugging Face.
        train_dataset: Обучающий датасет.
        eval_dataset: Валидационный датасет или словарь сплитов.
        data_collator: Коллатор для обучения.
        val_data_collator: Коллатор для валидации.
        compute_metrics: Callback для вычисления метрик.
        training_args: Аргументы обучения. Если не указаны, создаются из конфига.

    Returns:
        Настроенный экземпляр TrainerDifCollators.
    """
    if training_args is None:
        training_args = create_training_args()

    return TrainerDifCollators(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        val_data_collator=val_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
