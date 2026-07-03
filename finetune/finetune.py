"""Точка входа для запуска дообучения."""

import argparse

import numpy as np
import pytorch_lightning as pl
from collator import create_collators
from config import load_train_config, resolve_data_paths
from data import load_and_prepare_dataset
from metrics import create_compute_metrics
from model import load_model_components, model_size
from trainer import create_trainer, create_training_args


def main():
    """Запускает полный пайплайн дообучения: загрузка данных, модели и обучение."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to train config YAML")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    data_cfg = cfg["data"]
    data_paths = resolve_data_paths(cfg)
    seed = cfg["experiment"]["seed"]

    np.random.seed(seed)
    pl.seed_everything(seed)

    model, feature_extractor, tokenizer, processor = load_model_components(
        model_name=cfg["model"]["name"],
    )
    print(f"N of parameters: {model_size(model.model)}")

    audio_dataset = load_and_prepare_dataset(
        feature_extractor,
        tokenizer,
        train_csv=data_paths["train_csv"],
        val_csv=data_paths["val_csv"],
        test_csv=data_paths["test_csv"],
        max_duration=data_cfg["max_duration"],
        text_column=data_cfg["text_column"],
        audio_column=data_cfg["audio_column"],
        sampling_rate=data_cfg["sampling_rate"],
    )

    data_collator, val_data_collator = create_collators(processor)
    compute_metrics = create_compute_metrics(tokenizer)

    training_args = create_training_args(config_path=args.config)
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=audio_dataset["train"],
        eval_dataset={"val": audio_dataset["validation"]},
        data_collator=data_collator,
        val_data_collator=val_data_collator,
        compute_metrics=compute_metrics,
        training_args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
