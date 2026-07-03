"""Загрузка модели."""

from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer


def load_model_components(model_name: str):
    """Загружает модель и процессор.

    Args:
        model_name: Имя или путь к предобученной модели.

    Returns:
        Кортеж (model, feature_extractor, tokenizer, processor).
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, feature_extractor, tokenizer, processor


def model_size(model) -> int:
    """Выводит количество обучаемых и общих параметров модели.

    Args:
        model: PyTorch-модель.

    Returns:
        Количество обучаемых параметров.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable}/{total}")
    return trainable
