"""Фабрика Silero VAD: однократная загрузка и кэш JIT-модели в памяти.

Каждый StreamingVAD получает свою копию модели через torch.jit.load,
но веса читаются из общего буфера — без повторного torch.hub.load.
"""

import io
import logging

import torch

logger = logging.getLogger(__name__)


class VADFactory:
    """Фабрика и кэш Silero VAD для потокового распознавания.

    Модель загружается один раз при initialize, прогревается JIT-компиляцией
    и сохраняется в байтовый буфер. Последующие вызовы create_vad_model
    создают изолированные экземпляры с отдельным внутренним состоянием RNN.
    """

    _model_buffer: bytes | None = None

    @classmethod
    def initialize(cls) -> None:
        """Загружает Silero VAD из torch.hub и кэширует сериализованную модель.

        При повторном вызове ничего не делает, если буфер уже заполнен.
        Выполняет прогрев на батчах размером 1 и 4 для JIT-компиляции.

        Returns:
            None.
        """
        if cls._model_buffer is not None:
            return

        logger.info("Pre-loading Silero VAD model into memory...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )

        logger.info("Warming up VAD model (JIT compilation)...")
        _ = model(torch.zeros(1, 512), 16000)
        _ = model(torch.zeros(4, 512), 16000)

        buffer = io.BytesIO()
        torch.jit.save(model, buffer)
        buffer.seek(0)
        cls._model_buffer = buffer.read()
        logger.info("Silero VAD model successfully cached and warmed up.")

    @classmethod
    def create_vad_model(cls) -> torch.jit.ScriptModule:
        """Создаёт новый изолированный экземпляр VAD-модели.

        У каждого экземпляра своё состояние reset_states, что необходимо
        для независимых потоковых сессий. При первом вызове автоматически
        инициализирует кэш через initialize.

        Returns:
            JIT-модель Silero VAD.
        """
        if cls._model_buffer is None:
            cls.initialize()
        model = torch.jit.load(io.BytesIO(cls._model_buffer), map_location="cpu")
        model.eval()
        return model
