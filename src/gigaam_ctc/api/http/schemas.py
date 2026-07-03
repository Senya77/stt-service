"""Pydantic-схемы запросов и ответов OpenAI-compatible HTTP API."""

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """Ответ эндпоинта POST /v1/audio/transcriptions в формате json.

    Attributes:
        text: Распознанный текст речи.
    """

    text: str


class ModelObject(BaseModel):
    """Описание одной модели в списке GET /v1/models.

    Attributes:
        id: Идентификатор модели.
        object: Тип объекта в OpenAI API.
        created: Unix timestamp создания.
        owned_by: Владелец модели.
    """

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Список моделей, совместимый с OpenAI GET /v1/models.

    Attributes:
        object: Тип коллекции.
        data: Список объектов ModelObject.
    """

    object: str = "list"
    data: list[ModelObject]


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /health.

    Attributes:
        status: Статус готовности сервиса.
    """

    status: str = "ok"
