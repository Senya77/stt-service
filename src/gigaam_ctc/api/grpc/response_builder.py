"""Сборка protobuf-ответов RecognizeResponse из доменных DTO распознавания."""

import logging

from src.gigaam_ctc.core.dtos import TranscriptionResult
from src.gigaam_ctc.grpc_stub import stt_pb2

logger = logging.getLogger(__name__)


def build_proto_result(
    transcription: TranscriptionResult, converted_text: str
) -> stt_pb2.RecognizeResponse.Result:
    """Формирует protobuf Result с текстом и пословной разметкой.

    Args:
        transcription: Результат распознавания с таймкодами слов.
        converted_text: Текст для поля result.text.

    Returns:
        Protobuf-структура Result с полями text и words.
    """
    word_infos = [
        stt_pb2.RecognizeResponse.Result.WordInfo(
            start_time=word.start_time,
            end_time=word.end_time,
            word=word.word,
            channel_tag=word.channel_tag,
        )
        for word in transcription.words
    ]
    return stt_pb2.RecognizeResponse.Result(
        text=converted_text,
        words=word_infos,
    )


def build_response(
    transcription: TranscriptionResult,
    version: str,
    is_final: bool = True,
) -> stt_pb2.RecognizeResponse:
    """Собирает полный RecognizeResponse для синхронного и потокового RPC.

    При ошибке построения пословной разметки возвращает ответ только с текстом
    без деградации всего запроса.

    Args:
        transcription: Результат распознавания из доменного слоя.
        version: Версия сервиса, записываемая в поле version ответа.
        is_final: Признак финального результата. False для промежуточных
        ответов в потоковом режиме.

    Returns:
        Готовый protobuf RecognizeResponse для отправки клиенту.
    """
    text = transcription.text

    try:
        proto_result = build_proto_result(transcription, text)
    except Exception as e:
        logger.warning(f"Failed to build proto result with word info: {e}")
        proto_result = stt_pb2.RecognizeResponse.Result(text=text)

    return stt_pb2.RecognizeResponse(
        recognition_result=text,
        is_final=is_final,
        recognition_duration=f"{transcription.duration:.3f}",
        status_code=200,
        version=version,
        result=proto_result,
    )
