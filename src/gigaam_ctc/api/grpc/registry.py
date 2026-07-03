"""Регистрация gRPC-обработчиков для разных режимов деплоя STT-сервиса."""

import grpc

from src.gigaam_ctc.grpc_stub import stt_pb2 as stt_pb2

SERVICE_NAME = "SpeechToTextService"


def register_servicer(server: grpc.aio.Server, servicer) -> None:
    """Регистрирует RPC-методы Recognize и TruthfullyRecognize.

    Используется в деплойменте с синхронным распознаванием
    обслуживается отдельным процессом без потокового API.

    Args:
        server: gRPC-сервер, на котором регистрируются обработчики.
        servicer: Экземпляр SpeechToTextServicer с реализацией синхронных-методов.
    """
    rpc_method_handlers = {
        "Recognize": grpc.unary_unary_rpc_method_handler(
            servicer.Recognize,
            request_deserializer=stt_pb2.RecognizeRequest.FromString,
            response_serializer=stt_pb2.RecognizeResponse.SerializeToString,
        ),
        "TruthfullyRecognize": grpc.unary_unary_rpc_method_handler(
            servicer.TruthfullyRecognize,
            request_deserializer=stt_pb2.RecognizeRequest.FromString,
            response_serializer=stt_pb2.RecognizeResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(SERVICE_NAME, rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(SERVICE_NAME, rpc_method_handlers)


def register_stream_servicer(server: grpc.aio.Server, servicer) -> None:
    """Регистрирует потоковый RPC-метод StreamingRecognize.

    Используется в потоковом-деплойменте, где распознавание выполняется
    по мере поступления аудио-чанков от клиента.

    Args:
        server: gRPC-сервер, на котором регистрируется обработчик.
        servicer: Экземпляр StreamSpeechToTextServicer с реализацией стриминга.
    """
    rpc_method_handlers = {
        "StreamingRecognize": grpc.stream_stream_rpc_method_handler(
            servicer.StreamingRecognize,
            request_deserializer=stt_pb2.StreamingRecognizeRequest.FromString,
            response_serializer=stt_pb2.RecognizeResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(SERVICE_NAME, rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(SERVICE_NAME, rpc_method_handlers)
