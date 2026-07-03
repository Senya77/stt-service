from __future__ import annotations

from functools import cache

from src.gigaam_ctc.stt_model.model import GigaAM


@cache
def get_gigaam_model() -> GigaAM:
    model = GigaAM()
    return model
