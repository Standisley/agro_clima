# agroclima_cultiva/ml/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Objective = Literal["renda_rapida", "baixo_risco", "seguranca_alimentar", "diversificacao"]


@dataclass(frozen=True)
class MLDatasetRow:
    # identificação
    ds_end: str  # data final da janela (YYYY-MM-DD)
    window_len: int  # 7 ou 14

    # localização / contexto
    lat: float
    lon: float
    municipio: Optional[str]
    uf: Optional[str]
    area_m2: float
    objetivo: Objective

    # cultura
    crop_id: str

    # features climáticas agregadas
    chuva_total: float
    et0_total: float
    balanco_hidrico: float
    dias_secos: int
    dias_chuva_forte: int
    tmax_media: float
    tmax_p95: float
    rh_media: float
    vento_medio: float

    # rótulo supervisionado fraco (teacher)
    score_teacher: float  # 0..1
    flags: str  # texto curto com penalidades/alertas

