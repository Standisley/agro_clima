# agroclima_cultiva/ml/dataset_builder.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence

import pandas as pd

from ..planner.catalog import CropSpec
from ..schemas.inputs import FarmInput
from ..climate.metrics import compute_features_7_14, HorizonFeatures
from .schema import MLDatasetRow
from .teacher import teacher_score_for_crop


def _feats_from_horizon(h: HorizonFeatures) -> Dict[str, float]:
    return {
        "chuva_total": float(h.chuva_total_mm),
        "et0_total": float(h.et0_total_mm),
        "balanco_hidrico": float(h.balanco_hidrico_mm),
        "dias_secos": float(h.dias_secos),
        "dias_chuva_forte": float(h.dias_chuva_forte),
        "tmax_media": float(h.tmax_media_c),
        "tmax_p95": float(h.tmax_p95_c),
        "rh_media": float(h.rh_media_pct),
        "vento_medio": float(h.vento_medio_kmh),
    }


def build_ml_dataset_from_history(
    inp: FarmInput,
    df_hist: pd.DataFrame,
    crops: Sequence[CropSpec],
    window_lens: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    """
    Dataset supervisionado fraco (teacher):
    - Usa histórico diário (1 ano) padronizado (Open-Meteo)
    - Rolling por ds_end
    - Para cada ds_end e janela (7/14), aplica teacher_score_for_crop(crop, feats)
    """
    inp.validate()

    if df_hist is None or df_hist.empty:
        raise ValueError("df_hist vazio.")

    required = {"ds", "om_prcp_mm", "om_tmax_c", "om_rh_max", "om_wind_kmh", "om_et0_fao_mm"}
    missing = required - set(df_hist.columns)
    if missing:
        raise ValueError(f"df_hist sem colunas obrigatórias: {sorted(missing)}")

    df = df_hist.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    max_w = max(int(x) for x in window_lens)
    rows: List[Dict] = []

    for end_idx in range(len(df)):
        df_slice = df.iloc[: end_idx + 1].copy()
        if len(df_slice) < max_w:
            continue

        metrics = compute_features_7_14(df_slice)
        ds_end = pd.to_datetime(df_slice["ds"].iloc[-1]).strftime("%Y-%m-%d")

        for w in window_lens:
            key = f"{int(w)}d"
            h = metrics.get(key)
            if h is None:
                continue

            feats = _feats_from_horizon(h)

            for crop in crops:
                tr = teacher_score_for_crop(crop, feats)

                row = MLDatasetRow(
                    ds_end=ds_end,
                    window_len=int(w),
                    lat=float(inp.lat),
                    lon=float(inp.lon),
                    municipio=inp.municipio,
                    uf=inp.uf,
                    area_m2=float(inp.area_m2),
                    objetivo=inp.objetivo,  # type: ignore[arg-type]
                    crop_id=crop.crop_id,
                    chuva_total=float(feats["chuva_total"]),
                    et0_total=float(feats["et0_total"]),
                    balanco_hidrico=float(feats["balanco_hidrico"]),
                    dias_secos=int(feats["dias_secos"]),
                    dias_chuva_forte=int(feats["dias_chuva_forte"]),
                    tmax_media=float(feats["tmax_media"]),
                    tmax_p95=float(feats["tmax_p95"]),
                    rh_media=float(feats["rh_media"]),
                    vento_medio=float(feats["vento_medio"]),
                    score_teacher=float(tr.score),
                    flags=";".join(tr.flags),
                )
                rows.append(asdict(row))

    return pd.DataFrame(rows)
