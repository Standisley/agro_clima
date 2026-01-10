# agroclima_cultiva/climate/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonFeatures:
    horizon_days: int
    chuva_total_mm: float
    et0_total_mm: float
    balanco_hidrico_mm: float
    dias_secos: int
    dias_chuva_forte: int
    tmax_media_c: float
    tmax_p95_c: float
    rh_media_pct: float
    vento_medio_kmh: float


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def compute_features_7_14(df_daily_forecast: pd.DataFrame) -> Dict[str, HorizonFeatures]:
    """
    Calcula features agregadas para 7 e 14 dias a partir do df do Open-Meteo.

    Espera colunas:
      ds, om_prcp_mm, om_tmax_c, om_rh_max, om_wind_kmh, om_et0_fao_mm
    """
    if df_daily_forecast is None or df_daily_forecast.empty:
        return {}

    df = df_daily_forecast.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    def compute(h: int) -> HorizonFeatures:
        d = df.head(h).copy()

        chuva_total = _safe_float(d["om_prcp_mm"].sum()) if "om_prcp_mm" in d.columns else 0.0
        et0_total = _safe_float(d["om_et0_fao_mm"].sum()) if "om_et0_fao_mm" in d.columns else 0.0
        balanco = chuva_total - et0_total

        dias_secos = int((d["om_prcp_mm"] < 0.5).sum()) if "om_prcp_mm" in d.columns else 0
        dias_chuva_forte = int((d["om_prcp_mm"] >= 20.0).sum()) if "om_prcp_mm" in d.columns else 0

        tmax_media = _safe_float(d["om_tmax_c"].mean()) if "om_tmax_c" in d.columns else 0.0
        tmax_p95 = _safe_float(d["om_tmax_c"].quantile(0.95)) if "om_tmax_c" in d.columns else 0.0

        # no Open-Meteo estamos usando UR máxima diária; aqui fazemos a média dessa proxy
        rh_media = _safe_float(d["om_rh_max"].mean()) if "om_rh_max" in d.columns else 0.0
        vento_medio = _safe_float(d["om_wind_kmh"].mean()) if "om_wind_kmh" in d.columns else 0.0

        return HorizonFeatures(
            horizon_days=h,
            chuva_total_mm=chuva_total,
            et0_total_mm=et0_total,
            balanco_hidrico_mm=balanco,
            dias_secos=dias_secos,
            dias_chuva_forte=dias_chuva_forte,
            tmax_media_c=tmax_media,
            tmax_p95_c=tmax_p95,
            rh_media_pct=rh_media,
            vento_medio_kmh=vento_medio,
        )

    return {"7d": compute(7), "14d": compute(14)}



