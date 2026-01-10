from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..planner.catalog import CropSpec
from .schema import FEATURE_COLUMNS
from .teacher import teacher_score_crop


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def build_window_features(
    df_daily: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """
    Recebe df_daily com colunas diárias:
      ds, om_prcp_mm, om_et0_fao_mm, om_tmax_c, om_rh_max, om_wind_kmh
    e devolve um df indexado por ds_ref com features agregadas (rolling).

    Observação: usa janela "passada" incluindo o próprio dia,
    ajuste depois se preferir "D-1 ... D-window".
    """
    df = df_daily.copy()

    # garantir ds datetime
    if "ds" not in df.columns:
        raise ValueError("df_daily precisa ter coluna 'ds'.")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # colunas-base
    prcp = pd.to_numeric(df.get("om_prcp_mm", 0.0), errors="coerce").fillna(0.0)
    et0 = pd.to_numeric(df.get("om_et0_fao_mm", 0.0), errors="coerce").fillna(0.0)
    tmax = pd.to_numeric(df.get("om_tmax_c", 0.0), errors="coerce")
    rh = pd.to_numeric(df.get("om_rh_max", 0.0), errors="coerce")
    wind = pd.to_numeric(df.get("om_wind_kmh", 0.0), errors="coerce")

    roll = prcp.rolling(window_days, min_periods=window_days)

    out = pd.DataFrame({"ds_ref": df["ds"]})

    out["chuva_total_mm"] = roll.sum().astype(float)
    out["dias_secos"] = prcp.rolling(window_days, min_periods=window_days).apply(lambda s: float((s < 0.5).sum()), raw=False)
    out["dias_chuva_forte"] = prcp.rolling(window_days, min_periods=window_days).apply(lambda s: float((s >= 20.0).sum()), raw=False)

    out["et0_total_mm"] = et0.rolling(window_days, min_periods=window_days).sum().astype(float)
    out["balanco_hidrico_mm"] = out["chuva_total_mm"] - out["et0_total_mm"]

    out["tmax_media_c"] = tmax.rolling(window_days, min_periods=window_days).mean()
    out["tmax_p95_c"] = tmax.rolling(window_days, min_periods=window_days).quantile(0.95)

    # RH: se só tiver max, usamos como proxy de média (por enquanto)
    out["rh_media_pct"] = rh.rolling(window_days, min_periods=window_days).mean()

    out["vento_medio_kmh"] = wind.rolling(window_days, min_periods=window_days).mean()

    # auxiliares
    out["mes"] = out["ds_ref"].dt.month.astype(int)
    out["doy"] = out["ds_ref"].dt.dayofyear.astype(int)

    # remover linhas sem janela completa
    out = out.dropna(subset=["chuva_total_mm", "et0_total_mm", "tmax_media_c", "rh_media_pct", "vento_medio_kmh"]).reset_index(drop=True)

    # inteiros
    out["dias_secos"] = out["dias_secos"].round(0).astype(int)
    out["dias_chuva_forte"] = out["dias_chuva_forte"].round(0).astype(int)

    return out


def build_ml_dataset(
    df_daily: pd.DataFrame,
    crops: Sequence[CropSpec],
    *,
    lat: float,
    lon: float,
    area_m2: float,
    objetivo: str,
    municipio: Optional[str] = None,
    uf: Optional[str] = None,
    windows: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    """
    Dataset supervisionado por teacher:
    - Gera features por janela (7/14)
    - Para cada cultura, calcula score_teacher
    """
    rows: List[Dict[str, Any]] = []

    for w in windows:
        fdf = build_window_features(df_daily, window_days=int(w))

        for _, r in fdf.iterrows():
            feats = {k: r.get(k) for k in FEATURE_COLUMNS if k in fdf.columns}

            for crop in crops:
                score, pen_hidrico, pen_doenca = teacher_score_crop(crop, feats)

                row = {
                    "ds_ref": pd.to_datetime(r["ds_ref"]).strftime("%Y-%m-%d"),
                    "window_days": int(w),
                    "crop_id": crop.crop_id,

                    "lat": _safe_float(lat),
                    "lon": _safe_float(lon),
                    "municipio": municipio,
                    "uf": uf,
                    "area_m2": _safe_float(area_m2),
                    "objetivo": str(objetivo or "").strip().lower(),
                }

                # features
                for k in FEATURE_COLUMNS:
                    row[k] = r.get(k)

                # targets
                row["score_teacher"] = float(score)
                row["pen_hidrico"] = float(pen_hidrico)
                row["pen_doenca"] = float(pen_doenca)

                rows.append(row)

    df_out = pd.DataFrame(rows)

    # limpeza final (evita NaN em features essenciais)
    essential = ["chuva_total_mm", "et0_total_mm", "balanco_hidrico_mm", "tmax_media_c", "rh_media_pct", "vento_medio_kmh"]
    df_out = df_out.dropna(subset=essential).reset_index(drop=True)

    return df_out
