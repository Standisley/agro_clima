# agroclima_cultiva/ml/make_dataset.py
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Reutiliza seu catálogo JSON e dataclass CropSpec
from ..planner.catalog import CropSpec, load_catalog_from_json


# =============================================================================
# Open-Meteo (Archive) — histórico diário (1 ano)
# =============================================================================

def fetch_daily_history_openmeteo(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    """
    Baixa histórico diário via Open-Meteo Archive API e normaliza para o schema do projeto:
      ds, om_prcp_mm, om_tmin_c, om_tmax_c, om_wind_kmh, om_rh_max, om_et0_fao_mm
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(
            [
                "precipitation_sum",
                "temperature_2m_min",
                "temperature_2m_max",
                "windspeed_10m_max",
                "relative_humidity_2m_max",
                "et0_fao_evapotranspiration",
            ]
        ),
        "timezone": timezone,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily") or {}
    if not daily:
        return pd.DataFrame(
            columns=[
                "ds",
                "om_prcp_mm",
                "om_tmin_c",
                "om_tmax_c",
                "om_wind_kmh",
                "om_rh_max",
                "om_et0_fao_mm",
            ]
        )

    df = pd.DataFrame(
        {
            "ds": daily.get("time", []),
            "om_prcp_mm": daily.get("precipitation_sum", []),
            "om_tmin_c": daily.get("temperature_2m_min", []),
            "om_tmax_c": daily.get("temperature_2m_max", []),
            "om_wind_kmh": daily.get("windspeed_10m_max", []),
            "om_rh_max": daily.get("relative_humidity_2m_max", []),
            "om_et0_fao_mm": daily.get("et0_fao_evapotranspiration", []),
        }
    )

    if df.empty:
        return df

    df["ds"] = pd.to_datetime(df["ds"])
    for c in ["om_prcp_mm", "om_tmin_c", "om_tmax_c", "om_wind_kmh", "om_rh_max", "om_et0_fao_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.sort_values("ds").reset_index(drop=True)


# =============================================================================
# Features por janela (7/14) — compatível com seu metrics.py (mas genérico)
# =============================================================================

def compute_window_features(df: pd.DataFrame, h: int) -> Dict[str, float]:
    """
    Calcula features na janela [últimos h dias] do DF acumulado até ds_end.
    """
    d = df.tail(h).copy()

    chuva_total = float(d["om_prcp_mm"].sum())
    et0_total = float(d["om_et0_fao_mm"].sum()) if "om_et0_fao_mm" in d.columns else 0.0
    balanco = chuva_total - et0_total

    dias_secos = int((d["om_prcp_mm"] < 0.5).sum())
    dias_chuva_forte = int((d["om_prcp_mm"] >= 20.0).sum())

    tmax_media = float(d["om_tmax_c"].mean()) if "om_tmax_c" in d.columns else 0.0
    tmax_p95 = float(d["om_tmax_c"].quantile(0.95)) if "om_tmax_c" in d.columns else 0.0

    rh_media = float(d["om_rh_max"].mean()) if "om_rh_max" in d.columns else 0.0
    vento_medio = float(d["om_wind_kmh"].mean()) if "om_wind_kmh" in d.columns else 0.0

    def nz(x: float) -> float:
        try:
            if x is None or np.isnan(float(x)):
                return 0.0
            return float(x)
        except Exception:
            return 0.0

    return {
        "chuva_total": nz(chuva_total),
        "et0_total": nz(et0_total),
        "balanco_hidrico": nz(balanco),
        "dias_secos": float(dias_secos),
        "dias_chuva_forte": float(dias_chuva_forte),
        "tmax_media": nz(tmax_media),
        "tmax_p95": nz(tmax_p95),
        "rh_media": nz(rh_media),
        "vento_medio": nz(vento_medio),
    }


# =============================================================================
# Teacher (weak supervision) — score 0..1 + flags
# =============================================================================

def teacher_score_for_crop(crop: CropSpec, feats: Dict[str, float], objetivo: str) -> Tuple[float, List[str]]:
    """
    Regra teacher minimalista para gerar score supervisionado (weak supervision).
    Objetivo: produzir um "alvo" numérico contínuo (0..1) + flags explicáveis.
    """
    score = 1.0
    flags: List[str] = []

    demanda = (str(crop.demanda_hidrica or "media")).strip().lower()
    grupo = (str(crop.grupo or "")).strip().lower()
    risco = (str(crop.risco or "medio")).strip().lower()
    investimento = (str(crop.investimento or "medio")).strip().lower()
    complexidade = (str(crop.complexidade or "media")).strip().lower()

    dias_secos = float(feats.get("dias_secos", 0))
    bal = float(feats.get("balanco_hidrico", 0.0))
    chuva = float(feats.get("chuva_total", 0.0))
    rh = float(feats.get("rh_media", 0.0))
    chuva_forte = float(feats.get("dias_chuva_forte", 0.0))
    tmax_p95 = float(feats.get("tmax_p95", 0.0))

    # 1) Penalidade hídrica (proxy)
    if demanda == "alta":
        if dias_secos >= 3:
            score -= 0.20
            flags.append("penal_hidrico:dias_secos>=3")
        if bal <= -10:
            score -= 0.20
            flags.append("penal_hidrico:bal<=-10")
    elif demanda == "media":
        if dias_secos >= 4:
            score -= 0.12
            flags.append("penal_hidrico_media:dias_secos>=4")
        if bal <= -15:
            score -= 0.15
            flags.append("penal_hidrico_media:bal<=-15")

    # 2) Doença/umidade (proxy): rh alto + chuva moderada; reforço por chuva forte
    if rh >= 90 and chuva >= 40:
        if grupo in {"folhosas", "hortaliça-fruto", "hortalica-fruto"}:
            score -= 0.22
            flags.append("penal_doenca:rh>=90_chuva>=40")
        else:
            score -= 0.10
            flags.append("penal_umidade:rh>=90_chuva>=40")

    if chuva_forte >= 1 and rh >= 88:
        score -= 0.10
        flags.append("penal_chuva_forte:>=1")

    # 3) Calor extremo (proxy simples)
    if tmax_p95 >= 34:
        score -= 0.08
        flags.append("penal_calor:tmax_p95>=34")
        if grupo == "folhosas":
            score -= 0.10
            flags.append("penal_calor_folhosas:tmax_p95>=34")

    # 4) Penalidade por objetivo "baixo_risco" usando atributos do catálogo
    obj = (objetivo or "").strip().lower()
    if obj == "baixo_risco":
        if risco == "alto":
            score -= 0.20
            flags.append("penal_obj:risco_alto")
        if investimento == "alto":
            score -= 0.10
            flags.append("penal_obj:investimento_alto")
        if complexidade == "alta":
            score -= 0.10
            flags.append("penal_obj:complexidade_alta")

    score = float(max(0.0, min(1.0, score)))
    return score, flags


# =============================================================================
# Builder do dataset (rolling windows 7/14)
# =============================================================================

def build_dataset(
    lat: float,
    lon: float,
    area_m2: float,
    objetivo: str,
    municipio: Optional[str],
    uf: Optional[str],
    df_hist: pd.DataFrame,
    crops: List[CropSpec],
    window_lens: Tuple[int, int] = (7, 14),
) -> pd.DataFrame:
    if df_hist is None or df_hist.empty:
        raise ValueError("Histórico vazio.")

    df = df_hist.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    max_w = max(int(x) for x in window_lens)
    rows: List[Dict] = []

    for end_idx in range(len(df)):
        # precisa ter pelo menos max_w dias até o ponto final
        if end_idx + 1 < max_w:
            continue

        ds_end = pd.to_datetime(df.loc[end_idx, "ds"]).strftime("%Y-%m-%d")
        df_up_to_end = df.iloc[: end_idx + 1].copy()

        for w in window_lens:
            feats = compute_window_features(df_up_to_end, int(w))

            for crop in crops:
                score_teacher, flags = teacher_score_for_crop(crop, feats, objetivo)

                rows.append(
                    {
                        "ds_end": ds_end,
                        "window_len": int(w),
                        "lat": float(lat),
                        "lon": float(lon),
                        "municipio": municipio or "",
                        "uf": uf or "",
                        "area_m2": float(area_m2),
                        "objetivo": str(objetivo),
                        "crop_id": crop.crop_id,
                        # features
                        "chuva_total": float(feats["chuva_total"]),
                        "et0_total": float(feats["et0_total"]),
                        "balanco_hidrico": float(feats["balanco_hidrico"]),
                        "dias_secos": int(feats["dias_secos"]),
                        "dias_chuva_forte": int(feats["dias_chuva_forte"]),
                        "tmax_media": float(feats["tmax_media"]),
                        "tmax_p95": float(feats["tmax_p95"]),
                        "rh_media": float(feats["rh_media"]),
                        "vento_medio": float(feats["vento_medio"]),
                        # label teacher
                        "score_teacher": float(score_teacher),
                        "flags": ";".join(flags) if flags else "",
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Gera dataset ML (teacher) para AgroClima Cultiva.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--area_m2", type=float, default=5000.0)
    p.add_argument(
        "--objetivo",
        type=str,
        default="baixo_risco",
        choices=["renda_rapida", "baixo_risco", "seguranca_alimentar", "diversificacao"],
    )
    p.add_argument("--municipio", type=str, default=None)
    p.add_argument("--uf", type=str, default=None)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--timezone", type=str, default="America/Sao_Paulo")
    p.add_argument("--catalog", type=str, default=None, help="Caminho opcional do catalog_v1.json")
    p.add_argument("--out", type=str, default="agroclima_cultiva_dataset.csv")
    args = p.parse_args()

    # datas
    # -1 dia para evitar "hoje" incompleto, e garantir janela fechada
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=max(30, int(args.days)))

    # catálogo
    if args.catalog:
        crops = load_catalog_from_json(path=Path(args.catalog))
    else:
        crops = load_catalog_from_json()

    # histórico
    df_hist = fetch_daily_history_openmeteo(
        lat=float(args.lat),
        lon=float(args.lon),
        start_date=start,
        end_date=end,
        timezone=str(args.timezone),
    )

    if df_hist.empty:
        raise RuntimeError("Open-Meteo retornou histórico vazio para o período solicitado.")

    # dataset
    df_ds = build_dataset(
        lat=float(args.lat),
        lon=float(args.lon),
        area_m2=float(args.area_m2),
        objetivo=str(args.objetivo),
        municipio=args.municipio,
        uf=args.uf,
        df_hist=df_hist,
        crops=crops,
        window_lens=(7, 14),
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ds.to_csv(out_path, index=False, encoding="utf-8")

    print(f"OK: dataset gerado com {len(df_ds):,} linhas.")
    print(f"Arquivo: {out_path}")
    print("Colunas:", ", ".join(df_ds.columns.tolist()))


if __name__ == "__main__":
    main()

