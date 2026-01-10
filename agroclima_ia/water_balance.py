# agroclima_ia/water_balance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SoilParams:
    """Parâmetros simplificados de solo para bucket model."""
    taw_mm: float  # Total Available Water (mm) na zona radicular efetiva
    p: float = 0.5  # fração facilmente disponível (RAW = p * TAW)
    init_frac: float = 0.70  # fração inicial do TAW no início do horizonte


def _norm_text(x: str) -> str:
    return str(x or "").strip().lower()


def _infer_soil_params(meta: Optional[Dict]) -> SoilParams:
    """
    Converte descrições textuais de solo (meta['solo']) em um TAW aproximado (mm).

    Observação:
    - Isso é um modelo simplificado. O TAW real depende de profundidade radicular,
      textura, densidade, capacidade de campo, PMP, etc.
    - Aqui usamos valores típicos para “zona efetiva” padrão e decisões operacionais.
    """
    meta = meta or {}
    solo = _norm_text(meta.get("solo", ""))

    # Heurística por palavras-chave
    # (valores típicos de TAW efetivo em mm; ajuste fino depois por cultura/raiz)
    if any(k in solo for k in ["arenoso", "areia", "sandy"]):
        taw = 70.0
    elif any(k in solo for k in ["argiloso", "clay", "barro"]):
        taw = 140.0
    elif any(k in solo for k in ["argilo-arenoso", "franco", "loam", "misto"]):
        taw = 110.0
    elif "gleissolo" in solo:
        taw = 120.0
    else:
        taw = 110.0  # default prudente

    # Ajuste leve por sistema (sequeiro tende a precisar “sensibilidade maior”)
    sistema = _norm_text(meta.get("sistema", ""))
    if any(k in sistema for k in ["alagado", "irrigado", "inundado"]):
        # Em arroz alagado, a dinâmica é outra; mantemos bucket “alto” e evitamos déficit.
        # Ainda assim calculamos saldo P-ET0 para diagnóstico.
        return SoilParams(taw_mm=taw, p=0.7, init_frac=0.9)

    # Ajuste leve por cultura (opcional; conservador)
    cultura = _norm_text(meta.get("cultura", ""))
    if "banana" in cultura:
        # banana é sensível a déficit; p um pouco menor (RAW menor -> alerta mais cedo)
        return SoilParams(taw_mm=taw, p=0.45, init_frac=0.75)

    return SoilParams(taw_mm=taw, p=0.5, init_frac=0.70)


def compute_water_balance_bucket(
    df_forecast: pd.DataFrame,
    meta: Optional[Dict] = None,
    rain_col: str = "y_ensemble_mm",
    et0_col: str = "om_et0_fao_mm",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Balanço hídrico em modelo “bucket” (reservatório de água no solo).

    Requer:
      - coluna de datas: 'ds'
      - chuva (mm): rain_col
      - ET0 (mm): et0_col

    Retorna:
      - df_out: df_forecast com colunas adicionais:
          water_storage_mm: água armazenada no bucket (0..TAW)
          raw_mm: água facilmente disponível (RAW = p*TAW)
          taw_mm: água total disponível (TAW)
          deficit_mm: déficit relativo ao bucket (mm)
          excess_mm: excesso/percolação (mm) quando storage ultrapassa TAW
          water_balance_mm: saldo diário chuva - ET0 (mm)
          water_status: texto simples (OK/ATENCAO/DEFICIT/INDISPONIVEL)
      - summary: dict com agregados (chuva_total, et0_total, saldo_total, etc.)
    """
    if df_forecast is None or df_forecast.empty:
        return df_forecast, {
            "ok": False,
            "reason": "df_forecast vazio",
        }

    df = df_forecast.copy()

    if "ds" not in df.columns:
        # não “quebra” o pipeline; só sinaliza indisponível
        df["water_balance_mm"] = np.nan
        df["water_storage_mm"] = np.nan
        df["deficit_mm"] = np.nan
        df["excess_mm"] = np.nan
        df["water_status"] = "INDISPONIVEL"
        return df, {"ok": False, "reason": "coluna ds ausente"}

    df["ds"] = pd.to_datetime(df["ds"])

    # Garante colunas numéricas
    if rain_col not in df.columns:
        df[rain_col] = np.nan
    if et0_col not in df.columns:
        df[et0_col] = np.nan

    rain = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0).astype(float)
    et0 = pd.to_numeric(df[et0_col], errors="coerce")

    # Se ET0 for todo NaN, marcamos indisponível (não inventamos)
    if et0.isna().all():
        df["water_balance_mm"] = rain.values * 1.0
        df["water_storage_mm"] = np.nan
        df["raw_mm"] = np.nan
        df["taw_mm"] = np.nan
        df["deficit_mm"] = np.nan
        df["excess_mm"] = np.nan
        df["water_status"] = "INDISPONIVEL"
        return df, {"ok": False, "reason": "ET0 indisponível"}

    # Para balanço, NaN residual de ET0 vira 0 (evita quebrar somas e regra diária)
    et0 = et0.fillna(0.0).astype(float)

    params = _infer_soil_params(meta)
    taw = float(params.taw_mm)
    raw = float(params.p * taw)

    # Estado inicial
    storage = float(np.clip(params.init_frac * taw, 0.0, taw))

    stor_list = []
    deficit_list = []
    excess_list = []
    status_list = []

    for p_mm, et0_mm in zip(rain.values, et0.values):
        # saldo “climático” diário
        wb = float(p_mm - et0_mm)

        # dinâmica do bucket
        new_storage = storage + wb

        excess = 0.0
        if new_storage > taw:
            excess = float(new_storage - taw)
            new_storage = taw

        if new_storage < 0.0:
            new_storage = 0.0

        # déficit relativo ao “cheio” (TAW)
        deficit = float(taw - new_storage)

        # classificação simples
        # - OK: acima de RAW (a planta tem água facilmente disponível)
        # - ATENCAO: entre RAW e (RAW - 20% TAW) -> começando a restringir
        # - DEFICIT: abaixo de (TAW - (TAW - RAW)) = RAW? (ou mais severo)
        if et0_mm == 0.0 and p_mm == 0.0:
            status = "OK"
        else:
            if new_storage >= raw:
                status = "OK"
            elif new_storage >= 0.30 * taw:
                status = "ATENCAO"
            else:
                status = "DEFICIT"

        stor_list.append(float(new_storage))
        deficit_list.append(deficit)
        excess_list.append(excess)
        status_list.append(status)

        storage = new_storage

    df["water_balance_mm"] = (rain - et0).round(3)
    df["taw_mm"] = taw
    df["raw_mm"] = raw
    df["water_storage_mm"] = np.round(stor_list, 3)
    df["deficit_mm"] = np.round(deficit_list, 3)
    df["excess_mm"] = np.round(excess_list, 3)
    df["water_status"] = status_list

    # Agregados para relatório
    summary = {
        "ok": True,
        "chuva_total_mm": float(rain.sum()),
        "et0_total_mm": float(et0.sum()),
        "saldo_total_mm": float((rain - et0).sum()),
        "storage_ini_mm": float(np.clip(params.init_frac * taw, 0.0, taw)),
        "storage_fim_mm": float(stor_list[-1]) if stor_list else np.nan,
        "solo_taw_mm": taw,
        "solo_raw_mm": raw,
    }

    return df, summary
