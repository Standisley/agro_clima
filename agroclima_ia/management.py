# agroclima_ia/management.py

from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from .spraying import calculate_spraying_window
# --- NOVO: Importa a inteligência agronômica que respeita estágios ---
from .agronomy import classify_planting_window, classify_nitrogen_window


# =============================================================================
# PERFIS DE MANEJO POR CULTURA / SISTEMA
# =============================================================================
# Mantemos os perfis para configurações como "heat_stress_threshold".
# As configurações de plantio/nitrogênio aqui ficam como referência legado,
# pois a lógica principal agora é delegada ao agronomy.py para maior inteligência.
# -----------------------------------------------------------------------------

MANAGEMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # SOJA – sequeiro típico de Cerrado
    # -------------------------------------------------------------------------
    "soja": {
        "planting": {
            "wb_ok_min": -5.0, "wb_ok_max": 20.0,
            "wb_attention_min": -20.0, "wb_attention_max": 30.0,
        },
        "nitrogen": {
            "rain_min_ok": 3.0, "rain_max_ok": 20.0, "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 36.0,  # Tmax >= 36°C = estresse forte
    },

    # -------------------------------------------------------------------------
    # TRIGO – regiões mais frias (Sul)
    # -------------------------------------------------------------------------
    "trigo": {
        "planting": {
            "wb_ok_min": -3.0, "wb_ok_max": 15.0,
            "wb_attention_min": -15.0, "wb_attention_max": 25.0,
        },
        "nitrogen": {
            "rain_min_ok": 2.0, "rain_max_ok": 15.0, "rain_max_attention": 30.0,
        },
        "heat_stress_threshold": 32.0,
    },

    # -------------------------------------------------------------------------
    # ARROZ – diferenciar irrigado/alagado vs sequeiro
    # -------------------------------------------------------------------------
    "arroz_alagado": {
        "planting": {
            "wb_ok_min": -10.0, "wb_ok_max": 30.0,
            "wb_attention_min": -25.0, "wb_attention_max": 40.0,
        },
        "nitrogen": {
            "rain_min_ok": 0.0, "rain_max_ok": 25.0, "rain_max_attention": 50.0,
        },
        "heat_stress_threshold": 35.0,
    },

    "arroz_sequeiro": {
        "planting": {
            "wb_ok_min": -5.0, "wb_ok_max": 20.0,
            "wb_attention_min": -20.0, "wb_attention_max": 30.0,
        },
        "nitrogen": {
            "rain_min_ok": 3.0, "rain_max_ok": 20.0, "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 34.0,
    },

    # -------------------------------------------------------------------------
    # DEFAULT
    # -------------------------------------------------------------------------
    "default": {
        "planting": {
            "wb_ok_min": -5.0, "wb_ok_max": 20.0,
            "wb_attention_min": -20.0, "wb_attention_max": 30.0,
        },
        "nitrogen": {
            "rain_min_ok": 3.0, "rain_max_ok": 20.0, "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 36.0,
    },
}


# =============================================================================
# HELPERS PARA ESCOLHER PERFIL
# =============================================================================

def _get_management_profile(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide qual perfil de manejo usar com base em cultura e sistema.
    """
    cultura = (meta.get("cultura") or "").lower()
    sistema = (meta.get("sistema") or "").lower()

    if "arroz" in cultura and "alagado" in sistema:
        return MANAGEMENT_PROFILES["arroz_alagado"]
    if "arroz" in cultura:
        return MANAGEMENT_PROFILES["arroz_sequeiro"]
    if "soja" in cultura:
        return MANAGEMENT_PROFILES["soja"]
    if "trigo" in cultura:
        return MANAGEMENT_PROFILES["trigo"]

    return MANAGEMENT_PROFILES["default"]


# =============================================================================
# CLASSIFICAÇÃO DE ESTRESSE TÉRMICO (Mantido Local)
# =============================================================================

def _calculate_heat_stress(
    df: pd.DataFrame,
    heat_threshold: float,
) -> pd.Series:
    """
    Retorna uma Series booleana indicando dias com Tmax >= limiar de estresse.
    """
    if "om_temp2m_max" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["om_temp2m_max"] >= float(heat_threshold)


# =============================================================================
# FUNÇÃO PRINCIPAL DE MANEJO (INTEGRADA)
# =============================================================================

def apply_management_windows(
    df_forecast: pd.DataFrame,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, str]:
    """
    Aplica regras de manejo (pulverização, plantio, N, estresse térmico).
    
    CORREÇÃO: Agora integra com agronomy.py para respeitar estágio fenológico.
    """
    df = df_forecast.copy()

    # 1. Seleciona o perfil para dados gerais (ex: limiar térmico)
    profile = _get_management_profile(meta)
    heat_thr = profile["heat_stress_threshold"]
    
    # Extrai metadados essenciais para as novas funções
    cultura = meta.get("cultura", "")
    solo = meta.get("solo", "")
    estagio = meta.get("estagio_fenologico", "")

    # 2. Pulverização (spraying.py)
    try:
        df = calculate_spraying_window(df)
    except Exception as e:
        print(f"[management] Aviso: erro ao calcular spray_status: {e}")

    # 3. Plantio (agronomy.py) 
    # --> Substitui a lógica antiga para garantir a trava de "Ciclo em Andamento"
    df, planting_status_val = classify_planting_window(
        df, 
        cultura=cultura, 
        solo=solo, 
        estagio=estagio
    )

    # 4. Nitrogênio (agronomy.py)
    # --> Substitui a lógica antiga para garantir a trava de "Soja" e "Fim de Ciclo"
    df = classify_nitrogen_window(
        df,
        cultura=cultura,
        estagio=estagio
    )

    # 5. Estresse térmico (Mantido localmente com o perfil)
    df["heat_stress"] = _calculate_heat_stress(df, heat_thr)

    # 6. Resumo da janela de plantio
    # Se o agronomy retornou uma string (ex: "CICLO_EM_ANDAMENTO" ou "PLANTIO_BOM"), usamos ela.
    if isinstance(planting_status_val, str):
        planting_summary = planting_status_val
    else:
        # Fallback de contagem clássica se necessário
        # Aceita tanto PLANTIO_OK (legado) quanto PLANTIO_BOM (novo)
        n_ok = int((df["planting_status"].isin(["PLANTIO_BOM", "PLANTIO_OK"])).sum())
        if n_ok > 0:
            planting_summary = "PLANTIO_OK"
        else:
            planting_summary = "PLANTIO_RUIM"

    return df, planting_summary


