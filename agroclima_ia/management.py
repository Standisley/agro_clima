# agroclima_ia/management.py

from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from .spraying import calculate_spraying_window


# =============================================================================
# PERFIS DE MANEJO POR CULTURA / SISTEMA
# =============================================================================
# Aqui definimos thresholds de manejo por cultura (e sistema, quando relevante).
# A ideia é: se quiser ajustar o comportamento para soja, trigo, arroz, basta
# mexer neste dicionário, SEM alterar o resto do código.
# -----------------------------------------------------------------------------

MANAGEMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # SOJA – sequeiro típico de Cerrado
    # -------------------------------------------------------------------------
    "soja": {
        "planting": {
            # Balanço hídrico (water_balance_mm) – janela boa/intermediária/ruim
            "wb_ok_min": -5.0,
            "wb_ok_max": 20.0,
            "wb_attention_min": -20.0,
            "wb_attention_max": 30.0,
        },
        "nitrogen": {
            # Chuva diária (mm) para N em cobertura
            # Muito seco = risco ; moderado = atenção ; moderado+chuva leve = OK ; chuva muito alta = risco
            "rain_min_ok": 3.0,
            "rain_max_ok": 20.0,
            "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 36.0,  # Tmax >= 36°C = estresse forte
    },

    # -------------------------------------------------------------------------
    # TRIGO – regiões mais frias (Sul)
    # -------------------------------------------------------------------------
    "trigo": {
        "planting": {
            # Trigo é mais sensível a encharcamento e déficit em perfilhamento
            "wb_ok_min": -3.0,
            "wb_ok_max": 15.0,
            "wb_attention_min": -15.0,
            "wb_attention_max": 25.0,
        },
        "nitrogen": {
            # Trigo usa muito N em cobertura – evitar tanto solo seco como chuva torrencial
            "rain_min_ok": 2.0,
            "rain_max_ok": 15.0,
            "rain_max_attention": 30.0,
        },
        "heat_stress_threshold": 32.0,  # Acima de ~32°C já é bem ruim em muitas fases
    },

    # -------------------------------------------------------------------------
    # ARROZ – diferenciar irrigado/alagado vs sequeiro
    # -------------------------------------------------------------------------
    "arroz_alagado": {
        "planting": {
            # Em sistemas alagados, plantio convencional direto na lâmina é diferente;
            # aqui usamos algo genérico e mais permissivo.
            "wb_ok_min": -10.0,
            "wb_ok_max": 30.0,
            "wb_attention_min": -25.0,
            "wb_attention_max": 40.0,
        },
        "nitrogen": {
            # N em arroz irrigado normalmente precisa de lâmina controlada;
            # chuva perde relevância, mas mantemos algo suave.
            "rain_min_ok": 0.0,
            "rain_max_ok": 25.0,
            "rain_max_attention": 50.0,
        },
        "heat_stress_threshold": 35.0,
    },

    "arroz_sequeiro": {
        "planting": {
            "wb_ok_min": -5.0,
            "wb_ok_max": 20.0,
            "wb_attention_min": -20.0,
            "wb_attention_max": 30.0,
        },
        "nitrogen": {
            "rain_min_ok": 3.0,
            "rain_max_ok": 20.0,
            "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 34.0,
    },

    # -------------------------------------------------------------------------
    # DEFAULT – usado quando a cultura não está explicitamente mapeada
    # -------------------------------------------------------------------------
    "default": {
        "planting": {
            "wb_ok_min": -5.0,
            "wb_ok_max": 20.0,
            "wb_attention_min": -20.0,
            "wb_attention_max": 30.0,
        },
        "nitrogen": {
            "rain_min_ok": 3.0,
            "rain_max_ok": 20.0,
            "rain_max_attention": 35.0,
        },
        "heat_stress_threshold": 36.0,
    },
}


# =============================================================================
# HELPERS PARA ESCOLHER PERFIL
# =============================================================================


def _get_management_profile(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide qual perfil de manejo usar com base em:
      - cultura (meta['cultura'])
      - sistema (meta['sistema'] - ex: 'alagado', 'sequeiro')
    """
    cultura = (meta.get("cultura") or "").lower()
    sistema = (meta.get("sistema") or "").lower()

    # arroz irrigado vs sequeiro
    if "arroz" in cultura and "alagado" in sistema:
        return MANAGEMENT_PROFILES["arroz_alagado"]
    if "arroz" in cultura:
        return MANAGEMENT_PROFILES["arroz_sequeiro"]

    if "soja" in cultura:
        return MANAGEMENT_PROFILES["soja"]
    if "trigo" in cultura:
        return MANAGEMENT_PROFILES["trigo"]

    # fallback genérico
    return MANAGEMENT_PROFILES["default"]


# =============================================================================
# CLASSIFICAÇÃO DE PLANTIO
# =============================================================================


def _classify_planting_row(
    row: pd.Series,
    plant_cfg: Dict[str, float],
) -> str:
    """
    Classifica plantio em PLANTIO_OK / PLANTIO_ATENCAO / PLANTIO_RUIM
    a partir do balanço hídrico diário (water_balance_mm).
    """
    wb = row.get("water_balance_mm", np.nan)
    if pd.isna(wb):
        return "PLANTIO_ATENCAO"

    wb_ok_min = plant_cfg["wb_ok_min"]
    wb_ok_max = plant_cfg["wb_ok_max"]
    wb_att_min = plant_cfg["wb_attention_min"]
    wb_att_max = plant_cfg["wb_attention_max"]

    if wb_ok_min <= wb <= wb_ok_max:
        return "PLANTIO_OK"
    if wb_att_min <= wb <= wb_att_max:
        return "PLANTIO_ATENCAO"
    return "PLANTIO_RUIM"


# =============================================================================
# CLASSIFICAÇÃO DE NITROGÊNIO (N em cobertura)
# =============================================================================


def _classify_nitrogen_row(
    row: pd.Series,
    nitro_cfg: Dict[str, float],
) -> str:
    """
    Classifica a janela diária para N:
      - N_OK
      - N_ATENCAO
      - N_RISCO

    Base: chuva diária prevista (y_ensemble_mm) e, indiretamente, saldo hídrico.
    """
    rain = row.get("y_ensemble_mm", np.nan)
    if pd.isna(rain):
        return "N_ATENCAO"

    rain_min_ok = nitro_cfg["rain_min_ok"]
    rain_max_ok = nitro_cfg["rain_max_ok"]
    rain_max_att = nitro_cfg["rain_max_attention"]

    # Muito seco → risco (não incorpora, fertilizante "fica na superfície")
    if rain < rain_min_ok:
        return "N_RISCO"

    # Faixa boa de chuva → boa incorporação
    if rain_min_ok <= rain <= rain_max_ok:
        return "N_OK"

    # Chuva moderada-alta → atenção (pode incorporar bem, mas risco de perdas)
    if rain_max_ok < rain <= rain_max_att:
        return "N_ATENCAO"

    # Chuva muito forte → risco de lixiviação / escorrimento
    return "N_RISCO"


# =============================================================================
# CLASSIFICAÇÃO DE ESTRESSE TÉRMICO
# =============================================================================


def _calculate_heat_stress(
    df: pd.DataFrame,
    heat_threshold: float,
) -> pd.Series:
    """
    Retorna uma Series booleana indicando dias com Tmax >= limiar de estresse.
    Usa a coluna 'om_temp2m_max' (Open-Meteo) se disponível.
    """
    if "om_temp2m_max" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["om_temp2m_max"] >= float(heat_threshold)


# =============================================================================
# FUNÇÃO PRINCIPAL DE MANEJO
# =============================================================================


def apply_management_windows(
    df_forecast: pd.DataFrame,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, str]:
    """
    Aplica regras de manejo (pulverização, plantio, N, estresse térmico)
    usando perfis específicos por cultura/sistema.

    Retorna:
      - df_mgmt: DataFrame de forecast com colunas adicionais
      - planting_status_resumo: string resumindo a "qualidade" da janela de plantio
    """
    df = df_forecast.copy()

    # 1. Seleciona o perfil de manejo (soja / trigo / arroz / default)
    profile = _get_management_profile(meta)
    plant_cfg = profile["planting"]
    nitro_cfg = profile["nitrogen"]
    heat_thr = profile["heat_stress_threshold"]

    # 2. Pulverização (usa função própria em spraying.py)
    #    – já leva em conta chuva, vento, temp, UR
    try:
        df = calculate_spraying_window(df)
    except Exception as e:
        # Se der algum problema, mantemos sem coluna e tocamos o fluxo
        print(f"[management] Aviso: erro ao calcular spray_status: {e}")

    # 3. Plantio – usa o balanço hídrico diário e perfil de cultura
    df["planting_status"] = df.apply(
        lambda row: _classify_planting_row(row, plant_cfg),
        axis=1,
    )

    # 4. Nitrogênio em cobertura – usa chuva diária + perfil da cultura
    df["nitrogen_status"] = df.apply(
        lambda row: _classify_nitrogen_row(row, nitro_cfg),
        axis=1,
    )

    # 5. Estresse térmico – baseado em Tmax e limiar da cultura
    df["heat_stress"] = _calculate_heat_stress(df, heat_thr)

    # 6. Resumo da janela de plantio (para o texto do relatório)
    n_ok = int((df["planting_status"] == "PLANTIO_OK").sum())
    n_att = int((df["planting_status"] == "PLANTIO_ATENCAO").sum())
    n_bad = int((df["planting_status"] == "PLANTIO_RUIM").sum())

    if n_ok > 0:
        planting_summary = "PLANTIO_OK"
    elif n_att > 0:
        planting_summary = "PLANTIO_ATENCAO"
    else:
        planting_summary = "PLANTIO_RUIM"

    return df, planting_summary


