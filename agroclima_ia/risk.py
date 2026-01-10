# agroclima_ia/risk.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def calculate_pest_risk(
    df_forecast: pd.DataFrame,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calcula o risco fitossanitário principal baseado na cultura.

    Atualmente implementado para Ferrugem da Soja (Phakopsora pachyrhizi),
    usando condições climáticas de temperatura, umidade relativa e, quando
    disponível, chuva recente como proxy de molhamento foliar.

    Saída:
        - df com uma coluna adicional:
            pest_risk ∈ { "RISCO_BAIXO",
                           "RISCO_ATENÇÃO",
                           "RISCO_ALTO_FERRUGEM" }
    """
    df = df_forecast.copy()
    cultura = str(meta.get("cultura", "")).lower()

    # 1) Se não for soja, por enquanto assume risco baixo genérico.
    if "soja" not in cultura:
        df["pest_risk"] = "RISCO_BAIXO"
        return df

    # 2) Pré-condições de colunas climáticas mínimas
    required_cols = ["om_temp2m_max", "om_rh2m_max"]
    if not all(col in df.columns for col in required_cols):
        df["pest_risk"] = "RISCO_BAIXO"
        return df

    # 3) Cálculo de proxies climáticos
    # --------------------------------
    # T média aproximada (se não temos Tmin, usamos Tmax - 5 °C como proxy)
    df["Tmean_proxy"] = df["om_temp2m_max"] - 5.0

    # Faixas de temperatura
    T_IDEAL = (df["Tmean_proxy"] >= 18.0) & (df["Tmean_proxy"] <= 28.0)
    T_MARGINAL_ALTA = (df["Tmean_proxy"] > 28.0) & (df["Tmean_proxy"] <= 32.0)

    # Umidade relativa
    UR_ALTA = df["om_rh2m_max"] >= 90.0      # muito favorável à ferrugem
    UR_BOA = (df["om_rh2m_max"] >= 80.0) & (df["om_rh2m_max"] < 90.0)

    # Chuva recente (3 dias) – se não houver coluna de chuva, assume 0.
    if "om_precipitation_sum" in df.columns:
        df["rain_rolling_3d"] = (
            df["om_precipitation_sum"]
            .fillna(0.0)
            .rolling(window=3, min_periods=1)
            .sum()
        )
    else:
        df["rain_rolling_3d"] = 0.0

    # 4) Condições diárias de risco
    # ------------------------------

    # Condição MUITO favorável (alto risco por dia):
    # - T ideal E UR alta
    cond_muito_favoravel = T_IDEAL & UR_ALTA

    # Condição favorável / atenção:
    # - T ideal E UR boa
    # - OU T marginal alta E UR alta
    cond_favoravel = (T_IDEAL & UR_BOA) | (T_MARGINAL_ALTA & UR_ALTA)

    # Molhamento foliar relevante (chuva acumulada ≥ 10 mm em 3 dias)
    cond_molhamento = df["rain_rolling_3d"] >= 10.0

    # 5) Acúmulo de dias favoráveis (janela móvel)
    # --------------------------------------------
    # Contagem de dias MUITO favoráveis em janela de 2 dias
    muito_fav_rolling_2d = cond_muito_favoravel.astype(int).rolling(
        window=2, min_periods=1
    ).sum()

    # Contagem de dias favoráveis (atenção) em janela de 2 dias
    fav_rolling_2d = cond_favoravel.astype(int).rolling(
        window=2, min_periods=1
    ).sum()

    # 6) Classificação final de risco
    # -------------------------------
    # Critérios (ordem importa):
    # - RISCO_ALTO_FERRUGEM:
    #     • Pelo menos 2 dias MUITO favoráveis na janela de 2 dias
    #       OU
    #     • Condição muito favorável E molhamento relevante
    #
    # - RISCO_ATENÇÃO:
    #     • 1 dia muito favorável (isolado)
    #       OU
    #     • Pelo menos 2 dias favoráveis em janela de 2 dias
    #
    # - Caso contrário: RISCO_BAIXO
    cond_alto = (muito_fav_rolling_2d >= 2) | (cond_muito_favoravel & cond_molhamento)
    cond_atencao = (~cond_alto) & (
        cond_muito_favoravel | (fav_rolling_2d >= 2)
    )

    df["pest_risk"] = np.select(
        [
            cond_alto,
            cond_atencao,
        ],
        [
            "RISCO_ALTO_FERRUGEM",
            "RISCO_ATENÇÃO",
        ],
        default="RISCO_BAIXO",
    )

    # 7) Limpeza e retorno
    return df.drop(columns=["Tmean_proxy", "rain_rolling_3d"])
