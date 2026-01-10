# agroclima_ia/anomalies.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def detect_agro_anomalies(
    df: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detecta anomalias agroclimáticas na janela de previsão.

    Hoje a detecção é feita apenas na janela atual (7 dias), mas a ideia
    é evoluir para usar também histórico de 10 anos (seca em janeiro, etc.).

    Retorna um dicionário com:
      - has_critical: bool
      - messages: lista de strings descritivas
      - tags: lista de rótulos curtos das anomalias
      - summary: texto pronto para impressão no terminal
    """
    if meta is None:
        meta = {}

    anomalies: Dict[str, Any] = {
        "has_critical": False,
        "messages": [],
        "tags": [],
    }

    if df.empty:
        anomalies["summary"] = "Nenhuma anomalia climática crítica identificada (sem dados na janela)."
        return anomalies

    # Garante colunas básicas para evitar KeyError
    rain = df["y_ensemble_mm"] if "y_ensemble_mm" in df.columns else pd.Series(
        [0.0] * len(df), index=df.index
    )

    wb = df["water_balance_mm"] if "water_balance_mm" in df.columns else pd.Series(
        [np.nan] * len(df), index=df.index
    )

    tmax = df["om_temp2m_max"] if "om_temp2m_max" in df.columns else pd.Series(
        [np.nan] * len(df), index=df.index
    )

    # ------------------------------------------------------------------
    # 1) “Estiagem” na janela de previsão (sequência de dias muito secos)
    # ------------------------------------------------------------------
    # Aqui ainda é só na janela de 7 dias. No futuro, isso vai olhar o
    # histórico de 10 anos (seca de 15 dias em janeiro, etc.).
    dry_threshold = 0.5  # mm/dia
    dry_days = (rain < dry_threshold).sum()

    if dry_days >= 5:
        anomalies["has_critical"] = True
        anomalies["messages"].append(
            f"Sequência de {dry_days} dias com chuva prevista < {dry_threshold:.1f} mm (sinal de estiagem na janela)."
        )
        anomalies["tags"].append("dry_spell_horizon")

    # ------------------------------------------------------------------
    # 2) Balanço hídrico muito negativo em vários dias
    # ------------------------------------------------------------------
    if not wb.isna().all():
        very_negative_wb = (wb < -8.0).sum()
        if very_negative_wb >= 3:
            anomalies["has_critical"] = True
            anomalies["messages"].append(
                f"{very_negative_wb} dias com balanço hídrico diário < -8 mm (déficit hídrico forte e repetitivo)."
            )
            anomalies["tags"].append("strong_deficit_horizon")

    # ------------------------------------------------------------------
    # 3) Calor extremo (Tmax muito alta)
    # ------------------------------------------------------------------
    if not tmax.isna().all():
        very_hot_days = (tmax > 36.0).sum()
        if very_hot_days >= 1:
            anomalies["has_critical"] = True
            anomalies["messages"].append(
                f"Pelo menos {very_hot_days} dia(s) com temperatura máxima > 36 °C na janela de previsão."
            )
            anomalies["tags"].append("extreme_heat_horizon")

    # ------------------------------------------------------------------
    # 4) Monta resumo textual
    # ------------------------------------------------------------------
    if anomalies["has_critical"]:
        summary_lines = ["Anomalias climáticas críticas identificadas:"]
        for msg in anomalies["messages"]:
            summary_lines.append(f"- {msg}")
        anomalies["summary"] = "\n".join(summary_lines)
    else:
        anomalies["summary"] = (
            "Nenhuma anomalia climática crítica identificada no horizonte desta previsão."
        )

    return anomalies


