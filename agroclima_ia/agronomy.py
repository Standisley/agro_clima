# agroclima_ia/agronomy.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

PlantingStatus = Literal["PLANTIO_BOM", "PLANTIO_ATENCAO", "PLANTIO_RUIM"]
NitrogenStatus = Literal["N_OK", "N_ATENCAO", "N_RISCO"]


def classify_planting_window(
    df_forecast: pd.DataFrame,
    cultura: str | None = None,
    solo: str | None = None,
) -> tuple[pd.DataFrame, PlantingStatus]:
    """
    Classifica a janela de plantio (no horizonte da previsão) em:
      - PLANTIO_BOM
      - PLANTIO_ATENCAO
      - PLANTIO_RUIM

    Regra simplificada (heurística climatológica):
    - Usamos chuva total, ET0 total e razão chuva/ET0.
    - Queremos evitar:
        * seca extrema (quase nada de chuva + ET0 alta)
        * encharcamento extremo (chuva >> ET0)
    - Para plantio em sequeiro, em geral:
        * Razão chuva/ET0 ~ 0.6–1.2 => janela boa ou com atenção
        * Muito abaixo disso => atenção/ruim (risco de falha de emergência)
        * Muito acima disso => atenção/ruim (risco de encharcamento em solo pesado)
    """

    df = df_forecast.copy()

    rain_col = "y_ensemble_mm"
    et0_col = "om_et0_fao_mm"

    if rain_col not in df.columns:
        raise ValueError("DataFrame de previsão precisa da coluna 'y_ensemble_mm' para plantio.")

    total_rain = float(df[rain_col].sum())
    total_et0 = float(df[et0_col].sum()) if et0_col in df.columns else None

    ratio = None
    if total_et0 is not None and total_et0 > 0:
        ratio = total_rain / total_et0

    # Heurística:
    # - ratio < 0.4  => PLANTIO_RUIM (pouca umidade prevista)
    # - 0.4 <= ratio < 0.7 => PLANTIO_ATENCAO
    # - 0.7 <= ratio <= 1.3 => PLANTIO_BOM (janela equilibrada)
    # - ratio > 1.3 => PLANTIO_ATENCAO (possível excesso / encharcamento em solo argiloso)
    if ratio is None:
        planting_status: PlantingStatus
        # fallback usando apenas chuva total
        if total_rain < 5:
            planting_status = "PLANTIO_RUIM"
        elif total_rain <= 20:
            planting_status = "PLANTIO_ATENCAO"
        else:
            planting_status = "PLANTIO_BOM"
    else:
        if ratio < 0.4:
            planting_status = "PLANTIO_RUIM"
        elif ratio < 0.7:
            planting_status = "PLANTIO_ATENCAO"
        elif ratio <= 1.3:
            planting_status = "PLANTIO_BOM"
        else:
            planting_status = "PLANTIO_ATENCAO"

    # Colocamos a mesma classificação em todas as linhas do horizonte
    df["planting_status"] = planting_status

    return df, planting_status


def classify_nitrogen_window(
    df_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classifica, dia a dia, janelas para adubação nitrogenada de cobertura.

    Ideia geral (simplificada):
    - Boa janela de N (N_OK):
        * chuva acumulada no dia + 2 dias seguintes entre 5 e 25 mm
        * sem estresse térmico no dia
    - Atenção (N_ATENCAO):
        * chuva acumulada entre 2 e 40 mm
        * OU leve estresse térmico, mas sem chuva extrema
    - Risco (N_RISCO):
        * chuva acumulada < 2 mm (risco de não incorporar o N)
        * OU chuva acumulada > 40 mm (risco de perda/lixiviação)
        * OU estresse térmico forte no dia (heat_stress = True)

    Usamos:
      - y_ensemble_mm (chuva)
      - heat_stress (se existir)
    """

    df = df_forecast.copy()

    if "y_ensemble_mm" not in df.columns:
        raise ValueError("DataFrame de previsão precisa da coluna 'y_ensemble_mm' para N.")

    rain = df["y_ensemble_mm"].to_numpy()
    n = len(rain)

    # Se não houver coluna heat_stress, consideramos False
    if "heat_stress" not in df.columns:
        heat = np.zeros(n, dtype=bool)
    else:
        heat = df["heat_stress"].fillna(False).to_numpy(dtype=bool)

    nitrogen_status: list[NitrogenStatus] = []

    for i in range(n):
        # chuva acumulada no dia i + 2 dias seguintes
        rain_window = float(rain[i : i + 3].sum())
        hs_today = bool(heat[i])

        if hs_today:
            # dia com estresse térmico considerado de maior risco para N
            status: NitrogenStatus = "N_RISCO"
        else:
            if rain_window < 2.0:
                status = "N_RISCO"
            elif rain_window > 40.0:
                status = "N_RISCO"
            elif 5.0 <= rain_window <= 25.0:
                status = "N_OK"
            else:
                status = "N_ATENCAO"

        nitrogen_status.append(status)

    df["nitrogen_status"] = nitrogen_status

    return df
