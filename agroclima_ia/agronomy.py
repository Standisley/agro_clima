# agroclima_ia/agronomy.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# Atualizei os tipos para incluir os novos status de "trava agronômica"
PlantingStatus = Literal["PLANTIO_BOM", "PLANTIO_ATENCAO", "PLANTIO_RUIM", "CICLO_EM_ANDAMENTO"]
NitrogenStatus = Literal["N_OK", "N_ATENCAO", "N_RISCO", "N_NAO_SE_APLICA"]


def classify_planting_window(
    df_forecast: pd.DataFrame,
    cultura: str | None = None,
    solo: str | None = None,
    estagio: str | None = None,  # <--- Novo parâmetro para filtrar fase
) -> tuple[pd.DataFrame, PlantingStatus]:
    """
    Classifica a janela de plantio.
    CORREÇÃO: Verifica se a cultura já está implantada para não recomendar plantio
    no meio do ciclo.
    """

    df = df_forecast.copy()
    
    # -------------------------------------------------------------------------
    # 1. TRAVA DE ESTÁGIO FENOLÓGICO (Correção Agronômica)
    # -------------------------------------------------------------------------
    # Se a planta já está no campo, não faz sentido calcular janela de plantio.
    estagio_lower = str(estagio or "").strip().lower()
    
    # Lista de fases que indicam que o plantio JÁ ACONTECEU
    fases_implantadas = [
        "v", "vegetativo", "perfilhamento", "crescimento",
        "r", "reprodutivo", "flor", "enchimento", "maturacao", 
        "colheita", "frutificacao", "espigamento"
    ]
    
    # Se encontrar qualquer termo acima no estágio, trava o status.
    if any(fase in estagio_lower for fase in fases_implantadas):
        planting_status: PlantingStatus = "CICLO_EM_ANDAMENTO"
        df["planting_status"] = planting_status
        return df, planting_status

    # -------------------------------------------------------------------------
    # 2. Lógica Climatológica Original (Chuva vs ET0)
    # -------------------------------------------------------------------------
    rain_col = "y_ensemble_mm"
    et0_col = "om_et0_fao_mm"

    if rain_col not in df.columns:
        raise ValueError("DataFrame de previsão precisa da coluna 'y_ensemble_mm' para plantio.")

    total_rain = float(df[rain_col].sum())
    total_et0 = float(df[et0_col].sum()) if et0_col in df.columns else None

    ratio = None
    if total_et0 is not None and total_et0 > 0:
        ratio = total_rain / total_et0

    # Heurística original mantida:
    if ratio is None:
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
            # Excesso de chuva/encharcamento
            planting_status = "PLANTIO_ATENCAO"

    df["planting_status"] = planting_status

    return df, planting_status


def classify_nitrogen_window(
    df_forecast: pd.DataFrame,
    cultura: str | None = None,   # <--- Necessário para regra da Soja
    estagio: str | None = None,   # <--- Necessário para regra de final de ciclo
) -> pd.DataFrame:
    """
    Classifica janelas para adubação nitrogenada.
    CORREÇÃO: Bloqueia N para Soja e para fases finais (enchimento/maturação).
    """

    df = df_forecast.copy()

    if "y_ensemble_mm" not in df.columns:
        raise ValueError("DataFrame de previsão precisa da coluna 'y_ensemble_mm' para N.")

    # -------------------------------------------------------------------------
    # 1. TRAVAS AGRONÔMICAS (Correção)
    # -------------------------------------------------------------------------
    cultura_lower = str(cultura or "").strip().lower()
    estagio_lower = str(estagio or "").strip().lower()

    # Regra A: Soja praticamente não usa N mineral (Fixação Biológica)
    if "soja" in cultura_lower:
        df["nitrogen_status"] = "N_NAO_SE_APLICA"
        return df

    # Regra B: Não se aplica N no final do ciclo (Enchimento/Maturação)
    fases_finais = ["enchimento", "maturacao", "colheita", "r5", "r6", "r7", "r8"]
    if any(f in estagio_lower for f in fases_finais):
        df["nitrogen_status"] = "N_NAO_SE_APLICA"
        return df

    # -------------------------------------------------------------------------
    # 2. Lógica Climatológica Original
    # -------------------------------------------------------------------------
    rain = df["y_ensemble_mm"].to_numpy()
    n = len(rain)

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
            # Estresse térmico = alto risco de volatilização
            status: NitrogenStatus = "N_RISCO"
        else:
            if rain_window < 2.0:
                # Muito seco = não incorpora
                status = "N_RISCO"
            elif rain_window > 40.0:
                # Muita chuva = lixiviação
                status = "N_RISCO"
            elif 5.0 <= rain_window <= 25.0:
                status = "N_OK"
            else:
                status = "N_ATENCAO"

        nitrogen_status.append(status)

    df["nitrogen_status"] = nitrogen_status

    return df
