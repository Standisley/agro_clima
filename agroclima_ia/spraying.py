# agroclima_ia/spraying.py

import pandas as pd
from typing import Literal

SprayStatus = Literal["VERDE", "AMARELO", "VERMELHO"]


def classify_spray_row(row) -> SprayStatus:
    rain = row.get("y_ensemble_mm")
    temp = row.get("om_temp2m_max")
    wind = row.get("om_windspeed10_max")
    rh   = row.get("om_rh2m_max")

    # --- Tratamento de NaN ---
    rain_is_nan = pd.isna(rain)
    temp_is_nan = pd.isna(temp)
    wind_is_nan = pd.isna(wind)
    rh_is_nan   = pd.isna(rh)

    # Se a chuva for NaN, não dá pra decidir nada com segurança
    if rain_is_nan:
        return "VERMELHO"

    # Regras agronômicas básicas (ajuste depois se quiser):
    # - Chuva baixa: < 1 mm   (ok)
    # - Chuva moderada: 1–4 mm (atenção)
    # - Chuva alta: ≥ 4 mm (evitar)
    # - Vento ideal: até 10 km/h
    # - Vento limite: até 15 km/h
    # - Temp ideal: 18–32 ºC
    # - UR ideal: 55–95 %

    # 1) Situações claramente ruins (vermelho direto)
    if rain >= 4.0:
        return "VERMELHO"

    # 2) Só podemos ter VERDE se TODOS os dados climáticos estiverem presentes
    if not (temp_is_nan or wind_is_nan or rh_is_nan):
        if (
            rain < 1.0 and
            wind <= 10.0 and
            18.0 <= temp <= 32.0 and
            55.0 <= rh <= 95.0
        ):
            return "VERDE"

    # 3) Situações intermediárias (AMARELO):
    # - chuva baixa ou moderada
    # - vento não estourou muito
    # - ou falta alguma variável (NaN), mas chuva ainda é baixa
    if (
        rain < 2.0 and
        (wind_is_nan or wind <= 15.0)
    ):
        return "AMARELO"

    # 4) Todo o resto: VERMELHO
    return "VERMELHO"


def calculate_spraying_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas:
      - ds
      - y_ensemble_mm
      - om_temp2m_max
      - om_windspeed10_max
      - om_rh2m_max

    e devolve o mesmo DF com a coluna 'spray_status'
    preenchida com VERDE / AMARELO / VERMELHO.
    """

    required_cols = [
        "ds",
        "y_ensemble_mm",
        "om_temp2m_max",
        "om_windspeed10_max",
        "om_rh2m_max",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltando colunas no DataFrame para pulverização: {missing}")

    df = df.copy()
    df["spray_status"] = df.apply(classify_spray_row, axis=1)
    return df
