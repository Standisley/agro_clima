# agroclima_ia/et0.py

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import requests
import streamlit as st  # <--- IMPORT NOVO OBRIGATÓRIO


def _to_date(x: Any) -> dt.date:
    """
    Converte entrada (date, datetime, Timestamp, str) para datetime.date.
    """
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"[et0] Não foi possível converter '{x}' para data.") from exc


# --- OTIMIZAÇÃO: Cache de 1 hora (3600s) para evitar chamadas repetidas à API ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_et0_fao_daily(
    lat: float,
    lon: float,
    start_date: Any,
    end_date: Any,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Busca ET0 FAO diária (Penman-Monteith FAO-56) na Open-Meteo.
    Cacheado pelo Streamlit para evitar lentidão.
    """

    # Normaliza datas para YYYY-MM-DD (sem 'T00:00:00')
    s_date = _to_date(start_date)
    e_date = _to_date(end_date)

    print(f"[et0] Baixando ET0 FAO diária {s_date} -> {e_date} (lat={lat}, lon={lon})")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": s_date.isoformat(),  # ex: 2025-11-24
        "end_date": e_date.isoformat(),    # ex: 2025-11-30
        "daily": "et0_fao_evapotranspiration",
        "timezone": timezone,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        # print(f"[et0] URL chamada: {resp.url}")
        if resp.status_code != 200:
            print(
                f"[et0] AVISO: resposta HTTP {resp.status_code} da Open-Meteo "
                f"para ET0. Corpo (início): {resp.text[:300]}... Usando ET0 = 0.0 mm."
            )
            idx = pd.date_range(s_date, e_date, freq="D")
            return pd.DataFrame(
                {
                    "ds": idx,
                    "om_et0_fao_mm": [0.0] * len(idx),
                }
            )

        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[et0] AVISO: erro ao chamar Open-Meteo para ET0 ({exc}). "
            "Usando ET0 = 0.0 mm."
        )
        idx = pd.date_range(s_date, e_date, freq="D")
        return pd.DataFrame(
            {
                "ds": idx,
                "om_et0_fao_mm": [0.0] * len(idx),
            }
        )

    # Verifica estrutura esperada
    if "daily" not in data or "time" not in data["daily"]:
        print(
            "[et0] AVISO: resposta da Open-Meteo sem 'daily.time'. "
            "Usando ET0 = 0.0 mm."
        )
        idx = pd.date_range(s_date, e_date, freq="D")
        return pd.DataFrame(
            {
                "ds": idx,
                "om_et0_fao_mm": [0.0] * len(idx),
            }
        )

    daily = data["daily"]
    # print(f"[et0] daily.keys(): {list(daily.keys())}")

    times = pd.to_datetime(daily["time"])

    if "et0_fao_evapotranspiration" not in daily:
        print(
            "[et0] AVISO: campo 'et0_fao_evapotranspiration' ausente em daily. "
            "Preenchendo ET0 = 0.0 mm para todas as datas."
        )
        et0_vals = [0.0] * len(times)
    else:
        et0_vals = daily["et0_fao_evapotranspiration"]
        # print(f"[et0] Exemplos de ET0 retornado: {et0_vals[:5]}")

    df = pd.DataFrame(
        {
            "ds": times,
            "om_et0_fao_mm": et0_vals,
        }
    )
    df["om_et0_fao_mm"] = df["om_et0_fao_mm"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    return df




