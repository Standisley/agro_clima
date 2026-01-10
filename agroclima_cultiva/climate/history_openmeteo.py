# agroclima_cultiva/climate/history_openmeteo.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests


def fetch_daily_history(
    lat: float,
    lon: float,
    days_back: int = 365,
    timezone: str = "America/Sao_Paulo",
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Histórico diário via Open-Meteo (Archive API), retornando colunas padronizadas:

    - ds (datetime)
    - om_prcp_mm
    - om_tmin_c
    - om_tmax_c
    - om_wind_kmh
    - om_rh_max
    - om_et0_fao_mm

    Observação:
    - days_back recomendado: 365 (1 ano) para MVP do dataset.
    """
    if days_back <= 0:
        raise ValueError("days_back deve ser > 0")

    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=int(days_back))

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

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
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
