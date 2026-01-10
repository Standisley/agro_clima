# agroclima_cultiva/climate/openmeteo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests


@dataclass(frozen=True)
class LocationMeta:
    lat: float
    lon: float
    municipio: Optional[str] = None
    uf: Optional[str] = None


def fetch_daily_forecast(
    lat: float,
    lon: float,
    days: int = 16,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    """
    Forecast diário Open-Meteo (até 16 dias), com colunas padronizadas:

    - ds (datetime)
    - om_prcp_mm
    - om_tmin_c
    - om_tmax_c
    - om_wind_kmh
    - om_rh_max
    - om_et0_fao_mm  (ET0 FAO diária)
    """
    if days <= 0:
        raise ValueError("days deve ser > 0")
    if days > 16:
        days = 16  # Open-Meteo forecast diário típico

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
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

    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily") or {}
    if not daily:
        return pd.DataFrame(columns=[
            "ds",
            "om_prcp_mm",
            "om_tmin_c",
            "om_tmax_c",
            "om_wind_kmh",
            "om_rh_max",
            "om_et0_fao_mm",
        ])

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

