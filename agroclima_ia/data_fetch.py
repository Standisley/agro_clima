# agroclima_ia/data_fetch.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variáveis climáticas horárias que vamos buscar (histórico)
HOURLY_VARS = [
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "shortwave_radiation",
]


# ==========================================
# FUNÇÃO: Baixar 1 bloco (1 ano ou menos) – histórico horário
# ==========================================

def fetch_hourly_block(lat: float, lon: float,
                       start_date: date, end_date: date) -> pd.DataFrame | None:
    """
    Busca dados horários reais (históricos) da Open-Meteo Archive API
    para o intervalo [start_date, end_date].
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/Sao_Paulo",
    }

    try:
        r = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if "hourly" not in data or "time" not in data["hourly"]:
            print("[OpenMeteo] ⚠ Resposta incompleta. Ignorando bloco.")
            return None

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)

        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"time": "ds"})

        return df

    except Exception as e:
        print(f"[OpenMeteo] ⚠ Erro no bloco {start_date} -> {end_date}: {e}")
        return None


# ==========================================
# FUNÇÃO: Baixar N anos em blocos de 365 dias – histórico horário
# ==========================================

def fetch_n_years_hourly(lat: float, lon: float, years: int = 10) -> pd.DataFrame | None:
    """
    Baixa vários anos de dados horários reais da Open-Meteo (API de archive).

    Retorna um DataFrame horário com colunas:
      ['ds', 'precipitation', 'temperature_2m',
       'relative_humidity_2m', 'windspeed_10m', 'shortwave_radiation']
    ou None se todos os blocos falharem.
    """
    end = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=365 * years)

    blocks: list[pd.DataFrame] = []
    cur_start = start

    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=365), end)

        print(f"[data_fetch] Baixando {cur_start} -> {cur_end}")

        df_block = fetch_hourly_block(lat, lon, cur_start, cur_end)
        if df_block is not None:
            blocks.append(df_block)

        cur_start = cur_end + timedelta(days=1)

    if not blocks:
        return None

    df_full = pd.concat(blocks, ignore_index=True)
    df_full = df_full.sort_values("ds").reset_index(drop=True)
    return df_full


# ==========================================
# FUNÇÃO: Agregar horários → Diário
# ==========================================

def aggregate_hourly_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Converte dados horários em dados diários com agregações adequadas.

    Saída com colunas:
      'ds', 'y', 'tmean_ext', 'ur_ext', 'vento_ext', 'rad_ext'
    """
    df_hourly = df_hourly.copy()
    df_hourly["ds"] = pd.to_datetime(df_hourly["ds"])
    df_hourly = df_hourly.set_index("ds")

    agg_dict = {
        "precipitation": "sum",        # soma da chuva (mm/dia)
        "temperature_2m": "mean",      # média de temperatura
        "relative_humidity_2m": "mean",
        "windspeed_10m": "mean",
        "shortwave_radiation": "sum",  # energia solar diária
    }

    df_daily = df_hourly.resample("D").agg(agg_dict)
    df_daily = df_daily.reset_index()

    # Renomear para manter padrão da pipeline
    df_daily = df_daily.rename(
        columns={
            "precipitation": "y",
            "temperature_2m": "tmean_ext",
            "relative_humidity_2m": "ur_ext",
            "windspeed_10m": "vento_ext",
            "shortwave_radiation": "rad_ext",
        }
    )

    return df_daily


# ==========================================
# FALLBACK: Série sintética caso API falhe
# ==========================================

def generate_synthetic_series(csv_path: Path | str, years: int) -> pd.DataFrame:
    """
    Fallback seguro: gera série temporal falsa, mas funcional para o pipeline.

    Colunas:
      'ds', 'y', 'tmean_ext', 'ur_ext', 'vento_ext', 'rad_ext'
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years)

    dates = pd.date_range(start=start, end=end, freq="D")

    rng = np.random.default_rng(42)
    rain = np.clip(rng.gamma(shape=1.2, scale=3.0, size=len(dates)) - 2, 0, None)

    df = pd.DataFrame(
        {
            "ds": dates,
            "y": rain,
            "tmean_ext": 25 + 5 * np.sin(np.linspace(0, 6.28, len(dates))),
            "ur_ext": rng.uniform(50, 90, size=len(dates)),
            "vento_ext": rng.uniform(1, 8, size=len(dates)),
            "rad_ext": rng.uniform(5, 25, size=len(dates)),
        }
    )

    csv_path = Path(csv_path)
    df.to_csv(csv_path, index=False)
    print(f"[data_fetch] ⚠ Série diária sintética salva em {csv_path}")
    return df


# ==========================================
# FUNÇÃO PRINCIPAL: Carregar ou criar série diária histórica
# ==========================================

def load_or_create_daily_series(
    lat: float,
    lon: float,
    years: int,
    csv_path: Path | str,
) -> pd.DataFrame:
    """
    Carrega a série diária histórica (chuva + variáveis externas) de CSV,
    se já existir; caso contrário, baixa do archive da Open-Meteo,
    agrega e salva.

    Colunas de saída:
      'ds', 'y', 'tmean_ext', 'ur_ext', 'vento_ext', 'rad_ext'
    """
    csv_path = Path(csv_path)

    if csv_path.exists():
        print(f"Carregando série diária de {csv_path}")
        return pd.read_csv(csv_path, parse_dates=["ds"])

    # Baixar dados horários
    df_hourly = fetch_n_years_hourly(lat, lon, years)

    if df_hourly is None:
        print("[data_fetch] ⚠ API falhou totalmente. Gerando dados sintéticos...")
        return generate_synthetic_series(csv_path, years)

    # Agregar para diário
    df_daily = aggregate_hourly_to_daily(df_hourly)
    df_daily = df_daily.sort_values("ds").reset_index(drop=True)

    df_daily.to_csv(csv_path, index=False)
    print(f"Série diária salva em {csv_path}")
    return df_daily


# ==========================================
# PREVISÃO FUTURA DIÁRIA (PARA ENSEMBLE)
# ==========================================

def fetch_future_daily_openmeteo(
    lat: float,
    lon: float,
    days_ahead: int = 7,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    """
    Busca previsão diária da Open-Meteo para os próximos N dias.

    Variáveis diárias:
      - precipitation_sum (mm)
      - temperature_2m_max (°C)
      - precipitation_probability_max (%)
      - windspeed_10m_max (km/h)
      - relative_humidity_2m_max (%)

    Retorna DataFrame com colunas:
      ['ds',
       'om_precipitation_sum',
       'om_temp2m_max',
       'om_precip_prob_max',
       'om_windspeed10_max',
       'om_rh2m_max']
    """
    # começamos em amanhã para alinhar com "previsão para frente"
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=days_ahead - 1)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum,temperature_2m_max,"
        "precipitation_probability_max,windspeed_10m_max,relative_humidity_2m_max"
        "&timezone={tz}"
        "&start_date={start}&end_date={end}"
    ).format(
        lat=lat,
        lon=lon,
        tz=timezone.replace("/", "%2F"),
        start=start.isoformat(),
        end=end.isoformat(),
    )

    print(f"[future_forecast] Baixando previsão diária {start} -> {end}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    if not daily:
        raise RuntimeError("Open-Meteo não retornou bloco 'daily'.")

    df = pd.DataFrame(
        {
            "ds": daily["time"],
            "om_precipitation_sum": daily["precipitation_sum"],
            "om_temp2m_max": daily["temperature_2m_max"],
            "om_precip_prob_max": daily["precipitation_probability_max"],
            "om_windspeed10_max": daily["windspeed_10m_max"],
            "om_rh2m_max": daily["relative_humidity_2m_max"],
        }
    )

    df["ds"] = pd.to_datetime(df["ds"])
    return df
