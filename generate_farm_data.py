import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta

from agroclima_ia.config import (
    DAILY_RAIN_CSV,
    FARM_OBS_CSV,
    DEFAULT_LAT,
    DEFAULT_LON,
)
from agroclima_ia.data_fetch import load_or_create_daily_series

# Quantos anos de histórico queremos simular
HIST_YEARS = 10


def gerar_serie_diaria_sintetica(years: int) -> pd.DataFrame:
    """
    Gera uma série diária sintética de chuva ('ds','y') para 'years' anos
    atrás até ontem, com uma sazonalidade simples + ruído.
    Serve como fallback quando a API da Open-Meteo falha.
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=years * 365)

    datas = pd.date_range(start=start, end=end, freq="D")
    n = len(datas)

    rng = np.random.default_rng(seed=42)

    # Sazonalidade anual simples (mais chuva em parte do ano)
    # você pode ajustar a "forma" depois, se quiser
    day_of_year = np.array([d.timetuple().tm_yday for d in datas], dtype=float)
    season = 2.5 + 2.0 * np.sin(2 * np.pi * (day_of_year - 30) / 365.0)

    # ruído aleatório
    noise = rng.gamma(shape=1.5, scale=1.2, size=n)

    y = season + noise
    y = np.clip(y, 0, None)  # não negativo
    # zera alguns dias aleatórios para parecer período seco
    dry_mask = rng.random(n) < 0.4  # 40% dos dias com 0 chuva
    y[dry_mask] = 0.0

    df = pd.DataFrame({"ds": datas, "y": y})
    return df


# 1) Carrega ou cria a série diária externa (Open-Meteo)
try:
    df_ext = load_or_create_daily_series(
        lat=DEFAULT_LAT,
        lon=DEFAULT_LON,
        years=HIST_YEARS,
        csv_path=DAILY_RAIN_CSV,
    )
    print(f"Série diária EXTERNA carregada de/gerada em {DAILY_RAIN_CSV}")
except (requests.exceptions.RequestException, ConnectionError) as e:
    print(f"[generate_farm_data] Erro ao acessar Open-Meteo: {e}")
    print("[generate_farm_data] Gerando série diária sintética para continuar o teste...")
    df_ext = gerar_serie_diaria_sintetica(HIST_YEARS)
    df_ext.to_csv(DAILY_RAIN_CSV, index=False)
    print(f"Série diária sintética salva em {DAILY_RAIN_CSV}")

# Ordena por data e garante índice limpo
df_ext = df_ext.sort_values("ds").reset_index(drop=True)

# 2) Monta DataFrame da fazenda
df_farm = pd.DataFrame()
df_farm["data"] = df_ext["ds"]

base_rain = df_ext["y"].values
n = len(df_farm)

rng = np.random.default_rng(seed=42)

# ================
# CHUVA (chuva_mm)
# ================
noise_factor = rng.normal(loc=1.0, scale=0.2, size=n)
noise_factor = np.clip(noise_factor, 0.5, 1.5)

chuva_fake = base_rain * noise_factor

dry_days = base_rain == 0
add_rain_mask = dry_days & (rng.random(n) < 0.05)
chuva_fake[add_rain_mask] = rng.uniform(0.5, 5.0, size=add_rain_mask.sum())

chuva_fake = np.clip(chuva_fake, 0, None)
df_farm["chuva_mm"] = chuva_fake.round(1)

# =========================
# VARIÁVEIS CLIMÁTICAS
# =========================
doy = df_farm["data"].dt.dayofyear.values.astype(float)

tmean = 24 + 5 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 1.5, size=n)
tdelta = 8 + rng.normal(0, 1.0, size=n)
tmin = tmean - tdelta / 2
tmax = tmean + tdelta / 2

df_farm["tmin"] = tmin.round(1)
df_farm["tmax"] = tmax.round(1)

ur = 70 + 15 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 5, size=n)
ur = np.clip(ur, 30, 100)
df_farm["ur"] = ur.round(0)

vento = rng.normal(loc=2.0, scale=0.7, size=n)
vento = np.clip(vento, 0.2, 8.0)
df_farm["vento"] = vento.round(2)

radiacao = 18 + 6 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 2, size=n)
radiacao = np.clip(radiacao, 5, 30)
df_farm["radiacao"] = radiacao.round(2)

et0 = 3 + 0.07 * (radiacao - 15) + 0.05 * (tmean - 25)
et0 += rng.normal(0, 0.3, size=n)
et0 = np.clip(et0, 0, 8)
df_farm["et0"] = et0.round(2)

# =========================
# VARIÁVEIS DE SOLO
# =========================
soil = np.zeros(n)
soil[0] = 0.30

for i in range(1, n):
    soil[i] = soil[i - 1] - 0.01 * et0[i] + 0.002 * chuva_fake[i]
    soil[i] = np.clip(soil[i], 0.10, 0.45)

df_farm["umidade_solo"] = soil.round(3)

temp_solo = tmean - 1.5 + rng.normal(0, 0.7, size=n)
df_farm["temperatura_solo"] = temp_solo.round(1)

# =========================
# SALVANDO
# =========================
df_farm.to_csv(FARM_OBS_CSV, index=False)
print(f"Arquivo de observações da fazenda salvo em: {FARM_OBS_CSV}")
print(df_farm.head())
