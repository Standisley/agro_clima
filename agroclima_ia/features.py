# agroclima_ia/features.py

import pandas as pd
import numpy as np
from typing import List, Tuple


def create_rain_features(
    df_daily: pd.DataFrame,
    target_col: str = "y",
    mode: str = "train",  # "train" ou "forecast"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cria features robustas para previsão de chuva (Lags, Médias Móveis, Sazonalidade).
    Otimizado para distribuição Tweedie (captura de picos de chuva).

    Espera:
      - Coluna 'ds' (data).
      - Coluna alvo 'y' (chuva mm).
      - Variáveis exógenas opcionais (tmin, tmax, ur, vento, radiacao).

    Retorna:
      - df_features: DataFrame pronto para o modelo.
      - feature_cols: Lista de colunas que devem entrar no treino.
    """

    # ------------------------------------------------------------------
    # 0) Casos degenerados
    # ------------------------------------------------------------------
    if df_daily is None or df_daily.empty:
        return pd.DataFrame(), []

    mode = (mode or "train").strip().lower()
    if mode not in ("train", "forecast"):
        raise ValueError("mode deve ser 'train' ou 'forecast'.")

    # ------------------------------------------------------------------
    # 1) Cópia + garantia da coluna de data 'ds'
    # ------------------------------------------------------------------
    df = df_daily.copy()

    if "ds" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["ds"] = df.index
            df = df.reset_index(drop=True)
        else:
            raise ValueError("df_daily precisa ter coluna 'ds' ou índice DatetimeIndex.")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2) Garantir coluna alvo e Limpeza Inicial
    # ------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(f"DataFrame não possui coluna alvo '{target_col}'.")

    # Coerção numérica
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Em forecast, se o alvo futuro for NaN, preenchemos com 0.0 para
    # permitir o cálculo das features (lags) sem quebrar o código.
    if mode == "forecast":
        df[target_col] = df[target_col].fillna(0.0)

    # ------------------------------------------------------------------
    # 3) Padronização de Variáveis Exógenas (Mapeamento Robusto)
    # ------------------------------------------------------------------
    # Tenta encontrar e renomear colunas para um padrão interno (tmean, rh, wind, rad)
    # Isso facilita a criação de features num loop único.
    
    # --- Temperatura Média (Importante para Convecção) ---
    tmean = None
    if "tmean" in df.columns:
        tmean = pd.to_numeric(df["tmean"], errors="coerce")
    else:
        # Tenta calcular pela média de min/max
        tmin_candidates = ["tmin", "om_temp2m_min", "temp2m_min", "temperature_2m_min"]
        tmax_candidates = ["tmax", "om_temp2m_max", "temp2m_max", "temperature_2m_max"]
        
        tmin_col = next((c for c in tmin_candidates if c in df.columns), None)
        tmax_col = next((c for c in tmax_candidates if c in df.columns), None)
        
        if tmin_col and tmax_col:
            tmean = (
                pd.to_numeric(df[tmin_col], errors="coerce") + 
                pd.to_numeric(df[tmax_col], errors="coerce")
            ) / 2.0

    # --- Umidade Relativa ---
    rh = None
    rh_candidates = ["ur", "om_rh2m_max", "rh", "relative_humidity", "relative_humidity_2m_max"]
    for c in rh_candidates:
        if c in df.columns:
            rh = pd.to_numeric(df[c], errors="coerce")
            break

    # --- Vento ---
    wind = None
    wind_candidates = ["vento", "om_windspeed10_max", "wind", "windspeed", "windspeed_10m_max"]
    for c in wind_candidates:
        if c in df.columns:
            wind = pd.to_numeric(df[c], errors="coerce")
            break

    # --- Radiação Solar ---
    rad = None
    rad_candidates = ["radiacao", "om_sw_rad_sum", "rad", "shortwave_radiation_sum"]
    for c in rad_candidates:
        if c in df.columns:
            rad = pd.to_numeric(df[c], errors="coerce")
            break

    # Atribui nomes padrão ao DataFrame se encontrou os dados
    if tmean is not None: df["tmean"] = tmean
    if rh is not None:    df["rh"] = rh
    if wind is not None:  df["wind"] = wind
    if rad is not None:   df["rad"] = rad

    # Lista das colunas que realmente temos disponíveis para gerar lags
    available_exog = [c for c in ["tmean", "rh", "wind", "rad"] if c in df.columns]

    # ------------------------------------------------------------------
    # 4) Features de Chuva (Lags e Janelas Móveis)
    # ------------------------------------------------------------------
    df_feat = df.copy()

    # Cria série auxiliar limpa para shifts. 
    # MUDANÇA: Usar fillna(0.0) em vez de ffill() para chuva é mais seguro 
    # (evita propagar tempestades inexistentes).
    y_clean = df_feat[target_col].fillna(0.0)

    # Lags (O que aconteceu ontem, anteontem, semana passada)
    df_feat["lag_1"] = y_clean.shift(1)
    df_feat["lag_2"] = y_clean.shift(2)
    df_feat["lag_3"] = y_clean.shift(3)
    df_feat["lag_7"] = y_clean.shift(7) # Sazonalidade semanal

    # Janelas Móveis (Tendência)
    # Curto prazo (1 semana)
    df_feat["rolling_mean_7"] = y_clean.rolling(window=7, min_periods=3).mean()
    df_feat["rolling_sum_7"]  = y_clean.rolling(window=7, min_periods=3).sum()
    
    # --- NOVO: Médio prazo (30 dias) ---
    # Ajuda o modelo a saber se o mês é chuvoso ou seco no geral.
    # Isso corrige a tendência de subestimar chuva em meses historicamente úmidos.
    df_feat["rolling_mean_30"] = y_clean.rolling(window=30, min_periods=10).mean()

    # ------------------------------------------------------------------
    # 5) Features Exógenas (Lags e Médias Móveis)
    # ------------------------------------------------------------------
    for col in available_exog:
        # Garante numérico e preenche falhas com ffill (para temperatura/vento ok)
        # e depois 0 se sobrar NaN no começo
        col_series = pd.to_numeric(df[col], errors="coerce").fillna(method="ffill").fillna(0.0)
        
        df_feat[f"{col}_lag_1"] = col_series.shift(1)
        df_feat[f"{col}_roll_7"] = col_series.rolling(window=7, min_periods=3).mean()

    # ------------------------------------------------------------------
    # 6) Sazonalidade Temporal
    # ------------------------------------------------------------------
    ds = df_feat["ds"]
    df_feat["month"] = ds.dt.month.astype(int)
    df_feat["day_of_year"] = ds.dt.dayofyear.astype(int)
    df_feat["day_of_week"] = ds.dt.weekday.astype(int)

    # ------------------------------------------------------------------
    # 7) Definição Final das Colunas de Feature
    # ------------------------------------------------------------------
    feature_candidates = [
        "lag_1", "lag_2", "lag_3", "lag_7",
        "rolling_mean_7", "rolling_sum_7", "rolling_mean_30",
        "month", "day_of_year", "day_of_week"
    ]
    
    # Adiciona as exógenas dinamicamente se existirem
    for col in available_exog:
        feature_candidates.append(f"{col}_lag_1")
        feature_candidates.append(f"{col}_roll_7")

    # Filtra apenas o que realmente foi criado no DataFrame
    final_feature_cols = [c for c in feature_candidates if c in df_feat.columns]

    # ------------------------------------------------------------------
    # 8) Limpeza Final (Crítico para Modelo Tweedie)
    # ------------------------------------------------------------------
    
    # 8.1) Preenche NaNs nas FEATURES com 0.0
    # Modelos como LightGBM lidam com NaN, mas preencher evita erros obscuros
    # e garante consistência nos cálculos de lag.
    df_feat[final_feature_cols] = df_feat[final_feature_cols].fillna(0.0)

    # 8.2) Modo TREINO: Remove linhas onde não temos o ALVO (y)
    # Não podemos treinar prevendo NaN.
    if mode == "train":
        df_feat = df_feat.dropna(subset=[target_col]).reset_index(drop=True)
        
        # Opcional: Remover os primeiros 30 dias para estabilizar o rolling_30
        # df_feat = df_feat.iloc[30:].reset_index(drop=True)
    
    # Modo FORECAST: Não removemos linhas! O alvo pode ser NaN (futuro),
    # mas as features (lags do passado) estarão lá preenchidas.

    return df_feat, final_feature_cols




