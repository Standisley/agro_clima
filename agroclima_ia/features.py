# agroclima_ia/features.py

import pandas as pd
from typing import List, Tuple


def create_rain_features(
    df_daily: pd.DataFrame,
    target_col: str = "y",
    mode: str = "train",  # "train" ou "forecast"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cria features para previsão de chuva a partir de uma série diária.

    Espera:
      - Uma coluna de data chamada 'ds' OU um índice DatetimeIndex.
      - Uma coluna alvo (por padrão 'y'), normalmente a chuva diária (mm).
      - Opcionalmente variáveis exógenas (temperatura, UR, vento, radiação).

    Retorna:
      - df_features: DataFrame com coluna 'ds', coluna alvo e features numéricas.
      - feature_cols: lista de nomes das colunas de features usadas pelo modelo.

    Observação importante (para forecast):
      - Este módulo NÃO deve "matar" datas futuras por falta de histórico.
      - Em forecast, é melhor manter as datas futuras e preencher NaNs das
        features com 0.0 do que retornar vazio e quebrar o pipeline.
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
            raise ValueError(
                "df_daily precisa ter coluna 'ds' ou índice DatetimeIndex."
            )

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2) Garantir coluna alvo
    # ------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(f"DataFrame não possui coluna alvo '{target_col}'.")

    # Coerção numérica do alvo (evita strings etc.)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # IMPORTANTE:
    # - Em treino, y precisa ser válido.
    # - Em forecast, pode existir y NaN nas datas futuras; não queremos derrubar essas linhas.
    if mode == "forecast":
        # mantém datas futuras; y NaN vira 0.0 apenas para não matar o horizonte
        df[target_col] = df[target_col].fillna(0.0)

    # ------------------------------------------------------------------
    # 3) Variáveis exógenas (tentamos mapear para nomes padrão)
    # ------------------------------------------------------------------
    tmean = None
    if "tmean" in df.columns:
        tmean = pd.to_numeric(df["tmean"], errors="coerce")
    else:
        tmin_candidates = ["tmin", "om_temp2m_min", "temp2m_min"]
        tmax_candidates = ["tmax", "om_temp2m_max", "temp2m_max"]
        tmin_col = next((c for c in tmin_candidates if c in df.columns), None)
        tmax_col = next((c for c in tmax_candidates if c in df.columns), None)
        if tmin_col and tmax_col:
            tmean = (
                pd.to_numeric(df[tmin_col], errors="coerce")
                + pd.to_numeric(df[tmax_col], errors="coerce")
            ) / 2.0

    rh = None
    for c in ["ur", "om_rh2m_max", "rh", "relative_humidity"]:
        if c in df.columns:
            rh = pd.to_numeric(df[c], errors="coerce")
            break

    wind = None
    for c in ["vento", "om_windspeed10_max", "wind", "windspeed"]:
        if c in df.columns:
            wind = pd.to_numeric(df[c], errors="coerce")
            break

    rad = None
    for c in ["radiacao", "om_sw_rad_sum", "rad", "shortwave_radiation_sum"]:
        if c in df.columns:
            rad = pd.to_numeric(df[c], errors="coerce")
            break

    if tmean is not None:
        df["tmean"] = tmean
    if rh is not None:
        df["rh"] = rh
    if wind is not None:
        df["wind"] = wind
    if rad is not None:
        df["rad"] = rad

    EXOG_COLS: List[str] = []
    for c in ["tmean", "rh", "wind", "rad"]:
        if c in df.columns:
            EXOG_COLS.append(c)

    # ------------------------------------------------------------------
    # 4) Features de chuva (lags e janelas)
    # ------------------------------------------------------------------
    df_feat = df.copy()

    # Série auxiliar para lags/rollings (evita NaN cascata)
    y_for_lags = df_feat[target_col].copy()
    if y_for_lags.isna().any():
        y_for_lags = y_for_lags.ffill()
    if y_for_lags.isna().all():
        y_for_lags = y_for_lags.fillna(0.0)

    df_feat["lag_1"] = y_for_lags.shift(1)
    df_feat["lag_2"] = y_for_lags.shift(2)
    df_feat["lag_3"] = y_for_lags.shift(3)
    df_feat["lag_7"] = y_for_lags.shift(7)

    df_feat["rolling_mean_7"] = y_for_lags.rolling(window=7, min_periods=3).mean()
    df_feat["rolling_sum_7"] = y_for_lags.rolling(window=7, min_periods=3).sum()

    # ------------------------------------------------------------------
    # 5) Features exógenas (lags e médias móveis)
    # ------------------------------------------------------------------
    for col in EXOG_COLS:
        df_feat[f"{col}_lag_1"] = pd.to_numeric(df[col], errors="coerce").shift(1)
        df_feat[f"{col}_roll_7"] = pd.to_numeric(df[col], errors="coerce").rolling(
            window=7, min_periods=3
        ).mean()

    # ------------------------------------------------------------------
    # 6) Features sazonais
    # ------------------------------------------------------------------
    ds = pd.to_datetime(df_feat["ds"])
    df_feat["month"] = ds.dt.month.astype(int)
    df_feat["day_of_year"] = ds.dt.dayofyear.astype(int)
    df_feat["day_of_week"] = ds.dt.weekday.astype(int)

    # ------------------------------------------------------------------
    # 7) Montar lista final de features
    # ------------------------------------------------------------------
    feature_cols: List[str] = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_7",
        "rolling_sum_7",
    ]

    for col in EXOG_COLS:
        if f"{col}_lag_1" in df_feat.columns:
            feature_cols.append(f"{col}_lag_1")
        if f"{col}_roll_7" in df_feat.columns:
            feature_cols.append(f"{col}_roll_7")

    feature_cols.extend(["month", "day_of_year", "day_of_week"])

    # ------------------------------------------------------------------
    # 8) Limpeza / robustez (CRÍTICO para não matar datas futuras)
    # ------------------------------------------------------------------
    # 8.1) Não derrubar linhas por NaN nas features -> preencher com 0.0
    for c in feature_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0.0)

    # 8.2) Para treino, y precisa ser válido
    if mode == "train":
        df_feat = df_feat.dropna(subset=[target_col]).reset_index(drop=True)
    else:
        # forecast: garantimos não-NaN (caso algo reintroduza NaN)
        df_feat[target_col] = pd.to_numeric(df_feat[target_col], errors="coerce").fillna(0.0)

    final_feature_cols = [c for c in feature_cols if c in df_feat.columns]

    return df_feat, final_feature_cols




