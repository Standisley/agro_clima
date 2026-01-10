# agroclima_ia/forecast.py

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .et0 import fetch_et0_fao_daily
from .config import (
    DAILY_RAIN_CSV,
    FARM_OBS_CSV,
    LGB_MODEL_PATH,
    DEFAULT_LAT,
    DEFAULT_LON,
)

# =============================================================================
# Utilitários básicos
# =============================================================================

def _print(msg: str) -> None:
    print(msg)

def _get_historical_date_range(years_back: int) -> Tuple[dt.date, dt.date]:
    """Calcula o período de histórico (ex: últimos 10 anos)"""
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=365 * years_back)
    return start_date, end_date

# ... [Mantenha _download_full_history_openmeteo e load_or_download_daily_series] ...

# =============================================================================
# 2) Treinamento Unificado (AGORA TREINA CLASSIFICAÇÃO E REGRESSÃO)
# =============================================================================

def train_lightgbm_models(df_daily: pd.DataFrame) -> Tuple[Any, Any, List[str]]: # ✅ Retorna 2 modelos
    from .features import create_rain_features, get_nonzero_rain_data
    from .model import train_test_split_time, train_lightgbm, evaluate_model, save_model
    
    _print("[train] Gerando features avançadas (Chuva + Clima Exógeno)...")
    df_features, all_feature_cols = create_rain_features(df_daily, target_col="y")
    
    # ----------------------------------------------------
    # ESTÁGIO 1: CLASSIFICAÇÃO (P(Chuva > 0))
    # ----------------------------------------------------
    target_col_binary = "y_class"
    X_train_b, X_test_b, y_train_b, y_test_b, feature_cols = train_test_split_time(
        df_features, target_col=target_col_binary, all_feature_cols=all_feature_cols
    )
    
    _print(f"[train] Treinando Modelo de CLASSIFICAÇÃO Binária com {len(feature_cols)} variáveis...")
    model_b = train_lightgbm(X_train_b, y_train_b, X_test_b, y_test_b, objective="binary")
    mae_b, rmse_b = evaluate_model(model_b, X_test_b, y_test_b)
    _print(f"[metrics][CLASSIF.] MAE: {mae_b:.3f} | RMSE: {rmse_b:.3f}") # RMSE/MAE em y_class é apenas um proxy
    
    binary_model_path = Path(str(LGB_MODEL_PATH).replace(".txt", "_binary.txt"))
    save_model(model_b, binary_model_path)
    
    # ----------------------------------------------------
    # ESTÁGIO 2: REGRESSÃO CONDICIONAL (E(Chuva | Chuva > 0))
    # ----------------------------------------------------
    df_nonzero = get_nonzero_rain_data(df_features) # ✅ Filtra só dias de chuva
    
    X_train_r, X_test_r, y_train_r, y_test_r, _ = train_test_split_time(
        df_nonzero, target_col="y", all_feature_cols=all_feature_cols
    )
    
    # CRÍTICO: Se não houver dados de teste com chuva, não treinamos a regressão
    if len(y_train_r) == 0 or len(y_test_r) == 0:
         _print("[train] AVISO: Dados de chuva ZERO ou INSUFICIENTES para Regressão Condicional. Usando Modelo Binário + Média.")
         return model_b, None, feature_cols
    
    _print(f"[train] Treinando Modelo de REGRESSÃO CONDICIONAL com {len(feature_cols)} variáveis (N={len(y_train_r)} dias)...")
    
    model_r = train_lightgbm(X_train_r, y_train_r, X_test_r, y_test_r, objective="regression")
    mae_r, rmse_r = evaluate_model(model_r, X_test_r, y_test_r)
    _print(f"[metrics][REGRESSÃO] MAE: {mae_r:.3f} | RMSE: {rmse_r:.3f}")

    regression_model_path = Path(str(LGB_MODEL_PATH).replace(".txt", "_nonzero.txt"))
    save_model(model_r, regression_model_path)

    return model_b, model_r, feature_cols


def predict_tomorrow(
    df_daily: pd.DataFrame,
    model_b: Any, # ✅ Modelo de Classificação
    model_r: Any | None, # ✅ Modelo de Regressão (pode ser None)
    feature_cols: List[str],
) -> float:
    from .features import create_rain_features
    
    df_features, _ = create_rain_features(df_daily, target_col="y")
    X_predict = df_features[feature_cols].iloc[[-1]]

    # 1. Previsão de Probabilidade (Classificação)
    # Retorna P(Chuva > 0)
    p_rain = model_b.predict(X_predict)[0] 
    
    # Se a probabilidade for muito baixa, ou se o modelo de regressão não foi treinado
    if model_r is None or p_rain < 0.2: 
        return 0.0

    # 2. Previsão de Volume Condicional (Regressão)
    y_pred_cond = model_r.predict(X_predict)[0]

    # 3. Ensemble Zero-Inflated: E(Y) = P(Y > 0) * E(Y | Y > 0)
    final_pred = p_rain * y_pred_cond
    
    # Garante que a previsão não seja negativa
    return max(0.0, final_pred)


def forecast_next_days_with_openmeteo(
    df_daily: pd.DataFrame,
    model_b: Any, 
    model_r: Any | None, 
    days: int = 7,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    mm_tomorrow: Optional[float] = None, 
) -> pd.DataFrame | None:
    """
    Combina (Ensemble) previsão Zero-Inflated para D+1 com previsão Open-Meteo para D+2 a D+N.
    """
    lat = lat if lat is not None else DEFAULT_LAT
    lon = lon if lon is not None else DEFAULT_LON

    # 1. Busca previsão futura (já blindada internamente)
    df_om = fetch_future_daily_openmeteo(lat=lat, lon=lon, days_ahead=days)
    
    if df_om is None:
        return None

    # 2. Busca ET0 para o período
    horizon_start = df_om["ds"].min()
    horizon_end = df_om["ds"].max()
    df_et0 = fetch_et0_fao_daily(lat, lon, horizon_start, horizon_end, "auto")
    df_om = df_om.merge(df_et0, on="ds", how="left")
    
    # Ensemble base = chuva da Open-Meteo
    df_om["y_ensemble_mm"] = df_om["om_precipitation_sum"]

    # 3. CRÍTICO: forçar o D+1 a usar a previsão Zero-Inflated (mm_tomorrow)
    if mm_tomorrow is not None and len(df_om) > 0:
        # A primeira data é o dia D+1
        df_om.loc[df_om.index[0], "y_ensemble_mm"] = mm_tomorrow
        
    return df_om