# agroclima_ia/model.py
"""
Camada de modelo (LightGBM) do AgroClima IA.
CORRIGIDO PARA DISTRIBUIÇÃO TWEEDIE (Melhor captura de picos de chuva).

Principais alterações:
- objective: 'tweedie' (em vez de regression)
- learning_rate reduzido / num_boost_round aumentado
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


# =============================================================================
# 1) Split temporal treino / teste
# =============================================================================

def train_test_split_time(
    df_features: pd.DataFrame,
    target_col: str = "y",
    test_size_days: int = 365,
    all_feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Split temporal (treino/teste) para série diária.
    """
    if "ds" not in df_features.columns:
        raise ValueError("train_test_split_time espera coluna 'ds' em df_features.")

    df_sorted = df_features.sort_values("ds").reset_index(drop=True)

    # Se o dataset for muito curto, reduz test_size_days para caber
    n_rows = len(df_sorted)
    test_size_days = min(test_size_days, max(1, n_rows // 5))

    if all_feature_cols is None:
        # Todas as colunas numéricas exceto o alvo
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
    else:
        feature_cols = [c for c in all_feature_cols if c in df_features.columns]

    if not feature_cols:
        raise ValueError("Nenhuma coluna de feature encontrada para treinar o modelo.")

    # Divisão temporal
    split_idx = max(1, n_rows - test_size_days)

    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    X_train = train[feature_cols].astype(float)
    y_train = train[target_col].astype(float)

    X_test = test[feature_cols].astype(float)
    y_test = test[target_col].astype(float)

    return X_train, X_test, y_train, y_test, feature_cols


# =============================================================================
# 2) Treino LightGBM (OTIMIZADO PARA CHUVA / TWEEDIE)
# =============================================================================

def _default_lgb_params() -> dict:
    """
    Hiperparâmetros ajustados para capturar picos de chuva (Tweedie).
    """
    return {
        # --- A MUDANÇA CRÍTICA ESTÁ AQUI ---
        "objective": "tweedie",       # Ideal para dados com muitos zeros e cauda longa (chuva)
        "tweedie_variance_power": 1.5, # 1.5 = Composto Poisson-Gamma (Padrão ouro para precipitação)
        "metric": "rmse",
        
        "boosting_type": "gbdt",
        
        # --- AJUSTES DE PRECISÃO ---
        "learning_rate": 0.05,    # Reduzido (era 0.1) para aprender picos com mais calma
        "num_leaves": 31,         # Aumentado (era 20) para permitir maior complexidade
        "max_depth": -1,          # Sem limite rígido de profundidade
        "min_child_samples": 10,  # Reduzido (era 20) para aceitar eventos mais raros (chuva forte)
        
        # Otimização de CPU
        "force_col_wise": True,
        "n_jobs": -1,
        "seed": 42,
        "verbose": -1,
    }


def train_lightgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    X_valid: pd.DataFrame | None = None, # alias antigo
    y_valid: pd.Series | None = None,    # alias antigo
    params: dict | None = None,
    num_boost_round: int = 300, # Aumentado de 150 para 300 (devido ao learning_rate menor)
    early_stopping_rounds: int = 30,
    **kwargs,
) -> lgb.Booster:
    """
    Treina um modelo LightGBM (Objective Tweedie) para chuva diária.
    """
    # Mescla params padrão otimizados com os recebidos
    final_params = {**_default_lgb_params(), **(params or {})}

    # Normalizar nomes (X_val vs X_valid)
    if X_val is None and X_valid is not None:
        X_val = X_valid
    if y_val is None and y_valid is not None:
        y_val = y_valid

    dtrain = lgb.Dataset(X_train, label=y_train)
    
    # Define valid sets se disponíveis
    valid_sets = [dtrain]
    if X_val is not None and y_val is not None and len(X_val) > 0:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)

    # Treino
    booster = lgb.train(
        final_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
    )

    return booster


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
    **kwargs,
) -> lgb.Booster:
    """
    Alias para train_lightgbm_regressor.
    """
    return train_lightgbm_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        **kwargs,
    )


# =============================================================================
# 3) Avaliação
# =============================================================================

def evaluate_model(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """
    Calcula métricas MAE e RMSE.
    """
    if X_test is None or len(X_test) == 0:
        return float("nan"), float("nan")

    y_pred = model.predict(X_test)
    y_true = y_test.values

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    
    # Cálculo do Viés (Bias) para debug no terminal
    bias = float(np.mean(y_pred - y_true))
    print(f"[metrics] MAE: {mae:.3f} | RMSE: {rmse:.3f} | Bias: {bias:.3f} (Negativo = Subestima)")
    
    return mae, rmse


# =============================================================================
# 4) Persistência
# =============================================================================

def save_model(model: lgb.Booster, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

def load_model(path: Path | str) -> lgb.Booster:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")
    print(f"[model] Carregando modelo LightGBM de {path}")
    return lgb.Booster(model_file=str(path))




