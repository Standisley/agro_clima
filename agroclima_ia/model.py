# agroclima_ia/model.py
"""
Camada de modelo (LightGBM) do AgroClima IA.

Aqui concentramos:
- split temporal treino/teste
- treino do LightGBM
- avaliação simples (MAE / RMSE)
- salvar / carregar modelo

Compatibilidade:
- Expomos dois nomes de função de treino, para suportar versões antigas
  de outros módulos:
    - train_lightgbm_regressor(...)
    - train_lightgbm(...)
  Ambos fazem a mesma coisa.

- train_lightgbm_regressor aceita tanto (X_val, y_val) quanto
  (X_valid, y_valid) como nomes de parâmetros, para não quebrar chamadas antigas.

- IMPORTANTE: não usamos mais:
    * early_stopping_rounds
    * verbose_eval
  na chamada de lightgbm.train, porque a sua versão não aceita esses argumentos.
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

    Parâmetros
    ----------
    df_features : DataFrame
        Saída de create_rain_features(), contendo coluna 'ds', target
        (y por padrão) e demais features numéricas.
    target_col : str
        Nome da coluna alvo.
    test_size_days : int
        Quantidade de dias (linhas finais) reservados para teste.
    all_feature_cols : list[str] | None
        Lista explícita de colunas de features. Se None, usa todas as
        colunas numéricas exceto o alvo.

    Retorno
    -------
    X_train, X_test, y_train, y_test, feature_cols
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
# 2) Treino LightGBM
# =============================================================================


def _default_lgb_params() -> dict:
    """Hiperparâmetros padrão, conservadores, para o modelo."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbose": -1,
    }


def train_lightgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    # nomes antigos ainda aceitos:
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,  # mantido na assinatura, mas NÃO usado
    **kwargs,
) -> lgb.Booster:
    """
    Treina um modelo LightGBM (regressão) para chuva diária.

    Compatível com chamadas antigas e novas:
    - Pode receber (X_val, y_val)
    - Ou (X_valid, y_valid)

    OBS: Não usamos early_stopping_rounds nem verbose_eval na chamada
    a lightgbm.train por compatibilidade com a sua versão da biblioteca.
    """
    params = {**_default_lgb_params(), **(params or {})}

    # Normalizar nomes (damos preferência a X_val/y_val; se estiverem None,
    # usamos X_valid/y_valid, que é como algumas versões antigas chamam).
    if X_val is None and X_valid is not None:
        X_val = X_valid
    if y_val is None and y_valid is not None:
        y_val = y_valid

    dtrain = lgb.Dataset(X_train, label=y_train)

    if X_val is not None and y_val is not None and len(X_val) > 0:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets = [dtrain, dval]
        # Algumas versões antigas não aceitam valid_names,
        # então vamos evitar também para máxima compatibilidade.
        # valid_names = ["train", "valid"]
    else:
        dval = None
        valid_sets = [dtrain]
        # valid_names = ["train"]

    # ⚠️ Chamamos lightgbm.train de forma mínima, sem early_stopping_rounds,
    # sem verbose_eval, sem valid_names, para ser compatível com versões antigas.
    booster = lgb.train(
        params,
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
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,  # mantido só para compatibilidade
    **kwargs,
) -> lgb.Booster:
    """
    Alias para train_lightgbm_regressor, mantido por compatibilidade.
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
    Calcula métricas MAE e RMSE para o modelo treinado.
    """
    if X_test is None or len(X_test) == 0:
        return float("nan"), float("nan")

    y_pred = model.predict(X_test)
    y_true = y_test.values

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    print(f"[metrics] MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    return mae, rmse


# =============================================================================
# 4) Persistência
# =============================================================================


def save_model(model: lgb.Booster, path: Path | str) -> None:
    """
    Salva o modelo LightGBM em formato .txt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"[model] Modelo LightGBM salvo em {path}")


def load_model(path: Path | str) -> lgb.Booster:
    """
    Carrega o modelo LightGBM a partir de um arquivo .txt.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")
    print(f"[model] Carregando modelo LightGBM de {path}")
    return lgb.Booster(model_file=str(path))




