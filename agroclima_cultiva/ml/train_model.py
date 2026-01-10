# agroclima_cultiva/ml/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _DEFAULT_MODEL = "hgb"
except Exception:
    _DEFAULT_MODEL = "rf"

from sklearn.ensemble import RandomForestRegressor

from ..planner.catalog import load_catalog_from_json, CropSpec


def _catalog_to_df(crops: List[CropSpec]) -> pd.DataFrame:
    rows = []
    all_tags: set[str] = set()

    for c in crops:
        tags = [str(t).strip().lower() for t in (c.tags or ())]
        for t in tags:
            if t:
                all_tags.add(t)

        rows.append(
            {
                "crop_id": c.crop_id,
                "grupo": str(c.grupo),
                "demanda_hidrica": str(c.demanda_hidrica),
                "complexidade": str(c.complexidade),
                "investimento": str(c.investimento),
                "risco": str(c.risco),
                "tags": tags,
            }
        )

    df = pd.DataFrame(rows)

    tag_cols = sorted(all_tags)
    for t in tag_cols:
        df[f"tag__{t}"] = df["tags"].apply(lambda xs: 1 if t in (xs or []) else 0)

    return df.drop(columns=["tags"])


def _prepare_training_frame(df: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ds_end"] = pd.to_datetime(out["ds_end"], errors="coerce")
    if out["ds_end"].isna().any():
        raise ValueError("Há ds_end inválido no dataset. Verifique o CSV gerado.")

    out["crop_id"] = out["crop_id"].astype(str).str.strip()
    out["objetivo"] = out["objetivo"].astype(str).str.strip().str.lower()

    out = out.merge(catalog_df, on="crop_id", how="left")

    miss = out["grupo"].isna().sum()
    if miss > 0:
        missing_ids = (
            out.loc[out["grupo"].isna(), "crop_id"]
            .value_counts()
            .head(10)
            .index.tolist()
        )
        raise ValueError(
            f"Existem {miss} linhas com crop_id sem match no catálogo. "
            f"Exemplos: {missing_ids}. Corrija o catalog_v1.json ou o CSV."
        )

    num_base = [
        "lat", "lon", "area_m2",
        "chuva_total", "et0_total", "balanco_hidrico",
        "dias_secos", "dias_chuva_forte",
        "tmax_media", "tmax_p95",
        "rh_media", "vento_medio",
    ]
    for c in num_base:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    tag_cols = [c for c in out.columns if c.startswith("tag__")]
    for c in tag_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["score_teacher"] = pd.to_numeric(out["score_teacher"], errors="coerce").fillna(0.0)
    out["score_teacher"] = out["score_teacher"].clip(0.0, 1.0)

    return out


def _build_pipeline(num_cols: List[str], cat_cols: List[str], model_name: str) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=None,
            learning_rate=0.08,
            max_iter=500,
        )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def _time_split(df: pd.DataFrame, test_days: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ds_end").reset_index(drop=True)
    max_day = df["ds_end"].max()
    cutoff = max_day - pd.Timedelta(days=int(test_days))
    train = df[df["ds_end"] <= cutoff].copy()
    test = df[df["ds_end"] > cutoff].copy()
    return train, test


def _train_one(
    df_all: pd.DataFrame,
    window_len: int,
    model_dir: Path,
    model_name: str,
    test_days: int = 60,
) -> None:
    df_w = df_all[df_all["window_len"].astype(int) == int(window_len)].copy()
    if df_w.empty:
        raise ValueError(f"Não há linhas para window_len={window_len} no dataset.")

    train_df, test_df = _time_split(df_w, test_days=test_days)
    if len(train_df) < 500:
        raise ValueError(f"Treino muito pequeno ({len(train_df)}). Ajuste --test_days ou gere mais histórico.")
    if len(test_df) < 50:
        raise ValueError(f"Teste muito pequeno ({len(test_df)}). Ajuste --test_days ou gere mais histórico.")

    tag_cols = sorted([c for c in df_all.columns if c.startswith("tag__")])

    num_cols = [
        "lat", "lon", "area_m2",
        "chuva_total", "et0_total", "balanco_hidrico",
        "dias_secos", "dias_chuva_forte",
        "tmax_media", "tmax_p95",
        "rh_media", "vento_medio",
        *tag_cols,
    ]
    cat_cols = ["objetivo", "grupo", "demanda_hidrica", "complexidade", "investimento", "risco", "crop_id"]

    pipe = _build_pipeline(num_cols=num_cols, cat_cols=cat_cols, model_name=model_name)

    X_train = train_df[num_cols + cat_cols].copy()
    y_train = train_df["score_teacher"].astype(float).values

    X_test = test_df[num_cols + cat_cols].copy()
    y_test = test_df["score_teacher"].astype(float).values

    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_test).astype(float), 0.0, 1.0)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)     # compatível com sklearn antigo
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, pred)

    bundle: Dict[str, object] = {
        "pipeline": pipe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "window_len": int(window_len),
        "test_days": int(test_days),
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"cultiva_score_model_{window_len}d.joblib"
    dump(bundle, out_path)

    print(f"[OK] Modelo {window_len}d salvo em: {out_path.resolve()}")
    print(f"     Métricas (teste): MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")
    print(f"     Split: train={len(train_df):,} | test={len(test_df):,} | test_days={test_days}")


def main() -> None:
    p = argparse.ArgumentParser(description="Treina modelo ML (score climático) a partir do dataset teacher.")
    p.add_argument("--data", type=str, required=True, help="CSV gerado pelo make_dataset.py")
    p.add_argument("--catalog", type=str, default=None, help="Caminho opcional do catalog_v1.json")
    p.add_argument("--model_dir", type=str, default="agroclima_cultiva/ml/models")
    p.add_argument("--test_days", type=int, default=60)
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL, choices=["hgb", "rf"])
    args = p.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8")

    if args.catalog:
        crops = load_catalog_from_json(path=Path(args.catalog))
    else:
        crops = load_catalog_from_json()

    cat_df = _catalog_to_df(crops)
    df_all = _prepare_training_frame(df, cat_df)

    model_dir = Path(args.model_dir).resolve()

    _train_one(df_all, window_len=7, model_dir=model_dir, model_name=str(args.model), test_days=int(args.test_days))
    _train_one(df_all, window_len=14, model_dir=model_dir, model_name=str(args.model), test_days=int(args.test_days))

    print("[DONE] Treino concluído.")


if __name__ == "__main__":
    main()



