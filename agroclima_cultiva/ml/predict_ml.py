# agroclima_cultiva/ml/predict_ml.py
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import load

from ..planner.catalog import CropSpec, load_catalog_from_json
from ..schemas.inputs import FarmInput
from ..climate.openmeteo import fetch_daily_forecast
from ..climate import metrics as metrics_mod


# -----------------------------------------------------------------------------
# Paths padrão (robustos: relativos ao arquivo, não ao diretório atual)
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = _THIS_DIR / "models"
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "planner" / "data" / "catalog_v1.json"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _get_compute_features_fn() -> Any:
    """
    Resolve a função de features no metrics.py, sem quebrar por ImportError.

    Esperado:
      - compute_features_7_14(df_daily_forecast) -> dict com chaves "7d" e "14d"
    """
    fn = getattr(metrics_mod, "compute_features_7_14", None)
    if callable(fn):
        return fn

    # fallback se você tiver salvado com nome diferente por engano
    fn2 = getattr(metrics_mod, "compute_features_7_14", None)
    if callable(fn2):
        return fn2

    # não achou: mostra diagnóstico
    public = sorted([n for n in dir(metrics_mod) if not n.startswith("_")])
    raise ImportError(
        "Não encontrei compute_features_7_14 em agroclima_cultiva.climate.metrics.\n"
        "Verifique se o arquivo agroclima_cultiva/climate/metrics.py realmente contém:\n"
        "  def compute_features_7_14(df_daily_forecast): ...\n\n"
        f"Nomes públicos encontrados no módulo metrics.py: {public}"
    )


def _catalog_to_df(crops: List[CropSpec], area_m2: float) -> pd.DataFrame:
    """
    Converte catálogo em DataFrame com one-hot de tags.
    Também filtra por área mínima.
    """
    rows: List[Dict[str, Any]] = []
    all_tags: set[str] = set()

    for c in crops:
        # filtro por área mínima
        try:
            if float(area_m2) < float(c.area_min_m2):
                continue
        except Exception:
            pass

        tags = [str(t).strip().lower() for t in (c.tags or ())]
        for t in tags:
            if t:
                all_tags.add(t)

        rows.append(
            {
                "crop_id": c.crop_id,
                "nome": c.nome,
                "grupo": str(c.grupo),
                "demanda_hidrica": str(c.demanda_hidrica),
                "complexidade": str(c.complexidade),
                "investimento": str(c.investimento),
                "risco": str(c.risco),
                "tags": tags,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # one-hot de tags
    tag_cols = sorted(all_tags)
    for t in tag_cols:
        df[f"tag__{t}"] = df["tags"].apply(lambda xs: 1 if t in (xs or []) else 0)

    return df.drop(columns=["tags"]).reset_index(drop=True)


def _hfeat_to_dict(hf: Any) -> Dict[str, float]:
    """
    Normaliza HorizonFeatures (dataclass) ou dict para o schema usado no ML.
    """
    if hf is None:
        return {}

    if is_dataclass(hf):
        d = asdict(hf)
        return {
            "chuva_total": float(d.get("chuva_total_mm", 0.0) or 0.0),
            "et0_total": float(d.get("et0_total_mm", 0.0) or 0.0),
            "balanco_hidrico": float(d.get("balanco_hidrico_mm", 0.0) or 0.0),
            "dias_secos": float(d.get("dias_secos", 0.0) or 0.0),
            "dias_chuva_forte": float(d.get("dias_chuva_forte", 0.0) or 0.0),
            "tmax_media": float(d.get("tmax_media_c", 0.0) or 0.0),
            "tmax_p95": float(d.get("tmax_p95_c", 0.0) or 0.0),
            "rh_media": float(d.get("rh_media_pct", 0.0) or 0.0),
            "vento_medio": float(d.get("vento_medio_kmh", 0.0) or 0.0),
        }

    dd = dict(hf)
    return {
        "chuva_total": float(dd.get("chuva_total", 0.0) or 0.0),
        "et0_total": float(dd.get("et0_total", 0.0) or 0.0),
        "balanco_hidrico": float(dd.get("balanco_hidrico", 0.0) or 0.0),
        "dias_secos": float(dd.get("dias_secos", 0.0) or 0.0),
        "dias_chuva_forte": float(dd.get("dias_chuva_forte", 0.0) or 0.0),
        "tmax_media": float(dd.get("tmax_media", 0.0) or 0.0),
        "tmax_p95": float(dd.get("tmax_p95", 0.0) or 0.0),
        "rh_media": float(dd.get("rh_media", 0.0) or 0.0),
        "vento_medio": float(dd.get("vento_medio", 0.0) or 0.0),
    }


def load_model_bundle(model_dir: Path, window_len: int) -> Dict[str, Any]:
    p = model_dir / f"cultiva_score_model_{window_len}d.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")

    bundle = load(p)
    for k in ("pipeline", "num_cols", "cat_cols"):
        if k not in bundle:
            raise ValueError(f"Bundle do modelo inválido ({p.name}): faltando chave '{k}'")
    return bundle


def predict_scores_for_horizon(
    inp: FarmInput,
    feats: Dict[str, float],
    window_len: int,
    catalog_df: pd.DataFrame,
    model_bundle: Dict[str, Any],
) -> pd.DataFrame:
    """
    Prediz score ML por cultura para um horizonte (7d ou 14d).
    """
    if catalog_df is None or catalog_df.empty:
        return pd.DataFrame(columns=["crop_id", "nome", "grupo", "score_ml", "window_len"])

    pipe = model_bundle["pipeline"]
    num_cols: List[str] = list(model_bundle["num_cols"])
    cat_cols: List[str] = list(model_bundle["cat_cols"])

    base: Dict[str, Any] = {
        "lat": float(inp.lat),
        "lon": float(inp.lon),
        "area_m2": float(inp.area_m2),
        "objetivo": str(inp.objetivo),
    }
    base.update(feats)

    d = catalog_df.copy()
    for k, v in base.items():
        d[k] = v

    for c in num_cols:
        if c not in d.columns:
            d[c] = 0.0
    for c in cat_cols:
        if c not in d.columns:
            d[c] = "desconhecido"

    X = d[num_cols + cat_cols].copy()
    yhat = pipe.predict(X)

    out = d[["crop_id", "nome", "grupo", "demanda_hidrica", "complexidade", "investimento", "risco"]].copy()
    out["window_len"] = int(window_len)
    out["score_ml"] = np.clip(np.asarray(yhat, dtype=float), 0.0, 1.0)

    return out.sort_values("score_ml", ascending=False).reset_index(drop=True)


def predict_topk(
    inp: FarmInput,
    features_7_14: Dict[str, Any],
    top_k: int = 5,
    model_dir: Optional[str] = None,
    catalog_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retorna TOP-K culturas com score combinado (média 7d/14d).
    """
    cat_path = Path(catalog_path).resolve() if catalog_path else DEFAULT_CATALOG_PATH
    crops = load_catalog_from_json(path=cat_path)
    cat_df = _catalog_to_df(crops, area_m2=float(inp.area_m2))

    if cat_df.empty:
        return pd.DataFrame(columns=["crop_id", "nome", "grupo", "score_ml_mean", "score_ml_7d", "score_ml_14d"])

    mdir = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR
    bundle7 = load_model_bundle(mdir, 7)
    bundle14 = load_model_bundle(mdir, 14)

    feats7 = _hfeat_to_dict(features_7_14.get("7d"))
    feats14 = _hfeat_to_dict(features_7_14.get("14d"))

    pred7 = predict_scores_for_horizon(inp, feats7, 7, cat_df, bundle7)
    pred14 = predict_scores_for_horizon(inp, feats14, 14, cat_df, bundle14)

    p7 = pred7[["crop_id", "nome", "grupo", "score_ml"]].rename(columns={"score_ml": "score_ml_7d"})
    p14 = pred14[["crop_id", "score_ml"]].rename(columns={"score_ml": "score_ml_14d"})

    out = p7.merge(p14, on="crop_id", how="inner")
    out["score_ml_mean"] = (out["score_ml_7d"] + out["score_ml_14d"]) / 2.0
    out = out.sort_values("score_ml_mean", ascending=False).reset_index(drop=True)

    return out.head(int(top_k)).copy()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Prediz score ML (7d/14d) e imprime top culturas.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--area_m2", type=float, default=5000.0)
    p.add_argument("--objetivo", type=str, default="baixo_risco")
    p.add_argument("--municipio", type=str, default=None)
    p.add_argument("--uf", type=str, default=None)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--model_dir", type=str, default=None)   # se None, usa DEFAULT_MODEL_DIR
    p.add_argument("--catalog", type=str, default=None)     # se None, usa DEFAULT_CATALOG_PATH
    p.add_argument("--timezone", type=str, default="America/Sao_Paulo")
    args = p.parse_args()

    inp = FarmInput(
        lat=float(args.lat),
        lon=float(args.lon),
        area_m2=float(args.area_m2),
        objetivo=str(args.objetivo),  # type: ignore[arg-type]
        municipio=args.municipio,
        uf=args.uf,
        perfil_produtor="agricultura_familiar",
        restricoes={},
        contexto={},
    )
    inp.validate()

    # forecast -> features (resolve função de forma robusta)
    compute_fn = _get_compute_features_fn()
    df_fc = fetch_daily_forecast(lat=inp.lat, lon=inp.lon, days=16, timezone=str(args.timezone))
    feats = compute_fn(df_fc)

    top = predict_topk(
        inp=inp,
        features_7_14=feats,
        top_k=int(args.top_k),
        model_dir=args.model_dir,
        catalog_path=args.catalog,
    )

    place = (inp.municipio or "").strip()
    uf = (inp.uf or "").strip()
    place_str = (f"{place}/{uf}" if place and uf else (place or uf or "sem_municipio_uf"))

    print(f"Top {len(top)} (ML) — {place_str} ({inp.lat:.4f}, {inp.lon:.4f}) | area={inp.area_m2:.0f} m² | objetivo={inp.objetivo}")

    if top.empty:
        print("Sem resultados (verifique catálogo/area_min_m2 e caminhos dos modelos).")
        return

    for i, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{i}. {row.nome} | grupo={row.grupo} | "
            f"score={row.score_ml_mean:.3f} | 7d={row.score_ml_7d:.3f} | 14d={row.score_ml_14d:.3f}"
        )


if __name__ == "__main__":
    main()


