# agroclima_ia/forecast.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .et0 import fetch_et0_fao_daily
from . import config as cfg


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


# =============================================================================
# 1) Aquisição de Dados (ETL)
# =============================================================================

def _download_openmeteo_historical(
    lat: float,
    lon: float,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    """
    Faz download do histórico diário (chuva + clima) da Open-Meteo
    para o período informado, retornando um DataFrame com colunas:
      ds, y, tmin, tmax, ur, vento, radiacao, tmean
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(
            [
                "precipitation_sum",
                "temperature_2m_min",
                "temperature_2m_max",
                "windspeed_10m_max",
                "relative_humidity_2m_max",
                "shortwave_radiation_sum",
            ]
        ),
        "timezone": "America/Sao_Paulo",
    }

    try:
        _print(f"[data_fetch] Baixando histórico completo ({start_date} -> {end_date})...")
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        if not daily:
            _print("[data_fetch] ERRO: Resposta da API sem campo 'daily'. Retornando DataFrame vazio.")
            return pd.DataFrame(
                columns=["ds", "y", "tmin", "tmax", "ur", "vento", "radiacao", "tmean"]
            )

        df = pd.DataFrame(
            {
                "ds": daily.get("time", []),
                "y": daily.get("precipitation_sum", []),
                "tmin": daily.get("temperature_2m_min", []),
                "tmax": daily.get("temperature_2m_max", []),
                "vento": daily.get("windspeed_10m_max", []),
                "ur": daily.get("relative_humidity_2m_max", []),
                "radiacao": daily.get("shortwave_radiation_sum", []),
            }
        )

        if df.empty:
            _print("[data_fetch] AVISO: DataFrame vazio após leitura da resposta 'daily'.")
            return pd.DataFrame(
                columns=["ds", "y", "tmin", "tmax", "ur", "vento", "radiacao", "tmean"]
            )

        df["ds"] = pd.to_datetime(df["ds"])
        numeric_cols = ["y", "tmin", "tmax", "ur", "vento", "radiacao"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["tmean"] = (df["tmin"] + df["tmax"]) / 2.0
        return df.sort_values("ds").reset_index(drop=True)

    except Exception as e:
        _print(f"[data_fetch] ERRO CRÍTICO no download: {e}. Retornando DataFrame vazio.")
        return pd.DataFrame(
            columns=["ds", "y", "tmin", "tmax", "ur", "vento", "radiacao", "tmean"]
        )


def _merge_farm_observations(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Tenta carregar e mesclar observações locais da fazenda, dando prioridade à chuva local."""
    farm_obs_path = Path(cfg.FARM_OBS_CSV)
    if not farm_obs_path.exists():
        return df_daily
    try:
        df_obs = pd.read_csv(farm_obs_path, parse_dates=["data"])
        df_obs = df_obs.rename(columns={"data": "ds", "chuva_mm": "y_local"})

        df_merged = df_daily.merge(df_obs[["ds", "y_local"]], on="ds", how="left")
        df_merged["y"] = df_merged["y_local"].combine_first(df_merged["y"])
        df_merged = df_merged.drop(columns=["y_local"])
        return df_merged
    except Exception as e:
        _print(f"[data_fetch] AVISO: Falha ao mesclar observações da fazenda ({e}).")
        return df_daily


def _delete_old_data_files() -> None:
    """Deleta arquivos de dados históricos e de observação da fazenda (o cache)."""
    _print("\n--- Limpando Cache de Dados ---")
    if cfg.DAILY_RAIN_CSV.exists():
        cfg.DAILY_RAIN_CSV.unlink()
        _print(f"🗑️ Arquivo de histórico diário deletado: {cfg.DAILY_RAIN_CSV.name}")
    if cfg.FARM_OBS_CSV.exists():
        cfg.FARM_OBS_CSV.unlink()
        _print(f"🗑️ Arquivo de observações da fazenda deletado: {cfg.FARM_OBS_CSV.name}")


def load_or_download_daily_series(
    lat: float,
    lon: float,
    force_refresh: bool = False,
    years_back: int = 10,
) -> pd.DataFrame:
    """
    Carrega a série diária de chuva + clima:
      1) Se existir um CSV local para esta fazenda (cfg.DAILY_RAIN_CSV) e
         'force_refresh=False', tenta usá-lo direto.
      2) Caso contrário, faz o download histórico completo na Open-Meteo.
    """
    daily_dir = cfg.DAILY_RAIN_CSV.parent
    daily_dir.mkdir(parents=True, exist_ok=True)

    if cfg.DAILY_RAIN_CSV.exists() and not force_refresh:
        try:
            df_local = pd.read_csv(cfg.DAILY_RAIN_CSV, parse_dates=["ds"])
            if not df_local.empty:
                print(f"[data_fetch] Usando série local em {cfg.DAILY_RAIN_CSV}")
                return df_local
        except Exception as e:
            print(
                f"[data_fetch] Erro ao ler CSV local '{cfg.DAILY_RAIN_CSV}': {e}. "
                "Tentando baixar histórico novamente..."
            )

    start_date, end_date = _get_historical_date_range(years_back)
    df_hist = _download_openmeteo_historical(lat, lon, start_date, end_date)

    if df_hist.empty:
        _print("[data_fetch] ERRO: Histórico baixado vazio. Não haverá dados suficientes para treinar.")
        return df_hist

    df_hist = _merge_farm_observations(df_hist)

    df_reset = df_hist.reset_index(drop=True)
    df_reset.to_csv(cfg.DAILY_RAIN_CSV, index=False)
    print(f"Série diária salva em {cfg.DAILY_RAIN_CSV}")
    return df_reset


def refresh_and_reload_daily_series(lat: float, lon: float) -> pd.DataFrame:
    _delete_old_data_files()
    return load_or_download_daily_series(lat=lat, lon=lon, force_refresh=True)


# =============================================================================
# 2) Treinamento e Previsão
# =============================================================================

def train_lightgbm_model(df_daily: pd.DataFrame) -> Tuple[Any, List[str]]:
    """
    Treina modelo LightGBM para prever 'y' (chuva diária).
    """
    from .features import create_rain_features
    from .model import (
        train_test_split_time,
        train_lightgbm_regressor,
        evaluate_model,
        save_model,
    )

    if df_daily is None or df_daily.empty:
        raise ValueError("df_daily está vazio. Não é possível treinar o modelo.")

    df_features, all_feature_cols = create_rain_features(df_daily, target_col="y")
    if df_features.empty:
        raise ValueError("Falha na criação de features: DataFrame de features veio vazio.")

    X_train, X_test, y_train, y_test, feature_cols = train_test_split_time(
        df_features,
        target_col="y",
        all_feature_cols=all_feature_cols,
        test_size_days=30,
    )

    model = train_lightgbm_regressor(
        X_train,
        y_train,
        X_valid=X_test,
        y_valid=y_test,
        feature_names=feature_cols,
    )

    evaluate_model(model, X_test, y_test)

    save_model(model, cfg.LGB_MODEL_PATH)
    print(f"[model] Modelo LightGBM salvo em {cfg.LGB_MODEL_PATH}")

    return model, feature_cols


def predict_tomorrow(
    df_daily: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
) -> float:
    from .features import create_rain_features

    if df_daily is None or df_daily.empty:
        raise ValueError("df_daily está vazio. Não é possível prever amanhã.")

    df_features, _ = create_rain_features(df_daily, target_col="y")
    if df_features.empty:
        raise ValueError("Falha na criação de features: DataFrame de features veio vazio.")

    X_last = df_features[feature_cols].tail(1)
    if X_last.empty:
        raise ValueError("Não foi possível gerar features suficientes para prever amanhã.")

    y_hat = model.predict(X_last)[0]
    return float(y_hat)


def forecast_next_days_with_openmeteo(
    df_daily: pd.DataFrame,
    model: Any,
    days: int,
    lat: float,
    lon: float,
    mm_tomorrow: Optional[float] = None,
    meta: Optional[Dict] = None,
) -> Optional[pd.DataFrame]:
    """
    Previsão dos próximos 'days' dias combinando:
      - Modelo local (quando há features futuras)
      - Open-Meteo (sempre)
      - ET0 FAO
      - Balanço hídrico (bucket model)
    """
    if df_daily is None or df_daily.empty:
        _print("[forecast] df_daily vazio. Não é possível gerar forecast.")
        return None

    if days <= 0:
        _print("[forecast] 'days' precisa ser > 0.")
        return None

    # 1) Open-Meteo forecast
    base_url = "https://api.open-meteo.com/v1/forecast"
    today = dt.date.today()
    end_date = today + dt.timedelta(days=days - 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": today.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(
            [
                "precipitation_sum",
                "temperature_2m_max",
                "windspeed_10m_max",
                "relative_humidity_2m_max",
            ]
        ),
        "timezone": "America/Sao_Paulo",
    }

    try:
        _print(f"[future_forecast] Baixando previsão diária {today} -> {end_date}")
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        _print(f"[future_forecast] ❌ ERRO DE CONEXÃO/API: {e}. Retornando None.")
        return None

    daily = data.get("daily", {})
    if not daily:
        _print("[future_forecast] ❌ ERRO: Resposta sem campo 'daily'. Retornando None.")
        return None

    df_om = pd.DataFrame(
        {
            "ds": daily.get("time", []),
            "om_prcp": daily.get("precipitation_sum", []),
            "om_temp2m_max": daily.get("temperature_2m_max", []),
            "om_windspeed10_max": daily.get("windspeed_10m_max", []),
            "om_rh2m_max": daily.get("relative_humidity_2m_max", []),
        }
    )
    if df_om.empty:
        _print("[future_forecast] ❌ ERRO: DataFrame de previsão futura vazio.")
        return None

    df_om["ds"] = pd.to_datetime(df_om["ds"])
    for col in ["om_prcp", "om_temp2m_max", "om_windspeed10_max", "om_rh2m_max"]:
        df_om[col] = pd.to_numeric(df_om[col], errors="coerce")

    # 2) Features + modelo (com fallback)
    from .features import create_rain_features

    # ------------------------------------------------------------------
    # CORREÇÃO CRÍTICA:
    # - Garantir df_concat definido e que as datas futuras não sejam eliminadas
    #   pelo create_rain_features (que faz dropna no target_col).
    # - Para isso, criamos linhas futuras com y=0.0 (não-NaN) + exógenas do OM.
    # ------------------------------------------------------------------
    df_daily2 = df_daily.copy()
    if "ds" in df_daily2.columns:
        df_daily2["ds"] = pd.to_datetime(df_daily2["ds"])

    df_future = df_om[["ds", "om_temp2m_max", "om_windspeed10_max", "om_rh2m_max"]].copy()
    df_future["y"] = 0.0  # evita dropna(subset=['y']) matar o horizonte futuro

    df_concat = pd.concat([df_daily2, df_future], ignore_index=True)
    df_concat = df_concat.sort_values("ds").reset_index(drop=True)

    df_features, _ = create_rain_features(df_concat, target_col="y")

    use_model = True
    y_hat_model = None

    if df_features.empty:
        use_model = False
        _print("[future_forecast] AVISO: df_features vazio; usando apenas precipitação da Open-Meteo.")
    else:
        future_mask = df_features["ds"] >= df_om["ds"].min()
        df_future_feats = df_features.loc[future_mask].copy()

        # IMPORTANTE:
        # Modelo treinado tem conjunto fixo de colunas; respeitar feature_name().
        try:
            model_feature_cols = list(getattr(model, "feature_name")())
        except Exception:
            model_feature_cols = None

        if model_feature_cols:
            for c in model_feature_cols:
                if c not in df_future_feats.columns:
                    df_future_feats[c] = 0.0
            X_future = df_future_feats[model_feature_cols]
        else:
            feature_cols = [c for c in df_future_feats.columns if c not in ["ds", "y"]]
            X_future = df_future_feats[feature_cols] if feature_cols else pd.DataFrame()

        if X_future is None or X_future.empty or X_future.shape[1] == 0:
            use_model = False
            _print("[future_forecast] AVISO: Nenhuma feature futura gerada; usando apenas precipitação da Open-Meteo.")
        else:
            try:
                y_hat_model = model.predict(X_future)
            except Exception as e:
                use_model = False
                _print(f"[future_forecast] AVISO: Falha ao prever com modelo ({e}); usando apenas Open-Meteo.")

    # 3) Merge final
    df_om2 = df_om.copy().sort_values("ds").reset_index(drop=True)
    df_om2["y_openmeteo_mm"] = df_om2["om_prcp"].clip(lower=0.0)

    if use_model and y_hat_model is not None:
        y_model = np.array(y_hat_model[: len(df_om2)], dtype=float)
        y_model = np.clip(y_model, 0.0, None)
        df_om2["y_model_mm"] = y_model
        df_om2["y_ensemble_mm"] = (df_om2["y_model_mm"] + df_om2["y_openmeteo_mm"]) / 2.0
    else:
        df_om2["y_model_mm"] = np.nan
        df_om2["y_ensemble_mm"] = df_om2["y_openmeteo_mm"]

    # 4) ET0 FAO
    try:
        df_et0 = fetch_et0_fao_daily(
            lat=lat,
            lon=lon,
            start_date=today,
            end_date=end_date,
        )

        if df_et0 is not None and not df_et0.empty:
            _print(f"[future_forecast] Colunas retornadas por df_et0: {list(df_et0.columns)}")

            date_col = "ds" if "ds" in df_et0.columns else ("time" if "time" in df_et0.columns else None)
            et0_col = "om_et0_fao_mm" if "om_et0_fao_mm" in df_et0.columns else (
                "et0_fao_mm" if "et0_fao_mm" in df_et0.columns else (
                    "et0_fao_evapotranspiration" if "et0_fao_evapotranspiration" in df_et0.columns else None
                )
            )

            if date_col and et0_col:
                tmp = df_et0[[date_col, et0_col]].copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col])
                tmp = tmp.rename(columns={date_col: "ds", et0_col: "om_et0_fao_mm"})
                df_om2 = df_om2.merge(tmp, on="ds", how="left")
            else:
                _print(
                    f"[future_forecast] AVISO: df_et0 sem colunas esperadas "
                    f"(datas: {date_col}, et0: {et0_col}). Definindo ET0 como NaN."
                )
                df_om2["om_et0_fao_mm"] = np.nan
        else:
            _print("[future_forecast] AVISO: df_et0 vazio. Definindo ET0 como NaN.")
            df_om2["om_et0_fao_mm"] = np.nan

    except Exception as e:
        _print(f"[future_forecast] AVISO: Erro ao buscar ET0 FAO: {e}. Definindo ET0 como NaN.")
        df_om2["om_et0_fao_mm"] = np.nan

    # 5) Balanço hídrico “real” (bucket model)
    try:
        from .water_balance import compute_water_balance_bucket

        df_om2, wb_summary = compute_water_balance_bucket(
            df_om2,
            meta=meta,
            rain_col="y_ensemble_mm",
            et0_col="om_et0_fao_mm",
        )
    except Exception as e:
        _print(f"[future_forecast] AVISO: Falha ao calcular balanço hídrico: {e}.")
        df_om2["water_balance_mm"] = np.nan
        df_om2["water_storage_mm"] = np.nan
        df_om2["deficit_mm"] = np.nan
        df_om2["excess_mm"] = np.nan
        df_om2["water_status"] = "INDISPONIVEL"

    # 6) validação mínima de colunas necessárias pro pipeline
    required_cols = [
        "ds",
        "y_ensemble_mm",
        "om_temp2m_max",
        "om_windspeed10_max",
        "om_rh2m_max",
        "om_et0_fao_mm",
    ]
    missing = [c for c in required_cols if c not in df_om2.columns]
    if missing:
        _print(f"[forecast] ❌ ERRO: Colunas necessárias ausentes: {missing}")
        return None

    return df_om2










