from __future__ import annotations

from typing import Any, List, Tuple, Dict
from pathlib import Path
import json
import hashlib

import lightgbm as lgb
import pandas as pd

# Imports internos
from .config import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_SERIES_ID,
    FARM_CONFIG,
    LGB_MODEL_PATH,
)
from . import config as cfg  # <- para acessar cfg.DAILY_RAIN_CSV com segurança

from .forecast import (
    load_or_download_daily_series,
    train_lightgbm_model,
    predict_tomorrow,
    forecast_next_days_with_openmeteo,
)
from .management import apply_management_windows
from .anomalies import detect_agro_anomalies
from .explain import explain_forecast_with_llm
from .risk import calculate_pest_risk


# =============================================================================
# META / LOCK DE TREINO (re-treino só quando dados mudarem)
# =============================================================================

def _model_meta_path() -> Path:
    # salva meta ao lado do modelo, com mesmo nome + ".meta.json"
    return Path(str(LGB_MODEL_PATH) + ".meta.json")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_model_meta() -> Dict[str, Any]:
    p = _model_meta_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_model_meta(meta: Dict[str, Any]) -> None:
    p = _model_meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _current_data_fingerprint() -> Dict[str, Any]:
    """
    Fingerprint objetivo do dataset local (CSV histórico).
    Usa SHA256 + mtime + tamanho.
    """
    csv_path = Path(cfg.DAILY_RAIN_CSV)
    if not csv_path.exists():
        return {}

    st = csv_path.stat()
    return {
        "csv_path": str(csv_path),
        "csv_size": int(st.st_size),
        "csv_mtime": float(st.st_mtime),
        "csv_sha256": _sha256_file(csv_path),
    }


def _data_changed_since_last_train() -> bool:
    """
    Retorna True se:
      - não existe meta anterior
      - não existe fingerprint atual
      - SHA do CSV mudou
    """
    meta = _read_model_meta()
    cur = _current_data_fingerprint()

    # se não temos CSV local fingerprintável, não dá para garantir; consideramos "mudou"
    if not cur:
        return True

    last_sha = meta.get("csv_sha256")
    return (last_sha is None) or (last_sha != cur.get("csv_sha256"))


# =============================================================================
# HELPERS
# =============================================================================

def _print_header(title: str) -> None:
    print()
    print("=" * 40)
    print(title)
    print("=" * 40)


def _check_model_consistency(model: lgb.Booster, df_daily: pd.DataFrame) -> bool:
    """
    Checa se o conjunto de features do modelo bate com o conjunto de features
    que o código atual produz (aproximação via tail + split).
    """
    try:
        model_features = model.feature_name()
        n_features_model = len(model_features)

        from .features import create_rain_features
        from .model import train_test_split_time

        df_feat, all_feature_cols = create_rain_features(
            df_daily.tail(50), target_col="y"
        )
        _, _, _, _, code_cols = train_test_split_time(
            df_feat,
            target_col="y",
            all_feature_cols=all_feature_cols,
            test_size_days=1,
        )
        n_features_code = len(code_cols)

        return n_features_model == n_features_code
    except Exception:
        return False


def _load_or_train_model(
    df_daily: pd.DataFrame,
    force_retrain: bool = False,
) -> Tuple[Any, List[str]]:
    """
    Carrega o modelo LightGBM de disco, se existir.

    NOVA REGRA (TRAVA DE RE-TREINO):
      - Só treina automaticamente se o CSV histórico (DAILY_RAIN_CSV) mudou
        (fingerprint SHA256 diferente) OU se o modelo não existe.
      - Se o modelo existe e os dados NÃO mudaram, mas o modelo está inconsistente
        com as features do código, NÃO re-treina automaticamente: aborta com
        instrução explícita (force_retrain=True ou apagar o modelo).
    """
    model_path = Path(LGB_MODEL_PATH)

    # Se usuário força, treina e sobrescreve meta
    if force_retrain:
        print("♻️ Re-treino FORÇADO (force_retrain=True).")
        model, feature_cols = train_lightgbm_model(df_daily)
        meta = _read_model_meta()
        meta.update({
            "trained_reason": "force_retrain",
            **_current_data_fingerprint(),
            "model_path": str(model_path),
            "feature_cols": feature_cols,
        })
        _write_model_meta(meta)
        return model, feature_cols

    # Se modelo existe, tentamos reaproveitar
    if model_path.exists():
        print(f"Modelo encontrado em {model_path}")

        # Se dados não mudaram, re-treino está bloqueado
        if not _data_changed_since_last_train():
            try:
                booster = lgb.Booster(model_file=str(model_path))

                # Consistência: se falhar, NÃO re-treina automaticamente
                if _check_model_consistency(booster, df_daily):
                    return booster, booster.feature_name()

                # se inconsistente, aborta com instrução clara
                raise RuntimeError(
                    "Modelo existe e os dados NÃO mudaram, mas o modelo ficou inconsistente "
                    "com as features do código atual.\n"
                    "Re-treino automático está travado por segurança.\n"
                    "Ações:\n"
                    "  (1) Rode com force_retrain=True (no app/CLI), OU\n"
                    "  (2) Apague o arquivo do modelo e o .meta.json para re-treinar.\n"
                    f"Modelo: {model_path}\n"
                    f"Meta:   {_model_meta_path()}\n"
                )
            except Exception as e:
                # Se não conseguimos sequer carregar o modelo, aqui NÃO devemos treinar
                # automaticamente se os dados não mudaram (trava). Então propagamos.
                raise RuntimeError(
                    "Falha ao carregar/reutilizar modelo existente com re-treino travado.\n"
                    "Se você quiser re-treinar, use force_retrain=True ou apague o modelo/meta.\n"
                    f"Detalhe: {e}"
                )

        # Se dados mudaram, aí sim treina novamente
        print("📌 Dados mudaram desde o último treino -> re-treinando automaticamente...")
        try:
            # tenta remover modelo antigo (opcional)
            try:
                model_path.unlink()
            except Exception:
                pass
            try:
                _model_meta_path().unlink()
            except Exception:
                pass
        except Exception:
            pass

        model, feature_cols = train_lightgbm_model(df_daily)
        meta = _read_model_meta()
        meta.update({
            "trained_reason": "data_changed",
            **_current_data_fingerprint(),
            "model_path": str(model_path),
            "feature_cols": feature_cols,
        })
        _write_model_meta(meta)
        return model, feature_cols

    # Se não existe modelo, treina
    print("Iniciando treinamento do modelo LightGBM (modelo inexistente)...")
    model, feature_cols = train_lightgbm_model(df_daily)

    meta = _read_model_meta()
    meta.update({
        "trained_reason": "model_missing",
        **_current_data_fingerprint(),
        "model_path": str(model_path),
        "feature_cols": feature_cols,
    })
    _write_model_meta(meta)
    return model, feature_cols


def _format_mgmt_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata a tabela de manejo para saída no terminal, incluindo uma coluna
    'Data' no formato 'dd/mm (DiaSemana)', baseada no índice (se for datetime)
    ou na coluna 'ds'.
    """
    out = df.copy()

    # --- Construir coluna de DATA: dd/mm (DiaSemana) ---
    if isinstance(out.index, pd.DatetimeIndex):
        datas = pd.Series(out.index)
    elif "ds" in out.columns:
        datas = pd.to_datetime(out["ds"])
    else:
        datas = pd.to_datetime(
            pd.Series(range(len(out))),
            unit="D",
            origin="2025-01-01",
        )

    datas = pd.to_datetime(datas)

    weekday_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    dias = datas.dt.strftime("%d/%m")
    dows = [weekday_labels[d] for d in datas.dt.dayofweek]

    data_fmt = [f"{d} ({dow})" for d, dow in zip(dias, dows)]

    out.insert(0, "Data", data_fmt)

    out = out.reset_index(drop=True)
    out.index.name = None

    cols_map = {
        "y_ensemble_mm": "Chuva (mm)",
        "om_et0_fao_mm": "ET0 (mm)",
        "water_balance_mm": "Saldo (mm)",
        "spray_status": "Pulverização",
        "planting_status": "Plantio",
        "nitrogen_status": "Nitrogênio",
        "heat_stress": "Calor > 36°C",
        "pest_risk": "Risco Fito",
    }

    cols_exist = [c for c in cols_map.keys() if c in out.columns]
    out = out[["Data"] + cols_exist].rename(columns=cols_map)

    numeric_cols = ["Chuva (mm)", "ET0 (mm)", "Saldo (mm)"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].astype(float).round(1)

    if "Plantio" in out.columns:
        out["Plantio"] = (
            out["Plantio"].astype(str).str.replace("PLANTIO_", "", regex=False)
        )
    if "Nitrogênio" in out.columns:
        out["Nitrogênio"] = (
            out["Nitrogênio"].astype(str).str.replace("N_", "", regex=False)
        )
    if "Calor > 36°C" in out.columns:
        out["Calor > 36°C"] = out["Calor > 36°C"].apply(
            lambda x: "SIM ⚠️" if bool(x) else "-"
        )

    return out.head(7)


# =============================================================================
# FUNÇÃO PRINCIPAL (CLI)
# =============================================================================

def main():
    _print_header(f"AGROCLIMA IA - RODADA DE TESTE: {DEFAULT_SERIES_ID.upper()}")

    farm_cfg = FARM_CONFIG.get(DEFAULT_SERIES_ID, FARM_CONFIG.get("default", {}))

    lat = farm_cfg.get("lat", DEFAULT_LAT)
    lon = farm_cfg.get("lon", DEFAULT_LON)

    regiao = farm_cfg.get("regiao", "Região não informada")
    cultura = farm_cfg.get("cultura", "N/D")
    solo = farm_cfg.get("solo", "N/D")

    print(f"📍 Região: {regiao}")
    print(f"📍 GPS: {lat}, {lon}")
    print(f"🌱 Cultura: {cultura} | Solo: {solo}")

    print("\n[1/5] Carregando dados e IA...")
    df_daily = load_or_download_daily_series(lat=lat, lon=lon)

    if df_daily is None or df_daily.empty:
        print(
            "\n❌ ERRO CRÍTICO: Histórico climático veio vazio para esta fazenda.\n"
            "Possível erro de limite da API (429 Too Many Requests) ou falha temporária.\n"
            "O relatório foi abortado. Tente novamente em alguns minutos.\n"
        )
        return

    # Aqui está a trava de re-treino aplicada
    model, feature_cols = _load_or_train_model(df_daily, force_retrain=False)

    mm_tomorrow = predict_tomorrow(df_daily, model, feature_cols=feature_cols)

    print("[3/5] Gerando previsão híbrida (Modelo + Satélite)...")
    forecast_df = forecast_next_days_with_openmeteo(
        df_daily=df_daily,
        model=model,
        days=7,
        lat=lat,
        lon=lon,
        mm_tomorrow=mm_tomorrow,
        meta=farm_cfg,
    )

    if forecast_df is None or forecast_df.empty:
        print(
            "\n❌ ERRO CRÍTICO: Falha ao obter a previsão futura (Open-Meteo / ensemble).\n"
            "O relatório foi abortado. Tente novamente em alguns minutos.\n"
        )
        return

    print("\n[4/5] Aplicando regras agronômicas...")

    forecast_df = calculate_pest_risk(forecast_df, meta=farm_cfg)

    forecast_mgmt, status_plantio = apply_management_windows(
        forecast_df, meta=farm_cfg
    )
    anomalies = detect_agro_anomalies(forecast_mgmt, meta=farm_cfg)

    print("\n[5/5] Gerando relatório técnico...\n")

    relatorio = explain_forecast_with_llm(
        forecast_mgmt,
        llm_fn=None,
        cultura=farm_cfg.get("cultura", ""),
        estagio_fenologico=farm_cfg.get("estagio_fenologico", ""),
        solo=farm_cfg.get("solo", ""),
        regiao=farm_cfg.get("regiao", ""),
        sistema=farm_cfg.get("sistema", ""),
        anomalies=anomalies,
    )

    print(relatorio)
    print("-" * 30)

    print("\nTABELA TÉCNICA SEMANAL:\n")
    tabela = _format_mgmt_table(forecast_mgmt)
    print(tabela.to_string(index=False))
    print("\n")


if __name__ == "__main__":
    main()







    
