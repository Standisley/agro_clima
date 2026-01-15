# agroclima_ia/app_streamlit.py

from __future__ import annotations

import sys
import time
import pickle
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------------
# Ajuste de caminho
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------------
# Importações do Projeto
# -------------------------------------------------------------------------
import agroclima_ia.config as cfg
from agroclima_ia.config import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_SERIES_ID,
    get_farm_profile,
)

from agroclima_ia.forecast import (
    load_or_download_daily_series,
    train_lightgbm_model,
    predict_tomorrow,
    forecast_next_days_with_openmeteo,
)

from agroclima_ia.management import apply_management_windows
from agroclima_ia.anomalies import detect_agro_anomalies
from agroclima_ia.risk import calculate_pest_risk
from agroclima_ia.main import _format_mgmt_table

# --- IMPORTAÇÃO SEGURA DA IA ---
try:
    from agroclima_ia.explain import explain_forecast_with_llm, call_gemini_llm
except ImportError:
    def explain_forecast_with_llm(df, llm_fn=None, **kwargs):
        return "⚠️ Erro no módulo de explicação (explain.py). Verifique as instalações."
    def call_gemini_llm(p, k): return None


# =============================================================================
# OTIMIZAÇÃO: Cache Inteligente (Automático - 30 DIAS)
# =============================================================================
@st.cache_resource
def get_trained_model_cached(df_daily: pd.DataFrame, series_id: str):
    """
    Gerencia o modelo automaticamente:
    - Se existe arquivo e tem menos de 30 DIAS: Carrega do disco (RÁPIDO).
    - Se é velho ou não existe: Treina e salva (LENTO).
    """
    model_checkpoint = cfg.MODELS_DIR / f"{series_id}_checkpoint.pkl"
    
    # Tempo limite de validade do modelo em segundos
    # 30 dias * 24 horas * 60 minutos * 60 segundos
    MODEL_TTL_SECONDS = 30 * 24 * 60 * 60 
    
    should_retrain = True

    # 1. Verifica se o arquivo existe e é recente
    if model_checkpoint.exists():
        last_modified = model_checkpoint.stat().st_mtime
        age_seconds = time.time() - last_modified
        
        if age_seconds < MODEL_TTL_SECONDS:
            should_retrain = False
        else:
            print(f"Modelo expirado ({age_seconds/86400:.1f} dias). Retreinando...")

    # 2. Carrega se estiver válido
    if not should_retrain:
        try:
            with open(model_checkpoint, "rb") as f:
                model, feature_cols = pickle.load(f)
            return model, feature_cols
        except Exception as e:
            print(f"Erro ao ler modelo, forçando retreino: {e}")
            should_retrain = True

    # 3. Treina e Salva (se necessário)
    # Removido st.toast/st.warning daqui para evitar erro de Cache
    print(f"Treinando IA para {series_id}...") 
    
    model, feature_cols = train_lightgbm_model(df_daily)

    try:
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(model_checkpoint, "wb") as f:
            pickle.dump((model, feature_cols), f)
            print(f"Modelo salvo em: {model_checkpoint}")
    except Exception as e:
        print(f"Aviso: Não foi possível salvar o modelo em disco: {e}")

    return model, feature_cols


# =============================================================================
# SETUP DE CONFIGURAÇÃO
# =============================================================================
def set_active_farm_in_config(farm_id: str) -> None:
    if farm_id not in cfg.FARM_CONFIG:
        farm_id = "default"

    cfg.ACTIVE_FARM_ID = farm_id
    cfg.ACTIVE_FARM = cfg.FARM_CONFIG.get(farm_id, cfg.FARM_CONFIG["default"])

    farm = cfg.ACTIVE_FARM
    series_id = farm["series_id"]

    cfg.DEFAULT_LAT = farm["lat"]
    cfg.DEFAULT_LON = farm["lon"]
    cfg.DEFAULT_SERIES_ID = series_id

    cfg.DAILY_RAIN_CSV = cfg.DATA_DIR / f"{series_id}_daily_rain.csv"
    cfg.FARM_OBS_CSV = cfg.DATA_DIR / f"{series_id}_farm_obs.csv"
    cfg.LGB_MODEL_PATH = cfg.MODELS_DIR / f"{series_id}_lightgbm.txt"


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def run_pipeline(farm_id: str, api_key: str = "") -> Tuple[str, pd.DataFrame, str]:
    set_active_farm_in_config(farm_id)
    farm_cfg = get_farm_profile(farm_id)
    lat = farm_cfg.get("lat", DEFAULT_LAT)
    lon = farm_cfg.get("lon", DEFAULT_LON)
    series_id = farm_cfg.get("series_id", DEFAULT_SERIES_ID)

    df_daily = load_or_download_daily_series(lat=lat, lon=lon, force_refresh=False)

    if df_daily is None or df_daily.empty:
        raise RuntimeError("Histórico climático vazio (erro API).")

    # Chama a função inteligente (sem pedir permissão ao usuário)
    model, feature_cols = get_trained_model_cached(df_daily, series_id)
    
    mm_tomorrow = predict_tomorrow(df_daily, model, feature_cols=feature_cols)

    forecast_df = forecast_next_days_with_openmeteo(
        df_daily=df_daily, model=model, days=7, lat=lat, lon=lon, mm_tomorrow=mm_tomorrow, meta=farm_cfg,
    )

    if forecast_df is None or forecast_df.empty:
        raise RuntimeError("Falha ao obter previsão futura (API).")

    forecast_df = calculate_pest_risk(forecast_df, meta=farm_cfg)
    forecast_mgmt, status_plantio = apply_management_windows(forecast_df, meta=farm_cfg)
    anomalies = detect_agro_anomalies(forecast_mgmt, meta=farm_cfg)

    # Integração LLM
    llm_function = None
    if api_key and api_key.strip():
        def _my_llm_wrapper(prompt_txt: str) -> str:
            return call_gemini_llm(prompt_txt, api_key)
        llm_function = _my_llm_wrapper

    relatorio = explain_forecast_with_llm(
        forecast_mgmt,
        llm_fn=llm_function,
        cultura=farm_cfg.get("cultura", ""),
        estagio_fenologico=farm_cfg.get("estagio_fenologico", ""),
        solo=farm_cfg.get("solo", ""),
        regiao=farm_cfg.get("regiao", ""),
        sistema=farm_cfg.get("sistema", ""),
        anomalies=anomalies,
    )

    tabela = _format_mgmt_table(forecast_mgmt)
    return relatorio, tabela, series_id


# =============================================================================
# APP STREAMLIT
# =============================================================================
def main():
    st.set_page_config(
        page_title="newClima IA",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ newClima IA – Painel Agronômico")

    # --- SIDEBAR ---
    farm_ids = sorted(cfg.FARM_CONFIG.keys())
    default_id = getattr(cfg, "ACTIVE_FARM_ID", DEFAULT_SERIES_ID)
    if default_id not in farm_ids: default_id = farm_ids[0]

    def _label(fid: str) -> str:
        f = cfg.FARM_CONFIG[fid]
        return f"{fid} – {f.get('regiao', '')} | {f.get('cultura', '')}"

    st.sidebar.header("🌾 Configuração")
    
    # -----------------------------------------------------------------
    # LÓGICA DE SEGREDOS
    # -----------------------------------------------------------------
    st.sidebar.markdown("**🤖 Inteligência Artificial (LLM)**")
    gemini_key = st.secrets.get("GEMINI_KEY", "")

    if gemini_key:
        st.sidebar.success("🔑 Chave carregada automaticamente!")
    else:
        st.sidebar.warning("⚠️ Chave não encontrada nos Secrets.")
        gemini_key = st.sidebar.text_input("Cole a chave aqui manualmente:", type="password")

    st.sidebar.markdown("---")

    selected_farm_id = st.sidebar.selectbox(
        "Perfil/Fazenda:",
        options=farm_ids,
        index=farm_ids.index(default_id),
        format_func=_label,
    )

    # -----------------------------------------------------------------
    # DETALHES DA FAZENDA
    # -----------------------------------------------------------------
    farm_cfg = get_farm_profile(selected_farm_id)
    
    st.sidebar.subheader("📍 Detalhes da fazenda")
    st.sidebar.write(f"**ID da Série:** `{farm_cfg.get('series_id', selected_farm_id)}`")
    st.sidebar.write(f"**Região:** {farm_cfg.get('regiao', 'N/D')}")
    st.sidebar.write(f"**Cultura:** {farm_cfg.get('cultura', 'N/D')}")
    st.sidebar.write(f"**Estágio:** {farm_cfg.get('estagio_fenologico', 'N/D')}")
    st.sidebar.write(f"**Sistema:** {farm_cfg.get('sistema', 'N/D')}")
    st.sidebar.write(f"**Solo:** {farm_cfg.get('solo', 'N/D')}")
    st.sidebar.write(f"**GPS:** {farm_cfg.get('lat', DEFAULT_LAT)}, {farm_cfg.get('lon', DEFAULT_LON)}")
    
    if st.button("🚀 Rodar previsão Agronômica (7 dias)", type="primary"):
        try:
            with st.spinner("Conectando satélites, IA e processando..."):
                relatorio, tabela, series_id = run_pipeline(
                    selected_farm_id, 
                    api_key=gemini_key
                )
            
            st.subheader("📋 Relatório Técnico")
            st.markdown(relatorio.replace("\n", "  \n"))
            st.markdown("---") 
            st.subheader("📑 Tabela Técnica Semanal")
            st.dataframe(tabela, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erro: {e}")

if __name__ == "__main__":
    main()





