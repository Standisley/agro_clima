# agroclima_ia/app_streamlit.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------------
# Ajuste de caminho para rodar:
#   streamlit run agroclima_ia/app_streamlit.py
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

# --- IMPORTAÇÃO SEGURA DA IA (Evita Tela Branca) ---
try:
    from agroclima_ia.explain import explain_forecast_with_llm, call_gemini_llm
except ImportError:
    # Se der erro no explain.py, define versões dummy para o app abrir
    print("AVISO: Falha ao importar explain.py ou google-generativeai.")
    def explain_forecast_with_llm(df, llm_fn=None, **kwargs):
        return "⚠️ Erro no módulo de explicação (explain.py). Verifique as instalações."
    def call_gemini_llm(p, k): return None


# =============================================================================
# OTIMIZAÇÃO: Cache do Modelo de IA (Machine Learning)
# =============================================================================
@st.cache_resource(show_spinner="Treinando inteligência artificial (cache)...")
def get_trained_model_cached(df_daily: pd.DataFrame):
    """Treina o modelo LightGBM e o mantém em memória."""
    return train_lightgbm_model(df_daily)


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
    # 1) Config
    set_active_farm_in_config(farm_id)
    farm_cfg = get_farm_profile(farm_id)
    lat = farm_cfg.get("lat", DEFAULT_LAT)
    lon = farm_cfg.get("lon", DEFAULT_LON)
    series_id = farm_cfg.get("series_id", DEFAULT_SERIES_ID)

    # 2) Histórico (Cacheado no forecast.py)
    df_daily = load_or_download_daily_series(
        lat=lat,
        lon=lon,
        force_refresh=False,
    )

    if df_daily is None or df_daily.empty:
        raise RuntimeError("Histórico climático vazio (erro API).")

    # 3) Treino OTIMIZADO (Cache resource)
    model, feature_cols = get_trained_model_cached(df_daily)

    # Previsão amanhã
    mm_tomorrow = predict_tomorrow(df_daily, model, feature_cols=feature_cols)

    # 4) Previsão híbrida (Cache interno APIs)
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
        raise RuntimeError("Falha ao obter previsão futura (API).")

    # 5) Processamento Agronômico
    forecast_df = calculate_pest_risk(forecast_df, meta=farm_cfg)

    forecast_mgmt, status_plantio = apply_management_windows(
        forecast_df,
        meta=farm_cfg,
    )
    anomalies = detect_agro_anomalies(forecast_mgmt, meta=farm_cfg)

    # ---------------------------------------------------------
    # INTEGRAÇÃO DO LLM (Google Gemini)
    # ---------------------------------------------------------
    # Definimos a função wrapper. Se não tiver KEY, passa None (modo offline)
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
# APP STREAMLIT (INTERFACE)
# =============================================================================
def main():
    st.set_page_config(
        page_title="newClima IA",
        page_icon="🌦️",
        layout="wide",
    )

    st.title("🌦️ newClima IA – Painel Agronômico")

    # ---------------------------------------------------------------------
    # SIDEBAR DETALHADA
    # ---------------------------------------------------------------------
    farm_ids = sorted(cfg.FARM_CONFIG.keys())
    default_id = getattr(cfg, "ACTIVE_FARM_ID", DEFAULT_SERIES_ID)
    if default_id not in farm_ids:
        default_id = farm_ids[0]

    def _label(fid: str) -> str:
        f = cfg.FARM_CONFIG[fid]
        regiao = f.get("regiao", "N/D")
        cultura = f.get("cultura", "N/D")
        estagio = f.get("estagio_fenologico", "")
        return (
            f"{fid} – {regiao} | {cultura} "
            f"{('(' + estagio + ')') if estagio else ''}"
        )

    st.sidebar.header("🌾 Configuração")
    
    # --- CAMPO PARA API KEY ---
    st.sidebar.markdown("**🤖 Inteligência Artificial (LLM)**")
    gemini_key = st.sidebar.text_input(
        "Google Gemini API Key:", 
        type="password", 
        help="Cole aqui sua chave do Google AI Studio. Se deixar vazio, usa o modo Offline."
    )
    st.sidebar.markdown("---")

    selected_farm_id = st.sidebar.selectbox(
        "Perfil/Fazenda:",
        options=farm_ids,
        index=farm_ids.index(default_id),
        format_func=_label,
    )

    farm_cfg = get_farm_profile(selected_farm_id)
    
    st.sidebar.subheader("📍 Detalhes da fazenda")
    
    st.sidebar.write(
        f"**ID da Série:** `{farm_cfg.get('series_id', selected_farm_id)}`"
    )
    st.sidebar.write(f"**Região:** {farm_cfg.get('regiao', 'N/D')}")
    st.sidebar.write(f"**Cultura:** {farm_cfg.get('cultura', 'N/D')}")
    st.sidebar.write(
        f"**Estágio:** {farm_cfg.get('estagio_fenologico', 'N/D')}"
    )
    st.sidebar.write(f"**Sistema:** {farm_cfg.get('sistema', 'N/D')}")
    st.sidebar.write(f"**Solo:** {farm_cfg.get('solo', 'N/D')}")
    st.sidebar.write(
        f"**GPS:** {farm_cfg.get('lat', DEFAULT_LAT)}, "
        f"{farm_cfg.get('lon', DEFAULT_LON)}"
    )

    st.markdown(
        """
        Esta interface usa **o mesmo núcleo de modelo e regras** do script de linha de comando,
        mas permite trocar de fazenda/perfil diretamente pela barra lateral.
        """
    )

    if st.button("🚀 Rodar previsão Agronômica (7 dias)", type="primary"):
        try:
            with st.spinner("Processando dados e IA (AgroClima)..."):
                # Passamos a API KEY para o pipeline
                relatorio, tabela, series_id = run_pipeline(selected_farm_id, api_key=gemini_key)
            
            # --------------------------------------------------------
            # LAYOUT VERTICAL (Relatório no topo, Tabela embaixo)
            # --------------------------------------------------------
            st.subheader("📋 Relatório Técnico")
            st.markdown(relatorio.replace("\n", "  \n"))
            
            st.markdown("---") 
            
            st.subheader("📑 Tabela Técnica Semanal")
            st.dataframe(tabela, use_container_width=True)
                
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            # Dica para debug: descomente abaixo para ver o erro completo no terminal
            # raise e
    else:
        st.info("Selecione o perfil e clique no botão para rodar a previsão.")


if __name__ == "__main__":
    main()





