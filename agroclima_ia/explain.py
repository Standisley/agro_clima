# agroclima_ia/explain.py

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import pandas as pd

# --- TENTA IMPORTAR O MÓDULO ZARC ---
try:
    from agroclima_ia.zarc import check_zarc_risk
except ImportError:
    # Se der erro no import, cria uma função "tapa-buraco" para não quebrar o app
    def check_zarc_risk(r, c, s): return "N/D (Módulo não encontrado)"

# Tenta importar a biblioteca do Google
try:
    import google.generativeai as genai
    HAS_GOOGLE_LIB = True
except ImportError:
    HAS_GOOGLE_LIB = False

# =============================================================================
# Funções Auxiliares
# =============================================================================
def _fmt_mm(v: float) -> str:
    return f"{v:.1f} mm"

def _format_monitoramento_block(anomalies: Optional[Dict[str, Any]]) -> str:
    if anomalies is None or not isinstance(anomalies, dict):
        return "2. ✅ MONITORAMENTO:\n• Sem riscos críticos de anomalia climática."

    has_critical = bool(anomalies.get("has_critical", False))
    messages: List[str] = anomalies.get("messages") or []
    summary: str = anomalies.get("summary") or ""

    if not messages and not has_critical:
        return "2. ✅ MONITORAMENTO:\n• Sem riscos críticos de anomalia climática."

    header = "2. ⚠ MONITORAMENTO (Riscos Climáticos Detectados):" if has_critical else "2. 🔎 MONITORAMENTO (Anomalias observadas):"
    linhas = [header, "⚠ O algoritmo identificou anomalias climáticas relevantes:"]
    for msg in messages: linhas.append(f"• {msg}")
    return "\n".join(linhas)

# =============================================================================
# Função Conexão LLM (ROBUSTA / AUTO-DISCOVERY)
# =============================================================================
def call_gemini_llm(prompt_text: str, api_key: str) -> str:
    if not HAS_GOOGLE_LIB: return "⚠️ Erro: Biblioteca 'google-generativeai' não instalada."
    if not api_key: return "⚠️ Erro: API Key não fornecida."

    try:
        genai.configure(api_key=api_key)
        config = genai.types.GenerationConfig(temperature=0.4)
        
        # 1. Tenta o modelo padrão mais rápido primeiro
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt_text, generation_config=config)
            return response.text
        except Exception:
            pass # Falhou? Vamos para a busca automática

        # 2. Busca Automática (Lista quais modelos sua conta TEM acesso)
        valid_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'gemini' in m.name:
                        valid_models.append(m.name)
        except Exception as e_list:
            return f"⚠️ Erro ao listar modelos: {e_list}"

        if not valid_models:
            return "⚠️ Erro: Nenhum modelo Gemini disponível na sua conta."

        # Ordena para tentar os 'flash' primeiro (mais rápidos)
        valid_models.sort(key=lambda x: 'flash' not in x)

        last_error = None
        for model_name in valid_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_text, generation_config=config)
                return response.text
            except Exception as e:
                last_error = e
                continue
        
        return f"⚠️ Falha na IA. Tentamos {valid_models} e todos falharam. Erro final: {last_error}"

    except Exception as e:
        return f"⚠️ Erro Geral LLM: {e}"

# =============================================================================
# Função Principal (COM O VISUAL DO ZARC)
# =============================================================================

def explain_forecast_with_llm(
    df_forecast: pd.DataFrame,
    llm_fn: Optional[Callable[[str], str]] = None,
    cultura: str = "",
    estagio_fenologico: str = "",
    solo: str = "",
    regiao: str = "",
    sistema: str = "",
    anomalies: Optional[Dict[str, Any]] = None,
) -> str:
    df = df_forecast.copy()
    
    # --- 1. CONSULTA O ZARC ---
    risco_zarc = check_zarc_risk(regiao, cultura, solo)
    
    # Formatação visual bonita para o relatório
    if "20%" in risco_zarc: 
        zarc_txt = f"✅ DENTRO DA JANELA (Risco: {risco_zarc})"
        zarc_status_llm = f"Favorável ({risco_zarc})"
    elif "30%" in risco_zarc or "40%" in risco_zarc: 
        zarc_txt = f"⚠️ RISCO MÉDIO/ALTO ({risco_zarc})"
        zarc_status_llm = f"Atenção ({risco_zarc})"
    elif "FORA" in risco_zarc: 
        zarc_txt = f"⛔ {risco_zarc} (Sem cobertura de seguro)"
        zarc_status_llm = "PROIBITIVO (Fora da janela)"
    else: 
        zarc_txt = f"ℹ️ {risco_zarc}"
        zarc_status_llm = risco_zarc

    # 2. Dados Climáticos
    chuva_col = "y_ensemble_mm" if "y_ensemble_mm" in df.columns else "y"
    et0_col = "om_et0_fao_mm"
    saldo_col = "water_balance_mm"

    chuva_total = float(df[chuva_col].sum()) if chuva_col in df.columns else 0.0
    et0_total = float(df[et0_col].sum()) if et0_col in df.columns else 0.0
    saldo_total = float(df[saldo_col].sum()) if saldo_col in df.columns else 0.0
    n_dias_secos = int((df[chuva_col] < 0.5).sum()) if chuva_col in df.columns else 0
    
    # Normalização de anomalias
    anomalies_dict = anomalies if isinstance(anomalies, dict) else None
    if anomalies and not isinstance(anomalies, dict): 
         anomalies_dict = {"has_critical": True, "messages": list(anomalies)}

    monitoramento_plain = "Sem riscos críticos."
    if anomalies_dict and anomalies_dict.get("messages"):
        monitoramento_plain = "\n".join(f"- {m}" for m in anomalies_dict["messages"])
    
    # Bloco formatado para o modo offline
    monitoramento_block = _format_monitoramento_block