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
# Função Conexão LLM
# =============================================================================
def call_gemini_llm(prompt_text: str, api_key: str) -> str:
    if not HAS_GOOGLE_LIB: return "⚠️ Erro: Biblioteca 'google-generativeai' não instalada."
    if not api_key: return "⚠️ Erro: API Key não fornecida."

    try:
        genai.configure(api_key=api_key)
        config = genai.types.GenerationConfig(temperature=0.4)
        
        # Tenta modelos em ordem de preferência
        valid_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        last_error = None
        for model_name in valid_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_text, generation_config=config)
                return response.text
            except Exception as e:
                last_error = e
                continue
        return f"⚠️ Falha na IA. Erro: {last_error}"
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
    
    # --- 1. CONSULTA O ZARC (A Mágica acontece aqui) ---
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
    monitoramento_block = _format_monitoramento_block(anomalies_dict)

    # 3. Janelas Operacionais (Resumido)
    pest_risk_txt = "BAIXO"
    if "pest_risk" in df.columns:
        vc = df["pest_risk"].value_counts()
        if vc.get("RISCO_ALTO_FERRUGEM", 0) > 0: pest_risk_txt = "ALTO"
        elif vc.get("RISCO_ATENÇÃO", 0) > 0: pest_risk_txt = "ATENÇÃO"

    pulverizacao_txt = "Sem janelas seguras."
    if "spray_status" in df.columns:
        verde = (df["spray_status"] == "VERDE").sum()
        if verde > 0: pulverizacao_txt = f"{verde} dias VERDE."
        else: pulverizacao_txt = "Restrito (Amarelo/Vermelho)."

    plantio_txt = "Inadequado."
    if "planting_status" in df.columns:
        ok = (df["planting_status"] == "PLANTIO_OK").sum()
        if ok > 0: plantio_txt = f"{ok} dias FAVORÁVEIS."

    adubacao_txt = "Verificar umidade."
    if "nitrogen_status" in df.columns:
        ok_n = (df["nitrogen_status"] == "N_OK").sum()
        if ok_n > 0: adubacao_txt = f"{ok_n} dias FAVORÁVEIS."

    # -------------------------------------------------------------------------
    # PROMPT PARA LLM (Aqui pedimos para a IA falar sobre o ZARC)
    # -------------------------------------------------------------------------
    if llm_fn is not None:
        prompt = f"""
        Você é o AgroClima IA. Gere um relatório técnico direto.

        DADOS:
        - Fazenda: {cultura.upper()} | {regiao}
        - Solo: {solo}
        - ZARC (Risco Oficial): {zarc_status_llm}
        - Clima (7d): Chuva {chuva_total:.1f}mm | Saldo {saldo_total:.1f}mm
        - Alertas: {monitoramento_plain}
        
        JANELAS:
        - Plantio: {plantio_txt}
        - Adubação: {adubacao_txt}

        IMPORTANTE:
        Se o ZARC estiver "FORA DA JANELA" ou "40%", ALERTE o produtor sobre perda de seguro.
        Se estiver "20%", confirme que está seguro plantar.

        FORMATO DE SAÍDA (Markdown):

        ### 📋 RELATÓRIO TÉCNICO: {cultura.upper()}
        📍 **{regiao}** | Solo: {solo}

        **1. STATUS ZARC (Risco Oficial):**
        👉 **{zarc_txt}**

        **2. CLIMA (7 dias):**
        • Chuva: {chuva_total:.1f} mm | Saldo: {saldo_total:.1f} mm

        **3. ANÁLISE E RECOMENDAÇÃO (IA):**
        (Sua análise aqui)
        """
        return llm_fn(prompt)

    # -------------------------------------------------------------------------
    # TEMPLATE OFFLINE
    # -------------------------------------------------------------------------
    return f"""### 📋 RELATÓRIO: {cultura.upper()}
📍 **{regiao}**

**1. STATUS ZARC:**
👉 **{zarc_txt}**

**CLIMA:** Chuva: {chuva_total:.1f}mm | Saldo: {saldo_total:.1f}mm
{monitoramento_block}

**MANEJO:**
🚜 Pulverização: {pulverizacao_txt}
🌱 Plantio: {plantio_txt}
🌿 Adubação: {adubacao_txt}

*(Modo Offline)*
"""