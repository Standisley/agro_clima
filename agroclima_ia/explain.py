# agroclima_ia/explain.py

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import pandas as pd

# --- TENTA IMPORTAR O MÓDULO ZARC ---
try:
    from agroclima_ia.zarc import check_zarc_risk
except ImportError:
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
        return "• Anomalias: Sem riscos críticos identificados."

    has_critical = bool(anomalies.get("has_critical", False))
    messages: List[str] = anomalies.get("messages") or []
    
    if not messages and not has_critical:
        return "• Anomalias: Sem riscos críticos de anomalia climática."

    texto = "• ⚠ **ALERTAS CLIMÁTICOS:**\n"
    for msg in messages:
        texto += f"  - {msg}\n"
    return texto.strip()

# =============================================================================
# Função Conexão LLM (ATUALIZADA)
# =============================================================================
def call_gemini_llm(prompt_text: str, api_key: str) -> str:
    if not HAS_GOOGLE_LIB: return "⚠️ Erro: Biblioteca 'google-generativeai' não instalada."
    if not api_key: return "⚠️ Erro: API Key não fornecida."

    try:
        genai.configure(api_key=api_key)
        config = genai.types.GenerationConfig(temperature=0.4)
        
        models_to_try = [
            'gemini-1.5-flash', 
            'gemini-1.5-pro', 
            'gemini-1.0-pro', 
            'gemini-pro'
        ]
        
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                models_to_try = [m for m in models_to_try if m in available_models]
                if not models_to_try:
                    models_to_try = available_models
        except: 
            pass 

        last_error = None
        for model_name in models_to_try:
            try:
                if "models/" in model_name:
                    model_name = model_name.replace("models/", "")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_text, generation_config=config)
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                continue 
        
        return f"⚠️ Falha na IA. Nenhum modelo funcionou. Erro final: {last_error}"

    except Exception as e:
        return f"⚠️ Erro Geral Conexão LLM: {e}"

# =============================================================================
# Função Principal de Explicação
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
    if "20%" in risco_zarc: 
        zarc_txt = f"✅ DENTRO DA JANELA (Risco: {risco_zarc})"
    elif "30%" in risco_zarc or "40%" in risco_zarc: 
        zarc_txt = f"⚠️ RISCO MÉDIO/ALTO ({risco_zarc})"
    elif "FORA" in risco_zarc: 
        zarc_txt = f"⛔ {risco_zarc} (Sem cobertura de seguro)"
    else: 
        zarc_txt = f"ℹ️ {risco_zarc}"

    # 2. Dados Climáticos
    chuva_col = "y_ensemble_mm" if "y_ensemble_mm" in df.columns else "y"
    et0_col = "om_et0_fao_mm"
    saldo_col = "water_balance_mm"

    chuva_total = float(df[chuva_col].sum()) if chuva_col in df.columns else 0.0
    et0_total = float(df[et0_col].sum()) if et0_col in df.columns else 0.0
    saldo_total = float(df[saldo_col].sum()) if saldo_col in df.columns else 0.0
    
    # 3. Monitoramento e Anomalias
    anomalies_dict = anomalies if isinstance(anomalies, dict) else None
    if anomalies and not isinstance(anomalies, dict): 
         anomalies_dict = {"has_critical": True, "messages": list(anomalies)}
    
    monitoramento_txt = _format_monitoramento_block(anomalies_dict)

    # 4. Janelas Operacionais
    pest_risk_txt = "BAIXO"
    if "pest_risk" in df.columns:
        vc = df["pest_risk"].value_counts()
        if vc.get("RISCO_ALTO_FERRUGEM", 0) > 0: pest_risk_txt = "ALTO 🚩"
        elif vc.get("RISCO_ATENÇÃO", 0) > 0: pest_risk_txt = "ATENÇÃO ⚠️"

    pulverizacao_txt = "Sem janelas."
    if "spray_status" in df.columns:
        verde = (df["spray_status"] == "VERDE").sum()
        if verde > 0: pulverizacao_txt = f"{verde} dias VERDE ✅"
        else: pulverizacao_txt = "Restrito (Amarelo/Vermelho) ⛔"

    plantio_txt = "Inadequado."
    if "planting_status" in df.columns:
        ok = (df["planting_status"] == "PLANTIO_OK").sum()
        if ok > 0: plantio_txt = f"{ok} dias FAVORÁVEIS ✅"

    # --- AJUSTE AGRONÔMICO PARA SOJA ---
    adubacao_txt = "Verificar umidade."
    if "soja" in cultura.lower():
        adubacao_txt = "Não se aplica (Fixação Biológica) 🦠"
    else:
        if "nitrogen_status" in df.columns:
            ok_n = (df["nitrogen_status"] == "N_OK").sum()
            if ok_n > 0: adubacao_txt = f"{ok_n} dias FAVORÁVEIS ✅"

    # =========================================================================
    # MONTAGEM DO CABEÇALHO FIXO
    # =========================================================================
    saldo_icon = '🔵 Superávit' if saldo_total >= 0 else '🟠 Déficit'
    
    header_report = f"""### 📋 RELATÓRIO TÉCNICO: {cultura.upper()}
📍 **{regiao}** | Solo: {solo}

**1. STATUS ZARC (Risco Oficial):**
👉 **{zarc_txt}**

**2. CLIMA (Acumulado 7 dias):**
• Chuva Prevista: **{chuva_total:.1f} mm**
• ET0 (Demanda): {et0_total:.1f} mm
• Saldo Hídrico: **{saldo_total:.1f} mm** ({saldo_icon})

**3. MONITORAMENTO & RISCOS:**
{monitoramento_txt}
• Risco Fitossanitário: {pest_risk_txt}

**4. JANELAS OPERACIONAIS:**
• 🚜 Pulverização: {pulverizacao_txt}
• 🌱 Plantio (Condição Solo): {plantio_txt}
• 🌿 Adubação (N): {adubacao_txt}
"""

    if llm_fn is None:
        return header_report + "\n*(Modo Offline - Sem análise de IA)*"

    # =========================================================================
    # LÓGICA DE CONTEXTO E PROMPT (REFINADA COM CAUSA DO ZARC)
    # =========================================================================
    estagio_lower = str(estagio_fenologico).lower()
    
    # Lógica de Contexto
    if any(x in estagio_lower for x in ["v", "vegetativo", "perfilhamento", "crescimento"]):
        contexto = (
            "A CULTURA JÁ ESTÁ PLANTADA E EM CRESCIMENTO VEGETATIVO. "
            "NÃO RECOMENDE PLANTIO. "
            "FOQUE EM: Adubação de cobertura (se não for soja) e Pragas (Lagartas)."
        )
    elif any(x in estagio_lower for x in ["r", "reprodutivo", "flor", "enchimento", "frutificacao"]):
        contexto = (
            "A CULTURA ESTÁ EM REPRODUÇÃO (Fase Crítica). "
            "IMPORTANTE SOBRE O ZARC: Como a planta já está no campo, se o ZARC indica risco alto (30/40%), "
            "EXPLIQUE O MOTIVO PROVÁVEL (Ex: Risco de Veranico/Deficiência Hídrica nesta época ou excesso de chuva). "
            "O ZARC aqui serve como ALERTA DE ESTRESSE CLIMÁTICO, não de plantio."
        )
    elif any(x in estagio_lower for x in ["colheita", "maturacao"]):
        contexto = "A CULTURA ESTÁ EM MATURAÇÃO/COLHEITA. Foque em logística e umidade do grão."
    else:
        contexto = "Geral."

    prompt = f"""
    Atue como o Agrônomo Sênior do AgroClima IA.
    
    DADOS DO RELATÓRIO:
    {header_report}
    
    ESTÁGIO ATUAL: {estagio_fenologico}
    CONTEXTO OBRIGATÓRIO: {contexto}
    
    SUA TAREFA:
    Escreva APENAS o item "5. ANÁLISE E RECOMENDAÇÃO AGRONÔMICA (IA)".
    
    DIRETRIZES ESPECÍFICAS:
    1. **Explique o ZARC:** Se o risco for 20%, diga que o ambiente está seguro. Se for 30% ou 40%, explique QUE TIPO de risco o produtor corre agora (provavelmente seca ou chuva excessiva, dependendo do saldo hídrico acima).
    2. **Soja em R1:** O foco é sanidade (Ferrugem) e Água. O risco ZARC indica vulnerabilidade climática.
    3. **Nitrogênio:** Se for Soja, confirme que não precisa de N.
    
    SAÍDA ESPERADA:
    **5. ANÁLISE E RECOMENDAÇÃO AGRONÔMICA (IA):**
    (Seu texto aqui, direto e técnico)
    """

    resposta_ia = llm_fn(prompt)
    
    if not resposta_ia:
        resposta_ia = "⚠️ A IA analisou os dados mas não retornou texto. Verifique sua conexão."

    return header_report + "\n" + resposta_ia