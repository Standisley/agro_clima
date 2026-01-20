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
# Função Conexão LLM (DESCOBERTA AUTOMÁTICA DE MODELO)
# =============================================================================
def call_gemini_llm(prompt_text: str, api_key: str) -> str:
    if not HAS_GOOGLE_LIB: return "⚠️ Erro: Biblioteca 'google-generativeai' não instalada."
    if not api_key: return "⚠️ Erro: API Key não fornecida."

    try:
        genai.configure(api_key=api_key)
        config = genai.types.GenerationConfig(temperature=0.4)
        
        # --- SOLUÇÃO DEFINITIVA: Listar modelos disponíveis em vez de adivinhar ---
        try:
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            # Estratégia de prioridade: Flash > Pro 1.5 > Pro 1.0 > Qualquer um
            chosen_model = None
            
            # 1. Tenta achar o Flash (mais rápido/barato)
            for m in available_models:
                if 'flash' in m.lower():
                    chosen_model = m
                    break
            
            # 2. Se não achar, tenta o Pro 1.5
            if not chosen_model:
                for m in available_models:
                    if '1.5-pro' in m.lower():
                        chosen_model = m
                        break
            
            # 3. Se não achar, pega o primeiro da lista (gemini-pro antigo)
            if not chosen_model and available_models:
                chosen_model = available_models[0]

            if not chosen_model:
                return "⚠️ Falha: Nenhum modelo de texto encontrado na sua API Key."

            # Gera com o modelo encontrado
            model = genai.GenerativeModel(chosen_model)
            response = model.generate_content(prompt_text, generation_config=config)
            
            if response and response.text:
                return response.text
                
        except Exception as e:
            return f"⚠️ Erro ao listar/chamar modelos: {e}"
        
        return "⚠️ Falha na IA. Resposta vazia."

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
    
    # 1. ZARC INTELIGENTE
    risco_zarc = check_zarc_risk(regiao, cultura, solo)
    zarc_label = "STATUS ZARC (Risco Oficial)"
    estagio_lower = str(estagio_fenologico).lower()
    
    fases_pos_plantio = [
        "vegetativo", "v1", "v2", "v3", "v4", "v5", "perfilhamento", 
        "crescimento", "reprodutivo", "r1", "r2", "r3", "r4", "r5", 
        "enchimento", "maturacao", "colheita", "frutificacao", "espigamento"
    ]
    
    if any(f in estagio_lower for f in fases_pos_plantio):
        zarc_label = "RISCO CLIMÁTICO REGIONAL (ZARC Atual)"

    if "20%" in risco_zarc: zarc_txt = f"✅ DENTRO DA JANELA (Risco: {risco_zarc})"
    elif "30%" in risco_zarc or "40%" in risco_zarc: zarc_txt = f"⚠️ RISCO MÉDIO/ALTO ({risco_zarc})"
    elif "FORA" in risco_zarc: zarc_txt = f"⛔ {risco_zarc} (Sem cobertura de seguro)"
    else: zarc_txt = f"ℹ️ {risco_zarc}"

    # 2. Dados Climáticos
    chuva_total = float(df["y_ensemble_mm"].sum()) if "y_ensemble_mm" in df.columns else 0.0
    et0_total = float(df["om_et0_fao_mm"].sum()) if "om_et0_fao_mm" in df.columns else 0.0
    saldo_total = float(df["water_balance_mm"].sum()) if "water_balance_mm" in df.columns else 0.0
    
    # 3. Monitoramento e Anomalias
    anomalies_dict = anomalies if isinstance(anomalies, dict) else None
    if anomalies and not isinstance(anomalies, dict): 
         anomalies_dict = {"has_critical": True, "messages": list(anomalies)}
    monitoramento_txt = _format_monitoramento_block(anomalies_dict)

    # 4. Janelas Operacionais
    pest_risk_level = "BAIXO" 
    pest_risk_txt = "BAIXO"
    if "pest_risk" in df.columns:
        vc = df["pest_risk"].value_counts()
        if vc.get("RISCO_ALTO_FERRUGEM", 0) > 0: 
            pest_risk_txt = "ALTO 🚩"
            pest_risk_level = "ALTO"
        elif vc.get("RISCO_ATENÇÃO", 0) > 0: 
            pest_risk_txt = "ATENÇÃO ⚠️"
            pest_risk_level = "MEDIO"

    pulverizacao_txt = "Sem janelas."
    if "spray_status" in df.columns:
        verde = (df["spray_status"] == "VERDE").sum()
        if verde > 0: pulverizacao_txt = f"{verde} dias VERDE ✅"
        else: pulverizacao_txt = "Restrito (Amarelo/Vermelho) ⛔"

    # Plantio
    plantio_txt = "Inadequado."
    if "planting_status" in df.columns:
        if (df["planting_status"] == "CICLO_EM_ANDAMENTO").any():
            plantio_txt = "Ciclo em andamento (Plantio já realizado) 🌾"
        else:
            ok = (df["planting_status"].isin(["PLANTIO_BOM", "PLANTIO_OK"])).sum()
            if ok > 0: plantio_txt = f"{ok} dias FAVORÁVEIS ✅"
            else:
                atencao = (df["planting_status"] == "PLANTIO_ATENCAO").sum()
                if atencao > 0: plantio_txt = f"{atencao} dias COM ATENÇÃO ⚠️"
                else: plantio_txt = "Restrito/Ruim ⛔"

    # Adubação
    adubacao_txt = "Verificar umidade."
    adubacao_status_code = "NORMAL"
    if "nitrogen_status" in df.columns:
        if (df["nitrogen_status"] == "N_NAO_SE_APLICA").any():
             adubacao_txt = "Não se aplica (Fase/Cultura) 🚫"
             adubacao_status_code = "NAO_APLICA"
        else:
            ok_n = (df["nitrogen_status"] == "N_OK").sum()
            if ok_n > 0: 
                adubacao_txt = f"{ok_n} dias FAVORÁVEIS ✅"
                adubacao_status_code = "FAVORAVEL"
            else:
                atencao_n = (df["nitrogen_status"] == "N_ATENCAO").sum()
                if atencao_n > 0:
                    adubacao_txt = f"{atencao_n} dias COM ATENÇÃO ⚠️"
                    adubacao_status_code = "ATENCAO"
                else:
                    adubacao_txt = "Restrito/Risco ⛔"
                    adubacao_status_code = "RISCO"
    
    if "soja" in cultura.lower() and "FAVORÁVEIS" in adubacao_txt:
        adubacao_txt = "Não se aplica (Fixação Biológica) 🦠"
        adubacao_status_code = "NAO_APLICA"

    saldo_icon = '🔵 Superávit' if saldo_total >= 0 else '🟠 Déficit'
    
    header_report = f"""### 📋 RELATÓRIO TÉCNICO: {cultura.upper()}
📍 **{regiao}** | Solo: {solo}

**1. {zarc_label}:**
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

    # --- DEFINIÇÃO DE DOENÇAS ---
    foco_sanidade = "Doenças fúngicas gerais"
    c_clean = cultura.lower()
    if "soja" in c_clean: foco_sanidade = "Ferrugem Asiática"
    elif "arroz" in c_clean: foco_sanidade = "Brusone"
    elif "trigo" in c_clean: foco_sanidade = "Giberela/Brusone"
    elif "milho" in c_clean: foco_sanidade = "Cercosporiose/Ferrugem Polissora"
    elif "cafe" in c_clean: foco_sanidade = "Ferrugem/Cercosporiose"

    # --- PROMPT BLINDADO ---
    prompt = f"""
    Atue como o Agrônomo Sênior do AgroClima IA.
    
    DADOS DO RELATÓRIO:
    {header_report}
    
    VARIÁVEIS DE CONTROLE:
    - Cultura: {cultura}
    - Risco Fitossanitário Calculado: {pest_risk_level}
    - Status Adubação: {adubacao_status_code}
    - Saldo Hídrico: {saldo_total:.1f} mm
    
    SUA TAREFA:
    Escreva APENAS o item "5. ANÁLISE E RECOMENDAÇÃO AGRONÔMICA (IA)".
    
    REGRAS DE OURO:
    
    1. **SOBRE DOENÇAS ({foco_sanidade}):**
       - OLHE A VARIÁVEL 'Risco Fitossanitário Calculado' ACIMA.
       - Se for "BAIXO": Você é PROIBIDO de dizer que há risco alto de doenças. Diga que "as condições climáticas atuais desfavorecem {foco_sanidade}, mas o monitoramento segue preventivo".
       - APENAS se for "ALTO" ou "MEDIO", você deve alertar perigo iminente.
       - Calor seco MATA fungo. Não associe calor > 36C com doença fúngica.
    
    2. **SOBRE ADUBAÇÃO (NITROGÊNIO):**
       - Se o status for "ATENÇÃO" ou "RISCO" e o Saldo Hídrico for negativo (Déficit): A recomendação é SUSPENDER ou TER EXTREMA CAUTELA.
       - Explique: "Com déficit hídrico de {saldo_total:.1f} mm, a aplicação de N tem baixa eficiência e alto risco de volatilização/fitotoxidez. Aguarde umidade."
    
    3. **SOBRE O CLIMA:**
       - Seja direto. Se tem déficit e calor, o foco é estresse hídrico.
    
    SAÍDA ESPERADA:
    **5. ANÁLISE E RECOMENDAÇÃO AGRONÔMICA (IA):**
    (Texto curto, técnico, sem inventar riscos que a tabela nega)
    """

    resposta_ia = llm_fn(prompt)
    if not resposta_ia: resposta_ia = "⚠️ Erro na IA."

    return header_report + "\n" + resposta_ia