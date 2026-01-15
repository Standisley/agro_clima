# agroclima_ia/explain.py

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import pandas as pd

# Tenta importar a biblioteca de forma segura
try:
    import google.generativeai as genai
    HAS_GOOGLE_LIB = True
except ImportError:
    HAS_GOOGLE_LIB = False

# =============================================================================
# Funções Auxiliares de Formatação (MANTIDAS)
# =============================================================================

def _fmt_mm(v: float) -> str:
    return f"{v:.1f} mm"

def _format_date_list(idx: pd.Index) -> str:
    if len(idx) == 0: return "nenhum"
    try:
        datas = pd.to_datetime(idx)
        return ", ".join(d.strftime("%d/%m") for d in datas)
    except Exception: return "datas inválidas"

def _safe_counts_and_days(df: pd.DataFrame, colname: str, categorias: List[str]):
    out = {}
    if colname not in df.columns:
        for cat in categorias:
            out[cat] = {"count": 0, "days": pd.Index([])}
        return out
    for cat in categorias:
        mask = df[colname] == cat
        out[cat] = {"count": int(mask.sum()), "days": df.index[mask]}
    return out

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
    if summary: linhas.append(f"• Resumo: {summary}")
    return "\n".join(linhas)

# =============================================================================
# NOVA FUNÇÃO: Conexão "Auto-Discovery" (Busca Automática)
# =============================================================================
def call_gemini_llm(prompt_text: str, api_key: str) -> str:
    """
    Envia o prompt para o Google Gemini.
    Usa list_models() para encontrar um modelo válido automaticamente.
    """
    if not HAS_GOOGLE_LIB:
        return "⚠️ Erro: Biblioteca 'google-generativeai' não instalada."
    
    if not api_key:
        return "⚠️ Erro: API Key não fornecida."

    try:
        genai.configure(api_key=api_key)
        config = genai.types.GenerationConfig(temperature=0.4)
        
        # 1. Tenta o nome padrão mais comum primeiro (rápido)
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt_text, generation_config=config)
            return response.text
        except Exception:
            pass # Se falhar, vamos para o plano B

        # 2. PLANO B: Busca automática
        valid_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'gemini' in m.name:
                        valid_models.append(m.name)
        except Exception as e_list:
            return f"⚠️ Erro ao listar modelos: {e_list}"

        if not valid_models:
            return "⚠️ Erro: Nenhum modelo 'Gemini' encontrado na sua conta API."

        # Ordena para preferir 'flash' ou 'pro'
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
        
        return f"⚠️ Falha na IA. Erro final: {last_error}"

    except Exception as e:
        return f"⚠️ Erro Geral LLM: {e}"

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
    
    # 1. Normalização de anomalias
    anomalies_dict = anomalies if isinstance(anomalies, dict) else None
    if anomalies and not isinstance(anomalies, dict): 
         anomalies_dict = {"has_critical": True, "messages": list(anomalies)}

    # Variável para o Prompt (Lista simples)
    if anomalies_dict and anomalies_dict.get("messages"):
        monitoramento_plain = "\n".join(f"- {m}" for m in anomalies_dict["messages"])
    else:
        monitoramento_plain = "Sem riscos críticos de anomalia climática."

    # 2. Métricas Climáticas
    chuva_col = "y_ensemble_mm" if "y_ensemble_mm" in df.columns else "y"
    et0_col = "om_et0_fao_mm"
    saldo_col = "water_balance_mm"

    chuva_total = float(df[chuva_col].sum()) if chuva_col in df.columns else 0.0
    et0_total = float(df[et0_col].sum()) if et0_col in df.columns else 0.0
    saldo_total = float(df[saldo_col].sum()) if saldo_col in df.columns else 0.0
    n_dias_secos = int((df[chuva_col] < 0.5).sum()) if chuva_col in df.columns else 0
    saldo_ok = (saldo_col in df.columns)

    # 3. Risco Fitossanitário
    pest_risk_txt = "BAIXO"
    if "pest_risk" in df.columns:
        vc = df["pest_risk"].value_counts()
        if vc.get("RISCO_ALTO_FERRUGEM", 0) > 0: pest_risk_txt = "ALTO (Ferrugem)"
        elif vc.get("RISCO_ATENÇÃO", 0) > 0: pest_risk_txt = "ATENÇÃO"

    # 4. Operacional
    pulverizacao_txt = "Sem janelas seguras."
    if "spray_status" in df.columns:
        verde = (df["spray_status"] == "VERDE").sum()
        amarelo = (df["spray_status"] == "AMARELO").sum()
        if verde > 0: pulverizacao_txt = f"{verde} dias VERDE / {amarelo} dias AMARELO."
        elif amarelo > 0: pulverizacao_txt = f"{amarelo} dias AMARELO."

    plantio_txt = "Inadequado."
    if "planting_status" in df.columns:
        ok = (df["planting_status"] == "PLANTIO_OK").sum()
        if ok > 0: plantio_txt = f"{ok} dias FAVORÁVEIS."
        elif (df["planting_status"] == "PLANTIO_ATENCAO").sum() > 0: plantio_txt = "Apenas dias de ATENÇÃO."

    # 5. Adubação
    adubacao_txt = "Verificar umidade."
    if "nitrogen_status" in df.columns:
        ok_n = (df["nitrogen_status"] == "N_OK").sum()
        if ok_n > 0: adubacao_txt = f"{ok_n} dias FAVORÁVEIS."
        elif (df["nitrogen_status"] == "N_RISCO").sum() > 0: adubacao_txt = "Risco de perda elevado."

    # 6. Irrigação
    irrigacao_txt = "Monitorar."
    if saldo_total < -20: irrigacao_txt = "Déficit severo."
    elif saldo_total > 10: irrigacao_txt = "Solo úmido."

    # -------------------------------------------------------------------------
    # PROMPT PARA LLM (ESTRUTURADO)
    # -------------------------------------------------------------------------
    if llm_fn is not None:
        prompt = f"""
        Você é um Engenheiro Agrônomo Sênior (AgroClima IA).
        Sua tarefa: Gerar um relatório técnico em formato Markdown rigoroso.

        DADOS DE ENTRADA:
        - Fazenda: {cultura.upper()} | {regiao}
        - Estágio: {estagio_fenologico}
        - Solo: {solo} | Sistema: {sistema}
        - Clima (7 dias): Chuva {chuva_total:.1f}mm | ET0 {et0_total:.1f}mm | Saldo {saldo_total:.1f}mm | Dias Secos {n_dias_secos}
        - Monitoramento/Anomalias: {monitoramento_plain}
        - Risco Doenças: {pest_risk_txt}
        - Pulverização: {pulverizacao_txt}
        - Plantio: {plantio_txt}
        - Adubação N: {adubacao_txt}
        - Irrigação: {irrigacao_txt}

        FORMATO DE SAÍDA (Obrigatório seguir este layout):

        ### 📋 RELATÓRIO TÉCNICO: {cultura.upper()}
        📍 **{regiao}** | Estágio: {estagio_fenologico} | Sistema: {sistema}

        **1. CLIMA (Acumulado 7 dias):**
        • Chuva: **{chuva_total:.1f} mm**
        • ET0 (Demanda): {et0_total:.1f} mm
        • Saldo Hídrico: **{saldo_total:.1f} mm** ({'🔵 Superávit' if saldo_total >= 0 else '🟠 Déficit'})
        • Dias Secos: {n_dias_secos}

        **2. MONITORAMENTO & RISCOS:**
        • Anomalias: {monitoramento_plain}
        • Risco Fitossanitário: {pest_risk_txt}

        **3. JANELAS OPERACIONAIS:**
        • 🚜 Pulverização: {pulverizacao_txt}
        • 🌱 Plantio: {plantio_txt}
        • 🌿 Adubação (N): {adubacao_txt}

        **4. ANÁLISE E RECOMENDAÇÃO AGRONÔMICA (IA):**
        (Escreva aqui sua análise interpretativa dos dados acima, focando na tomada de decisão para {cultura} em {estagio_fenologico}. Seja técnico e direto.)
        """
        return llm_fn(prompt)

    # -------------------------------------------------------------------------
    # TEMPLATE OFFLINE (Fallback)
    # -------------------------------------------------------------------------
    # Só usado se não tiver chave de API
    monitoramento_block = _format_monitoramento_block(anomalies_dict)
    
    return f"""### 📋 RELATÓRIO: {cultura.upper()}
📍 **{regiao}** | Estágio: {estagio_fenologico}

**CLIMA:** Chuva: {chuva_total:.1f}mm | Saldo: {saldo_total:.1f}mm
{monitoramento_block}
**RISCOS:** {pest_risk_txt}

**MANEJO:**
🚜 Pulverização: {pulverizacao_txt}
🌱 Plantio: {plantio_txt}
🌿 Adubação: {adubacao_txt}
💧 Irrigação: {irrigacao_txt}

*(Modo Offline - Sem análise de IA)*
"""