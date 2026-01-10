# agroclima_ia/explain.py

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import pandas as pd


def _fmt_mm(v: float) -> str:
    return f"{v:.1f} mm"


def _format_date_list(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "nenhum"
    try:
        datas = pd.to_datetime(idx)
        return ", ".join(d.strftime("%d/%m") for d in datas)
    except Exception:
        return "datas inválidas"


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
    """
    Gera o bloco '2. MONITORAMENTO' do relatório, usando o dicionário de anomalias.
    Espera um dict no padrão:
      {
        "has_critical": bool,
        "messages": [ "texto 1", "texto 2", ... ],
        "tags": [...],
        "summary": "resumo opcional"
      }
    """
    # Caso não tenha nada ou venha None → mensagem padrão
    if anomalies is None or not isinstance(anomalies, dict):
        return (
            "2. ✅ MONITORAMENTO:\n"
            "• Sem riscos críticos de anomalia climática."
        )

    has_critical = bool(anomalies.get("has_critical", False))
    messages: List[str] = anomalies.get("messages") or []
    summary: str = anomalies.get("summary") or ""

    # Se não tiver mensagens relevantes, cai no texto padrão
    if not messages and not has_critical:
        return (
            "2. ✅ MONITORAMENTO:\n"
            "• Sem riscos críticos de anomalia climática."
        )

    # Cabeçalho depende do nível de risco
    if has_critical:
        header = "2. ⚠ MONITORAMENTO (Riscos Climáticos Detectados):"
    else:
        header = "2. 🔎 MONITORAMENTO (Anomalias observadas):"

    linhas = [
        header,
        "⚠ O algoritmo identificou anomalias climáticas relevantes. Veja os principais pontos:",
    ]
    for msg in messages:
        linhas.append(f"• {msg}")

    if summary:
        linhas.append(f"• Resumo: {summary}")

    return "\n".join(linhas)


def explain_forecast_with_llm(
    df_forecast: pd.DataFrame,
    llm_fn: Optional[Callable[[str], str]] = None,
    cultura: str = "",
    estagio_fenologico: str = "",
    solo: str = "",
    regiao: str = "",
    sistema: str = "",              # tipo de sistema ("sequeiro", "alagado", etc.)
    anomalies: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Gera um texto explicativo do horizonte de 7 dias.

    Se llm_fn for None, usa um template "regra-fixa" (sem chamada de LLM).
    Caso contrário, monta um prompt e delega para a LLM.
    """
    df = df_forecast.copy()

    cultura = (cultura or "").strip().lower()
    estagio_fenologico = (estagio_fenologico or "").strip()
    solo = (solo or "").strip()
    regiao = (regiao or "").strip()
    sistema = (sistema or "").strip().lower()

    # -------------------------------------------------------------------------
    # NORMALIZAÇÃO DE ANOMALIAS (dict novo ou lista antiga)
    # -------------------------------------------------------------------------
    anomalies_dict: Optional[Dict[str, Any]]
    if anomalies is None:
        anomalies_dict = None
    elif isinstance(anomalies, dict):
        anomalies_dict = anomalies
    else:
        # caso algum código antigo ainda passe lista de strings
        try:
            anomalies_dict = {
                "has_critical": True if anomalies else False,
                "messages": list(anomalies),  # type: ignore[arg-type]
                "tags": [],
                "summary": "",
            }
        except Exception:
            anomalies_dict = None

    monitoramento_block = _format_monitoramento_block(anomalies_dict)

    # Versão "plana" só com bullets, para prompt de LLM (sem cabeçalho 2.)
    if anomalies_dict and anomalies_dict.get("messages"):
        monitoramento_plain = "\n".join(
            f"- {m}" for m in (anomalies_dict.get("messages") or [])
        )
    else:
        monitoramento_plain = "Sem riscos críticos de anomalia climática."

    # -------------------------------------------------------------------------
    # MÉTRICAS CLIMÁTICAS BÁSICAS
    # -------------------------------------------------------------------------
    chuva_col = "y_ensemble_mm" if "y_ensemble_mm" in df.columns else "y"
    et0_col = "om_et0_fao_mm" if "om_et0_fao_mm" in df.columns else None
    saldo_col = "water_balance_mm" if "water_balance_mm" in df.columns else None

    chuva_total = float(df[chuva_col].sum()) if chuva_col in df.columns else float("nan")
    et0_total = float(df[et0_col].sum()) if et0_col and et0_col in df.columns else float("nan")
    saldo_total = float(df[saldo_col].sum()) if saldo_col and saldo_col in df.columns else float("nan")

    # Flags de validade (não-NaN)
    chuva_ok = (chuva_total == chuva_total)
    et0_ok = (et0_total == et0_total)
    saldo_ok = (saldo_total == saldo_total)

    # Dias secos = chuva muito baixa
    n_dias_secos = int((df[chuva_col] < 0.5).sum()) if chuva_col in df.columns else 0

    # -------------------------------------------------------------------------
    # RISCO FITOSSANITÁRIO (resumo)
    # -------------------------------------------------------------------------
    pest_risk_txt = "BAIXO"
    if "pest_risk" in df.columns:
        vc = df["pest_risk"].value_counts()
        alto = int(vc.get("RISCO_ALTO_FERRUGEM", 0))
        atencao = int(vc.get("RISCO_ATENÇÃO", 0))

        if alto > 0:
            pest_risk_txt = (
                f"ALTO (Ferrugem da Soja): {alto} dia(s) com condição favorável à doença."
            )
        elif atencao > 0:
            pest_risk_txt = (
                f"ATENÇÃO: {atencao} dia(s) com ambiente parcialmente favorável à doença."
            )
        else:
            pest_risk_txt = (
                "BAIXO: Condições climáticas desfavoráveis para desenvolvimento de doenças."
            )

    # -------------------------------------------------------------------------
    # TEXTO OPERACIONAL (pulverização / plantio)
    # -------------------------------------------------------------------------
    pulverizacao_txt = "Sem condições seguras na semana."
    plantio_txt = "Solo sem condições ideais (muito seco ou encharcado)."

    if "spray_status" in df.columns:
        n_verde = int((df["spray_status"] == "VERDE").sum())
        n_amarelo = int((df["spray_status"] == "AMARELO").sum())
        n_vermelho = int((df["spray_status"] == "VERMELHO").sum())

        if n_verde > 0:
            pulverizacao_txt = (
                f"{n_verde} dia(s) VERDE (janelas preferenciais) e "
                f"{n_amarelo} dia(s) AMARELO. Priorize VERDE; use AMARELO com cautela."
            )
        elif n_amarelo > 0:
            pulverizacao_txt = (
                f"Apenas janelas AMARELO ({n_amarelo} dia(s)). "
                "Planeje com cuidado por risco de vento/chuva."
            )
        else:
            pulverizacao_txt = (
                "Sem janelas seguras (VERDE/AMARELO); evitar pulverizações, se possível."
            )

    if "planting_status" in df.columns:
        n_ok = int((df["planting_status"] == "PLANTIO_OK").sum())
        n_atencao_p = int((df["planting_status"] == "PLANTIO_ATENCAO").sum())
        n_ruim = int((df["planting_status"] == "PLANTIO_RUIM").sum())

        if n_ok > 0:
            plantio_txt = (
                f"{n_ok} dia(s) com PLANTIO_OK e {n_atencao_p} dia(s) de PLANTIO_ATENCAO. "
                "Priorize os dias PLANTIO_OK para maior segurança."
            )
        elif n_atencao_p > 0:
            plantio_txt = (
                f"Apenas janelas de PLANTIO_ATENCAO ({n_atencao_p} dia(s)). "
                "Exige avaliação fina de umidade de solo e logística."
            )
        else:
            plantio_txt = (
                "Predomínio de PLANTIO_RUIM; o plantio deve ser evitado, salvo necessidade extrema."
            )

    # -------------------------------------------------------------------------
    # TEXTO DE ADUBAÇÃO / NUTRIÇÃO (principalmente N em cobertura)
    # -------------------------------------------------------------------------
    adubacao_txt = (
        "O modelo avalia janelas climáticas para adubação nitrogenada em cobertura, "
        "considerando chuva, demanda atmosférica (ET0) e risco de perdas."
    )

    if "nitrogen_status" in df.columns:
        n_ok_n = int((df["nitrogen_status"] == "N_OK").sum())
        n_atencao_n = int((df["nitrogen_status"] == "N_ATENCAO").sum())
        n_risco_n = int((df["nitrogen_status"] == "N_RISCO").sum())

        linhas_n: List[str] = []
        if n_ok_n > 0:
            linhas_n.append(
                f"- **{n_ok_n} dia(s) N_OK**: boas janelas climáticas para aplicação de N, "
                "com maior probabilidade de incorporação eficiente no solo."
            )
        if n_atencao_n > 0:
            linhas_n.append(
                f"- **{n_atencao_n} dia(s) N_ATENCAO**: janelas intermediárias; "
                "nesses dias, prefira doses menores, fracionamento ou fontes menos sujeitas a perdas."
            )
        if n_risco_n > 0:
            linhas_n.append(
                f"- **{n_risco_n} dia(s) N_RISCO**: alto risco de perda de N (volatilização ou lixiviação). "
                "Evite adubações, principalmente com ureia superficial ou solos mais arenosos."
            )

        # Ajuste em função do balanço hídrico semanal
        if saldo_ok and saldo_total <= -20:
            linhas_n.append(
                "- O saldo hídrico **muito negativo** indica solo mais seco; aumenta o risco de volatilização "
                "quando se aplica ureia sem chuva de incorporação nas horas seguintes."
            )
        elif saldo_ok and -20 < saldo_total <= -10:
            linhas_n.append(
                "- O saldo hídrico **moderadamente negativo** sugere atenção: se o solo estiver muito seco, "
                "a eficiência da adubação de cobertura cai, especialmente em solos rasos ou arenosos."
            )
        elif saldo_ok and saldo_total >= 10:
            linhas_n.append(
                "- O saldo hídrico positivo indica ambiente mais úmido; atenção ao risco de lixiviação e "
                "perdas em profundidade em solos mais arenosos, caso ocorram chuvas fortes logo após a aplicação."
            )

        if linhas_n:
            adubacao_txt = "\n".join(linhas_n)
        else:
            adubacao_txt = (
                "Não foram identificadas janelas específicas de N_OK / N_ATENCAO / N_RISCO na semana. "
                "Use a combinação de chuva prevista e ET0 para escolher dias com menor risco de perdas."
            )
    else:
        adubacao_txt += (
            "  Nesta execução, a coluna de classificação de nitrogênio (nitrogen_status) não está disponível; "
            "use a informação de chuva e ET0 para definir as melhores janelas de cobertura."
        )

    # -------------------------------------------------------------------------
    # TEXTO DE IRRIGAÇÃO / MANEJO HÍDRICO
    # -------------------------------------------------------------------------
    irrigacao_txt = ""

    if not (saldo_ok and et0_ok and chuva_ok):
        irrigacao_txt = (
            "Não foi possível calcular um balanço hídrico confiável para recomendações "
            "mais detalhadas de irrigação. Use a previsão diária de chuva e ET0 como apoio."
        )
    else:
        # Indicadores auxiliares (reservado para evoluções)
        razao_chuva_et0 = chuva_total / et0_total if et0_total > 0 else 0.0  # noqa: F841

        sistema_irrigado = ("alagado" in sistema) or ("irrig" in sistema)

        if sistema_irrigado:
            # Texto específico por cultura (evita “vazar” cultura errada)
            if "arroz" in cultura:
                # 🟦 Arroz irrigado/alagado (lâmina)
                irrigacao_txt = (
                    "No arroz irrigado/alagado, o déficit atmosférico é compensado pela lâmina de água. "
                    "Com esse saldo hídrico, o foco deve ser manter uma lâmina estável, evitando tanto "
                    "exposição do solo quanto excesso de profundidade que pode aumentar risco de acamamento.\n\n"
                )
                if saldo_total < -10:
                    irrigacao_txt += (
                        "- O saldo hídrico moderadamente negativo indica maior demanda evaporativa. "
                        "Ajuste o turno de irrigação (intervalo entre lâminas) para não deixar a lâmina "
                        "abaixo do nível de segurança nos talhões mais sensíveis.\n"
                    )
                elif saldo_total > 10:
                    irrigacao_txt += (
                        "- O saldo positivo sugere aporte de água acima da demanda. "
                        "Monitore sinais de excesso e, se necessário, reduza brevemente a lâmina para favorecer "
                        "aeração do sistema radicular, conforme recomendação técnica local.\n"
                    )
                else:
                    irrigacao_txt += (
                        "- O balanço hídrico está próximo do neutro. Mantenha o manejo atual, "
                        "ajustando a lâmina em função de ventos e picos de temperatura.\n"
                    )
            else:
                # 🟩 Irrigado genérico (sem citar cultura)
                irrigacao_txt = (
                    "Em sistemas irrigados, o manejo hídrico deve considerar a demanda atmosférica (ET0), "
                    "a ocorrência de chuvas e o objetivo de manter umidade adequada na zona radicular, evitando "
                    "tanto déficit quanto excesso prolongado.\n\n"
                )
                if saldo_total <= -20:
                    irrigacao_txt += (
                        "- O saldo hídrico está **muito negativo**, indicando demanda elevada. "
                        "Se houver irrigação disponível, considere reforço de lâmina/turno para reduzir o déficit.\n"
                    )
                elif -20 < saldo_total <= -10:
                    irrigacao_txt += (
                        "- O saldo hídrico está **moderadamente negativo**. "
                        "Uma irrigação complementar moderada pode estabilizar o ambiente hídrico.\n"
                    )
                elif saldo_total >= 10:
                    irrigacao_txt += (
                        "- O saldo hídrico positivo indica aporte acima da demanda. "
                        "Monitore sinais de excesso e ajuste a irrigação para evitar saturação persistente.\n"
                    )
                else:
                    irrigacao_txt += (
                        "- O balanço hídrico está próximo do neutro. Mantenha o manejo atual, "
                        "ajustando conforme variações de vento e temperatura.\n"
                    )

        else:
            # 🌾 Sistemas de sequeiro (ou irrigação suplementar)
            irrigacao_txt = (
                "Para sistemas de sequeiro (ou irrigação apenas suplementar), o balanço hídrico da semana "
                "é um indicador importante de risco de estresse hídrico e da necessidade de complementar "
                "com lâminas de irrigação, se houver infraestrutura disponível.\n\n"
            )

            if saldo_total <= -20:
                irrigacao_txt += (
                    "- O saldo hídrico está **muito negativo**, sugerindo estresse hídrico relevante. "
                    "Se houver irrigação, priorize talhões em estágios mais sensíveis "
                    f"(como {estagio_fenologico or 'fases reprodutivas'}) e planeje lâminas que ao menos "
                    "reduzam o déficit acumulado.\n"
                )
            elif -20 < saldo_total <= -10:
                irrigacao_txt += (
                    "- O saldo hídrico está **moderadamente negativo**. "
                    "Em áreas com irrigação, uma lâmina complementar moderada pode evitar queda de rendimento, "
                    "especialmente em solos mais arenosos ou rasos.\n"
                )
            elif -10 < saldo_total < 5:
                irrigacao_txt += (
                    "- O balanço hídrico está levemente negativo ou próximo do neutro. "
                    "Mantenha o solo coberto (palhada, cobertura vegetal) e evite operações que exponham o solo "
                    "ao sol e vento, para preservar umidade.\n"
                )
            else:  # saldo >= 5
                irrigacao_txt += (
                    "- O saldo hídrico está levemente positivo, com boa reposição de água no solo. "
                    "Use essa janela para operações que exigem melhor umidade (plantio, adubações de base), "
                    "observando sempre a capacidade de campo do solo para evitar encharcamento.\n"
                )

            if n_dias_secos >= 4:
                irrigacao_txt += (
                    f"- Foram identificados **{n_dias_secos} dias secos** na janela, o que reforça "
                    "a importância de monitorar a umidade de solo (tensiômetros, sondagens) e antecipar "
                    "irrigação suplementar onde possível.\n"
                )

    # -------------------------------------------------------------------------
    # CONCLUSÃO GERAL (OPÇÃO 3 - contextual)
    # -------------------------------------------------------------------------
    # Classificação textual do cenário hídrico (para a conclusão)
    if saldo_ok:
        if saldo_total > 10:
            concl_saldo_lbl = "SUPERÁVIT"
        elif saldo_total < -10:
            concl_saldo_lbl = "DÉFICIT"
        else:
            concl_saldo_lbl = "EQUILÍBRIO"
    else:
        concl_saldo_lbl = "COND. INDEFINIDAS"

    concl_sistema_lbl = "SEQUEIRO"
    if ("alagado" in sistema) or ("irrig" in sistema):
        concl_sistema_lbl = "ALAGADO/IRRIGADO"

    conclusao_txt = (
        f"• O cenário climático da semana apresenta **{concl_saldo_lbl}** hídrico, "
        f"influenciando diretamente o manejo da cultura no sistema **{concl_sistema_lbl}**.\n"
        "• As operações agrícolas devem ser concentradas nos dias com melhor classificação operacional, "
        "evitando intervenções em períodos de maior risco climático ou fitossanitário.\n"
        "• O uso integrado da previsão de chuva, ET0 e balanço hídrico permite reduzir riscos operacionais "
        "e aumentar a eficiência do manejo ao longo da semana."
    )

    # -------------------------------------------------------------------------
    # TEXTO FINAL (TEMPLATE FIXO OU VIA LLM)
    # -------------------------------------------------------------------------
    # Linha do saldo hídrico no cabeçalho (evita mostrar 'nan mm')
    if saldo_ok:
        saldo_header = (
            f"   • Saldo Hídrico:   **{saldo_total:.1f} mm** "
            f"{'(🔵 Superávit)' if saldo_total >= 0 else '(🟠 Déficit)'}"
        )
    else:
        saldo_header = "   • Saldo Hídrico:   N/D"

    if llm_fn is not None:
        base_prompt = f"""
Você é um engenheiro agrônomo. Abaixo há um resumo do clima previsto e do manejo:

Região: {regiao}
Cultura: {cultura}
Estágio: {estagio_fenologico}
Solo: {solo}
Sistema: {sistema}

Chuva total (7 dias): {chuva_total:.1f} mm
ET0 total (7 dias): {et0_total:.1f} mm
Saldo hídrico (7 dias): {saldo_total:.1f} mm
Dias secos (<0.5 mm): {n_dias_secos}

Monitoramento/anomalias:
{monitoramento_plain}

Resumo de risco fitossanitário:
{pest_risk_txt}

Situação operacional:
- Pulverização: {pulverizacao_txt}
- Plantio: {plantio_txt}

Janelas para adubação nitrogenada:
{adubacao_txt}

Comentários sobre irrigação e manejo hídrico:
{irrigacao_txt}

Com base nessas informações, redija um parecer técnico curto (até 15 linhas)
para o produtor, com linguagem clara e objetiva.
"""
        return llm_fn(base_prompt)

    texto = f"""### 📋 RELATÓRIO: {cultura.upper() if cultura else 'CULTURA'} ({estagio_fenologico or 'estágio não informado'})
📍 **{regiao or 'Região não informada'}** | Solo: {solo or 'N/D'} | Sistema: {sistema or 'N/D'}

**1. CLIMA (7 DIAS):**
   • Chuva Acumulada: **{chuva_total:.1f} mm**
   • Demanda (ET0):   {f"{et0_total:.1f} mm" if et0_ok else "N/D"}
{saldo_header}

{monitoramento_block}

**3. 🦠 RISCO FITOSSANITÁRIO:**
   • {pest_risk_txt}

**4. OPERACIONAL:**
   • 🚜 Pulverização: {pulverizacao_txt}
   • 🌱 Plantio: {plantio_txt}

**5. CONCLUSÃO GERAL:**
{conclusao_txt}

**6. 💧 IRRIGAÇÃO / MANEJO HÍDRICO:**
{irrigacao_txt}

**7. 🌿 ADUBAÇÃO / NUTRIÇÃO (N em cobertura):**
{adubacao_txt}
"""
    return texto



















