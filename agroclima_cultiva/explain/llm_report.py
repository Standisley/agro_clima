from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..schemas.inputs import FarmInput
from ..planner.windows import CandidateWindow


def _to_plain(obj: Any) -> Any:
    """
    Converte objetos complexos (dataclasses, pandas, etc.)
    para estruturas JSON-safe.
    """
    if obj is None:
        return None

    # Dataclasses (ex.: HorizonFeatures)
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}

    # Dict
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}

    # List / Tuple
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    # Tipos primitivos
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback seguro
    return str(obj)


def build_evidence_pack(
    inp: FarmInput,
    climate_features: Dict[str, Any],
    top_windows: List[CandidateWindow],
    climate_df_7d: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Pacote único de evidências que a LLM deve usar.
    Importante: aqui entra apenas dado calculado/observado (determinístico).
    """
    def slim_climate(df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        if df is None or df.empty:
            return None

        d = df.copy().head(7).reset_index(drop=True)

        # mantém apenas colunas mais “comunicáveis”
        cols = [c for c in [
            "ds",
            "om_prcp_mm",
            "om_tmin_c",
            "om_tmax_c",
            "om_wind_kmh",
            "om_rh_max",
        ] if c in d.columns]

        d = d[cols]
        d["ds"] = pd.to_datetime(d["ds"]).dt.strftime("%Y-%m-%d")

        # arredondamento leve para texto
        for c in ["om_prcp_mm", "om_tmin_c", "om_tmax_c", "om_wind_kmh", "om_rh_max"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").round(1)

        return d.to_dict(orient="records")

    loc = {
        "lat": float(inp.lat),
        "lon": float(inp.lon),
        "municipio": inp.municipio,
        "uf": inp.uf,
        "area_m2": float(inp.area_m2),
        "objetivo": str(inp.objetivo),
        "perfil_produtor": inp.perfil_produtor,
    }

    ranking: List[Dict[str, Any]] = []
    for w in top_windows[:3]:
        evid = getattr(w, "evidencias", {}) or {}
        ranking.append({
            "cultura": w.cultura,
            "grupo": w.grupo,
            "demanda_agua": w.demanda_agua,
            "complexidade": w.complexidade_manejo,
            "aptidao_area": w.aptidao_area,
            "score_total": float(getattr(w, "score_total", 0.0) or 0.0),
            "evidencias": _to_plain(evid),
        })

    pack = {
        "localizacao": loc,
        "clima": {
            "tabela_7d": slim_climate(climate_df_7d),
            "features": _to_plain(climate_features),
        },
        "ranking_top3": ranking,
        "observacao_transparencia": (
            "Este relatório deve ser baseado estritamente nas evidências fornecidas "
            "e não deve afirmar integrações não implementadas (ex.: ZARC)."
        ),
    }
    return pack


def build_llm_prompt(evidence_pack: Dict[str, Any]) -> str:
    """
    Prompt com regras explícitas anti-alucinação:
    - usar apenas as evidências
    - não inventar dados/culturas/município/UF
    - não citar ZARC como se estivesse aplicado
    """
    evidence_json = json.dumps(evidence_pack, ensure_ascii=False, indent=2)

    return f"""
Você é um agrônomo e redator técnico. Gere um relatório curto e direto para o produtor.

REGRAS OBRIGATÓRIAS (ANTI-ALUCINAÇÃO):
1) Use APENAS os fatos e números do JSON de EVIDÊNCIAS.
2) Não invente dados (município/UF, valores, prazos) e não inclua culturas fora do ranking_top3.
3) Não cite ZARC como se estivesse aplicado. Se precisar mencionar, diga que é "próxima etapa" (apenas se estiver nas evidências).
4) Se algum campo estiver ausente, escreva "não disponível nas evidências".
5) Justifique cada recomendação com 1–2 evidências objetivas (ex.: chuva_7d, dias_secos_14d, rh, tmax_p95, penalidades registradas).
6) Se o balanço hídrico for positivo, NÃO mencionar déficit hídrico.


FORMATO (simples e limpo, sem texto longo):
- Linha 1: Localização e área (usar município/UF se existir, senão lat/lon).
- Bloco "Clima (7d e 14d)" com 2 linhas (somente resumo das features).
- Bloco "Top 3 recomendadas" com 3 itens numerados, cada item com:
  cultura — score — 1 justificativa objetiva (clima/penalidades)
- Bloco final "Atenções" (2–4 bullets) baseado em penalidades/evidências.

EVIDÊNCIAS (JSON):
{evidence_json}
""".strip()


def render_report_llm_only(
    evidence_pack: Dict[str, Any],
    llm_fn: Callable[[str], str],
) -> str:
    prompt = build_llm_prompt(evidence_pack)
    out = (llm_fn(prompt) or "").strip()
    if not out:
        raise RuntimeError("LLM retornou texto vazio.")
    return out
