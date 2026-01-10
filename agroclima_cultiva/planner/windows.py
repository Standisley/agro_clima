# agroclima_cultiva/planner/windows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from ..schemas.inputs import FarmInput
from .catalog import CropSpec, load_catalog_from_json


@dataclass(frozen=True)
class CandidateWindow:
    cultura: str
    grupo: str
    janela: str
    ciclo_dias: Tuple[int, int]
    demanda_agua: str
    complexidade_manejo: str
    aptidao_area: str
    racional: str
    riscos: List[str]
    observacoes: List[str]
    score_total: float  # <- NOVO: score final após regras climáticas


def _area_bucket(area_m2: float) -> str:
    if area_m2 <= 5_000:
        return "muito_pequena"
    if area_m2 <= 10_000:
        return "pequena"
    if area_m2 <= 20_000:
        return "media"
    return "grande"


def _aptidao_por_area(grupo: str, area_bucket: str) -> str:
    g = (grupo or "").strip().lower()

    if g in {"folhosas", "hortaliça", "hortalica", "hortaliça-fruto", "hortalica-fruto", "raízes", "raizes", "medicinal"}:
        return "alta" if area_bucket in {"muito_pequena", "pequena", "media"} else "media"

    if g in {"frutífera", "frutifera", "perene"}:
        return "media" if area_bucket in {"muito_pequena"} else "alta"

    if g in {"grãos", "graos"}:
        return "baixa" if area_bucket in {"muito_pequena", "pequena"} else "media"

    return "media"


def _objective_weights(objetivo: str) -> Dict[str, float]:
    objetivo = (objetivo or "").strip().lower()
    if objetivo == "renda_rapida":
        return {"ciclo_curto": 0.45, "baixo_risco": 0.15, "manejo_simples": 0.20, "aptidao": 0.20}
    if objetivo == "baixo_risco":
        return {"ciclo_curto": 0.15, "baixo_risco": 0.45, "manejo_simples": 0.20, "aptidao": 0.20}
    if objetivo == "seguranca_alimentar":
        return {"ciclo_curto": 0.20, "baixo_risco": 0.30, "manejo_simples": 0.25, "aptidao": 0.25}
    if objetivo == "diversificacao":
        return {"ciclo_curto": 0.20, "baixo_risco": 0.25, "manejo_simples": 0.15, "aptidao": 0.40}
    return {"ciclo_curto": 0.20, "baixo_risco": 0.35, "manejo_simples": 0.20, "aptidao": 0.25}


def _score_candidate_spec(c: CropSpec, area_bucket: str, weights: Dict[str, float]) -> float:
    ciclo_min, ciclo_max = c.ciclo_dias
    ciclo_med = (float(ciclo_min) + float(ciclo_max)) / 2.0
    ciclo_curto = 1.0 if ciclo_med <= 120 else (0.6 if ciclo_med <= 240 else 0.2)

    demanda_agua = (c.demanda_hidrica or "media").strip().lower()
    risco_hidrico = 1.0 if demanda_agua == "baixa" else (0.6 if demanda_agua == "media" else 0.3)

    complexidade = (c.complexidade or "media").strip().lower()
    manejo_simples = 1.0 if complexidade == "baixa" else (0.6 if complexidade == "media" else 0.3)

    apt = _aptidao_por_area(c.grupo, area_bucket)
    apt_score = 1.0 if apt == "alta" else (0.6 if apt == "media" else 0.2)

    score = (
        weights["ciclo_curto"] * ciclo_curto
        + weights["baixo_risco"] * risco_hidrico
        + weights["manejo_simples"] * manejo_simples
        + weights["aptidao"] * apt_score
    )
    return float(score)


def _janela_operacional_por_grupo(grupo: str) -> str:
    g = (grupo or "").strip().lower()
    if g in {"folhosas", "hortaliça", "hortalica", "raízes", "raizes", "medicinal"}:
        return "Ano todo (ajustar por calor/chuva)"
    if g in {"hortaliça-fruto", "hortalica-fruto"}:
        return "Preferir períodos menos chuvosos (varia por região)"
    if g in {"frutífera", "frutifera", "perene"}:
        return "Estabelecimento preferencial no início das chuvas"
    if g in {"grãos", "graos"}:
        return "Safra verão (varia por região) | Safrinha (se aplicável)"
    return "Ano todo"


def _riscos_base_por_tag(tags: Tuple[str, ...]) -> List[str]:
    t = {str(x).strip().lower() for x in (tags or ())}
    riscos: List[str] = []
    if "ciclo_curto" in t:
        riscos.append("sensível a falhas operacionais (plantio/colheita) por janelas curtas")
    if "valor_agregado" in t:
        riscos.append("depende de canal de venda (feira, cestas, processamento/embalagem)")
    if "secagem" in t:
        riscos.append("qualidade depende de pós-colheita (secagem/armazenamento)")
    return riscos


# -------------------------------
# ETAPA 2: penalizações climáticas
# -------------------------------

def _climate_penalties(
    c: CropSpec,
    climate_7d: Optional[Any],  # HorizonFeatures (dataclass) do metrics.py
) -> Tuple[float, List[str]]:
    """
    Retorna (penalty, reasons). penalty é subtraído do score base.
    """
    if climate_7d is None:
        return 0.0, []

    reasons: List[str] = []
    penalty = 0.0

    dias_secos = int(getattr(climate_7d, "dias_secos", 0))
    balanco = float(getattr(climate_7d, "balanco_hidrico_mm", 0.0))
    chuva_total = float(getattr(climate_7d, "chuva_total_mm", 0.0))
    rh_media = float(getattr(climate_7d, "rh_media_pct", 0.0))

    demanda = (c.demanda_hidrica or "media").strip().lower()
    tags = {str(t).strip().lower() for t in (c.tags or ())}

    # Regra A: penalizar alta demanda hídrica quando seco + balanço muito negativo
    # (proxy de estresse hídrico / déficit atmosférico)
    if dias_secos >= 4 and balanco <= -20:
        if demanda == "alta":
            penalty += 0.25
            reasons.append("penalizado: demanda hídrica alta sob dias secos e balanço hídrico muito negativo (7d)")
        elif demanda == "media":
            penalty += 0.12
            reasons.append("penalizado: demanda hídrica média sob dias secos e balanço hídrico muito negativo (7d)")

    # Regra B: penalizar sensibilidade a doenças com chuva alta + umidade alta (proxy)
    # Ajuste simples: ambiente úmido persistente tende a elevar risco fúngico/viroses
    humid_risky = (chuva_total >= 60.0) and (rh_media >= 85.0)
    if humid_risky:
        disease_sensitive = any(x in tags for x in {"doencas_fungicas", "viroses", "mosca_das_frutas", "tripes"})
        if disease_sensitive:
            penalty += 0.18
            reasons.append("penalizado: proxy de pressão de doenças (chuva alta + umidade alta) e cultura/tag sensível")

        # regra leve por risco declarado alto
        if (c.risco or "").strip().lower() == "alto":
            penalty += 0.08
            reasons.append("penalizado: risco base alto em cenário úmido (proxy)")

    return penalty, reasons


def build_candidate_windows(
    inp: FarmInput,
    top_k: int = 8,
    catalog_path: Optional[str] = None,
    climate_features: Optional[Dict[str, Any]] = None,  # <- NOVO
) -> List[CandidateWindow]:
    """
    Etapa 2:
      - score base (objetivo + área)
      - aplica penalizações climáticas (rule-based mínimo) usando features 7d
      - retorna lista ranqueada por score_total
    """
    inp.validate()

    area_bucket = _area_bucket(float(inp.area_m2))
    weights = _objective_weights(str(inp.objetivo))

    # carregar catálogo
    if catalog_path:
        from pathlib import Path
        catalog = load_catalog_from_json(path=Path(catalog_path))
    else:
        catalog = load_catalog_from_json()

    # filtro por área
    candidates: List[CropSpec] = [c for c in catalog if float(inp.area_m2) >= float(c.area_min_m2)]

    # extrair clima 7d
    climate_7d = None
    if climate_features and isinstance(climate_features, dict):
        climate_7d = climate_features.get("7d")

    scored: List[Tuple[float, float, CropSpec, List[str]]] = []
    for c in candidates:
        base = _score_candidate_spec(c, area_bucket, weights)
        penalty, reasons = _climate_penalties(c, climate_7d)
        total = max(0.0, base - penalty)
        scored.append((total, base, c, reasons))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[CandidateWindow] = []
    for total_score, base_score, c, climate_reasons in scored[: max(1, int(top_k))]:
        apt = _aptidao_por_area(c.grupo, area_bucket)
        janela = _janela_operacional_por_grupo(c.grupo)

        riscos: List[str] = []
        riscos.extend(_riscos_base_por_tag(c.tags))
        if apt == "baixa":
            riscos.append("baixa aptidão econômica para esta escala de área (MVP)")

        observacoes: List[str] = []
        if (c.demanda_hidrica or "").strip().lower() == "alta":
            observacoes.append("Cultura com maior exigência hídrica: avaliar irrigação, mulching e manejo conservacionista.")
        if c.observacao:
            observacoes.append(c.observacao)

        # registrar motivos climáticos (transparência)
        if climate_reasons:
            observacoes.append("Ajuste climático (7d): " + " | ".join(climate_reasons))

        racional = (
            f"Score base={base_score:.2f}; score final={total_score:.2f} "
            f"(objetivo='{inp.objetivo}', porte='{area_bucket}', aptidão='{apt}'). "
            "Catálogo: JSON (curadoria inicial; não-ZARC)."
        )

        out.append(
            CandidateWindow(
                cultura=c.nome,
                grupo=c.grupo,
                janela=janela,
                ciclo_dias=(int(c.ciclo_dias[0]), int(c.ciclo_dias[1])),
                demanda_agua=str(c.demanda_hidrica),
                complexidade_manejo=str(c.complexidade),
                aptidao_area=apt,
                racional=racional,
                riscos=riscos,
                observacoes=observacoes,
                score_total=float(total_score),
            )
        )

    return out




