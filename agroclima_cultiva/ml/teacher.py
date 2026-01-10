# agroclima_cultiva/ml/teacher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..planner.catalog import CropSpec


@dataclass(frozen=True)
class TeacherResult:
    score: float           # 0..1
    flags: Tuple[str, ...] # explicações curtas


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def teacher_score_for_crop(
    crop: CropSpec,
    feats: Dict[str, float],
) -> TeacherResult:
    """
    Heurística mínima e transparente:
    - Penaliza demanda hídrica alta quando dias_secos alto e balanco muito negativo
    - Penaliza risco sanitário quando chuva_total alta + RH alta (proxy)
    - Penaliza calor quando tmax_p95 alta (proxy de estresse)
    Retorna score 0..1 e flags (explicações).
    """
    score = 1.0
    flags = []

    chuva = float(feats.get("chuva_total", 0.0))
    et0 = float(feats.get("et0_total", 0.0))
    bal = float(feats.get("balanco_hidrico", chuva - et0))
    secos = int(feats.get("dias_secos", 0))
    rh = float(feats.get("rh_media", 0.0))
    tmax_p95 = float(feats.get("tmax_p95", 0.0))

    demanda = (crop.demanda_hidrica or "media").lower()

    # 1) Risco hídrico: seco + déficit
    if demanda == "alta":
        if secos >= 5 and bal <= -20:
            score -= 0.35
            flags.append("penal_hidrica:demanda_alta+secos>=5+bal<=-20")
        elif secos >= 4 and bal <= -10:
            score -= 0.20
            flags.append("penal_hidrica:demanda_alta+secos>=4+bal<=-10")

    if demanda == "media":
        if secos >= 6 and bal <= -20:
            score -= 0.20
            flags.append("penal_hidrica:demanda_media+secos>=6+bal<=-20")

    # 2) Proxy sanitário: chuva alta + umidade alta
    # (mais importante para hortaliça-fruto/folhosas)
    grupo = (crop.grupo or "").lower()
    grupo_sensivel = grupo in {"folhosas", "hortaliça", "hortalica", "hortaliça-fruto", "hortalica-fruto"}
    if grupo_sensivel and chuva >= 80 and rh >= 92:
        score -= 0.20
        flags.append("penal_sanitaria:chuva>=80+rh>=92")

    # 3) Calor extremo (proxy)
    if tmax_p95 >= 35:
        score -= 0.10
        flags.append("penal_calor:tmax_p95>=35")

    return TeacherResult(score=_clip01(score), flags=tuple(flags))

