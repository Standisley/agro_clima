# agroclima_cultiva/planner/catalog.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

Objective = Literal["renda_rapida", "baixo_risco", "seguranca_alimentar", "diversificacao"]

# Grupo = tipo de cultura (não é objetivo)
CropGroup = Literal[
    "folhosas",
    "hortaliça",
    "hortaliça-fruto",
    "raízes",
    "frutífera",
    "grãos",
    "medicinal",
]

WaterDemand = Literal["baixa", "media", "alta"]
Complexity = Literal["baixa", "media", "alta"]
Investment = Literal["baixo", "medio", "alto"]
Risk = Literal["baixo", "medio", "alto"]

CATALOG_PATH = Path(__file__).parent / "data" / "catalog_v1.json"


@dataclass(frozen=True)
class CropSpec:
    crop_id: str
    nome: str
    grupo: CropGroup
    ciclo_dias: Tuple[int, int]
    area_min_m2: int
    demanda_hidrica: WaterDemand
    complexidade: Complexity
    investimento: Investment
    risco: Risk
    tags: Tuple[str, ...]
    observacao: str = ""


def _parse_ciclo_dias(value) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ciclo_dias deve ser uma lista/tupla com 2 itens: [min, max].")
    a, b = value[0], value[1]
    a_i, b_i = int(a), int(b)
    if a_i <= 0 or b_i <= 0 or b_i < a_i:
        raise ValueError("ciclo_dias inválido: esperava min>0, max>0 e max>=min.")
    return (a_i, b_i)


def load_catalog_from_json(path: Path = CATALOG_PATH) -> List[CropSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Catálogo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "crops" not in raw:
        raise ValueError("JSON inválido: esperado um objeto com a chave 'crops'.")

    crops: List[CropSpec] = []

    for item in raw.get("crops", []):
        if not isinstance(item, dict):
            continue

        crop_id = str(item.get("crop_id", "")).strip()
        nome = str(item.get("nome", "")).strip()
        grupo = str(item.get("grupo", "")).strip()
        if not crop_id or not nome or not grupo:
            # item incompleto: ignora para não quebrar o pipeline
            continue

        ciclo = _parse_ciclo_dias(item.get("ciclo_dias"))

        crops.append(
            CropSpec(
                crop_id=crop_id,
                nome=nome,
                grupo=grupo,  # type: ignore[assignment]
                ciclo_dias=ciclo,
                area_min_m2=int(item.get("area_min_m2", 0)),
                demanda_hidrica=str(item.get("demanda_hidrica", "media")).lower(),  # type: ignore[assignment]
                complexidade=str(item.get("complexidade", "media")).lower(),        # type: ignore[assignment]
                investimento=str(item.get("investimento", "medio")).lower(),        # type: ignore[assignment]
                risco=str(item.get("risco", "medio")).lower(),                      # type: ignore[assignment]
                tags=tuple(item.get("tags", []) or []),
                observacao=str(item.get("observacao", "") or ""),
            )
        )

    if not crops:
        raise ValueError(f"Catálogo carregado, mas sem itens válidos em: {path}")

    return crops

