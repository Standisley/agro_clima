from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass(frozen=True)
class CropRecommendation:
    cultura: str
    area_m2: float
    janela_inicio: str
    janela_fim: str
    justificativa: str

@dataclass(frozen=True)
class RecommendationOutput:
    projeto: str
    versao: str
    lat: float
    lon: float
    area_m2: float
    objetivo: str
    municipio: Optional[str] = None
    uf: Optional[str] = None

    culturas_recomendadas: List[CropRecommendation] = field(default_factory=list)
    explicabilidade: str = ""
    observacoes_legais: str = ""
    rastreabilidade: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projeto": self.projeto,
            "versao": self.versao,
            "lat": self.lat,
            "lon": self.lon,
            "area_m2": self.area_m2,
            "objetivo": self.objetivo,
            "municipio": self.municipio,
            "uf": self.uf,
            "culturas_recomendadas": [
                {
                    "cultura": c.cultura,
                    "area_m2": c.area_m2,
                    "janela_inicio": c.janela_inicio,
                    "janela_fim": c.janela_fim,
                    "justificativa": c.justificativa,
                }
                for c in self.culturas_recomendadas
            ],
            "explicabilidade": self.explicabilidade,
            "observacoes_legais": self.observacoes_legais,
            "rastreabilidade": self.rastreabilidade,
        }
