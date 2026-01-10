from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

Objective = Literal["renda_rapida", "baixo_risco", "seguranca_alimentar", "diversificacao"]

@dataclass(frozen=True)
class FarmInput:
    lat: float
    lon: float
    area_m2: float
    objetivo: Objective = "baixo_risco"
    municipio: Optional[str] = None
    uf: Optional[str] = None
    perfil_produtor: str = ""
    restricoes: Dict[str, Any] = field(default_factory=dict)
    contexto: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not (-90.0 <= float(self.lat) <= 90.0):
            raise ValueError("lat inválida: deve estar entre -90 e 90.")
        if not (-180.0 <= float(self.lon) <= 180.0):
            raise ValueError("lon inválida: deve estar entre -180 e 180.")
        if float(self.area_m2) <= 0:
            raise ValueError("area_m2 inválida: deve ser > 0.")
        if self.objetivo not in {"renda_rapida", "baixo_risco", "seguranca_alimentar", "diversificacao"}:
            raise ValueError("objetivo inválido.")
