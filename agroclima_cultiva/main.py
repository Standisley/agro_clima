from __future__ import annotations

from datetime import datetime
from typing import Optional

from agroclima_cultiva.schemas.inputs import FarmInput
from agroclima_cultiva.planner.windows import build_candidate_windows

from agroclima_cultiva.climate.openmeteo import fetch_daily_forecast
from agroclima_cultiva.climate.metrics import compute_features_7_14

from agroclima_cultiva.explain.llm_report import (
    build_evidence_pack,
    render_report_llm_only,
)
from agroclima_cultiva.explain.local_llm import llm_local


PROJECT_NAME = "AgroClima Cultiva"
PROJECT_VERSION = "0.1.0"


def main(
    lat: float = -17.7923,
    lon: float = -50.9191,
    area_m2: float = 5000.0,
    objetivo: str = "baixo_risco",
    municipio: Optional[str] = "Rio Verde",
    uf: Optional[str] = "GO",
) -> None:
    """
    Fluxo principal do AgroClima Cultiva (LLM ONLY).

    Pipeline:
    1) Entrada do produtor (FarmInput)
    2) Clima diário (Open-Meteo)
    3) Features climáticas 7d / 14d
    4) Classificador heurístico (janelas candidatas)
    5) Evidence Pack (determinístico)
    6) LLM local gera o relatório final
    """

    # ------------------------------------------------------------------
    # 1) Entrada
    # ------------------------------------------------------------------
    inp = FarmInput(
        lat=lat,
        lon=lon,
        area_m2=area_m2,
        objetivo=objetivo,  # type: ignore[arg-type]
        municipio=municipio,
        uf=uf,
        perfil_produtor="agricultura_familiar",
        restricoes={},
        contexto={},
    )
    inp.validate()

    # ------------------------------------------------------------------
    # 2) Clima – Open-Meteo (forecast diário até 16 dias)
    # ------------------------------------------------------------------
    df_climate = fetch_daily_forecast(
        lat=inp.lat,
        lon=inp.lon,
        days=16,
    )

    # ------------------------------------------------------------------
    # 3) Features climáticas agregadas
    # ------------------------------------------------------------------
    climate_features = compute_features_7_14(df_climate)

    # ------------------------------------------------------------------
    # 4) Classificador determinístico (heurístico + clima)
    # ------------------------------------------------------------------
    windows = build_candidate_windows(inp, top_k=8)

    # ------------------------------------------------------------------
    # 5) Evidence Pack (ANTI-ALUCINAÇÃO)
    # ------------------------------------------------------------------
    evidence = build_evidence_pack(
        inp=inp,
        climate_features=climate_features,
        top_windows=windows,
        climate_df_7d=df_climate,
    )

    # ------------------------------------------------------------------
    # 6) Relatório FINAL – SOMENTE LLM
    # ------------------------------------------------------------------
    report = render_report_llm_only(
        evidence_pack=evidence,
        llm_fn=llm_local,
    )

    header = (
        f"{PROJECT_NAME} (v{PROJECT_VERSION}) — "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
    )

    print(header)
    print(report)




if __name__ == "__main__":
    main(
        municipio="Rio Verde",
        uf="GO"
    )


