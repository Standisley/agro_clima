# agroclima_cultiva/explain/report_generator.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..schemas.inputs import FarmInput
from ..planner.windows import CandidateWindow


# =============================================================================
# Helpers
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")


def _area_bucket(area_m2: float) -> str:
    if area_m2 <= 5_000:
        return "muito pequena"
    if area_m2 <= 10_000:
        return "pequena"
    if area_m2 <= 20_000:
        return "média"
    return "grande"


def _loc_str(inp: FarmInput) -> str:
    if inp.municipio and inp.uf:
        return f"{inp.municipio}/{inp.uf} — ({inp.lat:.5f}, {inp.lon:.5f})"
    if inp.municipio:
        return f"{inp.municipio} — ({inp.lat:.5f}, {inp.lon:.5f})"
    if inp.uf:
        return f"{inp.uf} — ({inp.lat:.5f}, {inp.lon:.5f})"
    return f"({inp.lat:.5f}, {inp.lon:.5f})"


def _fmt(v: Any, digits: int = 1) -> str:
    try:
        x = float(v)
        if x != x:
            return "N/D"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/D"


def _climate_table(df: Optional[pd.DataFrame], n_days: int = 7) -> str:
    """
    Espera colunas padronizadas:
      ds, om_prcp_mm, om_et0_fao_mm, om_tmin_c, om_tmax_c, om_rh_max, om_wind_kmh
    """
    if df is None or df.empty:
        return "sem dados"

    d = df.copy()
    d["ds"] = pd.to_datetime(d["ds"])
    d = d.sort_values("ds").head(n_days).reset_index(drop=True)

    weekday = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    d["Data"] = d["ds"].dt.strftime("%d/%m") + " (" + d["ds"].dt.dayofweek.map(lambda x: weekday[int(x)]) + ")"

    # colunas (se existirem)
    cols = ["Data"]
    for c in ["om_prcp_mm", "om_et0_fao_mm", "om_tmin_c", "om_tmax_c", "om_rh_max", "om_wind_kmh"]:
        if c in d.columns:
            cols.append(c)

    out = d[cols].copy()
    rename = {
        "om_prcp_mm": "Chuva",
        "om_et0_fao_mm": "ET0",
        "om_tmin_c": "Tmin",
        "om_tmax_c": "Tmax",
        "om_rh_max": "URmax",
        "om_wind_kmh": "Vento",
    }
    out = out.rename(columns=rename)

    # formatar
    for c in ["Chuva", "ET0", "Tmin", "Tmax", "Vento"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: _fmt(x, 1))
    if "URmax" in out.columns:
        out["URmax"] = out["URmax"].map(lambda x: _fmt(x, 0))

    header = "Data | Chuva | ET0 | Tmin | Tmax | URmax | Vento"
    sep = "-" * len(header)

    def get(r, k) -> str:
        return str(r.get(k, "") if k in r else "")

    rows: List[str] = []
    for _, r in out.iterrows():
        rows.append(
            f"{get(r,'Data'):>10} | "
            f"{get(r,'Chuva'):>5} | "
            f"{get(r,'ET0'):>4} | "
            f"{get(r,'Tmin'):>4} | "
            f"{get(r,'Tmax'):>4} | "
            f"{get(r,'URmax'):>5} | "
            f"{get(r,'Vento'):>5}"
        )

    return "\n".join([header, sep] + rows)


def _features_line(features: Optional[Dict[str, Any]]) -> str:
    if not features:
        return "7d: N/D | 14d: N/D"

    def _to_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "__dataclass_fields__"):
            # dataclass -> dict
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}
        if isinstance(obj, dict):
            return obj
        return {}

    f7 = _to_dict(features.get("7d"))
    f14 = _to_dict(features.get("14d"))

    def line(label: str, d: Dict[str, Any]) -> str:
        if not d:
            return f"{label}: N/D"
        chuva = _fmt(d.get("chuva_total_mm", float("nan")), 1)
        et0 = _fmt(d.get("et0_total_mm", float("nan")), 1)
        bal = _fmt(d.get("balanco_hidrico_mm", float("nan")), 1)
        secos = int(d.get("dias_secos", 0) or 0)
        forte = int(d.get("dias_chuva_forte", 0) or 0)
        return f"{label}: chuva={chuva} | et0={et0} | bal={bal} | secos={secos} | forte={forte}"

    return f"{line('7d', f7)}\n{line('14d', f14)}"


# =============================================================================
# Report
# =============================================================================

def generate_report(
    project_name: str,
    project_version: str,
    inp: FarmInput,
    windows: List[CandidateWindow],
    climate_df: Optional[pd.DataFrame] = None,
    climate_features: Optional[Dict[str, Any]] = None,
) -> str:
    inp.validate()

    porte = _area_bucket(float(inp.area_m2))
    objetivo = (str(inp.objetivo) if inp.objetivo else "").replace("_", " ")

    top3 = windows[:3] if windows else []

    lines: List[str] = []
    lines.append(f"{project_name} (v{project_version}) — {_now_str()}")
    lines.append(f"Local: {_loc_str(inp)} | Área: {int(inp.area_m2):,} m² ({porte}) | Objetivo: {objetivo}".replace(",", "."))
    lines.append("")
    lines.append("CLIMA (7 DIAS) — Open-Meteo")
    lines.append(_climate_table(climate_df, n_days=7))
    lines.append("")
    lines.append("FEATURES")
    lines.append(_features_line(climate_features))
    lines.append("")
    lines.append("TOP 3 (após regras climáticas)")
    if not top3:
        lines.append("sem candidatos")
    else:
        for i, w in enumerate(top3, start=1):
            # score_total precisa existir (Etapa 2). Se não existir, cai para 0.0.
            score = getattr(w, "score_total", None)
            score_str = _fmt(score, 2) if score is not None else "N/D"
            lines.append(f"{i}. {w.cultura} | grupo={w.grupo} | score={score_str}")

    return "\n".join(lines)



