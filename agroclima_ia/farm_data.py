import pandas as pd
from pathlib import Path


def load_farm_observations(
    path: Path,
    date_col: str = "data",
    rain_col: str = "chuva_mm",
) -> pd.DataFrame:
    """
    Carrega dados históricos da fazenda e padroniza:
      - 'ds' = data
      - 'y'  = chuva (mm)

    Mantém as demais colunas (tmin, tmax, ur, vento, radiacao, et0,
    umidade_solo, temperatura_solo, etc.).
    """

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de observações da fazenda não encontrado: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if date_col not in df.columns or rain_col not in df.columns:
        raise ValueError(
            f"Arquivo {path} precisa ter colunas '{date_col}' e '{rain_col}'. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    df = df.copy()
    df.rename(columns={date_col: "ds", rain_col: "y"}, inplace=True)

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)

    # Se tiver linhas duplicadas na mesma data:
    # - soma chuva (y)
    # - faz média das demais numéricas
    group_cols = [c for c in df.columns if c != "ds"]

    agg_dict = {}
    for c in group_cols:
        if c == "y":
            agg_dict[c] = "sum"
        else:
            # média para demais variáveis numéricas
            agg_dict[c] = "mean"

    df = df.groupby("ds", as_index=False).agg(agg_dict)

    return df

def merge_external_and_farm(
    df_external: pd.DataFrame,
    df_farm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mescla chuva diária externa (open-meteo) com observações da fazenda.

    Regras:
      - índice diário contínuo de min(data) até max(data) dos dois datasets.
      - onde houver y_farm (não nulo), usa y_farm.
      - onde não houver, usa y_external.
      - onde não houver nenhum, assume 0.0 mm.

    Mantém as colunas adicionais da fazenda (tmin, tmax, ur, vento, radiacao, et0,
    umidade_solo, temperatura_solo, etc.).
    """

    df_ext = df_external.copy()
    df_fm = df_farm.copy()

    if "ds" not in df_ext.columns or "y" not in df_ext.columns:
        raise ValueError("df_external precisa ter colunas ['ds','y']")
    if "ds" not in df_fm.columns or "y" not in df_fm.columns:
        raise ValueError("df_farm precisa ter colunas ['ds','y']")

    df_ext["ds"] = pd.to_datetime(df_ext["ds"])
    df_fm["ds"] = pd.to_datetime(df_fm["ds"])

    df_ext = df_ext.sort_values("ds")
    df_fm = df_fm.sort_values("ds")

    # período coberto por qualquer uma das fontes
    start = min(df_ext["ds"].min(), df_fm["ds"].min())
    end = max(df_ext["ds"].max(), df_fm["ds"].max())

    full_dates = pd.date_range(start=start, end=end, freq="D")
    base = pd.DataFrame({"ds": full_dates})

    # externo: só ds + y_external
    base = base.merge(
        df_ext[["ds", "y"]].rename(columns={"y": "y_external"}),
        on="ds", how="left"
    )

    # fazenda: renomeia y -> y_farm, mantém as demais colunas (tmin, tmax, etc.)
    farm_cols = [c for c in df_fm.columns if c != "y"]  # ds + extras
    df_fm_ren = df_fm[farm_cols + ["y"]].rename(columns={"y": "y_farm"})

    base = base.merge(df_fm_ren, on="ds", how="left")

    # prioridade para dado da fazenda
    base["y"] = base["y_farm"].where(base["y_farm"].notna(), base["y_external"])

    # onde não há nem fazenda nem externo, assume 0
    base["y"] = base["y"].fillna(0.0)

    # monta lista final de colunas:
    # ds, y, + tudo da fazenda exceto ds e y_farm (já incorporado em y)
    extra_cols = [c for c in base.columns if c not in ["ds", "y", "y_external", "y_farm"]]

    df_daily_final = base[["ds", "y"] + extra_cols].copy()
    return df_daily_final
