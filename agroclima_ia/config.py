# agroclima_ia/config.py

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# PASTAS BÁSICAS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
#PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
#PLOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# PERFIS DE FAZENDA / MANEJO
# -----------------------------------------------------------------------------
# Cada chave (ex: "goiania", "rio_verde") é um perfil de fazenda.
# No CLI você pode usar ACTIVE_FARM_ID como padrão.
# NO APP/STREAMLIT:
#   → NÃO use apenas as constantes globais.
#   → Use get_farm_profile(farm_id) + get_paths_for_farm(farm_id),
#     passando o farm_id escolhido na barra lateral.
# =============================================================================

FARM_CONFIG: Dict[str, Dict[str, Any]] = {
    # -------------------------
    # PERFIS QUE VOCÊ JÁ ESTÁ USANDO
    # -------------------------
    "rio_verde": {
        "regiao": "Rio Verde (GO)",
        "cultura": "soja",
        "estagio_fenologico": "R1",           # fase reprodutiva, sensível a déficit
        "sistema": "sequeiro",
        "solo": "argiloso",
        # Coordenadas aproximadas de Rio Verde - GO
        "lat": -17.7923,
        "lon": -50.9191,
        "series_id": "rio_verde",
    },
    "campo_grande": {
        "regiao": "Campo Grande (MS)",
        "cultura": "milho",
        "estagio_fenologico": "V4",           # vegetativo avançado
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        # Coordenadas aproximadas de Campo Grande - MS
        "lat": -20.4697,
        "lon": -54.6201,
        "series_id": "campo_grande",
    },
    "goiania": {
        "regiao": "Goiânia (GO)",
        "cultura": "soja",
        "estagio_fenologico": "R1",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        # Coordenadas de Goiânia - GO
        "lat": -16.6869,
        "lon": -49.2648,
        "series_id": "goiania",
    },

    # -------------------------
    # NOVOS PERFIS – CAFÉ
    # -------------------------
    "sul_mg_cafe_frutificacao": {
        "regiao": "Sul de Minas (MG)",
        "cultura": "cafe",
        "estagio_fenologico": "frutificacao",
        "sistema": "sequeiro",
        "solo": "argiloso",
        # Coordenadas aproximadas de Varginha (representando Sul de Minas)
        "lat": -21.5514,
        "lon": -45.4303,
        "series_id": "sul_mg_cafe_frutificacao",
    },
    "sul_mg_cafe_pos_colheita": {
        "regiao": "Sul de Minas (MG)",
        "cultura": "cafe",
        "estagio_fenologico": "pos_colheita",
        "sistema": "sequeiro",
        "solo": "argiloso",
        "lat": -21.5514,
        "lon": -45.4303,
        "series_id": "sul_mg_cafe_pos_colheita",
    },

    # -------------------------
    # NOVOS PERFIS – LARANJA (CITROS)
    # -------------------------
    "citros_sp_florescimento": {
        "regiao": "Centro-leste de São Paulo",
        "cultura": "laranja",
        "estagio_fenologico": "florescimento",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        # Coordenadas aproximadas de Limeira (SP), polo citrícola
        "lat": -22.5645,
        "lon": -47.4012,
        "series_id": "citros_sp_florescimento",
    },
    "citros_sp_frutificacao": {
        "regiao": "Centro-leste de São Paulo",
        "cultura": "laranja",
        "estagio_fenologico": "frutificacao",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        "lat": -22.5645,
        "lon": -47.4012,
        "series_id": "citros_sp_frutificacao",
    },

    # -------------------------
    # NOVOS PERFIS – TRIGO
    # -------------------------
    "trigo_rs_perfilhamento": {
        "regiao": "Norte do Rio Grande do Sul",
        "cultura": "trigo",
        "estagio_fenologico": "perfilhamento",
        "sistema": "sequeiro",
        "solo": "argiloso",
        # Coordenadas aproximadas de Passo Fundo - RS
        "lat": -28.2620,
        "lon": -52.4064,
        "series_id": "trigo_rs_perfilhamento",
    },
    "trigo_rs_espigamento": {
        "regiao": "Norte do Rio Grande do Sul",
        "cultura": "trigo",
        "estagio_fenologico": "espigamento",
        "sistema": "sequeiro",
        "solo": "argiloso",
        "lat": -28.2620,
        "lon": -52.4064,
        "series_id": "trigo_rs_espigamento",
    },

    # -------------------------
    # NOVOS PERFIS – ARROZ
    # -------------------------
    "arroz_irrigado_rs_enchimento": {
        "regiao": "Campanha Gaúcha (RS)",
        "cultura": "arroz",
        "estagio_fenologico": "enchimento",
        "sistema": "alagado",
        "solo": "gleissolo",
        # Coordenadas aproximadas de Pelotas - RS (região arrozeira)
        "lat": -31.7654,
        "lon": -52.3376,
        "series_id": "arroz_irrigado_rs_enchimento",
    },
    "arroz_sequeiro_go": {
        "regiao": "Goiás (GO)",
        "cultura": "arroz",
        "estagio_fenologico": "vegetativo",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        # Coordenadas aproximadas do estado de Goiás (centro)
        "lat": -15.8270,
        "lon": -49.8362,
        "series_id": "arroz_sequeiro_go",
    },

    # -------------------------
    # NOVOS PERFIS – BANANA
    # -------------------------
    "banana_vale_ribeira_vegetativo": {
        "regiao": "Vale do Ribeira (SP)",
        "cultura": "banana",
        "estagio_fenologico": "crescimento_vegetativo",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        # Coordenadas aproximadas de Registro - SP (Vale do Ribeira)
        "lat": -24.4971,
        "lon": -47.8449,
        "series_id": "banana_vale_ribeira_vegetativo",
    },
    "banana_vale_ribeira_enchimento": {
        "regiao": "Vale do Ribeira (SP)",
        "cultura": "banana",
        "estagio_fenologico": "enchimento_cachos",
        "sistema": "sequeiro",
        "solo": "argilo-arenoso",
        "lat": -24.4971,
        "lon": -47.8449,
        "series_id": "banana_vale_ribeira_enchimento",
    },

    # -------------------------
    # DEFAULT GENÉRICO
    # -------------------------
    "default": {
        "regiao": "Centro-Oeste (Brasil)",
        "cultura": "soja",
        "estagio_fenologico": "vegetativo",
        "sistema": "sequeiro",
        "solo": "argiloso",
        # Coordenadas genéricas para Centro-Oeste
        "lat": -16.0,
        "lon": -49.0,
        "series_id": "default",
    },
}

# =============================================================================
# ESCOLHA DA FAZENDA ATIVA (USO PRINCIPALMENTE NO CLI)
# -----------------------------------------------------------------------------
# → No script de linha de comando, você pode usar ACTIVE_FARM_ID como padrão.
# → No app, o ideal é NÃO depender só disso, e sim chamar get_paths_for_farm(farm_id)
#   passando explicitamente o id selecionado na interface.
# =============================================================================

ACTIVE_FARM_ID = "campo_grande"   # ex: "rio_verde", "campo_grande", "citros_sp_frutificacao", etc.
ACTIVE_FARM = FARM_CONFIG.get(ACTIVE_FARM_ID, FARM_CONFIG["default"])

# Coordenadas e série padrão derivadas da fazenda ativa (LEGADO / DEFAULT)
DEFAULT_LAT = ACTIVE_FARM["lat"]
DEFAULT_LON = ACTIVE_FARM["lon"]
DEFAULT_SERIES_ID = ACTIVE_FARM["series_id"]

# =============================================================================
# CAMINHOS DE ARQUIVOS PADRÃO (baseados na fazenda ativa) – USO LEGADO
# -----------------------------------------------------------------------------
# Para código novo (especialmente o app), prefira SEMPRE:
#   get_paths_for_farm(farm_id)
# =============================================================================
DAILY_RAIN_CSV = DATA_DIR / f"{DEFAULT_SERIES_ID}_daily_rain.csv"
FARM_OBS_CSV   = DATA_DIR / f"{DEFAULT_SERIES_ID}_farm_obs.csv"
LGB_MODEL_PATH = MODELS_DIR / f"{DEFAULT_SERIES_ID}_lightgbm.txt"
#PLOT_PATH      = PLOTS_DIR / f"{DEFAULT_SERIES_ID}_forecast.png"


# =============================================================================
# HELPERS RECOMENDADOS PARA APP / CÓDIGO NOVO
# =============================================================================

def get_farm_profile(farm_id: str | None = None) -> Dict[str, Any]:
    """
    Retorna o dicionário de configuração de uma fazenda.

    - Se farm_id for None, retorna a fazenda ativa (ACTIVE_FARM).
    - Caso o farm_id não exista, cai no perfil "default".
    """
    if farm_id is None:
        return ACTIVE_FARM
    return FARM_CONFIG.get(farm_id, FARM_CONFIG["default"])


def get_paths_for_farm(farm_id: str | None = None) -> Dict[str, Path]:
    """
    Retorna um dicionário com os caminhos CSV/modelo/gráfico para uma fazenda específica.

    - Se farm_id for None, usa a fazenda ativa.
    - Isso garante que CADA fazenda tenha seus próprios arquivos:
        data/{series_id}_daily_rain.csv
        data/{series_id}_farm_obs.csv
        models/{series_id}_lightgbm.txt
        plots/{series_id}_forecast.png
    """
    farm = get_farm_profile(farm_id)
    series_id = farm["series_id"]

    return {
        "daily_rain_csv": DATA_DIR / f"{series_id}_daily_rain.csv",
        "farm_obs_csv":   DATA_DIR / f"{series_id}_farm_obs.csv",
        "model_path":     MODELS_DIR / f"{series_id}_lightgbm.txt",
        #"plot_path":      PLOTS_DIR / f"{series_id}_forecast.png",
    }


