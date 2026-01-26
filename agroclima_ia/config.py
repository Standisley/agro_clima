# agroclima_ia/config.py

import json
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

# Caminho para o arquivo de registro de fazendas (JSON)
FARM_REGISTRY_JSON = DATA_DIR / "farm_registry.json"

# =============================================================================
# CARREGAMENTO DE PERFIS (ARQUITETURA DE DADOS EXTERNA)
# =============================================================================

def _load_farm_registry() -> Dict[str, Dict[str, Any]]:
    """
    Carrega as configurações das fazendas do arquivo JSON externo.
    Se o arquivo não existir ou der erro, carrega um fallback mínimo para não quebrar.
    """
    # Configuração mínima de segurança (Fallback)
    fallback_config = {
        "default": {
            "regiao": "Centro-Oeste (Brasil)",
            "cultura": "soja",
            "estagio_fenologico": "vegetativo",
            "sistema": "sequeiro",
            "solo": "argiloso",
            "lat": -16.0,
            "lon": -49.0,
            "series_id": "default",
        }
    }

    if not FARM_REGISTRY_JSON.exists():
        print(f"⚠️ AVISO: Arquivo {FARM_REGISTRY_JSON} não encontrado. Usando configuração padrão.")
        return fallback_config

    try:
        with open(FARM_REGISTRY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Garante que sempre exista a chave 'default' para evitar KeyErrors
            if "default" not in data:
                data["default"] = fallback_config["default"]
            return data
    except Exception as e:
        print(f"⚠️ ERRO CRÍTICO ao ler JSON de fazendas: {e}. Usando fallback.")
        return fallback_config

# Carrega a configuração na inicialização do módulo
FARM_CONFIG = _load_farm_registry()

# =============================================================================
# ESCOLHA DA FAZENDA ATIVA
# =============================================================================

ACTIVE_FARM_ID = "campo_grande" 
# Se o ID configurado não existir no JSON, cai no default
ACTIVE_FARM = FARM_CONFIG.get(ACTIVE_FARM_ID, FARM_CONFIG["default"])

# Coordenadas e série padrão (LEGADO / DEFAULT)
DEFAULT_LAT = ACTIVE_FARM["lat"]
DEFAULT_LON = ACTIVE_FARM["lon"]
DEFAULT_SERIES_ID = ACTIVE_FARM["series_id"]

# =============================================================================
# CAMINHOS DE ARQUIVOS PADRÃO (LEGADO)
# =============================================================================
DAILY_RAIN_CSV = DATA_DIR / f"{DEFAULT_SERIES_ID}_daily_rain.csv"
FARM_OBS_CSV   = DATA_DIR / f"{DEFAULT_SERIES_ID}_farm_obs.csv"
LGB_MODEL_PATH = MODELS_DIR / f"{DEFAULT_SERIES_ID}_lightgbm.txt"


# =============================================================================
# HELPERS RECOMENDADOS PARA O APP
# =============================================================================

def get_farm_profile(farm_id: str | None = None) -> Dict[str, Any]:
    """
    Retorna o dicionário de configuração de uma fazenda.
    """
    if farm_id is None:
        return ACTIVE_FARM
    return FARM_CONFIG.get(farm_id, FARM_CONFIG["default"])


def get_paths_for_farm(farm_id: str | None = None) -> Dict[str, Path]:
    """
    Retorna os caminhos dos arquivos para uma fazenda específica.
    """
    farm = get_farm_profile(farm_id)
    series_id = farm["series_id"]

    return {
        "daily_rain_csv": DATA_DIR / f"{series_id}_daily_rain.csv",
        "farm_obs_csv":   DATA_DIR / f"{series_id}_farm_obs.csv",
        "model_path":     MODELS_DIR / f"{series_id}_lightgbm.txt",
    }


