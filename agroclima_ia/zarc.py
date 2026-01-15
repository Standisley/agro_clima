import pandas as pd
from datetime import datetime
from pathlib import Path
import agroclima_ia.config as cfg

def get_current_decendio(date_obj=None):
    if date_obj is None: date_obj = datetime.now()
    day = date_obj.day
    return (date_obj.month - 1) * 3 + (1 if day <= 10 else 2 if day <= 20 else 3)

def check_zarc_risk(regiao: str, cultura: str, solo: str) -> str:
    csv_path = cfg.DATA_DIR / "zarc_rules.csv"
    if not csv_path.exists(): return "N/D (Sem CSV)"
    try:
        df = pd.read_csv(csv_path)
        # Normalização para evitar erros de busca
        df['regiao'] = df['regiao'].astype(str).str.strip()
        df['cultura'] = df['cultura'].astype(str).str.lower().str.strip()
        df['solo'] = df['solo'].astype(str).str.lower().str.strip()
        
        current_decendio = get_current_decendio()
        
        filtro = (
            (df['regiao'] == regiao.strip()) & 
            (df['cultura'] == cultura.lower().strip()) & 
            (df['solo'] == solo.lower().strip()) &
            (df['decendio'] == current_decendio)
        )
        
        resultado = df[filtro]
        if not resultado.empty: return resultado.iloc[0]['risco']
        else: return "FORA DA JANELA ⛔"
    except Exception as e: return f"Erro ZARC: {e}"