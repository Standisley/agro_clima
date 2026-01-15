# agroclima_ia/zarc.py

import pandas as pd
from datetime import datetime
from pathlib import Path
import agroclima_ia.config as cfg

def get_current_decendio(date_obj=None):
    """Calcula o decêndio do ano (1 a 36)."""
    if date_obj is None:
        date_obj = datetime.now()
    
    day = date_obj.day
    month = date_obj.month
    
    # 1 a 10 = decendio 1 | 11 a 20 = decendio 2 | 21+ = decendio 3
    if day <= 10:
        d_month = 1
    elif day <= 20:
        d_month = 2
    else:
        d_month = 3
        
    # Fórmula: (Mes - 1) * 3 + Decendio_do_mes
    return (month - 1) * 3 + d_month

def check_zarc_risk(regiao: str, cultura: str, solo: str) -> str:
    """
    Verifica o risco ZARC para a data de HOJE baseada no CSV local.
    """
    # Garante caminho correto
    csv_path = cfg.DATA_DIR / "zarc_rules.csv"
    
    if not csv_path.exists():
        return "N/D (Sem CSV)"

    try:
        df = pd.read_csv(csv_path)
        
        # Normalização rigorosa (remove espaços e converte para minuscula)
        df['regiao'] = df['regiao'].astype(str).str.strip()
        df['cultura'] = df['cultura'].astype(str).str.lower().str.strip()
        df['solo'] = df['solo'].astype(str).str.lower().str.strip()
        
        current_decendio = get_current_decendio()
        
        # Filtra
        filtro = (
            (df['regiao'] == regiao.strip()) & 
            (df['cultura'] == cultura.lower().strip()) & 
            (df['solo'] == solo.lower().strip()) &
            (df['decendio'] == current_decendio)
        )
        
        resultado = df[filtro]
        
        if not resultado.empty:
            return resultado.iloc[0]['risco']
        else:
            return "FORA DA JANELA ⛔"
            
    except Exception as e:
        return f"Erro ZARC: {e}"