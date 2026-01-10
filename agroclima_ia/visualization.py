# agroclima_ia/visualization.py

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_rain_et0_balance(
    df_forecast: pd.DataFrame,
    fazenda_id: str | None = None,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plota chuva (y_ensemble_mm), ET0 (om_et0_fao_mm) e balanço hídrico (water_balance_mm).
    """
    df = df_forecast.copy()

    if "ds" not in df.columns or "y_ensemble_mm" not in df.columns:
        # No fluxo main.py, isso não deve acontecer
        raise ValueError("DataFrame precisa ter as colunas 'ds' e 'y_ensemble_mm' para plot.")

    # --- Prepara Dados ---
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")
    
    has_et0 = "om_et0_fao_mm" in df.columns
    has_balance = "water_balance_mm" in df.columns

    # --- Configuração do Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    if fazenda_id:
        title = f"Previsão de Manejo Climático (7 dias) - {fazenda_id.upper()}"
        plt.title(title)

    # Barras de chuva (Eixo 1)
    ax1.bar(
        df["ds"],
        df["y_ensemble_mm"],
        width=0.8,
        align="center",
        color="skyblue",
        label="Chuva (ensemble, mm)",
    )
    ax1.set_ylabel("Chuva (mm)")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    
    # Linha ET0 (Eixo 1)
    if has_et0:
        ax1.plot(
            df["ds"],
            df["om_et0_fao_mm"],
            marker="o",
            linestyle="-",
            color="orange",
            label="ET0 FAO (mm/dia)",
        )

    # Eixo Secundário para Balanço Hídrico (Eixo 2)
    if has_balance:
        ax2 = ax1.twinx()
        
        # Corrigir cor do balanço (Superávit/Déficit)
        colors = ["green" if b >= 0 else "red" for b in df["water_balance_mm"]]
        
        ax2.plot(
            df["ds"],
            df["water_balance_mm"],
            marker="s",
            linestyle="--",
            color="gray", 
            alpha=0.6,
            label="Balanço hídrico (Saldo, mm)",
        )
        ax2.set_ylabel("Balanço Hídrico (mm)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        
        # Adiciona a linha de referência zero
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1)

    # --- Finalização ---
    ax1.set_xlabel("Data")
    
    # Rotação das datas no eixo X
    fig.autofmt_xdate(rotation=45) 

    # Unir legendas (para mostrar no Eixo 1)
    lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    if has_balance:
        lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_ax1 + lines_ax2, labels_ax1 + labels_ax2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    # 1. Salva a Figura
    if save_path:
        # Cria o diretório se não existir
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        
    # 2. Exibe ou Fecha (Fechamento é CRÍTICO)
    if show:
        plt.show()

    # ✅ CORREÇÃO CRÍTICA: Fecha a figura para liberar memória e evitar pop-up
    plt.close('all')
