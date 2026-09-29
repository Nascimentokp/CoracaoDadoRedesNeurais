"""Gravação padronizada de figuras e tabelas.

Regra herdada do projeto CoracaoDadoTema6: **nenhuma figura é gravada sem a
tabela que a originou**. Os notebooks são versionados sem saída
(`nbstripout`), de modo que o que não for persistido em disco desaparece do
repositório.

* figuras → `reports/figures/<secao>/<nome>.png`
* tabelas → `reports/tables/<secao>/<nome>.csv` (separador `;`, decimal `,`,
  abre direto no Excel em português)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def configurar_estilo() -> None:
    """Aplica o estilo visual único do projeto ao matplotlib."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": config.DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C9D3DB",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6EBEF",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": config.AZUL_PROFUNDO,
            "axes.labelsize": 10,
            "axes.prop_cycle": plt.cycler(color=config.PALETA),
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def salvar_figura(
    fig: plt.Figure, secao: str, nome: str, dados: pd.DataFrame | None = None
) -> Path:
    """Grava a figura e, opcionalmente, a tabela-fonte com o mesmo nome."""
    destino = config.FIGURES / secao
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.png"
    fig.savefig(caminho)
    if dados is not None:
        salvar_tabela(dados, secao, nome)
    return caminho


def salvar_tabela(dados: pd.DataFrame, secao: str, nome: str, indice: bool = True) -> Path:
    """Grava uma tabela em `reports/tables/<secao>/<nome>.csv`."""
    destino = config.TABLES / secao
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.csv"
    dados.to_csv(caminho, sep=";", decimal=",", index=indice, encoding="utf-8-sig")
    return caminho


def ler_tabela(secao: str, nome: str) -> pd.DataFrame:
    """Lê uma tabela gravada por :func:`salvar_tabela`."""
    caminho = config.TABLES / secao / f"{nome}.csv"
    return pd.read_csv(caminho, sep=";", decimal=",", index_col=0, encoding="utf-8-sig")


def salvar_json(conteudo: Any, caminho: Path) -> Path:
    """Grava um dicionário como JSON legível, criando o diretório se preciso."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return caminho


def ler_json(caminho: Path) -> Any:
    """Lê um JSON gravado por :func:`salvar_json`."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))
