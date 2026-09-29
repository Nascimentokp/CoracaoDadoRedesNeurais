"""Gráficos repetidos em vários notebooks: curvas de treino, ROC e matriz de confusão."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src import config


def curvas_de_treino(
    historico: dict[str, list[float]], titulo: str
) -> tuple[plt.Figure, pd.DataFrame]:
    """Perda (e AUC, se houver) por época, treino × validação.

    Aceita tanto ``History.history`` do Keras quanto os dicionários dos
    treinos em NumPy e PyTorch.
    """
    dados = pd.DataFrame(
        {chave: pd.Series(valores) for chave, valores in historico.items()}
    ).rename_axis("epoca")
    pares = [("loss", "val_loss"), ("perda", "perda_val"), ("auc", "val_auc")]
    presentes = [(a, b) for a, b in pares if b in dados]

    fig, eixos = plt.subplots(1, len(presentes), figsize=(5.5 * len(presentes), 3.8), squeeze=False)
    for eixo, (treino, val) in zip(eixos[0], presentes):
        if treino in dados:
            eixo.plot(dados.index + 1, dados[treino], label="Treino", color=config.AZUL_COBALTO)
        eixo.plot(dados.index + 1, dados[val], label="Validação", color=config.CORAL)
        eixo.set_xlabel("Época")
        eixo.set_ylabel("AUC" if "auc" in val else "Entropia cruzada")
        eixo.legend()
    fig.suptitle(titulo, fontweight="bold", color=config.AZUL_PROFUNDO)
    fig.tight_layout()
    return fig, dados


def curvas_roc(
    probabilidades: dict[str, np.ndarray], y: np.ndarray
) -> tuple[plt.Figure, pd.DataFrame]:
    """Curvas ROC de vários modelos sobre o mesmo conjunto."""
    from sklearn.metrics import roc_auc_score

    fig, eixo = plt.subplots(figsize=(5.5, 5))
    linhas = []
    for (nome, p), cor in zip(probabilidades.items(), config.PALETA):
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        eixo.plot(fpr, tpr, label=f"{nome} (AUC {auc:.3f})", color=cor, linewidth=1.6)
        linhas.append(pd.DataFrame({"modelo": nome, "fpr": fpr, "tpr": tpr}))
    eixo.plot([0, 1], [0, 1], linestyle="--", color="#9AA7B0", linewidth=1)
    eixo.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    eixo.set_ylabel("Sensibilidade")
    eixo.set_title("Curva ROC — teste")
    eixo.legend(loc="lower right", fontsize=8)
    return fig, pd.concat(linhas, ignore_index=True)


def matriz_confusao(
    y: np.ndarray, p: np.ndarray, limiar: float, titulo: str
) -> tuple[plt.Figure, pd.DataFrame]:
    """Matriz de confusão anotada com contagem e percentual da linha."""
    rotulos = ["Sem doença", "Com doença"]
    cm = confusion_matrix(y, (p >= limiar).astype(int), labels=[0, 1])
    tabela = pd.DataFrame(
        cm, index=[f"real: {r}" for r in rotulos], columns=[f"prev.: {r}" for r in rotulos]
    )

    fig, eixo = plt.subplots(figsize=(4.8, 4.2))
    eixo.imshow(cm, cmap="Blues")
    eixo.grid(False)
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum()
            cor = "white" if cm[i, j] > cm.max() / 2 else config.AZUL_PROFUNDO
            eixo.text(
                j,
                i,
                f"{cm[i, j]:,}\n({pct:.0%})".replace(",", "."),
                ha="center",
                va="center",
                color=cor,
            )
    eixo.set_xticks([0, 1], rotulos)
    eixo.set_yticks([0, 1], rotulos)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    eixo.set_title(f"{titulo}\nlimiar = {limiar:.2f}")
    return fig, tabela
