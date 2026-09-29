"""Métricas, escolha de limiar e explicação local compartilhadas por todos os modelos.

Em triagem cardiovascular, a acurácia sozinha engana: o que importa é
quantos doentes a rede encontra (sensibilidade / recall) e quanto do tempo
da equipe é gasto com alarmes falsos (precisão). Por isso o limiar de 0,5 é
só ponto de partida — ele é escolhido na **validação**, nunca no teste.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def metricas(y: np.ndarray, p: np.ndarray, limiar: float = 0.5) -> dict[str, float]:
    """Resumo das métricas de um modelo em um limiar de decisão."""
    pred = (p >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "limiar": limiar,
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "acuracia": accuracy_score(y, pred),
        "sensibilidade": recall_score(y, pred, zero_division=0),
        "especificidade": vn / (vn + fp) if (vn + fp) else 0.0,
        "precisao": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "vp": int(vp),
        "fp": int(fp),
        "fn": int(fn),
        "vn": int(vn),
    }


def escolher_limiar(
    y_val: np.ndarray, p_val: np.ndarray, recall_minimo: float = config.RECALL_MINIMO
) -> float:
    """Maior limiar que ainda garante a sensibilidade mínima na validação.

    Quanto maior o limiar, menos alarmes falsos; a restrição de recall impede
    que ele suba a ponto de deixar doentes passarem.

    ``drop_intermediate=False`` é necessário: por padrão o scikit-learn descarta
    pontos colineares da curva ROC, e o limiar exato em que o recall cruza o
    mínimo pode estar entre eles.
    """
    fpr, tpr, limiares = roc_curve(y_val, p_val, drop_intermediate=False)
    validos = limiares[(tpr >= recall_minimo) & np.isfinite(limiares)]
    return float(validos.max()) if len(validos) else 0.5


def bootstrap_diferenca_auc(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_amostras: int = 1000,
    semente: int = config.SEMENTE,
) -> dict[str, float]:
    """Diferença de AUC (A − B) com IC 95 % por bootstrap **pareado**.

    Os dois modelos são avaliados nos mesmos pacientes reamostrados. Assim a
    variação que vem da amostra (pacientes fáceis ou difíceis) se cancela, e o
    intervalo mede só a diferença entre os modelos. Intervalos calculados
    separadamente para cada modelo se sobrepõem mesmo quando a diferença é real.
    """
    rng = np.random.default_rng(semente)
    n = len(y)
    diferencas = []
    for _ in range(n_amostras):
        i = rng.integers(0, n, n)
        diferencas.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    inf, sup = np.percentile(diferencas, [2.5, 97.5])
    return {
        "diferenca": roc_auc_score(y, p_a) - roc_auc_score(y, p_b),
        "ic95_inf": float(inf),
        "ic95_sup": float(sup),
        "significativa": bool(inf > 0 or sup < 0),
    }


def tabela_comparativa(resultados: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Junta os dicionários de :func:`metricas` em uma tabela ordenada pela AUC."""
    tabela = pd.DataFrame(resultados).T
    colunas = [
        "auc_roc",
        "auc_pr",
        "acuracia",
        "sensibilidade",
        "especificidade",
        "precisao",
        "f1",
        "brier",
        "limiar",
    ]
    return tabela[colunas].astype(float).sort_values("auc_roc", ascending=False).round(4)


def contribuicoes_por_oclusao(
    prever_bruto: Callable[[pd.DataFrame], np.ndarray],
    paciente: pd.DataFrame,
    referencia: pd.DataFrame,
) -> pd.Series:
    """Quanto cada variável empurra a probabilidade deste paciente.

    Para cada variável, troca o valor do paciente pelo valor "típico" da
    referência (mediana, ou moda nas categóricas) e mede a queda na
    probabilidade. Positivo = a variável aumenta o risco deste paciente.

    É uma explicação local simples e rápida o bastante para a aplicação
    Streamlit; os notebooks usam SHAP e LIME para a análise completa.
    """
    base = float(prever_bruto(paciente)[0])
    tipico = {}
    for coluna in config.ENTRADAS:
        serie = referencia[coluna]
        if coluna in config.CONTINUAS:
            tipico[coluna] = serie.median()
        else:
            tipico[coluna] = serie.mode().iloc[0]

    variantes = pd.concat([paciente] * len(config.ENTRADAS), ignore_index=True).astype(float)
    for i, coluna in enumerate(config.ENTRADAS):
        variantes.loc[i, coluna] = tipico[coluna]
    probs = prever_bruto(variantes)
    return pd.Series(base - probs, index=config.ENTRADAS).sort_values(key=np.abs)
