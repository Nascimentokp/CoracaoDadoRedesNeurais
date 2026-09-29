"""Rede em NumPy, redes em Keras/PyTorch e utilitários de avaliação."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import avaliacao
from src.model.mlp_numpy import MLPNumpy, entropia_cruzada_binaria


def _dados_xor(n=800, semente=0):
    """Problema não linear clássico: um neurônio sozinho não resolve."""
    rng = np.random.default_rng(semente)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32)
    return X, y


def test_gradiente_analitico_bate_com_numerico():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 5))
    y = (rng.random(32) > 0.5).astype(float)
    rede = MLPNumpy(n_entradas=5, ocultas=(6, 4), l2=0.0, semente=3)
    rede.pesos = [W.astype(np.float64) for W in rede.pesos]
    rede.vieses = [b.astype(np.float64) for b in rede.vieses]

    pre, ativ = rede._forward(X)
    grad_W, _ = rede._backward(pre, ativ, y)

    eps = 1e-6
    for camada in range(len(rede.pesos)):
        original = rede.pesos[camada][0, 0]
        rede.pesos[camada][0, 0] = original + eps
        mais = entropia_cruzada_binaria(y, rede.predict_proba(X))
        rede.pesos[camada][0, 0] = original - eps
        menos = entropia_cruzada_binaria(y, rede.predict_proba(X))
        rede.pesos[camada][0, 0] = original
        numerico = (mais - menos) / (2 * eps)
        assert abs(numerico - grad_W[camada][0, 0]) < 1e-6


def test_mlp_numpy_aprende_xor():
    X, y = _dados_xor()
    rede = MLPNumpy(n_entradas=2, ocultas=(16, 8), taxa=0.3, l2=0.0)
    rede.fit(X, y, epocas=300, lote=64)
    acuracia = ((rede.predict_proba(X) >= 0.5) == y).mean()
    assert acuracia > 0.9


def test_keras_e_torch_aprendem_xor():
    from src.model import mlp_keras, mlp_torch

    X, y = _dados_xor()
    Xv, yv = _dados_xor(semente=1)

    keras_rede = mlp_keras.construir_mlp(2, ocultas=(16, 8), dropout=0.0, taxa=0.01)
    mlp_keras.treinar(keras_rede, X, y, Xv, yv, epocas=80, lote=64, paciencia=80)
    assert avaliacao.metricas(yv, mlp_keras.prever(keras_rede, Xv))["auc_roc"] > 0.9

    torch_rede = mlp_torch.CardioNet(2, ocultas=(16, 8), dropout=0.0)
    mlp_torch.treinar(torch_rede, X, y, Xv, yv, epocas=80, lote=64, taxa=0.01, paciencia=80)
    assert avaliacao.metricas(yv, mlp_torch.prever(torch_rede, Xv))["auc_roc"] > 0.9


def test_limiar_respeita_sensibilidade_minima():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    limiar = avaliacao.escolher_limiar(y, p, recall_minimo=0.8)
    assert avaliacao.metricas(y, p, limiar)["sensibilidade"] >= 0.8
    # E é o maior possível: nenhum escore acima dele ainda atinge 80 % de recall.
    acima = np.unique(p[p > limiar])
    assert all(avaliacao.metricas(y, p, t)["sensibilidade"] < 0.8 for t in acima[:50])


def test_limiar_nao_perde_pontos_colineares_da_roc():
    # Muitos positivos seguidos formam um trecho vertical da ROC; o limiar exato
    # em que o recall chega a 80 % fica no meio dele.
    y = np.array([0] * 10 + [1] * 10)
    p = np.concatenate([np.linspace(0.0, 0.1, 10), np.linspace(0.2, 1.0, 10)])
    limiar = avaliacao.escolher_limiar(y, p, recall_minimo=0.8)
    assert limiar == pytest.approx(p[12])  # 8 dos 10 positivos acima dele


def test_bootstrap_pareado_distingue_modelo_melhor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    ruido = rng.normal(0, 1, 3000)
    bom = y + ruido
    pior = y + ruido + rng.normal(0, 1, 3000)
    resultado = avaliacao.bootstrap_diferenca_auc(y, bom, pior, n_amostras=200)
    assert resultado["diferenca"] > 0
    assert resultado["significativa"]
    # Um modelo contra ele mesmo: diferença nula, não significativa.
    igual = avaliacao.bootstrap_diferenca_auc(y, bom, bom, n_amostras=50)
    assert igual["diferenca"] == 0
    assert not igual["significativa"]


def test_oclusao_zera_quando_paciente_e_tipico():
    referencia = pd.DataFrame([{c: 1.0 for c in config.ENTRADAS}] * 3)
    # Colunas inteiras, como chegam do formulário da aplicação.
    paciente = referencia.iloc[[0]].reset_index(drop=True).astype(int)

    def prever(df):
        return df[config.ENTRADAS].sum(axis=1).to_numpy() / 100

    contrib = avaliacao.contribuicoes_por_oclusao(prever, paciente, referencia)
    assert np.allclose(contrib, 0)


def test_oclusao_aceita_mediana_fracionaria_em_coluna_inteira():
    referencia = pd.DataFrame({c: [1, 2] for c in config.ENTRADAS})  # medianas = 1,5
    paciente = pd.DataFrame([{c: 2 for c in config.ENTRADAS}])

    def prever(df):
        return df[config.ENTRADAS].sum(axis=1).to_numpy() / 100

    contrib = avaliacao.contribuicoes_por_oclusao(prever, paciente, referencia)
    for coluna in config.CONTINUAS:
        assert contrib[coluna] == pytest.approx(0.005)
