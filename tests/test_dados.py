"""Partições e pré-processamento: formato, estratificação e ausência de vazamento."""

import numpy as np
import pytest

from src import config
from src.data.dados import carregar_base, dividir


@pytest.fixture(scope="module")
def particao():
    return dividir(carregar_base())


def test_base_tem_variaveis_derivadas():
    base = carregar_base()
    assert {"imc", "sexo_masculino"} <= set(base.columns)
    assert base["sexo_masculino"].isin([0, 1]).all()
    assert base[config.ENTRADAS + [config.ALVO]].notna().all().all()


def test_proporcoes_60_20_20(particao):
    n = len(particao.y_treino) + len(particao.y_val) + len(particao.y_teste)
    assert len(particao.y_treino) / n == pytest.approx(0.60, abs=0.01)
    assert len(particao.y_val) / n == pytest.approx(0.20, abs=0.01)
    assert len(particao.y_teste) / n == pytest.approx(0.20, abs=0.01)


def test_estratificacao(particao):
    prevalencias = [particao.y_treino.mean(), particao.y_val.mean(), particao.y_teste.mean()]
    assert max(prevalencias) - min(prevalencias) < 0.005


def test_16_entradas_float32(particao):
    # 6 contínuas + 2 ordinais × 3 níveis + 4 binárias
    assert particao.n_entradas == 16
    assert particao.X_treino.dtype == np.float32


def test_padronizacao_ajustada_so_no_treino(particao):
    continuas = particao.X_treino[:, : len(config.CONTINUAS)]
    np.testing.assert_allclose(continuas.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(continuas.std(axis=0), 1, atol=1e-3)
    # No teste, a média não é exatamente zero — sinal de que o scaler não o viu.
    assert not np.allclose(particao.X_teste[:, : len(config.CONTINUAS)].mean(axis=0), 0, atol=1e-4)
