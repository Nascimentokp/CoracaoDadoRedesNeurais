"""Gráficos da aplicação: cada função gera uma especificação Vega-Lite válida."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.deployment import graficos_app as graf

LIMIAR = 0.38


def _valido(grafico) -> dict:
    return grafico.to_dict(validate=True)  # levanta erro se a especificação for inválida


@pytest.fixture
def curva():
    ap_hi = np.arange(90, 201, 2.5)
    return pd.DataFrame({"ap_hi": ap_hi, "probabilidade": 1 / (1 + np.exp(-(ap_hi - 135) / 8))})


def test_pressao_de_cruzamento(curva):
    assert graf.pressao_de_cruzamento(curva, 0.5) == pytest.approx(135, abs=2.5)
    assert graf.pressao_de_cruzamento(curva, 1.1) is None


def test_graficos_do_paciente(curva):
    contrib = pd.Series([0.2, -0.05], index=["ap_hi", "active"])
    _valido(graf.fatores(contrib))
    _valido(graf.e_se_pressao(curva, 150, 0.85, LIMIAR))
    _valido(graf.posicao_na_populacao(np.random.default_rng(0).random(500), 0.7, LIMIAR))
    pa, idade = np.meshgrid(np.arange(90, 201, 5.0), np.arange(30, 66, 1.0))
    grade = pd.DataFrame({"ap_hi": pa.ravel(), "idade_anos": idade.ravel()})
    grade["probabilidade"] = np.clip((grade["ap_hi"] - 90) / 110, 0, 1)
    _valido(graf.mapa_pressao_idade(grade, 150, 52, LIMIAR))


def test_graficos_de_desempenho():
    limiares = np.round(np.arange(0.05, 0.951, 0.05), 2)
    varredura = pd.DataFrame(
        {
            "limiar": limiares,
            "Sensibilidade": 1 - limiares,
            "Especificidade": limiares,
            "Precisão": 0.7,
        }
    )
    _valido(graf.curvas_limiar(varredura, LIMIAR))
    _valido(graf.matriz_confusao(5000, 2500, 1300, 4900))
    pontos = pd.DataFrame(
        {
            "modelo": ["A"] * 3 + ["B"] * 3,
            "previsto": [0.1, 0.5, 0.9] * 2,
            "real": [0.12, 0.5, 0.85] * 2,
        }
    )
    _valido(graf.calibracao(pontos))
    ic = pd.DataFrame(
        {"auc": [0.80, 0.79], "ic95_inf": [0.79, 0.78], "ic95_sup": [0.81, 0.80]},
        index=["MLP Keras", "Regressão logística"],
    )
    _valido(graf.comparacao_modelos(ic))
    perm = pd.DataFrame(
        {"queda_auc": [0.18, 0.03]}, index=pd.Index(["Pressão sistólica", "Idade"], name="variavel")
    )
    _valido(graf.importancia(perm))


def test_matriz_confusao_mostra_os_quatro_tipos():
    spec = _valido(graf.matriz_confusao(1, 2, 3, 4))
    valores = next(iter(spec["datasets"].values()))
    assert {v["significado"] for v in valores} == {
        "doentes encontrados",
        "doentes perdidos",
        "alarmes falsos",
        "saudáveis liberados",
    }


def test_rotulos_usados_existem():
    assert set(config.ROTULOS) >= {"ap_hi", "active"}
