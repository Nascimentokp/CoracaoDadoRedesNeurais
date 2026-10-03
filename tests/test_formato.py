"""Números no padrão brasileiro (APP-04)."""

from src.utils.formato import numero, pct


def test_numero_e_porcentagem_com_virgula():
    assert numero(27.53) == "27,5"
    assert numero(0.80139, 3) == "0,801"
    assert numero(1.8, sinal=True) == "+1,8"
    assert pct(0.808) == "81%"
    assert pct(0.8078, 1) == "80,8%"
    assert pct(0.004, 1, sinal=True) == "+0,4%"


def test_graficos_usam_locale_brasileiro():
    import numpy as np

    from src.deployment import graficos_app as graf

    spec = graf.posicao_na_populacao(np.random.default_rng(0).random(50), 0.5, 0.38).to_dict()
    assert spec["config"]["locale"]["number"]["decimal"] == ","
