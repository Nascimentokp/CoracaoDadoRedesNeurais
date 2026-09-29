"""Teste de fumaça da aplicação Streamlit: carrega, prevê e explica sem erro."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.model import treino

pytestmark = pytest.mark.skipif(
    not treino.ARQUIVO_MODELO.exists(), reason="modelo não treinado (uv run invoke treinar)"
)

APP = str(config.RAIZ / "src" / "deployment" / "app.py")


def test_app_abre_e_mostra_previsao():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Probabilidade" in m.label for m in app.metric)
    assert app.success or app.error  # decisão de triagem exibida


def test_app_paciente_de_alto_risco_e_encaminhado():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.slider[0].set_value(64)
    entradas = {n.label: n for n in app.number_input}
    entradas["Pressão sistólica (mmHg)"].set_value(180)
    entradas["Pressão diastólica (mmHg)"].set_value(110)
    app.selectbox[0].set_value("Muito acima do normal")
    app.button[0].click().run()
    assert not app.exception
    assert any("Encaminhar" in e.value for e in app.error)


def test_app_rejeita_diastolica_maior_que_sistolica():
    app = AppTest.from_file(APP, default_timeout=60).run()
    entradas = {n.label: n for n in app.number_input}
    entradas["Pressão sistólica (mmHg)"].set_value(100)
    entradas["Pressão diastólica (mmHg)"].set_value(120)
    app.button[0].click().run()
    assert not app.exception
    assert app.warning


def test_aba_assistente_orienta_quando_falta_chave(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("OPENAI_API_KEY" in i.value for i in app.info)


def _graficos(app):
    return app.get("vega_lite_chart")


def test_app_mostra_todos_os_graficos():
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    # Paciente: e se, população, fatores, mapa · Desempenho: matriz, curvas, calibração,
    # comparação, importância.
    assert len(_graficos(app)) == 9


def test_slider_do_limiar_atualiza_metricas():
    app = AppTest.from_file(APP, default_timeout=90).run()

    def sensibilidade_explorada() -> float:
        # A métrica do controle é a que traz a comparação ("vs. triagem") no delta.
        metrica = next(m for m in app.metric if m.label == "Sensibilidade" and m.delta)
        return float(metrica.value.rstrip("%"))

    assert next(m for m in app.metric if m.label == "Sensibilidade" and m.delta).delta.startswith(
        "+0.0%"
    )
    antes = sensibilidade_explorada()
    app.slider(key="limiar_explorado").set_value(0.2).run()
    assert not app.exception
    assert sensibilidade_explorada() > antes
