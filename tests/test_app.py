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
    """APP-01."""
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
    """APP-01 / VAL-01."""
    app = AppTest.from_file(APP, default_timeout=60).run()
    entradas = {n.label: n for n in app.number_input}
    entradas["Pressão sistólica (mmHg)"].set_value(100)
    entradas["Pressão diastólica (mmHg)"].set_value(120)
    app.button[0].click().run()
    assert not app.exception
    assert app.warning


def test_assistente_em_modo_demo_avalia_e_mostra_o_rastreio():
    """APP-03 / AGT-03: sem chave, o fluxo roda em modo demonstração."""
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.chat_input[0].set_value(
        "Homem de 58 anos, 1,72 m e 94 kg, pressão 15 por 9,5, colesterol acima do normal, "
        "glicose normal, fuma, não bebe e é sedentário."
    ).run()
    assert not app.exception
    respostas = " ".join(m.value for m in app.markdown)
    assert "Modo demonstração" in respostas
    assert "encaminhar para investigação" in respostas
    assert app.session_state.parcial_paciente["idade"] == 58


def test_assistente_acumula_dados_entre_mensagens():
    """FLX-04 no app: o que faltou pode vir na mensagem seguinte."""
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.chat_input[0].set_value("Mulher, 61 anos, pressão 14 por 9.").run()
    assert "ainda preciso" in " ".join(m.value for m in app.markdown)
    app.chat_input[0].set_value(
        "1,60 m, 70 kg, colesterol e glicose normais, não fuma, não bebe, faz caminhada"
    ).run()
    assert not app.exception
    assert "Probabilidade" in " ".join(m.value for m in app.markdown)


def _graficos(app):
    return app.get("vega_lite_chart")


def test_numeros_com_virgula_decimal():
    """APP-04: IMC, AUC e porcentagens no padrão brasileiro."""
    app = AppTest.from_file(APP, default_timeout=90).run()
    valores = {m.label: m.value for m in app.metric}
    assert valores["IMC"] == "27,5"  # 75 kg, 165 cm (valores padrão do formulário)
    assert valores["AUC-ROC"].startswith("0,")
    assert "," in valores["Sensibilidade"] and "." not in valores["Sensibilidade"]


def test_app_mostra_todos_os_graficos():
    """APP-01."""
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    # Paciente: e se, população, fatores, mapa · Desempenho: matriz, curvas, calibração,
    # comparação, importância.
    assert len(_graficos(app)) == 9


def test_slider_do_limiar_atualiza_metricas():
    """APP-02."""
    app = AppTest.from_file(APP, default_timeout=90).run()

    def sensibilidade_explorada() -> float:
        # A métrica do controle é a que traz a comparação ("vs. triagem") no delta.
        metrica = next(m for m in app.metric if m.label == "Sensibilidade" and m.delta)
        return float(metrica.value.rstrip("%").replace(",", "."))

    assert next(m for m in app.metric if m.label == "Sensibilidade" and m.delta).delta.startswith(
        "+0,0%"
    )
    antes = sensibilidade_explorada()
    app.slider(key="limiar_explorado").set_value(0.2).run()
    assert not app.exception
    assert sensibilidade_explorada() > antes
