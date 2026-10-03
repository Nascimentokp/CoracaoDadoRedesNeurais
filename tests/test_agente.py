"""Ferramentas do agente e o ciclo ReAct completo, sem chamar a API da OpenAI.

O LLM é substituído por um modelo falso com respostas roteirizadas: primeiro pede a
ferramenta, depois responde. Isso testa a ligação agente → ferramenta → rede neural
sem custo nem chave de API.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.agente import ferramentas
from src.model import treino

pytestmark = pytest.mark.skipif(
    not treino.ARQUIVO_MODELO.exists(), reason="modelo não treinado (uv run invoke treinar)"
)

PACIENTE = dict(
    idade=58,
    sexo="masculino",
    altura_cm=172,
    peso_kg=94,
    pressao_sistolica=150,
    pressao_diastolica=95,
    colesterol=2,
    glicose=1,
    fumante=True,
    consome_alcool=False,
    fisicamente_ativo=False,
)


def test_avaliar_devolve_probabilidade_e_decisao():
    resultado = ferramentas.avaliar(**PACIENTE)
    assert 0 <= resultado["probabilidade"] <= 1
    assert resultado["decisao"] in {"encaminhar para investigação", "acompanhamento de rotina"}
    assert resultado["imc"] == pytest.approx(94 / 1.72**2, abs=0.1)
    assert resultado["avisos"] == []
    fator = resultado["fatores_principais"][0]
    assert {"variavel", "valor_do_paciente", "efeito_pontos_percentuais"} <= set(fator)


def test_pressao_alta_pesa_mais_que_normal():
    alta = ferramentas.avaliar(**{**PACIENTE, "pressao_sistolica": 170})["probabilidade"]
    normal = ferramentas.avaliar(**{**PACIENTE, "pressao_sistolica": 115, "pressao_diastolica": 75})
    assert alta > normal["probabilidade"]


def test_avaliar_recusa_diastolica_maior_que_sistolica():
    resultado = ferramentas.avaliar(**{**PACIENTE, "pressao_diastolica": 160})
    assert "erro" in resultado


def test_avaliar_avisa_extrapolacao():
    resultado = ferramentas.avaliar(**{**PACIENTE, "idade": 80})
    assert any("extrapolação" in aviso for aviso in resultado["avisos"])


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("15 por 9,5", (150, 95)),
        ("15x9", (150, 90)),
        ("12 por 8", (120, 80)),
        ("150/95", (150, 95)),
        ("pressão de 14,5 por 9", (145, 90)),
    ],
)
def test_converte_pressao_no_formato_brasileiro(texto, esperado):
    resultado = ferramentas.interpretar_pressao(texto)
    assert (resultado["pressao_sistolica"], resultado["pressao_diastolica"]) == esperado


def test_pressao_sem_dois_valores_da_erro():
    assert "erro" in ferramentas.interpretar_pressao("pressão alta")


def test_achados_explicam_paradoxo_do_fumo():
    temas = [a["tema"] for a in ferramentas.buscar_achados("fumar protege o coração?")]
    assert temas[0] == "fumo e álcool"
    assert "artefato" in ferramentas.ACHADOS["fumo e álcool"]


class _LLMRoteirizado(FakeMessagesListChatModel):
    """Modelo falso que aceita ``bind_tools`` (o agente chama isso ao montar o grafo)."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_ciclo_react_chama_a_rede_e_responde():
    from src.agente.agente import conversar, criar_agente

    llm = _LLMRoteirizado(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "avaliar_paciente", "args": PACIENTE, "id": "chamada-1"}],
            ),
            AIMessage(content="Probabilidade alta: encaminhar para investigação."),
        ]
    )
    resposta, chamadas = conversar(
        criar_agente(llm), [{"role": "user", "content": "Homem, 58 anos, pressão 15 por 9..."}]
    )
    assert resposta.startswith("Probabilidade alta")
    assert len(chamadas) == 1
    assert chamadas[0]["ferramenta"] == "avaliar_paciente"
    assert '"probabilidade"' in chamadas[0]["resultado"]


def test_sem_chave_de_api_da_erro_claro(monkeypatch):
    from src.agente.agente import criar_modelo_llm

    for variavel in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_PROVEDOR"):
        monkeypatch.delenv(variavel, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        criar_modelo_llm()
