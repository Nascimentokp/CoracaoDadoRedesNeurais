"""Fluxo LangGraph de triagem (specs/01-triagem.md, FLX-01 a FLX-05, AGT-01 a AGT-04).

O modo demonstração roda o grafo inteiro sem API. O modo LLM é testado com um
modelo roteirizado, sem custo.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.agente.fluxo import _fatores_legiveis, construir_fluxo, extrair_por_regras, triar
from src.model import treino

pytestmark = pytest.mark.skipif(
    not treino.ARQUIVO_MODELO.exists(), reason="modelo não treinado (uv run invoke treinar)"
)

COMPLETO = (
    "Homem de 58 anos, 1,72 m e 94 kg, pressão 15 por 9,5, colesterol acima do normal, "
    "glicose normal, fuma, não bebe e é sedentário."
)


@pytest.fixture(scope="module")
def demo():
    return construir_fluxo()


def _nos(rastreio):
    return [linha.split(" ")[0] for linha in rastreio]


def test_caminho_completo(demo):
    """FLX-01 / AGT-01 / AGT-03: interpretar → validar → avaliar → explicar, em modo demo."""
    r = triar(COMPLETO, fluxo=demo)
    assert _nos(r["rastreio"]) == ["interpretar", "validar", "avaliar", "explicar"]
    assert 0 < r["resultado"]["probabilidade"] < 1
    assert "Modo demonstração" in r["resposta"]
    assert "150/95" in r["resposta"]  # FLX-03: pressão convertida em código


def test_faltando_dados_pergunta_sem_chamar_a_rede(demo):
    """FLX-05."""
    r = triar("Mulher, 61 anos, pressão 14 por 9. Devo encaminhar?", fluxo=demo)
    assert _nos(r["rastreio"]) == ["interpretar", "validar", "perguntar"]
    assert r["resultado"] is None
    assert "peso" in r["resposta"]


def test_dados_acumulam_e_e_se_recalcula(demo):
    """FLX-04."""
    r = triar("Mulher, 61 anos, pressão 14 por 9.", fluxo=demo)
    r = triar(
        "1,60 m, 70 kg, colesterol e glicose normais, não fuma, não bebe, faz caminhada",
        r["parcial"],
        fluxo=demo,
    )
    antes = r["resultado"]["probabilidade"]
    r = triar("e se a pressão fosse 12 por 8?", r["parcial"], fluxo=demo)
    assert r["resultado"]["probabilidade"] < antes
    assert r["parcial"]["idade"] == 61  # o resto do paciente foi mantido


def test_valor_incoerente_pede_correcao(demo):
    """VAL-01 no fluxo: diastólica ≥ sistólica não chega à rede."""
    r = triar(COMPLETO.replace("15 por 9,5", "12 por 13"), fluxo=demo)
    assert _nos(r["rastreio"])[-1] == "corrigir"
    assert r["resultado"] is None


def test_pergunta_sobre_o_modelo(demo):
    """FLX-01: dúvidas vão para responder_pergunta."""
    r = triar("O modelo diz que fumar reduz o risco? Isso está certo?", fluxo=demo)
    assert _nos(r["rastreio"]) == ["interpretar", "responder_pergunta"]
    assert "artefato" in r["resposta"]


@pytest.mark.parametrize(
    "texto, campo, valor",
    [
        ("pressão 12x8", "pressao_texto", "12x8"),
        ("colesterol e glicose normais", "colesterol", 1),
        ("colesterol muito alto", "colesterol", 3),
        ("ela é sedentária", "fisicamente_ativo", False),
        ("devo encaminhar?", "fisicamente_ativo", None),  # "encaminhar" não é "caminha"
        ("caso ativo de hipertensão", "fisicamente_ativo", None),  # "ativo" sozinho não é exercício
        ("princípio ativo do remédio", "fisicamente_ativo", None),
        ("é fisicamente ativa", "fisicamente_ativo", True),
        ("faz musculação três vezes por semana", "fisicamente_ativo", True),
        ("não é fisicamente ativo", "fisicamente_ativo", False),
        ("e se ela parasse a caminhada?", "fisicamente_ativo", False),
        # idade do paciente, não duração nem idade de parente
        ("fumou por 10 anos mas parou", "idade", None),
        ("fumou por 10 anos mas parou", "fumante", False),
        ("paciente de 45 anos, mãe faleceu aos 70 anos", "idade", 45),
        ("filho de 30 anos acompanha; paciente com 62 anos", "idade", 62),
        # hábitos de parentes não são do paciente
        ("o marido fuma, ela não", "fumante", None),
        ("nunca bebeu", "consome_alcool", False),
        # data não é pressão; "/" só conta com a palavra pressão
        ("homem, 50 anos, data 12/08", "pressao_texto", None),
        ("pressão 120/80", "pressao_texto", "120/80"),
        # negação e variações de nome
        ("não tem colesterol alto", "colesterol", 1),
        ("colesterol total alto", "colesterol", 2),
        ("glicemia de jejum normal", "glicose", 1),
        ("altura 1,75", "altura_cm", 1.75),
        ("nunca fumou, ex-fumante", "fumante", False),
        ("1,60 m", "altura_cm", 1.6),
    ],
)
def test_regras_de_extracao(texto, campo, valor):
    """VAL-02 / AGT-03: extração por regras do modo demonstração."""
    assert getattr(extrair_por_regras(texto).paciente, campo) == valor


class _LLMRoteirizado(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_saida_invalida_do_llm_volta_para_correcao():
    """AGT-04 / FLX-02: Pydantic rejeita a 1ª extração; a 2ª passa; a rede calcula."""
    valido = dict(
        idade=58, sexo="masculino", altura_cm=172, peso_kg=94, pressao_texto="15 por 9,5",
        colesterol=2, glicose=1, fumante=True, consome_alcool=False, fisicamente_ativo=False,
    )  # fmt: skip

    def extracao(id_, paciente):
        # intenção "pergunta" com dados: o código força triagem (FLX-02)
        args = {"intencao": "pergunta", "paciente": paciente}
        return AIMessage("", tool_calls=[{"name": "Extracao", "id": id_, "args": args}])

    llm = _LLMRoteirizado(
        responses=[
            extracao("1", {**valido, "colesterol": 7}),  # rejeitada pelo Pydantic
            extracao("2", valido),
            AIMessage("Explicação redigida."),
        ]
    )
    r = triar(COMPLETO, fluxo=construir_fluxo(llm))
    assert _nos(r["rastreio"]) == ["interpretar", "validar", "avaliar", "explicar"]
    assert "(LLM) → triagem" in r["rastreio"][0]
    assert r["resposta"] == "Explicação redigida."


def test_virgula_decimal_so_no_numero_do_efeito():
    """AGT-02: o valor do paciente com decimal não rouba a vírgula do efeito."""
    fator = {"variavel": "IMC", "valor_do_paciente": "27.5", "efeito_pontos_percentuais": 1.8,
             "sentido": "aumenta o risco"}  # fmt: skip
    assert _fatores_legiveis({"fatores_principais": [fator]}) == [
        "imc (27.5): +1,8 p.p. (aumenta o risco)"
    ]
