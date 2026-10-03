"""Fluxo de triagem em LangGraph: cada passo é um nó, e o caminho é explícito.

O agente ReAct (``agente.py``) deixa o LLM decidir a ordem das ferramentas. Aqui o
caminho é fixo e auditável — o LLM só **extrai** dados do texto e **redige** a
explicação; validar e calcular são código (specs/01-triagem.md, FLX-01 a FLX-05)::

    START → interpretar ─┬─ pergunta sobre o modelo → responder_pergunta → END
                         └─ triagem → validar ─┬─ faltam dados → perguntar → END
                                               ├─ valor inválido → corrigir → END
                                               └─ completo → avaliar → explicar → END

Sem chave de API, o fluxo roda em **modo demonstração**: a extração usa regras
(expressões regulares) e a explicação, um modelo de texto. Os mesmos nós, a mesma
validação e a mesma rede neural — só sem LLM e sem custo.

Os dados do paciente se acumulam entre mensagens (``parcial``): a pessoa pode
mandar o que faltou depois, ou perguntar "e se a pressão fosse 12 por 8?".
"""

from __future__ import annotations

import json
import operator
import re
import unicodedata
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from src.agente.esquemas import NIVEIS, Paciente, PacienteParcial, nivel, validar
from src.agente.ferramentas import avaliar_validado, buscar_achados, desempenho
from src.utils.formato import numero, pct

LIMITE_MENSAGEM = 1000

# --------------------------------------------------------------------------- #
# Estado compartilhado entre os nós
# --------------------------------------------------------------------------- #


class Estado(TypedDict, total=False):
    mensagem: str
    parcial: dict  # o que já se sabe do paciente (acumula entre mensagens)
    intencao: Literal["triagem", "pergunta"]
    paciente: dict
    faltando: list[str]
    erros: list[str]
    resultado: dict
    resposta: str
    rastreio: Annotated[list[str], operator.add]  # cada nó acrescenta uma linha


class Extracao(BaseModel):
    """O que o LLM devolve no nó ``interpretar`` (saída estruturada)."""

    intencao: Literal["triagem", "pergunta"] = Field(
        description="triagem: descreve ou altera dados de um paciente; "
        "pergunta: dúvida sobre o modelo, o limiar, a base ou os resultados"
    )
    paciente: PacienteParcial = Field(
        default_factory=PacienteParcial,
        description="Somente os dados ditos NESTA mensagem; o resto fica null",
    )


# --------------------------------------------------------------------------- #
# Extração por regras (modo demonstração)
# --------------------------------------------------------------------------- #


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


_ACIMA = r"(?:alt[oa]s?|elevad[oa]s?|acima(?:\s+do\s+normal)?)"
_NIVEL = rf"(muito\s+{_ACIMA}|{_ACIMA}|limitrofe|normais|normal)"
_PRESSAO = r"(\d{1,3}(?:[.,]\d)?\s*(?:por|x|/)\s*\d{1,3}(?:[.,]\d)?)"
#: Sem a palavra "pressão" ou "PA", só "por" e "x" contam: "12/08" sozinho pode ser uma data.
_PRESSAO_SOLTA = r"(\d{1,3}(?:[.,]\d)?\s*(?:por|x)\s*\d{1,3}(?:[.,]\d)?)"
#: Parentes: o que eles fazem ("o marido fuma") não é dado do paciente.
_TERCEIRO = r"\b(?:o |a )?(?:marido|esposa|companheir[oa]|pai|mae|filh[oa]|irma[oa])\b"


def _idade(t: str) -> int | None:
    """A idade do paciente, não a de um parente nem uma duração ("fumou por 10 anos")."""
    candidatos = []
    for m in re.finditer(r"(\d{2,3})\s*anos", t):
        antes = t[max(0, m.start() - 25) : m.start()]
        if re.search(r"\b(por|ha|faz|durante|aos|desde|mais de)\s*$", antes):
            continue  # duração ou idade em que algo aconteceu
        if re.search(_TERCEIRO + r"\s+(de|com|tem)\s*$", antes):
            continue  # idade de um parente
        preferido = bool(
            re.search(r"(paciente|idade|tem|com|homem|mulher|senhora?)\D{0,12}$", antes)
        )
        candidatos.append((not preferido, m.start(), int(m.group(1))))
    return min(candidatos)[2] if candidatos else None


_PERGUNTA = (
    "modelo", "rede", "confiavel", "confia", "acerta", "auc", "sensibilidade", "especificidade",
    "limiar", "por que", "porque", "como funciona", "protege", "base de dados", "limitac",
)  # fmt: skip


def extrair_por_regras(texto: str) -> Extracao:
    """Extrai intenção e dados de um texto em português com expressões regulares."""
    t = _sem_acento(texto)
    d: dict = {}

    if (idade := _idade(t)) is not None:
        d["idade"] = idade
    if re.search(r"\b(homem|masculino|senhor)\b", t):
        d["sexo"] = "masculino"
    elif re.search(r"\b(mulher|feminino|senhora)\b", t):
        d["sexo"] = "feminino"
    if m := re.search(r"\b(\d[.,]\d{1,2})\s*(?:m\b|metro)", t):
        d["altura_cm"] = float(m.group(1).replace(",", "."))  # o esquema converte para cm
    elif m := re.search(r"altura\D{0,10}(\d[.,]\d{1,2})\b", t):
        d["altura_cm"] = float(m.group(1).replace(",", "."))
    elif m := re.search(r"(\d{3})\s*cm|altura\D{0,10}(\d{3})", t):
        d["altura_cm"] = int(m.group(1) or m.group(2))
    if m := re.search(r"(\d{2,3}(?:[.,]\d)?)\s*(?:kg|quilo|kilo)|peso\D{0,10}(\d{2,3})", t):
        d["peso_kg"] = float((m.group(1) or m.group(2)).replace(",", "."))
    if m := re.search(r"(?:pressao|\bpa\b)\D{0,20}" + _PRESSAO, t) or re.search(_PRESSAO_SOLTA, t):
        d["pressao_texto"] = m.group(1)

    if m := re.search(r"colesterol\s+e\s+glicose\s+(?:estao\s+)?" + _NIVEL, t):
        d["colesterol"] = d["glicose"] = nivel(m.group(1))
    else:
        for campo, padrao in (
            ("colesterol", r"colesterol(?:\s+(?:total|ldl))?"),
            ("glicose", r"(?:glicose|glicemia)(?:\s+de\s+jejum)?"),
        ):
            # Negação antes ("não está com colesterol alto", "nega", "sem") ou depois do nome
            # ("colesterol não está alto"), sem atravessar vírgula ou ponto. Negar "alto" =
            # normal; negar "muito alto" não diz o nível, então o campo fica em aberto.
            negacao = re.search(
                rf"\b(?:nao|nega\w*|sem)\b[^.,;]{{0,30}}?(?:{padrao})\s*(?:esta\s*|e\s*|:\s*)?"
                rf"(muito\s+)?{_ACIMA}"
                rf"|(?:{padrao})\s+nao\s+(?:esta\s+|e\s+)?(muito\s+)?{_ACIMA}",
                t,
            )
            if negacao:
                if not (negacao.group(1) or negacao.group(2)):
                    d[campo] = 1
            elif m := re.search(rf"(?:{padrao})\s*(?:esta\s*|e\s*|:\s*)?" + _NIVEL, t):
                d[campo] = nivel(m.group(1))

    # Hábitos de parentes saem do texto antes de procurar os do paciente.
    habitos = re.sub(_TERCEIRO + r"\s+(?:dela |dele )?(?:fuma|bebe|e fumante)\b", " ", t)

    if re.search(
        r"(nao|nunca)\s+(e\s+)?(fuma|fumante)|ex-?fumante|parou de fumar|parasse de fumar"
        r"|\bfumou\b.{0,30}\bparou|\bnega\w*\s+(?:o\s+)?(?:tabagismo|fumo)",
        habitos,
    ):
        d["fumante"] = False
    elif re.search(r"\b(fuma|fumante|tabagista)\b", habitos):
        d["fumante"] = True
    if re.search(
        r"(nao|nunca)\s+(bebe|bebeu|consome alcool|ingere alcool)|abstemi|parasse de beber"
        r"|\bnega\w*\s+[^.,;]{0,20}\b(?:etilismo|alcool)",
        habitos,
    ):
        d["consome_alcool"] = False
    elif re.search(r"\b(bebe|etilista|consome alcool)\b", habitos):
        d["consome_alcool"] = True
    # "ativo" sozinho não basta ("caso ativo", "princípio ativo"): só conta com o contexto
    # de exercício — "fisicamente ativo", "pratica esporte", "faz caminhada", "academia".
    if re.search(
        r"sedentari|inativ|nao\s+(pratica|faz)\s+(atividade|exercicio)"
        r"|nao\s+e\s+(fisicamente\s+)?ativ[oa]"
        r"|(parasse|parou|deixasse|deixou)\s+(a|de)\s+(caminhada|caminhar|malhar|treinar|academia)",
        t,
    ):
        d["fisicamente_ativo"] = False
    elif re.search(
        r"fisicamente ativ|\bativ[oa] fisicamente|atividade fisica|academia"
        r"|(pratica|faz)\s+(atividade|exercicio|esporte|musculacao)|\bse exercita"
        r"|\bcaminha(?:da|r)?\b",
        t,
    ):
        d["fisicamente_ativo"] = True

    # Um valor que o esquema recusa é descartado, não derruba a extração: o fluxo
    # pergunta por ele depois.
    validos = {}
    for campo, valor in d.items():
        try:
            PacienteParcial.model_validate({campo: valor})
            validos[campo] = valor
        except ValidationError:
            pass
    parcial = PacienteParcial.model_validate(validos)
    pergunta = not d and any(p in t for p in _PERGUNTA)
    return Extracao(intencao="pergunta" if pergunta else "triagem", paciente=parcial)


# --------------------------------------------------------------------------- #
# Textos fixos (modo demonstração) e instruções do LLM
# --------------------------------------------------------------------------- #

AVISO_DEMO = (
    "_[Modo demonstração — sem IA: dados extraídos por regras e resposta montada por "
    "modelo de texto.]_"
)

INSTRUCOES_EXTRACAO = """\
Extraia dados de um paciente de uma mensagem em português do Brasil, para triagem cardiovascular.
- Preencha SOMENTE o que a mensagem diz; o resto fica null. Não invente nem deduza.
- pressao_texto: copie o trecho da pressão exatamente como escrito ("15 por 9,5", "120/80").
- altura em metros ("1,72 m") ou centímetros; peso em kg; idade em anos.
- colesterol e glicose: 1 normal, 2 acima do normal/alto, 3 muito acima/muito alto.
- "sedentário" = fisicamente_ativo false. "parasse de fumar" = fumante false.
- Perguntas hipotéticas sobre o paciente em andamento ("e se a pressão fosse 12 por 8?",
  "e se ela parasse a caminhada?") são triagem: extraia só o valor que mudaria.
- intencao = pergunta somente para dúvidas sobre o modelo, o limiar, a base ou os resultados,
  sem dados de paciente. "Devo encaminhar?" junto com dados é triagem.
"""

INSTRUCOES_EXPLICACAO = """\
Você explica, em português do Brasil, o resultado de uma rede neural de triagem cardiovascular
para uma equipe de saúde. Use SOMENTE os dados do JSON recebido; não invente números.
1. Comece com "Dados usados:" e copie `dados_usados` como está.
2. Dê a probabilidade e a decisão de triagem, comparadas ao limiar, como vierem formatadas.
3. Cite os fatores de `fatores`, na ordem e com o sentido que vierem. Não fale de nenhuma
   variável fora dessa lista.
4. Se `nota_fumo_alcool` vier preenchida, inclua-a. Repita os avisos, se houver.
5. Termine lembrando que a decisão é da equipe clínica. Seja conciso (até 8 linhas).
"""

INSTRUCOES_PERGUNTA = """\
Responda, em português do Brasil e em até 6 linhas, a uma pergunta sobre um modelo de triagem
cardiovascular. Use SOMENTE os achados e métricas do JSON recebido. Se não houver informação
suficiente, diga isso.
"""


def _descrever_paciente(p: Paciente) -> str:
    return (
        f"{numero(p.idade, compacto=True)} anos, {p.sexo}, "
        f"{numero(p.altura_cm, compacto=True)} cm, {numero(p.peso_kg, compacto=True)} kg, "
        f"pressão {p.pressao_sistolica}/{p.pressao_diastolica}, colesterol {NIVEIS[p.colesterol]}, "
        f"glicose {NIVEIS[p.glicose]}, fuma: {'sim' if p.fumante else 'não'}, "
        f"álcool: {'sim' if p.consome_alcool else 'não'}, "
        f"fisicamente ativo: {'sim' if p.fisicamente_ativo else 'não'}"
    )


def _fatores_legiveis(r: dict, n: int = 3) -> list[str]:
    """'pressão sistólica (150): +22,5 p.p.' para os n primeiros fatores."""

    def efeito(f: dict) -> str:
        # Vírgula decimal só no número do efeito: o valor do paciente (ex.: IMC 27.5)
        # e o "p.p." têm pontos próprios que não podem ser trocados.
        return numero(f["efeito_pontos_percentuais"], 1, sinal=True)

    return [
        f"{re.sub(r'\s*\(.*?\)', '', f['variavel']).lower()} ({f['valor_do_paciente']}): "
        f"{efeito(f)} p.p. ({f['sentido']})"
        for f in r["fatores_principais"][:n]
    ]


def _para_o_llm(p: Paciente, r: dict) -> dict:
    """Resultado já formatado: o LLM só redige, sem converter códigos nem números."""
    fumo_reduz = any(
        f["variavel"] in {"Fumante", "Consome álcool"} and f["efeito_pontos_percentuais"] < 0
        for f in r["fatores_principais"][:3]
    )
    return {
        "dados_usados": _descrever_paciente(p),
        "probabilidade": f"{r['probabilidade']:.0%}",
        "limiar_de_triagem": f"{r['limiar_de_triagem']:.0%}",
        "decisao": r["decisao"],
        "fatores": _fatores_legiveis(r),
        "nota_fumo_alcool": (
            "Fumo/álcool aparecem reduzindo o risco: é um artefato desta base, não um efeito "
            "protetor; continuam sendo fatores de risco reais."
            if fumo_reduz
            else ""
        ),
        "avisos": r["avisos"],
    }


def _explicacao_por_modelo(p: Paciente, r: dict) -> str:
    fatores = "; ".join(_fatores_legiveis(r))
    linhas = [
        AVISO_DEMO,
        f"**Dados usados:** {_descrever_paciente(p)}.",
        f"**Probabilidade:** {r['probabilidade']:.0%} → **{r['decisao']}** "
        f"(limiar de triagem {r['limiar_de_triagem']:.0%}).",
        f"**Fatores que mais pesaram:** {fatores}" if fatores else "",
        *[f"⚠️ {aviso}" for aviso in r["avisos"]],
        "A decisão final é da equipe clínica.",
    ]
    return "\n\n".join(linha for linha in linhas if linha)


# --------------------------------------------------------------------------- #
# Montagem do grafo
# --------------------------------------------------------------------------- #


def construir_fluxo(llm: BaseChatModel | None = None, tentativas: int = 2):
    """Compila o grafo. Sem ``llm``, roda em modo demonstração."""
    modo = "llm" if llm is not None else "demo"
    extrator = (
        llm.with_structured_output(Extracao, method="function_calling", include_raw=True)
        if llm
        else None
    )

    def interpretar(estado: Estado) -> dict:
        texto = estado["mensagem"][:LIMITE_MENSAGEM]
        if extrator is None:
            extracao = extrair_por_regras(texto)
        else:
            # Como no exemplo da disciplina: se a saída não passar no Pydantic, o erro volta
            # ao LLM para ele corrigir, até `tentativas` vezes.
            contexto = INSTRUCOES_EXTRACAO
            if estado.get("parcial"):
                contexto += f"\nPaciente em andamento (já conhecido): {estado['parcial']}"
            mensagens = [SystemMessage(contexto), HumanMessage(texto)]
            extracao, ultimo_erro = None, None
            for _ in range(tentativas):
                saida = extrator.invoke(mensagens)
                if saida["parsed"] is not None:
                    extracao = saida["parsed"]
                    break
                ultimo_erro = saida["parsing_error"]
                mensagens.append(
                    HumanMessage(f"Saída rejeitada pela validação. Corrija:\n{ultimo_erro}")
                )
            if extracao is None:
                raise ValueError(f"Extração inválida após {tentativas} tentativas: {ultimo_erro}")
        novos = extracao.paciente.model_dump(exclude_none=True)
        # Regra em código, não no LLM: se a mensagem trouxe dados, é triagem (FLX-02).
        intencao = "triagem" if novos else extracao.intencao
        parcial = PacienteParcial.model_validate(estado.get("parcial") or {}).juntar(
            extracao.paciente
        )
        return {
            "intencao": intencao,
            "parcial": parcial.model_dump(exclude_none=True),
            "rastreio": [
                f"interpretar ({'LLM' if extrator else 'regras'}) → {intencao}; "
                f"campos nesta mensagem: {', '.join(novos) or 'nenhum'}"
            ],
        }

    def validar_dados(estado: Estado) -> dict:
        paciente, faltando, erros = validar(PacienteParcial.model_validate(estado["parcial"]))
        if paciente is not None:
            situacao = "completo"
        elif erros:
            situacao = "inválido: " + "; ".join(erros)
        else:
            situacao = "faltando: " + ", ".join(faltando)
        return {
            "paciente": paciente.model_dump() if paciente else {},
            "faltando": faltando,
            "erros": erros,
            "rastreio": [f"validar (Pydantic) → {situacao}"],
        }

    def perguntar(estado: Estado) -> dict:
        lista = "\n".join(f"- {campo}" for campo in estado["faltando"])
        resposta = f"Para calcular o risco, ainda preciso de:\n{lista}"
        if modo == "demo":
            resposta = f"{AVISO_DEMO}\n\n{resposta}"
        return {"resposta": resposta, "rastreio": ["perguntar (modelo de texto) → dados faltando"]}

    def corrigir(estado: Estado) -> dict:
        lista = "\n".join(f"- {erro}" for erro in estado["erros"])
        return {
            "resposta": f"Não consegui usar estes valores:\n{lista}\n\nPode confirmar?",
            "rastreio": ["corrigir (modelo de texto) → pede confirmação"],
        }

    def avaliar(estado: Estado) -> dict:
        resultado = avaliar_validado(Paciente.model_validate(estado["paciente"]))
        return {
            "resultado": resultado,
            "rastreio": [
                f"avaliar (rede neural) → P = {pct(resultado['probabilidade'], 1)}, "
                f"{resultado['decisao']}"
            ],
        }

    def explicar(estado: Estado) -> dict:
        paciente = Paciente.model_validate(estado["paciente"])
        if llm is None:
            texto = _explicacao_por_modelo(paciente, estado["resultado"])
        else:
            dados = _para_o_llm(paciente, estado["resultado"])
            texto = llm.invoke(
                [
                    SystemMessage(INSTRUCOES_EXPLICACAO),
                    HumanMessage(json.dumps(dados, ensure_ascii=False)),
                ]
            ).content
        return {
            "resposta": texto,
            "rastreio": [f"explicar ({'LLM' if llm else 'modelo de texto'})"],
        }

    def responder_pergunta(estado: Estado) -> dict:
        achados = buscar_achados(estado["mensagem"])
        if llm is None:
            texto = "\n\n".join(
                [AVISO_DEMO, *(f"**{a['tema'].capitalize()}:** {a['achado']}" for a in achados)]
            )
        else:
            dados = {"pergunta": estado["mensagem"], "achados": achados, "metricas": desempenho()}
            texto = llm.invoke(
                [
                    SystemMessage(INSTRUCOES_PERGUNTA),
                    HumanMessage(json.dumps(dados, ensure_ascii=False)),
                ]
            ).content
        temas = ", ".join(a["tema"] for a in achados)
        return {"resposta": texto, "rastreio": [f"responder_pergunta → achados: {temas}"]}

    grafo = StateGraph(Estado)
    for nome, funcao in [
        ("interpretar", interpretar),
        ("validar", validar_dados),
        ("perguntar", perguntar),
        ("corrigir", corrigir),
        ("avaliar", avaliar),
        ("explicar", explicar),
        ("responder_pergunta", responder_pergunta),
    ]:
        grafo.add_node(nome, funcao)
    grafo.add_edge(START, "interpretar")
    grafo.add_conditional_edges(
        "interpretar",
        lambda e: e["intencao"],
        {"triagem": "validar", "pergunta": "responder_pergunta"},
    )
    grafo.add_conditional_edges(
        "validar",
        lambda e: "avaliar" if e["paciente"] else ("corrigir" if e["erros"] else "perguntar"),
        {"avaliar": "avaliar", "corrigir": "corrigir", "perguntar": "perguntar"},
    )
    grafo.add_edge("avaliar", "explicar")
    for fim in ("explicar", "perguntar", "corrigir", "responder_pergunta"):
        grafo.add_edge(fim, END)
    return grafo.compile()


def triar(mensagem: str, parcial: dict | None = None, fluxo=None) -> dict:
    """Uma rodada do fluxo. Devolve resposta, dados acumulados, rastreio e resultado."""
    if fluxo is None:
        fluxo = construir_fluxo()
    estado = fluxo.invoke({"mensagem": mensagem, "parcial": parcial or {}, "rastreio": []})
    return {
        "resposta": estado["resposta"],
        "parcial": estado.get("parcial", {}),
        "rastreio": estado["rastreio"],
        "resultado": estado.get("resultado"),
    }


__all__ = [
    "Estado",
    "Extracao",
    "construir_fluxo",
    "extrair_por_regras",
    "triar",
    "ValidationError",
]
