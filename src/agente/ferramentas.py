"""Ferramentas que o agente pode chamar.

O LLM não calcula risco nenhum: ele conversa, extrai os dados do texto e
decide **qual ferramenta** chamar. Os números vêm sempre da rede neural
treinada (notebook 05) e dos resultados gravados em `reports/`. É o padrão
ReAct dos slides: raciocinar → agir com uma ferramenta → observar → responder.

Cada ferramenta é uma função Python comum (testável sem API) embrulhada com
``@tool`` do LangChain; a docstring e os tipos viram a descrição que o LLM lê.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from src import config
from src.data.dados import derivar_variaveis
from src.model import avaliacao, treino

#: Faixas vistas no treino. Fora delas a rede extrapola, e o agente avisa.
FAIXAS = {
    "idade": (30, 65),
    "altura_cm": (130, 207),
    "peso_kg": (30, 183),
    "pressao_sistolica": (60, 240),
    "pressao_diastolica": (40, 160),
}

NIVEIS = {1: "normal", 2: "acima do normal", 3: "muito acima do normal"}


@lru_cache(maxsize=1)
def _modelo():
    return treino.carregar_modelo_final()


def _descrever(variavel: str, valor) -> str:
    """Valor legível de uma variável (ex.: 'sim', 'acima do normal', '150')."""
    if variavel in config.ORDINAIS:
        return NIVEIS[int(valor)]
    if variavel in config.BINARIAS:
        return "sim" if int(valor) else "não"
    return f"{float(valor):.1f}".removesuffix(".0")


def avaliar(
    idade: float,
    sexo: Literal["feminino", "masculino"],
    altura_cm: float,
    peso_kg: float,
    pressao_sistolica: int,
    pressao_diastolica: int,
    colesterol: Literal[1, 2, 3],
    glicose: Literal[1, 2, 3],
    fumante: bool,
    consome_alcool: bool,
    fisicamente_ativo: bool,
) -> dict:
    """Probabilidade prevista, decisão de triagem e fatores que mais pesaram."""
    if pressao_diastolica >= pressao_sistolica:
        return {"erro": "A diastólica precisa ser menor que a sistólica. Confirme os valores."}
    if colesterol not in NIVEIS or glicose not in NIVEIS:
        return {"erro": "Colesterol e glicose vão de 1 (normal) a 3 (muito acima do normal)."}

    valores = locals()
    avisos = [
        f"{nome} = {valores[nome]} está fora da faixa vista no treino ({mi}–{ma}); "
        "a previsão é uma extrapolação e merece menos confiança."
        for nome, (mi, ma) in FAIXAS.items()
        if not mi <= valores[nome] <= ma
    ]

    modelo, preprocessador, metadados = _modelo()
    paciente = derivar_variaveis(
        pd.DataFrame(
            [
                {
                    "gender": 2 if sexo == "masculino" else 1,
                    "height": altura_cm,
                    "weight": peso_kg,
                    "ap_hi": pressao_sistolica,
                    "ap_lo": pressao_diastolica,
                    "cholesterol": colesterol,
                    "gluc": glicose,
                    "smoke": int(fumante),
                    "alco": int(consome_alcool),
                    "active": int(fisicamente_ativo),
                    "idade_anos": idade,
                }
            ]
        )
    )[config.ENTRADAS]

    def prever(dados: pd.DataFrame):
        return treino.prever_bruto(modelo, preprocessador, dados)

    prob = float(prever(paciente)[0])
    limiar = metadados["limiar"]
    contrib = avaliacao.contribuicoes_por_oclusao(
        prever, paciente, pd.DataFrame([metadados["referencia"]])
    )
    principais = contrib[contrib.abs() > 0.005].sort_values(key=abs, ascending=False).head(5)

    return {
        "probabilidade": round(prob, 3),
        "limiar_de_triagem": round(limiar, 3),
        "decisao": "encaminhar para investigação" if prob >= limiar else "acompanhamento de rotina",
        "imc": round(float(paciente["imc"].iat[0]), 1),
        # Efeito = probabilidade deste paciente − probabilidade se a variável tivesse o
        # valor típico da base. Perto da saturação (pressão muito alta) os efeitos das
        # demais variáveis ficam pequenos e podem inverter de sinal.
        "fatores_principais": [
            {
                "variavel": config.ROTULOS[nome],
                "valor_do_paciente": _descrever(nome, paciente[nome].iat[0]),
                "valor_tipico_da_base": _descrever(nome, metadados["referencia"][nome]),
                "efeito_pontos_percentuais": round(100 * valor, 1),
                "sentido": "aumenta o risco" if valor > 0 else "reduz o risco",
            }
            for nome, valor in principais.items()
        ],
        "avisos": avisos,
    }


def desempenho() -> dict:
    """Métricas do modelo no conjunto de teste (pacientes nunca vistos no treino)."""
    _, _, metadados = _modelo()
    escolhido, padrao = metadados["teste_limiar_escolhido"], metadados["teste_limiar_05"]
    return {
        "pacientes_de_teste": metadados["n_teste"],
        "auc_roc": round(escolhido["auc_roc"], 3),
        "limiar_de_triagem": round(metadados["limiar"], 3),
        "no_limiar_de_triagem": {
            "sensibilidade": round(escolhido["sensibilidade"], 3),
            "especificidade": round(escolhido["especificidade"], 3),
            "precisao": round(escolhido["precisao"], 3),
        },
        "no_limiar_0_5": {
            "sensibilidade": round(padrao["sensibilidade"], 3),
            "especificidade": round(padrao["especificidade"], 3),
        },
        "arquitetura": metadados["arquitetura"],
    }


#: Base de conhecimento com os achados do projeto. A busca é por palavras-chave;
#: numa base maior, seria trocada por embeddings e um banco vetorial (RAG).
ACHADOS = {
    "fumo e álcool": (
        "Na base, fumo e álcool autodeclarados aparecem associados a risco MENOR. É um "
        "artefato conhecido desta base (hábitos autodeclarados e confundidos com sexo e "
        "idade), não um efeito protetor. Fumar e beber continuam sendo fatores de risco "
        "cardiovascular estabelecidos; o modelo apenas reflete um viés dos dados."
    ),
    "pressão arterial": (
        "A pressão sistólica é, com folga, a variável mais importante (permutação, SHAP e "
        "LIME concordam). Sozinha, a rede chegou a uma fronteira de 130–135 mmHg, perto do "
        "limiar clínico de hipertensão, e o risco previsto satura acima de ~150 mmHg."
    ),
    "efeitos contraintuitivos": (
        "Os fatores mostram quanto a probabilidade mudaria se a variável tivesse o valor típico "
        "da base. Quando a pressão já é alta, o risco previsto está perto da saturação "
        "(~85–90 %) e os efeitos das demais variáveis ficam pequenos (1–2 pontos) e podem "
        "inverter de sinal, como não ser ativo aparecer reduzindo o risco. É uma interação "
        "aprendida pela rede, não uma recomendação: atividade física continua protetora."
    ),
    "limiar de triagem": (
        "O limiar foi escolhido na validação para encontrar pelo menos 80 % dos doentes. "
        "Com ele a rede encontra ~81 % dos doentes, contra ~70 % no limiar 0,5, ao custo de "
        "mais alarmes falsos (especificidade cai de ~77 % para ~63 %)."
    ),
    "comparação de modelos": (
        "Redes (MLP em NumPy, Keras e PyTorch, e Transformer tabular) e Gradient Boosting "
        "ficam no mesmo patamar de AUC (~0,80); o Gradient Boosting fica à frente por no "
        "máximo 0,002. A regressão logística fica ~0,008 atrás. O teto vem dos dados "
        "(16 variáveis tabulares, rótulos ruidosos), não da arquitetura."
    ),
    "limitações": (
        "O modelo não diagnostica: prioriza pacientes para investigação. A base não tem ECG, "
        "histórico familiar, medicação nem exames laboratoriais detalhados; colesterol e "
        "glicose vêm em 3 níveis; hábitos são autodeclarados. Treinado em adultos de 30 a 65 "
        "anos: fora dessa faixa a previsão é extrapolação."
    ),
    "variáveis": (
        "Entradas: idade, sexo, altura, peso, IMC (calculado), pressão sistólica e "
        "diastólica, colesterol e glicose (1 = normal, 2 = acima do normal, 3 = muito acima), "
        "fumo, álcool e atividade física (sim/não)."
    ),
}

_PALAVRAS = {
    "fumo e álcool": ["fum", "cigarr", "tabag", "álcool", "alcool", "beb"],
    "pressão arterial": ["press", "hiperten", "sistól", "sistol", "mmhg"],
    "efeitos contraintuitivos": [
        "contraintuit",
        "invert",
        "sedent",
        "ativ",
        "saturação",
        "estranho",
    ],
    "limiar de triagem": ["limiar", "corte", "sensibilidade", "especificidade", "triagem"],
    "comparação de modelos": ["compar", "boosting", "logíst", "logist", "transformer", "auc"],
    "limitações": ["limita", "confia", "diagnóst", "diagnost", "idade", "ecg"],
    "variáveis": ["variáve", "variave", "entrada", "dado", "colesterol", "glicose"],
}


def buscar_achados(pergunta: str) -> list[dict[str, str]]:
    """Achados do projeto relacionados à pergunta (busca por palavras-chave)."""
    texto = pergunta.lower()
    encontrados = [
        {"tema": tema, "achado": ACHADOS[tema]}
        for tema, palavras in _PALAVRAS.items()
        if any(p in texto for p in palavras)
    ]
    return encontrados or [{"tema": tema, "achado": ACHADOS[tema]} for tema in ("limitações",)]


def interpretar_pressao(texto: str) -> dict:
    """Converte pressão escrita à brasileira em mmHg: "15 por 9,5" → 150/95.

    Aceita "15 por 9", "15x9", "15/9,5", "150/95" e "150 por 95". Valores abaixo de 30
    estão em cmHg (o "12 por 8" do consultório) e são multiplicados por 10.
    """
    numeros = re.findall(r"\d+(?:[.,]\d+)?", texto)
    if len(numeros) != 2:
        return {"erro": f"Não encontrei dois valores de pressão em {texto!r}."}
    sistolica, diastolica = (float(n.replace(",", ".")) for n in numeros)
    sistolica, diastolica = (v * 10 if v < 30 else v for v in (sistolica, diastolica))
    return {"pressao_sistolica": round(sistolica), "pressao_diastolica": round(diastolica)}


# --------------------------------------------------------------------------- #
# Versões para o LangChain: a docstring é o que o LLM lê para decidir a chamada.
# --------------------------------------------------------------------------- #


@tool
def avaliar_paciente(
    idade: float,
    sexo: Literal["feminino", "masculino"],
    altura_cm: float,
    peso_kg: float,
    pressao_sistolica: int,
    pressao_diastolica: int,
    colesterol: Literal[1, 2, 3],
    glicose: Literal[1, 2, 3],
    fumante: bool,
    consome_alcool: bool,
    fisicamente_ativo: bool,
) -> dict:
    """Calcula com a rede neural a probabilidade de doença cardiovascular de um paciente.

    Use SEMPRE esta ferramenta para qualquer número de risco; nunca estime de cabeça.
    Só chame quando tiver os 11 dados; se faltar algum, pergunte ao usuário antes.
    Colesterol e glicose: 1 = normal, 2 = acima do normal, 3 = muito acima do normal.
    Devolve probabilidade, limiar de triagem, decisão, IMC, os fatores que mais pesaram
    (em pontos percentuais) e avisos de valores fora da faixa do treino.
    """
    return avaliar(**locals())


@tool
def converter_pressao(texto: str) -> dict:
    """Converte a pressão arterial como a pessoa escreveu para mmHg.

    Use SEMPRE antes de `avaliar_paciente` quando a pressão vier em texto, por exemplo
    "15 por 9,5", "12x8" ou "150/95". Passe o trecho exatamente como foi escrito.
    """
    return interpretar_pressao(texto)


@tool
def desempenho_do_modelo() -> dict:
    """Métricas da rede no teste: AUC, sensibilidade, especificidade e precisão nos dois limiares.

    Use quando perguntarem se o modelo é confiável, quanto ele acerta ou quantos doentes encontra.
    """
    return desempenho()


@tool
def consultar_achados(pergunta: str) -> list[dict[str, str]]:
    """Busca achados e limitações documentados no projeto.

    Use para perguntas sobre por que o modelo se comporta de certo jeito: fumo e álcool,
    pressão arterial, limiar de triagem, comparação entre modelos, limitações e variáveis.
    """
    return buscar_achados(pergunta)


FERRAMENTAS = [converter_pressao, avaliar_paciente, desempenho_do_modelo, consultar_achados]
