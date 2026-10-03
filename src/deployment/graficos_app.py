"""Gráficos interativos (Altair) da aplicação Streamlit.

Cada função recebe dados prontos e devolve um ``alt.Chart``. Nenhuma chama o
modelo nem o Streamlit — assim podem ser testadas sem abrir a aplicação, e os
cálculos caros ficam em cache no ``app.py``.
"""

from __future__ import annotations

from functools import wraps

import altair as alt
import numpy as np
import pandas as pd

from src import config

CINZA = "#8A99A6"
COR_SIM, COR_NAO = config.CORAL, config.AZUL_COBALTO
ESCALA_RISCO_RANGE = [config.AZUL_COBALTO, "#EEF2F5", config.CORAL]


#: Números no padrão brasileiro em eixos, legendas e dicas: 0,801 · 80,8% · 13.711.
LOCALE_PT_BR = {
    "number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]}
}


def _em_portugues(funcao):
    """Aplica o formato numérico brasileiro ao gráfico devolvido."""

    @wraps(funcao)
    def grafico(*args, **kwargs):
        return funcao(*args, **kwargs).configure(locale=LOCALE_PT_BR)

    return grafico


def _escala_risco(limiar: float) -> alt.Scale:
    """Azul abaixo do limiar, coral acima — branco exatamente no limiar."""
    # Interpolação em RGB: a padrão (HCL) passa por lilás entre o branco e o coral.
    return alt.Scale(domain=[0, limiar, 1], range=ESCALA_RISCO_RANGE, interpolate="rgb")


def _linha_limiar(
    limiar: float, eixo: str = "y", nome: str = "limiar de triagem"
) -> alt.LayerChart:
    """Linha tracejada do limiar, com rótulo."""
    dados = pd.DataFrame({"limiar": [limiar], "rotulo": [f"{nome} ({limiar:.0%})"]})
    campo = alt.Y("limiar:Q") if eixo == "y" else alt.X("limiar:Q")
    regra = (
        alt.Chart(dados)
        .mark_rule(color=CINZA, strokeDash=[5, 4])
        .encode(campo, tooltip=[alt.Tooltip("limiar:Q", title="Limiar de triagem", format=".0%")])
    )
    if eixo == "y":
        texto = regra.mark_text(
            align="right", baseline="bottom", x="width", dx=-4, dy=-3, color=CINZA, fontSize=11
        )
    else:
        texto = regra.mark_text(
            align="left", baseline="top", y=0, dx=4, dy=4, color=CINZA, fontSize=11
        )
    return regra + texto.encode(text="rotulo:N")


# --------------------------------------------------------------------------- #
# Aba Paciente
# --------------------------------------------------------------------------- #


@_em_portugues
def fatores(contrib: pd.Series) -> alt.Chart:
    """Barras horizontais: quanto cada variável empurra a probabilidade do paciente."""
    dados = pd.DataFrame(
        {
            "variavel": [config.ROTULOS[c] for c in contrib.index],
            "efeito": contrib.to_numpy() * 100,
        }
    )
    dados["sentido"] = np.where(dados["efeito"] > 0, "aumenta o risco", "reduz o risco")
    return (
        alt.Chart(dados)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("efeito:Q", title="Pontos percentuais de probabilidade"),
            y=alt.Y(
                "variavel:N",
                sort=alt.EncodingSortField("efeito", op="max", order="descending"),
                title=None,
            ),
            color=alt.Color(
                "sentido:N",
                scale=alt.Scale(
                    domain=["aumenta o risco", "reduz o risco"], range=[COR_SIM, COR_NAO]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("variavel:N", title="Variável"),
                alt.Tooltip("efeito:Q", title="Efeito (p.p.)", format="+.1f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def pressao_de_cruzamento(curva: pd.DataFrame, limiar: float) -> float | None:
    """Menor pressão sistólica a partir da qual a probabilidade passa do limiar."""
    acima = curva[curva["probabilidade"] >= limiar]
    return float(acima["ap_hi"].min()) if len(acima) else None


@_em_portugues
def e_se_pressao(curva: pd.DataFrame, ap_hi: float, prob: float, limiar: float) -> alt.Chart:
    """Probabilidade do paciente se só a pressão sistólica mudasse."""
    base = alt.Chart(curva).encode(
        x=alt.X("ap_hi:Q", title="Pressão sistólica (mmHg)", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "probabilidade:Q",
            title="Probabilidade prevista",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
    )
    linha = base.mark_line(color=config.AZUL_PROFUNDO, strokeWidth=2.5)
    pontos_hover = base.mark_point(opacity=0, size=80).encode(
        tooltip=[
            alt.Tooltip("ap_hi:Q", title="Sistólica (mmHg)"),
            alt.Tooltip("probabilidade:Q", title="Probabilidade", format=".0%"),
        ]
    )
    atual = (
        alt.Chart(pd.DataFrame({"ap_hi": [ap_hi], "probabilidade": [prob]}))
        .mark_point(
            filled=True,
            size=160,
            color=COR_SIM if prob >= limiar else COR_NAO,
            stroke="white",
            strokeWidth=1.5,
        )
        .encode(
            x="ap_hi:Q",
            y="probabilidade:Q",
            tooltip=[
                alt.Tooltip("ap_hi:Q", title="Pressão atual"),
                alt.Tooltip("probabilidade:Q", title="Probabilidade atual", format=".0%"),
            ],
        )
    )
    return (_linha_limiar(limiar) + linha + pontos_hover + atual).properties(height=280)


@_em_portugues
def posicao_na_populacao(p_teste: np.ndarray, prob: float, limiar: float) -> alt.Chart:
    """Histograma das probabilidades dos pacientes de teste, com o paciente marcado."""
    bordas = np.linspace(0, 1, 41)
    contagem, _ = np.histogram(p_teste, bins=bordas)
    dados = pd.DataFrame({"inicio": bordas[:-1], "fim": bordas[1:], "pacientes": contagem})
    dados["faixa"] = np.where(dados["inicio"] >= limiar, "acima do limiar", "abaixo do limiar")
    barras = (
        alt.Chart(dados)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X(
                "inicio:Q", bin="binned", title="Probabilidade prevista", axis=alt.Axis(format="%")
            ),
            x2="fim:Q",
            y=alt.Y("pacientes:Q", title="Pacientes de teste"),
            color=alt.Color(
                "faixa:N",
                scale=alt.Scale(
                    domain=["abaixo do limiar", "acima do limiar"], range=[COR_NAO, COR_SIM]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("inicio:Q", title="De", format=".0%"),
                alt.Tooltip("fim:Q", title="Até", format=".0%"),
                alt.Tooltip("pacientes:Q", title="Pacientes", format=","),
            ],
        )
    )
    paciente = (
        alt.Chart(pd.DataFrame({"prob": [prob], "rotulo": ["este paciente"]}))
        .mark_rule(color=config.AZUL_PROFUNDO, strokeWidth=3)
        .encode(x="prob:Q", tooltip=[alt.Tooltip("prob:Q", title="Este paciente", format=".0%")])
    )
    rotulo = paciente.mark_text(
        align="left", dx=5, dy=-120, fontWeight="bold", color=config.AZUL_PROFUNDO
    ).encode(text="rotulo:N")
    return (barras + _linha_limiar(limiar, eixo="x") + paciente + rotulo).properties(height=280)


@_em_portugues
def mapa_pressao_idade(grade: pd.DataFrame, ap_hi: float, idade: float, limiar: float) -> alt.Chart:
    """Probabilidade do paciente em todo o plano pressão × idade, com ele marcado."""
    passo_pa = float(np.diff(np.unique(grade["ap_hi"]))[0])
    passo_id = float(np.diff(np.unique(grade["idade_anos"]))[0])
    # Células 2 % maiores que o passo: sobrepõem-se e eliminam as frestas brancas.
    grade = grade.assign(
        pa_fim=grade["ap_hi"] + passo_pa * 1.02, id_fim=grade["idade_anos"] + passo_id * 1.02
    )
    celulas = (
        alt.Chart(grade)
        .mark_rect()
        .encode(
            x=alt.X(
                "ap_hi:Q",
                title="Pressão sistólica (mmHg)",
                axis=alt.Axis(grid=False),
                scale=alt.Scale(zero=False, nice=False),
            ),
            x2="pa_fim:Q",
            y=alt.Y(
                "idade_anos:Q",
                title="Idade (anos)",
                axis=alt.Axis(grid=False),
                scale=alt.Scale(zero=False, nice=False),
            ),
            y2="id_fim:Q",
            color=alt.Color(
                "probabilidade:Q",
                scale=_escala_risco(limiar),
                legend=alt.Legend(title="Probabilidade", format="%"),
            ),
            tooltip=[
                alt.Tooltip("ap_hi:Q", title="Sistólica"),
                alt.Tooltip("idade_anos:Q", title="Idade", format=".0f"),
                alt.Tooltip("probabilidade:Q", title="Probabilidade", format=".0%"),
            ],
        )
    )
    atual = pd.DataFrame(
        {
            "ap_hi": [ap_hi + passo_pa / 2],
            "idade_anos": [idade + passo_id / 2],
            "r": ["você está aqui"],
        }
    )
    ponto = (
        alt.Chart(atual)
        .mark_point(
            shape="diamond",
            filled=True,
            size=260,
            color=config.AZUL_PROFUNDO,
            stroke="white",
            strokeWidth=2,
        )
        .encode(x="ap_hi:Q", y="idade_anos:Q")
    )
    texto = ponto.mark_text(
        align="left", dx=10, fontWeight="bold", color=config.AZUL_PROFUNDO
    ).encode(text="r:N")
    return (celulas + ponto + texto).properties(height=360)


# --------------------------------------------------------------------------- #
# Aba Desempenho
# --------------------------------------------------------------------------- #


@_em_portugues
def curvas_limiar(varredura: pd.DataFrame, limiar: float) -> alt.Chart:
    """Sensibilidade, especificidade e precisão conforme o limiar, com o escolhido marcado."""
    longo = varredura.melt(id_vars="limiar", var_name="metrica", value_name="valor")
    linhas = (
        alt.Chart(longo)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("limiar:Q", title="Limiar de decisão"),
            y=alt.Y(
                "valor:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color(
                "metrica:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(
                    domain=["Sensibilidade", "Especificidade", "Precisão"],
                    range=[COR_SIM, COR_NAO, config.VERDE],
                ),
            ),
            tooltip=[
                alt.Tooltip("limiar:Q", format=".2f"),
                alt.Tooltip("metrica:N", title="Métrica"),
                alt.Tooltip("valor:Q", title="Valor", format=".1%"),
            ],
        )
    )
    return (linhas + _linha_limiar(limiar, eixo="x", nome="limiar escolhido")).properties(
        height=300
    )


@_em_portugues
def matriz_confusao(vp: int, fp: int, fn: int, vn: int) -> alt.Chart:
    """Matriz de confusão com contagens e o significado de cada célula."""
    dados = pd.DataFrame(
        [
            ("Com doença", "Encaminhado", vp, "doentes encontrados"),
            ("Com doença", "Rotina", fn, "doentes perdidos"),
            ("Sem doença", "Encaminhado", fp, "alarmes falsos"),
            ("Sem doença", "Rotina", vn, "saudáveis liberados"),
        ],
        columns=["real", "decisao", "pacientes", "significado"],
    )
    dados["tipo"] = np.where(
        dados["significado"].isin(["doentes encontrados", "saudáveis liberados"]), "acerto", "erro"
    )
    dados["texto"] = dados["pacientes"].map(lambda n: f"{n:,}".replace(",", "."))
    base = alt.Chart(dados).encode(
        x=alt.X(
            "decisao:N",
            title="Decisão do modelo",
            sort=["Encaminhado", "Rotina"],
            axis=alt.Axis(labelAngle=0, orient="top"),
        ),
        y=alt.Y("real:N", title="Situação real", sort=["Com doença", "Sem doença"]),
    )
    celulas = base.mark_rect(cornerRadius=4, opacity=0.85).encode(
        color=alt.Color(
            "tipo:N",
            scale=alt.Scale(domain=["acerto", "erro"], range=[config.VERDE, config.ROSA]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("significado:N", title="Célula"),
            alt.Tooltip("pacientes:Q", format=","),
        ],
    )
    numeros = base.mark_text(fontSize=20, fontWeight="bold", dy=-8, color="#1B2A35").encode(
        text="texto:N"
    )
    legenda = base.mark_text(fontSize=11, dy=14, color="#1B2A35").encode(text="significado:N")
    return (celulas + numeros + legenda).properties(height=300)


@_em_portugues
def calibracao(pontos: pd.DataFrame) -> alt.Chart:
    """Diagrama de confiabilidade: probabilidade prevista × proporção real de doentes."""
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color=CINZA, strokeDash=[5, 4])
        .encode(x="x:Q", y="y:Q")
    )
    destaque = alt.selection_point(fields=["modelo"], bind="legend")
    linhas = (
        alt.Chart(pontos)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "previsto:Q",
                title="Probabilidade prevista",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "real:Q",
                title="Proporção real de doentes",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "modelo:N",
                title=None,
                legend=alt.Legend(orient="bottom", columns=2),
                scale=alt.Scale(range=config.PALETA),
            ),
            opacity=alt.condition(destaque, alt.value(1), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("previsto:Q", title="Previsto", format=".0%"),
                alt.Tooltip("real:Q", title="Real", format=".0%"),
            ],
        )
        .add_params(destaque)
    )
    return (diagonal + linhas).properties(height=340)


@_em_portugues
def comparacao_modelos(ic: pd.DataFrame, destaque: str = "MLP Keras") -> alt.Chart:
    """AUC de cada modelo no teste com o intervalo de confiança de 95 %."""
    dados = ic.reset_index().rename(columns={"index": "modelo"})
    dados["tipo"] = np.where(dados["modelo"] == destaque, "modelo da aplicação", "outros")
    ordem = dados.sort_values("auc", ascending=False)["modelo"].tolist()
    base = alt.Chart(dados).encode(y=alt.Y("modelo:N", sort=ordem, title=None))
    barras = base.mark_rule(strokeWidth=2.5).encode(
        x=alt.X("ic95_inf:Q", title="AUC no teste (IC 95 %)", scale=alt.Scale(zero=False)),
        x2="ic95_sup:Q",
        color=alt.Color(
            "tipo:N",
            scale=alt.Scale(
                domain=["modelo da aplicação", "outros"], range=[COR_SIM, config.AZUL_CLARO]
            ),
            legend=alt.Legend(orient="bottom", title=None),
        ),
    )
    pontos = base.mark_point(filled=True, size=90).encode(
        x="auc:Q",
        color=alt.Color(
            "tipo:N",
            legend=None,
            scale=alt.Scale(
                domain=["modelo da aplicação", "outros"], range=[COR_SIM, config.AZUL_PROFUNDO]
            ),
        ),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip("auc:Q", title="AUC", format=".3f"),
            alt.Tooltip("ic95_inf:Q", title="IC inferior", format=".3f"),
            alt.Tooltip("ic95_sup:Q", title="IC superior", format=".3f"),
        ],
    )
    return (barras + pontos).properties(height=alt.Step(40))


@_em_portugues
def importancia(permutacao: pd.DataFrame) -> alt.Chart:
    """Queda da AUC ao embaralhar cada variável (importância global)."""
    dados = permutacao.reset_index()
    return (
        alt.Chart(dados)
        .mark_bar(color=config.AZUL_COBALTO, cornerRadiusEnd=3)
        .encode(
            x=alt.X("queda_auc:Q", title="Queda da AUC ao embaralhar a variável"),
            y=alt.Y("variavel:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("variavel:N", title="Variável"),
                alt.Tooltip("queda_auc:Q", title="Queda da AUC", format=".4f"),
            ],
        )
        .properties(height=alt.Step(24))
    )
