"""Aplicação Streamlit — triagem de risco cardiovascular com rede neural.

Execute com `uv run invoke app`. Requer o modelo gravado por
`uv run invoke treinar` (ou pelo notebook 05).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.agente import agente as _agente  # noqa: E402, F401  (carrega o .env)
from src.data.dados import derivar_variaveis, preparar  # noqa: E402
from src.deployment import graficos_app as graf  # noqa: E402
from src.model import avaliacao, treino  # noqa: E402
from src.utils.formato import numero, pct  # noqa: E402
from src.utils.io import ler_tabela  # noqa: E402

st.set_page_config(page_title="Coração de Dados · Redes Neurais", page_icon="🫀", layout="wide")


@st.cache_resource
def carregar():
    return treino.carregar_modelo_final()


if not treino.ARQUIVO_MODELO.exists():
    st.error("Modelo não encontrado. Rode `uv run invoke treinar` antes de abrir a aplicação.")
    st.stop()

modelo, preprocessador, metadados = carregar()
LIMIAR = metadados["limiar"]
referencia = pd.DataFrame([metadados["referencia"]])


def prever(pacientes: pd.DataFrame):
    return treino.prever_bruto(modelo, preprocessador, pacientes)


def inteiro(n: int) -> str:
    """Inteiro com separador de milhar brasileiro (68.551)."""
    return f"{n:,}".replace(",", ".")


@st.cache_data(show_spinner=False)
def probabilidades_teste() -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades e rótulos dos pacientes de teste (calculados uma vez)."""
    particao = preparar()
    return prever(particao.bruto_teste), particao.y_teste


@st.cache_data(show_spinner=False)
def varredura_limiar() -> pd.DataFrame:
    """Sensibilidade, especificidade e precisão no teste para limiares de 0,05 a 0,95."""
    p, y = probabilidades_teste()
    linhas = []
    for limiar in np.round(np.arange(0.05, 0.951, 0.01), 2):
        m = avaliacao.metricas(y, p, limiar)
        linhas.append(
            {
                "limiar": limiar,
                "Sensibilidade": m["sensibilidade"],
                "Especificidade": m["especificidade"],
                "Precisão": m["precisao"],
            }
        )
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner=False, max_entries=64)
def analisar_paciente(dados: tuple[tuple[str, float], ...]) -> dict:
    """Probabilidade, fatores, curva "e se" da pressão e mapa pressão × idade de um paciente.

    Recebe o paciente como tupla de pares (hashável) para que o resultado fique em cache
    enquanto os dados do formulário não mudam.
    """
    paciente = derivar_variaveis(pd.DataFrame([dict(dados)]))[config.ENTRADAS]
    registro = paciente.iloc[0]

    def variantes(**colunas) -> pd.DataFrame:
        grade = pd.DataFrame(colunas)
        outros = paciente.drop(columns=list(colunas)).iloc[[0] * len(grade)].reset_index(drop=True)
        return pd.concat([outros, grade], axis=1)[config.ENTRADAS]

    # Curva "e se": só a sistólica varia (acima da diastólica do paciente).
    pressoes = np.arange(max(90, registro["ap_lo"] + 5), 201, 2.5)
    curva = pd.DataFrame({"ap_hi": pressoes})
    curva["probabilidade"] = prever(variantes(ap_hi=pressoes))

    # Mapa: sistólica × idade, com o restante do paciente fixo.
    pa, idade = np.meshgrid(np.arange(90, 201, 5.0), np.arange(30, 66, 1.0))
    grade = pd.DataFrame({"ap_hi": pa.ravel(), "idade_anos": idade.ravel()})
    grade = grade[grade["ap_hi"] > registro["ap_lo"]].reset_index(drop=True)
    grade["probabilidade"] = prever(variantes(**{c: grade[c] for c in ["ap_hi", "idade_anos"]}))

    return {
        "prob": float(prever(paciente)[0]),
        "imc": float(registro["imc"]),
        "contrib": avaliacao.contribuicoes_por_oclusao(prever, paciente, referencia),
        "curva": curva,
        "grade": grade,
    }


st.title("🫀 Triagem de risco cardiovascular")
st.caption(
    "Rede neural (MLP em Keras) treinada em 41 mil pacientes da base Coração de Dados. "
    "Ferramenta de apoio à priorização — não substitui avaliação clínica."
)

aba_paciente, aba_assistente, aba_modelo, aba_sobre = st.tabs(
    ["Paciente", "Assistente (agente)", "Desempenho do modelo", "Sobre"]
)

# --------------------------------------------------------------------------- #
# Paciente
# --------------------------------------------------------------------------- #

with aba_paciente:
    with st.form("paciente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            idade = st.slider("Idade (anos)", 30, 65, 52)
            sexo = st.radio("Sexo", ["Feminino", "Masculino"], horizontal=True)
            altura = st.number_input("Altura (cm)", 130, 210, 165)
            peso = st.number_input("Peso (kg)", 40.0, 180.0, 75.0, step=0.5)
        with c2:
            ap_hi = st.number_input("Pressão sistólica (mmHg)", 80, 240, 130)
            ap_lo = st.number_input("Pressão diastólica (mmHg)", 50, 160, 85)
            niveis = {"Normal": 1, "Acima do normal": 2, "Muito acima do normal": 3}
            colesterol = st.selectbox("Colesterol", list(niveis))
            glicose = st.selectbox("Glicose", list(niveis))
        with c3:
            fumante = st.checkbox("Fumante")
            alcool = st.checkbox("Consome álcool")
            ativo = st.checkbox("Pratica atividade física", value=True)
        st.form_submit_button("Calcular risco", type="primary")

    if ap_lo >= ap_hi:
        st.warning("A pressão diastólica deve ser menor que a sistólica.")
    else:
        # O formulário devolve os últimos valores enviados (ou os padrões na
        # primeira abertura), então a previsão aparece desde o início.
        dados_paciente = {
            "gender": 2 if sexo == "Masculino" else 1,
            "height": altura,
            "weight": peso,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": niveis[colesterol],
            "gluc": niveis[glicose],
            "smoke": int(fumante),
            "alco": int(alcool),
            "active": int(ativo),
            "idade_anos": idade,
        }
        analise = analisar_paciente(tuple(dados_paciente.items()))
        prob = analise["prob"]
        p_teste, _ = probabilidades_teste()
        percentil = (p_teste < prob).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probabilidade de doença cardiovascular", pct(prob))
        m2.metric("Limiar de triagem", pct(LIMIAR))
        m3.metric("Pacientes com risco menor", pct(percentil))
        m4.metric("IMC", numero(analise["imc"]))
        if prob >= LIMIAR:
            st.error("**Encaminhar para investigação** — probabilidade acima do limiar de triagem.")
        else:
            st.success("**Acompanhamento de rotina** — probabilidade abaixo do limiar de triagem.")

        col_curva, col_populacao = st.columns(2)
        with col_curva, st.container(border=True):
            st.markdown("**E se a pressão sistólica fosse outra?**")
            cruzamento = graf.pressao_de_cruzamento(analise["curva"], LIMIAR)
            if cruzamento is None:
                texto = "Mesmo com pressão alta, este paciente ficaria abaixo do limiar."
            elif cruzamento <= analise["curva"]["ap_hi"].min():
                texto = "Mesmo com pressão baixa, este paciente ficaria acima do limiar."
            else:
                texto = f"Com os demais dados iguais, o limiar é cruzado a ~{cruzamento:.0f} mmHg."
            st.caption(f"Só a sistólica varia; o resto do paciente fica fixo. {texto}")
            st.altair_chart(graf.e_se_pressao(analise["curva"], ap_hi, prob, LIMIAR))
        with col_populacao, st.container(border=True):
            st.markdown("**Onde este paciente está entre os outros**")
            st.caption(
                f"Probabilidades previstas para os {inteiro(len(p_teste))} pacientes de teste. "
                "A linha tracejada é o limiar de triagem."
            )
            st.altair_chart(graf.posicao_na_populacao(p_teste, prob, LIMIAR))

        col_fatores, col_mapa = st.columns([2, 3])
        with col_fatores, st.container(border=True):
            st.markdown("**O que mais pesou nesta previsão**")
            st.caption(
                "Quanto a probabilidade mudaria se a variável tivesse o valor típico da base "
                "(mediana ou valor mais comum)."
            )
            contrib = analise["contrib"]
            contrib = contrib[contrib.abs() > 0.002]
            if contrib.empty:
                st.info("Todas as variáveis estão próximas do perfil típico da base.")
            else:
                st.altair_chart(graf.fatores(contrib))
        with col_mapa, st.container(border=True):
            st.markdown("**O paciente no mapa pressão × idade**")
            st.caption(
                "Probabilidade para cada combinação de pressão sistólica e idade, com os demais "
                "dados do paciente fixos. Azul = abaixo do limiar; coral = acima."
            )
            st.altair_chart(graf.mapa_pressao_idade(analise["grade"], ap_hi, idade, LIMIAR))

# --------------------------------------------------------------------------- #
# Assistente — fluxo LangGraph (interpretar → validar → avaliar → explicar)
# --------------------------------------------------------------------------- #

EXEMPLOS = {
    ":material/person: Paciente completo": (
        "Homem de 58 anos, 1,72 m e 94 kg, pressão 15 por 9,5, colesterol acima do normal, "
        "glicose normal, fuma, não bebe e é sedentário."
    ),
    ":material/help: Faltando dados": "Mulher, 61 anos, pressão 14 por 9. Devo encaminhar?",
    ":material/quiz: Pergunta sobre o modelo": "O modelo diz que fumar reduz o risco? Está certo?",
}


@st.cache_resource
def carregar_fluxo(provedor_llm: str):
    """Grafo compilado uma vez por provedor (demo, openai ou deepseek)."""
    from src.agente.agente import criar_modelo_llm
    from src.agente.fluxo import construir_fluxo

    return construir_fluxo(None if provedor_llm == "demo" else criar_modelo_llm())


def mostrar_rastreio(rastreio: list[str]) -> None:
    """Linha do tempo com os nós do grafo percorridos nesta resposta."""
    if not rastreio:
        return
    with st.status(f"Fluxo: {len(rastreio)} passos", type="compact", state="complete"):
        for linha in rastreio:
            no, _, detalhe = linha.partition(" ")
            with st.status(no, type="step", state="complete"):
                st.caption(detalhe)


def novo_paciente() -> None:
    st.session_state.conversa = []
    st.session_state.parcial_paciente = {}


with aba_assistente:
    from src.agente.agente import provedor
    from src.agente.fluxo import LIMITE_MENSAGEM, triar

    provedor_llm = provedor()
    st.session_state.setdefault("conversa", [])
    st.session_state.setdefault("parcial_paciente", {})

    cab_texto, cab_botao = st.columns([4, 1], vertical_alignment="center")
    with cab_texto:
        if provedor_llm == "demo":
            st.caption(
                ":orange-badge[Modo demonstração] Sem chave de API: os dados são extraídos por "
                "regras e a resposta é montada por modelo de texto, sem IA e sem custo. O risco "
                "vem da mesma rede neural."
            )
        else:
            st.caption(
                f":green-badge[LLM: {provedor_llm}] O LLM só extrai os dados e redige a "
                "explicação; validação (Pydantic) e risco (rede neural) são calculados em código."
            )
    with cab_botao:
        st.button("Novo paciente", icon=":material/restart_alt:", on_click=novo_paciente,
                  width="stretch")  # fmt: skip

    # Contêiner reservado antes do campo de texto: as mensagens novas entram nele e
    # ficam acima do campo, e não abaixo (dentro de abas o campo não fica fixo no rodapé).
    conversa = st.container()
    with conversa:
        for mensagem in st.session_state.conversa:
            with st.chat_message(mensagem["role"]):
                mostrar_rastreio(mensagem.get("rastreio", []))
                st.markdown(mensagem["content"])

    pergunta = st.chat_input(
        "Ex.: homem, 58 anos, 1,72 m, 94 kg, pressão 15 por 9, colesterol alto, fuma",
        max_chars=LIMITE_MENSAGEM,
        submit_mode="disable",
    )
    if not st.session_state.conversa:
        with st.container(horizontal=True):
            for rotulo, texto in EXEMPLOS.items():
                if st.button(rotulo, key=f"exemplo_{rotulo}"):
                    pergunta = texto

    if pergunta:
        st.session_state.conversa.append({"role": "user", "content": pergunta})
        with conversa:
            with st.chat_message("user"):
                st.markdown(pergunta)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Executando o fluxo…"):
                        r = triar(pergunta, st.session_state.parcial_paciente,
                                  fluxo=carregar_fluxo(provedor_llm))  # fmt: skip
                except Exception as erro:  # falha do provedor: avisa, sem fingir resposta (APP-03)
                    st.error(
                        f"Assistente indisponível no momento ({type(erro).__name__}). "
                        "Tente novamente em instantes.",
                        icon=":material/error:",
                    )
                    st.session_state.conversa.pop()
                else:
                    mostrar_rastreio(r["rastreio"])
                    st.markdown(r["resposta"])
                    st.session_state.parcial_paciente = r["parcial"]
                    st.session_state.conversa.append(
                        {"role": "assistant", "content": r["resposta"], "rastreio": r["rastreio"]}
                    )
                    st.session_state.conversa = st.session_state.conversa[-20:]

# --------------------------------------------------------------------------- #
# Desempenho
# --------------------------------------------------------------------------- #

with aba_modelo:
    escolhido = metadados["teste_limiar_escolhido"]
    padrao = metadados["teste_limiar_05"]
    st.markdown(
        f"Avaliado em **{inteiro(metadados['n_teste'])} pacientes de teste** que a rede nunca "
        f"viu. O limiar de {numero(LIMIAR, 2)} foi escolhido na validação para encontrar "
        "pelo menos "
        f"{pct(metadados['recall_minimo'])} dos doentes."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", numero(escolhido["auc_roc"], 3))
    c2.metric("Sensibilidade", pct(escolhido["sensibilidade"], 1))
    c3.metric("Especificidade", pct(escolhido["especificidade"], 1))
    c4.metric("Precisão", pct(escolhido["precisao"], 1))

    @st.fragment
    def explorar_limiar() -> None:
        """Controle do limiar — roda sozinho, sem recalcular o resto da página."""
        with st.container(border=True):
            st.markdown("**Explore o limiar de decisão**")
            st.caption(
                "Arraste para ver o compromisso: limiar mais baixo encontra mais doentes, mas "
                "gera mais alarmes falsos. A linha tracejada no gráfico é o limiar escolhido."
            )
            limiar = st.slider(
                "Limiar", 0.05, 0.95, round(LIMIAR, 2), 0.01, format="%.2f", key="limiar_explorado"
            )
            p, y = probabilidades_teste()
            m = avaliacao.metricas(y, p, limiar)
            # Referência no mesmo arredondamento do controle, para o delta partir de zero.
            triagem = avaliacao.metricas(y, p, round(LIMIAR, 2))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Sensibilidade",
                pct(m["sensibilidade"], 1),
                pct(m["sensibilidade"] - triagem["sensibilidade"], 1, sinal=True) + " vs. triagem",
            )
            k2.metric(
                "Especificidade",
                pct(m["especificidade"], 1),
                pct(m["especificidade"] - triagem["especificidade"], 1, sinal=True)
                + " vs. triagem",
            )
            k3.metric("Precisão", pct(m["precisao"], 1))
            k4.metric("Pacientes encaminhados", pct((m["vp"] + m["fp"]) / len(y)))
            c_matriz, c_curvas = st.columns([2, 3])
            with c_matriz:
                st.altair_chart(graf.matriz_confusao(m["vp"], m["fp"], m["fn"], m["vn"]))
            with c_curvas:
                st.altair_chart(graf.curvas_limiar(varredura_limiar(), limiar))

    explorar_limiar()

    col_cal, col_comp = st.columns(2)
    with col_cal, st.container(border=True):
        st.markdown("**Calibração: 70 % quer dizer 70 %?**")
        st.caption(
            "Pacientes agrupados em 10 faixas de probabilidade prevista. Quanto mais perto da "
            "diagonal, mais a probabilidade pode ser lida como frequência real. Clique na "
            "legenda para destacar um modelo."
        )
        try:
            pontos = ler_tabela("05-comparacao-final", "calibracao")
            st.altair_chart(graf.calibracao(pontos))
        except FileNotFoundError:
            st.info("Execute o notebook 05 para gerar a calibração.")
    with col_comp, st.container(border=True):
        st.markdown("**Comparação com outros modelos**")
        st.caption(
            "AUC no teste com intervalo de confiança de 95 % (bootstrap). Intervalos que se "
            "sobrepõem quase por inteiro indicam modelos equivalentes na prática."
        )
        try:
            st.altair_chart(
                graf.comparacao_modelos(ler_tabela("05-comparacao-final", "auc-bootstrap"))
            )
        except FileNotFoundError:
            st.info("Execute o notebook 05 para gerar a comparação.")

    with st.container(border=True):
        st.markdown("**Quais variáveis a rede mais usa**")
        st.caption(
            "Importância por permutação: quanto a AUC cai quando a variável é embaralhada. "
            "A pressão sistólica domina; hábitos autodeclarados quase não pesam."
        )
        try:
            st.altair_chart(
                graf.importancia(ler_tabela("06-explicabilidade", "importancia-permutacao"))
            )
        except FileNotFoundError:
            st.info("Execute o notebook 06 para gerar a importância das variáveis.")

    with st.expander("Tabela: limiar 0,50 × limiar de triagem"):
        tabela = pd.DataFrame(
            {"Limiar 0,50": padrao, f"Limiar {numero(LIMIAR, 2)} (triagem)": escolhido}
        ).loc[
            [
                "acuracia",
                "sensibilidade",
                "especificidade",
                "precisao",
                "f1",
                "vp",
                "fp",
                "fn",
                "vn",
            ]
        ]
        tabela.index = [
            "Acurácia",
            "Sensibilidade",
            "Especificidade",
            "Precisão",
            "F1",
            "Doentes encontrados (VP)",
            "Alarmes falsos (FP)",
            "Doentes perdidos (FN)",
            "Saudáveis liberados (VN)",
        ]
        st.dataframe(
            tabela.style.format(lambda v: numero(v, 3), subset=pd.IndexSlice[tabela.index[:5], :])
        )

# --------------------------------------------------------------------------- #
# Sobre
# --------------------------------------------------------------------------- #

with aba_sobre:
    arq = metadados["arquitetura"]
    st.markdown(
        f"""
**Arquitetura:** {len(arq["ocultas"])} camadas ocultas
({"-".join(map(str, arq["ocultas"]))} neurônios,
ReLU, Dropout {arq["dropout"]}), saída sigmoide · otimizador {arq["otimizador"]} ·
{inteiro(metadados["n_parametros"])} parâmetros · {metadados["epocas"]} épocas com parada
antecipada.

**Dados:** base curada do projeto Coração de Dados (Tema 6), 68.551 pacientes — divisão
60/20/20 estratificada.

**Limitações:** a base não tem exames como ECG, histórico familiar ou medicação; os hábitos
(fumo, álcool, atividade) são autodeclarados. O modelo prioriza pacientes, não diagnostica.
"""
    )
