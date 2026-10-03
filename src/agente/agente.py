"""Agente ReAct (LangChain + OpenAI) que usa a rede neural como ferramenta.

A pessoa descreve o paciente em texto livre ("homem, 58 anos, pressão 15 por 9,
fuma…"). O agente extrai os dados, pergunta o que faltar, chama a rede pela
ferramenta ``avaliar_paciente`` e explica o resultado em linguagem simples.

Requer ``OPENAI_API_KEY`` ou ``DEEPSEEK_API_KEY`` (pode ficar num arquivo ``.env``
na raiz do projeto — veja ``.env.exemplo``). Sem chave, o fluxo LangGraph de
``src/agente/fluxo.py`` funciona em modo demonstração.

Uso no terminal: ``uv run invoke agente``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from src import config
from src.agente.ferramentas import FERRAMENTAS

load_dotenv(config.RAIZ / ".env")

MODELO_PADRAO = "gpt-4o-mini"

INSTRUCOES = """\
Você é um assistente de triagem cardiovascular para uma equipe de saúde, em português do Brasil.

Como trabalhar:
1. Extraia do texto os 11 dados do paciente: idade, sexo, altura (cm), peso (kg), pressão
   sistólica e diastólica (mmHg), colesterol e glicose (1 = normal, 2 = acima do normal,
   3 = muito acima do normal), fumante, consome álcool e fisicamente ativo.
   - Pressão: passe o trecho exatamente como foi escrito ("15 por 9,5", "12x8") para a
     ferramenta `converter_pressao` e use os valores que ela devolver. Não converta de cabeça.
   - Altura "1,72 m" significa 172 cm.
   - "Normal" = 1, "acima do normal" / "alto" = 2, "muito acima do normal" / "muito alto" = 3.
   - "Sedentário" = não é fisicamente ativo; "não fuma", "não bebe" = não.
   - Essas conversões são diretas: aplique-as sem pedir confirmação.
   - Se algum dado realmente não foi informado, pergunte por ele. Não invente nem assuma
     valores ausentes.
2. Todo número de risco vem da ferramenta `avaliar_paciente`. Nunca estime risco por conta própria.
3. Comece a resposta listando os dados que usou (ex.: "58 anos, masculino, 172 cm, 94 kg,
   pressão 150/95, colesterol 2…"), para que a equipe possa conferir a extração.
   Depois dê a probabilidade, a decisão de triagem (comparada ao limiar) e os 2 ou 3
   fatores que mais pesaram, em linguagem simples. Repita os avisos que a ferramenta devolver.
   Fale SOMENTE dos fatores listados em `fatores_principais`, com o sentido que a ferramenta
   deu. Não atribua efeito a variáveis que não estão na lista.
4. Só chame `consultar_achados` sobre fumo ou álcool se, NA LISTA devolvida, fumo ou
   álcool aparecerem reduzindo o risco; nesse caso explique que é um artefato desta base e
   que continuam sendo fatores de risco reais. Se outro fator aparecer
   com sentido contraintuitivo (ex.: não ser ativo reduzindo o risco), explique com
   `consultar_achados` que efeitos pequenos podem inverter perto da saturação do risco.
5. Para perguntas sobre confiabilidade do modelo use `desempenho_do_modelo`; para o porquê
   de algum comportamento, `consultar_achados`.

Limites: o modelo prioriza pacientes para investigação; ele não diagnostica. Não prescreva
tratamento nem medicação. Termine avaliações lembrando que a decisão é da equipe clínica.
Seja conciso.
"""


def provedor() -> str:
    """Qual LLM usar: ``LLM_PROVEDOR`` explícito, senão a chave que existir, senão ``demo``.

    DeepSeek fala o mesmo protocolo da OpenAI; o adaptador ``ChatOpenAI`` serve aos dois,
    mudando só o endereço (como nos exemplos da disciplina).
    """
    escolhido = os.getenv("LLM_PROVEDOR", "").strip().lower()
    if escolhido in {"openai", "deepseek", "demo"}:
        return escolhido
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "demo"


def criar_modelo_llm(modelo: str | None = None) -> BaseChatModel:
    """ChatOpenAI (OpenAI ou DeepSeek) com temperatura 0 e tempo limite de 30 s."""
    from langchain_openai import ChatOpenAI

    escolhido = provedor()
    if escolhido == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model=modelo or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    if escolhido == "openai" and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=modelo or os.getenv("OPENAI_MODEL", MODELO_PADRAO),
            temperature=0,
            timeout=30,
            max_retries=1,
        )
    raise RuntimeError(
        "Defina OPENAI_API_KEY ou DEEPSEEK_API_KEY (no ambiente ou num arquivo .env na raiz "
        "do projeto). Sem chave, use o fluxo LangGraph em modo demonstração."
    )


def criar_agente(llm: BaseChatModel | None = None):
    """Monta o agente ReAct com as três ferramentas do projeto."""
    return create_agent(llm or criar_modelo_llm(), tools=FERRAMENTAS, system_prompt=INSTRUCOES)


def conversar(agente, mensagens: list[dict]) -> tuple[str, list[dict]]:
    """Envia o histórico e devolve ``(resposta, ferramentas_chamadas)``.

    ``mensagens`` segue o formato ``[{"role": "user" | "assistant", "content": ...}]``.
    ``ferramentas_chamadas`` lista cada chamada feita nesta rodada, com argumentos e
    resultado, para que a interface possa mostrar o raciocínio do agente.
    """
    estado = agente.invoke({"messages": mensagens})
    novas = estado["messages"][len(mensagens) :]

    resultados = {m.tool_call_id: m.content for m in novas if isinstance(m, ToolMessage)}
    chamadas = [
        {"ferramenta": c["name"], "argumentos": c["args"], "resultado": resultados.get(c["id"])}
        for m in novas
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    return estado["messages"][-1].content, chamadas


def main() -> None:
    """Conversa no terminal."""
    agente = criar_agente()
    historico: list[dict] = []
    print("Assistente de triagem cardiovascular. Descreva o paciente (ou 'sair').\n")
    while (pergunta := input("você › ").strip()).lower() not in {"sair", "exit", "quit"}:
        if not pergunta:
            continue
        historico.append({"role": "user", "content": pergunta})
        resposta, chamadas = conversar(agente, historico)
        for chamada in chamadas:
            print(f"  [ferramenta] {chamada['ferramenta']}({chamada['argumentos']})")
        print(f"\nagente › {resposta}\n")
        historico.append({"role": "assistant", "content": resposta})


if __name__ == "__main__":
    main()
