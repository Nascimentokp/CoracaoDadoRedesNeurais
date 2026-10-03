"""Esquemas Pydantic: o contrato dos dados de um paciente.

Toda entrada — formulário, texto livre extraído por um LLM ou pelo modo
demonstração — passa por aqui antes de chegar à rede neural. O esquema:

* **normaliza** o que chega em formato humano ("1,72 m" → 172 cm, "homem" →
  masculino, "acima do normal" → 2, "94 kg" → 94);
* **rejeita** valores fisiologicamente impossíveis e pressão diastólica maior
  ou igual à sistólica;
* **avisa** (sem rejeitar) valores plausíveis mas fora da faixa vista no treino.

Quando o esquema é a assinatura de uma ferramenta do LangChain, os limites
viram parte do JSON Schema que o LLM lê; se ele mandar um valor inválido, o
erro de validação volta para ele corrigir (specs/01-triagem.md, VAL-03).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.utils.formato import numero

#: Faixas vistas no treino. Fora delas a rede extrapola (aviso, não erro).
FAIXAS_TREINO = {
    "idade": (30, 65),
    "altura_cm": (130, 207),
    "peso_kg": (30, 183),
    "pressao_sistolica": (60, 240),
    "pressao_diastolica": (40, 160),
}

NIVEIS = {1: "normal", 2: "acima do normal", 3: "muito acima do normal"}

#: Nome legível de cada campo, para mensagens ao usuário.
NOMES = {
    "idade": "idade",
    "sexo": "sexo",
    "altura_cm": "altura",
    "peso_kg": "peso",
    "pressao_sistolica": "pressão sistólica",
    "pressao_diastolica": "pressão diastólica",
    "colesterol": "colesterol (normal, acima do normal ou muito acima)",
    "glicose": "glicose (normal, acima do normal ou muito acima)",
    "fumante": "se fuma",
    "consome_alcool": "se consome álcool",
    "fisicamente_ativo": "se pratica atividade física",
}


def _numero(valor):
    """'1,72 m', '94 kg', '58 anos' → float. Números passam direto."""
    if isinstance(valor, str):
        achado = re.search(r"\d+(?:[.,]\d+)?", valor)
        if not achado:
            return valor  # deixa o Pydantic acusar o erro de tipo
        return float(achado.group().replace(",", "."))
    return valor


def sexo(valor):
    """'homem', 'm', 'masc' → masculino; 'mulher', 'f', 'fem' → feminino."""
    if isinstance(valor, str):
        texto = valor.strip().lower()
        if texto in {"m", "masc", "homem", "masculino"}:
            return "masculino"
        if texto in {"f", "fem", "mulher", "feminino"}:
            return "feminino"
    return valor


def nivel(valor):
    """'normal' → 1, 'acima do normal' / 'alto' → 2, 'muito acima' / 'muito alto' → 3."""
    if isinstance(valor, str):
        texto = valor.strip().lower()
        if texto.isdigit():
            return int(texto)
        if "muito" in texto:
            return 3
        if re.search(r"acima|\balt[oa]s?\b|elevad|lim[ií]trofe", texto):
            return 2
        if "norma" in texto:  # normal, normais
            return 1
    return valor


class Paciente(BaseModel):
    """Os 11 dados que a rede usa. Todos obrigatórios."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    idade: float = Field(ge=18, le=110, description="Idade em anos")
    sexo: Literal["feminino", "masculino"]
    altura_cm: float = Field(ge=100, le=230, description="Altura em centímetros (1,72 m = 172)")
    peso_kg: float = Field(ge=30, le=250, description="Peso em quilogramas")
    pressao_sistolica: int = Field(ge=60, le=260, description="Pressão sistólica em mmHg (150)")
    pressao_diastolica: int = Field(ge=30, le=200, description="Pressão diastólica em mmHg (95)")
    colesterol: Literal[1, 2, 3] = Field(description="1 normal, 2 acima do normal, 3 muito acima")
    glicose: Literal[1, 2, 3] = Field(description="1 normal, 2 acima do normal, 3 muito acima")
    fumante: bool
    consome_alcool: bool
    fisicamente_ativo: bool

    @field_validator("idade", "peso_kg", mode="before")
    @classmethod
    def _texto_para_numero(cls, valor):
        return _numero(valor)

    @field_validator("altura_cm", mode="before")
    @classmethod
    def _altura_em_metros(cls, valor):
        valor = _numero(valor)
        if isinstance(valor, int | float) and 0.5 < valor < 2.5:
            return round(valor * 100, 1)  # 1,72 → 172
        return valor

    @field_validator("pressao_sistolica", "pressao_diastolica", mode="before")
    @classmethod
    def _cmhg_para_mmhg(cls, valor):
        valor = _numero(valor)
        if isinstance(valor, int | float) and valor < 30:
            return round(valor * 10)  # "15 por 9" → 150/90
        return round(valor) if isinstance(valor, float) else valor

    @field_validator("sexo", mode="before")
    @classmethod
    def _sexo(cls, valor):
        return sexo(valor)

    @field_validator("colesterol", "glicose", mode="before")
    @classmethod
    def _nivel(cls, valor):
        return nivel(valor)

    @model_validator(mode="after")
    def _pressao_coerente(self) -> Paciente:
        if self.pressao_diastolica >= self.pressao_sistolica:
            raise ValueError("a pressão diastólica precisa ser menor que a sistólica")
        return self

    @property
    def imc(self) -> float:
        return self.peso_kg / (self.altura_cm / 100) ** 2

    def avisos(self) -> list[str]:
        """Valores fora da faixa do treino: a previsão vira extrapolação."""
        return [
            f"{nome} = {numero(getattr(self, nome), compacto=True)} está fora da faixa vista no "
            f"treino ({mi}–{ma}); "
            "a previsão é uma extrapolação e merece menos confiança."
            for nome, (mi, ma) in FAIXAS_TREINO.items()
            if not mi <= getattr(self, nome) <= ma
        ]


class PacienteParcial(BaseModel):
    """O que se conseguiu extrair até agora de uma conversa — tudo opcional.

    ``pressao_texto`` guarda o trecho da pressão como foi escrito ("15 por 9,5"):
    a conversão é feita em código, não pelo LLM (specs, FLX-03).
    """

    model_config = ConfigDict(extra="ignore")

    idade: float | None = None
    sexo: Literal["feminino", "masculino"] | None = None
    altura_cm: float | None = None
    peso_kg: float | None = None
    pressao_texto: str | None = Field(
        default=None, description='Trecho da pressão exatamente como escrito, ex. "15 por 9,5"'
    )
    colesterol: Literal[1, 2, 3] | None = None
    glicose: Literal[1, 2, 3] | None = None
    fumante: bool | None = None
    consome_alcool: bool | None = None
    fisicamente_ativo: bool | None = None

    @field_validator("idade", "altura_cm", "peso_kg", mode="before")
    @classmethod
    def _texto_para_numero(cls, valor):
        return _numero(valor)

    @field_validator("sexo", mode="before")
    @classmethod
    def _sexo(cls, valor):
        return sexo(valor)

    @field_validator("colesterol", "glicose", mode="before")
    @classmethod
    def _nivel(cls, valor):
        return nivel(valor)

    def juntar(self, novo: PacienteParcial) -> PacienteParcial:
        """Valores novos substituem os antigos; ausentes mantêm o que já se sabia."""
        return self.model_copy(update=novo.model_dump(exclude_none=True))


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


def validar(parcial: PacienteParcial) -> tuple[Paciente | None, list[str], list[str]]:
    """Tenta montar um ``Paciente`` completo.

    Devolve ``(paciente, faltando, erros)``: só um dos três caminhos acontece —
    paciente válido, campos faltando (perguntar) ou valores inválidos (corrigir).
    """
    dados = parcial.model_dump(exclude_none=True)
    erros: list[str] = []
    if texto := dados.pop("pressao_texto", None):
        pressao = interpretar_pressao(texto)
        if "erro" in pressao:
            erros.append(pressao["erro"])
        else:
            dados.update(pressao)

    faltando = [NOMES[c] for c in Paciente.model_fields if c not in dados]
    if "pressão sistólica" in faltando and not erros:
        faltando = [f for f in faltando if not f.startswith("pressão")] + ["pressão arterial"]
    if faltando or erros:
        return None, faltando, erros
    try:
        return Paciente.model_validate(dados), [], []
    except ValidationError as erro:
        return None, [], traduzir_erros(erro)


def traduzir_erros(erro: ValidationError) -> list[str]:
    """Mensagens de validação em português, sem jargão do Pydantic."""
    mensagens = []
    for item in erro.errors():
        campo = NOMES.get(str(item["loc"][0]), "dados") if item["loc"] else "dados"
        contexto = item.get("ctx", {})
        if item["type"] == "greater_than_equal":
            mensagens.append(f"{campo}: valor abaixo do mínimo plausível ({contexto['ge']}).")
        elif item["type"] == "less_than_equal":
            mensagens.append(f"{campo}: valor acima do máximo plausível ({contexto['le']}).")
        elif item["type"] == "value_error":
            mensagens.append(str(contexto.get("error", item["msg"])).capitalize() + ".")
        else:
            mensagens.append(f"{campo}: valor não reconhecido.")
    return mensagens
