"""Esquema Pydantic do paciente (specs/01-triagem.md, VAL-01 a VAL-03)."""

import pytest
from pydantic import ValidationError

from src.agente.esquemas import Paciente, PacienteParcial, validar

VALIDO = dict(
    idade=58, sexo="masculino", altura_cm=172, peso_kg=94, pressao_sistolica=150,
    pressao_diastolica=95, colesterol=2, glicose=1, fumante=True, consome_alcool=False,
    fisicamente_ativo=False,
)  # fmt: skip


@pytest.mark.parametrize(
    "alteracao",
    [
        {"idade": 5},
        {"altura_cm": 300},
        {"pressao_sistolica": 400},
        {"pressao_diastolica": 160},  # ≥ sistólica
        {"colesterol": 4},
        {"sexo": "x"},
        {"campo_extra": 1},
    ],
)
def test_rejeita_valores_impossiveis_e_campos_extras(alteracao):
    """VAL-01."""
    with pytest.raises(ValidationError):
        Paciente.model_validate({**VALIDO, **alteracao})


def test_normaliza_formatos_humanos():
    """VAL-02."""
    p = Paciente.model_validate(
        {**VALIDO, "idade": "58 anos", "sexo": "homem", "altura_cm": "1,72 m", "peso_kg": "94 kg",
         "pressao_sistolica": 15, "pressao_diastolica": "9,5", "colesterol": "acima do normal",
         "glicose": "normais"}
    )  # fmt: skip
    assert (p.idade, p.sexo, p.altura_cm, p.peso_kg) == (58, "masculino", 172, 94)
    assert (p.pressao_sistolica, p.pressao_diastolica) == (150, 95)
    assert (p.colesterol, p.glicose) == (2, 1)


def test_fora_da_faixa_do_treino_gera_aviso_e_nao_erro():
    """VAL-03."""
    p = Paciente.model_validate({**VALIDO, "idade": 78})
    assert any("extrapolação" in aviso for aviso in p.avisos())
    assert Paciente.model_validate(VALIDO).avisos() == []


def test_validar_separa_faltando_de_invalido():
    """FLX-05 / VAL-01: faltando e inválido são caminhos diferentes."""
    paciente, faltando, erros = validar(PacienteParcial(idade=61, pressao_texto="14 por 9"))
    assert paciente is None and erros == [] and "peso" in faltando

    completo = PacienteParcial(**{k: v for k, v in VALIDO.items() if not k.startswith("pressao")})
    paciente, faltando, erros = validar(completo.juntar(PacienteParcial(pressao_texto="12 por 13")))
    assert paciente is None and faltando == [] and erros

    paciente, _, _ = validar(completo.juntar(PacienteParcial(pressao_texto="15 por 9,5")))
    assert (paciente.pressao_sistolica, paciente.pressao_diastolica) == (150, 95)
