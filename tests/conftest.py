"""Configuração comum dos testes."""

import pytest


@pytest.fixture(autouse=True)
def sem_llm_pago(monkeypatch):
    """Nenhum teste chama uma API paga: o .env pode ter chave, mas os testes usam o modo demo."""
    monkeypatch.setenv("LLM_PROVEDOR", "demo")
