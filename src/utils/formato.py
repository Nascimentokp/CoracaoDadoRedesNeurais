"""Números no padrão brasileiro: vírgula decimal.

A conversão é feita **no número formatado**, nunca na frase pronta: trocar o primeiro
ponto de uma frase pega o ponto errado quando há outro número antes ("IMC 27.5 … +1.8").
"""


def numero(valor: float, casas: int = 1, sinal: bool = False) -> str:
    """27.53 → '27,5'; com sinal, 1.8 → '+1,8'."""
    return f"{valor:{'+' if sinal else ''}.{casas}f}".replace(".", ",")


def pct(valor: float, casas: int = 0, sinal: bool = False) -> str:
    """0.808 → '81%'; com casas=1, '80,8%'; com sinal, '+0,4%'."""
    return f"{valor:{'+' if sinal else ''}.{casas}%}".replace(".", ",")
