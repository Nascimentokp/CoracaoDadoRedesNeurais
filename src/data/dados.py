"""Carga, divisão e pré-processamento da base curada.

Todo modelo do projeto — NumPy, Keras, PyTorch ou baseline clássico — recebe
exatamente as mesmas partições e o mesmo pré-processador, para que a
comparação entre eles seja justa.

Regra de ouro (repetida em todos os exercícios da disciplina): o
pré-processador é ajustado **somente no treino** e apenas aplicado em
validação e teste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def carregar_base() -> pd.DataFrame:
    """Lê a base curada e deriva as variáveis usadas pelas redes.

    * `imc` — peso / altura², mais informativo que peso e altura isolados;
    * `sexo_masculino` — recodifica `gender` (1/2) como 0/1.
    """
    base = pd.read_csv(config.BASE_CURADA, sep=";")
    return derivar_variaveis(base)


def derivar_variaveis(base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as variáveis derivadas a um DataFrame com as colunas originais."""
    base = base.copy()
    base["imc"] = base["weight"] / (base["height"] / 100) ** 2
    base["sexo_masculino"] = (base["gender"] == 2).astype(int)
    return base


def criar_preprocessador() -> ColumnTransformer:
    """Padroniza contínuas, faz one-hot das ordinais e repassa as binárias."""
    return ColumnTransformer(
        [
            ("continuas", StandardScaler(), config.CONTINUAS),
            (
                "ordinais",
                OneHotEncoder(categories=[[1, 2, 3]] * len(config.ORDINAIS), dtype=np.float32),
                config.ORDINAIS,
            ),
            ("binarias", "passthrough", config.BINARIAS),
        ],
        verbose_feature_names_out=False,
    )


@dataclass
class Particao:
    """Partições prontas para treino, já transformadas em `float32`."""

    X_treino: np.ndarray
    X_val: np.ndarray
    X_teste: np.ndarray
    y_treino: np.ndarray
    y_val: np.ndarray
    y_teste: np.ndarray
    #: Mesmas partições antes do pré-processamento (para explicações e app).
    bruto_treino: pd.DataFrame
    bruto_val: pd.DataFrame
    bruto_teste: pd.DataFrame
    preprocessador: ColumnTransformer
    nomes: list[str]

    @property
    def n_entradas(self) -> int:
        return self.X_treino.shape[1]


def dividir(base: pd.DataFrame, semente: int = config.SEMENTE) -> Particao:
    """Divide 60/20/20 estratificado e ajusta o pré-processador no treino."""
    X = base[config.ENTRADAS]
    y = base[config.ALVO].to_numpy(dtype=np.float32)

    X_resto, X_teste, y_resto, y_teste = train_test_split(
        X, y, test_size=config.PROPORCAO_TESTE, random_state=semente, stratify=y
    )
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_resto,
        y_resto,
        test_size=config.PROPORCAO_VALIDACAO,
        random_state=semente,
        stratify=y_resto,
    )

    prep = criar_preprocessador()
    prep.fit(X_treino)

    def transformar(parte: pd.DataFrame) -> np.ndarray:
        return prep.transform(parte).astype(np.float32)

    return Particao(
        X_treino=transformar(X_treino),
        X_val=transformar(X_val),
        X_teste=transformar(X_teste),
        y_treino=y_treino,
        y_val=y_val,
        y_teste=y_teste,
        bruto_treino=X_treino.reset_index(drop=True),
        bruto_val=X_val.reset_index(drop=True),
        bruto_teste=X_teste.reset_index(drop=True),
        preprocessador=prep,
        nomes=list(prep.get_feature_names_out()),
    )


def preparar(semente: int = config.SEMENTE) -> Particao:
    """Atalho usado pelos notebooks: carrega a base e devolve as partições."""
    return dividir(carregar_base(), semente)
