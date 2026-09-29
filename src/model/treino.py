"""Treino e persistência do modelo final consumido pela aplicação Streamlit.

Executado por `uv run invoke treinar` ou pelo notebook 05. Grava em `models/`:

* `mlp_cardio.keras` — a rede treinada;
* `preprocessador.joblib` — o ColumnTransformer ajustado no treino;
* `metadados.json` — arquitetura, limiar escolhido na validação, métricas no
  teste e perfil de referência usado nas explicações da aplicação.
"""

from __future__ import annotations

import joblib
import keras
import pandas as pd

from src import config
from src.data.dados import Particao, preparar
from src.model import avaliacao, mlp_keras
from src.utils.io import ler_json, salvar_json

ARQUIVO_MODELO = config.MODELS / "mlp_cardio.keras"
ARQUIVO_PREPROCESSADOR = config.MODELS / "preprocessador.joblib"
ARQUIVO_METADADOS = config.MODELS / "metadados.json"

#: Arquitetura escolhida no notebook 04 (empate técnico com as demais; poucos parâmetros).
ARQUITETURA = {"ocultas": (64, 32, 16), "dropout": 0.2, "otimizador": "Adam", "taxa": 1e-3}


def treinar_modelo_final(particao: Particao | None = None, verbose: int = 0) -> dict:
    """Treina o MLP final, escolhe o limiar na validação e grava os artefatos."""
    config.garantir_diretorios()
    particao = particao or preparar()

    modelo = mlp_keras.construir_mlp(particao.n_entradas, **ARQUITETURA)
    historico = mlp_keras.treinar(
        modelo,
        particao.X_treino,
        particao.y_treino,
        particao.X_val,
        particao.y_val,
        verbose=verbose,
    )

    p_val = mlp_keras.prever(modelo, particao.X_val)
    p_teste = mlp_keras.prever(modelo, particao.X_teste)
    limiar = avaliacao.escolher_limiar(particao.y_val, p_val)

    referencia = particao.bruto_treino
    metadados = {
        "arquitetura": {**ARQUITETURA, "ocultas": list(ARQUITETURA["ocultas"])},
        "n_parametros": modelo.count_params(),
        "epocas": len(historico.history["loss"]),
        "limiar": limiar,
        "recall_minimo": config.RECALL_MINIMO,
        "teste_limiar_05": avaliacao.metricas(particao.y_teste, p_teste, 0.5),
        "teste_limiar_escolhido": avaliacao.metricas(particao.y_teste, p_teste, limiar),
        "n_treino": len(particao.y_treino),
        "n_validacao": len(particao.y_val),
        "n_teste": len(particao.y_teste),
        "referencia": {
            coluna: (
                float(referencia[coluna].median())
                if coluna in config.CONTINUAS
                else int(referencia[coluna].mode().iloc[0])
            )
            for coluna in config.ENTRADAS
        },
    }

    modelo.save(ARQUIVO_MODELO)
    joblib.dump(particao.preprocessador, ARQUIVO_PREPROCESSADOR)
    salvar_json(metadados, ARQUIVO_METADADOS)
    return metadados


def carregar_modelo_final() -> tuple[keras.Model, object, dict]:
    """Carrega rede, pré-processador e metadados gravados por :func:`treinar_modelo_final`."""
    modelo = keras.models.load_model(ARQUIVO_MODELO)
    preprocessador = joblib.load(ARQUIVO_PREPROCESSADOR)
    return modelo, preprocessador, ler_json(ARQUIVO_METADADOS)


def prever_bruto(modelo: keras.Model, preprocessador, pacientes: pd.DataFrame):
    """Probabilidade a partir de variáveis na escala original (sem padronização)."""
    X = preprocessador.transform(pacientes[config.ENTRADAS]).astype("float32")
    return mlp_keras.prever(modelo, X)
