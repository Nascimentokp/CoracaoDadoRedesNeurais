"""Rede densa em Keras 3 (TensorFlow) para P(cardio = 1).

Receita dos exercícios da disciplina (câncer de mama, crédito alemão):
Dense + ReLU com Dropout entre as camadas, saída sigmoide, perda de entropia
cruzada binária, EarlyStopping monitorando a AUC de validação com
``restore_best_weights=True`` e, se a base fosse desbalanceada, class_weight.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from src import config

# Os notebooks criam várias redes em sequência; o aviso de "retracing" do
# TensorFlow nesse cenário é esperado e só polui a saída.
tf.get_logger().setLevel("ERROR")

OTIMIZADORES = {
    "SGD": lambda taxa: keras.optimizers.SGD(learning_rate=taxa),
    "SGD + momentum": lambda taxa: keras.optimizers.SGD(
        learning_rate=taxa, momentum=0.9, nesterov=True
    ),
    "RMSprop": lambda taxa: keras.optimizers.RMSprop(learning_rate=taxa),
    "Adam": lambda taxa: keras.optimizers.Adam(learning_rate=taxa),
    "AdamW": lambda taxa: keras.optimizers.AdamW(learning_rate=taxa, weight_decay=1e-4),
    "Nadam": lambda taxa: keras.optimizers.Nadam(learning_rate=taxa),
}

#: Taxa de aprendizado padrão de cada otimizador (SGD puro precisa de mais).
TAXA_PADRAO = {"SGD": 0.05, "SGD + momentum": 0.01}


def construir_mlp(
    n_entradas: int,
    ocultas: tuple[int, ...] = (64, 32, 16),
    dropout: float = 0.2,
    batch_norm: bool = False,
    otimizador: str = "Adam",
    taxa: float | keras.optimizers.schedules.LearningRateSchedule | None = None,
) -> keras.Model:
    """Monta e compila o MLP.

    ``taxa`` aceita um número fixo ou uma agenda (ex.: ``CosineDecay``), que
    muda a taxa de aprendizado ao longo do treino.
    """
    camadas: list[keras.layers.Layer] = [keras.Input(shape=(n_entradas,))]
    for neuronios in ocultas:
        camadas.append(keras.layers.Dense(neuronios, activation="relu"))
        if batch_norm:
            camadas.append(keras.layers.BatchNormalization())
        if dropout:
            # Dropout só age no treino: desliga neurônios ao acaso para a
            # rede não memorizar pacientes específicos.
            camadas.append(keras.layers.Dropout(dropout))
    camadas.append(keras.layers.Dense(1, activation="sigmoid"))  # P(cardio = 1)

    modelo = keras.Sequential(camadas, name="mlp_cardio")
    taxa = taxa if taxa is not None else TAXA_PADRAO.get(otimizador, 1e-3)
    modelo.compile(
        optimizer=OTIMIZADORES[otimizador](taxa),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Recall(name="recall"),
            "accuracy",
        ],
    )
    return modelo


def pesos_de_classe(y: np.ndarray) -> dict[int, float]:
    """Peso inversamente proporcional à frequência de cada classe."""
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return {0: 1.0, 1: n_neg / n_pos}


def treinar(
    modelo: keras.Model,
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 200,
    lote: int = 256,
    paciencia: int = 15,
    class_weight: dict[int, float] | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Treina com parada antecipada pela AUC de validação."""
    keras.utils.set_random_seed(config.SEMENTE)
    parada = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=paciencia, restore_best_weights=True
    )
    return modelo.fit(
        X_treino,
        y_treino,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=lote,
        class_weight=class_weight,
        callbacks=[parada, *(callbacks or [])],
        verbose=verbose,
    )


def prever(modelo: keras.Model, X: np.ndarray) -> np.ndarray:
    """Probabilidade de doença cardiovascular, como vetor 1-D."""
    return modelo.predict(np.asarray(X, dtype=np.float32), verbose=0, batch_size=4096).ravel()


def modelo_para_busca(hp, n_entradas: int) -> keras.Model:
    """Espaço de busca do Keras Tuner: profundidade, largura, Dropout, BatchNorm e taxa.

    As camadas encolhem pela metade a cada nível (ex.: 128 → 64 → 32), o mesmo
    formato de funil das arquiteturas do notebook 04.
    """
    camadas = hp.Int("camadas", 1, 4)
    largura = hp.Choice("largura_inicial", [16, 32, 64, 128, 256])
    ocultas = tuple(max(largura // 2**i, 8) for i in range(camadas))
    return construir_mlp(
        n_entradas,
        ocultas=ocultas,
        dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
        batch_norm=hp.Boolean("batch_norm"),
        otimizador="Adam",
        taxa=hp.Float("taxa", 1e-4, 1e-2, sampling="log"),
    )
