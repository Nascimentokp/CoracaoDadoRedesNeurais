"""Autoencoder denso: a rede aprende a comprimir o paciente sem ver o diagnóstico.

O codificador espreme as 16 entradas num gargalo de poucas dimensões (o
*espaço latente*); o decodificador tenta reconstruir as 16 entradas a partir
dele. A única supervisão é a própria entrada — aprendizado **não
supervisionado**, como o agrupamento do projeto Tema 6, só que com uma rede.

Dois usos no notebook 08:

* **mapa dos pacientes** — com gargalo de 2 dimensões, cada paciente vira um
  ponto num plano, que pode ser comparado com o PCA (a versão linear da mesma
  ideia) e colorido pelo diagnóstico que a rede nunca viu;
* **registros atípicos** — pacientes que a rede reconstrói mal são os que
  fogem do padrão da base (combinações raras ou suspeitas de erro de digitação).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers

from src import config


def construir_autoencoder(
    n_entradas: int = 16,
    ocultas: tuple[int, ...] = (32, 16),
    latente: int = 2,
    taxa: float = 1e-3,
) -> tuple[keras.Model, keras.Model]:
    """Monta o autoencoder e devolve ``(autoencoder, codificador)``.

    O decodificador espelha o codificador. A saída é linear e a perda é o erro
    quadrático médio: as entradas já estão padronizadas ou em 0/1.
    """
    entrada = keras.Input(shape=(n_entradas,), name="paciente")
    x = entrada
    for neuronios in ocultas:
        x = layers.Dense(neuronios, activation="relu")(x)
    codigo = layers.Dense(latente, name="espaco_latente")(x)

    x = codigo
    for neuronios in reversed(ocultas):
        x = layers.Dense(neuronios, activation="relu")(x)
    reconstrucao = layers.Dense(n_entradas, name="reconstrucao")(x)

    autoencoder = keras.Model(entrada, reconstrucao, name="autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=taxa), loss="mse")
    codificador = keras.Model(entrada, codigo, name="codificador")
    return autoencoder, codificador


def treinar(
    autoencoder: keras.Model,
    X_treino: np.ndarray,
    X_val: np.ndarray,
    epocas: int = 200,
    lote: int = 256,
    paciencia: int = 10,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Treina a reconstruir a entrada, com parada antecipada pela perda de validação."""
    keras.utils.set_random_seed(config.SEMENTE)
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=paciencia, restore_best_weights=True
    )
    return autoencoder.fit(
        X_treino,
        X_treino,  # o alvo é a própria entrada
        validation_data=(X_val, X_val),
        epochs=epocas,
        batch_size=lote,
        callbacks=[parada],
        verbose=verbose,
    )


def erro_reconstrucao(autoencoder: keras.Model, X: np.ndarray) -> np.ndarray:
    """Erro quadrático médio de reconstrução de cada paciente."""
    X = np.asarray(X, dtype=np.float32)
    reconstruido = autoencoder.predict(X, verbose=0, batch_size=4096)
    return ((X - reconstruido) ** 2).mean(axis=1)
