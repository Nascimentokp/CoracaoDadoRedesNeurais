"""Transformer para dados tabulares (no estilo do FT-Transformer), em Keras 3.

Em texto, cada palavra vira um *token* (um vetor de embedding) e a atenção
decide quais palavras olhar ao interpretar cada uma. Aqui, cada **variável do
paciente** vira um token:

* contínuas (idade, pressão…) — ``valor × w_j + b_j``: um vetor próprio de
  cada variável, esticado pelo valor padronizado;
* categóricas (colesterol, glicose, fumo…) — uma tabela de embeddings, um
  vetor por nível (é o que uma camada densa sem viés faz sobre o one-hot).

Um token extra, ``[CLS]``, não carrega dado nenhum: ele junta, pela atenção,
o que precisa das demais variáveis, e é dele que sai a previsão (o mesmo truque
do BERT). Não há codificação posicional: como cada variável tem pesos de
embedding próprios, o token já "sabe" de qual variável veio — a ordem das
colunas não significa nada.

Bloco do codificador (pré-normalização):
``x + Atenção(LN(x))`` → ``x + FFN(LN(x))``.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from src import config

#: Como as 16 colunas pré-processadas se agrupam nas 12 variáveis originais
#: (ordem do ColumnTransformer: contínuas, one-hot das ordinais, binárias).
GRUPOS: list[tuple[str, int, int]] = (
    [(nome, i, i + 1) for i, nome in enumerate(config.CONTINUAS)]
    + [
        (nome, len(config.CONTINUAS) + 3 * k, len(config.CONTINUAS) + 3 * (k + 1))
        for k, nome in enumerate(config.ORDINAIS)
    ]
    + [
        (nome, len(config.CONTINUAS) + 3 * len(config.ORDINAIS) + k,
         len(config.CONTINUAS) + 3 * len(config.ORDINAIS) + k + 1)
        for k, nome in enumerate(config.BINARIAS)
    ]
)  # fmt: skip

#: Rótulos dos tokens na ordem em que entram na atenção.
TOKENS = ["[CLS]"] + [nome for nome, _, _ in GRUPOS]


@keras.saving.register_keras_serializable(package="cardio")
class TokenCLS(layers.Layer):
    """Acrescenta um token aprendido no início da sequência."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.cls = self.add_weight(shape=(1, 1, self.dim), initializer="random_normal", name="cls")

    def call(self, tokens):
        n = ops.shape(tokens)[0]
        return ops.concatenate([ops.broadcast_to(self.cls, (n, 1, self.dim)), tokens], axis=1)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


def construir_transformer(
    n_entradas: int = 16,
    dim: int = 32,
    cabecas: int = 4,
    blocos: int = 2,
    dropout: float = 0.1,
    taxa: float = 1e-3,
) -> tuple[keras.Model, keras.Model]:
    """Monta o Transformer tabular.

    Devolve ``(modelo, modelo_atencao)``: o segundo compartilha os pesos do
    primeiro e devolve os pesos de atenção de cada bloco, com forma
    ``(pacientes, cabecas, 13 tokens, 13 tokens)``.
    """
    entrada = keras.Input(shape=(n_entradas,), name="variaveis")

    # 1. Tokenização: uma camada densa (sem ativação) por variável.
    tokens = []
    for nome, inicio, fim in GRUPOS:
        fatia = entrada[:, inicio:fim]
        token = layers.Dense(dim, name=f"embedding_{nome}")(fatia)
        tokens.append(layers.Reshape((1, dim))(token))
    x = layers.Concatenate(axis=1, name="tokens")(tokens)
    x = TokenCLS(dim, name="token_cls")(x)

    # 2. Blocos do codificador.
    mapas = []
    for b in range(blocos):
        normalizado = layers.LayerNormalization(name=f"ln_atencao_{b}")(x)
        atencao, pesos = layers.MultiHeadAttention(
            num_heads=cabecas, key_dim=dim // cabecas, dropout=dropout, name=f"atencao_{b}"
        )(normalizado, normalizado, return_attention_scores=True)
        x = layers.Add()([x, layers.Dropout(dropout)(atencao)])
        mapas.append(pesos)

        normalizado = layers.LayerNormalization(name=f"ln_ffn_{b}")(x)
        ffn = layers.Dense(2 * dim, activation="relu")(normalizado)
        ffn = layers.Dense(dim)(layers.Dropout(dropout)(ffn))
        x = layers.Add()([x, ffn])

    # 3. A previsão sai só do token [CLS].
    cls = layers.LayerNormalization(name="ln_saida")(x[:, 0, :])
    saida = layers.Dense(1, activation="sigmoid", name="p_cardio")(cls)

    modelo = keras.Model(entrada, saida, name="transformer_tabular")
    modelo.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=taxa, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall"), "accuracy"],
    )
    modelo_atencao = keras.Model(entrada, mapas, name="mapas_de_atencao")
    return modelo, modelo_atencao


def atencao_media(modelo_atencao: keras.Model, X: np.ndarray, bloco: int = -1) -> np.ndarray:
    """Matriz 13 × 13 de atenção média (pacientes e cabeças) de um bloco.

    Linha = token que "pergunta" (query); coluna = token consultado (key).
    """
    mapas = modelo_atencao.predict(np.asarray(X, dtype=np.float32), verbose=0, batch_size=2048)
    if not isinstance(mapas, list):
        mapas = [mapas]
    return np.asarray(mapas[bloco]).mean(axis=(0, 1))
