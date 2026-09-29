"""Rede neural multicamada escrita do zero, só com NumPy.

Serve para abrir a "caixa-preta" antes de usar Keras e PyTorch: cada passo
— soma ponderada, ativação, perda, gradiente e atualização dos pesos — está
escrito à mão, como nos exercícios da disciplina (rede que aprende a soma,
perceptron sigmoide e MLP do EEG com inicialização He e regularização L2).

Arquitetura: entrada → [Dense + ReLU] × n → Dense + sigmoide → P(cardio = 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-7


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivada(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoide(z: np.ndarray) -> np.ndarray:
    # Recorte evita overflow de exp() para |z| muito grande.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def entropia_cruzada_binaria(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class MLPNumpy:
    """MLP binário treinado por gradiente descendente em mini-lotes.

    Args:
        n_entradas: número de variáveis de entrada.
        ocultas: neurônios em cada camada oculta, ex. ``(32, 16)``.
        taxa: taxa de aprendizado.
        l2: força da regularização L2 (penaliza pesos grandes).
        semente: semente da inicialização.
    """

    n_entradas: int
    ocultas: tuple[int, ...] = (32, 16)
    taxa: float = 0.05
    l2: float = 1e-4
    semente: int = 42
    pesos: list[np.ndarray] = field(init=False, repr=False)
    vieses: list[np.ndarray] = field(init=False, repr=False)
    historico: dict[str, list[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.semente)
        tamanhos = [self.n_entradas, *self.ocultas, 1]
        # Inicialização He: variância 2/n_entrada mantém a escala do sinal
        # estável ao atravessar camadas ReLU.
        self.pesos = [
            rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)).astype(np.float32)
            for n_in, n_out in zip(tamanhos[:-1], tamanhos[1:])
        ]
        self.vieses = [np.zeros((1, n_out), dtype=np.float32) for n_out in tamanhos[1:]]
        self.historico = {"perda": [], "perda_val": []}

    # ------------------------------------------------------------------ #
    # Propagação
    # ------------------------------------------------------------------ #

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Devolve as pré-ativações (z) e ativações (a) de cada camada."""
        ativacoes, pre = [X], []
        for i, (W, b) in enumerate(zip(self.pesos, self.vieses)):
            z = ativacoes[-1] @ W + b
            pre.append(z)
            ultima = i == len(self.pesos) - 1
            ativacoes.append(sigmoide(z) if ultima else relu(z))
        return pre, ativacoes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[1][-1].ravel()

    def _backward(self, pre, ativacoes, y) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Regra da cadeia, da saída para a entrada."""
        n = y.shape[0]
        # Sigmoide + entropia cruzada: o gradiente em z da saída é simplesmente p - y.
        delta = ativacoes[-1] - y.reshape(-1, 1)
        grad_W, grad_b = [], []
        for camada in reversed(range(len(self.pesos))):
            grad_W.insert(0, ativacoes[camada].T @ delta / n + self.l2 * self.pesos[camada])
            grad_b.insert(0, delta.mean(axis=0, keepdims=True))
            if camada > 0:
                delta = (delta @ self.pesos[camada].T) * relu_derivada(pre[camada - 1])
        return grad_W, grad_b

    # ------------------------------------------------------------------ #
    # Treino
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epocas: int = 100,
        lote: int = 256,
        paciencia: int = 10,
        verbose: bool = False,
    ) -> MLPNumpy:
        """Treina com mini-lotes embaralhados e parada antecipada na validação."""
        rng = np.random.default_rng(self.semente)
        melhor, sem_melhora, melhor_estado = np.inf, 0, None

        for epoca in range(epocas):
            ordem = rng.permutation(len(X))
            for inicio in range(0, len(X), lote):
                idx = ordem[inicio : inicio + lote]
                pre, ativ = self._forward(X[idx])
                grad_W, grad_b = self._backward(pre, ativ, y[idx])
                for i in range(len(self.pesos)):
                    self.pesos[i] -= self.taxa * grad_W[i]
                    self.vieses[i] -= self.taxa * grad_b[i]

            perda = entropia_cruzada_binaria(y, self.predict_proba(X))
            self.historico["perda"].append(perda)
            if X_val is None:
                continue

            perda_val = entropia_cruzada_binaria(y_val, self.predict_proba(X_val))
            self.historico["perda_val"].append(perda_val)
            if verbose and epoca % 10 == 0:
                print(f"época {epoca:3d}  perda {perda:.4f}  perda_val {perda_val:.4f}")

            if perda_val < melhor - 1e-5:
                melhor, sem_melhora = perda_val, 0
                melhor_estado = ([W.copy() for W in self.pesos], [b.copy() for b in self.vieses])
            else:
                sem_melhora += 1
                if sem_melhora >= paciencia:
                    break

        # Equivalente ao restore_best_weights=True do Keras.
        if melhor_estado is not None:
            self.pesos, self.vieses = melhor_estado
        return self

    @property
    def n_parametros(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.pesos, self.vieses))
