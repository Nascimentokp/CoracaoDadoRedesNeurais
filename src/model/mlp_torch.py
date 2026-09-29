"""A mesma rede densa em PyTorch, com o laço de treino escrito à mão.

Em Keras, ``fit`` esconde o laço; aqui ele aparece inteiro — zerar
gradientes, forward, perda, backward, passo do otimizador — como no exercício
da Iris em PyTorch. A saída é um *logit* e a perda é
``BCEWithLogitsLoss``, numericamente mais estável que sigmoide + BCE.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


def fixar_semente(semente: int = config.SEMENTE) -> None:
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CardioNet(nn.Module):
    """Entrada → [Linear + ReLU + Dropout] × n → Linear (logit)."""

    def __init__(self, n_entradas: int, ocultas=(64, 32, 16), dropout: float = 0.2):
        super().__init__()
        camadas: list[nn.Module] = []
        anterior = n_entradas
        for neuronios in ocultas:
            camadas += [nn.Linear(anterior, neuronios), nn.ReLU(), nn.Dropout(dropout)]
            anterior = neuronios
        camadas.append(nn.Linear(anterior, 1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x).squeeze(-1)


def prever(modelo: nn.Module, X: np.ndarray) -> np.ndarray:
    """Probabilidade de doença cardiovascular (sigmoide do logit)."""
    disp = next(modelo.parameters()).device
    modelo.eval()
    with torch.no_grad():
        logits = modelo(torch.as_tensor(X, dtype=torch.float32, device=disp))
        return torch.sigmoid(logits).cpu().numpy()


def treinar(
    modelo: nn.Module,
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 200,
    lote: int = 256,
    taxa: float = 1e-3,
    paciencia: int = 15,
    disp: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Treina com Adam e parada antecipada pela AUC de validação."""
    fixar_semente()
    disp = disp or torch.device("cpu")
    modelo.to(disp)

    carregador = DataLoader(
        TensorDataset(torch.as_tensor(X_treino), torch.as_tensor(y_treino)),
        batch_size=lote,
        shuffle=True,
    )
    X_val_t = torch.as_tensor(X_val, device=disp)
    y_val_t = torch.as_tensor(y_val, device=disp)
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)

    historico = {"perda": [], "perda_val": [], "auc_val": []}
    melhor_auc, sem_melhora, melhor_estado = -np.inf, 0, None

    for epoca in range(epocas):
        modelo.train()
        soma = 0.0
        for xb, yb in carregador:
            xb, yb = xb.to(disp), yb.to(disp)
            otimizador.zero_grad()  # 1. zera gradientes acumulados
            perda = criterio(modelo(xb), yb)  # 2. forward + perda
            perda.backward()  # 3. backpropagation
            otimizador.step()  # 4. atualiza os pesos
            soma += perda.item() * len(xb)

        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(X_val_t)
            perda_val = criterio(logits_val, y_val_t).item()
            auc_val = roc_auc_score(y_val, torch.sigmoid(logits_val).cpu().numpy())

        historico["perda"].append(soma / len(carregador.dataset))
        historico["perda_val"].append(perda_val)
        historico["auc_val"].append(auc_val)
        if verbose and epoca % 5 == 0:
            print(f"época {epoca:3d}  perda {historico['perda'][-1]:.4f}  AUC val {auc_val:.4f}")

        if auc_val > melhor_auc + 1e-5:
            melhor_auc, sem_melhora = auc_val, 0
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    return historico
