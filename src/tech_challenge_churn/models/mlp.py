"""Componentes PyTorch para a MLP de previsão de churn."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tech_challenge_churn.config import RANDOM_SEED


@dataclass(frozen=True)
class MLPConfig:
    """Hiperparâmetros de treinamento da MLP."""

    hidden_layers: tuple[int, ...] = (64, 32)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 10
    seed: int = RANDOM_SEED

    def to_dict(self) -> dict[str, int | float | str]:
        """Serializa a configuração para logs e artefatos."""
        values = asdict(self)
        values["hidden_layers"] = "-".join(str(layer) for layer in self.hidden_layers)
        return values


class TelcoMLP(nn.Module):
    """Rede neural MLP compacta para dados tabulares pré-processados."""

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Aplica inicialização Kaiming nas camadas lineares."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Retorna logits para classificação binária."""
        return self.network(features).squeeze(1)


def to_numpy_array(features: object) -> np.ndarray:
    """Converte matriz densa ou esparsa do sklearn para `float32`."""
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def make_tensor_dataset(features: np.ndarray, target: np.ndarray) -> TensorDataset:
    """Cria dataset PyTorch com tipos estáveis."""
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(target, dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def make_data_loader(
    features: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    """Cria DataLoader determinístico para treinamento ou avaliação."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = make_tensor_dataset(features, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_pos_weight(target: np.ndarray) -> torch.Tensor:
    """Calcula peso da classe positiva para BCEWithLogitsLoss."""
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def predict_proba(model: TelcoMLP, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Gera probabilidades de churn a partir dos logits da MLP."""
    model.eval()
    probabilities: list[np.ndarray] = []
    dummy_target = np.zeros(features.shape[0], dtype=np.float32)
    loader = make_data_loader(
        features,
        dummy_target,
        batch_size=batch_size,
        seed=RANDOM_SEED,
        shuffle=False,
    )

    with torch.no_grad():
        for batch_features, _ in loader:
            logits = model(batch_features)
            batch_proba = torch.sigmoid(logits).cpu().numpy()
            probabilities.append(batch_proba)

    return np.concatenate(probabilities)


def train_torch_model(
    train_features: np.ndarray,
    train_target: np.ndarray,
    valid_features: np.ndarray,
    valid_target: np.ndarray,
    config: MLPConfig,
) -> tuple[TelcoMLP, dict[str, float]]:
    """Treina uma MLP com early stopping monitorando PR-AUC."""
    torch.manual_seed(config.seed)
    input_dim = train_features.shape[1]
    model = TelcoMLP(
        input_dim=input_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_target))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )
    train_loader = make_data_loader(
        train_features,
        train_target,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )

    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()

        valid_proba = predict_proba(model, valid_features, batch_size=config.batch_size)
        valid_pr_auc = float(average_precision_score(valid_target, valid_proba))
        scheduler.step(valid_pr_auc)

        if valid_pr_auc > best_score:
            best_score = valid_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "best_epoch": float(best_epoch),
        "best_valid_pr_auc": float(best_score),
        "epochs_trained": float(best_epoch + epochs_without_improvement),
    }
    return model, history
