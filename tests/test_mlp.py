"""Testes unitários da MLP PyTorch."""

import numpy as np
import torch

from tech_challenge_churn.models.mlp import MLPConfig, TelcoMLP, make_tensor_dataset, predict_proba


def test_mlp_forward_output_shape() -> None:
    """Confirma que a MLP retorna um logit por cliente."""
    model = TelcoMLP(input_dim=10, hidden_layers=(8, 4), dropout=0.1)
    features = torch.randn(5, 10)

    logits = model(features)

    assert logits.shape == (5,)


def test_predict_proba_range() -> None:
    """Garante que a inferência produz probabilidades válidas."""
    model = TelcoMLP(input_dim=3, hidden_layers=(4,), dropout=0.1)
    features = np.ones((2, 3), dtype=np.float32)

    probabilities = predict_proba(model, features, batch_size=2)

    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_mlp_config_serializes_hidden_layers() -> None:
    """Valida serialização da arquitetura para MLflow e JSON."""
    config = MLPConfig(hidden_layers=(64, 32), dropout=0.2)

    assert config.to_dict()["hidden_layers"] == "64-32"


def test_tensor_dataset_uses_float32() -> None:
    """Confirma os tipos esperados pelo BCEWithLogitsLoss."""
    dataset = make_tensor_dataset(
        np.ones((3, 2), dtype=np.float32),
        np.array([0, 1, 0]),
    )
    features, target = dataset.tensors

    assert features.dtype == torch.float32
    assert target.dtype == torch.float32
