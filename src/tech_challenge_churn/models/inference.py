"""Carregamento de artefatos para inferência em produção."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch

from tech_challenge_churn.config import MODELS_DIR
from tech_challenge_churn.models.mlp import TelcoMLP


def load_mlp_for_inference(
    model_dir: Path | None = None,
) -> tuple[object, TelcoMLP, dict[str, object]]:
    """Carrega preprocessor, modelo PyTorch e metadados sem dependências de treino."""
    model_dir = model_dir or MODELS_DIR / "mlp"
    metadata = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = TelcoMLP(
        input_dim=int(metadata["input_dim"]),
        hidden_layers=tuple(int(layer) for layer in metadata["config"]["hidden_layers"].split("-")),
        dropout=float(metadata["config"]["dropout"]),
    )
    model.load_state_dict(torch.load(model_dir / "model_state_dict.pt", weights_only=True))
    model.eval()
    return preprocessor, model, metadata
