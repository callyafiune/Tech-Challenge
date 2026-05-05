"""Serviço de inferência que encapsula preprocessor e MLP."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from tech_challenge_churn.config import MODELS_DIR
from tech_challenge_churn.models.mlp import predict_proba, to_numpy_array
from tech_challenge_churn.models.train_mlp import load_mlp_for_inference


class InferenceService:
    """Carrega artefatos e executa predições sem reler o modelo por request."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or MODELS_DIR / "mlp"
        self.preprocessor, self.model, self.metadata = load_mlp_for_inference(self.model_dir)
        self.threshold_business = float(self.metadata.get("threshold_business", 0.5))
        self.input_dim = int(self.metadata["input_dim"])
        self.model_version = f"mlp_v1_{self._model_hash()[:8]}"

    def _model_hash(self) -> str:
        """Calcula hash do state_dict para versionamento simples."""
        model_path = self.model_dir / "model_state_dict.pt"
        digest = hashlib.sha256()
        with model_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def predict_one(
        self,
        payload: dict[str, object],
        threshold: float | None = None,
    ) -> dict[str, object]:
        """Prediz churn para um cliente."""
        selected_threshold = self.threshold_business if threshold is None else threshold
        frame = pd.DataFrame([payload])
        transformed = to_numpy_array(self.preprocessor.transform(frame))
        probability = float(predict_proba(self.model, transformed)[0])

        return {
            "churn_probability": probability,
            "churn_prediction": probability >= selected_threshold,
            "threshold": selected_threshold,
            "model_version": self.model_version,
        }
