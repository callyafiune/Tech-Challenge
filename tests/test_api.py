"""Testes da API FastAPI."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tech_challenge_churn.api.app import app
from tech_challenge_churn.config import MODELS_DIR


def _payload() -> dict[str, object]:
    """Payload válido representativo de cliente Telco."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": "958.20",
    }


def _model_available() -> bool:
    """Verifica se os artefatos da MLP estão disponíveis localmente."""
    return (MODELS_DIR / "mlp" / "model_config.json").exists()


def test_health_endpoint() -> None:
    """Valida o endpoint de saúde."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


@pytest.mark.skipif(not _model_available(), reason="Modelo MLP ainda nao treinado.")
def test_predict_smoke() -> None:
    """Valida inferência ponta a ponta com payload correto."""
    with TestClient(app) as client:
        response = client.post("/predict", json=_payload())

    body = response.json()
    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn_prediction"], bool)
    assert body["model_version"].startswith("mlp_v1_")


@pytest.mark.skipif(not _model_available(), reason="Modelo MLP ainda nao treinado.")
def test_predict_is_deterministic() -> None:
    """Garante resposta determinística para o mesmo cliente."""
    with TestClient(app) as client:
        first = client.post("/predict", json=_payload()).json()
        second = client.post("/predict", json=_payload()).json()

    assert first["churn_probability"] == second["churn_probability"]


@pytest.mark.skipif(not _model_available(), reason="Modelo MLP ainda nao treinado.")
def test_predict_threshold_override_changes_decision_rule() -> None:
    """Valida override explícito de threshold."""
    with TestClient(app) as client:
        response = client.post("/predict?threshold=0.99", json=_payload())

    body = response.json()
    assert response.status_code == 200
    assert body["threshold"] == 0.99
    assert body["churn_prediction"] is False


def test_predict_rejects_invalid_schema() -> None:
    """Confirma que valores fora do domínio retornam 422."""
    payload = _payload()
    payload["Contract"] = "Weekly"

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_extra_fields() -> None:
    """Confirma bloqueio de campos fora do contrato."""
    payload = _payload()
    payload["customerID"] = "1234"

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_model_artifact_path_is_project_local() -> None:
    """Protege o teste contra caminhos acidentais fora do projeto."""
    model_path = (MODELS_DIR / "mlp").resolve()
    assert Path("C:/estudos/Tech-Challenge").resolve() in model_path.parents
