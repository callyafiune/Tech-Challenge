"""Testes da bateria de refinamento de F1."""

import pandas as pd

from tech_challenge_churn.models.f1_refinement import (
    REFINEMENT_CATEGORICAL_FEATURES,
    REFINEMENT_NUMERIC_FEATURES,
    add_refinement_features,
    build_candidate_registry,
    build_feature_specs,
)


def _sample_features() -> pd.DataFrame:
    """Cria linhas mínimas no schema Telco sem a coluna alvo."""
    return pd.DataFrame(
        [
            {
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
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 48,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 59.0,
                "TotalCharges": "2832.00",
            },
        ]
    )


def test_add_refinement_features_creates_expected_columns_without_target() -> None:
    """Garante que as interações novas usam apenas features de entrada."""
    engineered = add_refinement_features(_sample_features())

    for column in (*REFINEMENT_NUMERIC_FEATURES, *REFINEMENT_CATEGORICAL_FEATURES):
        assert column in engineered.columns
    assert "Churn" not in engineered.columns


def test_candidate_registry_uses_allowed_families_and_feature_sets() -> None:
    """Valida que a busca fica dentro do escopo técnico definido."""
    allowed_families = {"random_forest", "extra_trees", "hgb", "stacking"}
    feature_sets = set(build_feature_specs())

    candidates = build_candidate_registry()

    assert candidates
    assert {candidate.family for candidate in candidates} <= allowed_families
    assert {candidate.feature_spec.name for candidate in candidates} <= feature_sets
