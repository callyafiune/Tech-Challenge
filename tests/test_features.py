"""Testes do pipeline de features."""

import pandas as pd

from tech_challenge_churn.features.build import (
    add_telco_features,
    build_feature_pipeline,
    normalize_service_absence_categories,
)


def _sample_features() -> pd.DataFrame:
    """Cria amostra mínima com o schema de features do Telco."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 0],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 70.00],
            "TotalCharges": ["29.85", " "],
        }
    )


def test_add_telco_features_creates_business_features() -> None:
    """Valida criação das features recomendadas para churn Telco."""
    engineered = add_telco_features(_sample_features())

    assert "avg_monthly_spend" in engineered.columns
    assert "charges_delta" in engineered.columns
    assert "tenure_bucket" in engineered.columns
    assert "num_services" in engineered.columns
    assert "has_protection_bundle" in engineered.columns
    assert "contract_tenure_segment" in engineered.columns
    assert "internet_security_profile" in engineered.columns
    assert "payment_contract_profile" in engineered.columns
    assert "total_to_monthly_ratio" in engineered.columns
    assert "fiber_without_security" in engineered.columns
    assert "electronic_check_month_to_month" in engineered.columns
    assert "month_to_month_low_tenure" in engineered.columns
    assert engineered.loc[0, "num_services"] == 1
    assert engineered.loc[1, "fiber_without_security"] == 1
    assert engineered.loc[1, "TotalCharges"] == 0.0


def test_service_absence_categories_are_collapsed_before_modeling() -> None:
    """Garante reducao de colinearidade deterministica antes do encoder."""
    normalized = normalize_service_absence_categories(_sample_features())

    assert normalized.loc[0, "MultipleLines"] == "No"
    assert normalized.loc[1, "OnlineSecurity"] == "No"


def test_feature_pipeline_transforms_without_leakage_steps_outside_pipeline() -> None:
    """Garante que o pipeline sklearn transforma a amostra ponta a ponta."""
    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(_sample_features())

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 5
