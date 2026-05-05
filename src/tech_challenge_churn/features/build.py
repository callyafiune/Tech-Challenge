"""Pipeline de pré-processamento para modelos tabulares."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from tech_challenge_churn.config import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    INTERNET_DEPENDENT_COLUMNS,
    PROTECTION_COLUMNS,
    SERVICE_COLUMNS,
)


def normalize_service_absence_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Colapsa categorias redundantes de ausencia de servico para reduzir colinearidade."""
    normalized = data.copy()
    internet_columns = [
        column for column in INTERNET_DEPENDENT_COLUMNS if column in normalized.columns
    ]
    normalized[internet_columns] = normalized[internet_columns].replace(
        {"No internet service": "No"}
    )
    if "MultipleLines" in normalized.columns:
        normalized["MultipleLines"] = normalized["MultipleLines"].replace(
            {"No phone service": "No"}
        )
    return normalized


def add_telco_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria features de gasto, relacionamento e intensidade de serviços."""
    engineered = normalize_service_absence_categories(data)
    engineered["TotalCharges"] = pd.to_numeric(engineered["TotalCharges"], errors="coerce")
    zero_tenure_missing_total = engineered["tenure"].eq(0) & engineered["TotalCharges"].isna()
    engineered.loc[zero_tenure_missing_total, "TotalCharges"] = 0.0

    safe_tenure = engineered["tenure"].replace(0, np.nan)
    safe_monthly_charges = engineered["MonthlyCharges"].replace(0, np.nan)
    engineered["avg_monthly_spend"] = engineered["TotalCharges"] / safe_tenure
    engineered["avg_monthly_spend"] = engineered["avg_monthly_spend"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    engineered["charges_delta"] = engineered["MonthlyCharges"] - engineered["avg_monthly_spend"]
    engineered["total_to_monthly_ratio"] = engineered["TotalCharges"] / safe_monthly_charges
    engineered["total_to_monthly_ratio"] = engineered["total_to_monthly_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    engineered["tenure_bucket"] = pd.cut(
        engineered["tenure"],
        bins=[-0.1, 6, 12, 24, 48, np.inf],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype("object")

    service_flags = [engineered[column].eq("Yes").astype(int) for column in SERVICE_COLUMNS]
    protection_flags = [engineered[column].eq("Yes").astype(int) for column in PROTECTION_COLUMNS]
    engineered["num_services"] = np.sum(service_flags, axis=0)
    engineered["num_protection_services"] = np.sum(protection_flags, axis=0)
    engineered["has_protection_bundle"] = (
        engineered[PROTECTION_COLUMNS].eq("Yes").all(axis=1).astype(int)
    )
    engineered["is_zero_tenure"] = engineered["tenure"].eq(0).astype(int)
    engineered["has_internet_service"] = engineered["InternetService"].ne("No").astype(int)
    engineered["fiber_without_security"] = (
        engineered["InternetService"].eq("Fiber optic") & engineered["OnlineSecurity"].eq("No")
    ).astype(int)
    engineered["electronic_check_month_to_month"] = (
        engineered["PaymentMethod"].eq("Electronic check")
        & engineered["Contract"].eq("Month-to-month")
    ).astype(int)
    engineered["month_to_month_low_tenure"] = (
        engineered["Contract"].eq("Month-to-month") & engineered["tenure"].le(12)
    ).astype(int)
    engineered["senior_month_to_month"] = (
        engineered["SeniorCitizen"].eq(1) & engineered["Contract"].eq("Month-to-month")
    ).astype(int)
    engineered["streaming_bundle"] = (
        engineered["StreamingTV"].eq("Yes") & engineered["StreamingMovies"].eq("Yes")
    ).astype(int)
    engineered["contract_tenure_segment"] = (
        engineered["Contract"].astype(str) + "__" + engineered["tenure_bucket"].astype(str)
    )
    engineered["internet_security_profile"] = (
        engineered["InternetService"].astype(str)
        + "__security_"
        + engineered["OnlineSecurity"].astype(str)
        + "__support_"
        + engineered["TechSupport"].astype(str)
    )
    engineered["payment_contract_profile"] = (
        engineered["PaymentMethod"].astype(str) + "__" + engineered["Contract"].astype(str)
    )

    return engineered


def build_feature_pipeline() -> Pipeline:
    """Monta o pipeline sklearn usado por baselines e pela futura MLP."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, BASE_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, BASE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_telco_features, validate=False)),
            ("preprocessor", preprocessor),
        ]
    )
