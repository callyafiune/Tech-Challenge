"""Schema Pandera do dataset Telco."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import DataFrame

from tech_challenge_churn.config import CATEGORICAL_DOMAIN_VALUES, EXPECTED_COLUMNS

TELCO_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["gender"])),
        "SeniorCitizen": pa.Column(int, checks=pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["Partner"])),
        "Dependents": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["Dependents"]),
        ),
        "tenure": pa.Column(int, checks=[pa.Check.ge(0), pa.Check.le(72)]),
        "PhoneService": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["PhoneService"]),
        ),
        "MultipleLines": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["MultipleLines"]),
        ),
        "InternetService": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["InternetService"]),
        ),
        "OnlineSecurity": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["OnlineSecurity"]),
        ),
        "OnlineBackup": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["OnlineBackup"]),
        ),
        "DeviceProtection": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["DeviceProtection"]),
        ),
        "TechSupport": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["TechSupport"]),
        ),
        "StreamingTV": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["StreamingTV"]),
        ),
        "StreamingMovies": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["StreamingMovies"]),
        ),
        "Contract": pa.Column(str, checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["Contract"])),
        "PaperlessBilling": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["PaperlessBilling"]),
        ),
        "PaymentMethod": pa.Column(
            str,
            checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["PaymentMethod"]),
        ),
        "MonthlyCharges": pa.Column(float, checks=pa.Check.ge(0)),
        "TotalCharges": pa.Column(str, nullable=False),
        "Churn": pa.Column(str, checks=pa.Check.isin(CATEGORICAL_DOMAIN_VALUES["Churn"])),
    },
    strict=True,
    coerce=True,
)

CLEAN_TELCO_SCHEMA = pa.DataFrameSchema(
    {
        **{
            column: TELCO_SCHEMA.columns[column]
            for column in EXPECTED_COLUMNS
            if column != "TotalCharges"
        },
        "TotalCharges": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
    },
    strict=True,
    coerce=True,
)


def validate_telco_schema(data: DataFrame) -> DataFrame:
    """Valida schema, ordem de colunas e regras basicas de dominio."""
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("A ordem ou o conjunto de colunas do dataset foi alterado.")
    return TELCO_SCHEMA.validate(data)


def validate_clean_telco_schema(data: DataFrame) -> DataFrame:
    """Valida o dataset apos coercao numerica de TotalCharges."""
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("A ordem ou o conjunto de colunas do dataset foi alterado.")
    return CLEAN_TELCO_SCHEMA.validate(data)
