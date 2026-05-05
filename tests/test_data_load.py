"""Testes de carregamento e schema do dataset."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from tech_challenge_churn.config import DATA_PATH, EXPECTED_COLUMNS
from tech_challenge_churn.data.load import clean_total_charges, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_clean_telco_schema, validate_telco_schema


def test_dataset_schema_contract() -> None:
    """Valida que o CSV preserva o contrato esperado de colunas."""
    data = validate_telco_schema(read_raw_data(DATA_PATH))

    assert data.shape == (7043, 21)
    assert list(data.columns) == EXPECTED_COLUMNS


def test_total_charges_coercion_handles_blank_values() -> None:
    """Garante que TotalCharges em branco vira nulo numérico controlado."""
    data = read_raw_data(DATA_PATH)
    cleaned = clean_total_charges(data)

    assert cleaned["TotalCharges"].isna().sum() == 11
    assert cleaned.loc[cleaned["TotalCharges"].isna(), "tenure"].eq(0).all()
    validate_clean_telco_schema(cleaned)


def test_split_features_target_removes_identifier_and_encodes_target() -> None:
    """Confirma remoção de customerID e codificação binária do alvo."""
    data = read_raw_data(DATA_PATH)
    features, target = split_features_target(data)

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_schema_rejects_invalid_categorical_domain() -> None:
    """Valida que dominios categoricos fora do contrato sao barrados."""
    data = read_raw_data(DATA_PATH)
    data.loc[0, "Contract"] = "Invalid contract"

    with pytest.raises(SchemaError):
        validate_telco_schema(data)


def test_clean_schema_rejects_negative_total_charges() -> None:
    """Valida regra pos-limpeza para TotalCharges numerico nao negativo."""
    data = clean_total_charges(read_raw_data(DATA_PATH))
    data.loc[0, "TotalCharges"] = -1.0

    with pytest.raises(SchemaError):
        validate_clean_telco_schema(pd.DataFrame(data))
