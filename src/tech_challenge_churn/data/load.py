"""Funções de leitura, limpeza mínima e separação do dataset Telco."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from tech_challenge_churn.config import DATA_PATH, EXPECTED_COLUMNS, ID_COLUMN, TARGET_COLUMN
from tech_challenge_churn.utils.logging import get_logger

logger = get_logger(__name__)


def compute_file_hash(path: Path = DATA_PATH) -> str:
    """Calcula o hash SHA256 do arquivo usado no experimento."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto e valida as colunas esperadas."""
    data = pd.read_csv(path)
    missing_columns = set(EXPECTED_COLUMNS) - set(data.columns)
    extra_columns = set(data.columns) - set(EXPECTED_COLUMNS)

    if missing_columns or extra_columns:
        raise ValueError(
            "Schema inesperado no CSV: "
            f"missing_columns={sorted(missing_columns)}, extra_columns={sorted(extra_columns)}"
        )

    logger.info(
        "dataset_carregado",
        extra={"rows": len(data), "columns": len(data.columns), "path": str(path)},
    )
    return data


def clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges para número preservando valores inválidos como NaN."""
    cleaned = data.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def encode_target(target: pd.Series) -> pd.Series:
    """Codifica Churn como 1 para cancelamento e 0 para permanência."""
    return target.map({"Yes": 1, "No": 0}).astype(int)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo, removendo identificadores sem valor preditivo."""
    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = encode_target(data[TARGET_COLUMN])
    return features, target
