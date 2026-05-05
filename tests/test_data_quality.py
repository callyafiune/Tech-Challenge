"""Testes da auditoria de qualidade dos dados."""

from tech_challenge_churn.config import DATA_PATH
from tech_challenge_churn.data.load import clean_total_charges, read_raw_data
from tech_challenge_churn.reports.data_quality import (
    _domain_values,
    _logical_anomalies,
    generate_data_quality_report,
)


def test_domain_audit_finds_no_invalid_values_in_reference_dataset() -> None:
    """Garante que a base de referencia respeita os dominios esperados."""
    raw_data = read_raw_data(DATA_PATH)
    _, invalid_values = _domain_values(raw_data)

    assert invalid_values.empty


def test_logical_anomaly_audit_keeps_total_charges_issue_explicit() -> None:
    """Confirma que os 11 blanks de TotalCharges continuam rastreados."""
    raw_data = read_raw_data(DATA_PATH)
    clean_data = clean_total_charges(raw_data)
    anomalies = _logical_anomalies(raw_data, clean_data)
    total_blank_count = anomalies.loc[
        anomalies["checagem"].eq("TotalCharges vazio no raw"),
        "ocorrencias",
    ].iloc[0]

    assert total_blank_count == 11


def test_generate_data_quality_report_creates_markdown_artifact() -> None:
    """Valida criacao do relatorio completo de qualidade de dados."""
    output_path = generate_data_quality_report()

    assert output_path.exists()
    assert "Revisao Rigorosa" in output_path.read_text(encoding="utf-8")
