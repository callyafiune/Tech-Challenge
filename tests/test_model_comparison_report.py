"""Testes da comparação estatística de modelos."""

import numpy as np

from tech_challenge_churn.reports.model_comparison import (
    exact_sign_test_p_value,
    paired_comparison,
)


def test_exact_sign_test_p_value_handles_all_positive_differences() -> None:
    """Valida p-valor bicaudal do teste de sinais para 5 diferenças positivas."""
    differences = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    assert exact_sign_test_p_value(differences) == 0.0625


def test_paired_comparison_reports_non_significant_small_difference() -> None:
    """Garante leitura conservadora quando o ganho é pequeno."""
    result = paired_comparison(
        "modelo_a",
        [0.64, 0.63, 0.65, 0.62, 0.64],
        "modelo_b",
        [0.63, 0.64, 0.64, 0.62, 0.63],
        "f1",
    )

    assert result["metric"] == "f1"
    assert result["significant"] is False
    assert "parcimônia" in result["conclusion"]
