"""Testes das métricas de negócio."""

import numpy as np

from tech_challenge_churn.evaluation.business import (
    BusinessAssumptions,
    compute_business_metrics,
    find_best_business_threshold,
    lift_at_top_fraction,
)


def test_business_metrics_returns_expected_counts() -> None:
    """Confirma contagens de TP, FP e FN em threshold fixo."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.1])
    monthly_charges = np.array([100.0, 50.0, 80.0, 40.0])

    metrics = compute_business_metrics(
        y_true,
        y_proba,
        monthly_charges,
        threshold=0.5,
        assumptions=BusinessAssumptions(retained_months=12, offer_cost_multiplier=1),
    )

    assert metrics["business_tp"] == 1
    assert metrics["business_fp"] == 1
    assert metrics["business_fn"] == 1
    assert metrics["business_incremental_savings"] == 1050.0


def test_business_threshold_and_lift_are_valid() -> None:
    """Valida busca de threshold de negócio e lift no topo do ranking."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.1])
    monthly_charges = np.array([100.0, 50.0, 80.0, 40.0])

    threshold, metrics = find_best_business_threshold(y_true, y_proba, monthly_charges)
    lift = lift_at_top_fraction(y_true, y_proba, fraction=0.5)

    assert 0.05 <= threshold <= 0.95
    assert metrics["business_incremental_savings"] > 0
    assert lift >= 1.0
