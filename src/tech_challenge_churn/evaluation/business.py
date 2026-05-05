"""Métricas de negócio para campanhas de retenção."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessAssumptions:
    """Premissas explícitas para estimar valor de retenção."""

    retained_months: int = 12
    offer_cost_multiplier: float = 1.0


def compute_business_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    monthly_charges: pd.Series | np.ndarray,
    threshold: float,
    assumptions: BusinessAssumptions | None = None,
) -> dict[str, float]:
    """Calcula valor líquido da ação de retenção em um threshold."""
    assumptions = assumptions or BusinessAssumptions()
    charges = np.asarray(monthly_charges, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    selected = y_proba >= threshold

    true_positive = selected & (y_true == 1)
    false_positive = selected & (y_true == 0)
    false_negative = (~selected) & (y_true == 1)

    expected_customer_value = charges * assumptions.retained_months
    offer_cost = charges * assumptions.offer_cost_multiplier

    saved_value = float(expected_customer_value[true_positive].sum())
    campaign_cost = float(offer_cost[selected].sum())
    missed_value = float(expected_customer_value[false_negative].sum())
    incremental_savings = saved_value - campaign_cost
    operational_net_value = incremental_savings - missed_value

    return {
        "business_threshold": float(threshold),
        "business_tp": float(true_positive.sum()),
        "business_fp": float(false_positive.sum()),
        "business_fn": float(false_negative.sum()),
        "business_campaign_cost": campaign_cost,
        "business_saved_value": saved_value,
        "business_incremental_savings": float(incremental_savings),
        "business_operational_net_value": float(operational_net_value),
    }


def find_best_business_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    monthly_charges: pd.Series | np.ndarray,
    assumptions: BusinessAssumptions | None = None,
) -> tuple[float, dict[str, float]]:
    """Encontra o threshold com maior economia incremental estimada."""
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_savings = float("-inf")

    for threshold in np.linspace(0.05, 0.95, 91):
        current_metrics = compute_business_metrics(
            y_true,
            y_proba,
            monthly_charges,
            threshold,
            assumptions,
        )
        current_savings = current_metrics["business_incremental_savings"]
        if current_savings > best_savings:
            best_threshold = float(threshold)
            best_metrics = current_metrics
            best_savings = current_savings

    if best_metrics is None:
        raise RuntimeError("Nao foi possivel calcular o melhor threshold de negocio.")
    return best_threshold, best_metrics


def lift_at_top_fraction(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraction: float = 0.2,
) -> float:
    """Calcula lift de churn nos clientes com maior score."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    selected_count = max(1, int(len(y_true) * fraction))
    ranked_index = np.argsort(y_proba)[::-1][:selected_count]
    base_rate = y_true.mean()
    if base_rate == 0:
        return 0.0
    return float(y_true[ranked_index].mean() / base_rate)


def precision_at_top_fraction(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraction: float = 0.2,
) -> float:
    """Calcula a precisao nos clientes com maior score de churn."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    selected_count = max(1, int(len(y_true) * fraction))
    ranked_index = np.argsort(y_proba)[::-1][:selected_count]
    return float(y_true[ranked_index].mean())


def recall_at_top_fraction(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraction: float = 0.2,
) -> float:
    """Calcula quanto dos churns reais aparece entre os maiores scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    selected_count = max(1, int(len(y_true) * fraction))
    ranked_index = np.argsort(y_proba)[::-1][:selected_count]
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    return float(y_true[ranked_index].sum() / positives)


def assumptions_to_dict(assumptions: BusinessAssumptions) -> dict[str, int | float]:
    """Serializa premissas para registro no MLflow e documentação."""
    return asdict(assumptions)
