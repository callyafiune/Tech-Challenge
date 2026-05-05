"""Testes da bateria de tuning Scikit-Learn."""

from sklearn.pipeline import Pipeline

from tech_challenge_churn.models.sklearn_tuning import (
    build_candidate_registry,
)


def test_candidate_registry_contains_expected_families() -> None:
    """Garante cobertura dos caminhos de tuning definidos."""
    candidates = build_candidate_registry()
    families = {candidate.family for candidate in candidates}

    assert {"hgb", "random_forest", "extra_trees", "stacking", "svc"}.issubset(families)


def test_candidates_keep_selector_inside_model_pipeline() -> None:
    """Valida que a selecao de features fica dentro do pipeline avaliado."""
    candidate = build_candidate_registry()[0]

    assert isinstance(candidate.model, Pipeline)
    assert list(candidate.model.named_steps) == ["selector", "classifier"]
