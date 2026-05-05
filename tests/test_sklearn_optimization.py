"""Testes dos experimentos Scikit-Learn permitidos."""

from sklearn.pipeline import Pipeline

from tech_challenge_churn.models.sklearn_optimization import (
    build_experiment_pipeline,
    build_model_registry,
)


def test_model_registry_contains_permitted_experiments() -> None:
    """Valida que a bateria contem modelos Scikit-Learn sem dependencias externas."""
    registry = build_model_registry()

    assert "logistic_elasticnet_expanded" in registry
    assert "hist_gradient_boosting_regularized" in registry
    assert "hist_gradient_boosting_calibrated_sigmoid" in registry


def test_experiment_pipeline_keeps_feature_engineering_inside_pipeline() -> None:
    """Garante que o pre-processamento fica dentro do pipeline avaliado em CV."""
    model = build_model_registry()["hist_gradient_boosting_regularized"]
    pipeline = build_experiment_pipeline(model)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["features", "model"]
