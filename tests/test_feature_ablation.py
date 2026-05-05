"""Testes da bateria de ablação de features."""

from tech_challenge_churn.models.feature_ablation import (
    _feature_lists,
    build_ablation_model,
    build_ablation_registry,
)


def test_ablation_registry_contains_baseline_and_summaries() -> None:
    """Garante que a bateria cobre referência e sumarizações relevantes."""
    specs = build_ablation_registry()
    names = {spec.name for spec in specs}

    assert "full_current" in names
    assert "no_gender" in names
    assert "relationship_summarized" in names
    assert "service_counts_only" in names


def test_relationship_summary_replaces_partner_and_dependents() -> None:
    """Valida que Partner e Dependents são removidos quando resumidos."""
    spec = next(
        spec for spec in build_ablation_registry() if spec.name == "relationship_summarized"
    )
    numeric_features, categorical_features = _feature_lists(spec)

    assert "Partner" not in categorical_features
    assert "Dependents" not in categorical_features
    assert "has_family_context" in numeric_features


def test_ablation_model_keeps_expected_pipeline_steps() -> None:
    """Confirma que o modelo final mantém pré-processamento e classificador."""
    spec = build_ablation_registry()[0]
    model = build_ablation_model(spec)

    assert list(model.named_steps) == ["features", "model"]
