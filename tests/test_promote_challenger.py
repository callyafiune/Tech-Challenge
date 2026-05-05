"""Testes da promocao formal do challenger."""

from tech_challenge_churn.models.promote_challenger import (
    PROMOTED_MODEL_ROLE,
    PROMOTED_VERSION,
    _promotion_spec,
)


def test_promotion_spec_uses_no_gender_feature_set() -> None:
    """Garante que a promocao formal usa a ablação recomendada."""
    spec = _promotion_spec()

    assert spec.name == "no_gender"
    assert "gender" in spec.drop_categorical


def test_promoted_version_and_role_are_explicit() -> None:
    """Evita promover modelos sem papel operacional definido."""
    assert PROMOTED_VERSION == "random_forest_no_gender_v1"
    assert PROMOTED_MODEL_ROLE == "operational_challenger"
