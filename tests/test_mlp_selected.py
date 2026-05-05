"""Testes dos experimentos de MLP com selecao de features."""

import pytest
from sklearn.feature_selection import SelectKBest

from tech_challenge_churn.models.train_mlp_selected import (
    _selector,
    selected_mlp_experiments,
)


def test_selected_mlp_experiments_cover_expected_selectors() -> None:
    """Garante que a bateria cobre filtros estatisticos e informacao mutua."""
    experiments = selected_mlp_experiments()
    selectors = {experiment.selector for experiment in experiments}
    feature_counts = {experiment.k for experiment in experiments}

    assert {"f_classif", "mutual_info"}.issubset(selectors)
    assert {35, 50, 65}.issubset(feature_counts)


def test_selector_factory_returns_select_k_best() -> None:
    """Valida a fabrica de seletores usada dentro de cada fold."""
    selector = _selector("f_classif", k=35)

    assert isinstance(selector, SelectKBest)
    assert selector.k == 35


def test_selector_factory_rejects_unknown_selector() -> None:
    """Evita nomes de seletores silenciosamente invalidos."""
    with pytest.raises(ValueError, match="Selector desconhecido"):
        _selector("nao_existe", k=10)
