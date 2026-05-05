"""Testes da estratégia de validação cruzada."""

from sklearn.model_selection import StratifiedKFold

from tech_challenge_churn.config import DATA_PATH, RANDOM_SEED
from tech_challenge_churn.data.load import read_raw_data, split_features_target


def test_stratified_kfold_preserves_churn_rate() -> None:
    """Valida que cada fold mantém proporção próxima do alvo positivo."""
    data = read_raw_data(DATA_PATH)
    features, target = split_features_target(data)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    global_rate = target.mean()

    for _, valid_index in splitter.split(features, target):
        fold_rate = target.iloc[valid_index].mean()
        assert abs(fold_rate - global_rate) < 0.01
