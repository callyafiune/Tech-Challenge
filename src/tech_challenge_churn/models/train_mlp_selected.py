"""Experimentos da MLP com selecao de features apos o pre-processamento."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split

from tech_challenge_churn.config import DATA_PATH, MLFLOW_TRACKING_URI, RANDOM_SEED, REPORTS_DIR
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import lift_at_top_fraction
from tech_challenge_churn.evaluation.metrics import (
    find_best_f1_threshold,
    probability_metrics,
    threshold_metrics,
)
from tech_challenge_churn.features.build import build_feature_pipeline
from tech_challenge_churn.models.mlp import (
    MLPConfig,
    predict_proba,
    to_numpy_array,
    train_torch_model,
)
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-mlp-feature-selection"


@dataclass(frozen=True)
class SelectedMLPExperiment:
    """Configuracao de uma tentativa de MLP com entrada reduzida."""

    name: str
    selector: str
    k: int
    config: MLPConfig


def _current_git_sha() -> str:
    """Obtem o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def selected_mlp_experiments() -> list[SelectedMLPExperiment]:
    """Define uma busca curta para testar reducao de dimensionalidade na MLP."""
    compact = MLPConfig(
        hidden_layers=(64, 32),
        dropout=0.4,
        learning_rate=5e-4,
        weight_decay=1e-3,
    )
    current_best = MLPConfig(
        hidden_layers=(128, 64, 32),
        dropout=0.3,
        learning_rate=5e-4,
        weight_decay=1e-4,
    )
    return [
        SelectedMLPExperiment("mlp_select_f_classif_k35", "f_classif", 35, current_best),
        SelectedMLPExperiment("mlp_select_f_classif_k50", "f_classif", 50, current_best),
        SelectedMLPExperiment("mlp_select_f_classif_k65", "f_classif", 65, current_best),
        SelectedMLPExperiment("mlp_select_mutual_info_k50", "mutual_info", 50, current_best),
        SelectedMLPExperiment("mlp_select_mutual_info_k65", "mutual_info", 65, current_best),
        SelectedMLPExperiment("mlp_compact_select_f_classif_k50", "f_classif", 50, compact),
    ]


def _selector(selector_name: str, k: int, seed: int = RANDOM_SEED) -> SelectKBest:
    """Cria seletor de features para os arrays ja pre-processados."""
    if selector_name == "f_classif":
        return SelectKBest(score_func=f_classif, k=k)
    if selector_name == "mutual_info":
        score_func = partial(mutual_info_classif, random_state=seed)
        return SelectKBest(score_func=score_func, k=k)
    raise ValueError(f"Selector desconhecido: {selector_name}")


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula medias e desvios por metrica."""
    summary: dict[str, float] = {}
    for metric_name in sorted(fold_metrics[0]):
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def evaluate_selected_mlp(
    experiment: SelectedMLPExperiment,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Avalia uma MLP com selecao de features dentro de cada fold."""
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=experiment.config.seed,
    )
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_valid_index, test_index) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        x_train_valid = features.iloc[train_valid_index]
        y_train_valid = target.iloc[train_valid_index]
        x_test = features.iloc[test_index]
        y_test = target.iloc[test_index]

        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train_valid,
            y_train_valid,
            test_size=0.15,
            stratify=y_train_valid,
            random_state=experiment.config.seed + fold_index,
        )
        preprocessor = build_feature_pipeline()
        train_array = to_numpy_array(preprocessor.fit_transform(x_train))
        valid_array = to_numpy_array(preprocessor.transform(x_valid))
        test_array = to_numpy_array(preprocessor.transform(x_test))

        selector = _selector(
            experiment.selector,
            min(experiment.k, train_array.shape[1]),
            experiment.config.seed,
        )
        train_array = selector.fit_transform(train_array, y_train)
        valid_array = selector.transform(valid_array)
        test_array = selector.transform(test_array)

        model, history = train_torch_model(
            train_array,
            y_train.to_numpy(),
            valid_array,
            y_valid.to_numpy(),
            experiment.config,
        )
        valid_proba = predict_proba(
            model,
            valid_array,
            batch_size=experiment.config.batch_size,
        )
        threshold, valid_best_f1 = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)
        test_proba = predict_proba(
            model,
            test_array,
            batch_size=experiment.config.batch_size,
        )
        metrics = probability_metrics(y_test.to_numpy(), test_proba)
        metrics.update(threshold_metrics(y_test.to_numpy(), test_proba, threshold=0.5))
        optimal_metrics = threshold_metrics(y_test.to_numpy(), test_proba, threshold=threshold)
        metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
        metrics["threshold_f1"] = threshold
        metrics["valid_f1_optimal"] = valid_best_f1
        metrics["lift_at_top_20pct"] = lift_at_top_fraction(y_test.to_numpy(), test_proba)
        metrics["selected_features"] = float(train_array.shape[1])
        metrics["best_epoch"] = history["best_epoch"]
        metrics["best_valid_pr_auc"] = history["best_valid_pr_auc"]
        fold_metrics.append(metrics)

        logger.info(
            "fold_mlp_selecionada_avaliado",
            extra={
                "experiment": experiment.name,
                "fold": fold_index,
                "selector": experiment.selector,
                "k": experiment.k,
                "optimal_f1": metrics["optimal_f1"],
                "pr_auc": metrics["pr_auc"],
            },
        )

    return _summarize_fold_metrics(fold_metrics), fold_metrics


def _log_selected_mlp_run(
    experiment: SelectedMLPExperiment,
    summary: dict[str, float],
    fold_metrics: list[dict[str, float]],
) -> None:
    """Registra uma tentativa de MLP com selecao de features no MLflow."""
    with mlflow.start_run(run_name=experiment.name):
        mlflow.set_tags(
            {
                "stage": "mlp_feature_selection",
                "framework": "pytorch",
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": experiment.name,
                "selector": experiment.selector,
            }
        )
        mlflow.log_params(
            {
                "selector": experiment.selector,
                "k": experiment.k,
                **experiment.config.to_dict(),
            }
        )
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)
        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)
        mlflow.log_dict(summary, "summary_metrics.json")
        mlflow.log_dict(fold_metrics, "fold_metrics.json")


def _write_report(comparison: pd.DataFrame) -> None:
    """Documenta se selecao de features ajudou a MLP."""
    output_path = Path("docs") / "mlp_feature_selection_report.md"
    best = comparison.iloc[0]
    report = f"""# MLP com Selecao de Features

## Protocolo

- O pre-processador foi ajustado apenas no treino de cada fold.
- `SelectKBest` foi ajustado apenas no treino de cada fold, depois do OneHotEncoder.
- A MLP manteve early stopping por PR-AUC e `pos_weight` para desbalanceamento.
- Todos os experimentos foram registrados no MLflow em `{EXPERIMENT_NAME}`.

## Melhor Resultado

- Experimento: `{best["model"]}`.
- AUC-ROC media: `{best["roc_auc_mean"]:.4f}`.
- PR-AUC media: `{best["pr_auc_mean"]:.4f}`.
- F1 medio em threshold 0,5: `{best["f1_mean"]:.4f}`.
- F1 medio com threshold interno: `{best["optimal_f1_mean"]:.4f}`.

## Leitura Critica

A selecao de features foi testada para verificar se a MLP estava sofrendo com ruido nas 80 features
codificadas. O resultado deve ser comparado com a MLP refinada sem selecao, que tinha F1 otimizado
medio de `0.6242` em CV e F1 holdout de `0.6396`.

Artefato comparativo: `reports/mlp_feature_selection/comparison.csv`.
"""
    output_path.write_text(report, encoding="utf-8")


def run_selected_mlp_experiments() -> pd.DataFrame:
    """Executa e registra as MLPs com entrada reduzida."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    results: list[dict[str, float | str]] = []

    for experiment in selected_mlp_experiments():
        summary, fold_metrics = evaluate_selected_mlp(experiment, features, target)
        _log_selected_mlp_run(experiment, summary, fold_metrics)
        results.append(
            {
                "model": experiment.name,
                "selector": experiment.selector,
                "k": experiment.k,
                **summary,
            }
        )

    comparison = pd.DataFrame(results).sort_values(
        ["optimal_f1_mean", "pr_auc_mean", "roc_auc_mean"],
        ascending=False,
    )
    output_dir = REPORTS_DIR / "mlp_feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    _write_report(comparison)
    logger.info(
        "experimentos_mlp_selecao_concluidos",
        extra={
            "best_model": comparison.iloc[0]["model"],
            "best_optimal_f1_mean": comparison.iloc[0]["optimal_f1_mean"],
        },
    )
    return comparison


def main() -> None:
    """Ponto de entrada do comando train-mlp-selected."""
    run_selected_mlp_experiments()


if __name__ == "__main__":
    main()
