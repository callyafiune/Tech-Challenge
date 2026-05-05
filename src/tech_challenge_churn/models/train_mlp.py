"""Treinamento e comparação da MLP PyTorch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from tech_challenge_churn.config import DATA_PATH, MLRUNS_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import (
    BusinessAssumptions,
    compute_business_metrics,
    find_best_business_threshold,
    lift_at_top_fraction,
)
from tech_challenge_churn.evaluation.metrics import (
    find_best_f1_threshold,
    probability_metrics,
    threshold_metrics,
)
from tech_challenge_churn.features.build import build_feature_pipeline
from tech_challenge_churn.models.mlp import (
    MLPConfig,
    TelcoMLP,
    predict_proba,
    to_numpy_array,
    train_torch_model,
)
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-mlp"


def candidate_configs() -> list[MLPConfig]:
    """Define uma busca local enxuta de hiperparâmetros."""
    return [
        MLPConfig(hidden_layers=(64, 32), dropout=0.2, learning_rate=1e-3, weight_decay=1e-4),
        MLPConfig(hidden_layers=(128, 64), dropout=0.3, learning_rate=1e-3, weight_decay=1e-4),
        MLPConfig(hidden_layers=(64, 32), dropout=0.4, learning_rate=5e-4, weight_decay=1e-3),
        MLPConfig(hidden_layers=(128, 64, 32), dropout=0.3, learning_rate=5e-4, weight_decay=1e-4),
    ]


def _current_git_sha() -> str:
    """Obtém o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula média e desvio por métrica."""
    summary: dict[str, float] = {}
    for metric_name in sorted(fold_metrics[0]):
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def _transform_fold_data(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta o pré-processador somente no treino e transforma validação/teste."""
    preprocessor = build_feature_pipeline()
    train_array = to_numpy_array(preprocessor.fit_transform(x_train))
    valid_array = to_numpy_array(preprocessor.transform(x_valid))
    test_array = to_numpy_array(preprocessor.transform(x_test))
    return train_array, valid_array, test_array


def evaluate_config(
    config: MLPConfig,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Avalia uma configuração com CV estratificada e early stopping interno."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    fold_metrics: list[dict[str, float]] = []
    assumptions = BusinessAssumptions()

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
            random_state=config.seed + fold_index,
        )
        train_array, valid_array, test_array = _transform_fold_data(x_train, x_valid, x_test)
        model, history = train_torch_model(
            train_array,
            y_train.to_numpy(),
            valid_array,
            y_valid.to_numpy(),
            config,
        )
        valid_proba = predict_proba(model, valid_array, batch_size=config.batch_size)
        threshold, f1_optimal = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)
        test_proba = predict_proba(model, test_array, batch_size=config.batch_size)

        metrics = probability_metrics(y_test.to_numpy(), test_proba)
        metrics.update(threshold_metrics(y_test.to_numpy(), test_proba, threshold=0.5))
        optimal_metrics = threshold_metrics(y_test.to_numpy(), test_proba, threshold=threshold)
        metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
        business_threshold, _ = find_best_business_threshold(
            y_valid.to_numpy(),
            valid_proba,
            x_valid["MonthlyCharges"],
            assumptions,
        )
        business_metrics = compute_business_metrics(
            y_test.to_numpy(),
            test_proba,
            x_test["MonthlyCharges"],
            business_threshold,
            assumptions,
        )
        metrics.update({f"business_{key}": value for key, value in business_metrics.items()})
        metrics["threshold_f1"] = threshold
        metrics["threshold_business"] = business_threshold
        metrics["f1_optimal"] = f1_optimal
        metrics["lift_at_top_20pct"] = lift_at_top_fraction(y_test.to_numpy(), test_proba)
        metrics["best_epoch"] = history["best_epoch"]
        metrics["best_valid_pr_auc"] = history["best_valid_pr_auc"]
        metrics["pos_weight"] = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        fold_metrics.append(metrics)

        logger.info(
            "fold_mlp_avaliado",
            extra={
                "fold": fold_index,
                "hidden_layers": config.to_dict()["hidden_layers"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1_optimal": metrics["optimal_f1"],
            },
        )

    return _summarize_fold_metrics(fold_metrics), fold_metrics


def _log_config_run(
    config_name: str,
    config: MLPConfig,
    summary: dict[str, float],
    fold_metrics: list[dict[str, float]],
) -> None:
    """Registra uma configuração da MLP no MLflow."""
    with mlflow.start_run(run_name=config_name):
        mlflow.set_tags(
            {
                "stage": "mlp_cross_validation",
                "framework": "pytorch",
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": config_name,
            }
        )
        mlflow.log_params(config.to_dict())
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)
        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)
        mlflow.log_dict(fold_metrics, "fold_metrics.json")
        mlflow.log_dict(summary, "summary_metrics.json")


def _train_final_model(
    config: MLPConfig,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Treina o modelo final e salva preprocessor, pesos e metadados."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=config.seed,
    )
    preprocessor = build_feature_pipeline()
    train_array = to_numpy_array(preprocessor.fit_transform(x_train))
    valid_array = to_numpy_array(preprocessor.transform(x_valid))
    model, history = train_torch_model(
        train_array,
        y_train.to_numpy(),
        valid_array,
        y_valid.to_numpy(),
        config,
    )
    valid_proba = predict_proba(model, valid_array, batch_size=config.batch_size)
    threshold, f1_optimal = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)
    business_threshold, business_metrics = find_best_business_threshold(
        y_valid.to_numpy(),
        valid_proba,
        x_valid["MonthlyCharges"],
        BusinessAssumptions(),
    )
    final_metrics = probability_metrics(y_valid.to_numpy(), valid_proba)
    final_metrics.update(threshold_metrics(y_valid.to_numpy(), valid_proba, threshold=0.5))
    optimal_metrics = threshold_metrics(y_valid.to_numpy(), valid_proba, threshold=threshold)
    final_metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
    final_metrics.update({f"business_{key}": value for key, value in business_metrics.items()})
    final_metrics["threshold_f1"] = threshold
    final_metrics["threshold_business"] = business_threshold
    final_metrics["f1_optimal"] = f1_optimal
    final_metrics["best_epoch"] = history["best_epoch"]
    final_metrics["best_valid_pr_auc"] = history["best_valid_pr_auc"]
    final_metrics["input_dim"] = float(train_array.shape[1])

    model_dir = MODELS_DIR / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model_state_dict.pt")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    metadata = {
        "config": config.to_dict(),
        "metrics": final_metrics,
        "input_dim": int(train_array.shape[1]),
        "threshold_f1": threshold,
        "threshold_business": business_threshold,
    }
    (model_dir / "model_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="mlp_final_model"):
        mlflow.set_tags(
            {
                "stage": "mlp_final",
                "framework": "pytorch",
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": "mlp_final_model",
            }
        )
        mlflow.log_params(config.to_dict())
        for metric_name, value in final_metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

    logger.info(
        "mlp_final_salva",
        extra={
            "path": str(model_dir),
            "roc_auc": final_metrics["roc_auc"],
            "f1_optimal": final_metrics["optimal_f1"],
        },
    )
    return final_metrics


def _write_deep_learning_report(
    comparison: pd.DataFrame,
    best_config: MLPConfig,
    final_metrics: dict[str, float],
) -> None:
    """Documenta os resultados de deep learning."""
    output_path = Path("docs") / "deep_learning_report.md"
    logistic_auc = 0.8471257696936277
    logistic_f1 = 0.62785162188775
    best_row = comparison.iloc[0]
    report = f"""# Relatório de Deep Learning e Otimização

## Estratégia Técnica

A estratégia escolhida foi usar uma MLP compacta e regularizada, com
`BCEWithLogitsLoss(pos_weight)`, AdamW, early stopping por PR-AUC e comparação contra os baselines
Scikit-Learn.

## Arquitetura Escolhida

- Camadas ocultas: `{best_config.to_dict()["hidden_layers"]}`.
- Ativação: ReLU.
- Regularização: BatchNorm1d, Dropout `{best_config.dropout}` e AdamW com weight decay
  `{best_config.weight_decay}`.
- Tratamento de desbalanceamento: `pos_weight = n_negativos / n_positivos`.
- Early stopping: paciência de `{best_config.patience}` épocas monitorando PR-AUC de validação.

## Comparação com Baseline

Baseline de referência:

- Regressão Logística balanceada: AUC-ROC média `{logistic_auc:.4f}`, F1 média `{logistic_f1:.4f}`.

Melhor MLP em validação cruzada:

- AUC-ROC média `{best_row["roc_auc_mean"]:.4f}`.
- PR-AUC média `{best_row["pr_auc_mean"]:.4f}`.
- F1 em threshold 0,5 média `{best_row["f1_mean"]:.4f}`.
- F1 com threshold otimizado média `{best_row["optimal_f1_mean"]:.4f}`.

Modelo final salvo em `models/mlp/`:

- AUC-ROC holdout `{final_metrics["roc_auc"]:.4f}`.
- PR-AUC holdout `{final_metrics["pr_auc"]:.4f}`.
- F1 otimizado holdout `{final_metrics["optimal_f1"]:.4f}`.
- Threshold F1 `{final_metrics["threshold_f1"]:.2f}`.

## Interpretação

Como é comum em datasets tabulares pequenos, a MLP compete com a Regressão Logística, mas a
decisão final deve considerar desempenho, simplicidade, estabilidade e interpretabilidade. A
comparação completa foi salva em `reports/mlp/comparison.csv` e os experimentos foram registrados
no MLflow em `telco-churn-mlp`.
"""
    output_path.write_text(report, encoding="utf-8")


def run_mlp_experiments() -> pd.DataFrame:
    """Executa busca enxuta de MLPs, registra MLflow e salva o melhor modelo."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)

    results: list[dict[str, float | str]] = []
    configs = candidate_configs()

    for index, config in enumerate(configs, start=1):
        config_name = f"mlp_config_{index}_{config.to_dict()['hidden_layers']}"
        summary, fold_metrics = evaluate_config(config, features, target)
        _log_config_run(config_name, config, summary, fold_metrics)
        results.append({"model": config_name, **summary})

    comparison = pd.DataFrame(results).sort_values(
        ["optimal_f1_mean", "pr_auc_mean", "roc_auc_mean"],
        ascending=False,
    )
    output_dir = REPORTS_DIR / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    best_index = int(best_model_name.split("_")[2]) - 1
    best_config = configs[best_index]
    final_metrics = _train_final_model(best_config, features, target)
    _write_deep_learning_report(comparison, best_config, final_metrics)

    logger.info(
        "experimentos_mlp_concluidos",
        extra={
            "best_model": best_model_name,
            "best_optimal_f1_mean": comparison.iloc[0]["optimal_f1_mean"],
            "best_roc_auc_mean": comparison.iloc[0]["roc_auc_mean"],
        },
    )
    return comparison


def load_mlp_for_inference(
    model_dir: Path | None = None,
) -> tuple[object, TelcoMLP, dict[str, object]]:
    """Carrega preprocessor, modelo PyTorch e metadados para inferência."""
    model_dir = model_dir or MODELS_DIR / "mlp"
    metadata = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = TelcoMLP(
        input_dim=int(metadata["input_dim"]),
        hidden_layers=tuple(int(layer) for layer in metadata["config"]["hidden_layers"].split("-")),
        dropout=float(metadata["config"]["dropout"]),
    )
    model.load_state_dict(torch.load(model_dir / "model_state_dict.pt", weights_only=True))
    model.eval()
    return preprocessor, model, metadata


def main() -> None:
    """Ponto de entrada do comando train-mlp."""
    run_mlp_experiments()


if __name__ == "__main__":
    main()
