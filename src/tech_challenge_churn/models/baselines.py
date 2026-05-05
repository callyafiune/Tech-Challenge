"""Treinamento dos baselines com MLflow."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from tech_challenge_churn.config import (
    DATA_PATH,
    MLRUNS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import (
    BusinessAssumptions,
    assumptions_to_dict,
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
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-baselines"


def build_dummy_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """Cria baseline ingênuo com a mesma interface dos demais modelos."""
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("model", DummyClassifier(strategy="stratified", random_state=seed)),
        ]
    )


def build_logistic_regression_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """Cria baseline linear com balanceamento de classes."""
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_baseline_registry(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Agrupa os modelos avaliados como baselines."""
    return {
        "dummy_stratified": build_dummy_pipeline(seed),
        "logistic_regression_balanced": build_logistic_regression_pipeline(seed),
    }


def _current_git_sha() -> str:
    """Obtém o commit atual para rastreabilidade dos experimentos."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _save_evaluation_artifacts(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Path:
    """Salva curvas, matriz de confusão e relatório textual."""
    artifact_dir = REPORTS_DIR / "baselines" / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(f"Curva ROC - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "roc_curve.png", dpi=140)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(f"Curva Precision-Recall - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "pr_curve.png", dpi=140)
    plt.close()

    y_pred = (y_proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"]).plot(values_format="d")
    plt.title(f"Matriz de confusão - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=["No", "Yes"], zero_division=0)
    (artifact_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    return artifact_dir


def _log_sklearn_model(model: Pipeline) -> None:
    """Registra o pipeline final no MLflow com compatibilidade entre versões."""
    try:
        mlflow.sklearn.log_model(sk_model=model, name="model")
    except TypeError:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula média e desvio das métricas por fold."""
    summary: dict[str, float] = {}
    metric_names = sorted(fold_metrics[0])
    for metric_name in metric_names:
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def evaluate_and_log_model(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
    n_splits: int = 5,
) -> dict[str, float]:
    """Executa validação cruzada estratificada e registra o run no MLflow."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []
    all_true: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    all_monthly_charges: list[pd.Series] = []

    for train_index, valid_index in splitter.split(features, target):
        fold_model = clone(pipeline)
        x_train = features.iloc[train_index]
        x_valid = features.iloc[valid_index]
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]

        fold_model.fit(x_train, y_train)
        y_proba = fold_model.predict_proba(x_valid)[:, 1]
        metrics = probability_metrics(y_valid.to_numpy(), y_proba)
        metrics.update(threshold_metrics(y_valid.to_numpy(), y_proba, threshold=0.5))
        metrics["valid_churn_rate"] = float(y_valid.mean())
        fold_metrics.append(metrics)
        all_true.append(y_valid.to_numpy())
        all_proba.append(y_proba)
        all_monthly_charges.append(x_valid["MonthlyCharges"])

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    monthly_charges = pd.concat(all_monthly_charges, ignore_index=True)
    summary_metrics = _summarize_fold_metrics(fold_metrics)

    best_f1_threshold, best_f1 = find_best_f1_threshold(y_true, y_proba)
    assumptions = BusinessAssumptions()
    best_business_threshold, business_metrics = find_best_business_threshold(
        y_true,
        y_proba,
        monthly_charges,
        assumptions,
    )
    threshold_05_business = compute_business_metrics(
        y_true,
        y_proba,
        monthly_charges,
        0.5,
        assumptions,
    )
    lift_top_20 = lift_at_top_fraction(y_true, y_proba, fraction=0.2)

    final_model = clone(pipeline)
    final_model.fit(features, target)

    model_dir = MODELS_DIR / "baselines"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(final_model, model_path)

    artifact_dir = _save_evaluation_artifacts(model_name, y_true, y_proba, best_business_threshold)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": model_name,
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "n_splits": n_splits,
                "rows": len(features),
                "positive_rate": float(target.mean()),
                **assumptions_to_dict(assumptions),
            }
        )
        model_params = pipeline.named_steps["model"].get_params()
        mlflow.log_params({f"model__{key}": value for key, value in model_params.items()})

        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)

        for metric_name, value in summary_metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_metric("f1_optimal", best_f1)
        mlflow.log_metric("threshold_f1_optimal", best_f1_threshold)
        mlflow.log_metric("threshold_business_optimal", best_business_threshold)
        mlflow.log_metric("lift_at_top_20pct", lift_top_20)

        for metric_name, value in business_metrics.items():
            mlflow.log_metric(f"optimal_{metric_name}", value)
        for metric_name, value in threshold_05_business.items():
            mlflow.log_metric(f"threshold_05_{metric_name}", value)

        mlflow.log_dict(fold_metrics, "fold_metrics.json")
        mlflow.log_dict(summary_metrics, "summary_metrics.json")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        mlflow.log_artifact(str(model_path), artifact_path="serialized_model")
        _log_sklearn_model(final_model)

    metrics_path = REPORTS_DIR / "baselines" / f"{model_name}_summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "summary_metrics": summary_metrics,
                "best_f1_threshold": best_f1_threshold,
                "best_business_threshold": best_business_threshold,
                "business_metrics": business_metrics,
                "lift_at_top_20pct": lift_top_20,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "baseline_registrado",
        extra={
            "model_name": model_name,
            "roc_auc_mean": summary_metrics["roc_auc_mean"],
            "f1_mean": summary_metrics["f1_mean"],
            "model_path": str(model_path),
        },
    )
    return summary_metrics


def run_all_baselines() -> dict[str, dict[str, float]]:
    """Carrega os dados, executa baselines e retorna o resumo comparativo."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    results: dict[str, dict[str, float]] = {}

    for model_name, pipeline in build_baseline_registry(RANDOM_SEED).items():
        results[model_name] = evaluate_and_log_model(model_name, pipeline, features, target)

    comparison = pd.DataFrame(results).T.sort_values("roc_auc_mean", ascending=False)
    output_path = REPORTS_DIR / "baselines" / "comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=True)
    logger.info("comparacao_baselines_salva", extra={"path": str(output_path)})
    return results


def main() -> None:
    """Ponto de entrada do comando train-baselines."""
    run_all_baselines()


if __name__ == "__main__":
    main()
