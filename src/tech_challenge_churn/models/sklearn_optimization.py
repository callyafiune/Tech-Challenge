"""Experimentos Scikit-Learn permitidos para otimizar churn com MLflow."""

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
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from tech_challenge_churn.config import DATA_PATH, MLRUNS_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import (
    BusinessAssumptions,
    assumptions_to_dict,
    compute_business_metrics,
    find_best_business_threshold,
    lift_at_top_fraction,
    precision_at_top_fraction,
    recall_at_top_fraction,
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
EXPERIMENT_NAME = "telco-churn-sklearn-optimization"
FEATURE_SET_NAME = "telco_expanded_v1"


def build_hgb_classifier(
    *,
    seed: int = RANDOM_SEED,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 15,
    l2_regularization: float = 0.1,
) -> HistGradientBoostingClassifier:
    """Cria um HistGradientBoosting regularizado e balanceado."""
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=250,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=25,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        scoring="loss",
        class_weight="balanced",
        random_state=seed,
    )


def build_logistic_elasticnet(seed: int = RANDOM_SEED) -> LogisticRegression:
    """Cria regressao logistica ElasticNet para as features expandidas."""
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.35,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        n_jobs=-1,
    )


def build_model_registry(seed: int = RANDOM_SEED) -> dict[str, BaseEstimator]:
    """Define a bateria de experimentos dentro das ferramentas permitidas."""
    logistic_l2 = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=seed,
        solver="liblinear",
    )
    hgb_regularized = build_hgb_classifier(seed=seed)
    hgb_deeper = build_hgb_classifier(
        seed=seed,
        learning_rate=0.035,
        max_leaf_nodes=31,
        l2_regularization=0.3,
    )
    stack_lr_hgb = StackingClassifier(
        estimators=[
            ("lr", logistic_l2),
            ("hgb", hgb_regularized),
        ],
        final_estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    return {
        "logistic_elasticnet_expanded": build_logistic_elasticnet(seed),
        "logistic_l2_calibrated_sigmoid": CalibratedClassifierCV(
            estimator=logistic_l2,
            method="sigmoid",
            cv=3,
        ),
        "hist_gradient_boosting_regularized": hgb_regularized,
        "hist_gradient_boosting_deeper": hgb_deeper,
        "hist_gradient_boosting_calibrated_sigmoid": CalibratedClassifierCV(
            estimator=build_hgb_classifier(seed=seed),
            method="sigmoid",
            cv=3,
        ),
        "stacking_lr_hgb": stack_lr_hgb,
    }


def build_experiment_pipeline(model: BaseEstimator) -> Pipeline:
    """Combina feature engineering e modelo em um pipeline sem vazamento."""
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("model", model),
        ]
    )


def _current_git_sha() -> str:
    """Obtem o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula medias e desvios das metricas por fold."""
    summary: dict[str, float] = {}
    for metric_name in sorted(fold_metrics[0]):
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def _safe_params(model: BaseEstimator) -> dict[str, str | int | float | bool | None]:
    """Normaliza parametros para respeitar limites do MLflow."""
    safe: dict[str, str | int | float | bool | None] = {}
    for key, value in model.get_params(deep=True).items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[f"model__{key}"] = value
        else:
            safe[f"model__{key}"] = str(value)[:500]
    return safe


def _save_evaluation_artifacts(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Salva curvas e relatorio de classificacao para auditoria."""
    artifact_dir = REPORTS_DIR / "sklearn_optimization" / model_name
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

    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10)
    plt.title(f"Calibracao - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "calibration_curve.png", dpi=140)
    plt.close()

    matrix = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"]).plot(values_format="d")
    plt.title(f"Matriz de confusao - {model_name}")
    plt.tight_layout()
    plt.savefig(artifact_dir / "confusion_matrix.png", dpi=140)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=["No", "Yes"], zero_division=0)
    (artifact_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    return artifact_dir


def _log_sklearn_model(model: Pipeline) -> None:
    """Registra o pipeline final no MLflow com compatibilidade entre versoes."""
    try:
        mlflow.sklearn.log_model(sk_model=model, name="model")
    except TypeError:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


def evaluate_and_log_model(
    model_name: str,
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
    n_splits: int = 5,
) -> dict[str, float]:
    """Avalia um modelo com CV externa e threshold escolhido em validacao interna."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, float]] = []
    all_true: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
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
            test_size=0.2,
            stratify=y_train_valid,
            random_state=seed + fold_index,
        )

        fold_pipeline = build_experiment_pipeline(clone(model))
        fold_pipeline.fit(x_train, y_train)

        valid_proba = fold_pipeline.predict_proba(x_valid)[:, 1]
        threshold_f1, valid_f1 = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)
        threshold_business, _ = find_best_business_threshold(
            y_valid.to_numpy(),
            valid_proba,
            x_valid["MonthlyCharges"],
            assumptions,
        )

        test_proba = fold_pipeline.predict_proba(x_test)[:, 1]
        y_test_array = y_test.to_numpy()
        y_pred_optimal = (test_proba >= threshold_f1).astype(int)

        metrics = probability_metrics(y_test_array, test_proba)
        metrics.update(threshold_metrics(y_test_array, test_proba, threshold=0.5))
        optimal_metrics = threshold_metrics(y_test_array, test_proba, threshold=threshold_f1)
        metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
        business_metrics = compute_business_metrics(
            y_test_array,
            test_proba,
            x_test["MonthlyCharges"],
            threshold_business,
            assumptions,
        )
        metrics.update(business_metrics)
        metrics["threshold_f1"] = threshold_f1
        metrics["threshold_business"] = threshold_business
        metrics["valid_f1_optimal"] = valid_f1
        metrics["lift_at_top_20pct"] = lift_at_top_fraction(y_test_array, test_proba)
        metrics["precision_at_top_20pct"] = precision_at_top_fraction(y_test_array, test_proba)
        metrics["recall_at_top_20pct"] = recall_at_top_fraction(y_test_array, test_proba)
        fold_metrics.append(metrics)

        all_true.append(y_test_array)
        all_proba.append(test_proba)
        all_pred.append(y_pred_optimal)

        logger.info(
            "fold_sklearn_otimizacao_avaliado",
            extra={
                "model_name": model_name,
                "fold": fold_index,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "optimal_f1": metrics["optimal_f1"],
                "threshold_f1": threshold_f1,
            },
        )

    summary_metrics = _summarize_fold_metrics(fold_metrics)
    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    y_pred = np.concatenate(all_pred)
    artifact_dir = _save_evaluation_artifacts(model_name, y_true, y_proba, y_pred)

    final_pipeline = build_experiment_pipeline(clone(model))
    final_pipeline.fit(features, target)
    model_dir = MODELS_DIR / "sklearn_optimization"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(final_pipeline, model_path)

    summary_path = REPORTS_DIR / "sklearn_optimization" / f"{model_name}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary_metrics": summary_metrics,
                "fold_metrics": fold_metrics,
                "feature_set": FEATURE_SET_NAME,
                "evaluation_protocol": "outer_cv_with_inner_threshold",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(
            {
                "stage": "optimization",
                "framework": "scikit-learn",
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "feature_set": FEATURE_SET_NAME,
                "evaluation_protocol": "outer_cv_with_inner_threshold",
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
                **_safe_params(model),
            }
        )
        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)
        for metric_name, value in summary_metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_dict(fold_metrics, "fold_metrics.json")
        mlflow.log_dict(summary_metrics, "summary_metrics.json")
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        mlflow.log_artifact(str(summary_path), artifact_path="reports")
        mlflow.log_artifact(str(model_path), artifact_path="serialized_model")
        _log_sklearn_model(final_pipeline)

    logger.info(
        "experimento_sklearn_registrado",
        extra={
            "model_name": model_name,
            "roc_auc_mean": summary_metrics["roc_auc_mean"],
            "pr_auc_mean": summary_metrics["pr_auc_mean"],
            "optimal_f1_mean": summary_metrics["optimal_f1_mean"],
            "model_path": str(model_path),
        },
    )
    return summary_metrics


def _write_report(comparison: pd.DataFrame) -> None:
    """Documenta resultados e cuidados metodologicos dos experimentos."""
    output_path = Path("docs") / "sklearn_optimization_report.md"
    best_row = comparison.iloc[0]
    best_pr_auc_row = comparison.sort_values("pr_auc_mean", ascending=False).iloc[0]
    report = f"""# Experimentos Scikit-Learn - Otimizacao Permitida

## Escopo

Esta bateria foi executada apenas com ferramentas já adotadas no projeto: Scikit-Learn para
modelagem tabular e MLflow para rastreamento integral. Nao foram adicionadas dependencias externas
como XGBoost, LightGBM ou CatBoost.

## Protocolo

- Validacao cruzada estratificada externa com 5 folds.
- Split interno dentro de cada fold para escolher o threshold de F1 e o threshold de negocio.
- Pipeline unico com feature engineering, imputacao, escala, one-hot encoding e modelo.
- Registro obrigatorio de todos os experimentos no MLflow (`{EXPERIMENT_NAME}`).
- Metricas: AUC-ROC, PR-AUC, Brier, log loss, F1, F1 otimizado, lift@20%, precision@20%,
  recall@20% e valor de negocio estimado.

## Melhor Resultado

- Modelo: `{best_row["model"]}`.
- AUC-ROC media: `{best_row["roc_auc_mean"]:.4f}`.
- PR-AUC media: `{best_row["pr_auc_mean"]:.4f}`.
- F1 em threshold 0,5 medio: `{best_row["f1_mean"]:.4f}`.
- F1 com threshold interno medio: `{best_row["optimal_f1_mean"]:.4f}`.
- Lift@20% medio: `{best_row["lift_at_top_20pct_mean"]:.4f}`.

Melhor PR-AUC entre os experimentos:

- Modelo: `{best_pr_auc_row["model"]}`.
- PR-AUC media: `{best_pr_auc_row["pr_auc_mean"]:.4f}`.
- F1 com threshold interno medio: `{best_pr_auc_row["optimal_f1_mean"]:.4f}`.

## Leitura Critica

O resultado deve ser comparado contra a Regressao Logistica balanceada registrada previamente no
MLflow. Como o threshold foi escolhido apenas no split interno de cada fold, a estimativa reduz o
risco de overfitting em relacao a otimizar diretamente no fold de teste. PR-AUC e lift@20% devem ter
mais peso que F1 isolado, pois o problema e desbalanceado e a operacao tende a acionar uma campanha
de retencao sobre uma fracao dos clientes.

Arquivo comparativo: `reports/sklearn_optimization/comparison.csv`.
"""
    output_path.write_text(report, encoding="utf-8")


def run_sklearn_optimization_experiments() -> pd.DataFrame:
    """Executa todos os experimentos permitidos e registra no MLflow."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    results: list[dict[str, float | str]] = []

    for model_name, model in build_model_registry(RANDOM_SEED).items():
        summary = evaluate_and_log_model(model_name, model, features, target)
        results.append({"model": model_name, **summary})

    comparison = pd.DataFrame(results).sort_values(
        ["optimal_f1_mean", "pr_auc_mean", "roc_auc_mean"],
        ascending=False,
    )
    output_dir = REPORTS_DIR / "sklearn_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    _write_report(comparison)

    logger.info(
        "experimentos_sklearn_otimizacao_concluidos",
        extra={
            "best_model": comparison.iloc[0]["model"],
            "best_optimal_f1_mean": comparison.iloc[0]["optimal_f1_mean"],
            "best_pr_auc_mean": comparison.iloc[0]["pr_auc_mean"],
        },
    )
    return comparison


def main() -> None:
    """Ponto de entrada do comando train-sklearn-optimization."""
    run_sklearn_optimization_experiments()


if __name__ == "__main__":
    main()
