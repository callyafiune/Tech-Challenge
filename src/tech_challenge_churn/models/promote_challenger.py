"""Promocao formal do challenger operacional com MLflow."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from tech_challenge_churn.config import DATA_PATH, MLFLOW_TRACKING_URI, MODELS_DIR, RANDOM_SEED
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
from tech_challenge_churn.models.feature_ablation import (
    build_ablation_model,
    build_ablation_registry,
)
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)

EXPERIMENT_NAME = "telco-churn-model-promotion"
REGISTERED_MODEL_NAME = "telco-churn-random-forest-challenger"
REGISTRY_ALIAS = "challenger"
PROMOTED_VERSION = "random_forest_no_gender_v1"
PROMOTED_MODEL_ROLE = "operational_challenger"


def _current_git_sha() -> str:
    """Obtem o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _log_sklearn_model(model: object) -> str:
    """Registra o modelo no MLflow e tenta criar entrada no registry local."""
    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    except Exception as error:
        mlflow.set_tag("registry_status", "registry_unavailable")
        mlflow.set_tag("registry_error", str(error)[:500])
        logger.warning(
            "registro_model_registry_indisponivel",
            extra={"error": str(error), "registered_model_name": REGISTERED_MODEL_NAME},
        )
        try:
            model_info = mlflow.sklearn.log_model(sk_model=model, name="model")
        except TypeError:
            model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    else:
        mlflow.set_tag("registry_status", "registered")

    return getattr(model_info, "model_uri", "unavailable")


def _tag_registered_model_version(run_id: str) -> str:
    """Marca a versão registrada como challenger quando o registry estiver disponível."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        matching_versions = [version for version in versions if version.run_id == run_id]
        if not matching_versions:
            return "unavailable"
        registry_version = str(matching_versions[0].version)
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            REGISTRY_ALIAS,
            registry_version,
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "model_role",
            PROMOTED_MODEL_ROLE,
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "promotion_status",
            "promoted_challenger",
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registry_version,
            "semantic_version",
            PROMOTED_VERSION,
        )
    except Exception as error:
        mlflow.set_tag("registry_alias_status", "alias_unavailable")
        mlflow.set_tag("registry_alias_error", str(error)[:500])
        logger.warning(
            "alias_model_registry_indisponivel",
            extra={"error": str(error), "registered_model_name": REGISTERED_MODEL_NAME},
        )
        return "unavailable"
    mlflow.set_tag("registry_alias_status", f"alias_{REGISTRY_ALIAS}_set")
    return registry_version


def _promotion_spec() -> Any:
    """Seleciona o feature set promovido a partir da ablação."""
    return next(spec for spec in build_ablation_registry() if spec.name == "no_gender")


def _evaluate_for_promotion(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, float], float, float, int]:
    """Treina em split de desenvolvimento e calcula metricas de promocao."""
    spec = _promotion_spec()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_SEED,
    )
    model = build_ablation_model(spec)
    model.fit(x_train, y_train)
    valid_proba = model.predict_proba(x_valid)[:, 1]
    threshold_f1, f1_optimal = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)
    threshold_business, business_selection_metrics = find_best_business_threshold(
        y_valid.to_numpy(),
        valid_proba,
        x_valid["MonthlyCharges"],
        BusinessAssumptions(),
    )

    metrics = probability_metrics(y_valid.to_numpy(), valid_proba)
    metrics.update(threshold_metrics(y_valid.to_numpy(), valid_proba, threshold=0.5))
    optimal_metrics = threshold_metrics(y_valid.to_numpy(), valid_proba, threshold=threshold_f1)
    metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
    business_metrics = compute_business_metrics(
        y_valid.to_numpy(),
        valid_proba,
        x_valid["MonthlyCharges"],
        threshold_business,
        BusinessAssumptions(),
    )
    metrics.update(business_metrics)
    metrics["threshold_f1"] = threshold_f1
    metrics["threshold_business"] = threshold_business
    metrics["f1_optimal_internal"] = f1_optimal
    metrics["lift_at_top_20pct"] = lift_at_top_fraction(y_valid.to_numpy(), valid_proba)
    metrics["business_selection_incremental_savings"] = business_selection_metrics[
        "business_incremental_savings"
    ]
    feature_count = int(
        len(model.named_steps["features"].named_steps["preprocessor"].get_feature_names_out())
    )
    return metrics, threshold_f1, threshold_business, feature_count


def _write_promotion_report(metadata: dict[str, Any]) -> Path:
    """Gera documento curto da promoção formal."""
    output_path = Path("docs") / "promocao_challenger.md"
    metrics = metadata["validation_metrics"]
    report = f"""# Promoção Formal - Challenger Operacional

## Modelo Promovido

- Nome: `{metadata["model_name"]}`.
- Versão: `{metadata["version"]}`.
- Papel: `{metadata["model_role"]}`.
- Status: `{metadata["promotion_status"]}`.
- Feature set: `{metadata["feature_set"]}`.
- Atributo removido: `gender`.

## Protocolo

- Split holdout estratificado de 20% para métricas de promoção.
- Treino final do artefato promovido com 100% da base após escolha dos thresholds.
- Registro no MLflow em `{EXPERIMENT_NAME}`.
- Hash SHA256 do dataset registrado no metadata.

## Métricas de Promoção

- AUC-ROC validação: `{metrics["roc_auc"]:.4f}`.
- PR-AUC validação: `{metrics["pr_auc"]:.4f}`.
- F1 em threshold 0,5: `{metrics["f1"]:.4f}`.
- F1 no threshold promovido: `{metrics["optimal_f1"]:.4f}`.
- Threshold F1 promovido: `{metadata["threshold_f1"]:.2f}`.
- Threshold de negócio promovido: `{metadata["threshold_business"]:.2f}`.
- Lift@20%: `{metrics["lift_at_top_20pct"]:.4f}`.

## Artefatos

- Modelo local: `{metadata["model_path"]}`.
- Metadata local: `{metadata["metadata_path"]}`.
- MLflow run ID: `{metadata["mlflow_run_id"]}`.
- MLflow model URI: `{metadata["mlflow_model_uri"]}`.

## Decisão

O RandomForest sem `gender` foi promovido como challenger operacional, não como substituto direto da
MLP. A MLP segue como modelo neural principal, enquanto o RandomForest permanece disponível para
validação operacional em shadow mode.
"""
    output_path.write_text(report, encoding="utf-8")
    return output_path


def promote_random_forest_challenger() -> dict[str, Any]:
    """Promove formalmente o RandomForest sem gender como challenger operacional."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    spec = _promotion_spec()
    dataset_hash = compute_file_hash(DATA_PATH)
    git_sha = _current_git_sha()
    metrics, threshold_f1, threshold_business, feature_count = _evaluate_for_promotion(
        features,
        target,
    )

    final_model = build_ablation_model(spec)
    final_model.fit(features, target)

    model_dir = MODELS_DIR / "challengers" / PROMOTED_VERSION
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    joblib.dump(final_model, model_path)

    metadata: dict[str, Any] = {
        "model_name": REGISTERED_MODEL_NAME,
        "version": PROMOTED_VERSION,
        "model_role": PROMOTED_MODEL_ROLE,
        "promotion_status": "promoted_challenger",
        "feature_set": spec.name,
        "removed_features": ["gender"],
        "dataset_path": str(DATA_PATH),
        "dataset_sha256": dataset_hash,
        "git_sha": git_sha,
        "promoted_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(features)),
        "positive_rate": float(target.mean()),
        "final_feature_count": feature_count,
        "threshold_f1": threshold_f1,
        "threshold_business": threshold_business,
        "validation_metrics": metrics,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "mlflow_run_id": "pending",
        "mlflow_model_uri": "pending",
        "mlflow_registry_version": "pending",
        "mlflow_registry_alias": REGISTRY_ALIAS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    with mlflow.start_run(run_name=PROMOTED_VERSION) as run:
        mlflow.set_tags(
            {
                "stage": "model_promotion",
                "promotion_status": "promoted_challenger",
                "model_role": PROMOTED_MODEL_ROLE,
                "model_name": REGISTERED_MODEL_NAME,
                "model_version": PROMOTED_VERSION,
                "feature_set": spec.name,
                "dataset_sha256": dataset_hash,
                "git_sha": git_sha,
            }
        )
        mlflow.log_params(
            {
                "seed": RANDOM_SEED,
                "rows": len(features),
                "positive_rate": float(target.mean()),
                "final_feature_count": feature_count,
                "threshold_f1": threshold_f1,
                "threshold_business": threshold_business,
                "removed_features": ",".join(metadata["removed_features"]),
                **assumptions_to_dict(BusinessAssumptions()),
            }
        )
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_artifact(str(model_path), artifact_path="promoted_model")
        mlflow.log_artifact(str(metadata_path), artifact_path="promoted_model")
        model_uri = _log_sklearn_model(final_model)
        registry_version = _tag_registered_model_version(run.info.run_id)

        metadata["mlflow_run_id"] = run.info.run_id
        metadata["mlflow_model_uri"] = model_uri
        metadata["mlflow_registry_version"] = registry_version
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(metadata_path), artifact_path="promoted_model")

    report_path = _write_promotion_report(metadata)
    metadata["promotion_report_path"] = str(report_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "challenger_promovido",
        extra={
            "model_version": PROMOTED_VERSION,
            "model_path": str(model_path),
            "mlflow_run_id": metadata["mlflow_run_id"],
            "threshold_f1": threshold_f1,
            "threshold_business": threshold_business,
        },
    )
    return metadata


def main() -> None:
    """Ponto de entrada do comando promote-challenger."""
    promote_random_forest_challenger()


if __name__ == "__main__":
    main()
