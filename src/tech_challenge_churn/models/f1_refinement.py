"""Refinamento controlado de F1 dentro do escopo Scikit-Learn."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from tech_challenge_churn.config import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    DATA_PATH,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import lift_at_top_fraction
from tech_challenge_churn.evaluation.metrics import (
    find_best_f1_threshold,
    probability_metrics,
    threshold_metrics,
)
from tech_challenge_churn.features.build import add_telco_features
from tech_challenge_churn.models.feature_ablation import build_champion_classifier
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-f1-refinement"


@dataclass(frozen=True)
class RefinementFeatureSpec:
    """Define um conjunto de features para refinamento de F1."""

    name: str
    description: str
    drop_numeric: tuple[str, ...] = ()
    drop_categorical: tuple[str, ...] = ()
    add_numeric: tuple[str, ...] = ()
    add_categorical: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RefinementCandidate:
    """Representa um candidato de modelo e feature set."""

    name: str
    family: str
    feature_spec: RefinementFeatureSpec
    classifier: BaseEstimator
    params: dict[str, Any]


def _current_git_sha() -> str:
    """Obtém o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _safe_params(params: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    """Normaliza parâmetros para logging no MLflow."""
    safe: dict[str, str | int | float | bool | None] = {}
    for key, value in params.items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)[:500]
    return safe


def add_refinement_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria interações Telco sem usar o alvo."""
    engineered = add_telco_features(data)
    service_denominator = engineered["num_services"].clip(lower=1)
    tenure_plus_one = engineered["tenure"] + 1
    manual_payment = ~engineered["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    )

    engineered["has_family_context"] = (
        engineered["Partner"].eq("Yes") | engineered["Dependents"].eq("Yes")
    ).astype(int)
    engineered["monthly_charge_per_service"] = (
        engineered["MonthlyCharges"] / service_denominator
    ).replace([np.inf, -np.inf], np.nan)
    engineered["tenure_charge_pressure"] = engineered["MonthlyCharges"] / tenure_plus_one
    engineered["no_protection_services"] = engineered["num_protection_services"].eq(0).astype(int)
    engineered["no_security_or_support"] = (
        engineered["OnlineSecurity"].eq("No") & engineered["TechSupport"].eq("No")
    ).astype(int)
    engineered["fiber_month_to_month"] = (
        engineered["InternetService"].eq("Fiber optic")
        & engineered["Contract"].eq("Month-to-month")
    ).astype(int)
    engineered["no_protection_fiber_month_to_month"] = (
        engineered["fiber_month_to_month"].eq(1)
        & engineered["num_protection_services"].eq(0)
    ).astype(int)
    engineered["manual_payment"] = manual_payment.astype(int)
    engineered["manual_payment_month_to_month"] = (
        manual_payment & engineered["Contract"].eq("Month-to-month")
    ).astype(int)
    engineered["low_tenure_high_monthly"] = (
        engineered["tenure"].le(12) & engineered["MonthlyCharges"].ge(70)
    ).astype(int)
    engineered["monthly_charge_bucket"] = pd.cut(
        engineered["MonthlyCharges"],
        bins=[-0.1, 35, 65, 85, np.inf],
        labels=["low", "medium", "high", "very_high"],
    ).astype("object")
    engineered["charge_tenure_segment"] = (
        engineered["monthly_charge_bucket"].astype(str)
        + "__"
        + engineered["tenure_bucket"].astype(str)
    )
    engineered["contract_payment_tenure_profile"] = (
        engineered["Contract"].astype(str)
        + "__"
        + engineered["PaymentMethod"].astype(str)
        + "__"
        + engineered["tenure_bucket"].astype(str)
    )
    return engineered


REFINEMENT_NUMERIC_FEATURES = (
    "has_family_context",
    "monthly_charge_per_service",
    "tenure_charge_pressure",
    "no_protection_services",
    "no_security_or_support",
    "fiber_month_to_month",
    "no_protection_fiber_month_to_month",
    "manual_payment",
    "manual_payment_month_to_month",
    "low_tenure_high_monthly",
)

REFINEMENT_CATEGORICAL_FEATURES = (
    "monthly_charge_bucket",
    "charge_tenure_segment",
    "contract_payment_tenure_profile",
)


def build_feature_specs() -> dict[str, RefinementFeatureSpec]:
    """Define conjuntos de features candidatos."""
    return {
        "no_gender_current": RefinementFeatureSpec(
            name="no_gender_current",
            description="Conjunto atual sem gender, usado como referência justa.",
            drop_categorical=("gender",),
            tags=("reference", "fairness"),
        ),
        "no_gender_refined": RefinementFeatureSpec(
            name="no_gender_refined",
            description="Remove gender e adiciona interações de contrato, cobrança e proteção.",
            drop_categorical=("gender",),
            add_numeric=REFINEMENT_NUMERIC_FEATURES,
            add_categorical=REFINEMENT_CATEGORICAL_FEATURES,
            tags=("fairness", "interactions"),
        ),
        "family_refined": RefinementFeatureSpec(
            name="family_refined",
            description="Remove gender e resume Partner/Dependents em has_family_context.",
            drop_categorical=("gender", "Partner", "Dependents"),
            add_numeric=REFINEMENT_NUMERIC_FEATURES,
            add_categorical=REFINEMENT_CATEGORICAL_FEATURES,
            tags=("fairness", "interactions", "family_summary"),
        ),
    }


def _feature_lists(spec: RefinementFeatureSpec) -> tuple[list[str], list[str]]:
    """Calcula as listas finais de features por especificação."""
    numeric_features = [
        feature for feature in BASE_NUMERIC_FEATURES if feature not in spec.drop_numeric
    ]
    categorical_features = [
        feature
        for feature in BASE_CATEGORICAL_FEATURES
        if feature not in spec.drop_categorical
    ]
    for feature in spec.add_numeric:
        if feature not in numeric_features:
            numeric_features.append(feature)
    for feature in spec.add_categorical:
        if feature not in categorical_features:
            categorical_features.append(feature)
    return numeric_features, categorical_features


def build_refinement_feature_pipeline(spec: RefinementFeatureSpec) -> Pipeline:
    """Monta o pré-processamento para um feature set de refinamento."""
    numeric_features, categorical_features = _feature_lists(spec)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_refinement_features, validate=False)),
            ("preprocessor", preprocessor),
        ]
    )


def build_refinement_pipeline(
    feature_spec: RefinementFeatureSpec,
    classifier: BaseEstimator,
) -> Pipeline:
    """Combina feature engineering e classificador."""
    return Pipeline(
        steps=[
            ("features", build_refinement_feature_pipeline(feature_spec)),
            ("model", classifier),
        ]
    )


def _rf_classifier(
    *,
    seed: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | float,
    class_weight: str | dict[int, float],
) -> RandomForestClassifier:
    """Cria RandomForest dentro da vizinhança do campeão atual."""
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )


def _extra_trees_classifier(
    *,
    seed: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
) -> ExtraTreesClassifier:
    """Cria ExtraTrees como alternativa robusta para tabular."""
    return ExtraTreesClassifier(
        n_estimators=700,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _hgb_classifier(
    *,
    seed: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
) -> HistGradientBoostingClassifier:
    """Cria HistGradientBoosting dentro do escopo Scikit-Learn."""
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=450,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=seed,
    )


def build_candidate_registry(seed: int = RANDOM_SEED) -> list[RefinementCandidate]:
    """Define uma busca curta e deliberada ao redor dos melhores modelos atuais."""
    feature_specs = build_feature_specs()
    return [
        RefinementCandidate(
            name="rf_no_gender_reference",
            family="random_forest",
            feature_spec=feature_specs["no_gender_current"],
            classifier=build_champion_classifier(seed),
            params={"feature_set": "no_gender_current", "classifier": "champion_random_forest"},
        ),
        RefinementCandidate(
            name="rf_no_gender_refined",
            family="random_forest",
            feature_spec=feature_specs["no_gender_refined"],
            classifier=build_champion_classifier(seed),
            params={"feature_set": "no_gender_refined", "classifier": "champion_random_forest"},
        ),
        RefinementCandidate(
            name="rf_family_refined",
            family="random_forest",
            feature_spec=feature_specs["family_refined"],
            classifier=build_champion_classifier(seed),
            params={"feature_set": "family_refined", "classifier": "champion_random_forest"},
        ),
        RefinementCandidate(
            name="rf_refined_depth10_leaf2",
            family="random_forest",
            feature_spec=feature_specs["no_gender_refined"],
            classifier=_rf_classifier(
                seed=seed,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
            ),
            params={
                "feature_set": "no_gender_refined",
                "n_estimators": 700,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
        ),
        RefinementCandidate(
            name="rf_refined_depth12_leaf2_weight14",
            family="random_forest",
            feature_spec=feature_specs["no_gender_refined"],
            classifier=_rf_classifier(
                seed=seed,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.35,
                class_weight={0: 1.0, 1: 1.4},
            ),
            params={
                "feature_set": "no_gender_refined",
                "n_estimators": 700,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": 0.35,
                "class_weight": "{0: 1.0, 1: 1.4}",
            },
        ),
        RefinementCandidate(
            name="extra_trees_family_refined",
            family="extra_trees",
            feature_spec=feature_specs["family_refined"],
            classifier=_extra_trees_classifier(
                seed=seed,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
            ),
            params={
                "feature_set": "family_refined",
                "n_estimators": 700,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        ),
        RefinementCandidate(
            name="hgb_no_gender_refined",
            family="hgb",
            feature_spec=feature_specs["no_gender_refined"],
            classifier=_hgb_classifier(
                seed=seed,
                learning_rate=0.045,
                max_leaf_nodes=23,
                min_samples_leaf=20,
                l2_regularization=0.3,
            ),
            params={
                "feature_set": "no_gender_refined",
                "learning_rate": 0.045,
                "max_leaf_nodes": 23,
                "min_samples_leaf": 20,
                "l2_regularization": 0.3,
                "class_weight": "balanced",
            },
        ),
        RefinementCandidate(
            name="stack_rf_hgb_refined",
            family="stacking",
            feature_spec=feature_specs["no_gender_refined"],
            classifier=StackingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1_000,
                            random_state=seed,
                            solver="liblinear",
                        ),
                    ),
                    ("rf", build_champion_classifier(seed)),
                    (
                        "hgb",
                        _hgb_classifier(
                            seed=seed,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            min_samples_leaf=25,
                            l2_regularization=0.1,
                        ),
                    ),
                ],
                final_estimator=LogisticRegression(
                    class_weight="balanced",
                    C=0.35,
                    max_iter=1_000,
                    random_state=seed,
                    solver="liblinear",
                ),
                stack_method="predict_proba",
                cv=5,
                n_jobs=1,
            ),
            params={
                "feature_set": "no_gender_refined",
                "classifier": "stacking_lr_rf_hgb",
                "meta_c": 0.35,
                "n_jobs": 1,
            },
        ),
    ]


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula média e desvio por métrica."""
    summary: dict[str, float] = {}
    for metric_name in sorted(fold_metrics[0]):
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def _feature_count(feature_spec: RefinementFeatureSpec, features: pd.DataFrame) -> int:
    """Conta colunas finais após pré-processamento."""
    pipeline = build_refinement_feature_pipeline(feature_spec)
    pipeline.fit(features)
    return int(len(pipeline.named_steps["preprocessor"].get_feature_names_out()))


def evaluate_candidate(
    candidate: RefinementCandidate,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
    n_splits: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Avalia um candidato com CV externa e threshold interno."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
            test_size=0.2,
            stratify=y_train_valid,
            random_state=seed + fold_index,
        )

        model = build_refinement_pipeline(candidate.feature_spec, clone(candidate.classifier))
        model.fit(x_train, y_train)
        valid_proba = model.predict_proba(x_valid)[:, 1]
        threshold_f1, valid_f1 = find_best_f1_threshold(y_valid.to_numpy(), valid_proba)

        test_proba = model.predict_proba(x_test)[:, 1]
        y_test_array = y_test.to_numpy()
        metrics = probability_metrics(y_test_array, test_proba)
        metrics.update(threshold_metrics(y_test_array, test_proba, threshold=0.5))
        optimal_metrics = threshold_metrics(y_test_array, test_proba, threshold=threshold_f1)
        metrics.update({f"optimal_{key}": value for key, value in optimal_metrics.items()})
        metrics["threshold_f1"] = threshold_f1
        metrics["valid_f1_optimal"] = valid_f1
        metrics["lift_at_top_20pct"] = lift_at_top_fraction(y_test_array, test_proba)
        fold_metrics.append(metrics)

        logger.info(
            "fold_refinamento_f1_avaliado",
            extra={
                "model_name": candidate.name,
                "fold": fold_index,
                "f1": metrics["f1"],
                "optimal_f1": metrics["optimal_f1"],
                "threshold_f1": threshold_f1,
            },
        )

    return _summarize_fold_metrics(fold_metrics), fold_metrics


def _log_candidate_run(
    candidate: RefinementCandidate,
    summary: dict[str, float],
    fold_metrics: list[dict[str, float]],
    feature_count: int,
) -> None:
    """Registra um candidato no MLflow."""
    numeric_features, categorical_features = _feature_lists(candidate.feature_spec)
    with mlflow.start_run(run_name=candidate.name):
        mlflow.set_tags(
            {
                "stage": "f1_refinement",
                "framework": "scikit-learn",
                "family": candidate.family,
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": candidate.name,
                "feature_set": candidate.feature_spec.name,
            }
        )
        mlflow.log_params(
            _safe_params(
                {
                    **candidate.params,
                    "seed": RANDOM_SEED,
                    "n_splits": 5,
                    "inner_validation_fraction": 0.2,
                    "evaluation_protocol": "outer_cv_with_inner_threshold",
                    "final_feature_count": feature_count,
                    "numeric_features": ",".join(numeric_features),
                    "categorical_features": ",".join(categorical_features),
                }
            )
        )
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)
        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)
        mlflow.log_dict(summary, "summary_metrics.json")
        mlflow.log_dict(fold_metrics, "fold_metrics.json")


def _log_failed_candidate(candidate: RefinementCandidate, error: Exception) -> None:
    """Registra falha de candidato sem interromper a bateria."""
    with mlflow.start_run(run_name=f"{candidate.name}_failed"):
        mlflow.set_tags(
            {
                "stage": "f1_refinement",
                "framework": "scikit-learn",
                "family": candidate.family,
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": candidate.name,
                "feature_set": candidate.feature_spec.name,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
            }
        )
        mlflow.log_params(_safe_params(candidate.params))


def _write_report(comparison: pd.DataFrame) -> None:
    """Documenta os achados do refinamento de F1."""
    output_path = Path("docs") / "f1_refinement_report.md"
    reference = comparison.query("model == 'rf_no_gender_reference'").iloc[0]
    best_fixed = comparison.sort_values("f1_mean", ascending=False).iloc[0]
    best_internal = comparison.sort_values("optimal_f1_mean", ascending=False).iloc[0]
    report = f"""# Refinamento de F1 sem Vazamento

## Objetivo

Testar ajustes finos sugeridos para churn tabular sem sair do escopo técnico do projeto. A bateria
mantém Scikit-Learn, PyTorch e MLflow como ferramentas de referência e não usa XGBoost, LightGBM,
CatBoost ou oversampling sintético.

## Protocolo

- Validação cruzada estratificada externa com 5 folds.
- Em cada fold, o threshold de F1 é escolhido apenas em split interno de validação.
- O fold externo é usado somente para estimativa final de métricas.
- Todos os candidatos são registrados no MLflow em `{EXPERIMENT_NAME}`.
- As novas features usam apenas atributos disponíveis no payload, sem usar `Churn`.

## Referência

- Modelo: `{reference["model"]}`.
- Feature set: `{reference["feature_set"]}`.
- Features finais: `{int(reference["final_feature_count"])}`.
- F1 médio em threshold 0,5: `{reference["f1_mean"]:.4f}`.
- F1 médio com threshold interno: `{reference["optimal_f1_mean"]:.4f}`.
- PR-AUC média: `{reference["pr_auc_mean"]:.4f}`.

## Melhor F1 em Threshold 0,5

- Modelo: `{best_fixed["model"]}`.
- Família: `{best_fixed["family"]}`.
- Feature set: `{best_fixed["feature_set"]}`.
- Features finais: `{int(best_fixed["final_feature_count"])}`.
- F1 médio: `{best_fixed["f1_mean"]:.4f}`.
- Delta contra referência: `{best_fixed["delta_f1_mean"]:+.4f}`.
- PR-AUC média: `{best_fixed["pr_auc_mean"]:.4f}`.

## Melhor F1 com Threshold Interno

- Modelo: `{best_internal["model"]}`.
- Família: `{best_internal["family"]}`.
- Feature set: `{best_internal["feature_set"]}`.
- F1 médio com threshold interno: `{best_internal["optimal_f1_mean"]:.4f}`.
- Delta contra referência: `{best_internal["delta_optimal_f1_mean"]:+.4f}`.

## Leitura Crítica

O protocolo evita vazamento ao escolher threshold somente no split interno. Diferenças pequenas de
F1 devem ser tratadas com cautela, porque podem estar dentro da variância entre folds. Se nenhum
candidato superar a referência de forma consistente, a recomendação permanece manter o RandomForest
sem `gender` como challenger operacional e a MLP como modelo neural principal.

Artefato comparativo: `reports/f1_refinement/comparison.csv`.
"""
    output_path.write_text(report, encoding="utf-8")


def run_f1_refinement() -> pd.DataFrame:
    """Executa a bateria incremental de refinamento de F1."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    rows: list[dict[str, Any]] = []

    for candidate in build_candidate_registry(RANDOM_SEED):
        try:
            summary, fold_metrics = evaluate_candidate(candidate, features, target)
            feature_count = _feature_count(candidate.feature_spec, features)
            _log_candidate_run(candidate, summary, fold_metrics, feature_count)
            rows.append(
                {
                    "model": candidate.name,
                    "family": candidate.family,
                    "feature_set": candidate.feature_spec.name,
                    "feature_description": candidate.feature_spec.description,
                    "final_feature_count": feature_count,
                    **summary,
                }
            )
        except Exception as error:
            _log_failed_candidate(candidate, error)
            logger.warning(
                "refinamento_f1_candidato_falhou",
                extra={"model_name": candidate.name, "error": str(error)},
            )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise RuntimeError("Nenhum candidato de refinamento de F1 foi avaliado com sucesso.")
    reference = comparison.query("model == 'rf_no_gender_reference'").iloc[0]
    comparison["delta_f1_mean"] = comparison["f1_mean"] - float(reference["f1_mean"])
    comparison["delta_optimal_f1_mean"] = comparison["optimal_f1_mean"] - float(
        reference["optimal_f1_mean"]
    )
    comparison["delta_pr_auc_mean"] = comparison["pr_auc_mean"] - float(reference["pr_auc_mean"])
    comparison = comparison.sort_values(
        ["f1_mean", "optimal_f1_mean", "pr_auc_mean"],
        ascending=False,
    )

    output_dir = REPORTS_DIR / "f1_refinement"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    best_name = str(comparison.iloc[0]["model"])
    best_candidate = {
        candidate.name: candidate for candidate in build_candidate_registry(RANDOM_SEED)
    }[best_name]
    best_model = build_refinement_pipeline(
        best_candidate.feature_spec,
        clone(best_candidate.classifier),
    )
    best_model.fit(features, target)
    model_dir = MODELS_DIR / "f1_refinement"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / f"{best_name}.joblib")

    _write_report(comparison)
    logger.info(
        "refinamento_f1_concluido",
        extra={
            "best_model": best_name,
            "best_f1_mean": comparison.iloc[0]["f1_mean"],
            "best_optimal_f1_mean": comparison.iloc[0]["optimal_f1_mean"],
        },
    )
    return comparison


def main() -> None:
    """Ponto de entrada do comando train-f1-refinement."""
    run_f1_refinement()


if __name__ == "__main__":
    main()
