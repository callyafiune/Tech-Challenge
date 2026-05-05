"""Experimentos de ablação de features com MLflow."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from tech_challenge_churn.config import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    DATA_PATH,
    MLFLOW_TRACKING_URI,
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
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-feature-ablation"


@dataclass(frozen=True)
class FeatureSetSpec:
    """Define uma versão do conjunto de features para teste de ablação."""

    name: str
    description: str
    drop_numeric: tuple[str, ...] = ()
    drop_categorical: tuple[str, ...] = ()
    add_numeric: tuple[str, ...] = ()
    add_categorical: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


def _current_git_sha() -> str:
    """Obtém o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def add_ablation_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona sumarizações candidatas antes da seleção de colunas."""
    engineered = add_telco_features(data)
    engineered["has_family_context"] = (
        engineered["Partner"].eq("Yes") | engineered["Dependents"].eq("Yes")
    ).astype(int)
    engineered["service_intensity_bucket"] = pd.cut(
        engineered["num_services"],
        bins=[-0.1, 1, 3, 5, np.inf],
        labels=["0-1", "2-3", "4-5", "6+"],
    ).astype("object")
    engineered["protection_level"] = pd.cut(
        engineered["num_protection_services"],
        bins=[-0.1, 0, 2, 4],
        labels=["none", "partial", "full"],
    ).astype("object")
    return engineered


def build_ablation_registry() -> list[FeatureSetSpec]:
    """Lista ablações e sumarizações candidatas para validação."""
    protection_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    )
    streaming_columns = ("StreamingTV", "StreamingMovies")
    service_columns = (
        "PhoneService",
        "MultipleLines",
        *protection_columns,
        *streaming_columns,
    )

    return [
        FeatureSetSpec(
            name="full_current",
            description="Conjunto atual completo usado pelo campeão.",
            tags=("baseline",),
        ),
        FeatureSetSpec(
            name="no_gender",
            description="Remove gender por baixo sinal esperado e risco de fairness.",
            drop_categorical=("gender",),
            tags=("fairness", "low_signal"),
        ),
        FeatureSetSpec(
            name="no_gender_partner",
            description="Remove gender e Partner mantendo demais variáveis relacionais.",
            drop_categorical=("gender", "Partner"),
            tags=("low_signal",),
        ),
        FeatureSetSpec(
            name="relationship_summarized",
            description="Substitui Partner e Dependents por has_family_context.",
            drop_categorical=("Partner", "Dependents"),
            add_numeric=("has_family_context",),
            tags=("summary", "relationship"),
        ),
        FeatureSetSpec(
            name="no_gender_relationship_summarized",
            description="Remove gender e resume Partner/Dependents em has_family_context.",
            drop_categorical=("gender", "Partner", "Dependents"),
            add_numeric=("has_family_context",),
            tags=("summary", "fairness", "relationship"),
        ),
        FeatureSetSpec(
            name="no_demographics",
            description="Remove variáveis demográficas e derivada de idoso com contrato mensal.",
            drop_numeric=("SeniorCitizen", "senior_month_to_month"),
            drop_categorical=("gender", "Partner", "Dependents"),
            tags=("fairness", "demographic"),
        ),
        FeatureSetSpec(
            name="streaming_summarized",
            description="Substitui StreamingTV e StreamingMovies por streaming_bundle.",
            drop_categorical=streaming_columns,
            tags=("summary", "services"),
        ),
        FeatureSetSpec(
            name="protection_count_only",
            description="Remove serviços de proteção individuais e mantém apenas contagem.",
            drop_numeric=("has_protection_bundle", "fiber_without_security"),
            drop_categorical=(*protection_columns, "internet_security_profile"),
            tags=("summary", "services"),
        ),
        FeatureSetSpec(
            name="protection_level_summary",
            description="Resume proteções em uma categoria ordinal de cobertura.",
            drop_numeric=("num_protection_services", "has_protection_bundle"),
            drop_categorical=(*protection_columns, "internet_security_profile"),
            add_categorical=("protection_level",),
            tags=("summary", "services"),
        ),
        FeatureSetSpec(
            name="service_counts_only",
            description="Remove serviços individuais e mantém contagens/perfis sintéticos.",
            drop_categorical=service_columns,
            add_categorical=("service_intensity_bucket",),
            tags=("summary", "services"),
        ),
        FeatureSetSpec(
            name="charges_simplified",
            description="Remove derivadas muito correlacionadas com tenure e MonthlyCharges.",
            drop_numeric=("avg_monthly_spend", "total_to_monthly_ratio"),
            tags=("redundancy", "charges"),
        ),
        FeatureSetSpec(
            name="compact_fair_relationship_charges",
            description="Remove gender, resume família e simplifica derivadas de cobrança.",
            drop_numeric=("avg_monthly_spend", "total_to_monthly_ratio"),
            drop_categorical=("gender", "Partner", "Dependents"),
            add_numeric=("has_family_context",),
            tags=("summary", "fairness", "redundancy"),
        ),
        FeatureSetSpec(
            name="compact_operational",
            description=(
                "Remove demografia e serviços individuais, preservando resumos operacionais."
            ),
            drop_numeric=(
                "SeniorCitizen",
                "senior_month_to_month",
                "has_protection_bundle",
                "fiber_without_security",
                "avg_monthly_spend",
                "total_to_monthly_ratio",
            ),
            drop_categorical=(
                "gender",
                "Partner",
                "Dependents",
                *service_columns,
                "internet_security_profile",
            ),
            add_numeric=("has_family_context",),
            add_categorical=("service_intensity_bucket", "protection_level"),
            tags=("summary", "compact", "fairness"),
        ),
    ]


def _feature_lists(spec: FeatureSetSpec) -> tuple[list[str], list[str]]:
    """Calcula listas finais de features numéricas e categóricas."""
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


def build_feature_set_pipeline(spec: FeatureSetSpec) -> Pipeline:
    """Monta o pré-processamento para uma especificação de ablação."""
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
            ("feature_engineering", FunctionTransformer(add_ablation_features, validate=False)),
            ("preprocessor", preprocessor),
        ]
    )


def build_champion_classifier(seed: int = RANDOM_SEED) -> RandomForestClassifier:
    """Recria o classificador do RandomForest campeão do tuning."""
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def build_ablation_model(spec: FeatureSetSpec, classifier: BaseEstimator | None = None) -> Pipeline:
    """Combina feature set ablado com o classificador campeão."""
    return Pipeline(
        steps=[
            ("features", build_feature_set_pipeline(spec)),
            ("model", classifier if classifier is not None else build_champion_classifier()),
        ]
    )


def _summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula médias e desvios por métrica."""
    summary: dict[str, float] = {}
    for metric_name in sorted(fold_metrics[0]):
        values = np.array([fold[metric_name] for fold in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def _feature_count(spec: FeatureSetSpec, features: pd.DataFrame, target: pd.Series) -> int:
    """Obtém a quantidade de colunas finais geradas pelo pré-processador."""
    pipeline = build_feature_set_pipeline(spec)
    pipeline.fit(features, target)
    return int(len(pipeline.named_steps["preprocessor"].get_feature_names_out()))


def _log_ablation_run(
    spec: FeatureSetSpec,
    summary: dict[str, float],
    fold_metrics: list[dict[str, float]],
    feature_count: int,
) -> None:
    """Registra uma ablação no MLflow."""
    numeric_features, categorical_features = _feature_lists(spec)
    with mlflow.start_run(run_name=spec.name):
        mlflow.set_tags(
            {
                "stage": "feature_ablation",
                "framework": "scikit-learn",
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": spec.name,
                "tags": ",".join(spec.tags),
            }
        )
        mlflow.log_params(
            {
                "description": spec.description,
                "classifier": "random_forest_005_without_selector",
                "drop_numeric": ",".join(spec.drop_numeric),
                "drop_categorical": ",".join(spec.drop_categorical),
                "add_numeric": ",".join(spec.add_numeric),
                "add_categorical": ",".join(spec.add_categorical),
                "numeric_features": len(numeric_features),
                "categorical_features": len(categorical_features),
                "final_feature_count": feature_count,
                "seed": RANDOM_SEED,
            }
        )
        for metric_name, value in summary.items():
            mlflow.log_metric(metric_name, value)
        for fold_index, metrics in enumerate(fold_metrics, start=1):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_index}_{metric_name}", value)
        mlflow.log_dict(summary, "summary_metrics.json")
        mlflow.log_dict(fold_metrics, "fold_metrics.json")


def evaluate_feature_set(
    spec: FeatureSetSpec,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
    n_splits: int = 5,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Avalia um conjunto de features com CV externa e threshold interno."""
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
        model = build_ablation_model(spec, clone(build_champion_classifier(seed)))
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
            "fold_ablation_avaliado",
            extra={
                "feature_set": spec.name,
                "fold": fold_index,
                "f1": metrics["f1"],
                "optimal_f1": metrics["optimal_f1"],
                "pr_auc": metrics["pr_auc"],
            },
        )

    return _summarize_fold_metrics(fold_metrics), fold_metrics


def _write_report(comparison: pd.DataFrame) -> None:
    """Documenta os resultados da ablação."""
    output_path = Path("docs") / "feature_ablation_report.md"
    reference = comparison.query("model == 'full_current'").iloc[0]
    best_f1 = comparison.sort_values("f1_mean", ascending=False).iloc[0]
    best_compact = (
        comparison.query("is_non_degrading_f1 == True")
        .sort_values(["final_feature_count", "f1_mean"], ascending=[True, False])
        .head(1)
    )
    compact_text = "Nenhum conjunto reduzido preservou F1 pelo criterio estrito."
    if not best_compact.empty:
        row = best_compact.iloc[0]
        compact_text = (
            f"`{row['model']}` preservou F1 com `{int(row['final_feature_count'])}` features "
            f"finais e F1 medio `{row['f1_mean']:.4f}`."
        )

    report = f"""# Ablacao de Features - Telco Churn

## Objetivo

Avaliar se atributos de baixo valor preditivo ou grupos redundantes podem ser removidos ou
sumarizados sem piorar o F1 do campeao tabular.

## Protocolo

- Modelo fixo: RandomForest com os hiperparametros do campeao `random_forest_005`.
- Validacao cruzada estratificada com 5 folds.
- Threshold de F1 escolhido em split interno de cada fold.
- Todos os experimentos foram registrados no MLflow em `{EXPERIMENT_NAME}`.
- Criterio estrito de nao piora: F1 medio em threshold 0,5 maior ou igual ao conjunto completo.

## Referencia

- Feature set: `full_current`.
- Features finais: `{int(reference["final_feature_count"])}`.
- F1 medio em threshold 0,5: `{reference["f1_mean"]:.4f}`.
- F1 medio com threshold interno: `{reference["optimal_f1_mean"]:.4f}`.
- PR-AUC media: `{reference["pr_auc_mean"]:.4f}`.

## Melhor F1

- Feature set: `{best_f1["model"]}`.
- Features finais: `{int(best_f1["final_feature_count"])}`.
- F1 medio em threshold 0,5: `{best_f1["f1_mean"]:.4f}`.
- Delta contra referencia: `{best_f1["delta_f1_mean"]:+.4f}`.
- F1 medio com threshold interno: `{best_f1["optimal_f1_mean"]:.4f}`.

## Melhor Reducao Sem Piora Estrita

{compact_text}

## Leitura Critica

Ablacao deve ser interpretada com cuidado porque as diferencas de F1 sao pequenas e podem estar
dentro da variancia entre folds. Quando duas versoes empatam em F1, a versao com menos features e
menor risco etico deve ser preferida, especialmente se remove `gender` ou resume atributos
demograficos.

Artefato comparativo: `reports/feature_ablation/comparison.csv`.
"""
    output_path.write_text(report, encoding="utf-8")


def run_feature_ablation() -> pd.DataFrame:
    """Executa a bateria de ablação e registra todos os resultados."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    rows: list[dict[str, Any]] = []

    for spec in build_ablation_registry():
        summary, fold_metrics = evaluate_feature_set(spec, features, target)
        feature_count = _feature_count(spec, features, target)
        _log_ablation_run(spec, summary, fold_metrics, feature_count)
        rows.append(
            {
                "model": spec.name,
                "description": spec.description,
                "tags": ",".join(spec.tags),
                "drop_numeric": ",".join(spec.drop_numeric),
                "drop_categorical": ",".join(spec.drop_categorical),
                "add_numeric": ",".join(spec.add_numeric),
                "add_categorical": ",".join(spec.add_categorical),
                "final_feature_count": feature_count,
                **summary,
            }
        )

    comparison = pd.DataFrame(rows)
    reference = comparison.query("model == 'full_current'").iloc[0]
    comparison["delta_f1_mean"] = comparison["f1_mean"] - float(reference["f1_mean"])
    comparison["delta_optimal_f1_mean"] = comparison["optimal_f1_mean"] - float(
        reference["optimal_f1_mean"]
    )
    comparison["delta_pr_auc_mean"] = comparison["pr_auc_mean"] - float(reference["pr_auc_mean"])
    comparison["removed_final_features"] = (
        int(reference["final_feature_count"]) - comparison["final_feature_count"]
    )
    comparison["is_non_degrading_f1"] = comparison["delta_f1_mean"] >= 0
    comparison = comparison.sort_values(
        ["f1_mean", "optimal_f1_mean", "pr_auc_mean"],
        ascending=False,
    )
    output_dir = REPORTS_DIR / "feature_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    _write_report(comparison)

    logger.info(
        "ablacao_features_concluida",
        extra={
            "feature_sets": len(comparison),
            "best_model": comparison.iloc[0]["model"],
            "best_f1_mean": comparison.iloc[0]["f1_mean"],
        },
    )
    return comparison


def main() -> None:
    """Ponto de entrada do comando train-feature-ablation."""
    run_feature_ablation()


if __name__ == "__main__":
    main()
