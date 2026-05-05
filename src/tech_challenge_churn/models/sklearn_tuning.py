"""Busca controlada de F1 com Scikit-Learn e rastreamento MLflow."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tech_challenge_churn.config import DATA_PATH, MLRUNS_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from tech_challenge_churn.data.load import compute_file_hash, read_raw_data, split_features_target
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.evaluation.business import lift_at_top_fraction
from tech_challenge_churn.evaluation.metrics import (
    find_best_f1_threshold,
    probability_metrics,
    threshold_metrics,
)
from tech_challenge_churn.models.sklearn_optimization import (
    build_experiment_pipeline,
    evaluate_and_log_model,
)
from tech_challenge_churn.utils.logging import configure_logging, get_logger
from tech_challenge_churn.utils.seed import set_global_seed

logger = get_logger(__name__)
EXPERIMENT_NAME = "telco-churn-sklearn-tuning"


@dataclass(frozen=True)
class Candidate:
    """Representa uma tentativa de tuning rastreavel."""

    name: str
    family: str
    model: BaseEstimator
    params: dict[str, Any]


def _current_git_sha() -> str:
    """Obtem o commit atual para rastreabilidade."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _safe_params(params: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    """Normaliza parametros para logging no MLflow."""
    safe: dict[str, str | int | float | bool | None] = {}
    for key, value in params.items():
        if isinstance(value, str | int | float | bool) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)[:500]
    return safe


def _selector_step(selector_name: str, k: int | str, seed: int = RANDOM_SEED) -> Any:
    """Cria seletor de features opcional apos o pre-processador."""
    if selector_name == "passthrough":
        return "passthrough"
    if selector_name == "f_classif":
        return SelectKBest(score_func=f_classif, k=k)
    if selector_name == "mutual_info":
        score_func = partial(mutual_info_classif, random_state=seed)
        return SelectKBest(score_func=score_func, k=k)
    raise ValueError(f"Selector desconhecido: {selector_name}")


def _with_selector(
    classifier: BaseEstimator,
    selector_name: str,
    k: int | str,
) -> Pipeline:
    """Encapsula seletor e classificador como etapa final do pipeline completo."""
    return Pipeline(
        steps=[
            ("selector", _selector_step(selector_name, k)),
            ("classifier", classifier),
        ]
    )


def _candidate_name(prefix: str, index: int) -> str:
    """Cria nome curto e estavel para uma tentativa."""
    return f"{prefix}_{index:03d}"


def _hgb_candidates(seed: int = RANDOM_SEED, n_iter: int = 36) -> list[Candidate]:
    """Amostra configuracoes de HistGradientBoosting."""
    distributions = {
        "learning_rate": np.geomspace(0.02, 0.18, 12).round(5).tolist(),
        "max_iter": [150, 200, 250, 350, 500, 650],
        "max_leaf_nodes": [7, 15, 23, 31, 47, 63],
        "min_samples_leaf": [10, 15, 20, 30, 45, 60],
        "l2_regularization": [0.0, 0.01, 0.05, 0.1, 0.3, 0.7, 1.0],
        "max_depth": [None, 3, 4, 6, 8],
        "class_weight": [None, "balanced", {0: 1.0, 1: 1.3}, {0: 1.0, 1: 1.6}],
        "selector": ["passthrough", "f_classif", "mutual_info"],
        "k": [35, 50, 65, 80],
    }
    candidates: list[Candidate] = []
    sampler = ParameterSampler(distributions, n_iter=n_iter, random_state=seed)
    for index, params in enumerate(sampler, start=1):
        selector_name = str(params.pop("selector"))
        k = int(params.pop("k"))
        classifier = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            scoring="loss",
            random_state=seed,
            **params,
        )
        model = _with_selector(classifier, selector_name, k)
        candidates.append(
            Candidate(
                name=_candidate_name("hgb_tuned", index),
                family="hgb",
                model=model,
                params={"selector": selector_name, "k": k, **params},
            )
        )
    return candidates


def _tree_candidates(seed: int = RANDOM_SEED, n_iter: int = 24) -> list[Candidate]:
    """Amostra RandomForest e ExtraTrees."""
    distributions = {
        "model_type": ["random_forest", "extra_trees"],
        "n_estimators": [250, 400, 600],
        "max_depth": [None, 8, 12, 18, 25],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8, 12],
        "max_features": ["sqrt", "log2", 0.35, 0.5, 0.75],
        "class_weight": ["balanced", "balanced_subsample", {0: 1.0, 1: 1.4}],
        "selector": ["passthrough", "f_classif", "mutual_info"],
        "k": [40, 60, 80],
    }
    candidates: list[Candidate] = []
    sampler = ParameterSampler(distributions, n_iter=n_iter, random_state=seed + 11)
    for index, params in enumerate(sampler, start=1):
        model_type = str(params.pop("model_type"))
        selector_name = str(params.pop("selector"))
        k = int(params.pop("k"))
        classifier_class = (
            RandomForestClassifier if model_type == "random_forest" else ExtraTreesClassifier
        )
        classifier = classifier_class(
            random_state=seed,
            n_jobs=-1,
            bootstrap=model_type == "random_forest",
            **params,
        )
        model = _with_selector(classifier, selector_name, k)
        candidates.append(
            Candidate(
                name=_candidate_name(model_type, index),
                family=model_type,
                model=model,
                params={"selector": selector_name, "k": k, "model_type": model_type, **params},
            )
        )
    return candidates


def _stacking_candidates(seed: int = RANDOM_SEED, n_iter: int = 14) -> list[Candidate]:
    """Amostra stackings diversificados sem sair do Scikit-Learn."""
    distributions = {
        "hgb_learning_rate": [0.03, 0.05, 0.07, 0.1],
        "hgb_max_leaf_nodes": [15, 23, 31],
        "hgb_min_samples_leaf": [15, 25, 40],
        "hgb_l2_regularization": [0.0, 0.05, 0.1, 0.3],
        "tree_type": ["random_forest", "extra_trees"],
        "tree_min_samples_leaf": [2, 4, 8],
        "tree_max_depth": [None, 10, 16],
        "meta_c": [0.15, 0.35, 0.7, 1.0],
        "selector": ["passthrough", "f_classif"],
        "k": [50, 65, 80],
    }
    candidates: list[Candidate] = []
    sampler = ParameterSampler(distributions, n_iter=n_iter, random_state=seed + 23)
    for index, params in enumerate(sampler, start=1):
        selector_name = str(params["selector"])
        k = int(params["k"])
        hgb = HistGradientBoostingClassifier(
            learning_rate=float(params["hgb_learning_rate"]),
            max_iter=300,
            max_leaf_nodes=int(params["hgb_max_leaf_nodes"]),
            min_samples_leaf=int(params["hgb_min_samples_leaf"]),
            l2_regularization=float(params["hgb_l2_regularization"]),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            class_weight="balanced",
            random_state=seed,
        )
        tree_params = {
            "n_estimators": 400,
            "max_depth": params["tree_max_depth"],
            "min_samples_leaf": int(params["tree_min_samples_leaf"]),
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
        }
        if params["tree_type"] == "random_forest":
            tree = RandomForestClassifier(**tree_params)
        else:
            tree = ExtraTreesClassifier(**tree_params)
        logistic = LogisticRegression(
            class_weight="balanced",
            C=float(params["meta_c"]),
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        )
        stack = StackingClassifier(
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
                ("hgb", hgb),
                ("tree", tree),
            ],
            final_estimator=logistic,
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
        )
        model = _with_selector(stack, selector_name, k)
        candidates.append(
            Candidate(
                name=_candidate_name("stack_tuned", index),
                family="stacking",
                model=model,
                params=dict(params),
            )
        )
    return candidates


def _svc_candidates(seed: int = RANDOM_SEED) -> list[Candidate]:
    """Cria poucas tentativas de SVC RBF calibrado, pois o custo e maior."""
    candidates: list[Candidate] = []
    configs = [
        {"C": 0.5, "gamma": "scale", "selector": "f_classif", "k": 50},
        {"C": 1.0, "gamma": "scale", "selector": "f_classif", "k": 65},
        {"C": 2.0, "gamma": 0.01, "selector": "f_classif", "k": 65},
        {"C": 1.0, "gamma": 0.03, "selector": "mutual_info", "k": 50},
    ]
    for index, params in enumerate(configs, start=1):
        svc = SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight="balanced",
        )
        classifier = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=3)
        model = _with_selector(classifier, str(params["selector"]), int(params["k"]))
        candidates.append(
            Candidate(
                name=_candidate_name("svc_rbf_calibrated", index),
                family="svc",
                model=model,
                params={"seed": seed, **params},
            )
        )
    return candidates


def build_candidate_registry(seed: int = RANDOM_SEED) -> list[Candidate]:
    """Agrupa todas as tentativas de busca curta."""
    return [
        *_hgb_candidates(seed),
        *_tree_candidates(seed),
        *_stacking_candidates(seed),
        *_svc_candidates(seed),
    ]


def _evaluate_trial(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = RANDOM_SEED,
) -> dict[str, float | str]:
    """Avalia uma tentativa em holdout de triagem e registra no MLflow."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.25,
        stratify=target,
        random_state=seed,
    )
    metrics: dict[str, float | str] = {
        "model": candidate.name,
        "family": candidate.family,
    }
    with mlflow.start_run(run_name=f"search_{candidate.name}"):
        mlflow.set_tags(
            {
                "stage": "tuning_search",
                "family": candidate.family,
                "dataset_path": str(DATA_PATH),
                "dataset_sha256": compute_file_hash(DATA_PATH),
                "git_sha": _current_git_sha(),
                "model_name": candidate.name,
            }
        )
        mlflow.log_params(_safe_params(candidate.params))
        mlflow.log_param("seed", seed)
        mlflow.log_param("evaluation_protocol", "single_stratified_holdout_for_screening")

        try:
            pipeline = build_experiment_pipeline(clone(candidate.model))
            pipeline.fit(x_train, y_train)
            y_proba = pipeline.predict_proba(x_valid)[:, 1]
            threshold, best_f1 = find_best_f1_threshold(y_valid.to_numpy(), y_proba)
            probability = probability_metrics(y_valid.to_numpy(), y_proba)
            threshold_05 = threshold_metrics(y_valid.to_numpy(), y_proba, threshold=0.5)
            optimal = threshold_metrics(y_valid.to_numpy(), y_proba, threshold=threshold)
            metrics.update({f"valid_{key}": value for key, value in probability.items()})
            metrics.update({f"valid_{key}": value for key, value in threshold_05.items()})
            metrics.update({f"valid_optimal_{key}": value for key, value in optimal.items()})
            metrics["valid_threshold_f1"] = threshold
            metrics["valid_f1_best_internal"] = best_f1
            metrics["valid_lift_at_top_20pct"] = lift_at_top_fraction(
                y_valid.to_numpy(),
                y_proba,
            )
            y_pred = (y_proba >= threshold).astype(int)
            report = classification_report(y_valid, y_pred, zero_division=0)
            artifact_dir = REPORTS_DIR / "sklearn_tuning" / "search" / candidate.name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "classification_report.txt").write_text(report, encoding="utf-8")
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
            for metric_name, value in metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(metric_name, float(value))
            mlflow.set_tag("status", "success")
        except Exception as error:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error_type", type(error).__name__)
            mlflow.set_tag("error_message", str(error)[:500])
            metrics["status"] = "failed"
            metrics["error"] = str(error)
            logger.warning(
                "tentativa_tuning_falhou",
                extra={"model_name": candidate.name, "error": str(error)},
            )
            return metrics

    metrics["status"] = "success"
    logger.info(
        "tentativa_tuning_registrada",
        extra={
            "model_name": candidate.name,
            "family": candidate.family,
            "valid_optimal_f1": metrics.get("valid_optimal_f1"),
            "valid_pr_auc": metrics.get("valid_pr_auc"),
        },
    )
    return metrics


def _select_finalists(
    candidates: list[Candidate],
    search_results: pd.DataFrame,
    max_finalists: int = 8,
) -> list[Candidate]:
    """Seleciona melhores tentativas e garante diversidade por familia."""
    successful = search_results.query("status == 'success'").copy()
    successful = successful.sort_values(
        ["valid_optimal_f1", "valid_pr_auc", "valid_roc_auc"],
        ascending=False,
    )
    selected_names = successful.head(max_finalists // 2)["model"].tolist()
    for _, row in successful.groupby("family", as_index=False).head(1).iterrows():
        selected_names.append(str(row["model"]))
    selected_names = list(dict.fromkeys(selected_names))[:max_finalists]
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    return [candidate_by_name[name] for name in selected_names if name in candidate_by_name]


def _write_report(search_results: pd.DataFrame, final_results: pd.DataFrame) -> None:
    """Documenta a busca de F1 e o resultado final."""
    output_path = Path("docs") / "sklearn_tuning_report.md"
    best_search = search_results.query("status == 'success'").sort_values(
        "valid_optimal_f1",
        ascending=False,
    ).iloc[0]
    best_final = final_results.sort_values("optimal_f1_mean", ascending=False).iloc[0]
    report = f"""# Tuning Avancado Scikit-Learn para F1

## Protocolo

- Triagem com `ParameterSampler` e holdout estratificado apenas para ordenar tentativas.
- Cada tentativa foi registrada no MLflow em `{EXPERIMENT_NAME}` com parametros e metricas.
- Finalistas foram reavaliados com validacao cruzada estratificada e threshold escolhido em split
  interno, reduzindo risco de overfitting.
- Ferramentas restritas ao escopo técnico do projeto: Scikit-Learn e MLflow.

## Melhor Tentativa de Triagem

- Modelo: `{best_search["model"]}`.
- Familia: `{best_search["family"]}`.
- F1 validacao com threshold interno: `{float(best_search["valid_optimal_f1"]):.4f}`.
- PR-AUC validacao: `{float(best_search["valid_pr_auc"]):.4f}`.

## Melhor Resultado Reavaliado

- Modelo: `{best_final["model"]}`.
- AUC-ROC media: `{best_final["roc_auc_mean"]:.4f}`.
- PR-AUC media: `{best_final["pr_auc_mean"]:.4f}`.
- F1 medio em threshold 0,5: `{best_final["f1_mean"]:.4f}`.
- F1 medio com threshold interno: `{best_final["optimal_f1_mean"]:.4f}`.

## Leitura Critica

A triagem pode superestimar F1 porque muitas tentativas olham o mesmo holdout de desenvolvimento.
Por isso, o numero que deve entrar na comparacao final e o F1 medio dos finalistas reavaliados por
CV. A busca testou HGB, RandomForest, ExtraTrees, stacking e SVC calibrado, com e sem selecao de
features (`SelectKBest` por ANOVA F ou informacao mutua).

Artefatos:

- `reports/sklearn_tuning/search_results.csv`
- `reports/sklearn_tuning/final_cv_comparison.csv`
"""
    output_path.write_text(report, encoding="utf-8")


def run_sklearn_tuning() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa busca por F1 e reavalia os finalistas."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = validate_telco_schema(read_raw_data(DATA_PATH))
    features, target = split_features_target(data)
    candidates = build_candidate_registry(RANDOM_SEED)

    search_rows = [
        _evaluate_trial(candidate, features, target, RANDOM_SEED) for candidate in candidates
    ]
    output_dir = REPORTS_DIR / "sklearn_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    search_results = pd.DataFrame(search_rows)
    search_results.to_csv(output_dir / "search_results.csv", index=False)

    finalists = _select_finalists(candidates, search_results)
    final_rows: list[dict[str, float | str]] = []
    for candidate in finalists:
        summary = evaluate_and_log_model(
            model_name=f"tuned_final_{candidate.name}",
            model=candidate.model,
            features=features,
            target=target,
        )
        final_rows.append({"model": candidate.name, "family": candidate.family, **summary})

    final_results = pd.DataFrame(final_rows).sort_values(
        ["optimal_f1_mean", "pr_auc_mean", "roc_auc_mean"],
        ascending=False,
    )
    final_results.to_csv(output_dir / "final_cv_comparison.csv", index=False)
    if not final_results.empty:
        best_name = str(final_results.iloc[0]["model"])
        best_candidate = {candidate.name: candidate for candidate in finalists}[best_name]
        best_model = build_experiment_pipeline(clone(best_candidate.model))
        best_model.fit(features, target)
        model_dir = MODELS_DIR / "sklearn_tuning"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_dir / f"{best_name}.joblib")
    _write_report(search_results, final_results)

    logger.info(
        "tuning_sklearn_concluido",
        extra={
            "tentativas": len(search_results),
            "finalistas": len(final_results),
            "best_model": None if final_results.empty else final_results.iloc[0]["model"],
            "best_optimal_f1_mean": None
            if final_results.empty
            else final_results.iloc[0]["optimal_f1_mean"],
        },
    )
    return search_results, final_results


def main() -> None:
    """Ponto de entrada do comando train-sklearn-tuning."""
    run_sklearn_tuning()


if __name__ == "__main__":
    main()
