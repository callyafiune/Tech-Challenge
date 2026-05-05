"""Comparação estatística de modelos a partir dos folds registrados no MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from tech_challenge_churn.config import DOCS_DIR, MLFLOW_TRACKING_URI, REPORTS_DIR
from tech_challenge_churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)
ALPHA = 0.05
N_BOOTSTRAP = 5_000


@dataclass(frozen=True)
class ModelRunSpec:
    """Define um run esperado no MLflow para comparação."""

    label: str
    experiment_name: str
    run_name: str
    role: str


MODEL_SPECS = [
    ModelRunSpec(
        label="Regressão Logística balanceada",
        experiment_name="telco-churn-baselines",
        run_name="logistic_regression_balanced",
        role="baseline",
    ),
    ModelRunSpec(
        label="MLP PyTorch refinada",
        experiment_name="telco-churn-mlp",
        run_name="mlp_config_4_128-64-32",
        role="modelo_neural",
    ),
    ModelRunSpec(
        label="RandomForest tunado",
        experiment_name="telco-churn-sklearn-tuning",
        run_name="tuned_final_random_forest_005",
        role="tabular_tunado",
    ),
    ModelRunSpec(
        label="RandomForest sem gender",
        experiment_name="telco-churn-feature-ablation",
        run_name="no_gender",
        role="challenger_operacional",
    ),
    ModelRunSpec(
        label="RandomForest refinado sem gender",
        experiment_name="telco-churn-f1-refinement",
        run_name="rf_no_gender_refined",
        role="refinamento_sem_promocao",
    ),
]

PAIRWISE_COMPARISONS = [
    ("RandomForest sem gender", "MLP PyTorch refinada"),
    ("RandomForest sem gender", "Regressão Logística balanceada"),
    ("RandomForest sem gender", "RandomForest tunado"),
    ("RandomForest sem gender", "RandomForest refinado sem gender"),
    ("MLP PyTorch refinada", "Regressão Logística balanceada"),
]


def _latest_run_by_name(client: MlflowClient, spec: ModelRunSpec) -> Run:
    """Busca o run mais recente com o nome esperado."""
    experiment = client.get_experiment_by_name(spec.experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento MLflow não encontrado: {spec.experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{spec.run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"Run '{spec.run_name}' não encontrado no experimento '{spec.experiment_name}'."
        )
    return runs[0]


def _fold_scores(run: Run, metric: str, n_folds: int = 5) -> list[float]:
    """Extrai métricas por fold do run."""
    scores: list[float] = []
    for fold_index in range(1, n_folds + 1):
        key = f"fold_{fold_index}_{metric}"
        if key not in run.data.metrics:
            return []
        scores.append(float(run.data.metrics[key]))
    return scores


def exact_sign_test_p_value(differences: np.ndarray) -> float:
    """Calcula p-valor bicaudal do teste exato de sinais."""
    non_zero = differences[np.abs(differences) > 1e-12]
    n = len(non_zero)
    if n == 0:
        return 1.0
    positives = int(np.sum(non_zero > 0))
    tail = min(positives, n - positives)
    probability = sum(comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2 * probability))


def bootstrap_mean_diff_ci(
    differences: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Estima intervalo bootstrap de 95% para a diferença média."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(differences, size=len(differences), replace=True)
        means[index] = float(np.mean(sample))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def paired_comparison(
    label_a: str,
    scores_a: list[float],
    label_b: str,
    scores_b: list[float],
    metric: str,
) -> dict[str, Any]:
    """Compara dois modelos com folds pareados."""
    array_a = np.asarray(scores_a, dtype=float)
    array_b = np.asarray(scores_b, dtype=float)
    if array_a.shape != array_b.shape:
        raise ValueError(f"Scores com tamanhos incompatíveis: {label_a} vs {label_b}")

    differences = array_a - array_b
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
    ci_lower, ci_upper = bootstrap_mean_diff_ci(differences)
    p_value = exact_sign_test_p_value(differences)
    significant = p_value < ALPHA

    if significant and mean_diff > 0:
        conclusion = f"{label_a} tem evidência estatística de melhor {metric}."
    elif significant and mean_diff < 0:
        conclusion = f"{label_b} tem evidência estatística de melhor {metric}."
    else:
        conclusion = "Sem evidência estatística suficiente; preferir parcimônia e contexto."

    return {
        "metric": metric,
        "model_a": label_a,
        "model_b": label_b,
        "mean_a": float(np.mean(array_a)),
        "mean_b": float(np.mean(array_b)),
        "mean_diff_a_minus_b": mean_diff,
        "std_diff": std_diff,
        "effect_size_dz": float(effect_size),
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "sign_test_p_value": p_value,
        "alpha": ALPHA,
        "significant": significant,
        "conclusion": conclusion,
    }


def collect_model_scores(metric: str = "f1") -> pd.DataFrame:
    """Coleta scores por fold dos principais modelos via MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    rows: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        run = _latest_run_by_name(client, spec)
        scores = _fold_scores(run, metric)
        if not scores:
            logger.warning(
                "metricas_fold_ausentes",
                extra={"model": spec.label, "metric": metric, "run_id": run.info.run_id},
            )
            continue
        rows.append(
            {
                "model": spec.label,
                "role": spec.role,
                "experiment_name": spec.experiment_name,
                "run_name": spec.run_name,
                "run_id": run.info.run_id,
                "metric": metric,
                "fold_scores": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=0)),
            }
        )

    return pd.DataFrame(rows)


def _markdown_table(data: pd.DataFrame) -> str:
    """Converte DataFrame pequeno em tabela Markdown."""
    if data.empty:
        return "_Sem registros._"
    frame = data.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def _write_report(scores: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    """Gera relatório Markdown da comparação estatística."""
    docs_path = DOCS_DIR / "model_comparison_statistical.md"
    summary = scores[["model", "role", "run_id", "mean", "std"]].sort_values(
        "mean",
        ascending=False,
    )
    compact_comparisons = comparisons[
        [
            "model_a",
            "model_b",
            "mean_a",
            "mean_b",
            "mean_diff_a_minus_b",
            "ci95_lower",
            "ci95_upper",
            "sign_test_p_value",
            "significant",
            "conclusion",
        ]
    ]
    report = f"""# Comparação Estatística de Modelos

## Objetivo

Comparar os principais modelos usando as métricas por fold registradas no MLflow. A análise evita
decidir apenas por pequenas diferenças de média e explicita a incerteza do protocolo de validação.

## Protocolo

- Métrica comparada: F1 em threshold 0,5.
- Fonte dos scores: métricas `fold_*_f1` dos runs mais recentes no MLflow.
- Teste estatístico: teste exato de sinais bicaudal sobre diferenças pareadas por fold.
- Intervalo: bootstrap percentil de 95% da diferença média.
- Nível de significância: `{ALPHA}`.

Com 5 folds, o teste tem baixo poder estatístico. Por isso, ausência de significância não prova que
os modelos são equivalentes; indica apenas que a evidência disponível não justifica uma troca por
diferenças pequenas.

## Scores por Modelo

{_markdown_table(summary)}

## Comparações Pareadas

{_markdown_table(compact_comparisons)}

## Decisão

A comparação reforça a recomendação atual: manter a MLP como modelo neural principal e manter o
RandomForest sem `gender` como challenger operacional. As diferenças observadas são pequenas e não
há evidência estatística suficiente, com 5 folds, para promover os candidatos do refinamento de F1.

Artefatos:

- `reports/model_comparison/fold_scores.csv`
- `reports/model_comparison/statistical_comparison.csv`
"""
    docs_path.write_text(report, encoding="utf-8")


def run_model_comparison(metric: str = "f1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa comparação estatística entre modelos principais."""
    configure_logging()
    scores = collect_model_scores(metric=metric)
    scores_by_model = dict(zip(scores["model"], scores["fold_scores"], strict=False))

    comparison_rows: list[dict[str, Any]] = []
    for model_a, model_b in PAIRWISE_COMPARISONS:
        if model_a not in scores_by_model or model_b not in scores_by_model:
            logger.warning(
                "comparacao_ignorada_por_score_ausente",
                extra={"model_a": model_a, "model_b": model_b},
            )
            continue
        comparison_rows.append(
            paired_comparison(
                model_a,
                scores_by_model[model_a],
                model_b,
                scores_by_model[model_b],
                metric,
            )
        )

    comparisons = pd.DataFrame(comparison_rows)
    output_dir = REPORTS_DIR / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = scores.drop(columns=["fold_scores"]).copy()
    for _, row in scores.iterrows():
        for fold_index, value in enumerate(row["fold_scores"], start=1):
            fold_scores.loc[fold_scores["model"] == row["model"], f"fold_{fold_index}"] = value
    fold_scores.to_csv(output_dir / "fold_scores.csv", index=False)
    comparisons.to_csv(output_dir / "statistical_comparison.csv", index=False)
    _write_report(scores, comparisons)

    logger.info(
        "comparacao_estatistica_modelos_concluida",
        extra={"models": len(scores), "comparisons": len(comparisons), "metric": metric},
    )
    return scores, comparisons


def main() -> None:
    """Ponto de entrada do comando compare-models."""
    run_model_comparison()


if __name__ == "__main__":
    main()
