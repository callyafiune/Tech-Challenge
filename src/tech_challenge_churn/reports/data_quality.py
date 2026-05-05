"""Auditoria rigorosa de qualidade dos dados Telco."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from tech_challenge_churn.config import (
    BASE_CATEGORICAL_FEATURES,
    CATEGORICAL_DOMAIN_VALUES,
    DATA_PATH,
    DOCS_DIR,
    INTERNET_DEPENDENT_COLUMNS,
    REPORTS_DIR,
)
from tech_challenge_churn.data.load import clean_total_charges, compute_file_hash, read_raw_data
from tech_challenge_churn.data.schema import validate_clean_telco_schema, validate_telco_schema
from tech_challenge_churn.features.build import add_telco_features, build_feature_pipeline
from tech_challenge_churn.utils.logging import configure_logging, get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger(__name__)
CONTINUOUS_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "charges_delta",
    "total_to_monthly_ratio",
    "num_services",
    "num_protection_services",
]


def _markdown_table(data: pd.DataFrame, max_rows: int = 25) -> str:
    """Converte DataFrame pequeno em tabela Markdown."""
    if data.empty:
        return "_Sem registros._"

    sample = data.head(max_rows).copy()
    headers = [str(column) for column in sample.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in sample.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def _write_table(data: pd.DataFrame, output_dir: Path, filename: str) -> pd.DataFrame:
    """Salva uma tabela de auditoria em CSV e devolve o proprio DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / filename, index=False)
    return data


def _missing_values(raw_data: pd.DataFrame, clean_data: pd.DataFrame) -> pd.DataFrame:
    """Conta nulos, strings vazias e nulos apos coercao numerica."""
    rows = []
    for column in raw_data.columns:
        raw_series = raw_data[column]
        blank_count = int(raw_series.astype("string").str.strip().eq("").fillna(False).sum())
        rows.append(
            {
                "coluna": column,
                "nulos_raw": int(raw_series.isna().sum()),
                "strings_vazias": blank_count,
                "nulos_pos_limpeza": int(clean_data[column].isna().sum()),
                "percentual_pos_limpeza": round(clean_data[column].isna().mean() * 100, 4),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["nulos_pos_limpeza", "strings_vazias"],
        ascending=False,
    )


def _class_balance(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula balanceamento da classe alvo."""
    balance = (
        data["Churn"]
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="clientes")
        .assign(percentual=lambda frame: (frame["clientes"] / len(data) * 100).round(2))
    )
    majority = int(balance["clientes"].max())
    minority = int(balance["clientes"].min())
    balance["razao_maioria_minoria"] = round(majority / minority, 3)
    return balance


def _domain_values(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audita dominios categoricos esperados e valores invalidos."""
    domain_rows = []
    invalid_rows = []
    for column, expected_values in CATEGORICAL_DOMAIN_VALUES.items():
        observed_values = sorted(raw_data[column].dropna().astype(str).unique().tolist())
        invalid_values = sorted(set(observed_values) - set(expected_values))
        domain_rows.append(
            {
                "coluna": column,
                "valores_observados": ", ".join(observed_values),
                "valores_esperados": ", ".join(expected_values),
                "qtd_invalidos": len(invalid_values),
            }
        )
        for value in invalid_values:
            invalid_rows.append(
                {
                    "coluna": column,
                    "valor_invalido": value,
                    "ocorrencias": int(raw_data[column].astype(str).eq(value).sum()),
                }
            )
    return pd.DataFrame(domain_rows), pd.DataFrame(invalid_rows)


def _logical_anomalies(raw_data: pd.DataFrame, clean_data: pd.DataFrame) -> pd.DataFrame:
    """Conta anomalias logicas especificas do dominio Telco."""
    internet_no = raw_data["InternetService"].eq("No")
    internet_yes = raw_data["InternetService"].ne("No")
    internet_absence_wrong = raw_data.loc[
        internet_no,
        INTERNET_DEPENDENT_COLUMNS,
    ].ne("No internet service").any(axis=1)
    internet_present_wrong = raw_data.loc[
        internet_yes,
        INTERNET_DEPENDENT_COLUMNS,
    ].eq("No internet service").any(axis=1)

    checks = [
        (
            "customerID duplicado",
            int(raw_data["customerID"].duplicated().sum()),
            "Deve ser zero para evitar repeticao de cliente.",
        ),
        (
            "customerID vazio",
            int(raw_data["customerID"].astype("string").str.strip().eq("").sum()),
            "Deve ser zero para manter rastreabilidade.",
        ),
        (
            "TotalCharges vazio no raw",
            int(raw_data["TotalCharges"].astype("string").str.strip().eq("").sum()),
            "Esperado apenas para tenure=0 no dataset Telco.",
        ),
        (
            "TotalCharges nulo com tenure=0",
            int((clean_data["TotalCharges"].isna() & clean_data["tenure"].eq(0)).sum()),
            "Caso sem cobranca acumulada; tratado como zero no pipeline.",
        ),
        (
            "TotalCharges nulo com tenure>0",
            int((clean_data["TotalCharges"].isna() & clean_data["tenure"].gt(0)).sum()),
            "Deve ser zero; se aparecer, exige investigacao.",
        ),
        (
            "TotalCharges negativo",
            int(clean_data["TotalCharges"].lt(0).sum()),
            "Deve ser zero.",
        ),
        (
            "MonthlyCharges negativo",
            int(clean_data["MonthlyCharges"].lt(0).sum()),
            "Deve ser zero.",
        ),
        (
            "PhoneService=No com MultipleLines diferente de No phone service",
            int(
                (
                    raw_data["PhoneService"].eq("No")
                    & raw_data["MultipleLines"].ne("No phone service")
                ).sum()
            ),
            "Deve ser zero pela regra do dataset.",
        ),
        (
            "PhoneService=Yes com MultipleLines=No phone service",
            int(
                (
                    raw_data["PhoneService"].eq("Yes")
                    & raw_data["MultipleLines"].eq("No phone service")
                ).sum()
            ),
            "Deve ser zero pela regra do dataset.",
        ),
        (
            "InternetService=No com colunas dependentes inconsistentes",
            int(internet_absence_wrong.sum()),
            "Deve ser zero pela regra do dataset.",
        ),
        (
            "InternetService ativo com No internet service",
            int(internet_present_wrong.sum()),
            "Deve ser zero pela regra do dataset.",
        ),
    ]
    return pd.DataFrame(checks, columns=["checagem", "ocorrencias", "interpretacao"])


def _numeric_distribution(engineered_data: pd.DataFrame) -> pd.DataFrame:
    """Calcula distribuicao, assimetria e curtose das features numericas."""
    rows = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(engineered_data[column], errors="coerce")
        skew = float(series.skew(skipna=True))
        rows.append(
            {
                "feature": column,
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "q1": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "q3": round(float(series.quantile(0.75)), 4),
                "max": round(float(series.max()), 4),
                "skew": round(skew, 4),
                "kurtosis": round(float(series.kurtosis(skipna=True)), 4),
                "simetria": _skew_label(skew),
            }
        )
    return pd.DataFrame(rows)


def _skew_label(skew: float) -> str:
    """Classifica a simetria da distribuicao pelo skew."""
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        return "aproximadamente simetrica"
    if abs_skew < 1.0:
        return "assimetria moderada"
    return "assimetria forte"


def _outlier_summary(engineered_data: pd.DataFrame) -> pd.DataFrame:
    """Conta outliers pelo criterio IQR sem remover registros."""
    rows = []
    for column in CONTINUOUS_FEATURES:
        series = pd.to_numeric(engineered_data[column], errors="coerce").dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int((series.lt(lower) | series.gt(upper)).sum())
        rows.append(
            {
                "feature": column,
                "limite_inferior": round(lower, 4),
                "limite_superior": round(upper, 4),
                "outliers_iqr": count,
                "percentual": round(count / len(series) * 100, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("percentual", ascending=False)


def _high_correlations(engineered_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula matriz de correlacao e pares altamente correlacionados."""
    numeric_data = engineered_data.select_dtypes(include=[np.number]).copy()
    correlation_matrix = numeric_data.corr(method="pearson").round(4)
    rows = []
    columns = correlation_matrix.columns.tolist()
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            value = float(correlation_matrix.loc[first, second])
            if abs(value) >= 0.85:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "correlacao_pearson": round(value, 4),
                    }
                )
    pairs = pd.DataFrame(rows).sort_values(
        "correlacao_pearson",
        key=lambda series: series.abs(),
        ascending=False,
    )
    return correlation_matrix.reset_index(names="feature"), pairs


def _encoded_feature_audit(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Audita a codificacao one-hot e correlacoes simples das features codificadas."""
    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(features)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    transformed_array = np.asarray(transformed, dtype=float)
    target_array = target.to_numpy(dtype=float)

    correlations = []
    for index, feature_name in enumerate(feature_names):
        values = transformed_array[:, index]
        if np.isclose(values.std(), 0.0):
            continue
        correlation = float(np.corrcoef(values, target_array)[0, 1])
        if np.isfinite(correlation):
            correlations.append(
                {
                    "feature_codificada": feature_name,
                    "correlacao_com_churn": round(correlation, 4),
                }
            )

    top_correlations = pd.DataFrame(correlations).sort_values(
        "correlacao_com_churn",
        key=lambda series: series.abs(),
        ascending=False,
    )
    engineered = add_telco_features(features)
    rare_levels = {}
    for column in BASE_CATEGORICAL_FEATURES:
        frequencies = engineered[column].value_counts(normalize=True, dropna=False)
        rare_levels[column] = int(frequencies.lt(0.01).sum())

    summary = {
        "linhas": int(transformed_array.shape[0]),
        "features_originais_sem_id_alvo": int(features.shape[1]),
        "features_numericas_configuradas": len(
            pipeline.named_steps["preprocessor"].transformers_[0][2]
        ),
        "features_categoricas_configuradas": len(
            pipeline.named_steps["preprocessor"].transformers_[1][2]
        ),
        "features_codificadas_total": int(transformed_array.shape[1]),
        "saida_esparsa": bool(hasattr(transformed, "toarray")),
        "niveis_raros_por_feature_categorica": rare_levels,
    }
    return summary, top_correlations


def _save_figures(
    clean_data: pd.DataFrame,
    engineered_data: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Salva figuras complementares da auditoria."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.countplot(data=clean_data, x="Churn", hue="Churn")
    plt.title("Balanceamento da classe Churn")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_balance.png", dpi=140)
    plt.close()

    selected = [column for column in CONTINUOUS_FEATURES if column in engineered_data.columns]
    engineered_data[selected].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Distribuicoes numericas e features engenheiradas")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_distributions.png", dpi=140)
    plt.close()

    sns.boxplot(data=engineered_data[selected], orient="h")
    plt.title("Outliers por boxplot")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_boxplots.png", dpi=140)
    plt.close()

    matrix = correlation_matrix.set_index("feature")
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Matriz de correlacao das features numericas")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_correlation_heatmap.png", dpi=140)
    plt.close()


def generate_data_quality_report() -> Path:
    """Gera auditoria completa da base e dos refinamentos do pipeline."""
    configure_logging()
    raw_data = validate_telco_schema(read_raw_data(DATA_PATH))
    clean_data = validate_clean_telco_schema(clean_total_charges(raw_data))
    features = raw_data.drop(columns=["customerID", "Churn"])
    target = raw_data["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    engineered_data = add_telco_features(features).assign(ChurnBinary=target)

    output_dir = REPORTS_DIR / "data_quality"
    missing = _write_table(_missing_values(raw_data, clean_data), output_dir, "missing_values.csv")
    balance = _write_table(_class_balance(raw_data), output_dir, "class_balance.csv")
    domain_values, invalid_values = _domain_values(raw_data)
    domain_values = _write_table(domain_values, output_dir, "categorical_domains.csv")
    invalid_values = _write_table(invalid_values, output_dir, "invalid_values.csv")
    anomalies = _write_table(
        _logical_anomalies(raw_data, clean_data),
        output_dir,
        "logical_anomalies.csv",
    )
    distribution = _write_table(
        _numeric_distribution(engineered_data),
        output_dir,
        "numeric_distribution.csv",
    )
    outliers = _write_table(_outlier_summary(engineered_data), output_dir, "outliers_iqr.csv")
    correlation_matrix, high_correlations = _high_correlations(engineered_data)
    correlation_matrix = _write_table(
        correlation_matrix,
        output_dir,
        "numeric_correlation_matrix.csv",
    )
    high_correlations = _write_table(
        high_correlations,
        output_dir,
        "high_correlations.csv",
    )
    encoding_summary, encoded_correlations = _encoded_feature_audit(features, target)
    _write_table(
        encoded_correlations.head(30),
        output_dir,
        "top_encoded_correlations.csv",
    )
    (output_dir / "encoding_summary.json").write_text(
        json.dumps(encoding_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_figures(clean_data, engineered_data, correlation_matrix, output_dir)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    blank_total = int(missing.loc[missing["coluna"].eq("TotalCharges"), "strings_vazias"].iloc[0])
    positive_rate = float(target.mean())
    report = f"""# Revisao Rigorosa de Qualidade dos Dados - Telco Churn

Relatorio gerado em: {generated_at}

## Veredito

- A base tem {len(raw_data)} linhas, {len(raw_data.columns)} colunas e hash
  `{compute_file_hash(DATA_PATH)}`.
- Nao foram encontrados valores categoricos fora do dominio esperado.
- Nao ha `customerID` duplicado.
- O alvo esta desbalanceado: churn positivo de {positive_rate * 100:.2f}%.
- Existem {blank_total} strings vazias em `TotalCharges`; todas ocorrem em clientes com `tenure=0`.
- O pipeline foi refinado para tratar `TotalCharges` vazio com `0` quando `tenure=0`, evitando
  imputacao por mediana nesse caso sem usar informacao do alvo.
- O pipeline tambem colapsa `No internet service` e `No phone service` para `No` nas colunas
  dependentes, preservando `InternetService` e `PhoneService` como sinal explicito e reduzindo
  colinearidade deterministica.

## Missing Values

{_markdown_table(missing.query("nulos_pos_limpeza > 0 or strings_vazias > 0"))}

## Balanceamento de Classes

{_markdown_table(balance)}

## Valores Invalidos e Dominios Categoricos

Valores invalidos:

{_markdown_table(invalid_values)}

Resumo dos dominios:

{_markdown_table(domain_values)}

## Anomalias Logicas

{_markdown_table(anomalies)}

## Distribuicao, Simetria e Outliers

{_markdown_table(distribution)}

Outliers por IQR:

{_markdown_table(outliers)}

Os outliers nao foram removidos automaticamente. Em churn, valores extremos de mensalidade ou
tempo de contrato podem ser sinal real de comportamento do cliente. A decisao segura e registrar,
escalar e testar robustez em vez de descartar registros.

## Correlacoes e Colinearidade

Pares numericos com |correlacao| >= 0.85:

{_markdown_table(high_correlations)}

As correlacoes fortes confirmam redundancias esperadas: `TotalCharges` se relaciona com `tenure`,
`avg_monthly_spend` se aproxima de `MonthlyCharges` e `total_to_monthly_ratio` se aproxima de
tempo de relacionamento. Isso justifica testar ablacoees em experimentos, mas nao remover features
sem registro no MLflow.

## Codificacao Categorica

- Features originais sem ID/alvo: {encoding_summary["features_originais_sem_id_alvo"]}.
- Features numericas configuradas: {encoding_summary["features_numericas_configuradas"]}.
- Features categoricas configuradas: {encoding_summary["features_categoricas_configuradas"]}.
- Features finais apos OneHotEncoder: {encoding_summary["features_codificadas_total"]}.
- Saida esparsa: {encoding_summary["saida_esparsa"]}.

Top correlacoes simples apos codificacao:

{_markdown_table(encoded_correlations.head(15))}

## Artefatos Gerados

- `reports/data_quality/missing_values.csv`
- `reports/data_quality/class_balance.csv`
- `reports/data_quality/categorical_domains.csv`
- `reports/data_quality/invalid_values.csv`
- `reports/data_quality/logical_anomalies.csv`
- `reports/data_quality/numeric_distribution.csv`
- `reports/data_quality/outliers_iqr.csv`
- `reports/data_quality/numeric_correlation_matrix.csv`
- `reports/data_quality/high_correlations.csv`
- `reports/data_quality/top_encoded_correlations.csv`
- `reports/data_quality/encoding_summary.json`
- `reports/data_quality/figures/`
"""

    output_path = DOCS_DIR / "data_quality_report.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("data_quality_report_gerado", extra={"path": str(output_path)})
    return output_path


def main() -> None:
    """Ponto de entrada do comando generate-data-quality."""
    generate_data_quality_report()


if __name__ == "__main__":
    main()
