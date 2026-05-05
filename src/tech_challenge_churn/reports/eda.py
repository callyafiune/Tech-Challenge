"""Geração do relatório de EDA."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from tech_challenge_churn.config import DATA_PATH, DOCS_DIR, REPORTS_DIR
from tech_challenge_churn.data.load import clean_total_charges, compute_file_hash, read_raw_data
from tech_challenge_churn.data.schema import validate_telco_schema
from tech_challenge_churn.features.build import add_telco_features
from tech_challenge_churn.utils.logging import configure_logging, get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger(__name__)


def _markdown_table(data: pd.DataFrame, max_rows: int = 20) -> str:
    """Converte DataFrame pequeno em tabela Markdown sem depender de tabulate."""
    if data.empty:
        return "_Sem registros._"

    sample = data.head(max_rows).copy()
    headers = [str(column) for column in sample.columns]
    rows = []
    for _, row in sample.iterrows():
        rows.append([str(value) for value in row.tolist()])

    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _save_eda_figures(data: pd.DataFrame, figures_dir: Path) -> None:
    """Salva visualizações principais da exploração de dados."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.countplot(data=data, x="Churn", hue="Churn")
    plt.title("Distribuição do alvo Churn")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=140)
    plt.close()

    contract_summary = (
        data.assign(ChurnBinary=data["Churn"].map({"Yes": 1, "No": 0}))
        .groupby("Contract", observed=False)["ChurnBinary"]
        .mean()
        .reset_index()
    )
    sns.barplot(data=contract_summary, x="Contract", y="ChurnBinary", hue="Contract")
    plt.title("Taxa de churn por tipo de contrato")
    plt.ylabel("Taxa de churn")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_by_contract.png", dpi=140)
    plt.close()

    sns.boxplot(data=data, x="Churn", y="MonthlyCharges", hue="Churn")
    plt.title("Mensalidade por classe de churn")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_charges_by_churn.png", dpi=140)
    plt.close()


def _churn_by_category(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calcula volume e taxa de churn por categoria."""
    summary = (
        data.assign(ChurnBinary=data["Churn"].map({"Yes": 1, "No": 0}))
        .groupby(column, observed=False)
        .agg(clientes=("ChurnBinary", "size"), taxa_churn=("ChurnBinary", "mean"))
        .reset_index()
        .sort_values("taxa_churn", ascending=False)
    )
    summary["taxa_churn"] = (summary["taxa_churn"] * 100).round(2).astype(str) + "%"
    return summary


def generate_eda_report() -> Path:
    """Gera o relatório Markdown de EDA completo."""
    configure_logging()
    raw_data = validate_telco_schema(read_raw_data(DATA_PATH))
    clean_data = clean_total_charges(raw_data)
    engineered_features = add_telco_features(clean_data.drop(columns=["Churn", "customerID"]))
    eda_data = engineered_features.assign(Churn=clean_data["Churn"])
    figures_dir = REPORTS_DIR / "figures"
    _save_eda_figures(clean_data, figures_dir)

    target_distribution = (
        clean_data["Churn"]
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="clientes")
        .assign(percentual=lambda frame: (frame["clientes"] / len(clean_data) * 100).round(2))
    )
    missing_after_cleaning = (
        clean_data.isna()
        .sum()
        .rename_axis("coluna")
        .reset_index(name="nulos")
        .query("nulos > 0")
    )
    numeric_summary = (
        clean_data[["tenure", "MonthlyCharges", "TotalCharges"]]
        .describe()
        .T.round(2)
        .reset_index(names="feature")
    )
    churn_binary = clean_data["Churn"].map({"Yes": 1, "No": 0})
    correlations = (
        clean_data[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]]
        .assign(ChurnBinary=churn_binary)
        .corr(numeric_only=True)["ChurnBinary"]
        .drop("ChurnBinary")
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .round(4)
        .rename_axis("feature")
        .reset_index(name="correlacao_com_churn")
    )

    total_charges_missing = int(clean_data["TotalCharges"].isna().sum())
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# EDA - Telco Customer Churn

Relatório gerado em: {generated_at}

## Resumo Executivo

- Volume: {len(clean_data)} clientes e {len(clean_data.columns)} colunas.
- Alvo positivo: churn (`Churn = Yes`), com taxa de {churn_binary.mean() * 100:.2f}%.
- `TotalCharges` foi recebido como texto e passou por coerção numérica.
- Valores nulos após coerção de `TotalCharges`: {total_charges_missing}.
- Hash SHA256 do dataset: `{compute_file_hash(DATA_PATH)}`.

## Distribuição do Alvo

{_markdown_table(target_distribution)}

## Qualidade dos Dados

{_markdown_table(missing_after_cleaning)}

Observação: os nulos de `TotalCharges` correspondem a clientes com `tenure = 0` e são tratados
como `0` dentro do pipeline de features, pois representam clientes ainda sem cobrança acumulada.

## Estatísticas Numéricas

{_markdown_table(numeric_summary)}

## Correlação Numérica com Churn

{_markdown_table(correlations)}

## Taxa de Churn por Contrato

{_markdown_table(_churn_by_category(clean_data, "Contract"))}

## Taxa de Churn por Serviço de Internet

{_markdown_table(_churn_by_category(clean_data, "InternetService"))}

## Taxa de Churn por Método de Pagamento

{_markdown_table(_churn_by_category(clean_data, "PaymentMethod"))}

## Taxa de Churn por Faixa de Tenure

{_markdown_table(_churn_by_category(eda_data, "tenure_bucket"))}

## Data Readiness

- O dataset tem volume suficiente para validação cruzada estratificada com 5 folds.
- O alvo é desbalanceado, então AUC-ROC deve ser acompanhada por PR-AUC, recall e F1.
- `customerID` será removido do treinamento para evitar identificador sem sinal generalizável.
- Features de gasto dependem de `TotalCharges`; por isso a coerção numérica fica dentro do fluxo
  reprodutível de preparação.
- As categorias de ausência de serviço (`No internet service`, `No phone service`) são colapsadas
  para `No` nas colunas dependentes antes do OneHotEncoder, mantendo `InternetService` e
  `PhoneService` como sinal explícito e reduzindo colinearidade.

## Figuras

- `reports/figures/target_distribution.png`
- `reports/figures/churn_by_contract.png`
- `reports/figures/monthly_charges_by_churn.png`
"""

    output_path = DOCS_DIR / "eda_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("eda_report_gerado", extra={"path": str(output_path)})
    return output_path


def main() -> None:
    """Ponto de entrada do comando generate-eda."""
    generate_eda_report()


if __name__ == "__main__":
    main()
