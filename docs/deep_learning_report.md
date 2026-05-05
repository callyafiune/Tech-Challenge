# Relatório de Deep Learning e Otimização

## Estratégia Técnica

A estratégia escolhida foi usar uma MLP compacta e regularizada, com
`BCEWithLogitsLoss(pos_weight)`, AdamW, early stopping por PR-AUC e comparação contra os baselines
Scikit-Learn.

## Arquitetura Escolhida

- Camadas ocultas: `128-64-32`.
- Ativação: ReLU.
- Regularização: BatchNorm1d, Dropout `0.3` e AdamW com weight decay
  `0.0001`.
- Tratamento de desbalanceamento: `pos_weight = n_negativos / n_positivos`.
- Early stopping: paciência de `10` épocas monitorando PR-AUC de validação.

## Comparação com Baseline

Baseline de referência:

- Regressão Logística balanceada: AUC-ROC média `0.8471`, F1 média `0.6279`.

Melhor MLP em validação cruzada:

- AUC-ROC média `0.8423`.
- PR-AUC média `0.6492`.
- F1 em threshold 0,5 média `0.6205`.
- F1 com threshold otimizado média `0.6242`.

Modelo final salvo em `models/mlp/`:

- AUC-ROC holdout `0.8463`.
- PR-AUC holdout `0.6580`.
- F1 otimizado holdout `0.6396`.
- Threshold F1 `0.51`.

Experimento adicional com seleção de features:

- Melhor configuração: `SelectKBest(mutual_info_classif, k=50)`.
- AUC-ROC média `0.8437`.
- PR-AUC média `0.6479`.
- F1 em threshold 0,5 média `0.6267`.
- F1 com threshold interno média `0.6333`.

## Interpretação

Como é comum em datasets tabulares pequenos, a MLP compete com a Regressão Logística, mas a
decisão final deve considerar desempenho, simplicidade, estabilidade e interpretabilidade. A
comparação completa foi salva em `reports/mlp/comparison.csv` e os experimentos foram registrados
no MLflow em `telco-churn-mlp`. A bateria com seleção de features foi registrada em
`telco-churn-mlp-feature-selection` e melhorou o F1 médio da MLP em CV, embora sem ganho claro sobre
o melhor modelo tabular tunado.
