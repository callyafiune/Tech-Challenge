# Model Card - Telco Churn MLP v1

## Identificação

- Nome do modelo: `telco-churn-mlp-v1`.
- Versão: `mlp-v1`, artefatos em `models/mlp/`.
- Data: 2026-05-04.
- Framework: PyTorch para a MLP, Scikit-Learn para pré-processamento.
- Experimentos: MLflow local nos experimentos `telco-churn-baselines`, `telco-churn-mlp`,
  `telco-churn-sklearn-optimization`, `telco-churn-sklearn-tuning` e
  `telco-churn-mlp-feature-selection`, `telco-churn-feature-ablation` e
  `telco-churn-f1-refinement`.

## Visão Geral

O modelo estima a probabilidade de churn de clientes de telecomunicações. O uso pretendido é
priorizar clientes para ações de retenção, apoiando times de marketing, atendimento e gestão de
receita.

O modelo não deve ser usado como decisão automática definitiva contra clientes. A predição deve
servir como sinal de priorização, com revisão operacional e monitoramento contínuo.

## Dados

- Fonte: `Telco-Customer-Churn.csv`.
- Volume: 7.043 clientes.
- Alvo: `Churn`, codificado como 1 para `Yes` e 0 para `No`.
- Taxa positiva: aproximadamente 26,5%.
- Particularidade relevante: `TotalCharges` chega como texto e pode conter espaços em branco;
  a conversão numérica é feita no pipeline com `errors="coerce"`. Quando `tenure=0`, o valor vazio
  é tratado como `0`, pois representa cliente recém-contratado ainda sem cobrança acumulada.

## Features

O pipeline remove `customerID` e usa variáveis de perfil, serviços contratados, contrato, forma de
pagamento, cobranças e tenure. Também cria features derivadas:

- `avg_monthly_spend`.
- `charges_delta`.
- `tenure_bucket`.
- `num_services`.
- `has_protection_bundle`.
- `total_to_monthly_ratio`.
- `num_protection_services`.
- `fiber_without_security`.
- `electronic_check_month_to_month`.
- `month_to_month_low_tenure`.
- Perfis categóricos de contrato, segurança e pagamento.

## Modelo

A MLP final usa:

- Camadas ocultas: `128-64-32`.
- Ativação: ReLU.
- Regularização: BatchNorm1d, Dropout `0.3` e weight decay `0.0001`.
- Otimizador: AdamW.
- Loss: `BCEWithLogitsLoss` com `pos_weight`.
- Early stopping: monitoramento de PR-AUC de validação.
- Threshold de negócio salvo: `0.18`.

## Métricas

| Modelo | Avaliação | AUC-ROC | PR-AUC | F1 |
|---|---:|---:|---:|---:|
| DummyClassifier | CV 5 folds | 0.5050 | 0.2675 | 0.2738 |
| Regressão Logística balanceada | CV 5 folds, baseline registrado | 0.8471 | 0.6596 | 0.6279 |
| MLP PyTorch refinada | CV 5 folds | 0.8423 | 0.6492 | 0.6242 |
| MLP PyTorch refinada | Holdout final | 0.8463 | 0.6580 | 0.6396 |
| HGB Scikit-Learn refinado | CV 5 folds, threshold interno | 0.8433 | 0.6531 | 0.6328 |
| Stacking LR + HGB | CV 5 folds, melhor PR-AUC Scikit-Learn | 0.8474 | 0.6633 | 0.6307 |
| RandomForest tunado | CV 5 folds, F1 em threshold 0,5 | 0.8457 | 0.6568 | 0.6395 |
| RandomForest tunado | CV 5 folds, threshold interno | 0.8457 | 0.6568 | 0.6340 |
| RandomForest sem `gender` | CV 5 folds, ablação de feature | 0.8452 | 0.6558 | 0.6402 |
| RandomForest sem `gender` | CV 5 folds, refinamento sem vazamento | 0.8452 | 0.6558 | 0.6402 |
| MLP + SelectKBest mutual_info k=50 | CV 5 folds, threshold interno | 0.8437 | 0.6479 | 0.6333 |

Interpretação: a MLP supera claramente o baseline ingênuo e fica competitiva com a Regressão
Logística. A auditoria de dados melhorou a coerência do pipeline e elevou o F1 holdout da MLP para
`0.6396`. Em validação cruzada, o RandomForest tunado teve o melhor F1 médio entre os modelos
Scikit-Learn, enquanto o stacking manteve a melhor PR-AUC. A MLP com seleção de features melhorou
o F1 médio da MLP em CV, mas não superou o RandomForest. Para produção, a escolha deve balancear
métrica, simplicidade, interpretabilidade, custo de manutenção e aderência ao requisito da rede
neural. A ablação de features indicou que `gender` pode ser removida sem perda de F1, reduzindo
risco ético e complexidade. A rodada de refinamento sem vazamento confirmou essa escolha: novas
interações aumentaram a dimensionalidade e não melhoraram o F1 médio em threshold 0,5.

## Comparação Estatística

Os principais modelos também foram comparados a partir das métricas por fold registradas no MLflow.
Foi usado teste exato de sinais bicaudal e intervalo bootstrap de 95% para a diferença média de F1.
Com apenas 5 folds, o teste tem baixo poder estatístico; por isso, pequenas diferenças de média não
devem ser tratadas como evidência suficiente para troca automática de modelo.

Resultado prático: a análise reforça manter a MLP como modelo neural principal e o RandomForest sem
`gender` como challenger operacional. A rodada de refinamento de F1 não gerou evidência suficiente
para promoção adicional. O relatório está em `docs/model_comparison_statistical.md`.

## Limitações

- Dataset estático, sem dimensão temporal real de coleta.
- Ausência de dados recentes de interação, reclamações, qualidade de rede ou histórico de ofertas.
- Métrica financeira usa premissas sintéticas de retenção e custo de campanha.
- O modelo pode refletir vieses presentes no histórico de clientes e contratos.
- O desempenho deve ser recalculado quando houver rótulo real pós-campanha.

## Riscos Éticos

Clientes com contratos mensais, idosos ou perfis de pagamento específicos podem receber tratamento
diferenciado. Recomenda-se auditoria por subgrupo e revisão humana antes de ofertas agressivas ou
ações que afetem a experiência do cliente.

## Manutenção

- Retreino sugerido: trimestral ou quando houver alerta de drift.
- Monitorar AUC, F1, recall, PR-AUC, latência e drift de features.
- Registrar toda nova versão no MLflow com hash do dataset, seed e métricas.
- Manter rollback para artefatos anteriores em `models/` ou registry.

## Reprodutibilidade

- Seed global: `42`.
- Validação: `StratifiedKFold` com shuffle e seed fixa.
- Configuração: `pyproject.toml`.
- Execução: `make eda`, `make train-baselines`, `make train-mlp` e `make check`.
- Auditoria de dados: `make data-quality`.
- Experimentos adicionais: `make train-sklearn-optimization`, `make train-sklearn-tuning`,
  `make train-mlp-selected`, `make train-f1-refinement` e `make compare-models`.
