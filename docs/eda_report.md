# EDA - Telco Customer Churn

Relatório gerado em: 2026-05-05 18:10:45 UTC

## Resumo Executivo

- Volume: 7043 clientes e 21 colunas.
- Alvo positivo: churn (`Churn = Yes`), com taxa de 26.54%.
- `TotalCharges` foi recebido como texto e passou por coerção numérica.
- Valores nulos após coerção de `TotalCharges`: 11.
- Hash SHA256 do dataset: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

## Distribuição do Alvo

| classe | clientes | percentual |
| --- | --- | --- |
| No | 5174 | 73.46 |
| Yes | 1869 | 26.54 |

## Qualidade dos Dados

| coluna | nulos |
| --- | --- |
| TotalCharges | 11 |

Observação: os nulos de `TotalCharges` correspondem a clientes com `tenure = 0` e são tratados
como `0` dentro do pipeline de features, pois representam clientes ainda sem cobrança acumulada.

## Estatísticas Numéricas

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenure | 7043.0 | 32.37 | 24.56 | 0.0 | 9.0 | 29.0 | 55.0 | 72.0 |
| MonthlyCharges | 7043.0 | 64.76 | 30.09 | 18.25 | 35.5 | 70.35 | 89.85 | 118.75 |
| TotalCharges | 7032.0 | 2283.3 | 2266.77 | 18.8 | 401.45 | 1397.48 | 3794.74 | 8684.8 |

## Correlação Numérica com Churn

| feature | correlacao_com_churn |
| --- | --- |
| tenure | -0.3522 |
| TotalCharges | -0.1995 |
| MonthlyCharges | 0.1934 |
| SeniorCitizen | 0.1509 |

## Taxa de Churn por Contrato

| Contract | clientes | taxa_churn |
| --- | --- | --- |
| Month-to-month | 3875 | 42.71% |
| One year | 1473 | 11.27% |
| Two year | 1695 | 2.83% |

## Taxa de Churn por Serviço de Internet

| InternetService | clientes | taxa_churn |
| --- | --- | --- |
| Fiber optic | 3096 | 41.89% |
| DSL | 2421 | 18.96% |
| No | 1526 | 7.4% |

## Taxa de Churn por Método de Pagamento

| PaymentMethod | clientes | taxa_churn |
| --- | --- | --- |
| Electronic check | 2365 | 45.29% |
| Mailed check | 1612 | 19.11% |
| Bank transfer (automatic) | 1544 | 16.71% |
| Credit card (automatic) | 1522 | 15.24% |

## Taxa de Churn por Faixa de Tenure

| tenure_bucket | clientes | taxa_churn |
| --- | --- | --- |
| 0-6 | 1481 | 52.94% |
| 7-12 | 705 | 35.89% |
| 13-24 | 1024 | 28.71% |
| 25-48 | 1594 | 20.39% |
| 49+ | 2239 | 9.51% |

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
