# Revisao Rigorosa de Qualidade dos Dados - Telco Churn

Relatorio gerado em: 2026-05-05 20:18:17 UTC

## Veredito

- A base tem 7043 linhas, 21 colunas e hash
  `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- Nao foram encontrados valores categoricos fora do dominio esperado.
- Nao ha `customerID` duplicado.
- O alvo esta desbalanceado: churn positivo de 26.54%.
- Existem 11 strings vazias em `TotalCharges`; todas ocorrem em clientes com `tenure=0`.
- O pipeline foi refinado para tratar `TotalCharges` vazio com `0` quando `tenure=0`, evitando
  imputacao por mediana nesse caso sem usar informacao do alvo.
- O pipeline tambem colapsa `No internet service` e `No phone service` para `No` nas colunas
  dependentes, preservando `InternetService` e `PhoneService` como sinal explicito e reduzindo
  colinearidade deterministica.

## Missing Values

| coluna | nulos_raw | strings_vazias | nulos_pos_limpeza | percentual_pos_limpeza |
| --- | --- | --- | --- | --- |
| TotalCharges | 0 | 11 | 11 | 0.1562 |

## Balanceamento de Classes

| classe | clientes | percentual | razao_maioria_minoria |
| --- | --- | --- | --- |
| No | 5174 | 73.46 | 2.768 |
| Yes | 1869 | 26.54 | 2.768 |

## Valores Invalidos e Dominios Categoricos

Valores invalidos:

_Sem registros._

Resumo dos dominios:

| coluna | valores_observados | valores_esperados | qtd_invalidos |
| --- | --- | --- | --- |
| gender | Female, Male | Female, Male | 0 |
| Partner | No, Yes | Yes, No | 0 |
| Dependents | No, Yes | Yes, No | 0 |
| PhoneService | No, Yes | Yes, No | 0 |
| MultipleLines | No, No phone service, Yes | No phone service, No, Yes | 0 |
| InternetService | DSL, Fiber optic, No | DSL, Fiber optic, No | 0 |
| OnlineSecurity | No, No internet service, Yes | No internet service, No, Yes | 0 |
| OnlineBackup | No, No internet service, Yes | No internet service, No, Yes | 0 |
| DeviceProtection | No, No internet service, Yes | No internet service, No, Yes | 0 |
| TechSupport | No, No internet service, Yes | No internet service, No, Yes | 0 |
| StreamingTV | No, No internet service, Yes | No internet service, No, Yes | 0 |
| StreamingMovies | No, No internet service, Yes | No internet service, No, Yes | 0 |
| Contract | Month-to-month, One year, Two year | Month-to-month, One year, Two year | 0 |
| PaperlessBilling | No, Yes | Yes, No | 0 |
| PaymentMethod | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | 0 |
| Churn | No, Yes | Yes, No | 0 |

## Anomalias Logicas

| checagem | ocorrencias | interpretacao |
| --- | --- | --- |
| customerID duplicado | 0 | Deve ser zero para evitar repeticao de cliente. |
| customerID vazio | 0 | Deve ser zero para manter rastreabilidade. |
| TotalCharges vazio no raw | 11 | Esperado apenas para tenure=0 no dataset Telco. |
| TotalCharges nulo com tenure=0 | 11 | Caso sem cobranca acumulada; tratado como zero no pipeline. |
| TotalCharges nulo com tenure>0 | 0 | Deve ser zero; se aparecer, exige investigacao. |
| TotalCharges negativo | 0 | Deve ser zero. |
| MonthlyCharges negativo | 0 | Deve ser zero. |
| PhoneService=No com MultipleLines diferente de No phone service | 0 | Deve ser zero pela regra do dataset. |
| PhoneService=Yes com MultipleLines=No phone service | 0 | Deve ser zero pela regra do dataset. |
| InternetService=No com colunas dependentes inconsistentes | 0 | Deve ser zero pela regra do dataset. |
| InternetService ativo com No internet service | 0 | Deve ser zero pela regra do dataset. |

## Distribuicao, Simetria e Outliers

| feature | count | missing | mean | std | min | q1 | median | q3 | max | skew | kurtosis | simetria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenure | 7043 | 0 | 32.3711 | 24.5595 | 0.0 | 9.0 | 29.0 | 55.0 | 72.0 | 0.2395 | -1.3874 | aproximadamente simetrica |
| MonthlyCharges | 7043 | 0 | 64.7617 | 30.09 | 18.25 | 35.5 | 70.35 | 89.85 | 118.75 | -0.2205 | -1.2573 | aproximadamente simetrica |
| TotalCharges | 7043 | 0 | 2279.7343 | 2266.7945 | 0.0 | 398.55 | 1394.55 | 3786.6 | 8684.8 | 0.9632 | -0.2286 | assimetria moderada |
| avg_monthly_spend | 7032 | 11 | 64.7994 | 30.1859 | 13.775 | 36.1799 | 70.3732 | 90.1796 | 121.4 | -0.2111 | -1.2467 | aproximadamente simetrica |
| charges_delta | 7032 | 11 | -0.0012 | 2.6162 | -18.9 | -1.1602 | 0.0 | 1.1478 | 19.125 | -0.1576 | 5.3616 | aproximadamente simetrica |
| total_to_monthly_ratio | 7043 | 0 | 32.3734 | 24.5959 | 0.0 | 8.7172 | 28.6731 | 55.2445 | 79.3418 | 0.2451 | -1.3792 | aproximadamente simetrica |
| num_services | 7043 | 0 | 3.3629 | 2.062 | 0.0 | 1.0 | 3.0 | 5.0 | 8.0 | 0.4508 | -0.8628 | aproximadamente simetrica |
| num_protection_services | 7043 | 0 | 1.2657 | 1.2869 | 0.0 | 0.0 | 1.0 | 2.0 | 4.0 | 0.6246 | -0.806 | assimetria moderada |

Outliers por IQR:

| feature | limite_inferior | limite_superior | outliers_iqr | percentual |
| --- | --- | --- | --- | --- |
| charges_delta | -4.6221 | 4.6097 | 530 | 7.537 |
| tenure | -60.0 | 124.0 | 0 | 0.0 |
| TotalCharges | -4683.525 | 8868.675 | 0 | 0.0 |
| MonthlyCharges | -46.025 | 171.375 | 0 | 0.0 |
| avg_monthly_spend | -44.8196 | 171.1791 | 0 | 0.0 |
| total_to_monthly_ratio | -61.0736 | 125.0353 | 0 | 0.0 |
| num_services | -5.0 | 11.0 | 0 | 0.0 |
| num_protection_services | -3.0 | 5.0 | 0 | 0.0 |

Os outliers nao foram removidos automaticamente. Em churn, valores extremos de mensalidade ou
tempo de contrato podem ser sinal real de comportamento do cliente. A decisao segura e registrar,
escalar e testar robustez em vez de descartar registros.

## Correlacoes e Colinearidade

Pares numericos com |correlacao| >= 0.85:

| feature_1 | feature_2 | correlacao_pearson |
| --- | --- | --- |
| tenure | total_to_monthly_ratio | 0.9989 |
| MonthlyCharges | avg_monthly_spend | 0.9962 |
| num_services | num_protection_services | 0.8547 |

As correlacoes fortes confirmam redundancias esperadas: `TotalCharges` se relaciona com `tenure`,
`avg_monthly_spend` se aproxima de `MonthlyCharges` e `total_to_monthly_ratio` se aproxima de
tempo de relacionamento. Isso justifica testar ablacoees em experimentos, mas nao remover features
sem registro no MLflow.

## Codificacao Categorica

- Features originais sem ID/alvo: 19.
- Features numericas configuradas: 17.
- Features categoricas configuradas: 19.
- Features finais apos OneHotEncoder: 80.
- Saida esparsa: False.

Top correlacoes simples apos codificacao:

| feature_codificada | correlacao_com_churn |
| --- | --- |
| categorical__Contract_Month-to-month | 0.4051 |
| categorical__internet_security_profile_Fiber optic__security_No__support_No | 0.373 |
| numeric__electronic_check_month_to_month | 0.3676 |
| categorical__payment_contract_profile_Electronic check__Month-to-month | 0.3676 |
| numeric__fiber_without_security | 0.3549 |
| numeric__month_to_month_low_tenure | 0.3532 |
| numeric__tenure | -0.3522 |
| numeric__total_to_monthly_ratio | -0.352 |
| categorical__contract_tenure_segment_Month-to-month__0-6 | 0.3252 |
| categorical__tenure_bucket_0-6 | 0.3085 |
| categorical__InternetService_Fiber optic | 0.308 |
| categorical__Contract_Two year | -0.3023 |
| categorical__PaymentMethod_Electronic check | 0.3019 |
| categorical__tenure_bucket_49+ | -0.2632 |
| categorical__contract_tenure_segment_Two year__49+ | -0.2457 |

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
