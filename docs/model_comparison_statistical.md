# Comparação Estatística de Modelos

## Objetivo

Comparar os principais modelos usando as métricas por fold registradas no MLflow. A análise evita
decidir apenas por pequenas diferenças de média e explicita a incerteza do protocolo de validação.

## Protocolo

- Métrica comparada: F1 em threshold 0,5.
- Fonte dos scores: métricas `fold_*_f1` dos runs mais recentes no MLflow.
- Teste estatístico: teste exato de sinais bicaudal sobre diferenças pareadas por fold.
- Intervalo: bootstrap percentil de 95% da diferença média.
- Nível de significância: `0.05`.

Com 5 folds, o teste tem baixo poder estatístico. Por isso, ausência de significância não prova que
os modelos são equivalentes; indica apenas que a evidência disponível não justifica uma troca por
diferenças pequenas.

## Scores por Modelo

| model | role | run_id | mean | std |
| --- | --- | --- | --- | --- |
| RandomForest sem gender | challenger_operacional | 0d3b65c9ae43420783c3441115508eb6 | 0.6402 | 0.0123 |
| RandomForest tunado | tabular_tunado | 2902690db5d74205b64d7e342ac82458 | 0.6395 | 0.0134 |
| RandomForest refinado sem gender | refinamento_sem_promocao | c3633f520b8e4ef9afe17325081ee018 | 0.6361 | 0.0157 |
| Regressão Logística balanceada | baseline | 55d9a25c936147a8be40d57f06d0a043 | 0.6279 | 0.0128 |
| MLP PyTorch refinada | modelo_neural | a66dd01773444cfdb639bbd23366233c | 0.6205 | 0.0118 |

## Comparações Pareadas

| model_a | model_b | mean_a | mean_b | mean_diff_a_minus_b | ci95_lower | ci95_upper | sign_test_p_value | significant | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest sem gender | MLP PyTorch refinada | 0.6402 | 0.6205 | 0.0197 | 0.0148 | 0.0240 | 0.0625 | False | Sem evidência estatística suficiente; preferir parcimônia e contexto. |
| RandomForest sem gender | Regressão Logística balanceada | 0.6402 | 0.6279 | 0.0123 | 0.0084 | 0.0167 | 0.0625 | False | Sem evidência estatística suficiente; preferir parcimônia e contexto. |
| RandomForest sem gender | RandomForest tunado | 0.6402 | 0.6395 | 0.0006 | -0.0013 | 0.0027 | 1.0000 | False | Sem evidência estatística suficiente; preferir parcimônia e contexto. |
| RandomForest sem gender | RandomForest refinado sem gender | 0.6402 | 0.6361 | 0.0041 | 0.0001 | 0.0090 | 0.3750 | False | Sem evidência estatística suficiente; preferir parcimônia e contexto. |
| MLP PyTorch refinada | Regressão Logística balanceada | 0.6205 | 0.6279 | -0.0074 | -0.0106 | -0.0043 | 0.0625 | False | Sem evidência estatística suficiente; preferir parcimônia e contexto. |

## Decisão

A comparação reforça a recomendação atual: manter a MLP como modelo neural principal e manter o
RandomForest sem `gender` como challenger operacional. As diferenças observadas são pequenas e não
há evidência estatística suficiente, com 5 folds, para promover os candidatos do refinamento de F1.

Artefatos:

- `reports/model_comparison/fold_scores.csv`
- `reports/model_comparison/statistical_comparison.csv`
