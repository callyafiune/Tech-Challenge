# Tuning Avancado Scikit-Learn para F1

## Protocolo

- Triagem com `ParameterSampler` e holdout estratificado apenas para ordenar tentativas.
- Cada tentativa foi registrada no MLflow em `telco-churn-sklearn-tuning` com parametros e metricas.
- Finalistas foram reavaliados com validacao cruzada estratificada e threshold escolhido em split
  interno, reduzindo risco de overfitting.
- Ferramentas restritas ao escopo técnico do projeto: Scikit-Learn e MLflow.

## Melhor Tentativa de Triagem

- Modelo: `stack_tuned_002`.
- Familia: `stacking`.
- F1 validacao com threshold interno: `0.6544`.
- PR-AUC validacao: `0.6548`.

## Melhor Resultado Reavaliado

- Modelo: `random_forest_005`.
- AUC-ROC media: `0.8457`.
- PR-AUC media: `0.6568`.
- F1 medio em threshold 0,5: `0.6395`.
- F1 medio com threshold interno: `0.6340`.

## Leitura Critica

A triagem pode superestimar F1 porque muitas tentativas olham o mesmo holdout de desenvolvimento.
Por isso, o numero que deve entrar na comparacao final e o F1 medio dos finalistas reavaliados por
CV. A busca testou HGB, RandomForest, ExtraTrees, stacking e SVC calibrado, com e sem selecao de
features (`SelectKBest` por ANOVA F ou informacao mutua).

Artefatos:

- `reports/sklearn_tuning/search_results.csv`
- `reports/sklearn_tuning/final_cv_comparison.csv`
