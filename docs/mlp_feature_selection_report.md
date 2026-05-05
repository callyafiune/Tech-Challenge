# MLP com Selecao de Features

## Protocolo

- O pre-processador foi ajustado apenas no treino de cada fold.
- `SelectKBest` foi ajustado apenas no treino de cada fold, depois do OneHotEncoder.
- A MLP manteve early stopping por PR-AUC e `pos_weight` para desbalanceamento.
- Todos os experimentos foram registrados no MLflow em `telco-churn-mlp-feature-selection`.

## Melhor Resultado

- Experimento: `mlp_select_mutual_info_k50`.
- AUC-ROC media: `0.8437`.
- PR-AUC media: `0.6479`.
- F1 medio em threshold 0,5: `0.6267`.
- F1 medio com threshold interno: `0.6333`.

## Leitura Critica

A selecao de features foi testada para verificar se a MLP estava sofrendo com ruido nas 80 features
codificadas. O resultado deve ser comparado com a MLP refinada sem selecao, que tinha F1 otimizado
medio de `0.6242` em CV e F1 holdout de `0.6396`.

Conclusao: houve ganho em CV contra a MLP sem selecao (`0.6333` contra `0.6242`), mas o resultado
ficou ligeiramente abaixo do RandomForest tunado reavaliado por CV (`0.6340` com threshold interno).
Assim, a selecao de features e util para reduzir ruido da MLP, mas nao muda o campeao operacional.

Artefato comparativo: `reports/mlp_feature_selection/comparison.csv`.
