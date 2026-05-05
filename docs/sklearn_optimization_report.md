# Experimentos Scikit-Learn - Otimizacao Permitida

## Escopo

Esta bateria foi executada apenas com ferramentas já adotadas no projeto: Scikit-Learn para
modelagem tabular e MLflow para rastreamento integral. Nao foram adicionadas dependencias externas
como XGBoost, LightGBM ou CatBoost.

## Protocolo

- Validacao cruzada estratificada externa com 5 folds.
- Split interno dentro de cada fold para escolher o threshold de F1 e o threshold de negocio.
- Pipeline unico com feature engineering, imputacao, escala, one-hot encoding e modelo.
- Registro obrigatorio de todos os experimentos no MLflow (`telco-churn-sklearn-optimization`).
- Metricas: AUC-ROC, PR-AUC, Brier, log loss, F1, F1 otimizado, lift@20%, precision@20%,
  recall@20% e valor de negocio estimado.

## Melhor Resultado

- Modelo: `hist_gradient_boosting_regularized`.
- AUC-ROC media: `0.8433`.
- PR-AUC media: `0.6531`.
- F1 em threshold 0,5 medio: `0.6284`.
- F1 com threshold interno medio: `0.6328`.
- Lift@20% medio: `2.5291`.

Melhor PR-AUC entre os experimentos:

- Modelo: `stacking_lr_hgb`.
- PR-AUC media: `0.6633`.
- F1 com threshold interno medio: `0.6307`.

## Leitura Critica

O resultado deve ser comparado contra a Regressao Logistica balanceada registrada previamente no
MLflow. Como o threshold foi escolhido apenas no split interno de cada fold, a estimativa reduz o
risco de overfitting em relacao a otimizar diretamente no fold de teste. PR-AUC e lift@20% devem ter
mais peso que F1 isolado, pois o problema e desbalanceado e a operacao tende a acionar uma campanha
de retencao sobre uma fracao dos clientes.

Arquivo comparativo: `reports/sklearn_optimization/comparison.csv`.
