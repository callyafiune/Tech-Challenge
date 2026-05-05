# Refinamento de F1 sem Vazamento

## Objetivo

Testar ajustes finos sugeridos para churn tabular sem sair do escopo técnico do projeto. A bateria
mantém Scikit-Learn, PyTorch e MLflow como ferramentas de referência e não usa XGBoost, LightGBM,
CatBoost ou oversampling sintético.

## Protocolo

- Validação cruzada estratificada externa com 5 folds.
- Em cada fold, o threshold de F1 é escolhido apenas em split interno de validação.
- O fold externo é usado somente para estimativa final de métricas.
- Todos os candidatos são registrados no MLflow em `telco-churn-f1-refinement`.
- As novas features usam apenas atributos disponíveis no payload, sem usar `Churn`.

## Referência

- Modelo: `rf_no_gender_reference`.
- Feature set: `no_gender_current`.
- Features finais: `79`.
- F1 médio em threshold 0,5: `0.6402`.
- F1 médio com threshold interno: `0.6344`.
- PR-AUC média: `0.6558`.

## Melhor F1 em Threshold 0,5

- Modelo: `rf_no_gender_reference`.
- Família: `random_forest`.
- Feature set: `no_gender_current`.
- Features finais: `79`.
- F1 médio: `0.6402`.
- Delta contra referência: `+0.0000`.
- PR-AUC média: `0.6558`.

## Melhor F1 com Threshold Interno

- Modelo: `rf_no_gender_refined`.
- Família: `random_forest`.
- Feature set: `no_gender_refined`.
- F1 médio com threshold interno: `0.6352`.
- Delta contra referência: `+0.0008`.

## Decisão

Nenhum modelo desta rodada será promovido. O ganho marginal em threshold interno não compensa o
aumento de complexidade do pipeline, e o F1 médio em threshold 0,5 ficou abaixo da referência.

## Leitura Crítica

O protocolo evita vazamento ao escolher threshold somente no split interno. Diferenças pequenas de
F1 devem ser tratadas com cautela, porque podem estar dentro da variância entre folds. Se nenhum
candidato superar a referência de forma consistente, a recomendação permanece manter o RandomForest
sem `gender` como challenger operacional e a MLP como modelo neural principal.

Artefato comparativo: `reports/f1_refinement/comparison.csv`.
