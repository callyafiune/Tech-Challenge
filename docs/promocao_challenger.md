# Promoção Formal - Challenger Operacional

## Modelo Promovido

- Nome: `telco-churn-random-forest-challenger`.
- Versão: `random_forest_no_gender_v1`.
- Papel: `operational_challenger`.
- Status: `promoted_challenger`.
- Feature set: `no_gender`.
- Atributo removido: `gender`.

## Protocolo

- Split holdout estratificado de 20% para métricas de promoção.
- Treino final do artefato promovido com 100% da base após escolha dos thresholds.
- Registro no MLflow em `telco-churn-model-promotion`.
- Hash SHA256 do dataset registrado no metadata.

## Métricas de Promoção

- AUC-ROC validação: `0.8453`.
- PR-AUC validação: `0.6531`.
- F1 em threshold 0,5: `0.6391`.
- F1 no threshold promovido: `0.6403`.
- Threshold F1 promovido: `0.46`.
- Threshold de negócio promovido: `0.24`.
- Lift@20%: `2.5339`.

## Artefatos

- Modelo local: `C:\estudos\Tech-Challenge\models\challengers\random_forest_no_gender_v1\model.joblib`.
- Metadata local: `C:\estudos\Tech-Challenge\models\challengers\random_forest_no_gender_v1\metadata.json`.
- MLflow run ID: `e04eb2c506144cefb37bbd20aa17797f`.
- MLflow model URI: `models:/m-dd5f937c35df406aa06cd2124ad06bb7`.
- MLflow registered model: `telco-churn-random-forest-challenger`.
- MLflow registry version: `1`.
- MLflow alias: `challenger`.

## Decisão

O RandomForest sem `gender` foi promovido como challenger operacional, não como substituto direto da
MLP. A MLP segue como modelo neural principal, enquanto o RandomForest permanece disponível para
validação operacional em shadow mode.
