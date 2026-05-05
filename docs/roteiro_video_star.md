# Roteiro do Vídeo STAR - 5 Minutos

## 0:00-0:20 - Abertura

Apresentar o objetivo: construir um pipeline profissional para prever churn em telecomunicações,
comparando baselines, MLP PyTorch e servindo o modelo por API.

## 0:20-1:00 - Situation

A operadora está perdendo clientes e precisa priorizar ações de retenção. O dataset Telco tem 7.043
clientes, 21 colunas e alvo desbalanceado, com cerca de 26,5% de churn.

## 1:00-1:40 - Task

O objetivo do projeto foi construir uma solução end-to-end: EDA, ML Canvas, baseline Scikit-Learn,
MLP PyTorch, MLflow, API FastAPI, testes, logging estruturado e documentação final.

## 1:40-3:50 - Action

Exploração e baseline:

- EDA e data readiness.
- Auditoria rigorosa de missing values, domínios, anomalias, outliers, distribuições e correlações.
- Tratamento semântico de `TotalCharges` vazio como `0` quando `tenure=0`.
- Baselines Dummy e Regressão Logística com validação cruzada estratificada.
- Registro de métricas e artefatos no MLflow.

Deep learning:

- MLP PyTorch com BatchNorm, Dropout, AdamW e early stopping.
- `pos_weight` para desbalanceamento.
- Threshold técnico e de negócio.
- Comparação transparente com os baselines.

Industrialização:

- Refatoração para módulos em `src/`.
- API FastAPI com `/health` e `/predict`.
- Pydantic para schema de entrada.
- Logging estruturado e testes automatizados.

## 3:50-4:40 - Result

Resultados principais:

- DummyClassifier: AUC 0,5050 e F1 0,2738.
- Regressão Logística: AUC 0,8471 e F1 0,6279.
- MLP final refinada: AUC holdout 0,8463 e F1 otimizado 0,6396.
- Melhor Scikit-Learn em F1 médio: HGB regularizado com F1 0,6328.

Conclusão: a auditoria de dados melhorou a coerência do pipeline e a MLP refinada alcança o melhor
F1 holdout. Em produção, a decisão deve balancear F1, PR-AUC, lift@20%, custo de retenção,
simplicidade e interpretabilidade.

## 4:40-5:00 - Fechamento

Mostrar próximos passos: monitoramento, calibração de threshold com dados reais de campanha,
retreino periódico e deploy em nuvem como evolução planejada.
