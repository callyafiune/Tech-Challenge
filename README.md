# Tech Challenge - Churn Telco

Projeto end-to-end para previsão de churn em telecomunicações, com EDA, baselines Scikit-Learn,
MLP em PyTorch, rastreamento com MLflow, API FastAPI, testes automatizados e documentação final.

## Resultados

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
| MLP + SelectKBest mutual_info k=50 | CV 5 folds, threshold interno | 0.8437 | 0.6479 | 0.6333 |

A MLP segue como modelo neural principal e foi retreinada após a auditoria rigorosa de qualidade
dos dados. O melhor F1 holdout da MLP ficou em `0.6396`. No protocolo de validação
cruzada dos modelos Scikit-Learn, o `RandomForest` tunado passou a liderar F1, com `0.6395` em
threshold fixo 0,5 e `0.6340` quando o threshold é escolhido internamente por fold. A MLP com
`SelectKBest(mutual_info_classif, k=50)` melhorou a MLP em validação cruzada, mas não superou o
RandomForest. O stacking manteve a melhor PR-AUC entre os modelos Scikit-Learn. A comparação
operacional deve priorizar também lift@20% e custo de retenção, não apenas F1. A rodada de ablação
indicou que `gender` pode ser removida sem piora de F1 e que `Partner`/`Dependents` podem ser
sumarizados em `has_family_context` com desempenho equivalente.

## Setup

```powershell
cd C:\estudos\Tech-Challenge
.\.venv\Scripts\python.exe -m pip install -e .
```

## Execução

Gerar EDA:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.reports.eda
```

Gerar auditoria rigorosa de qualidade dos dados:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.reports.data_quality
```

Treinar baselines:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.baselines
```

Treinar MLP:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.train_mlp
```

Treinar MLP com seleção de features:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.train_mlp_selected
```

Treinar experimentos Scikit-Learn otimizados:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.sklearn_optimization
```

Treinar tuning avançado de F1 no Scikit-Learn:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.sklearn_tuning
```

Rodar ablação de features:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.feature_ablation
```

Promover challenger operacional:

```powershell
.\.venv\Scripts\python.exe -m tech_challenge_churn.models.promote_challenger
```

Rodar API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn tech_challenge_churn.api.app:app --reload
```

## Docker

Construir a imagem da API:

```powershell
docker build -t tech-challenge-churn:latest .
```

Subir o container:

```powershell
docker run -d --restart unless-stopped -p 8000:8000 --name tech-challenge-churn tech-challenge-churn:latest
```

Para alterar a porta interna usada pelo Uvicorn:

```powershell
docker run -d --restart unless-stopped -p 8080:8080 -e PORT=8080 --name tech-challenge-churn tech-challenge-churn:latest
```

O container inclui `models/mlp` quando os artefatos existem localmente no momento do build. Se a
imagem for construída em uma instância sem os artefatos treinados, monte o diretório de modelos:

```powershell
docker run -d --restart unless-stopped -p 8000:8000 -v ${PWD}\models:/app/models --name tech-challenge-churn tech-challenge-churn:latest
```

Em uma instância Linux, use `-v "$PWD/models:/app/models"` para montar o mesmo diretório.

Ver logs:

```powershell
docker logs -f tech-challenge-churn
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Exemplo de predição:

```powershell
$body = @{
  gender = "Female"
  SeniorCitizen = 0
  Partner = "Yes"
  Dependents = "No"
  tenure = 12
  PhoneService = "Yes"
  MultipleLines = "No"
  InternetService = "Fiber optic"
  OnlineSecurity = "No"
  OnlineBackup = "Yes"
  DeviceProtection = "No"
  TechSupport = "No"
  StreamingTV = "Yes"
  StreamingMovies = "No"
  Contract = "Month-to-month"
  PaperlessBilling = "Yes"
  PaymentMethod = "Electronic check"
  MonthlyCharges = 79.85
  TotalCharges = "958.20"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body $body -ContentType "application/json"
```

Abrir MLflow:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns
```

Acesse `http://localhost:5000`.

## Qualidade

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

## Estrutura

```text
src/tech_challenge_churn/  Código do projeto
tests/                     Testes automatizados
notebooks/                 Notebooks de análise e interpretação
docs/                      ML Canvas, Model Card, deploy, monitoramento e roteiro STAR
models/                    Artefatos treinados gerados localmente e ignorados pelo Git
reports/                   Comparações versionáveis e saídas geradas
mlruns/                    Histórico local do MLflow ignorado pelo Git
```

## Documentação

- `docs/ml_canvas.md`
- `docs/business_metric.md`
- `docs/eda_report.md`
- `docs/data_quality_report.md`
- `docs/dicionario_dados_telco.md`
- `docs/deep_learning_report.md`
- `docs/sklearn_optimization_report.md`
- `docs/sklearn_tuning_report.md`
- `docs/mlp_feature_selection_report.md`
- `docs/feature_ablation_report.md`
- `docs/model_card.md`
- `docs/recomendacao_final.md`
- `docs/promocao_challenger.md`
- `docs/monitoramento.md`
- `docs/arquitetura_deploy.md`
- `docs/roteiro_video_star.md`
- `docs/checklist_entrega.md`

## Dataset

O arquivo esperado é `Telco-Customer-Churn.csv` na raiz do projeto. O pipeline remove
`customerID`, codifica `Churn` como alvo binário, trata `TotalCharges` como numérico e converte
`TotalCharges` vazio para `0` quando `tenure=0`. Categorias redundantes como `No internet service`
e `No phone service` são normalizadas dentro do pipeline para reduzir colinearidade.

## Deploy

O deploy em nuvem ainda não foi executado. A arquitetura recomendada está documentada em
`docs/arquitetura_deploy.md`.
