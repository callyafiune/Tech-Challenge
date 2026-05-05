# Checklist Final de Entrega

| Item | Status | Evidência |
|---|---|---|
| Estrutura `src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/` | Concluído | Pastas na raiz do projeto |
| README com setup e execução | Concluído | `README.md` |
| `pyproject.toml` como fonte de dependências e ferramentas | Concluído | `pyproject.toml` |
| `.gitignore` de projeto ML | Concluído | `.gitignore` |
| EDA completa | Concluído | `docs/eda_report.md`, `reports/figures/` |
| Auditoria rigorosa de qualidade dos dados | Concluído | `docs/data_quality_report.md`, `reports/data_quality/` |
| ML Canvas | Concluído | `docs/ml_canvas.md` |
| Baselines Scikit-Learn | Concluído | `src/tech_challenge_churn/models/baselines.py` |
| MLflow com histórico de experimentos | Concluído | `mlruns/` |
| Tracking MLflow configurável por ambiente | Concluído | `MLFLOW_TRACKING_URI`, `src/tech_challenge_churn/config.py` |
| Experimentos Scikit-Learn otimizados registrados | Concluído | `telco-churn-sklearn-optimization`, `reports/sklearn_optimization/comparison.csv` |
| Tuning avançado de F1 registrado | Concluído | `telco-churn-sklearn-tuning`, `reports/sklearn_tuning/final_cv_comparison.csv` |
| Refinamento de F1 sem vazamento registrado | Concluído | `telco-churn-f1-refinement`, sem promoção adicional |
| Comparação estatística por folds registrados | Concluído | `docs/model_comparison_statistical.md`, `reports/model_comparison/` |
| MLP PyTorch | Concluído | `src/tech_challenge_churn/models/mlp.py` |
| MLP com seleção de features registrada | Concluído | `telco-churn-mlp-feature-selection`, `reports/mlp_feature_selection/comparison.csv` |
| Ablação de features registrada | Concluído | `telco-churn-feature-ablation`, `reports/feature_ablation/comparison.csv` |
| Early stopping e desbalanceamento | Concluído | `train_torch_model`, `BCEWithLogitsLoss(pos_weight)` |
| Comparação MLP vs baselines | Concluído | `docs/deep_learning_report.md`, `reports/mlp/comparison.csv` |
| FastAPI `/health` e `/predict` | Concluído | `src/tech_challenge_churn/api/app.py` |
| Validação Pydantic | Concluído | `src/tech_challenge_churn/api/schemas.py` |
| Testes automatizados | Concluído | `tests/` |
| Schema Pandera | Concluído | `src/tech_challenge_churn/data/schema.py` |
| Logging estruturado sem `print()` | Concluído | `src/tech_challenge_churn/utils/logging.py` |
| Ruff sem erros | Concluído | `make lint` |
| Pytest passando | Concluído | `make test` |
| Model Card | Concluído | `docs/model_card.md` |
| Recomendação final | Concluído | `docs/recomendacao_final.md` |
| Promoção formal do challenger operacional | Concluído | `docs/promocao_challenger.md`, MLflow `telco-churn-model-promotion` |
| Plano de monitoramento | Concluído | `docs/monitoramento.md` |
| Arquitetura de deploy documentada | Concluído | `docs/arquitetura_deploy.md` |
| Roteiro STAR de 5 minutos | Concluído | `docs/roteiro_video_star.md` |
| Deploy em nuvem | Planejado | Arquitetura documentada como próximo passo |
| Vídeo de apresentação | Roteirizado | `docs/roteiro_video_star.md` |
