PYTHON ?= .venv/Scripts/python.exe

.PHONY: setup lint test check eda data-quality train-baselines train-mlp train-mlp-selected train-feature-ablation promote-challenger train-sklearn-optimization train-sklearn-tuning api mlflow-ui

setup:
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

check: lint test

eda:
	$(PYTHON) -m tech_challenge_churn.reports.eda

data-quality:
	$(PYTHON) -m tech_challenge_churn.reports.data_quality

train-baselines:
	$(PYTHON) -m tech_challenge_churn.models.baselines

train-mlp:
	$(PYTHON) -m tech_challenge_churn.models.train_mlp

train-mlp-selected:
	$(PYTHON) -m tech_challenge_churn.models.train_mlp_selected

train-feature-ablation:
	$(PYTHON) -m tech_challenge_churn.models.feature_ablation

promote-challenger:
	$(PYTHON) -m tech_challenge_churn.models.promote_challenger

train-sklearn-optimization:
	$(PYTHON) -m tech_challenge_churn.models.sklearn_optimization

train-sklearn-tuning:
	$(PYTHON) -m tech_challenge_churn.models.sklearn_tuning

api:
	$(PYTHON) -m uvicorn tech_challenge_churn.api.app:app --reload

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns
