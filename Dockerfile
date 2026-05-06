FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TECH_CHALLENGE_PROJECT_ROOT=/app \
    PORT=8000

WORKDIR /app

# Dependências mínimas de sistema para instalação e execução das bibliotecas Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        fastapi==0.136.0 \
        joblib==1.5.3 \
        numpy==2.4.4 \
        pandas==2.3.3 \
        pydantic==2.13.2 \
        python-json-logger==4.1.0 \
        scikit-learn==1.8.0 \
        torch==2.11.0 \
        uvicorn==0.44.0 \
    && python -m pip install --no-deps .

# Copia apenas os arquivos necessários para servir a API. O .dockerignore remove dados brutos,
# MLflow, ambientes locais e caches; se models/mlp existir localmente, ele entra na imagem.
COPY . .

RUN groupadd --system app \
    && useradd --system --create-home --gid app app \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); raise SystemExit(0 if data.get('model_loaded') else 1)"

CMD ["sh", "-c", "uvicorn tech_challenge_churn.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
