"""Aplicação FastAPI para servir o modelo de churn."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from tech_challenge_churn.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from tech_challenge_churn.api.service import InferenceService
from tech_challenge_churn.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo no startup da aplicação."""
    try:
        app.state.inference_service = InferenceService()
        logger.info("modelo_api_carregado")
    except Exception as exc:
        app.state.inference_service = None
        logger.exception("falha_ao_carregar_modelo_api", extra={"erro": str(exc)})
    yield


app = FastAPI(
    title="Tech Challenge Churn API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def latency_logging_middleware(request: Request, call_next):
    """Registra latência por request sem logar o payload."""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "request_concluida",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


def get_inference_service(request: Request) -> InferenceService:
    """Recupera o serviço carregado no lifespan."""
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Modelo indisponivel.")
    return service


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Retorna o estado operacional da API."""
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        input_dim=service.input_dim,
        threshold_business=service.threshold_business,
        model_version=service.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: CustomerFeatures,
    service: Annotated[InferenceService, Depends(get_inference_service)],
    threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> PredictionResponse:
    """Prediz a probabilidade de churn de um cliente."""
    prediction = service.predict_one(payload.model_dump(), threshold=threshold)
    return PredictionResponse(**prediction)
