"""Schemas Pydantic da API de inferência."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    """Payload de entrada com as 19 features usadas pelo modelo."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    OnlineBackup: Literal["No internet service", "No", "Yes"]
    DeviceProtection: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]
    StreamingTV: Literal["No internet service", "No", "Yes"]
    StreamingMovies: Literal["No internet service", "No", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def validate_total_charges(cls, value: object) -> str:
        """Aceita strings numéricas ou espaços, como ocorre no CSV original."""
        if value is None:
            raise ValueError("TotalCharges nao pode ser nulo.")
        text = str(value)
        if text.strip() == "":
            return text
        try:
            float(text)
        except ValueError as exc:
            raise ValueError("TotalCharges deve ser numerico ou vazio.") from exc
        return text


class PredictionResponse(BaseModel):
    """Resposta de inferência para um cliente."""

    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    threshold: float = Field(ge=0, le=1)
    model_version: str


class HealthResponse(BaseModel):
    """Status operacional da API."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    input_dim: int | None = None
    threshold_business: float | None = None
    model_version: str | None = None
