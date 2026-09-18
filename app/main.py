from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_service import MODEL_VERSION, TicketModelService
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    TicketPrediction,
    TicketRequest,
)

model_service = TicketModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Support Ticket Intelligence API",
    description="Classifies support tickets by category and business priority.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/", tags=["Meta"])
def root() -> dict[str, str]:
    return {
        "message": "Support Ticket Intelligence API",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=model_service.loaded)


def make_prediction(ticket: TicketRequest) -> TicketPrediction:
    try:
        result = model_service.predict(ticket.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TicketPrediction(text=ticket.text, model_version=MODEL_VERSION, **result)


@app.post("/predict", response_model=TicketPrediction, tags=["Predictions"])
def predict(ticket: TicketRequest) -> TicketPrediction:
    return make_prediction(ticket)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=[make_prediction(ticket) for ticket in request.tickets])

