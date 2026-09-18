from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(
        min_length=10,
        max_length=2000,
        examples=["I cannot sign in after resetting my password."],
    )


class Prediction(BaseModel):
    category: str
    category_confidence: float = Field(ge=0, le=1)
    priority: str
    priority_confidence: float = Field(ge=0, le=1)


class TicketPrediction(Prediction):
    text: str
    model_version: str


class BatchRequest(BaseModel):
    tickets: list[TicketRequest] = Field(min_length=1, max_length=50)


class BatchResponse(BaseModel):
    predictions: list[TicketPrediction]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

