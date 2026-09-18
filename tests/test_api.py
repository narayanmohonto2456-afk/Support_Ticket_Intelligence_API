from fastapi.testclient import TestClient

from app.main import app
from ml.train import train_models


def test_health_and_prediction() -> None:
    train_models()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "models_loaded": True}

        response = client.post(
            "/predict",
            json={
                "text": "Production is unavailable for all users and the dashboard shows an error."
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["category"] in {
            "account_access",
            "billing",
            "feature_request",
            "technical_issue",
        }
        assert body["priority"] in {"low", "medium", "high", "critical"}
        assert 0 <= body["category_confidence"] <= 1
        assert 0 <= body["priority_confidence"] <= 1


def test_validation_and_batch_prediction() -> None:
    train_models()
    with TestClient(app) as client:
        invalid = client.post("/predict", json={"text": "too short"})
        assert invalid.status_code == 422

        batch = client.post(
            "/predict/batch",
            json={
                "tickets": [
                    {"text": "I was charged twice for my monthly subscription."},
                    {"text": "Please add dark mode in a future release."},
                ]
            },
        )
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2

