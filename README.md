# Support Ticket Intelligence API

An end-to-end learning project that trains NLP models and serves predictions through FastAPI.
It classifies customer-support tickets by **category** and **priority** using TF-IDF features and
logistic regression.

## Why this is more than an API demo

- Deterministic synthetic dataset generation with clearly disclosed limitations
- Reproducible Scikit-learn pipelines for preprocessing and classification
- Stratified train/test split and saved evaluation metrics
- Category and priority confidence scores
- Single and batch prediction endpoints
- Pydantic validation, OpenAPI documentation, automated tests and Docker support

## Architecture

```text
Ticket text -> FastAPI -> Saved Scikit-learn pipelines -> Predictions
                              |
Synthetic data -> Training -> TF-IDF + Logistic Regression
                              |
                         metrics.json
```

## Predicted labels

**Categories:** `account_access`, `billing`, `technical_issue`, `feature_request`

**Priorities:** `low`, `medium`, `high`, `critical`

## Local setup

Use Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m ml.train
uvicorn app.main:app --reload
```

Open:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Example prediction

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"text":"Production is unavailable for all users and the dashboard shows an error."}'
```

Example response:

```json
{
  "text": "Production is unavailable for all users and the dashboard shows an error.",
  "category": "technical_issue",
  "category_confidence": 0.81,
  "priority": "critical",
  "priority_confidence": 0.88,
  "model_version": "1.0.0"
}
```

## Train and evaluate

```powershell
python -m ml.train
```

This creates:

- `artifacts/category_model.joblib`
- `artifacts/priority_model.joblib`
- `artifacts/metrics.json`
- `data/generated_tickets.csv`

The generated dataset is synthetic and intended for learning. Metrics on it do **not** represent
real production performance. A portfolio-ready next version should use an anonymized public dataset,
perform error analysis, compare multiple algorithms and document model limitations.

## Test and lint

```powershell
pytest -q
ruff check .
```

## Docker

```powershell
docker build -t ticket-intelligence-api .
docker run --rm -p 8000:8000 ticket-intelligence-api
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API metadata |
| GET | `/health` | Readiness check |
| POST | `/predict` | Predict one ticket |
| POST | `/predict/batch` | Predict up to 50 tickets |

## Skills demonstrated

Python, FastAPI, Pydantic, Scikit-learn, NLP, TF-IDF, classification, model evaluation,
REST API design, testing, Docker and Git.

## Suggested portfolio upgrade path

1. Replace synthetic examples with a documented public support-ticket dataset.
2. Add exploratory data analysis and class-distribution charts.
3. Compare logistic regression, Linear SVM and Naive Bayes.
4. Add cross-validation and hyperparameter tuning.
5. Track experiments with MLflow.
6. Store prediction feedback in PostgreSQL and monitor performance drift.
7. Deploy the Docker image and add a live demo URL.

