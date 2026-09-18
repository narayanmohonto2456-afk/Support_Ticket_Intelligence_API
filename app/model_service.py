from dataclasses import dataclass
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float


class TicketModelService:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.category_model = None
        self.priority_model = None

    @property
    def loaded(self) -> bool:
        return self.category_model is not None and self.priority_model is not None

    def load(self) -> None:
        category_path = self.artifact_dir / "category_model.joblib"
        priority_path = self.artifact_dir / "priority_model.joblib"
        missing = [str(path) for path in (category_path, priority_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Model artifacts are missing. Run `python -m ml.train` first. Missing: "
                + ", ".join(missing)
            )
        self.category_model = joblib.load(category_path)
        self.priority_model = joblib.load(priority_path)

    @staticmethod
    def _predict_one(model, text: str) -> ModelPrediction:
        probabilities = model.predict_proba([text])[0]
        best_index = int(probabilities.argmax())
        return ModelPrediction(
            label=str(model.classes_[best_index]),
            confidence=round(float(probabilities[best_index]), 4),
        )

    def predict(self, text: str) -> dict[str, str | float]:
        if not self.loaded:
            raise RuntimeError("Models have not been loaded.")
        category = self._predict_one(self.category_model, text)
        priority = self._predict_one(self.priority_model, text)
        return {
            "category": category.label,
            "category_confidence": category.confidence,
            "priority": priority.label,
            "priority_confidence": priority.confidence,
        }

