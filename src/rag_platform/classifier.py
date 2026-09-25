"""Machine learning failure diagnosis layer: feature extraction, probability estimation, and active learning queue."""

from __future__ import annotations

import re
from typing import Any

try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    classification_report = None  # type: ignore
    f1_score = None  # type: ignore
    SKLEARN_AVAILABLE = False

from rag_platform.models import (
    Answerability,
    MetricResult,
    RagTrace,
    TestCase,
)

FAILURE_CLASSES = ["PASS", "RET-01", "RET-02", "GEN-01", "CIT-01", "ABS-01", "OPS-01"]


class FeatureExtractor:
    """Extracts standardized tabular feature vectors from traces and cases.
    
    Prefers structural, distributional, and retrieval signals over rule-generated metrics
    to prevent target leakage during training.
    """

    FEATURE_NAMES = [
        "q_len",
        "ans_len",
        "abstained",
        "num_chunks",
        "top_chunk_score",
        "score_gap",
        "avg_chunk_len",
        "latency_ms",
        "num_citations",
        "citation_density",
        "is_infra_error",
        "is_unanswerable",
        "lexical_overlap_ratio",
    ]

    @classmethod
    def extract(cls, trace: RagTrace, case: TestCase, metrics: list[MetricResult] | None = None) -> list[float]:
        chunks = trace.retrieved_chunks
        top_score = float(chunks[0].score) if chunks else 0.0
        score_gap = float(chunks[0].score - chunks[1].score) if len(chunks) >= 2 else 0.0

        ans = trace.answer or ""
        total_chunk_len = sum(len(c.text) for c in chunks)
        avg_chunk_len = (total_chunk_len / len(chunks)) if chunks else 0.0

        ans_words = set(re.findall(r"\w+", ans.lower()))
        chunk_words = set(re.findall(r"\w+", " ".join(c.text for c in chunks).lower())) if chunks else set()
        overlap_ratio = len(ans_words & chunk_words) / max(len(ans_words), 1)

        sentences = [s.strip() for s in re.split(r"[.!?]+", ans) if s.strip()]
        citation_density = len(trace.citations) / max(len(sentences), 1)

        return [
            float(len(case.question)),
            float(len(ans)),
            1.0 if trace.abstained else 0.0,
            float(len(chunks)),
            round(top_score, 4),
            round(score_gap, 4),
            round(avg_chunk_len, 2),
            float(trace.latency_ms),
            float(len(trace.citations)),
            round(citation_density, 4),
            1.0 if trace.error_code == "OPS-01" else 0.0,
            1.0 if case.answerability == Answerability.UNANSWERABLE else 0.0,
            round(overlap_ratio, 4),
        ]


class MLFailureClassifier:
    """Gradient-boosted failure diagnosis model with uncalibrated probability estimates.
    
    Outputs raw model probability estimates. Does not claim calibrated confidence.
    """

    def __init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            self.model = None
            self.classes = list(FAILURE_CLASSES)
            self.is_fitted = False
            return

        self.classes = list(FAILURE_CLASSES)
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        self.is_fitted = False

    def train(self, X: list[list[float]] | Any, y: list[str]) -> None:
        """Fit classifier on extracted features and failure labels."""
        if not SKLEARN_AVAILABLE or self.model is None:
            raise RuntimeError("scikit-learn and numpy are required to train the MLFailureClassifier. Install with `pip install .[ml]`.")

        X_arr = np.array(X, dtype=float)
        self.model.fit(X_arr, y)
        self.classes = list(self.model.classes_)
        self.is_fitted = True

    def predict(
        self,
        trace: RagTrace,
        case: TestCase,
        metrics: list[MetricResult] | None = None,
    ) -> tuple[str, float]:
        """Predict likely failure category and diagnostic probability estimate."""
        if not SKLEARN_AVAILABLE or not self.is_fitted or self.model is None:
            # Fallback deterministic heuristic if ML subsystem is not fitted
            return ("PASS", 1.0) if not trace.error_code else ("OPS-01", 1.0)

        vec = FeatureExtractor.extract(trace, case, metrics)
        probs = self.model.predict_proba([vec])[0]
        max_idx = int(np.argmax(probs))
        predicted_class = self.classes[max_idx]
        probability_estimate = float(probs[max_idx])

        return predicted_class, round(probability_estimate, 4)

    def evaluate_model(self, X_test: list[list[float]], y_test: list[str]) -> dict[str, Any]:
        """Compute Macro F1 and evaluation metrics on held-out test data."""
        if not SKLEARN_AVAILABLE or not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before evaluation and scikit-learn must be available.")

        X_arr = np.array(X_test, dtype=float)
        y_pred = self.model.predict(X_arr)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        return {
            "macro_f1": round(float(macro_f1), 4),
            "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        }

    @staticmethod
    def filter_active_learning_queue(
        predictions: list[tuple[RagTrace, str, float]],
        probability_threshold: float = 0.70,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Identify low-probability predictions that require human verification to enrich training data."""
        thresh = confidence_threshold if confidence_threshold is not None else probability_threshold
        queue = []
        for trace, predicted_label, prob_estimate in predictions:
            if prob_estimate < thresh and predicted_label != "PASS":
                queue.append({
                    "trace_id": trace.trace_id,
                    "predicted_label": predicted_label,
                    "probability_estimate": prob_estimate,
                    "confidence": prob_estimate,
                    "question": trace.question,
                    "answer": trace.answer,
                })
        return queue
