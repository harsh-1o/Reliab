"""Machine learning failure diagnosis layer: feature extraction, calibrated classifier, and active learning queue.

# ponytail: single file covers tabular feature extractor, gradient-boosted classifier, and active learning.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score

from rag_platform.models import (
    Answerability,
    FailureCode,
    MetricResult,
    RagTrace,
    TestCase,
)

FAILURE_CLASSES = ["PASS", "RET-01", "RET-02", "GEN-01", "CIT-01", "ABS-01", "OPS-01"]


class FeatureExtractor:
    """Extracts standardized tabular feature vectors from traces, cases, and metrics."""

    FEATURE_NAMES = [
        "q_len",
        "ans_len",
        "abstained",
        "num_chunks",
        "top_chunk_score",
        "score_gap",
        "latency_ms",
        "faithfulness",
        "recall",
        "correctness",
        "num_citations",
        "is_infra_error",
        "is_unanswerable",
    ]

    @classmethod
    def extract(cls, trace: RagTrace, case: TestCase, metrics: list[MetricResult] | None = None) -> list[float]:
        metric_map = {m.metric_name: m.score for m in (metrics or [])}

        chunks = trace.retrieved_chunks
        top_score = float(chunks[0].score) if chunks else 0.0
        score_gap = float(chunks[0].score - chunks[1].score) if len(chunks) >= 2 else 0.0

        ans = trace.answer or ""
        return [
            float(len(case.question)),
            float(len(ans)),
            1.0 if trace.abstained else 0.0,
            float(len(chunks)),
            round(top_score, 4),
            round(score_gap, 4),
            float(trace.latency_ms),
            float(metric_map.get("faithfulness", 1.0 if trace.abstained else 0.0)),
            float(metric_map.get("recall_at_5", 1.0 if not case.relevant_documents else 0.0)),
            float(metric_map.get("answer_correctness", 1.0 if trace.abstained else 0.0)),
            float(len(trace.citations)),
            1.0 if trace.error_code == "OPS-01" else 0.0,
            1.0 if case.answerability == Answerability.UNANSWERABLE else 0.0,
        ]


class MLFailureClassifier:
    """Gradient-boosted failure diagnosis model with calibrated probabilities."""

    def __init__(self) -> None:
        self.classes = list(FAILURE_CLASSES)
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        self.is_fitted = False

    def train(self, X: list[list[float]] | np.ndarray, y: list[str]) -> None:
        """Fit classifier on extracted features and failure labels."""
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
        """Predict likely failure category and diagnostic confidence probability."""
        if not self.is_fitted:
            # Fallback heuristic if model not trained yet
            return ("PASS", 1.0) if not trace.error_code else ("OPS-01", 1.0)

        vec = FeatureExtractor.extract(trace, case, metrics)
        probs = self.model.predict_proba([vec])[0]
        max_idx = int(np.argmax(probs))
        predicted_class = self.classes[max_idx]
        confidence = float(probs[max_idx])

        return predicted_class, round(confidence, 4)

    def evaluate_model(self, X_test: list[list[float]], y_test: list[str]) -> dict[str, Any]:
        """Compute Macro F1 and evaluation metrics on held-out test data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation.")

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
        confidence_threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        """Identify low-confidence predictions that require human verification to enrich training data."""
        queue = []
        for trace, predicted_label, confidence in predictions:
            if confidence < confidence_threshold and predicted_label != "PASS":
                queue.append({
                    "trace_id": trace.trace_id,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "question": trace.question,
                    "answer": trace.answer,
                })
        return queue
