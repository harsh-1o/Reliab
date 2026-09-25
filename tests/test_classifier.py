"""Unit tests for Phase 10: ML failure diagnosis classifier, feature extractor, and active learning queue."""

from __future__ import annotations

import pytest

from rag_platform.classifier import FeatureExtractor, MLFailureClassifier
from rag_platform.models import (
    Answerability,
    DocumentReference,
    MetricFamily,
    MetricResult,
    RagTrace,
    RetrievedChunk,
    TestCase,
)


@pytest.fixture
def sample_trace_and_case():
    case = TestCase(
        id="c1",
        question="What was 2024 revenue?",
        expected_answer="$10B",
        relevant_documents=[DocumentReference(document_id="doc1")],
    )
    trace = RagTrace(
        trace_id="tr1",
        run_id="run1",
        test_case_id="c1",
        question="What was 2024 revenue?",
        answer="$10B",
        latency_ms=150,
        retrieved_chunks=[RetrievedChunk(document_id="doc1", chunk_id="p1", rank=1, score=0.95, text="text")],
    )
    metrics = [
        MetricResult(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, score=1.0),
        MetricResult(metric_name="recall_at_5", metric_family=MetricFamily.RETRIEVAL, score=1.0),
    ]
    return trace, case, metrics


def test_feature_extractor(sample_trace_and_case):
    trace, case, metrics = sample_trace_and_case
    features = FeatureExtractor.extract(trace, case, metrics)

    assert len(features) == len(FeatureExtractor.FEATURE_NAMES)
    assert features[2] == 0.0  # abstained = False
    assert features[3] == 1.0  # num_chunks = 1
    assert features[7] == 1.0  # faithfulness = 1.0


def test_ml_classifier_training_and_prediction():
    classifier = MLFailureClassifier()

    # Synthetic training samples: [features], label
    # 0: PASS, 1: RET-01, 2: GEN-01, 3: OPS-01
    X_train = [
        # PASS: high faithfulness (1.0), high recall (1.0), low latency, no infra error
        [30, 20, 0, 5, 0.9, 0.1, 120, 1.0, 1.0, 1.0, 1, 0, 0],
        [25, 15, 0, 4, 0.85, 0.1, 110, 0.95, 1.0, 0.9, 1, 0, 0],
        # RET-01: recall = 0.0, top score = 0.2
        [30, 20, 0, 5, 0.2, 0.05, 150, 0.9, 0.0, 0.5, 0, 0, 0],
        [40, 30, 0, 3, 0.15, 0.02, 160, 0.8, 0.0, 0.4, 0, 0, 0],
        # GEN-01: faithfulness = 0.1, recall = 1.0
        [30, 50, 0, 5, 0.9, 0.1, 140, 0.1, 1.0, 0.1, 0, 0, 0],
        [35, 60, 0, 4, 0.88, 0.1, 130, 0.2, 1.0, 0.15, 0, 0, 0],
        # OPS-01: infra error = 1.0, latency = 5000
        [30, 0, 0, 0, 0.0, 0.0, 5000, 0.0, 0.0, 0.0, 0, 1, 0],
        [28, 0, 0, 0, 0.0, 0.0, 5000, 0.0, 0.0, 0.0, 0, 1, 0],
    ]
    y_train = ["PASS", "PASS", "RET-01", "RET-01", "GEN-01", "GEN-01", "OPS-01", "OPS-01"]

    classifier.train(X_train, y_train)
    assert classifier.is_fitted is True

    # Test evaluation
    eval_res = classifier.evaluate_model(X_train, y_train)
    assert eval_res["macro_f1"] >= 0.80

    # Test prediction on passing trace
    test_case = TestCase(id="t1", question="What was revenue?")
    pass_trace = RagTrace(trace_id="tr1", run_id="r1", test_case_id="t1", question="What was revenue?", answer="$10B")
    pred_label, conf = classifier.predict(pass_trace, test_case, [
        MetricResult(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, score=1.0),
        MetricResult(metric_name="recall_at_5", metric_family=MetricFamily.RETRIEVAL, score=1.0),
    ])
    assert 0.0 <= conf <= 1.0


def test_active_learning_review_queue():
    t_low_conf = RagTrace(trace_id="tr_ambiguous", run_id="r1", test_case_id="c1", question="Q", answer="Ans")
    t_high_conf = RagTrace(trace_id="tr_clear", run_id="r1", test_case_id="c2", question="Q2", answer="Ans2")

    predictions = [
        (t_low_conf, "GEN-01", 0.55),   # Low confidence -> should be flagged
        (t_high_conf, "GEN-01", 0.95),  # High confidence -> excluded
    ]

    queue = MLFailureClassifier.filter_active_learning_queue(predictions, confidence_threshold=0.70)
    assert len(queue) == 1
    assert queue[0]["trace_id"] == "tr_ambiguous"
    assert queue[0]["confidence"] == 0.55
