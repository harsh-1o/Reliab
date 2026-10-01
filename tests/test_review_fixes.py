import asyncio

import pytest

from rag_platform.adapters import HttpRagResponsePayload
from rag_platform.evaluators import CitationSupportMetric, RecallAtKMetric
from rag_platform.models import (
    Citation,
    MetricFamily,
    MetricResult,
    MetricSummary,
    RagTrace,
    RetrievedChunk,
    RunMetricsSummary,
    RunProvenance,
    TestCase,
)
from rag_platform.regression import RegressionEngine
from rag_platform.gate import resolve_release_policy


def _summary(case_count: int = 2) -> RunMetricsSummary:
    metric = MetricSummary(
        metric_name="faithfulness",
        metric_family=MetricFamily.GENERATION,
        mean=1.0,
        p50=1.0,
        p95=1.0,
        min=1.0,
        max=1.0,
        count=case_count,
        applicable_count=case_count,
    )
    return RunMetricsSummary(
        metrics={"faithfulness": metric},
        total_cases=case_count,
        scored_cases=case_count,
        required_case_count=case_count,
        evaluated_case_count=case_count,
        coverage_ratio=1.0,
        is_full_evaluation=True,
        eligible_for_release_gate=True,
    )


def test_regression_distinguishes_missing_case_from_unmeasured_metric() -> None:
    result = RegressionEngine().compare(
        _summary(),
        _summary(),
        "baseline",
        "candidate",
        baseline_case_scores={"a": 1.0, "b": None},
        candidate_case_scores={"a": 1.0, "b": None},
        baseline_case_ids={"a", "b"},
        candidate_case_ids={"a", "b"},
    )

    assert result.candidate_missing_cases == []
    assert result.baseline_missing_cases == []
    assert result.case_transitions["b"]["transition"] == "NOT_APPLICABLE"


def test_regression_detects_actual_candidate_case_omission() -> None:
    result = RegressionEngine().compare(
        _summary(),
        _summary(),
        "baseline",
        "candidate",
        baseline_case_scores={"a": 1.0, "b": 0.8},
        candidate_case_scores={"a": 1.0},
        baseline_case_ids={"a", "b"},
        candidate_case_ids={"a"},
    )

    assert result.candidate_missing_cases == ["b"]
    assert result.case_transitions["b"]["transition"] == "CANDIDATE_MISSING"


def test_provenance_nested_inputs_are_immutable() -> None:
    provenance = RunProvenance(
        dataset_checksum="a" * 64,
        model_parameters={"temperature": 0.2, "nested": {"enabled": True}},
        environment_info={"labels": ["ci"]},
    )

    with pytest.raises(TypeError):
        provenance.model_parameters["temperature"] = 0.9

    with pytest.raises(TypeError):
        provenance.model_parameters["nested"]["enabled"] = False

    with pytest.raises(TypeError):
        provenance.environment_info["labels"].append("prod")

    assert provenance.manifest_hash == provenance.compute_hash()


def test_provenance_rejects_incorrect_supplied_manifest_hash() -> None:
    with pytest.raises(ValueError, match="manifest_hash"):
        RunProvenance(
            dataset_checksum="a" * 64,
            manifest_hash="b" * 64,
        )


def test_recall_uses_chunk_rank_for_top_k() -> None:
    case = TestCase(
        id="case",
        question="q",
        relevant_documents=[{"document_id": "doc-good", "chunk_id": "good"}],
    )
    trace = RagTrace(
        trace_id="t",
        run_id="r",
        test_case_id="case",
        question="q",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc-noise", chunk_id="noise", rank=5, text="noise"),
            RetrievedChunk(document_id="doc-good", chunk_id="good", rank=1, text="good"),
        ],
    )

    result = asyncio.run(RecallAtKMetric(k=1).compute(trace, case))
    assert result.score == 1.0


def test_contradicted_citation_is_not_rescued_by_lexical_fallback() -> None:
    case = TestCase(id="case", question="q")
    trace = RagTrace(
        trace_id="t",
        run_id="r",
        test_case_id="case",
        question="q",
        answer="Revenue increased.",
        retrieved_chunks=[
            RetrievedChunk(
                document_id="doc",
                chunk_id="c1",
                rank=1,
                text="Revenue decreased significantly and revenue increased is false.",
            )
        ],
        citations=[
            Citation(
                claim_id="cl1",
                claim_text="Revenue increased.",
                document_id="doc",
                chunk_id="c1",
            )
        ],
    )

    result = asyncio.run(CitationSupportMetric().compute(trace, case))
    assert result.score == 0.0


def test_http_citation_span_contract() -> None:
    with pytest.raises(Exception):
        HttpRagResponsePayload.model_validate(
            {
                "answer": "ok",
                "citations": [
                    {
                        "claim_text": "claim",
                        "document_id": "doc",
                        "chunk_id": "c",
                        "span": [10, 2],
                    }
                ],
            }
        )


def test_release_policy_resolution_does_not_silently_default_unknown_policy() -> None:
    assert resolve_release_policy("prod-default").policy_id == "prod-default"
    with pytest.raises(ValueError, match="Unknown release policy"):
        resolve_release_policy("does-not-exist")


def test_aggregate_run_deduplicates_case_ids() -> None:
    from rag_platform.evaluators import EvaluationEngine

    trace = RagTrace(
        trace_id="t1",
        run_id="r",
        test_case_id="case",
        question="q",
        answer="answer",
    )
    metrics = [MetricResult(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, score=1.0)]
    duplicate = trace.model_copy(update={"trace_id": "t2"})
    summary = EvaluationEngine().aggregate_run(
        [(trace, metrics), (duplicate, metrics)],
        required_case_count=2,
    )
    assert summary.evaluated_case_count == 1
    assert summary.missing_case_count == 1
    assert summary.coverage_ratio == 0.5
    assert summary.metrics["faithfulness"].count == 1
