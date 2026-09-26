"""Unit tests for Phase 1 foundation: domain contracts, hashing, provenance."""

from __future__ import annotations

from rag_platform.core import (
    canonical_json,
    generate_id,
)
from rag_platform.models import (
    BenchmarkDataset,
    Citation,
    DatasetStatus,
    FailureAttribution,
    FailureCode,
    RagTrace,
    RetrievedChunk,
    RunProvenance,
    Severity,
    TestCase,
    compute_dataset_checksum,
)


def test_canonical_json_sorting_and_determinism():
    d1 = {"b": 2, "a": 1, "nested": {"z": 26, "y": 25}}
    d2 = {"a": 1, "nested": {"y": 25, "z": 26}, "b": 2}
    assert canonical_json(d1) == canonical_json(d2)
    assert canonical_json(d1) == '{"a":1,"b":2,"nested":{"y":25,"z":26}}'


def test_dataset_checksum_invariance_to_case_order():
    case_a = TestCase(id="case-001", question="What is Q1 revenue?", expected_answer="$10M")
    case_b = TestCase(id="case-002", question="What is Q2 revenue?", expected_answer="$12M")
    assert compute_dataset_checksum([case_a, case_b]) == compute_dataset_checksum([case_b, case_a])


def test_dataset_checksum_detects_mutation():
    case1 = TestCase(id="c1", question="What is GDP?", expected_answer="3 trillion")
    case_mod = TestCase(id="c1", question="What is GDP?", expected_answer="4 trillion")
    assert compute_dataset_checksum([case1]) != compute_dataset_checksum([case_mod])


def test_dataset_publishing_lifecycle():
    ds = BenchmarkDataset(
        id="ds_123",
        project_id="proj_1",
        name="finance_bench",
        version="v1.0.0",
        cases=[TestCase(id="c1", question="Revenue?", expected_answer="$15B")],
    )
    assert ds.status == DatasetStatus.DRAFT
    ds.publish()
    assert ds.status == DatasetStatus.PUBLISHED
    assert len(ds.checksum_sha256) == 64


def test_six_dimension_provenance_manifest():
    base_args = {
        "dataset_checksum": "aaa111",
        "rag_version": "git-commit-1",
        "model_config_hash": "model-hash-1",
        "prompt_hash": "prompt-hash-1",
        "evaluator_version": "eval-v1",
        "experiment_hash": "exp-hash-1",
    }
    prov1 = RunProvenance(**base_args)
    args_altered = dict(base_args, prompt_hash="prompt-hash-2")
    prov2 = RunProvenance(**args_altered)
    assert prov1.manifest_hash != prov2.manifest_hash


def test_trace_and_chunk_provenance():
    trace = RagTrace(
        trace_id="tr_001",
        run_id="run_100",
        test_case_id="case_001",
        question="What was EBITDA?",
        answer="EBITDA was $2.1B [1].",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc-1", chunk_id="c-4", rank=1, score=0.94, text="EBITDA: $2.1B")
        ],
        citations=[
            Citation(claim_id="c1", claim_text="EBITDA was $2.1B", document_id="doc-1", chunk_id="c-4")
        ],
    )
    dumped = trace.model_dump_json()
    loaded = RagTrace.model_validate_json(dumped)
    assert loaded.trace_id == "tr_001"
    assert loaded.retrieved_chunks[0].score == 0.94


def test_failure_attribution_model():
    attribution = FailureAttribution(
        trace_id="tr_001",
        failure_type=FailureCode.RET_01,
        severity=Severity.HIGH,
        confidence=0.95,
        explanation="Gold document absent from top-5.",
    )
    assert attribution.failure_type == FailureCode.RET_01


def test_id_generator_sortability():
    id1 = generate_id("run")
    id2 = generate_id("run")
    assert id1.startswith("run_")
    assert id1 != id2
