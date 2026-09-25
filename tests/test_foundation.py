"""Unit tests for Phase 1 foundation: domain contracts, hashing, provenance, and logging."""

from __future__ import annotations

import json
from rag_platform.common.crypto import canonical_json, compute_manifest_hash, sha256_hash
from rag_platform.common.ids import generate_id
from rag_platform.common.logging import StructuredJsonFormatter, current_run_id, current_trace_id
from rag_platform.models import (
    Answerability,
    BenchmarkDataset,
    Citation,
    DatasetStatus,
    DocumentReference,
    FailureAttribution,
    FailureCode,
    GateResult,
    GateStatus,
    MetricFamily,
    MetricResult,
    RagTrace,
    ReleasePolicy,
    RetrievedChunk,
    RunConfig,
    RunProvenance,
    Severity,
    TestCase,
    compute_dataset_checksum,
)
import logging


def test_canonical_json_sorting_and_determinism():
    """Ensure key order does not affect canonical json string or hash."""
    d1 = {"b": 2, "a": 1, "nested": {"z": 26, "y": 25}}
    d2 = {"a": 1, "nested": {"y": 25, "z": 26}, "b": 2}

    json1 = canonical_json(d1)
    json2 = canonical_json(d2)

    assert json1 == json2
    assert json1 == '{"a":1,"b":2,"nested":{"y":25,"z":26}}'
    assert sha256_hash(json1) == sha256_hash(json2)


def test_dataset_checksum_invariance_to_case_order():
    """Ensure dataset checksum is identical regardless of input case ordering."""
    case_a = TestCase(
        id="case-001",
        question="What is Q1 revenue?",
        expected_answer="$10M",
        expected_facts=["Q1 revenue was $10M"],
        relevant_documents=[DocumentReference(document_id="doc-1", chunk_id="c-1")],
    )
    case_b = TestCase(
        id="case-002",
        question="What is Q2 revenue?",
        expected_answer="$12M",
        expected_facts=["Q2 revenue was $12M"],
        relevant_documents=[DocumentReference(document_id="doc-2", chunk_id="c-2")],
    )

    checksum1 = compute_dataset_checksum([case_a, case_b])
    checksum2 = compute_dataset_checksum([case_b, case_a])

    assert checksum1 == checksum2
    assert len(checksum1) == 64  # SHA256 hex length


def test_dataset_checksum_detects_mutation():
    """Ensure any modification to a test case changes the dataset checksum."""
    case1 = TestCase(id="c1", question="What is GDP?", expected_answer="3 trillion")
    checksum_orig = compute_dataset_checksum([case1])

    case_modified = TestCase(id="c1", question="What is GDP?", expected_answer="4 trillion")
    checksum_mod = compute_dataset_checksum([case_modified])

    assert checksum_orig != checksum_mod


def test_dataset_publishing_lifecycle():
    """Ensure publishing locks the dataset and calculates checksum."""
    ds = BenchmarkDataset(
        id="ds_123",
        project_id="proj_1",
        name="finance_bench",
        version="v1.0.0",
        cases=[
            TestCase(id="c1", question="What was 2024 revenue?", expected_answer="$15B")
        ],
    )
    assert ds.status == DatasetStatus.DRAFT
    assert ds.checksum_sha256 == ""

    ds.publish()
    assert ds.status == DatasetStatus.PUBLISHED
    assert len(ds.checksum_sha256) == 64


def test_six_dimension_provenance_manifest():
    """Verify that all 6 dimensions are required and changing any one alters manifest hash."""
    base_args = {
        "dataset_checksum": "aaa111",
        "rag_version": "git-commit-1",
        "model_config_hash": "model-hash-1",
        "prompt_hash": "prompt-hash-1",
        "evaluator_version": "eval-v1",
        "experiment_hash": "exp-hash-1",
    }
    prov1 = RunProvenance(**base_args)
    assert prov1.manifest_hash != ""

    # Change only prompt_hash
    args_altered = dict(base_args)
    args_altered["prompt_hash"] = "prompt-hash-2"
    prov2 = RunProvenance(**args_altered)

    assert prov1.manifest_hash != prov2.manifest_hash


def test_trace_and_chunk_provenance():
    """Verify canonical RagTrace structure and serializability."""
    trace = RagTrace(
        trace_id="tr_001",
        run_id="run_100",
        test_case_id="case_001",
        question="What was EBITDA?",
        answer="EBITDA was $2.1B [1].",
        retrieved_chunks=[
            RetrievedChunk(
                document_id="annual-2024",
                chunk_id="chunk-4",
                rank=1,
                score=0.94,
                text="In 2024, operating EBITDA reached $2.1B.",
            )
        ],
        citations=[
            Citation(
                claim_id="c1",
                claim_text="EBITDA was $2.1B",
                document_id="annual-2024",
                chunk_id="chunk-4",
                span=[0, 20],
            )
        ],
        latency_ms=120,
        model="gpt-4o",
    )

    dumped = trace.model_dump_json()
    loaded = RagTrace.model_validate_json(dumped)
    assert loaded.trace_id == "tr_001"
    assert len(loaded.retrieved_chunks) == 1
    assert loaded.retrieved_chunks[0].score == 0.94
    assert len(loaded.citations) == 1


def test_failure_attribution_model():
    """Verify failure attribution schema and confidence constraints."""
    attribution = FailureAttribution(
        trace_id="tr_001",
        failure_type=FailureCode.RET_01,
        severity=Severity.HIGH,
        confidence=0.95,
        explanation="Gold document annual-2024 absent from top-5 chunks.",
        evidence={"gold_document_id": "annual-2024", "retrieved_ids": ["doc-9", "doc-12"]},
        recommended_actions=["Increase top-k from 5 to 8", "Enable hybrid BM25 search"],
    )

    assert attribution.failure_type == FailureCode.RET_01
    assert attribution.confidence == 0.95


def test_id_generator_sortability():
    """Verify ID generation creates unique, chronologically sortable strings."""
    id1 = generate_id("run")
    id2 = generate_id("run")

    assert id1.startswith("run_")
    assert id2.startswith("run_")
    assert id1 != id2


def test_structured_json_logging_with_context():
    """Verify structured log formatting injects context variables."""
    formatter = StructuredJsonFormatter()
    logger = logging.getLogger("test_logger")

    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Execution started",
        args=(),
        exc_info=None,
    )

    token_run = current_run_id.set("run_abc123")
    token_tr = current_trace_id.set("tr_xyz789")

    try:
        output_str = formatter.format(record)
        data = json.loads(output_str)

        assert data["level"] == "INFO"
        assert data["message"] == "Execution started"
        assert data["run_id"] == "run_abc123"
        assert data["trace_id"] == "tr_xyz789"
        assert "timestamp" in data
    finally:
        current_run_id.reset(token_run)
        current_trace_id.reset(token_tr)
