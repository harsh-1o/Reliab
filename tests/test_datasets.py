"""Unit tests for Phase 6: benchmark dataset import/export, adversarial generation, and splits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_platform.datasets import (
    AdversarialGenerator,
    export_dataset_csv,
    export_dataset_jsonl,
    import_dataset_csv,
    import_dataset_jsonl,
    split_benchmark_dataset,
)
from rag_platform.models import (
    Answerability,
    BenchmarkDataset,
    DocumentReference,
    TestCase,
)


@pytest.fixture
def sample_dataset() -> BenchmarkDataset:
    cases = [
        TestCase(
            id=f"case_{i}",
            question=f"Question {i}?",
            expected_answer=f"Answer {i}",
            expected_facts=[f"Fact {i}a", f"Fact {i}b"],
            relevant_documents=[DocumentReference(document_id=f"doc_{i}")],
            tags=["finance", "q3"],
        )
        for i in range(10)
    ]
    return BenchmarkDataset(
        id="ds_test",
        project_id="proj_1",
        name="finance_master",
        version="v1.0",
        cases=cases,
    ).publish()


def test_jsonl_roundtrip_export_import(sample_dataset: BenchmarkDataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "benchmark.jsonl"
        export_dataset_jsonl(sample_dataset, path)
        assert path.exists()

        imported = import_dataset_jsonl(path, "proj_1", "finance_imported", "v1.0")
        assert len(imported.cases) == len(sample_dataset.cases)
        assert imported.cases[0].id == sample_dataset.cases[0].id
        assert imported.checksum_sha256 == sample_dataset.checksum_sha256


def test_csv_roundtrip_export_import(sample_dataset: BenchmarkDataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "benchmark.csv"
        export_dataset_csv(sample_dataset, path)
        assert path.exists()

        imported = import_dataset_csv(path, "proj_1", "finance_csv", "v1.0")
        assert len(imported.cases) == len(sample_dataset.cases)
        assert imported.cases[1].expected_answer == "Answer 1"
        assert imported.cases[1].relevant_documents[0].document_id == "doc_1"


def test_adversarial_generators(sample_dataset: BenchmarkDataset):
    base_case = sample_dataset.cases[0]

    # Unanswerable attack
    adv_unans = AdversarialGenerator.make_unanswerable(base_case, "warp drive engine")
    assert adv_unans.answerability == Answerability.UNANSWERABLE
    assert adv_unans.expected_answer is None
    assert "warp drive" in adv_unans.question

    # Distractor attack
    adv_dist = AdversarialGenerator.make_distractor_case(base_case)
    assert adv_dist.answerability == Answerability.ANSWERABLE
    assert adv_dist.expected_answer == base_case.expected_answer
    assert "distractor" in adv_dist.tags


def test_deterministic_benchmark_split(sample_dataset: BenchmarkDataset):
    dev1, val1, test1 = split_benchmark_dataset(sample_dataset, dev_ratio=0.2, val_ratio=0.2, test_ratio=0.6, seed=42)
    dev2, val2, test2 = split_benchmark_dataset(sample_dataset, dev_ratio=0.2, val_ratio=0.2, test_ratio=0.6, seed=42)

    assert len(dev1.cases) == 2
    assert len(val1.cases) == 2
    assert len(test1.cases) == 6

    # Verify seed reproducibility
    assert dev1.checksum_sha256 == dev2.checksum_sha256
    assert test1.checksum_sha256 == test2.checksum_sha256
