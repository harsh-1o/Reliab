"""Regression tests for reliability hardening fixes."""

from __future__ import annotations

import csv

import pytest

from rag_platform.core import _resolve_database_url
from rag_platform.datasets import export_dataset_csv, import_dataset_csv, split_benchmark_dataset
from rag_platform.models import Answerability, BenchmarkDataset, TestCase


def _dataset(case_count: int = 3) -> BenchmarkDataset:
    return BenchmarkDataset(
        id="ds_hardening",
        project_id="proj_hardening",
        name="hardening",
        version="1",
        cases=[
            TestCase(
                id=f"case_{i}",
                question=f"Question {i}",
                expected_answer=f"Answer {i}",
                expected_facts=[f"fact-{i}"],
                answerability=Answerability.ANSWERABLE,
                tags=["regression"],
                metadata={"source": "test", "ordinal": i},
            )
            for i in range(case_count)
        ],
    )


def test_production_database_configuration_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        _resolve_database_url()


def test_production_sqlite_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./unsafe.db")
    with pytest.raises(RuntimeError, match="SQLite"):
        _resolve_database_url()


def test_csv_round_trip_preserves_metadata(tmp_path) -> None:
    original = _dataset(3)
    path = tmp_path / "dataset.csv"
    export_dataset_csv(original, path)

    imported = import_dataset_csv(path, "proj_hardening", "roundtrip", "1")
    assert [c.metadata for c in imported.cases] == [c.metadata for c in original.cases]
    assert [c.tags for c in imported.cases] == [c.tags for c in original.cases]


def test_small_dataset_split_never_creates_empty_partition() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        split_benchmark_dataset(_dataset(2))

    dev, val, test = split_benchmark_dataset(_dataset(3))
    assert len(dev.cases) == 1
    assert len(val.cases) == 1
    assert len(test.cases) == 1


def test_csv_contains_metadata_column(tmp_path) -> None:
    path = tmp_path / "dataset.csv"
    export_dataset_csv(_dataset(), path)
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "metadata_json" in header
