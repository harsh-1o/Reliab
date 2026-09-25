"""Benchmark dataset management, multi-format import/export, and adversarial attack generation.

# ponytail: single file covers JSONL/CSV import/export, dataset splitting, and attack generators.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from rag_platform.core import generate_id
from rag_platform.models import (
    Answerability,
    BenchmarkDataset,
    DatasetStatus,
    DocumentReference,
    TestCase,
)


# --- Import / Export ---
def export_dataset_jsonl(dataset: BenchmarkDataset, target_path: str | Path) -> None:
    """Export benchmark dataset cases to a JSONL file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in dataset.cases:
            f.write(case.model_dump_json() + "\n")


def import_dataset_jsonl(
    source_path: str | Path,
    project_id: str,
    name: str,
    version: str,
    description: str | None = None,
) -> BenchmarkDataset:
    """Import benchmark dataset from a JSONL file."""
    path = Path(source_path)
    cases: list[TestCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(TestCase.model_validate_json(line))

    ds = BenchmarkDataset(
        id=generate_id("ds"),
        project_id=project_id,
        name=name,
        version=version,
        description=description,
        cases=cases,
    )
    return ds.publish()


def export_dataset_csv(dataset: BenchmarkDataset, target_path: str | Path) -> None:
    """Export benchmark dataset cases to a flat CSV file."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "expected_answer", "expected_facts", "doc_ids", "answerability", "tags"])
        for c in dataset.cases:
            writer.writerow([
                c.id,
                c.question,
                c.expected_answer or "",
                ";".join(c.expected_facts),
                ";".join(d.document_id for d in c.relevant_documents),
                c.answerability.value,
                ";".join(c.tags),
            ])


def import_dataset_csv(
    source_path: str | Path,
    project_id: str,
    name: str,
    version: str,
    description: str | None = None,
) -> BenchmarkDataset:
    """Import benchmark dataset from a CSV file."""
    path = Path(source_path)
    cases: list[TestCase] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_ids = [d.strip() for d in row.get("doc_ids", "").split(";") if d.strip()]
            facts = [ft.strip() for ft in row.get("expected_facts", "").split(";") if ft.strip()]
            tags = [t.strip() for t in row.get("tags", "").split(";") if t.strip()]

            cases.append(
                TestCase(
                    id=row["id"],
                    question=row["question"],
                    expected_answer=row.get("expected_answer") or None,
                    expected_facts=facts,
                    relevant_documents=[DocumentReference(document_id=did) for did in doc_ids],
                    answerability=Answerability(row.get("answerability", "ANSWERABLE")),
                    tags=tags,
                )
            )

    ds = BenchmarkDataset(
        id=generate_id("ds"),
        project_id=project_id,
        name=name,
        version=version,
        description=description,
        cases=cases,
    )
    return ds.publish()


# --- Adversarial Attack Generator ---
class AdversarialGenerator:
    """Generates synthetic adversarial test cases while preserving provenance."""

    @staticmethod
    def make_unanswerable(case: TestCase, missing_subject: str = "quantum teleportation battery") -> TestCase:
        """Create an adversarial unanswerable case that requires strict abstention."""
        return TestCase(
            id=f"{case.id}_adv_unans",
            question=f"Regarding {missing_subject}, {case.question.lower()}",
            expected_answer=None,
            expected_facts=[],
            relevant_documents=[],
            answerability=Answerability.UNANSWERABLE,
            tags=sorted(list(set(case.tags + ["adversarial", "unanswerable"]))),
            metadata={"source_case_id": case.id, "attack_type": "unanswerable"},
        )

    @staticmethod
    def make_distractor_case(case: TestCase, distractor_topic: str = "unrelated solar flare impact") -> TestCase:
        """Create a case testing robustness against semantic distractors."""
        return TestCase(
            id=f"{case.id}_adv_distractor",
            question=f"{case.question} (Ignore any reports regarding {distractor_topic}).",
            expected_answer=case.expected_answer,
            expected_facts=case.expected_facts,
            relevant_documents=case.relevant_documents,
            answerability=case.answerability,
            tags=sorted(list(set(case.tags + ["adversarial", "distractor"]))),
            metadata={"source_case_id": case.id, "attack_type": "distractor"},
        )

    @staticmethod
    def make_citation_trap(case: TestCase) -> TestCase:
        """Create a case testing whether citation checker detects swapped chunk references."""
        return TestCase(
            id=f"{case.id}_adv_citation_trap",
            question=case.question,
            expected_answer=case.expected_answer,
            expected_facts=case.expected_facts,
            relevant_documents=case.relevant_documents,
            answerability=case.answerability,
            tags=sorted(list(set(case.tags + ["adversarial", "citation_trap"]))),
            metadata={"source_case_id": case.id, "attack_type": "citation_trap"},
        )


# --- Deterministic Benchmark Splits ---
def split_benchmark_dataset(
    dataset: BenchmarkDataset,
    dev_ratio: float = 0.20,
    val_ratio: float = 0.20,
    test_ratio: float = 0.60,
    seed: int = 42,
) -> tuple[BenchmarkDataset, BenchmarkDataset, BenchmarkDataset]:
    """Split a dataset deterministically into (dev, val, locked_test) sets."""
    cases = list(dataset.cases)
    rng = random.Random(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_dev = int(n * dev_ratio)
    n_val = int(n * val_ratio)

    dev_cases = cases[:n_dev]
    val_cases = cases[n_dev : n_dev + n_val]
    test_cases = cases[n_dev + n_val :]

    dev_ds = BenchmarkDataset(
        id=f"{dataset.id}_dev",
        project_id=dataset.project_id,
        name=f"{dataset.name}_dev",
        version=dataset.version,
        cases=dev_cases,
    ).publish()

    val_ds = BenchmarkDataset(
        id=f"{dataset.id}_val",
        project_id=dataset.project_id,
        name=f"{dataset.name}_val",
        version=dataset.version,
        cases=val_cases,
    ).publish()

    test_ds = BenchmarkDataset(
        id=f"{dataset.id}_locked_test",
        project_id=dataset.project_id,
        name=f"{dataset.name}_locked_test",
        version=dataset.version,
        cases=test_cases,
    ).publish()

    return dev_ds, val_ds, test_ds
