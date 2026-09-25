"""Domain enums and taxonomy codes."""

from enum import Enum


class RunStatus(str, Enum):
    """Evaluation run lifecycle states."""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SCORING = "SCORING"
    ATTRIBUTING = "ATTRIBUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DatasetStatus(str, Enum):
    """Dataset version lifecycle status."""

    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


class Answerability(str, Enum):
    """Expected answerability label for benchmark test cases."""

    ANSWERABLE = "ANSWERABLE"
    UNANSWERABLE = "UNANSWERABLE"


class MetricFamily(str, Enum):
    """Metric classification family."""

    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"
    CITATION = "CITATION"
    ABSTENTION = "ABSTENTION"
    SYSTEM = "SYSTEM"


class FailureCode(str, Enum):
    """Diagnostic failure taxonomy codes."""

    RET_01 = "RET-01"  # Retrieval miss: gold chunk absent from top-K
    RET_02 = "RET-02"  # Bad ranking: relevant chunk buried below distractors
    GEN_01 = "GEN-01"  # Unsupported claim: answer contains claims absent from context
    GEN_02 = "GEN-02"  # Contradiction: answer contradicts retrieved context
    CIT_01 = "CIT-01"  # Mis-citation: citation points to incorrect passage
    CIT_02 = "CIT-02"  # Missing citation: external claim without citation annotation
    KNW_01 = "KNW-01"  # Knowledge gap: corpus lacks evidence, system should abstain
    ABS_01 = "ABS-01"  # Abstention failure: unanswerable question answered with hallucination
    NUM_01 = "NUM-01"  # Numerical reasoning: arithmetic or calculation mismatch
    ENT_01 = "ENT-01"  # Entity confusion: facts confused between similar entities
    OPS_01 = "OPS-01"  # Infrastructure error: timeout, rate limit, worker crash


class Severity(str, Enum):
    """Failure attribution severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GateStatus(str, Enum):
    """CI/CD release quality gate decision."""

    PASS = "PASS"
    FAIL = "FAIL"
