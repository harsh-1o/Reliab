"""Metric evaluation engine: claim-level faithfulness, fact-anchored correctness,
evidence retrieval, citation validation, and statistical confidence intervals.
"""

from __future__ import annotations

import math
import re
import statistics
from abc import ABC, abstractmethod

from rag_platform.core import canonical_json, sha256_hash
from rag_platform.models import (
    Answerability,
    ClaimStatus,
    ClaimVerification,
    DocumentReference,
    MetricFamily,
    MetricResult,
    MetricStatus,
    MetricSummary,
    RagTrace,
    RetrievedChunk,
    RunMetricsSummary,
    TestCase,
)

EQUIVALENCE_MAPPINGS: list[tuple[str, str]] = [
    (r"\bone month\b", "30 days"),
    (r"\b1 month\b", "30 days"),
    (r"\btwo months\b", "60 days"),
    (r"\b2 months\b", "60 days"),
    (r"\bone year\b", "12 months"),
    (r"\b1 year\b", "12 months"),
    (r"\bannually\b", "yearly"),
    (r"\bannual\b", "yearly"),
    (r"\bpermitted\b", "allowed"),
    (r"\bcan receive\b", "allowed"),
    (r"\beligible for\b", "allowed"),
    (r"\bprohibited\b", "forbidden"),
    (r"\bcustomers\b", "clients"),
    (r"\busers\b", "clients"),
    (r"\bconsumers\b", "clients"),
]


def normalize_text_equivalences(text: str) -> str:
    """Normalize common semantic synonyms, duration phrasing, and entity aliases."""
    normalized = text.lower()
    for pattern, replacement in EQUIVALENCE_MAPPINGS:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _decompose_sentence_into_clauses(sentence: str) -> list[str]:
    """Decompose a complex sentence into atomic factual clauses."""
    s = sentence.strip()
    if not s:
        return []

    # Semicolons are strong proposition separators
    raw_subparts = [p.strip() for p in s.split(";") if p.strip()]
    clauses: list[str] = []

    for part in raw_subparts:
        # Check for serial comma or coordinating conjunction clause boundaries
        # e.g., "Revenue increased 20%, profit increased 12%, and headcount increased by 500."
        # or "Revenue increased 20% and profit increased 15%."
        candidate_splits = re.split(
            r"(?:,\s+(?:and\s+|but\s+|while\s+|whereas\s+|as well as\s+)?|\s+(?:and|but|while|whereas)\s+)",
            part,
            flags=re.IGNORECASE,
        )
        if len(candidate_splits) > 1:
            valid_clauses: list[str] = []
            for c in candidate_splits:
                c_clean = c.strip().rstrip(".!?")
                c_clean = re.sub(r"^(?:and|or|but|also|as well as)\s+", "", c_clean, flags=re.IGNORECASE).strip()
                # A factual clause typically has numbers/symbols or at least 2 content tokens
                if len(c_clean) > 5 and (re.search(r"\b\d+|\$|%", c_clean) or len(c_clean.split()) >= 2):
                    # Capitalize first character
                    c_formatted = c_clean[0].upper() + c_clean[1:] if len(c_clean) > 1 else c_clean.upper()
                    valid_clauses.append(c_formatted)
                elif valid_clauses:
                    # Merge fragment back into previous clause
                    valid_clauses[-1] = f"{valid_clauses[-1]} and {c_clean}"

            if len(valid_clauses) > 1:
                for vc in valid_clauses:
                    clauses.append(vc if vc.endswith((".", "!", "?")) else f"{vc}.")
                continue

        # Single clause fallback
        cleaned = part.rstrip(".!?").strip()
        if cleaned:
            formatted = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
            clauses.append(formatted if formatted.endswith((".", "!", "?")) else f"{formatted}.")

    return clauses


def extract_claims(text: str) -> list[str]:
    """Extract atomic factual claim units from answer text.

    Decomposes paragraphs and compound sentences into atomic declarative clauses,
    filtering conversational padding and meta-discourse.
    """
    if not text or not text.strip():
        return []

    # Clean bracketed citation tags [1], [doc-1] for pure claim analysis
    clean_text = re.sub(r"\[\s*[\w\-]+\s*\]", "", text).strip()

    # Split lines and bullet points
    lines = [
        re.sub(r"^[-*•\d+\.]\s+", "", line).strip()
        for line in clean_text.splitlines()
        if line.strip()
    ]

    fillers = (
        "based on the provided context",
        "according to the documents",
        "as stated in the text",
        "in conclusion",
        "to summarize",
        "thank you",
        "here is the answer",
    )

    claims: list[str] = []

    for line in lines:
        # Split into sentences
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", line)
            if len(s.strip()) > 5
        ]

        for s in raw_sentences:
            lower_s = s.lower().strip()
            if any(lower_s == f for f in fillers):
                continue
            for f in fillers:
                if lower_s.startswith(f + ",") or lower_s.startswith(f + ":"):
                    s = s[len(f) + 1 :].strip()
                    break

            decomposed = _decompose_sentence_into_clauses(s)
            for c in decomposed:
                if len(c) > 6:
                    claims.append(c)

    return claims if claims else ([clean_text] if len(clean_text) > 8 else [])


def verify_claim_against_chunks(
    claim: str, chunks: list[RetrievedChunk]
) -> tuple[ClaimStatus, RetrievedChunk | None, str]:
    """Verify an individual claim against retrieved context chunks using Lexical Claim Grounding.

    Returns (ClaimStatus, supporting_chunk, explanation).
    Classifies as:
      - SUPPORTED: Key entities, predicates, and semantic alignments confirmed in evidence chunk.
      - CONTRADICTED: Directly conflicts with numerical amounts, explicit negation, or opposing predicates.
      - UNSUPPORTED: Facts not substantiated by any retrieved context chunk.
    """
    claim_norm = normalize_text_equivalences(claim)
    claim_lower = claim_norm.lower()
    claim_tokens = {
        t.strip(".,;:!?\"'()[]{}")
        for t in re.findall(r"[a-zA-Z0-9_\-\.%]+", claim_lower)
        if t.strip(".,;:!?\"'()[]{}")
    }

    best_match_chunk: RetrievedChunk | None = None
    best_overlap = 0.0

    # Explicit antonym / predicate opposition pairs (strict semantic opposites only)
    contradiction_pairs = [
        ("increased", "decreased"),
        ("expanded", "contracted"),
        ("grew", "declined"),
        ("approved", "rejected"),
        ("allowed", "prohibited"),
        ("acquired", "divested"),
        ("positive", "negative"),
        ("higher", "lower"),
        ("rose", "fell"),
        ("free", "paid"),
        ("enabled", "disabled"),
        ("success", "failure"),
        ("supported", "unsupported"),
        ("permitted", "forbidden"),
        ("active", "inactive"),
        ("valid", "invalid"),
        ("present", "absent"),
        ("true", "false"),
    ]

    for chunk in chunks:
        chunk_norm = normalize_text_equivalences(chunk.text)
        chunk_text_lower = chunk_norm.lower()
        chunk_tokens = {
            t.strip(".,;:!?\"'()[]{}")
            for t in re.findall(r"[a-zA-Z0-9_\-\.%]+", chunk_text_lower)
            if t.strip(".,;:!?\"'()[]{}")
        }

        # 1. Antonym / Predicate Opposition Check with shared topic
        for w1, w2 in contradiction_pairs:
            if (w1 in claim_lower and w2 in chunk_text_lower) or (w2 in claim_lower and w1 in chunk_text_lower):
                shared = {
                    t for t in claim_tokens.intersection(chunk_tokens)
                    if len(t) > 2 and t not in (w1, w2, "the", "and", "for", "with", "this", "that", "was", "were", "are", "is")
                }
                if len(shared) >= 1:
                    return (
                        ClaimStatus.CONTRADICTED,
                        chunk,
                        f"Heuristic contradiction: claim asserts '{w1}' while evidence states '{w2}' for '{', '.join(sorted(shared))}' (chunk {chunk.chunk_id}).",
                    )

        # 2. Numerical / Monetary / Quantitative / Date Conflict Check (e.g. $100M vs $80M, 2021 vs 2024)
        claim_amounts = {
            a.strip(".,;:!?")
            for a in re.findall(r"\$?\d+(?:\.\d+)?(?:%|[bkmBKM]|(?:\s*(?:million|billion|thousand)))?\b|\b(?:19|20)\d{2}\b", claim_lower)
            if a.strip(".,;:!?")
        }
        chunk_amounts = {
            a.strip(".,;:!?")
            for a in re.findall(r"\$?\d+(?:\.\d+)?(?:%|[bkmBKM]|(?:\s*(?:million|billion|thousand)))?\b|\b(?:19|20)\d{2}\b", chunk_text_lower)
            if a.strip(".,;:!?")
        }

        if claim_amounts and chunk_amounts and not claim_amounts.intersection(chunk_amounts):
            # Contradiction requires shared subject/noun entities, not just common predicate verbs
            predicate_verbs = {
                "the", "and", "for", "with", "this", "that", "was", "were", "are", "is",
                "has", "had", "have", "been", "increased", "decreased", "grew", "rose",
                "fell", "dropped", "added", "expanded", "reported", "reached", "to", "by",
                "in", "at", "on", "from", "of", "an", "a",
            }
            shared_topic = {
                t for t in claim_tokens.intersection(chunk_tokens)
                if len(t) > 2 and t not in predicate_verbs
            }
            if shared_topic:
                return (
                    ClaimStatus.CONTRADICTED,
                    chunk,
                    f"Heuristic contradiction (numerical/quantitative conflict with chunk {chunk.chunk_id}): claim asserts {claim_amounts} while evidence states {chunk_amounts} for '{', '.join(sorted(shared_topic))}'.",
                )

        # 3. Explicit Negation / Polarity Conflict on Shared Predicate & Object
        neg_nouns_claim = set(re.findall(r"\b(?:no|without|zero)\s+([a-zA-Z]+)\b", claim_lower))
        neg_nouns_chunk = set(re.findall(r"\b(?:no|without|zero)\s+([a-zA-Z]+)\b", chunk_text_lower))

        conflict_nouns = (neg_nouns_claim.intersection(chunk_tokens) - neg_nouns_chunk) | (
            neg_nouns_chunk.intersection(claim_tokens) - neg_nouns_claim
        )
        if conflict_nouns:
            action_verbs = {
                "requires", "required", "needs", "needed", "provides", "provided",
                "supports", "supported", "includes", "included", "allows", "allowed",
                "contains", "contained", "has", "have", "had", "uses", "used", "shows", "showed",
            }
            shared_action = claim_tokens.intersection(chunk_tokens).intersection(action_verbs)
            if shared_action:
                return (
                    ClaimStatus.CONTRADICTED,
                    chunk,
                    f"Heuristic contradiction (negation conflict: action '{', '.join(sorted(shared_action))}' with negated '{', '.join(sorted(conflict_nouns))}' in chunk {chunk.chunk_id}).",
                )

        # 4. Explicit Predicate Negation Check (e.g. "is not supported" vs "is supported")
        key_predicates = [
            "supported", "approved", "permitted", "allowed", "included",
            "required", "active", "enabled", "available", "completed", "found",
        ]
        for pred in key_predicates:
            if pred in claim_tokens and pred in chunk_tokens:
                claim_neg = any(f"{n} {pred}" in claim_lower or f"{n} be {pred}" in claim_lower for n in ("not", "never", "cannot"))
                chunk_neg = any(f"{n} {pred}" in chunk_text_lower or f"{n} be {pred}" in chunk_text_lower for n in ("not", "never", "cannot"))
                if claim_neg != chunk_neg:
                    shared_subj = {
                        t for t in claim_tokens.intersection(chunk_tokens)
                        if len(t) > 2 and t != pred and t not in ("the", "and", "for", "with", "this", "that", "was", "were", "are", "is")
                    }
                    if shared_subj:
                        return (
                            ClaimStatus.CONTRADICTED,
                            chunk,
                            f"Heuristic contradiction (predicate negation on '{pred}' for '{', '.join(sorted(shared_subj))}' in chunk {chunk.chunk_id}).",
                        )

        # 3. Normalized Token / Entity Alignment
        if claim_tokens:
            overlap = len(claim_tokens.intersection(chunk_tokens)) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_chunk = chunk

    # Supported threshold: key entities and >= 45% normalized token alignment
    if best_overlap >= 0.45 and best_match_chunk is not None:
        return (
            ClaimStatus.SUPPORTED,
            best_match_chunk,
            f"Supported by chunk {best_match_chunk.chunk_id} ({round(best_overlap * 100, 1)}% lexical grounding).",
        )

    return (
        ClaimStatus.UNSUPPORTED,
        None,
        f"Unsupported: no retrieved chunk substantiates claim (max lexical alignment {round(best_overlap * 100, 1)}%).",
    )


def is_evidence_match(gold: DocumentReference, chunk: RetrievedChunk) -> bool:
    """Matches evidence reference at chunk-level if chunk_id is specified, else document-level."""
    if gold.document_id != chunk.document_id:
        return False
    if gold.chunk_id is not None and chunk.chunk_id != gold.chunk_id:
        return False
    return True


def wilson_score_interval(p: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion at the specified confidence level."""
    if n <= 0:
        return (0.0, 0.0)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
    # Compute the standard normal two-tailed quantile corresponding to the requested confidence level
    alpha = 1.0 - confidence
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denominator = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denominator
    half_width = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4 * (n**2))))
    lower = max(0.0, round(centre - half_width, 4))
    upper = min(1.0, round(centre + half_width, 4))
    return (lower, upper)


class BaseMetric(ABC):
    """Abstract base metric evaluator."""

    name: str
    family: MetricFamily
    version: str = "2.1.0"

    @abstractmethod
    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        """Compute atomic metric score for a given trace and test case."""
        ...


# --- Retrieval Metrics ---
class RecallAtKMetric(BaseMetric):
    """Measures whether required evidence (document or chunk level) was retrieved in top-K."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.name = f"recall_at_{k}"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.NOT_APPLICABLE,
                reason="No relevant gold documents specified in test case.",
                evaluator_version=self.version,
            )

        gold_refs = case.relevant_documents
        retrieved_k = trace.retrieved_chunks[: self.k]

        matched_gold = 0
        chunk_hits = 0
        doc_hits = 0
        for gold in gold_refs:
            if any(is_evidence_match(gold, chunk) for chunk in retrieved_k):
                matched_gold += 1
                if gold.chunk_id:
                    chunk_hits += 1
            if any(gold.document_id == chunk.document_id for chunk in retrieved_k):
                doc_hits += 1

        score = matched_gold / len(gold_refs)
        spec_level = "chunk" if any(g.chunk_id for g in gold_refs) else "document"

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            status=MetricStatus.PASS if score >= 0.70 else MetricStatus.FAIL,
            reason=f"Found {matched_gold}/{len(gold_refs)} gold evidence targets ({spec_level}-level) in top-{self.k}.",
            evaluator_version=self.version,
            metadata={
                "matched_count": matched_gold,
                "total_gold": len(gold_refs),
                "k": self.k,
                "chunk_level_hits": chunk_hits,
                "doc_level_hits": doc_hits,
            },
        )


class MeanReciprocalRankMetric(BaseMetric):
    """Computes Reciprocal Rank (1/rank) for the highest-ranked gold evidence chunk."""

    def __init__(self) -> None:
        self.name = "mrr"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.NOT_APPLICABLE,
                reason="No golden documents required for query.",
                evaluator_version=self.version,
            )

        gold_refs = case.relevant_documents
        for chunk in trace.retrieved_chunks:
            if any(is_evidence_match(gold, chunk) for gold in gold_refs):
                rr = 1.0 / max(1, chunk.rank)
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=round(rr, 4),
                    status=MetricStatus.PASS if rr >= 0.50 else MetricStatus.FAIL,
                    reason=f"First relevant evidence '{chunk.document_id}:{chunk.chunk_id}' found at rank {chunk.rank}.",
                    evaluator_version=self.version,
                )

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=0.0,
            status=MetricStatus.FAIL,
            reason="No golden evidence targets found in retrieved chunks.",
            evaluator_version=self.version,
        )


class ContextualPrecisionMetric(BaseMetric):
    """Evaluates whether relevant chunks are ranked above distractors."""

    def __init__(self) -> None:
        self.name = "contextual_precision"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.NOT_APPLICABLE,
                reason="No gold evidence was supplied.",
                evaluator_version=self.version,
            )

        if not trace.retrieved_chunks:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.INSUFFICIENT_DATA,
                reason="No retrieved chunks available to evaluate contextual precision.",
                evaluator_version=self.version,
            )

        gold_refs = case.relevant_documents
        running_relevant = 0
        precisions: list[float] = []

        for i, chunk in enumerate(trace.retrieved_chunks, start=1):
            if any(is_evidence_match(gold, chunk) for gold in gold_refs):
                running_relevant += 1
                precisions.append(running_relevant / i)

        score = sum(precisions) / len(precisions) if precisions else 0.0
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            status=MetricStatus.PASS if score >= 0.60 else MetricStatus.FAIL,
            reason=f"Precision computed across {len(precisions)} relevant evidence chunk hits.",
            evaluator_version=self.version,
        )


# --- Generation Metrics ---
class FaithfulnessMetric(BaseMetric):
    """Claim-level Grounding / Faithfulness: verifies that each generated claim is supported by retrieved evidence."""

    def __init__(self) -> None:
        self.name = "faithfulness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if trace.abstained:
            if case.answerability == Answerability.UNANSWERABLE:
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=1.0,
                    status=MetricStatus.PASS,
                    reason="Correctly abstained on unanswerable query; zero hallucinated claims produced.",
                    evaluator_version=self.version,
                    metadata={"claims": [], "abstained": True, "is_refusal": True},
                )
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                status=MetricStatus.FAIL,
                reason="System abstained on an answerable query; no grounded claims produced.",
                evaluator_version=self.version,
                metadata={"claims": [], "abstained": True},
            )

        if not trace.answer or not trace.answer.strip():
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                status=MetricStatus.FAIL,
                reason="Answer is empty; no claims produced.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        if not trace.retrieved_chunks:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                status=MetricStatus.FAIL,
                reason="No retrieved context available to support answer claims.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        # 1. Extract atomic factual claims
        claims = extract_claims(trace.answer)
        if not claims:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                status=MetricStatus.PASS,
                reason="No verifiable factual claims detected in response.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        # 2. Verify each claim
        verifications: list[ClaimVerification] = []
        supported_count = 0
        contradiction_count = 0

        for idx, claim_text in enumerate(claims):
            status, matched_chunk, reason = verify_claim_against_chunks(claim_text, trace.retrieved_chunks)
            verifications.append(
                ClaimVerification(
                    claim_id=f"clm_{idx+1}",
                    claim_text=claim_text,
                    status=status,
                    supporting_chunk_id=matched_chunk.chunk_id if matched_chunk else None,
                    confidence=None,
                    confidence_type="not_calibrated",
                    reason=reason,
                )
            )
            if status == ClaimStatus.SUPPORTED:
                supported_count += 1
            elif status == ClaimStatus.CONTRADICTED:
                contradiction_count += 1

        raw_score = supported_count / len(claims)
        penalty = 0.25 * contradiction_count
        final_score = max(0.0, round(raw_score - penalty, 4))

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=final_score,
            status=MetricStatus.PASS if final_score >= 0.70 else MetricStatus.FAIL,
            reason=f"Lexical Claim Grounding verdict: {supported_count}/{len(claims)} supported, {contradiction_count} contradicted.",
            evaluator_version=self.version,
            metadata={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "contradictions": contradiction_count,
                "contradicted_claims": contradiction_count,
                "claims": [c.model_dump() for c in verifications],
            },
        )


class AnswerCorrectnessMetric(BaseMetric):
    """Fact-anchored answer correctness: evaluates golden fact coverage, lexical/entity precision & recall."""

    def __init__(self) -> None:
        self.name = "answer_correctness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.expected_answer and not case.expected_facts:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.NOT_APPLICABLE,
                reason="No expected answer or expected facts specified in golden test case.",
                evaluator_version=self.version,
            )

        if not trace.answer or not trace.answer.strip():
            score = 1.0 if case.answerability == Answerability.UNANSWERABLE else 0.0
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=score,
                status=MetricStatus.PASS if score >= 0.70 else MetricStatus.FAIL,
                reason="Answer is empty.",
                evaluator_version=self.version,
            )

        # 1. Fact Coverage
        fact_coverage = 1.0
        covered_facts: list[str] = []
        missing_facts: list[str] = []
        ans_norm = normalize_text_equivalences(trace.answer)

        if case.expected_facts:
            for fact in case.expected_facts:
                fact_norm = normalize_text_equivalences(fact)
                fact_tokens = [w for w in re.findall(r"\w+", fact_norm) if len(w) > 2]
                if fact_tokens and all(w in ans_norm for w in fact_tokens[: max(1, int(len(fact_tokens) * 0.7))]):
                    covered_facts.append(fact)
                else:
                    missing_facts.append(fact)
            fact_coverage = len(covered_facts) / len(case.expected_facts)

        # 2. Token / Entity F1
        token_f1 = 1.0
        if case.expected_answer:
            ans_words = set(re.findall(r"\w+", ans_norm))
            exp_words = set(re.findall(r"\w+", normalize_text_equivalences(case.expected_answer)))

            if ans_words and exp_words:
                intersection = ans_words.intersection(exp_words)
                precision = len(intersection) / len(ans_words)
                recall = len(intersection) / len(exp_words)
                token_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                token_f1 = 0.0

        if case.expected_facts and case.expected_answer:
            score = round(0.60 * fact_coverage + 0.40 * token_f1, 4)
            reason = f"Fact coverage: {len(covered_facts)}/{len(case.expected_facts)} ({round(fact_coverage, 2)}), Token F1: {round(token_f1, 3)}."
        elif case.expected_facts:
            score = round(fact_coverage, 4)
            reason = f"Fact coverage: {len(covered_facts)}/{len(case.expected_facts)} facts."
        else:
            score = round(token_f1, 4)
            reason = f"Token overlap F1: {round(token_f1, 3)}."

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            status=MetricStatus.PASS if score >= 0.70 else MetricStatus.FAIL,
            reason=reason,
            evaluator_version=self.version,
            metadata={
                "fact_coverage": round(fact_coverage, 4),
                "token_f1": round(token_f1, 4),
                "covered_facts": covered_facts,
                "missing_facts": missing_facts,
            },
        )


class CitationSupportMetric(BaseMetric):
    """Citation validation: checks that citations refer to retrieved chunks and actually substantiate claims."""

    def __init__(self) -> None:
        self.name = "citation_accuracy"
        self.family = MetricFamily.CITATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not trace.citations:
            claims = extract_claims(trace.answer or "")
            if claims and len(claims) > 0 and trace.answer and len(trace.answer.strip()) > 30:
                # Factual claims were produced without citing evidence
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=0.0,
                    status=MetricStatus.FAIL,
                    reason=f"Generated {len(claims)} factual statements without citing evidence.",
                    evaluator_version=self.version,
                    metadata={"missing_citations": True},
                )
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=None,
                status=MetricStatus.NOT_APPLICABLE,
                reason="No citations required for query.",
                evaluator_version=self.version,
            )

        chunk_map = {c.chunk_id: c for c in trace.retrieved_chunks}
        doc_map = {c.document_id: c for c in trace.retrieved_chunks}

        valid_count = 0
        heuristic_validated: list[dict] = []
        total = len(trace.citations)

        for cit in trace.citations:
            matched = chunk_map.get(cit.chunk_id) or doc_map.get(cit.document_id)
            if not matched:
                continue

            # Primary: verify claim against matched chunk using the grounding evaluator
            citation_status, _, _ = verify_claim_against_chunks(cit.claim_text, [matched])
            if citation_status == ClaimStatus.SUPPORTED:
                valid_count += 1
            else:
                # Heuristic lexical fallback — explicitly disclosed in metadata.
                # A weak lexical match must NOT silently override the primary evaluator.
                claim_words = [w for w in re.findall(r"\w+", cit.claim_text.lower()) if len(w) > 2]
                chunk_words = set(re.findall(r"\w+", matched.text.lower()))
                lexical_overlap = sum(1 for w in claim_words if w in chunk_words) / len(claim_words) if claim_words else 0.0
                if lexical_overlap >= 0.40:
                    valid_count += 1
                    # Tag this citation as heuristically validated (not semantically)
                    heuristic_validated.append({
                        "claim_id": cit.claim_id,
                        "evaluation_method": "HEURISTIC_FALLBACK",
                        "lexical_overlap": round(lexical_overlap, 3),
                    })

        score = round(valid_count / total, 4)
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            status=MetricStatus.PASS if score >= 0.70 else MetricStatus.FAIL,
            reason=f"{valid_count}/{total} citations substantiated by retrieved evidence chunks.",
            evaluator_version=self.version,
            metadata={
                "valid_citations": valid_count,
                "total_citations": total,
                "heuristic_fallback_count": len(heuristic_validated),
                "heuristic_validated": heuristic_validated,
                "evaluation_method": "HEURISTIC_FALLBACK" if heuristic_validated else "LEXICAL_GROUNDING",
            },
        )


class AbstentionAccuracyMetric(BaseMetric):
    """Verifies that unanswerable cases are refused and answerable cases are attempted."""

    def __init__(self) -> None:
        self.name = "abstention_accuracy"
        self.family = MetricFamily.ABSTENTION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        is_unanswerable = case.answerability == Answerability.UNANSWERABLE
        score = 1.0 if (is_unanswerable == trace.abstained) else 0.0
        reason = (
            "Valid refusal on unanswerable case."
            if (is_unanswerable and trace.abstained)
            else "Valid answer on answerable case."
            if (not is_unanswerable and not trace.abstained)
            else "Failed: answered unanswerable case with hallucination."
            if is_unanswerable
            else "Failed: incorrectly refused an answerable case."
        )

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            status=MetricStatus.PASS if score == 1.0 else MetricStatus.FAIL,
            reason=reason,
            evaluator_version=self.version,
        )


import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BaseEvaluationCache(ABC):
    """Abstract interface for evaluation metric result caches."""

    @abstractmethod
    def get(self, key: str) -> MetricResult | None:
        """Retrieve cached result or None."""

    @abstractmethod
    def set(self, key: str, value: MetricResult) -> None:
        """Store result in cache."""

    @abstractmethod
    def clear(self) -> None:
        """Clear cache entries."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cache metrics and diagnostics."""


class BoundedLRUCache(BaseEvaluationCache):
    """Thread-safe bounded in-process LRU cache for evaluation metric results.

    Prevents unbounded memory growth via strict capacity enforcement and
    least-recently-used eviction. Process restarts naturally clear the cache.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = max(1, capacity)
        self._cache: OrderedDict[str, MetricResult] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> MetricResult | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: MetricResult) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)  # Evict least recently used entry
            self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: str) -> MetricResult:
        with self._lock:
            val = self._cache[key]
            self._cache.move_to_end(key)
            return val

    def __setitem__(self, key: str, value: MetricResult) -> None:
        self.set(key, value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
            }


class DatabaseEvaluationCache(BaseEvaluationCache):
    """Database-backed persistent evaluation cache shared across multi-process workers.

    Allows worker processes on separate nodes or containers to share evaluated
    metric results without requiring an external Redis instance.
    """

    def __init__(self, capacity: int = 50_000, session_factory: Callable | None = None) -> None:
        self.capacity = max(1, capacity)
        self.session_factory = session_factory
        self.hits: int = 0
        self.misses: int = 0

    def _get_session(self):
        if self.session_factory is not None:
            return self.session_factory()
        from rag_platform.db import create_session
        return create_session()

    def get(self, key: str) -> MetricResult | None:
        from datetime import datetime, timezone

        from rag_platform.db import EvaluationCacheRow

        try:
            with self._get_session() as sess:
                row = sess.get(EvaluationCacheRow, key)
                if row:
                    now = datetime.now(timezone.utc)
                    last_acc = row.last_accessed_at.replace(tzinfo=timezone.utc) if row.last_accessed_at.tzinfo is None else row.last_accessed_at
                    # Throttle last_accessed_at DB writes to at most once per 5 minutes per cached entry
                    if (now - last_acc).total_seconds() > 300:
                        row.last_accessed_at = now
                        sess.commit()
                    self.hits += 1
                    data = json.loads(row.result_json)
                    return MetricResult.model_validate(data)
                self.misses += 1
                return None
        except Exception as exc:
            logger.debug("DatabaseEvaluationCache get error: %s", exc)
            self.misses += 1
            return None

    def set(self, key: str, value: MetricResult) -> None:
        from datetime import datetime, timezone

        from sqlalchemy import delete, func, select

        from rag_platform.db import EvaluationCacheRow

        try:
            with self._get_session() as sess:
                now = datetime.now(timezone.utc)
                row = sess.get(EvaluationCacheRow, key)
                if row:
                    row.result_json = value.model_dump_json()
                    row.last_accessed_at = now
                else:
                    new_row = EvaluationCacheRow(
                        cache_key=key,
                        result_json=value.model_dump_json(),
                        created_at=now,
                        last_accessed_at=now,
                    )
                    sess.add(new_row)
                    sess.flush()

                    # Prune when count exceeds capacity (bounded by oldest accessed)
                    self._set_counter = getattr(self, "_set_counter", 0) + 1
                    prune_interval = max(1, min(50, self.capacity))
                    if self._set_counter >= self.capacity and (self._set_counter % prune_interval == 0 or self._set_counter > self.capacity):
                        total_count = sess.scalar(select(func.count(EvaluationCacheRow.cache_key))) or 0
                        if total_count > self.capacity:
                            oldest_subq = (
                                select(EvaluationCacheRow.cache_key)
                                .order_by(EvaluationCacheRow.last_accessed_at.asc(), EvaluationCacheRow.created_at.asc())
                                .limit(total_count - self.capacity)
                            ).scalar_subquery()
                            sess.execute(
                                delete(EvaluationCacheRow).where(
                                    EvaluationCacheRow.cache_key.in_(oldest_subq)
                                )
                            )
                sess.commit()
        except Exception as exc:
            logger.debug("DatabaseEvaluationCache set error: %s", exc)

    def clear(self) -> None:
        from sqlalchemy import delete

        from rag_platform.db import EvaluationCacheRow

        try:
            with self._get_session() as sess:
                sess.execute(delete(EvaluationCacheRow))
                sess.commit()
                self.hits = 0
                self.misses = 0
        except Exception as exc:
            logger.debug("DatabaseEvaluationCache clear error: %s", exc)

    def stats(self) -> dict[str, Any]:
        from sqlalchemy import func, select

        from rag_platform.db import EvaluationCacheRow

        try:
            with self._get_session() as sess:
                count = sess.scalar(select(func.count(EvaluationCacheRow.cache_key))) or 0
                return {
                    "size": count,
                    "capacity": self.capacity,
                    "hits": self.hits,
                    "misses": self.misses,
                }
        except Exception:
            return {
                "size": 0,
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
            }


class TwoTierEvaluationCache(BaseEvaluationCache):
    """Two-tier evaluation cache combining L1 in-process memory LRU and L2 shared DB table.

    Eliminates multi-process cache isolation while preserving high-speed in-memory reads:
    - L1 (In-Memory BoundedLRUCache): Sub-millisecond hits for the local worker process.
    - L2 (DatabaseEvaluationCache): Shared across all distributed worker nodes/processes.
    """

    def __init__(
        self,
        l1_capacity: int = 10_000,
        l2_capacity: int = 50_000,
        session_factory: Callable | None = None,
    ) -> None:
        self.l1 = BoundedLRUCache(capacity=l1_capacity)
        self.l2 = DatabaseEvaluationCache(capacity=l2_capacity, session_factory=session_factory)
        self.capacity = l1_capacity + l2_capacity

    @property
    def hits(self) -> int:
        return self.l1.hits + self.l2.hits

    @property
    def misses(self) -> int:
        return self.l2.misses

    def get(self, key: str) -> MetricResult | None:
        # Check L1 memory first
        res = self.l1.get(key)
        if res is not None:
            return res
        # Check L2 shared DB
        res = self.l2.get(key)
        if res is not None:
            # Populate L1 for future fast local reads
            self.l1.set(key, res)
            return res
        return None

    def set(self, key: str, value: MetricResult) -> None:
        self.l1.set(key, value)
        self.l2.set(key, value)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
            "hits": self.hits,
            "misses": self.misses,
            "capacity": self.capacity,
        }


# --- Evaluation Engine, Cache, & Statistics ---
class EvaluationEngine:
    """Orchestrates metric execution with hashed caching, statistical confidence intervals, and aggregation."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        cache: BaseEvaluationCache | dict[str, MetricResult] | None = None,
    ) -> None:
        self.metrics = metrics or [
            RecallAtKMetric(k=5),
            MeanReciprocalRankMetric(),
            ContextualPrecisionMetric(),
            FaithfulnessMetric(),
            AnswerCorrectnessMetric(),
            CitationSupportMetric(),
            AbstentionAccuracyMetric(),
        ]
        self._cache = cache if cache is not None else BoundedLRUCache(capacity=10_000)

    def _cache_key(self, metric: BaseMetric, trace: RagTrace, case: TestCase) -> str:
        """Cache key incorporating ALL inputs that affect evaluation output.

        Includes case questions, facts, relevant documents, answer, chunks (with content hash),
        citations, and metric version to prevent stale cache entries.
        """
        payload = {
            "metric": metric.name,
            "version": metric.version,
            "case_id": case.id,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "expected_facts": sorted(case.expected_facts),
            "relevant_documents": [
                {"document_id": d.document_id, "chunk_id": d.chunk_id, "page": d.page, "span": d.span}
                for d in case.relevant_documents
            ],
            "trace_answer": trace.answer,
            "trace_abstained": trace.abstained,
            # Include chunk content hash, not just IDs (fixes stale cache on content change)
            "chunks": [
                {"document_id": c.document_id, "chunk_id": c.chunk_id, "content_hash": sha256_hash(c.text)}
                for c in trace.retrieved_chunks
            ],
            "citations": [
                {"claim_id": cit.claim_id, "chunk_id": cit.chunk_id, "claim_text": cit.claim_text}
                for cit in trace.citations
            ],
        }
        return sha256_hash(canonical_json(payload))

    async def evaluate_trace(self, trace: RagTrace, case: TestCase, use_cache: bool = True) -> list[MetricResult]:
        results: list[MetricResult] = []
        for metric in self.metrics:
            key = self._cache_key(metric, trace, case)
            if use_cache:
                if hasattr(self._cache, "get"):
                    cached_val = self._cache.get(key)
                    if cached_val is not None:
                        results.append(cached_val.model_copy(update={"cached": True}))
                        continue
                elif key in self._cache:  # type: ignore[operator]
                    cached_res = self._cache[key].model_copy(update={"cached": True})  # type: ignore[index]
                    results.append(cached_res)
                    continue

            result = await metric.compute(trace, case)
            if use_cache:
                if hasattr(self._cache, "set"):
                    self._cache.set(key, result)
                else:
                    self._cache[key] = result
            results.append(result)
        return results

    def aggregate_run(self, traces_with_metrics: list[tuple[RagTrace, list[MetricResult]]]) -> RunMetricsSummary:
        metric_values: dict[str, list[float]] = {}
        metric_families: dict[str, MetricFamily] = {}
        metric_total_counts: dict[str, int] = {}
        hallucinations = 0
        abstention_scores = []
        latencies = []
        total_cost = 0.0

        for trace, m_list in traces_with_metrics:
            latencies.append(trace.latency_ms)
            if trace.cost_usd:
                total_cost += trace.cost_usd

            for m in m_list:
                metric_total_counts[m.metric_name] = metric_total_counts.get(m.metric_name, 0) + 1
                metric_families[m.metric_name] = m.metric_family
                # Only include applicable metrics with a real numerical score
                if m.score is not None:
                    metric_values.setdefault(m.metric_name, []).append(m.score)

                if m.metric_name == "faithfulness" and m.score is not None and m.score < 0.60 and not trace.abstained:
                    hallucinations += 1
                if m.metric_name == "abstention_accuracy" and m.score is not None:
                    abstention_scores.append(m.score)

        summaries: dict[str, MetricSummary] = {}
        for name, total_evals in metric_total_counts.items():
            vals = metric_values.get(name, [])
            n = len(vals)
            if n == 0:
                summaries[name] = MetricSummary(
                    metric_name=name,
                    metric_family=metric_families[name],
                    mean=0.0,
                    p50=0.0,
                    p95=0.0,
                    min=0.0,
                    max=0.0,
                    count=0,
                    applicable_count=0,
                    std_dev=0.0,
                    ci_lower=0.0,
                    ci_upper=0.0,
                    sample_warning="No applicable test cases evaluated for this metric.",
                )
                continue

            sorted_vals = sorted(vals)
            mean_val = sum(vals) / n

            # Calculate sample standard deviation
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_dev = math.sqrt(variance)

            # Wilson score interval for bounded proportion metrics
            ci_lower, ci_upper = wilson_score_interval(mean_val, n)

            p50_idx = int(n * 0.50)
            p95_idx = min(int(n * 0.95), n - 1)

            sample_warning = (
                f"Small sample size (N={n} < 30); statistical variance is elevated."
                if n < 30
                else None
            )

            summaries[name] = MetricSummary(
                metric_name=name,
                metric_family=metric_families[name],
                mean=round(mean_val, 4),
                p50=round(sorted_vals[p50_idx], 4),
                p95=round(sorted_vals[p95_idx], 4),
                min=round(sorted_vals[0], 4),
                max=round(sorted_vals[-1], 4),
                count=n,
                applicable_count=n,
                std_dev=round(std_dev, 4),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sample_warning=sample_warning,
            )

        total_cases = len(traces_with_metrics)
        hallucination_rate = round(hallucinations / total_cases, 4) if total_cases > 0 else None
        abstention_acc = round(sum(abstention_scores) / len(abstention_scores), 4) if abstention_scores else None

        sorted_latencies = sorted(latencies) if latencies else [0]
        p95_lat = sorted_latencies[min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)]

        run_warning = (
            f"Run evaluation based on small sample size (N={total_cases} < 30). Regression results may lack statistical power."
            if total_cases < 30
            else None
        )

        return RunMetricsSummary(
            metrics=summaries,
            total_cases=total_cases,
            scored_cases=total_cases,
            hallucination_rate=hallucination_rate,
            low_faithfulness_rate=hallucination_rate,
            abstention_accuracy=abstention_acc,
            p95_latency_ms=float(p95_lat),
            total_cost_usd=round(total_cost, 4),
            sample_warning=run_warning,
        )
