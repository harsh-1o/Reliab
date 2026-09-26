# Failure Taxonomy & Attribution

Reliab automatically diagnoses failed traces into standardized failure codes. Every failed evaluation receives a **Primary Failure Code**, optional **Contributing Failure Codes**, and a diagnostic evidence payload.

---

## 1. Canonical Taxonomy Table

| Code | Name | Family | Diagnostic Condition | Remediation Guidance |
|:---|:---|:---|:---|:---|
| **RET-01** | Empty Retrieval | Retrieval | Top chunk similarity < 0.35 or zero relevant chunks retrieved | Tune embedding model, inspect chunk indexing pipeline, verify document ingestion |
| **RET-02** | Low-Precision Retrieval | Retrieval | Recall < 0.60 or contextual precision < 0.40 | Integrate cross-encoder reranker, increase chunk overlap, adjust top-K |
| **RET-03** | Distractor Overload | Retrieval | Top-ranked chunks are irrelevant noise crowding out gold evidence | Implement dense-sparse hybrid search with reranking |
| **GEN-01** | Factual Hallucination | Generation | Faithfulness < 0.60 despite sufficient retrieval support | Constrain prompt instructions, lower model temperature, enforce strict citation grounding |
| **GEN-02** | Contradiction | Generation | Direct numerical or predicate conflict with retrieved evidence | Introduce verification self-correction step in generator prompt |
| **CIT-01** | Missing Citation | Citation | Factual claim generated without supporting document citation | Constrain citation output format with structured JSON schema |
| **CIT-02** | Unsubstantiated Citation | Citation | Cited chunk does not contain factual evidence for the claim | Enforce citation alignment verification before returning answer |
| **ABS-01** | Over-Generation on Unanswerable | Abstention | System answered query marked as unanswerable | Strengthen system refusal prompt and confidence thresholding |
| **ABS-02** | False Refusal | Abstention | System refused query despite relevant evidence being retrieved | Relax strict refusal heuristic in system prompt |
| **OPS-01** | Infrastructure Timeout / Error | Infrastructure | HTTP connection error, request timeout, or 5xx provider status | Implement connection pooling, exponential backoff, and circuit breakers |

---

## 2. Deterministic Attribution Engine

Attribution is powered by a deterministic, rule-based decision tree (`FailureAttributionEngine`).

### Evaluation Priority Order
1. **Infrastructure Faults (`OPS-01`)**: Evaluated first. If network transport or adapter timeouts occur, operational errors take precedence.
2. **Abstention Violations (`ABS-01`, `ABS-02`)**: Checked next for queries with specific answerability constraints.
3. **Retrieval Deficits (`RET-01`, `RET-02`, `RET-03`)**: If retrieved chunks fail relevance or recall thresholds, retrieval is flagged before blaming the generation model.
4. **Generation & Grounding (`GEN-01`, `GEN-02`)**: Analyzed when retrieval succeeded but the LLM hallucinated or contradicted the evidence.
5. **Citation Integrity (`CIT-01`, `CIT-02`)**: Evaluated when factual claims are present but citations are missing or improperly mapped.

### Primary vs. Contributing Codes
- **Primary Code**: The root cause that most directly caused the test failure.
- **Contributing Codes**: Secondary deficiencies that compounded the failure (e.g., poor retrieval precision contributing to a downstream hallucination).

---

## 3. Auxiliary Active Learning Classifier

In addition to deterministic rules, Reliab includes an optional scikit-learn classifier (`classifier.py`):
- Operates on numerical metric vectors and text length ratios.
- Outputs probabilistic predictions across failure categories.
- Surfaces high-uncertainty traces to a triage queue for human review and active learning feedback.
- Deterministic rules remain the ultimate source of truth for CI/CD gates.
