# Supported Metrics & Evaluation Methodology

Reliab defines seven core metrics across retrieval, generation, citation grounding, and abstention behavior. 

Crucially, **precondition failures emit `NOT_APPLICABLE` (`score=None`)** rather than default zeros or artificial penalties, ensuring aggregate statistics remain mathematically valid.

---

## 1. Metric Applicability Matrix

| Metric Name | Family | Applicable Condition | Failure Definition | Precondition Absent Behavior |
|:---|:---|:---|:---|:---|
| `recall_at_k` | Retrieval | Gold evidence supplied | Gold documents not found in top-K | `status=NOT_APPLICABLE`, `score=None` |
| `mrr` | Retrieval | Gold evidence supplied | Gold document rank > threshold | `status=NOT_APPLICABLE`, `score=None` |
| `contextual_precision` | Retrieval | Gold evidence supplied | Relevant chunks ranked behind noise | `status=NOT_APPLICABLE`, `score=None` |
| `faithfulness` | Generation | Answerable, text produced | Unsupported or contradicted claims | `status=PASS`, `score=1.0` (if clean abstention) |
| `answer_correctness` | Generation | Expected facts supplied | Expected facts missing or contradicted | `status=NOT_APPLICABLE`, `score=None` |
| `citation_accuracy` | Citation | Factual claims generated | Claims cite wrong chunks or missing citations | `status=NOT_APPLICABLE`, `score=None` (if unanswerable) |
| `abstention_accuracy` | Abstention | Always evaluated | Unanswerable query answered, or answerable query refused | Scored on all test cases |

---

## 2. Statistical Reporting & Summary Semantics

### Confidence Intervals
- **Wilson Score Interval**: Strictly applied to binary/binomial metric outcomes ($\{0.0, 1.0\}$) such as binary exact match, binary groundings, or discrete abstentions. This provides robust lower and upper error bounds at a 95% confidence level without Gaussian skew.
- **Normal Approximation CI**: Continuous metric distributions (e.g. continuous similarity or partial overlap scores) compute 95% confidence intervals using standard normal distribution error bounds ($\bar{x} \pm z \cdot \frac{s}{\sqrt{n}}$).
- **Sample Size Warnings**: Runs with $N < 30$ cases automatically emit a warning:
  `Small sample size (N < 30); statistical variance is elevated.`

### Percentile Methodology
All latency and metric percentiles (p50, p95) use a single shared implementation based on **NIST Method 7 / standard linear interpolation**:
$$\text{rank} = p \times (N - 1)$$
where $p \in [0.0, 1.0]$ and linear interpolation is computed between adjacent ranks $v_{\lfloor \text{rank} \rfloor}$ and $v_{\lceil \text{rank} \rceil}$. This avoids the bias of nearest-rank / integer-truncation methods (which for $N=20$ incorrectly select the maximum observation).

### `scored_cases` vs `evaluated_cases` Semantics
- **`evaluated_cases`**: Total test cases executed in the run (bounded by `max_cases` if partial/smoke run).
- **`scored_cases`**: Distinct test cases where at least one metric yielded an applicable numeric evaluation (`score is not None`). Cases where all metrics legitimately evaluated to `NOT_APPLICABLE` (e.g. unanswerable queries on retrieval metrics) do not artificially increment `scored_cases`.

---

## 3. Claim-Level Evaluation (Lexical Grounding)

Rather than relying on noisy LLM judges or surface-level token overlap (ROUGE/BLEU), Reliab decomposes answers into atomic propositions:

```text
Generated Answer
       ↓
Sentence & Clause Decomposition (conjunctions, semicolons, punctuation)
       ↓
Atomic Proposition Units
       ↓
Equivalence Normalization ("one month" ↔ "30 days")
       ↓
Evidence Chunk Alignment
       ↓
[SUPPORTED] | [UNSUPPORTED] | [CONTRADICTED]
```

### Clause Decomposition
Handles complex sentences with coordinating conjunctions (`and`, `but`, `while`), semicolons, and numeric phrases.

- **Example Answer**: `"Revenue increased 20% and profit increased 15%."`
  - `Claim 1`: `"Revenue increased 20%."` (Supported by evidence chunk)
  - `Claim 2`: `"Profit increased 15%."` (Unsupported if evidence chunk only discusses revenue)
  - **Result**: Faithfulness = 0.50, `status=FAIL`.

### Contradiction Detection
Detects opposing predicates and conflicting numerical metrics:
- **Numerical Conflicts**: Evidence states `"Revenue was $80M."`, Answer asserts `"Revenue was $100M."` $\to$ `CONTRADICTED`.
- **Antonym Pairs**: Identifies semantic negations (`"increased"` vs `"decreased"`, `"approved"` vs `"rejected"`, `"allowed"` vs `"prohibited"`).

---

## 4. Retrieval Evaluation

Retrieval metrics assess the retriever component independently from the generation model:

### Recall@K
Proportion of golden document references successfully present in the top-$K$ retrieved chunks ($K=5$ by default):
$$\text{Recall@K} = \frac{|\text{Retrieved Gold Documents in top-K}|}{|\text{Total Gold Documents}|}$$

### Mean Reciprocal Rank (MRR)
Measures the reciprocal rank of the first relevant chunk:
$$\text{MRR} = \frac{1}{\text{rank}_{\text{first}}}$$
If no relevant document is found in retrieved results, $\text{MRR} = 0.0$.

### Contextual Precision
Evaluates whether relevant chunks are concentrated at top ranks versus diluted by irrelevant distractor chunks. Penalizes systems that rank irrelevant noise ahead of gold evidence.

---

## 5. Citation Evaluation

Reliab verifies that citations point to chunks that substantiate each specific claim:

- **Claim-Citation Alignment**: Matches the cited `document_id` and `chunk_id` in the answer to retrieved chunks.
- **Grounded Verification**: Confirms that the cited chunk contains lexical and factual support for the proposition.
- **Missing Citation Penalties**: If factual statements are generated on answerable queries with zero citations, `citation_accuracy = 0.0` (`FAIL`).
- **Unanswerable Abstentions**: If the system correctly abstains or makes no factual claims, `citation_accuracy = NOT_APPLICABLE` (`score=None`).

---

## 6. Abstention & Refusal Quality

Reliab evaluates system behavior on queries designed to elicit refusal:

- **Unanswerable Benchmark Cases**: Test queries with zero relevant context in the corpus or adversarial unanswerable questions.
- **Refusal Scoring (`abstention_accuracy`)**:
  - Unanswerable query + system abstains: **Pass (1.0)**.
  - Unanswerable query + system hallucinates an answer: **Fail (0.0, Code ABS-01)**.
  - Answerable query + system falsely refuses: **Fail (0.0, Code ABS-02)**.

---

## 7. Diagnostic Failure Attribution & Secondary Triage

Reliab provides root-cause failure taxonomy codes (e.g. `RET-01`, `GEN-02`, `CIT-01`, `ABS-01`) via deterministic heuristic rule engines and secondary classification models.

> [!NOTE]
> **Heuristic Diagnostics Notice**: Rule-based attribution and classifier outputs are designed for **secondary diagnostic triage and workflow acceleration**, not ground-truth causal proof or calibrated probabilistic inferences. Metric scores and raw traces remain the primary source of truth for release gate decisions.
