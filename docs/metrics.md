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

## 2. Statistical Reporting

To avoid misleading averages on small test sets, Reliab computes confidence bounds:

- **Wilson Score Interval**: Computed for bounded binomial metrics (Recall, Faithfulness, Precision, Citation Accuracy, Abstention Accuracy). This provides realistic lower and upper error bounds at a 95% confidence level.
- **Sample Size Warnings**: Runs with $N < 30$ cases automatically emit a warning:
  `Small sample size (N < 30); statistical variance is elevated.`

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
