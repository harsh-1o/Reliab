# Metrics & Evaluation Methodology

Reliab uses deterministic metrics across retrieval, generation, citation grounding, and abstention. The metrics are reproducible benchmark signals, not universal measures of intelligence or truth.

## Metric matrix

| Metric | Family | Measures | Key limitation |
|---|---|---|---|
| Recall@K | Retrieval | Gold evidence in top K | Depends on complete gold annotations |
| MRR | Retrieval | Rank of first relevant result | Depends on relevance labels |
| Contextual Precision | Retrieval | Relevant evidence concentrated near top | Current relevance checks are heuristic |
| Faithfulness | Generation | Claims supported by retrieved evidence | Lexical/entity/rule-based; not full semantic entailment |
| Answer Correctness | Generation | Expected facts represented | Depends on expected-fact annotations |
| Citation Accuracy | Citation | Citation resolution/support | Structural correctness is not proof of semantic support |
| Abstention Accuracy | Abstention | Correct answer/refusal behavior | Depends on answerability labels |

## Applicability

Conditional metrics use:

```text
status = NOT_APPLICABLE
score  = None
```

when their preconditions are absent. Aggregation does not turn these into artificial zeros.

Examples: retrieval metrics require gold evidence; answer correctness requires expected facts; citation accuracy can be inapplicable for an unanswerable case with no factual claim.

Abstention accuracy is designed to remain applicable to answerability cases.

## Retrieval limitations

### Recall@K

Uses explicit chunk `rank` for rank-sensitive evaluation.

Limitations:

- equivalent evidence absent from the gold set can be marked missing;
- gold document/chunk annotations can be incomplete;
- retrieval quality alone does not establish answer quality.

### MRR

Rewards high-ranked relevant evidence. It inherits the benchmark's relevance definition.

### Contextual Precision

Measures ranking/concentration of relevant context. It should not be interpreted as proof that context is semantically sufficient.

## Generation limitations

### Faithfulness

The current implementation performs deterministic claim extraction, lexical/entity overlap, and numerical/polarity conflict checks.

It is:

- fast;
- reproducible;
- explainable.

It is not a complete semantic entailment judge. Paraphrases with low lexical overlap can be under-scored, while superficial overlap can sometimes look supportive.

### Answer Correctness

Expected facts define the benchmark target. A high score means the generated answer matches those expected facts under the implemented verifier, not that every real-world nuance was checked.

## Citation limitations

Reliab verifies citation document/chunk resolution and supporting evidence heuristics.

A correct document ID does not automatically prove that the cited text semantically entails the claim.

Citation spans have structural validation, but provider-specific token/character semantics are not universally inferred.

## Abstention limitations

| Input | Expected | Outcome |
|---|---|---|
| Unanswerable | Abstain | Pass |
| Unanswerable | Answer/hallucinate | Fail |
| Answerable | Answer | Pass |
| Answerable | Refuse | Fail |

The metric inherits the quality of dataset answerability labels.

## Statistical reporting

Binary/binomial outcomes use Wilson score intervals. Continuous metrics use normal-approximation intervals based on observed sample statistics.

Confidence intervals quantify uncertainty around the benchmark statistic; they are not probabilities that a model is correct.

Small samples can be statistically unstable; Reliab emits a small-sample warning.

## Percentiles

Latency p50/p95 uses standard linear interpolation:

```text
rank = p × (N - 1)
```

with interpolation between neighboring sorted observations.

## Aggregation

`evaluated_cases` counts distinct executed case IDs. Duplicate traces do not inflate coverage, metric sample counts, latency totals, or cost totals.

`scored_cases` is distinct from `evaluated_cases`: a case may execute while all conditional metrics are legitimately inapplicable.

## Attribution is not a metric

Failure attribution provides deterministic diagnostic hypotheses such as retrieval, generation, citation, abstention, and operational codes.

Attribution confidence is heuristic and not a calibrated probability. Release gates use metric/gate evidence rather than treating attribution confidence as truth.

## Known limitations

1. Semantic entailment is not fully modeled.
2. Retrieval metrics inherit gold-evidence quality.
3. Answer correctness inherits expected-fact quality.
4. Citation support is harder to establish semantically than structurally.
5. Abstention quality inherits answerability labels.
6. Benchmarks may not represent production distribution.
7. Small datasets have higher statistical uncertainty.
8. Attribution confidence is not calibrated.

A future semantic verifier can complement the deterministic evaluator, but should remain distinguishable from the reproducible release-gate source of truth.
