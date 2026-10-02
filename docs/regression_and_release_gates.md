# Regression Detection & Release Gates

Reliab compares baseline and candidate runs and evaluates them against a named release policy.

## Current built-in policy

The named policy currently available through the resolver is:

```text
prod-default
```

Current implementation defaults:

| Field | Value |
|---|---:|
| `min_faithfulness` | 0.90 |
| `min_retrieval_recall` | 0.92 |
| `min_citation_accuracy` | 0.95 |
| `max_hallucination_rate` | 0.05 |
| `min_abstention_accuracy` | 0.90 |
| `max_latency_regression_pct` | 20.0 |
| `max_cost_regression_pct` | 25.0 |
| `min_cost_budget_usd` | 0.05 |
| `max_critical_regressions` | 0 |
| `min_case_coverage` | 1.0 |
| `allow_candidate_missing` | false |
| `metric_policies` | [] |

These are Reliab defaults, not universal definitions of acceptable RAG quality.

## Policy selection

Supported named policy IDs are resolved explicitly. Unknown IDs are rejected instead of silently falling back to `prod-default`.

The CLI uses `--policy prod-default`. Programmatic code may construct a `ReleasePolicy` directly.

Treat policy changes as release-control changes and review them like code.

## Coverage

Coverage is:

```text
distinct evaluated case IDs / required dataset case count
```

Duplicate traces do not increase coverage.

The default `min_case_coverage=1.0` means a release run must cover the complete required dataset. A partial/smoke run can still be useful for development but should not be treated as a complete release evaluation.

## Candidate missing cases

```text
baseline IDs - candidate IDs = CANDIDATE_MISSING
candidate IDs - baseline IDs = BASELINE_MISSING
```

A case with an inapplicable metric (`score=None`) is not automatically considered missing.

## Case transitions

| State | Meaning |
|---|---|
| `NEW_FAILURE` | Baseline pass → candidate fail |
| `CANDIDATE_MISSING` | Baseline case absent from candidate |
| `BASELINE_MISSING` | Candidate case absent from baseline |
| `RECOVERED` | Baseline fail → candidate pass |
| `UNCHANGED_PASS` | Pass → pass |
| `UNCHANGED_FAIL` | Fail → fail |
| `NOT_APPLICABLE` | Both cases exist but compared metric is inapplicable |

## Gate dimensions

The gate can reject a candidate because:

### Quality floors

```text
faithfulness >= 0.90
recall@5    >= 0.92
citation    >= 0.95
abstention  >= 0.90
```

under the current default policy.

### Hallucination cap

```text
hallucination_rate <= 0.05
```

### Regression budget

Critical regressions must remain within `max_critical_regressions`; the default allows zero.

### Completeness

Coverage and candidate omission are separate checks so difficult cases cannot be selectively removed to improve an aggregate.

### Latency and cost

Candidate latency may increase by at most 20% under the default. Cost may increase by at most 25%. The `min_cost_budget_usd` floor avoids unstable percentage comparisons when baseline cost is near zero.

## Gate result

`GateResult` contains status, policy ID, candidate run ID, violations, critical regression count, and evaluation timestamp.

CLI exit codes:

- `0` = PASS
- `1` = FAIL

JUnit output converts policy dimensions into CI-visible test cases.

## Example

```bash
python -m rag_platform.gate \
  --project proj_prod --dataset ds_gold \
  --system-version "$(git rev-parse HEAD)" \
  --policy prod-default \
  --adapter-type http \
  --endpoint-url https://rag.example/query
```

## Interpretation

PASS means the observed benchmark satisfied the selected policy. It does not prove universal factual correctness.

FAIL means the observed benchmark violated at least one policy constraint. Attribution can explain likely failure classes, but attribution is not causal proof.

The gate is therefore a reproducible **benchmark decision**, not a claim about every possible production query.
