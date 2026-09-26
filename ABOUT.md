# About Reliab

**Reliab** is developer infrastructure for evaluating and release-gating Retrieval-Augmented Generation (RAG) and LLM systems.

It focuses on a simple engineering problem: **how do you know that a change to an AI system did not make it worse?**

Reliab turns that question into a repeatable evaluation workflow:

```text
RAG / LLM System
       ↓
   Evaluation
       ↓
 Metrics + Grounding
       ↓
 Failure Attribution
       ↓
 Regression Detection
       ↓
   Release Gate
       ↓
   PASS / FAIL
```

### What Reliab evaluates

- Retrieval quality against golden evidence
- Claim-level grounding and contradictions
- Citation accuracy
- Abstention behavior
- Per-case regressions between candidate and baseline runs
- Release policies based on quality thresholds and regression budgets

### Design goals

Reliab is built around four principles:

- **Deterministic** — evaluation should produce predictable results.
- **Reproducible** — runs should retain the information needed to understand and compare them.
- **Actionable** — a failed evaluation should provide more information than a single score.
- **CI-friendly** — quality checks should be able to fail a build just like tests and lint checks do.

### How it fits into an engineering workflow

Reliab can be used locally during development, through its web dashboard for inspecting evaluation runs, or as a headless CLI in CI/CD.

A typical workflow is:

1. Maintain a versioned golden dataset.
2. Run the candidate RAG/LLM system against it.
3. Evaluate retrieval and generation behavior.
4. Attribute failures to diagnostic categories.
5. Compare the candidate with a baseline.
6. Apply the release policy.
7. Block the release when configured quality or regression limits are violated.

Reliab currently uses deterministic and heuristic evaluation primitives and provides pluggable adapter/evaluator interfaces for extending the platform to different systems and evaluation methods.

For implementation details, see the [README](README.md) and the documentation in [`docs/`](docs/).
