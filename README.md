# Reliab

<p align="center">
  <img src="assets/brand/reliab-wordmark.png" alt="Reliab" width="420" />
</p>

<p align="center">
  <strong>Developer infrastructure for testing, failure diagnosis, regression gating, and reliability engineering in RAG &amp; LLM systems.</strong>
</p>

<p align="center">
  <a href="https://github.com/harsh-1o/reliab/actions/workflows/reliab-evaluation.yml"><img src="https://github.com/harsh-1o/reliab/actions/workflows/reliab-evaluation.yml/badge.svg" alt="Reliab Quality Gate &amp; CI/CD Pipeline"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/pydantic-v2.0%2B-e92063.svg" alt="Pydantic"></a>
  <a href="https://alembic.sqlalchemy.org/"><img src="https://img.shields.io/badge/migrations-Alembic-red.svg" alt="Alembic"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

## What is Reliab?

Reliab is developer infrastructure designed to evaluate, diagnose, and gate Retrieval-Augmented Generation (RAG) and LLM applications. It provides automated failure attribution, claim-level evaluation, and statistical regression testing to prevent silent quality degradation.

Unlike surface-level LLM evaluation tools that rely on ungrounded heuristics or opaque model judges, Reliab decomposes answers into atomic claims, measures retrieval and generation independently, and diagnoses trace failures into a canonical taxonomy.

Reliab runs locally, as a standalone evaluation service with a web dashboard, or as a headless CLI gate in CI/CD pipelines to block failing pull requests before they reach production.

## Why Reliab?

RAG systems fail across multiple interdependent layers, making root-cause diagnosis difficult when looking only at final model outputs:

- **Retrieval Failures**: The retriever misses golden documents, surfaces low-similarity noise, or drowns out key evidence with distractors.
- **Hallucinations & Grounding Failures**: The generation model asserts claims unsupported by the retrieved context or directly contradicts evidence.
- **Citation Failures**: Citations point to irrelevant chunks, misattribute claims, or omit references altogether.
- **Regressions**: Upstream prompt tweaks, chunking updates, or embedding changes silently break previously working queries while aggregate averages stay flat.
- **Release Gating**: Engineering teams lack automated, fail-closed gates in CI/CD to prevent quality degradation before deployment.

Reliab isolates each failure mode with actionable diagnostic codes and enforces statistical release policies on every commit.

## How it works

Reliab executes a deterministic evaluation and release gating pipeline:

```text
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────────────┐
│  RAG / LLM SUT  │ ──> │ Evaluation Engine │ ──> │ Metrics & Failure Attribution│
└─────────────────┘     └───────────────────┘     └──────────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────────────┐
│  PASS / FAIL    │ <── │   Release Gate    │ <── │     Regression Detection     │
│ (Exit Code 0/1) │     │ (JUnit XML / CLI) │     │    (4-Way Case Transitions)  │
└─────────────────┘     └───────────────────┘     └──────────────────────────────┘
```

1. **Execute**: The SUT is queried via Python, synthetic mocks, or SSRF-protected HTTP endpoints.
2. **Evaluate**: Traces are stripped of secrets and evaluated across retrieval, claim grounding, citations, and abstention.
3. **Attribute**: Failures are mapped to root-cause diagnostic codes (`RET-01`, `GEN-01`, `CIT-02`, etc.) with supporting evidence.
4. **Compare**: Candidate metrics and individual test cases are compared against a golden baseline run.
5. **Gate**: Statistical thresholds and regression budgets determine the release verdict, returning exit code `0` or `1`.

### Evaluation Methodology & Architecture

Reliab currently provides **deterministic and heuristic evaluation primitives** (lexical claim decomposition, normalized token overlap, numerical/entity verification, and antonym opposition detection) designed for predictable, reproducible, zero-cost CI/CD quality gating without external API dependencies or nondeterministic LLM scoring variance.

For applications requiring deep semantic inference beyond lexical grounding, Reliab features a **pluggable evaluator and adapter architecture** allowing engineering teams to seamlessly plug in neural cross-encoders, natural language inference (NLI) models, or model-graded evaluators.

## Key capabilities

- **Retrieval Evaluation**: Measures Recall@K, Mean Reciprocal Rank (MRR), and Contextual Precision against golden documents.
- **Claim-Level Grounding**: Breaks generated answers into atomic propositions and verifies lexical support and contradictions against context.
- **Citation Evaluation**: Verifies that citations map directly to evidence chunks supporting the specific proposition.
- **Abstention Evaluation**: Checks that unanswerable queries trigger clean refusals rather than hallucinations, while penalizing false refusals.
- **Failure Attribution**: Maps trace failures to a canonical taxonomy with primary codes, contributing codes, and actionable remediation guidance.
- **Regression Detection**: Tracks 4-way per-case transitions (`NEW_FAILURE`, `RECOVERED`, `UNCHANGED_PASS`, `UNCHANGED_FAIL`) to catch masked regressions.
- **CI/CD Release Gates**: Headless CLI tool (`reliab-gate`) with fail-closed exit codes and standard JUnit XML reporting.
- **Reproducible Runs**: Immutable provenance manifests capturing dataset SHA-256 hashes, system versions, and environment metadata.
- **Security & SSRF Protection**: Fail-closed authentication, recursive secret sanitization in traces, and socket-pinned SSRF defense for HTTP adapters.

## Quickstart

### 1. Installation

```bash
# Clone and enter repository
git clone https://github.com/harsh-1o/Reliab.git
cd Reliab

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install locked dependencies
pip install -r requirements.lock
pip install -e . --no-deps

# Apply database migrations
python -m alembic upgrade head
```

### 2. Run Headless Quality Gate

Run a self-contained evaluation gate with automatic database bootstrapping:

```bash
python -m rag_platform.gate \
  --bootstrap \
  --project proj_ci \
  --dataset ds_ci_benchmark \
  --system-version $(git rev-parse --short HEAD 2>/dev/null || echo "v0.2.0") \
  --policy prod-default
```

### 3. Launch the Dashboard

```bash
python -m uvicorn rag_platform.server:app --host 127.0.0.1 --port 8080 --reload
```
Open [http://127.0.0.1:8080/dashboard](http://127.0.0.1:8080/dashboard) to inspect runs, traces, and metrics.

## Example

Evaluating a candidate RAG revision against a production baseline:

```bash
# Run candidate evaluation against release policy
reliab-gate \
  --project proj_finance_assistant \
  --dataset ds_golden_v1 \
  --system-version git-a3f91c2 \
  --policy prod-default \
  --junit-xml test-results/gate.xml
```

### Evaluation Output

```text
============================================================
RELIAB CI/CD RELEASE QUALITY GATE: [FAIL]
Candidate Run ID: run_01hx5m8q3v9 | Policy: prod-default
============================================================

✓ Faithfulness:         0.92  (Threshold: >= 0.90)
✓ Evidence Retrieval:   0.88  (Threshold: >= 0.85)
✓ Citation Accuracy:    0.94  (Threshold: >= 0.90)
✓ Abstention Accuracy:  1.00  (Threshold: >= 0.90)
✗ New Regressions:         2  (Budget: 0 allowed)

VIOLATIONS DETECTED (1):
  [FAIL] [regression_budget]: 2 test cases suffered NEW_FAILURE transitions
         - Case tc_tax_credit_04: RET-02 (Low-Precision Retrieval)
         - Case tc_quarterly_rev: GEN-01 (Factual Hallucination)

RESULT: FAIL (Exit Code 1)
============================================================
```

The pull request is blocked because previously passing test cases regressed, even though aggregate faithfulness exceeded the global threshold.

## Architecture

Reliab is structured into modular layers separating execution, evaluation, regression tracking, and storage:

- **Data Plane**: Versioned datasets with cryptographic SHA-256 checksums, adapter registry, and SSRF-safe networking.
- **Evaluation Plane**: Asynchronous evaluators with decoupled metric applicability (precondition misses emit `NOT_APPLICABLE`).
- **Attribution Subsystem**: Deterministic diagnostic engine pairing failures with remediation advice and an active learning triage classifier.
- **Control Plane**: FastAPI service, background durable workers with database-backed lease locking, and server-rendered console.

For the full architectural diagram, component breakdown, and concurrency model, see [Platform Architecture](docs/architecture.md).

## Documentation

Comprehensive technical documentation is available in the [`docs/`](docs/) directory:

- [Quickstart Guide](docs/quickstart.md) — Detailed setup, database configuration, and first evaluation.
- [Platform Architecture](docs/architecture.md) — Subsystems, pipeline workflow, durable worker, and design tenets.
- [Metrics & Methodology](docs/metrics.md) — Applicability matrix, Wilson score confidence intervals, and claim grounding.
- [Failure Taxonomy](docs/failure_taxonomy.md) — Canonical diagnostic codes (`RET-01` to `OPS-01`), conditions, and remediation.
- [Regression & Release Gates](docs/regression_and_release_gates.md) — Release policies, 4-way transitions, and CI/CD integration.
- [Adapters & SUT Integration](docs/adapters.md) — HTTP adapter, Synthetic simulator modes, and custom adapter interface.
- [REST API Reference](docs/api.md) — Endpoints, request schemas, API keys, and session authentication.
- [Security & Provenance](docs/security.md) — SSRF protection, secret scrubbing, and run manifests.
- [Development & Testing](docs/development.md) — Local testing, linting (Ruff), type checks (mypy), and roadmap.

## Development

Run local checks and validation suites:

```bash
# Run unit, concurrency, and integration tests (200+ tests)
python -m pytest tests/ -v

# Run linting
python -m ruff check src/

# Run type checking
python -m mypy src/rag_platform/

# Check database migration consistency
python -m alembic check
```

For detailed contributor guidelines and testing scenarios, see [Development Guide](docs/development.md).

## License

This project is licensed under the [MIT License](LICENSE).
