# Reliab

<p align="center">
  <img src="assets/brand/reliab-wordmark.png" alt="Reliab" width="360" />
</p>

<p align="center">
  <strong>Deterministic evaluation and release gates for RAG & LLM systems.</strong>
</p>

<p align="center">
  <a href="https://github.com/harsh-1o/Reliab/actions/workflows/reliab-evaluation.yml"><img src="https://github.com/harsh-1o/Reliab/actions/workflows/reliab-evaluation.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://alembic.sqlalchemy.org/"><img src="https://img.shields.io/badge/Migrations-Alembic-red.svg" alt="Alembic"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

---

## What is Reliab?

Reliab is developer infrastructure for evaluating, diagnosing, and release-gating RAG and LLM applications. It evaluates retrieval, grounding, citations, and abstention, attributes failures to actionable diagnostic codes, and detects regressions against a baseline.

It is designed to be **deterministic, reproducible, and CI-friendly**: run it locally, use the web dashboard to inspect results, or run the headless gate in CI/CD and fail a build when a release violates your policy.

## Why Reliab?

RAG quality can regress even when aggregate metrics look healthy. A prompt change, retriever update, embedding change, or context change can turn individual cases from passing to failing without an obvious signal.

Reliab gives those failures a structured evaluation path:

- **Retrieval** — Recall@K, MRR, and contextual precision against golden evidence.
- **Grounding** — claim-level support and contradiction checks.
- **Citations** — verify that cited evidence supports the relevant claims.
- **Abstention** — evaluate whether unanswerable queries are handled appropriately.
- **Failure attribution** — map failures to diagnostic codes with remediation context.
- **Regression detection** — track per-case transitions against a baseline.
- **Release gates** — enforce metric thresholds, complete dataset coverage, and regression budgets with CI-friendly exit codes.

## How it works

```text
             ┌──────────────────┐
             │    RAG / LLM     │
             │      System      │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │    Evaluation    │
             │      Engine      │
             └────────┬─────────┘
                      │
             ┌────────┴─────────┐
             ▼                  ▼
      ┌──────────────┐   ┌──────────────┐
      │   Metrics    │   │    Failure   │
      │ & Grounding  │   │ Attribution  │
      └──────┬───────┘   └──────┬───────┘
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │    Regression    │
             │    Detection     │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │   Release Gate   │
             │    PASS / FAIL   │
             └──────────────────┘
```

1. **Execute** — query a RAG/LLM system through an adapter.
2. **Evaluate** — measure retrieval, grounding, citations, and abstention.
3. **Attribute** — classify failures using Reliab’s diagnostic taxonomy.
4. **Compare** — compare candidate results with a golden baseline.
5. **Gate** — enforce the configured release policy and return a CI-friendly exit code.

### Evaluation approach

Reliab currently uses deterministic and heuristic evaluation primitives such as lexical claim decomposition, normalized token overlap, numerical/entity checks, and contradiction detection. This keeps CI evaluation predictable and reproducible without requiring an external LLM judge.

The evaluator and adapter layers are pluggable, so teams can add stronger semantic evaluators or connect different RAG systems when their use case requires them.

## Quick Start

### Install

```bash
git clone https://github.com/harsh-1o/Reliab.git
cd Reliab

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.lock
pip install -e . --no-deps
python -m alembic upgrade head
```

### Run an evaluation gate

```bash
reliab-gate \
  --bootstrap \
  --project proj_quickstart \
  --dataset ds_quickstart_bench \
  --system-version v0.1.0 \
  --policy prod-default
```

Exit codes are CI-friendly:

- `0` — gate passed
- `1` — gate violations detected

> **Note on Migration Ownership**: `--bootstrap` is strictly a local/offline development convenience for auto-initializing ephemeral SQLite tables and fixture datasets during initial setup. In staging and production environments, **Alembic is the authoritative schema authority** (`python -m alembic upgrade head`). Production deployment pipelines should never use `--bootstrap` as a substitute for tracked migrations.

### Launch the dashboard

```bash
python -m uvicorn rag_platform.server:app --host 127.0.0.1 --port 8080 --reload
```

Open `http://127.0.0.1:8080/dashboard` to inspect runs, traces, metrics, and evaluation results.

For the complete setup and integration walkthrough, see the [Quickstart Guide](docs/quickstart.md).

## CI/CD

Reliab can run as a headless release gate and emit JUnit XML for CI systems:

```bash
reliab-gate \
  --project proj_example \
  --dataset ds_golden_v1 \
  --system-version git-a3f91c2 \
  --policy prod-default \
  --junit-xml test-results/gate.xml
```

A gate can fail because of metric thresholds, insufficient dataset coverage, omitted baseline test cases, or newly introduced per-case regressions, even when aggregate metrics remain above their minimum thresholds.

## Architecture

Reliab separates system execution, evaluation, regression tracking, and control/storage concerns into modular layers. The platform includes adapters for SUT integration, deterministic evaluators, a failure-attribution subsystem, a FastAPI control plane, durable background workers, and persistent run provenance.

See [Platform Architecture](docs/architecture.md) for the detailed component and concurrency model.

## Documentation

- [Quickstart](docs/quickstart.md) — setup and first evaluation
- [Architecture](docs/architecture.md) — platform design and workflow
- [Metrics & Methodology](docs/metrics.md) — evaluation metrics and applicability
- [Failure Taxonomy](docs/failure_taxonomy.md) — diagnostic codes and remediation
- [Regression & Release Gates](docs/regression_and_release_gates.md) — baselines, policies, and CI/CD
- [Adapters](docs/adapters.md) — SUT integration and adapter behavior
- [REST API](docs/api.md) — API endpoints and authentication
- [Security & Provenance](docs/security.md) — SSRF, secret handling, and provenance
- [Development](docs/development.md) — testing, linting, typing, and contribution workflow

## Development

Run the core validation checks locally:

```bash
python -m pytest tests/ -v
python -m ruff check src/
python -m mypy src/rag_platform/
python -m alembic check
```

See the [Development Guide](docs/development.md) for the full workflow.

## License

Reliab is released under the [MIT License](LICENSE).

---

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=harsh-1o&repo=Reliab&label=Visitors&color=0e75b6&style=flat" alt="Repository visitors" />
</p>
