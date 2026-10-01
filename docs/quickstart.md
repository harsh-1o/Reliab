# Quickstart Guide

This guide walks you through setting up Reliab locally, running your first evaluation, and executing a headless CI/CD release gate.

---

## 1. Prerequisites

- Python 3.11 or higher
- SQLite (default) or PostgreSQL

---

## 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/harsh-1o/Reliab.git
cd Reliab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install locked dependencies and package
pip install -r requirements.lock
pip install -e . --no-deps
```

---

## 3. Database Initialization

Apply database migrations using Alembic:

```bash
python -m alembic upgrade head
```

By default, Reliab creates a local SQLite database at `rag_platform.db`.

---

## 4. Run Headless Quality Gate (CLI)

Reliab provides a built-in CLI gate runner that seeds a golden benchmark dataset and evaluates a RAG candidate:

```bash
python -m rag_platform.gate \
  --bootstrap \
  --project proj_quickstart \
  --dataset ds_quickstart_bench \
  --system-version $(git rev-parse --short HEAD 2>/dev/null || echo "v0.2.0") \
  --policy prod-default \
  --junit-xml test-results/reliab-gate.xml
```

You can also use the installed console script:

```bash
reliab-gate --bootstrap --project proj_quickstart --dataset ds_quickstart_bench --system-version v0.2.0
```

### Exit Codes
- **`0` (`PASS`)**: All metric thresholds and regression budgets satisfied.
- **`1` (`FAIL`)**: Gate violations detected (e.g., faithfulness drop or new regressions).

> **Schema Ownership Note**: `--bootstrap` is provided solely for rapid local quickstart and ephemeral tests. Production databases must run Alembic migrations (`python -m alembic upgrade head`) for schema management.

---

## 5. Launch the Web Dashboard

Start the FastAPI control plane and interactive dashboard:

```bash
python -m uvicorn rag_platform.server:app --host 127.0.0.1 --port 8080 --reload
```

Open [http://127.0.0.1:8080/dashboard](http://127.0.0.1:8080/dashboard) to:
- Inspect aggregate run metrics (Faithfulness, Recall, MRR, Citation Accuracy, Abstention).
- View per-trace evidence, extracted claims, and attribution codes.
- Compare candidate runs against production baselines.

---

## 6. Evaluating a Real RAG System (HTTP Adapter)

To evaluate an external HTTP endpoint serving your RAG system:

```bash
python -m rag_platform.gate \
  --adapter-type http \
  --endpoint-url https://api.internal.corp/rag/v1/query \
  --project proj_production \
  --dataset ds_golden_eval \
  --system-version $(git rev-parse HEAD) \
  --policy prod-default
```

The endpoint must accept a JSON payload with `query` and return `{ answer, retrieved_documents, citations }`. See [Adapters Documentation](adapters.md) for details.
