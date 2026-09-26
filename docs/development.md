# Development & Testing Guide

This guide covers setting up your local environment for contributing to Reliab, running verification suites, and understanding project standards.

---

## 1. Local Environment Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
```bash
git clone https://github.com/harsh-1o/Reliab.git
cd Reliab

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install locked dependencies
pip install -r requirements.lock
pip install -e . --no-deps

# Apply database migrations
python -m alembic upgrade head
```

---

## 2. Running Local Services

### Start Development API & Dashboard
```bash
python -m uvicorn rag_platform.server:app --host 127.0.0.1 --port 8080 --reload
```
The web dashboard is accessible at `http://127.0.0.1:8080/dashboard`.

### Start Background Durable Worker
```bash
python -m rag_platform.worker
# Or via console script:
reliab-worker
```

---

## 3. Running Verification Suites

All four verification checks must pass before opening pull requests:

### 1. Test Suite (pytest)
```bash
python -m pytest tests/ -v
```

### 2. Linting (Ruff)
```bash
python -m ruff check src/
```

### 3. Static Type Checking (mypy)
```bash
python -m mypy src/rag_platform/
```

### 4. Database Schema Migration Consistency
```bash
python -m alembic check
```

---

## 4. Headless Quality Gate Verification

Test both release gate pass and failure scenarios locally:

```bash
# PASS scenario (PERFECT mock adapter)
python -m rag_platform.gate \
  --bootstrap \
  --project proj_dev \
  --dataset ds_dev_bench \
  --system-version dev-local \
  --policy prod-default

# FAIL scenario (Simulated Hallucinations)
python -m rag_platform.gate \
  --bootstrap \
  --project proj_dev \
  --dataset ds_dev_bench \
  --system-version dev-local \
  --policy prod-default \
  --mock-mode HALLUCINATING
```

---

## 5. Known Limitations & Architecture Notes

1. **Lexical Claim Grounding**: Claim verification currently uses lexical token alignment, synonym normalization, and rule-based numerical/antonym conflict detection. It is a fast, deterministic baseline, not a full cross-encoder semantic entailment model.
2. **ML Classifier Probability Estimates**: The ML classifier outputs raw tree probabilities from `predict_proba`. These are model probability estimates, not mathematically calibrated Bayesian posterior probabilities.
3. **Database Leases vs Distributed Message Queues**: The durable worker utilizes PostgreSQL/SQLite lease locking with LISTEN/NOTIFY and polling backoff. For hyper-scale multi-datacenter topologies, an external message broker can be slotted in via the adapter interface.
4. **Starlette/httpx Test Client Deprecation**: The test suite uses `starlette.testclient` with `httpx` 0.28.1, which emits a deprecation warning (`install httpx2 instead`). This is a Starlette 1.3.1 internal deprecation. Upgrading requires a coordinated bump of FastAPI, Starlette, and httpx; deferred until a stable upgrade path is confirmed. No functional impact.

---

## 6. Engineering Roadmap

- [ ] **Cross-Encoder Semantic Entailment**: Optional local model (e.g. DeBERTa-v3-NLI) for high-fidelity semantic verification.
- [ ] **Multi-Tenant Organizations**: Fine-grained permissions per organization and project workspace.
- [ ] **Distributed Execution Engine**: Optional Redis/Celery queue integration for hyper-scale benchmark runs across heterogeneous worker pools.
- [ ] **Human-in-the-Loop Active Learning**: Direct UI triage interface for annotating ambiguous failure predictions.
