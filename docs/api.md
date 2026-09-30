# REST API Reference

Reliab exposes a RESTful API built on FastAPI for managing projects, benchmark datasets, evaluation runs, and regression comparisons.

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/v1/health` | Service health status and timestamp |
| `POST` | `/v1/projects` | Register a new project workspace |
| `GET` | `/v1/projects` | List projects (with pagination `limit`, `offset`) |
| `POST` | `/v1/datasets` | Create draft benchmark dataset |
| `POST` | `/v1/datasets/{id}/publish` | Publish and lock benchmark dataset version (computes SHA-256) |
| `POST` | `/v1/datasets/{id}/cases/bulk` | Bulk insert benchmark test cases into draft dataset |
| `POST` | `/v1/runs` | Launch evaluation run (`async_exec=true` returns `202 Accepted` + `QUEUED`) |
| `POST` | `/v1/runs/{id}/cancel` | Cancel an in-flight evaluation run |
| `GET` | `/v1/runs` | List evaluation runs with summary metrics |
| `GET` | `/v1/runs/{id}` | Fetch run details, provenance manifest, and status |
| `GET` | `/v1/runs/{id}/traces` | List individual traces with metrics and failure attributions |
| `POST` | `/v1/compare` | Compare candidate run against baseline run (4-way transitions) |
| `POST` | `/v1/maintenance/cleanup` | Purge expired runs based on data retention policy |
| `GET` | `/dashboard` | Interactive web engineering console |

---

## 2. Authentication & Security

When `RAG_AUTH_ENABLED=true` is set (default in production):
- **API Key**: Pass the configured secret key in the `X-API-Key` or `Authorization: Bearer <key>` HTTP header.
- **Web UI Session**: Browser clients authenticate via `POST /v1/auth/session` with `{"api_key": "..."}` to obtain an opaque random session token (`sess_<random>`) stored in an `HttpOnly`, `SameSite=Lax`, `Secure` cookie (`reliab_session` / `session_id`). The raw API key is never stored in browser cookies. Call `POST /v1/auth/logout` to destroy the server-side session.
- **Trusted Proxies**: Set `TRUSTED_PROXIES=10.0.0.0/8,127.0.0.1` when operating behind reverse proxies. Untrusted client IP spoofing in `X-Forwarded-For` is automatically rejected.

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8080/v1/projects
```

---

## 3. Key Endpoint Examples

### Launch Evaluation Run (`POST /v1/runs`)
```bash
curl -X POST http://localhost:8080/v1/runs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "project_id": "proj_prod",
    "dataset_id": "ds_golden_v1",
    "system_version": "git-commit-abc1234",
    "adapter_type": "http",
    "adapter_config": {
      "endpoint_url": "https://api.internal/rag/query",
      "timeout_seconds": 15.0
    },
    "async_exec": true
  }'
```

#### Response (`202 Accepted`)
```json
{
  "id": "run_01hx5m8q3v9",
  "project_id": "proj_prod",
  "dataset_id": "ds_golden_v1",
  "status": "QUEUED",
  "created_at": "2026-09-26T12:00:00Z"
}
```

---

### Compare Candidate vs. Baseline (`POST /v1/compare`)
```bash
curl -X POST http://localhost:8080/v1/compare \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "baseline_run_id": "run_baseline_123",
    "candidate_run_id": "run_candidate_456"
  }'
```

#### Response (`200 OK`)
```json
{
  "baseline_run_id": "run_baseline_123",
  "candidate_run_id": "run_candidate_456",
  "metric_deltas": {
    "faithfulness": -0.04,
    "recall_at_5": 0.02,
    "citation_accuracy": -0.01
  },
  "transitions": {
    "NEW_FAILURE": 2,
    "RECOVERED": 5,
    "UNCHANGED_PASS": 88,
    "UNCHANGED_FAIL": 5
  },
  "verdict": "FAIL",
  "violations": [
    "2 new regressions detected on previously passing test cases."
  ]
}
```
