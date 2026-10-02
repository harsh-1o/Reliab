# REST API Reference

Reliab's FastAPI control plane manages projects, benchmark datasets, evaluation runs, traces, failures, authentication, comparison, maintenance, and the dashboard.

## Routes implemented by the current server

| Method | Route | Purpose |
|---|---|---|
| GET | `/health`, `/v1/health` | Health |
| POST | `/v1/auth/session` | API key → browser session |
| POST | `/v1/auth/logout` | Invalidate browser session |
| POST/GET | `/v1/auth/keys` | Create/list API keys (admin) |
| POST | `/v1/auth/keys/{key_hash}/revoke` | Revoke key (admin) |
| POST | `/v1/auth/keys/{key_hash}/rotate` | Rotate key (admin) |
| POST | `/v1/maintenance/cleanup` | Cleanup (admin) |
| POST/GET | `/v1/projects` | Create/list projects |
| GET | `/v1/projects/{project_id}` | Get project |
| POST | `/v1/datasets` | Create draft dataset |
| POST | `/v1/datasets/{dataset_id}/cases/bulk` | Add cases to draft |
| GET | `/v1/datasets` | List datasets |
| GET | `/v1/datasets/{dataset_id}` | Get dataset |
| GET/POST | `/v1/runs` | List/create runs |
| POST | `/v1/runs/{run_id}/cancel` | Cancel active run |
| GET | `/v1/runs/{run_id}` | Get run |
| GET | `/v1/runs/{run_id}/traces` | Get traces |
| GET | `/v1/failures` | Get failure attributions |
| POST | `/v1/compare` | Baseline/candidate comparison |
| POST | `/v1/demo-run` | Local demonstration run |
| GET | `/dashboard` | Web dashboard |

There is currently **no public dataset-publish route**. Runs require a dataset that is already published.

## Authentication

API clients use either:

```http
X-API-Key: <key>
Authorization: Bearer <key>
```

Browser clients POST the key to `/v1/auth/session`. The server returns an opaque random `session_id` cookie with HttpOnly/SameSite=Strict attributes. The raw API key is not stored in browser storage.

API keys and sessions are stored as hashes. Revoking/rotating an API key invalidates linked browser sessions.

## Creating a run

```http
POST /v1/runs
Content-Type: application/json
X-API-Key: <key>
```

A run identifies a project, published dataset/version, system version, adapter, and release policy. Dataset project ownership, version, checksum, and provenance integrity are checked before persistence.

Set `async_exec=true` for durable worker execution. The run is queued and later claimed by a worker lease.

## Run lifecycle

```text
CREATED → QUEUED → RUNNING → COMPLETED
                         ├── FAILED
                         └── CANCELLED
```

Terminal runs are immutable. Workers use leases and heartbeats so stale workers cannot overwrite recovered runs.

## Traces and summaries

`GET /v1/runs/{id}` returns persisted run information, provenance, summary, coverage, and gate result.

`GET /v1/runs/{id}/traces?failure_only=true` filters traces to failures.

Completed run summaries/gate results are persisted; the API does not reconstruct a weaker summary that could lose authoritative coverage information.

## Compare

```json
{
  "baseline_run_id": "run_baseline",
  "candidate_run_id": "run_candidate"
}
```

Comparison is case-aware. It distinguishes `NEW_FAILURE`, `CANDIDATE_MISSING`, `BASELINE_MISSING`, `RECOVERED`, `UNCHANGED_PASS`, `UNCHANGED_FAIL`, and `NOT_APPLICABLE`.

Missing cases are determined from explicit case-ID membership, not from a `None` metric score.

## Maintenance

```http
POST /v1/maintenance/cleanup?trace_retention_days=90
```

The API bounds the retention parameter and requires administrator access.

## Important invariants

- Runs require published datasets.
- Dataset and run projects must match.
- Dataset versions and provenance checksums must match.
- Published datasets are immutable.
- Python adapters reference pre-registered server-side callables; arbitrary code is never accepted from JSON.
- Resources are project-scoped.
- Unknown named release policies are rejected on supported policy-selection paths.

## CLI gate

```bash
python -m rag_platform.gate --project proj_prod --dataset ds_gold \
  --system-version "$(git rev-parse HEAD)" --policy prod-default \
  --adapter-type http --endpoint-url https://rag.example/query
```

Exit code `0` means PASS; `1` means FAIL.
