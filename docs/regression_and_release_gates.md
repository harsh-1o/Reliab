# Regression Detection & Release Gates

Reliab prevents silent quality regressions in RAG applications by comparing candidate evaluation runs against baseline runs and enforcing statistical release policies in CI/CD pipelines.

---

## 1. Release Policy Configuration

Release policies define quality floors and allowable regression budgets.

### Policy Definition
```python
from rag_platform.models import ReleasePolicy

policy = ReleasePolicy(
    id="prod-default",
    name="Production Default Gate Policy",
    min_faithfulness=0.90,          # Minimum acceptable average faithfulness
    min_retrieval_recall=0.85,      # Minimum acceptable Recall@5
    min_citation_accuracy=0.90,     # Minimum acceptable citation accuracy
    max_hallucination_rate=0.05,    # Maximum allowable rate of unsupported claims
    max_latency_regression_pct=20.0,# Max latency increase compared to baseline (%)
    max_cost_regression_pct=25.0,   # Max cost increase compared to baseline (%)
    min_cost_budget_usd=0.05,       # Absolute floor preventing zero-baseline division errors
    allow_new_failures=False,       # Block gate if any previously passing case now fails
)
```

---

## 2. Four-Way Per-Case Transition Analysis

Averaging metrics across hundreds of test cases can mask regressions if improvements on some queries cancel out regressions on others.

Reliab tracks case-level transitions between the baseline and candidate run:

| Transition State | Definition | Gate Impact |
|:---|:---|:---|
| **`NEW_FAILURE`** | Case passed in baseline, but fails in candidate | **Critical Blocker**; violates gate when `allow_new_failures=False` |
| **`RECOVERED`** | Case failed in baseline, but passes in candidate | Quality improvement |
| **`UNCHANGED_PASS`** | Case passed in baseline and passes in candidate | Maintained quality standard |
| **`UNCHANGED_FAIL`** | Case failed in baseline and still fails in candidate | Known defect / technical debt |

---

## 3. Headless CI/CD CLI Gate

Run quality checks directly from terminal or CI runner:

```bash
python -m rag_platform.gate \
  --project proj_production \
  --dataset ds_golden_bench \
  --system-version $(git rev-parse HEAD) \
  --policy prod-default \
  --adapter-type http \
  --endpoint-url https://api.staging.internal/rag/query \
  --junit-xml test-results/reliab-gate.xml
```

You can also use the installed console script:

```bash
reliab-gate \
  --project proj_production \
  --dataset ds_golden_bench \
  --system-version $(git rev-parse HEAD) \
  --policy prod-default
```

### CLI Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `--project` | string | *required* | Target project workspace ID |
| `--dataset` | string | *required* | Published dataset ID to benchmark against |
| `--system-version` | string | *required* | Git commit SHA or version tag of candidate RAG |
| `--policy` | string | `prod-default` | Release policy ID to evaluate against |
| `--adapter-type` | `synthetic` \| `http` | `synthetic` | Adapter communication protocol |
| `--endpoint-url` | string | `None` | Target HTTP URL when `--adapter-type http` |
| `--mock-mode` | enum | `PERFECT` | Mock behavior when `--adapter-type synthetic` |
| `--junit-xml` | string | `None` | Destination path for JUnit XML report |
| `--bootstrap` | flag | `False` | Auto-initialize database and seed baseline dataset |

### Exit Codes
- **`0`**: Gate status is `PASS`. All thresholds and budgets satisfied.
- **`1`**: Gate status is `FAIL`. One or more violations detected.

---

## 4. JUnit XML CI/CD Integration

When `--junit-xml` is provided, Reliab generates test reports adhering to standard JUnit XML schemas, enabling native test visualizers in GitHub Actions, GitLab CI, Jenkins, and Azure DevOps.

Example GitHub Actions step:

```yaml
- name: Run Reliab Release Gate
  run: |
    python -m rag_platform.gate \
      --bootstrap \
      --project proj_ci \
      --dataset ds_ci_benchmark \
      --system-version ${{ github.sha }} \
      --policy prod-default \
      --junit-xml test-results/reliab-gate.xml

- name: Publish Test Results
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: gate-test-results
    path: test-results/reliab-gate.xml
```
