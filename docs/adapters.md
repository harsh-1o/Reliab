# Adapters & SUT Integration

Reliab connects to Systems Under Test (SUT) through an extensible adapter interface (`RagAdapter`). Adapters abstract communication protocols, concurrency controls, timeout handling, and transport security.

---

## 1. Adapter Interface

Every adapter implements the asynchronous `run` method:

```python
class RagAdapter(ABC):
    @abstractmethod
    async def run(self, test_case: TestCase, config: RunConfig) -> RagTrace:
        """Execute a single benchmark test case and return a normalized RagTrace."""
        pass
```

### Trace Output Format
Adapters return a normalized `RagTrace` object containing:
- `answer`: Generated textual response.
- `retrieved_chunks`: List of retrieved chunks (`chunk_id`, `document_id`, `text`, `similarity_score`, `rank`).
- `citations`: Extracted citations linking claim text to `document_id` and `chunk_id`.
- `latency_ms`: Total execution time in milliseconds.
- `cost_usd`: Computed invocation cost based on token consumption.
- `metadata`: Execution metadata (model name, prompt tokens, completion tokens).

---

## 2. Built-in Adapters

### 1. `HttpRagAdapter`
Connects to remote or local HTTP microservices serving RAG pipelines:
- **SSRF Defense**: Uses `SSRFProtectedTransport` to validate destination IP addresses against private networks (RFC 1918, RFC 3927) and pin sockets to prevent DNS rebinding.
- **Strict Canonical Response Validation**: Validates HTTP RAG responses through a canonical strict Pydantic model (`HttpRagResponsePayload`). Structural types, nested chunk definitions, citations with token spans, and abstention fields are strictly validated. Malformed responses (e.g. invalid chunk types, bad spans, missing required non-abstained answers) consistently become `OPS-01` operational errors with diagnostic telemetry rather than leaking unhandled exceptions.
- **Environment-Backed Secret References**: To avoid storing plaintext API tokens in the database, `HttpRagAdapter` supports secret references in header configurations (`header_secret_refs` or `${ENV_VAR}`). Plaintext headers are redacted prior to database persistence, and background execution workers resolve credentials at runtime from their environment. If a referenced environment secret is missing, execution safely produces an `OPS-01` error.
- **Retry & Idempotency Contract**: Intended for read-only evaluation requests. Outbound requests automatically include `X-Request-ID` and `Idempotency-Key` headers (`eval_{project}_{dataset}_{case}_{trace}`). Automatic retries with exponential backoff and jitter are performed for transient HTTP status codes (`429`, `502`, `503`, `504`) and connection drops without duplicating side-effects.
- **Bounded Concurrency & Clean Resource Teardown**: Throttles outbound traffic using `asyncio.Semaphore` and guarantees async client closure via asynchronous context management and `finally` cleanup.

#### Expected Request Payload
```json
{
  "question": "What is the warranty period for Model X?",
  "test_case_id": "tc_001",
  "metadata": {}
}
```

#### Expected Response Payload Contract
```json
{
  "answer": "The warranty period for Model X is 3 years or 36,000 miles [1].",
  "abstained": false,
  "abstention_reason": null,
  "retrieved_chunks": [
    {
      "chunk_id": "chk_102",
      "document_id": "doc_warranty_guide",
      "text": "Model X coverage includes a 3-year or 36,000-mile limited warranty.",
      "score": 0.94,
      "rank": 1
    }
  ],
  "citations": [
    {
      "claim_id": "cl_0",
      "claim_text": "Model X coverage includes a 3-year or 36,000-mile limited warranty.",
      "document_id": "doc_warranty_guide",
      "chunk_id": "chk_102"
    }
  ],
  "telemetry": {
    "latency_ms": 320.5
  }
}
```

---

### 2. `SyntheticRagAdapter`
Deterministic simulator used for unit testing, CI validation, and release gate calibration. Supports predefined mock behaviors:

| Mode | Behavior |
|:---|:---|
| `PERFECT` | Generates fully faithful, perfectly grounded answers with accurate citations |
| `HALLUCINATING` | Introduces unsupported claims absent from the retrieved evidence |
| `CONTRADICTING` | Produces direct predicate or numerical conflicts with the retrieved evidence |
| `RETRIEVAL_FAILURE` | Simulates retrieval misses by returning irrelevant chunks or empty results |
| `UNANSWERABLE_FAILED` | Answers questions marked as unanswerable instead of refusing |
| `HIGH_LATENCY` | Simulates slow network or LLM response times |

---

## 3. Implementing a Custom Adapter

You can implement custom adapters in Python:

```python
from rag_platform.adapters import RagAdapter
from rag_platform.models import TestCase, RunConfig, RagTrace, RetrievedChunk, CitationReference

class CustomPipelineAdapter(RagAdapter):
    def __init__(self, my_rag_pipeline):
        self.pipeline = my_rag_pipeline

    async def run(self, test_case: TestCase, config: RunConfig) -> RagTrace:
        result = await self.pipeline.query_async(test_case.question)
        
        return RagTrace(
            test_case_id=test_case.id,
            question=test_case.question,
            answer=result.text,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.doc_id,
                    text=c.content,
                    similarity_score=c.score,
                    rank=idx,
                )
                for idx, c in enumerate(result.chunks)
            ],
            citations=[
                CitationReference(document_id=cite.doc_id, chunk_id=cite.chunk_id)
                for cite in result.citations
            ],
            latency_ms=result.elapsed_ms,
        )
```

### 4. Process-Local Python Adapter Registry
To evaluate internal Python pipelines without exposing the platform to arbitrary remote code execution via HTTP payloads:
- Callables are registered explicitly in code:
  ```python
  from rag_platform.adapters import PythonAdapterRegistry
  PythonAdapterRegistry.register("my_model_v1", my_eval_fn)
  ```
- **Multi-Process Architecture**: When running distributed workers across separate processes, callables must be registered during application bootstrap in each worker process.
- **Security**: The REST API accepts only pre-registered string identifiers (`"adapter_type": "python"`, `"adapter_config": {"model_name": "my_model_v1"}`). Arbitrary code submission via JSON or network payload is strictly prohibited.

