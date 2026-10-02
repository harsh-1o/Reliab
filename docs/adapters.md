# Adapters & System-Under-Test Contract

Reliab evaluates an existing RAG/LLM system. It does not host or train the evaluated model. Adapters are integration boundaries that convert a benchmark `TestCase` into a normalized `RagTrace`.

## Canonical interface

```python
class RagAdapter(Protocol):
    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        ...
```

A normalized trace contains the test-case ID/question, answer or abstention, retrieved chunks, citations, latency, tokens/cost, model metadata, telemetry, and operational error information where applicable.

## Adapter types

| Type | Purpose | Trust boundary |
|---|---|---|
| `synthetic` | deterministic CI/self-test | Built in |
| `http` | external/local RAG service | SSRF-protected network |
| `python` | in-process pipeline | trusted server-side registry |

Adapters are selected by **interface**, not model family. A GPT/Llama/local/custom RAG service does not need a model-specific Reliab adapter if it implements one of these interfaces.

## HTTP contract

Request:

```json
{
  "question": "What is the warranty period for Model X?",
  "test_case_id": "tc_001",
  "metadata": {}
}
```

Response:

```json
{
  "answer": "The warranty is 3 years.",
  "abstained": false,
  "abstention_reason": null,
  "retrieved_chunks": [
    {
      "document_id": "doc_warranty",
      "chunk_id": "chunk_01",
      "text": "Model X has a 3-year limited warranty.",
      "score": 0.94,
      "rank": 1,
      "metadata": {}
    }
  ],
  "citations": [
    {
      "claim_id": "cl_0",
      "claim_text": "The warranty is 3 years.",
      "document_id": "doc_warranty",
      "chunk_id": "chunk_01"
    }
  ],
  "telemetry": {"latency_ms": 320}
}
```

Strict validation requires a JSON object; boolean `abstained`; a string answer for non-abstained responses; structured chunks/citations; string chunk document IDs; and valid citation spans when supplied. Citation spans are exactly two non-negative integers with `start <= end`. Unknown provider fields are ignored.

Malformed responses become operational adapter failures rather than plausible evaluation data.

## HTTP retries and identity

Transient 429/502/503/504 responses and connection drops may be retried with backoff. Requests include `X-Request-ID` and `Idempotency-Key`.

The evaluated endpoint should therefore make repeated read/evaluation requests safe.

HTTP execution also uses SSRF protection, bounded concurrency, timeout handling, and resource cleanup.

## HTTP secrets

Use environment-backed secret references rather than persisting long-lived provider credentials in run configuration. Workers resolve those references at runtime.

## Python adapter

Register a trusted callable:

```python
PythonAdapterRegistry.register("my_rag_pipeline", my_eval_fn)
```

The REST payload contains only:

```json
{
  "adapter_type": "python",
  "adapter_config": {"model_name": "my_rag_pipeline"}
}
```

No Python source, pickle, or callable is accepted from the API.

The registry is process-local. Every worker that may execute the adapter must register it during bootstrap.

## Synthetic adapter

Available deterministic modes include:

- `PERFECT`
- `HALLUCINATING`
- `CONTRADICTING`
- `RETRIEVAL_FAILURE`
- `UNANSWERABLE_FAILED`
- `HIGH_LATENCY`

These calibrate the evaluator/gate; they are not simulations of general model quality.

## Custom adapters

Normalize provider-specific data before returning a `RagTrace`. Preserve real document IDs, chunk IDs, and ranks. Do not fabricate evidence to satisfy the schema.

## Failure semantics

An adapter/network failure is an operational failure, not automatically a hallucination:

```text
timeout / malformed provider response
          ↓
       OPS-01
          ↓
operational evaluation evidence
```

This separation prevents provider outages from being mislabeled as generation defects.

## Adapter checklist

Production adapters should bound timeouts/concurrency, clean up resources, preserve case identity/ranks, avoid plaintext secrets, make retries safe, and never execute arbitrary caller-supplied code.
