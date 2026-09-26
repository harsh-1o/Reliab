# Security, Privacy & Provenance

Reliab is built with enterprise defense-in-depth principles to safely evaluate internal RAG applications and manage sensitive test traces.

---

## 1. Multi-Layer Security Architecture

### Fail-Closed Authentication
- **Default Enabled**: `AUTH_ENABLED=true` is the default configuration.
- **Environment Gating**: The application rejects startup if `DEV_MODE=true` is detected in staging or production environments.
- **Hashed API Keys**: API keys are securely hashed using cryptographic primitives before comparison to mitigate timing attacks.
- **Session Protection**: Web UI sessions use HTTP-only, SameSite cookies with strict expiration.

### Socket-Level SSRF Defense
When evaluating external HTTP RAG endpoints, attackers or rogue configurations could target internal infrastructure (e.g. AWS/GCP instance metadata services at `169.254.169.254` or internal microservices).

Reliab implements `SSRFProtectedTransport`:
1. **Pre-flight DNS Resolution**: Resolves the target hostname before initiating the HTTP handshake.
2. **IP Range Blocking**: Rejects private, loopback, link-local, and reserved IPv4/IPv6 ranges (RFC 1918, RFC 3927, RFC 4193).
3. **Socket IP Pinning**: Binds the TCP socket connection directly to the pre-validated IP address, defeating DNS rebinding (Time-of-Check to Time-of-Use) attacks.

### Recursive Secret Sanitization
Traces often capture real user queries or raw LLM completions that contain accidentally leaked credentials.

Before any trace, metric, or attribution evidence is persisted:
- A recursive scrubbing filter traverses all dictionary keys, lists, and strings.
- Automatically redacts API keys (`sk-...`, `Bearer ...`, `token`), JWT strings, database connection strings, and common authentication headers.

### Tenant & Workspace Isolation
- Projects enforce strict boundary isolation.
- Datasets, runs, and traces are scoped to project IDs, preventing cross-tenant leakage.

---

## 2. Reproducible Run Manifests & Provenance

Every evaluation run creates an immutable `RunProvenance` record containing:
- **`dataset_checksum`**: SHA-256 hash of all test cases in the dataset version.
- **`rag_version`**: Evaluated system version or Git commit SHA.
- **`adapter_config`**: Configuration payload stripped of API keys and credentials.
- **`dependency_lock_hash`**: SHA-256 hash of `requirements.lock` ensuring reproducible dependencies.
- **`environment_info`**: Python runtime version, platform architecture, and worker identifier.

---

## 3. Adversarial & Robustness Benchmarking

Reliab includes specialized test case patterns to stress-test RAG robustness:
- **Multi-Hop Synthesis**: Questions requiring reasoning across disjoint chunks.
- **Unanswerable Boundary Queries**: Questions with absent context to evaluate refusal boundaries.
- **Adversarial Distractors**: Documents with lexical overlap but contradictory semantic facts.
- **Prompt Injection Probes**: Attempts to hijack model instructions via injected instructions in retrieved documents.
