# Security, Privacy & Provenance

Reliab is built with enterprise defense-in-depth principles to safely evaluate internal RAG applications and manage sensitive test traces.

---

## 1. Multi-Layer Security Architecture

### Fail-Closed Authentication & Opaque Sessions
- **Default Enabled**: `AUTH_ENABLED=true` is the default configuration.
- **Environment Gating**: The application rejects startup if `DEV_MODE=true` is detected in staging or production environments.
- **Hashed API Keys**: API keys are securely hashed using cryptographic primitives before comparison to mitigate timing attacks.
- **Opaque Browser Session Tokens**: Authenticating via `POST /v1/auth/session` exchanges the API key for an opaque random session token (`sess_<token_urlsafe(32)>`). The raw API key is **never** stored in browser cookies. Sessions are stored in a server-side `SessionStore` with a 24-hour TTL, and `/v1/auth/logout` explicitly invalidates the server-side session.
- **Trusted Reverse Proxy Defense**: Client IP extraction respects `TRUSTED_PROXIES` (default `127.0.0.1,::1,testclient`). If a request does not originate from a configured trusted proxy, `X-Forwarded-For` headers are ignored and the direct socket IP is enforced, preventing spoofing and rate-limiting bypasses.

### Socket-Level SSRF Defense
When evaluating external HTTP RAG endpoints, attackers or rogue configurations could target internal infrastructure (e.g. cloud instance metadata at `169.254.169.254` or internal microservices).

Reliab implements `SSRFProtectedTransport`:
1. **Pre-flight DNS Resolution**: Resolves target hostnames before initiating the HTTP handshake.
2. **IP Range Blocking**: Rejects private, loopback, link-local, and reserved IPv4/IPv6 ranges (RFC 1918, RFC 3927, RFC 4193).
3. **Redirect Hop Re-validation**: Re-validates target destinations on every redirect hop up to the redirect limit.
4. **Socket IP Pinning**: Binds TCP connections directly to pre-validated IP addresses, defeating DNS rebinding (TOCTOU) attacks.
5. **Air-Gapped & Offline Support**: Offline evaluation environments can register static IP mappings via `register_static_dns()` or `STATIC_DNS_MAP` env without opening network access.

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
