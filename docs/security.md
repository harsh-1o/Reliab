# Security, Privacy & Provenance

Reliab is built with enterprise defense-in-depth principles to safely evaluate internal RAG applications and manage sensitive test traces.

---

## 1. Multi-Layer Security Architecture

### Fail-Closed Authentication & Opaque Sessions
- **Default Enabled**: `AUTH_ENABLED=true` is the default configuration.
- **Environment Gating**: The application rejects startup if `DEV_MODE=true` is detected in staging or production environments.
- **Hashed API Keys**: API keys are securely hashed using cryptographic primitives (SHA-256) before comparison and database storage, mitigating timing attacks and credential leakage.
- **Cryptographically Hashed Session Tokens**: Authenticating via `POST /v1/auth/session` exchanges an API key for a cryptographically secure, random bearer token (`sess_<token_urlsafe(32)>`).
  - The raw bearer token is returned to the client and stored exclusively in an HttpOnly, SameSite=Strict `session_id` cookie.
  - The database stores **only** the SHA-256 hash of the token (`token_hash`), ensuring that a database compromise or SQL dump never leaks valid session bearer credentials.
  - Incoming session requests hash the presented token and perform lookup by hash.
- **Credential Linkage & Instant Revocation**:
  - Each browser session is linked to the issuing API key (`api_key_hash`).
  - Revoking or rotating an API key immediately invalidates and deletes all active sessions associated with that key.
  - Active sessions dynamically reflect live database permissions (`project_roles_json` and `is_admin`) of the originating credential.
- **Session Cleanup & Expiration**:
  - Sessions default to a 24-hour TTL (`expires_at`).
  - Expired sessions are rejected automatically during authentication and can be purged safely in batches via `POST /v1/maintenance/cleanup` utilizing the database index on `expires_at`.
- **API-Key Cache & Revocation Consistency**:
  - API key identities are cached in-memory with a short TTL (configurable via `API_KEY_CACHE_TTL_SECONDS`, default 2.0 seconds).
  - Key revocation and rotation immediately purge the local process cache. Cross-process propagation is guaranteed within the 2.0-second window (or 0.0 seconds if `API_KEY_CACHE_TTL_SECONDS=0` is configured for strict instantaneous cross-process consistency).
- **Brute-Force Rate Limiting Scope & Bounded Memory**:
  - `AuthRateLimiter` enforces a sliding window (default 5 failed attempts per 60s per IP) for defense-in-depth against brute-force attacks on `/v1/auth/session`.
  - Memory consumption is strictly bounded with capacity-based eviction (`max_tracked_ips=10_000`) and active TTL pruning, preventing memory exhaustion attacks from unbounded numbers of unique source IPs.
  - In distributed multi-process or containerized deployments behind load balancers, edge gateways (such as NGINX, Cloudflare, Envoy, or AWS WAF) handle centralized rate limiting without requiring external Redis dependencies for offline/air-gapped environments.
- **Trusted Reverse Proxy Defense**: Client IP extraction respects `TRUSTED_PROXIES` (default `127.0.0.1,::1,testclient`). If a request does not originate from a configured trusted proxy, `X-Forwarded-For` headers are completely ignored. If an attacker connects directly and provides a forged all-trusted forwarded chain, the proxy extractor rejects the forwarded chain and safely falls back to the direct socket IP.

### Socket-Level SSRF Defense
When evaluating external HTTP RAG endpoints, attackers or rogue configurations could target internal infrastructure (e.g. cloud instance metadata at `169.254.169.254` or internal microservices).

Reliab implements `SSRFProtectedTransport`:
1. **Pre-flight DNS Resolution**: Resolves target hostnames before initiating the HTTP handshake.
2. **IP Range Blocking**: Rejects private, loopback, link-local, and reserved IPv4/IPv6 ranges (RFC 1918, RFC 3927, RFC 4193).
3. **Redirect Hop Re-validation**: Re-validates target destinations on every redirect hop up to the redirect limit.
4. **Socket IP Pinning**: Binds TCP connections directly to pre-validated IP addresses, defeating DNS rebinding (TOCTOU) attacks.
5. **Air-Gapped & Offline Support**: Offline evaluation environments can register static IP mappings via `register_static_dns()` or `STATIC_DNS_MAP` env without opening network access.

### Recursive Secret Sanitization & Secret References
Traces often capture real user queries or raw LLM completions that contain accidentally leaked credentials.

Before any trace, metric, or attribution evidence is persisted:
- A recursive scrubbing filter traverses all dictionary keys, lists, and strings.
- Automatically redacts API keys (`sk-...`, `Bearer ...`, `token`), JWT strings, database connection strings, and common authentication headers.
- **Environment-Backed Secret References**: HTTP adapters use secret references (`header_secret_refs` or `${ENV_VAR}`). Plaintext credentials are redacted before persistence, while secret references are preserved so workers resolve credentials at runtime from the worker environment. Database storage never contains plaintext secrets.

### Tenant & Workspace Isolation
- Projects enforce strict boundary isolation.
- Datasets, runs, and traces are scoped to project IDs, preventing cross-tenant leakage.

---

## 2. Reproducible Run Manifests & Provenance Immutability

Every evaluation run creates a cryptographically verified, immutable `RunProvenance` record (`ConfigDict(frozen=True)`):
- **Immutability Guarantee**: Provenance objects cannot be modified after construction. Any mutation attempts raise frozen instance errors, ensuring derived manifest hashes cannot become stale.
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
