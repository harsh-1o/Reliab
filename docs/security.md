# Security & Deployment Configuration

Reliab evaluates systems that may contain sensitive prompts, retrieved documents, generated answers, credentials, and internal endpoints. The platform treats these inputs as untrusted.

## Authentication

### AUTH_ENABLED

Authentication defaults to enabled in deployment environments and disabled for local development. With authentication enabled and `DEV_MODE=false`, a strong `RAG_PLATFORM_API_KEY` is required.

### DEV_MODE

Explicit local-development authentication bypass. The application refuses to start with `DEV_MODE=true` in a deployment environment.

### RAG_PLATFORM_API_KEY

Bootstrap API credential. Keep it outside source control and inject it through the environment/secret manager.

## Database

### DATABASE_URL

Persistent deployments must explicitly configure a database URL and cannot silently use SQLite. PostgreSQL is the intended deployment database.

Alembic is authoritative for persistent schema changes. `Base.metadata.create_all()` is restricted to ephemeral in-memory test databases.

## Trusted proxies

### TRUSTED_PROXIES

Comma-separated addresses/CIDRs allowed to supply `X-Forwarded-For`. Default:

```text
127.0.0.1,::1,testclient
```

Forwarded headers are ignored when the direct peer is not trusted. Do not configure broad networks unless every address is a trusted proxy.

## API-key cache

### API_KEY_CACHE_TTL_SECONDS

In-process API-key identity cache TTL. Default: `2.0` seconds. Set `0` for strict database validation on each request. Revocation/rotation invalidates the local cache immediately; a positive TTL can leave another process with a cached identity until expiry.

## Authentication rate limiting

The session endpoint has a process-local sliding-window limiter: 5 failed attempts per 60 seconds, with bounded tracking for 10,000 source IPs. In a distributed deployment, use an edge/WAF limiter as the centralized control.

## Browser sessions

The browser exchanges an API key for a cryptographically random opaque token. Only its SHA-256 hash is persisted. Sessions are HttpOnly, SameSite=Strict, linked to the issuing API key when applicable, and invalidated when that key is revoked/rotated.

The frontend must not persist API keys in localStorage/sessionStorage.

## SSRF protection

HTTP adapters:

1. validate the destination URL;
2. resolve DNS before connecting;
3. reject private, loopback, link-local, reserved, and metadata-style addresses by default;
4. validate all resolved addresses;
5. revalidate redirect hops;
6. pin connections to validated addresses;
7. disable ambient proxy environment configuration.

`HTTP_RAG_ALLOWED_HOSTS` can narrow destinations to an explicit host allowlist.

`SSRF_STATIC_DNS_MAP` supplies controlled DNS mappings for offline/air-gapped environments. Static mappings do not bypass private/reserved-address checks.

The low-level validator exposes `allow_private_ips` for trusted environments. Enabling it weakens the SSRF boundary and should not be used for untrusted endpoint configuration.

## Secrets

HTTP adapter credentials should use environment-backed secret references. Trace persistence recursively redacts API keys, bearer tokens, JWTs, connection strings, authentication headers, and private-key material.

Workers resolve runtime secrets from their own environment; persisted run configuration should not contain plaintext provider credentials.

## Prompt injection

Retrieved documents are treated as passive, untrusted evidence and are wrapped by the defensive prompt-formatting helpers. This protects Reliab's evaluation boundary; it does not make the evaluated RAG system immune to prompt injection.

## Project isolation

Projects are the primary authorization boundary. Datasets, runs, traces, and failures are checked against project permissions. Administrator credentials manage API keys and maintenance.

## Provenance

Run provenance records dataset identity/checksum, system/evaluator/adapter information, and environment/configuration data. Its manifest hash is recomputed and verified before persistence. Treat the resulting provenance as immutable after run creation.

## Deployment checklist

For an exposed deployment:

- PostgreSQL + Alembic migrations;
- `AUTH_ENABLED=true`;
- `DEV_MODE=false`;
- strong `RAG_PLATFORM_API_KEY` from a secret manager;
- narrow `TRUSTED_PROXIES`;
- centralized edge/WAF rate limiting;
- narrow HTTP host allowlist;
- private-IP access disabled unless explicitly required;
- TLS and network segmentation;
- identical worker/API secret configuration.

These controls complement, rather than replace, TLS, network isolation, least privilege, and host security.
