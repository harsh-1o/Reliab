"""Comprehensive authentication regression tests for browser session cookies and programmatic clients."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.core import Settings
from rag_platform.db import Base
from rag_platform.security import ApiKeyRegistry, Role
from rag_platform.server import app, get_db


@pytest.fixture
def auth_client(monkeypatch):
    # Isolated in-memory DB shared across requests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    def _get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = _get_db

    # Configure strictly fail-closed production settings
    settings = Settings(
        auth_enabled=True,
        dev_mode=False,
        api_key="master-admin-key-999-secure-secret-32-chars-long",
    )
    monkeypatch.setattr("rag_platform.core.get_settings", lambda: settings)
    monkeypatch.setattr("rag_platform.server.get_settings", lambda: settings)

    # Register a granular client key in ApiKeyRegistry
    ApiKeyRegistry.clear()
    ApiKeyRegistry.register_key(
        api_key="client-user-key-123",
        client_id="registered_dev_user",
        is_admin=False,
        project_roles={"proj_auth_test": Role.EDITOR},
    )

    client = TestClient(app, base_url="https://testserver")
    yield client
    app.dependency_overrides.clear()
    ApiKeyRegistry.clear()


def test_unauthenticated_request_rejected(auth_client):
    """Unauthenticated requests must be rejected with 401 when auth_enabled=True."""
    resp = auth_client.get("/v1/projects")
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]


def test_failed_session_creation(auth_client):
    """Submitting invalid credentials to /v1/auth/session must return 401 and not set a cookie."""
    resp = auth_client.post("/v1/auth/session", json={"api_key": "wrong-invalid-key"})
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["detail"]
    assert "api_key" not in resp.cookies


def test_browser_cookie_login_authenticated_request_and_logout(auth_client):
    """Verify browser flow: login -> HttpOnly cookie -> authenticated requests -> logout -> rejected."""
    # 1. Login with master API key
    raw_api_key = "master-admin-key-999-secure-secret-32-chars-long"
    login_resp = auth_client.post(
        "/v1/auth/session",
        json={"api_key": raw_api_key},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["status"] == "SUCCESS"
    assert login_resp.json()["is_admin"] is True

    # Priority 1 requirement: Cookie must NOT contain the raw API key!
    cookie_val = login_resp.cookies.get("session_id") or login_resp.cookies.get("api_key")
    assert cookie_val is not None
    assert cookie_val != raw_api_key
    assert raw_api_key not in cookie_val
    assert cookie_val.startswith("sess_")

    # 2. Make authenticated API request relying strictly on session cookie (no X-API-Key header)
    resp = auth_client.get("/v1/projects")
    assert resp.status_code == 200
    assert isinstance(resp.json()["projects"], list)

    # 3. Create project via cookie-authenticated admin session
    create_resp = auth_client.post("/v1/projects", json={"name": "Cookie Auth Test Project"})
    assert create_resp.status_code == 200
    assert create_resp.json()["name"] == "Cookie Auth Test Project"

    # 4. Logout: invalidate server-side session
    logout_resp = auth_client.post("/v1/auth/logout")
    assert logout_resp.status_code == 200
    assert logout_resp.json()["status"] == "SUCCESS"

    # 5. Subsequent request with the same session token must be rejected by server-side invalidation
    # Even if client keeps sending the cookie, server-side session is revoked
    unauth_resp = auth_client.get("/v1/projects", cookies={"session_id": cookie_val})
    assert unauth_resp.status_code == 401
    assert "Invalid or expired session" in unauth_resp.json()["detail"]


def test_session_cookie_does_not_contain_api_key(auth_client):
    """Explicitly verify that session cookie contains an opaque random token, never the API key."""
    secret_key = "client-user-key-123"
    login_resp = auth_client.post("/v1/auth/session", json={"api_key": secret_key})
    assert login_resp.status_code == 200

    session_cookie = login_resp.cookies.get("session_id")
    assert session_cookie is not None
    assert secret_key not in session_cookie
    assert session_cookie.startswith("sess_")
    assert len(session_cookie) >= 32


def test_expired_or_invalid_session_token_rejected(auth_client):
    """Submitting non-existent or fabricated session cookies must return 401."""
    resp = auth_client.get("/v1/projects", cookies={"session_id": "sess_fabricated_bogus_token_12345"})
    assert resp.status_code == 401
    assert "Invalid or expired session" in resp.json()["detail"]


def test_browser_cookie_login_with_registered_client_key(auth_client):
    """Verify registered non-admin user can also establish cookie session."""
    login_resp = auth_client.post("/v1/auth/session", json={"api_key": "client-user-key-123"})
    assert login_resp.status_code == 200
    assert login_resp.json()["client_id"] == "registered_dev_user"
    assert login_resp.json()["is_admin"] is False

    # Editor role: can read projects
    resp = auth_client.get("/v1/projects")
    assert resp.status_code == 200


def test_programmatic_client_x_api_key_header(auth_client):
    """Non-browser API clients using X-API-Key header continue to work without cookies."""
    resp = auth_client.get(
        "/v1/projects",
        headers={"X-API-Key": "master-admin-key-999-secure-secret-32-chars-long"},
    )
    assert resp.status_code == 200

    resp_client = auth_client.get("/v1/projects", headers={"X-API-Key": "client-user-key-123"})
    assert resp_client.status_code == 200

    resp_bad = auth_client.get("/v1/projects", headers={"X-API-Key": "completely-invalid"})
    assert resp_bad.status_code == 401


def test_programmatic_client_bearer_token(auth_client):
    """Non-browser API clients using Authorization: Bearer continue to work."""
    resp = auth_client.get(
        "/v1/projects",
        headers={"Authorization": "Bearer master-admin-key-999-secure-secret-32-chars-long"},
    )
    assert resp.status_code == 200

    resp_bad = auth_client.get("/v1/projects", headers={"Authorization": "Bearer bogus"})
    assert resp_bad.status_code == 401


def test_trusted_proxy_forwarded_for_and_spoof_defense(monkeypatch):
    """Priority 2: Verify trusted proxy resolution and spoofing defense."""
    from starlette.requests import Request

    from rag_platform.server import extract_client_ip

    trusted = ("127.0.0.1", "10.0.0.0/8")

    # 1. Direct untrusted client trying to spoof X-Forwarded-For
    scope_untrusted = {
        "type": "http",
        "client": ("203.0.113.195", 12345),
        "headers": [(b"x-forwarded-for", b"198.51.100.5")],
    }
    req_untrusted = Request(scope_untrusted)
    # Direct peer 203.0.113.195 is NOT trusted; X-Forwarded-For must be ignored!
    assert extract_client_ip(req_untrusted, trusted_proxies=trusted) == "203.0.113.195"

    # 2. Trusted proxy with single forwarded IP
    scope_trusted_proxy = {
        "type": "http",
        "client": ("127.0.0.1", 54321),
        "headers": [(b"x-forwarded-for", b"198.51.100.5")],
    }
    req_trusted = Request(scope_trusted_proxy)
    assert extract_client_ip(req_trusted, trusted_proxies=trusted) == "198.51.100.5"

    # 3. Trusted proxy with multiple forwarded IPs in chain: client, internal_lb
    scope_chain = {
        "type": "http",
        "client": ("127.0.0.1", 54321),
        "headers": [(b"x-forwarded-for", b"198.51.100.5, 10.0.0.2")],
    }
    req_chain = Request(scope_chain)
    # 10.0.0.2 is in trusted 10.0.0.0/8, so rightmost untrusted client is 198.51.100.5
    assert extract_client_ip(req_chain, trusted_proxies=trusted) == "198.51.100.5"

    # 4. Fallback when no client socket host is present
    scope_no_client = {
        "type": "http",
        "client": None,
        "headers": [],
    }
    req_no_client = Request(scope_no_client)
    assert extract_client_ip(req_no_client, trusted_proxies=trusted) == "127.0.0.1"


def test_database_backed_cross_process_sessions():
    """P1: Verify that sessions are persisted in database and shared across independent worker/API processes."""
    from rag_platform.security import ClientIdentity, Role, SessionStore

    # Shared database engine simulating shared storage between Process A and Process B
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    identity = ClientIdentity(
        client_id="user_distributed",
        is_admin=False,
        project_roles={"proj_shared": Role.EDITOR},
    )

    # 1. Process A creates session in shared database
    with Session(engine) as sess_a:
        session_id = SessionStore.create_session(identity, ttl_seconds=3600.0, db_session=sess_a)
        assert session_id.startswith("sess_")

    # 2. Process B (separate database connection/process) retrieves session
    with Session(engine) as sess_b:
        retrieved_ident = SessionStore.get_session(session_id, db_session=sess_b)
        assert retrieved_ident is not None
        assert retrieved_ident.client_id == "user_distributed"
        assert retrieved_ident.project_roles["proj_shared"] == Role.EDITOR
        assert not retrieved_ident.is_admin

    # 3. Test expired session behavior across processes
    with Session(engine) as sess_a:
        expired_id = SessionStore.create_session(identity, ttl_seconds=-10.0, db_session=sess_a)

    with Session(engine) as sess_b:
        assert SessionStore.get_session(expired_id, db_session=sess_b) is None

    # 4. Process A invalidates session -> Process B immediately sees rejection
    with Session(engine) as sess_a:
        assert SessionStore.invalidate(session_id, db_session=sess_a) is True

    with Session(engine) as sess_b:
        assert SessionStore.get_session(session_id, db_session=sess_b) is None


def test_api_key_short_ttl_multi_process_revocation():
    """P2: Verify that short API key cache TTL ensures revoked keys are rejected across processes."""
    import time

    from rag_platform.db import DatabaseRepo
    from rag_platform.security import ApiKeyRegistry, Role, generate_secure_api_key

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    raw_key = generate_secure_api_key()
    with Session(engine) as sess:
        repo = DatabaseRepo(sess)
        created_key, row = repo.create_api_key(
            client_id="client_multi_proc",
            api_key=raw_key,
            project_roles={"proj_1": Role.VIEWER},
        )
        sess.commit()
        key_hash = row.key_hash

    # Save original TTL and set short TTL for test
    orig_ttl = ApiKeyRegistry.CACHE_TTL_SECONDS
    try:
        ApiKeyRegistry.CACHE_TTL_SECONDS = 0.05
        ApiKeyRegistry.clear()

        # Process B authenticates and caches key
        with Session(engine) as sess_b:
            ident = ApiKeyRegistry.get(raw_key, db_session=sess_b)
            assert ident is not None
            assert ident.client_id == "client_multi_proc"

        # Process A revokes key in DB (without touching Process B's local memory)
        with Session(engine) as sess_a:
            repo_a = DatabaseRepo(sess_a)
            revoked = repo_a.revoke_api_key(key_hash)
            sess_a.commit()
            assert revoked is True

        # Wait for short TTL to expire
        time.sleep(0.06)

        # Process B attempts authentication again -> must be rejected from DB
        with Session(engine) as sess_b:
            ident_after = ApiKeyRegistry.get(raw_key, db_session=sess_b)
            assert ident_after is None
    finally:
        ApiKeyRegistry.CACHE_TTL_SECONDS = orig_ttl
        ApiKeyRegistry.clear()

