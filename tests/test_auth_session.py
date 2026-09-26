"""Comprehensive authentication regression tests for browser session cookies and programmatic clients."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.core import Settings
from rag_platform.db import Base
from rag_platform.security import ApiKeyRegistry, ClientIdentity, Role
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
    login_resp = auth_client.post(
        "/v1/auth/session",
        json={"api_key": "master-admin-key-999-secure-secret-32-chars-long"},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["status"] == "SUCCESS"
    assert login_resp.json()["is_admin"] is True
    assert "api_key" in login_resp.cookies

    # 2. Make authenticated API request relying strictly on session cookie (no X-API-Key header)
    resp = auth_client.get("/v1/projects")
    assert resp.status_code == 200
    assert isinstance(resp.json()["projects"], list)

    # 3. Create project via cookie-authenticated admin session
    create_resp = auth_client.post("/v1/projects", json={"name": "Cookie Auth Test Project"})
    assert create_resp.status_code == 200
    assert create_resp.json()["name"] == "Cookie Auth Test Project"

    # 4. Logout: invalidate server-side cookie
    logout_resp = auth_client.post("/v1/auth/logout")
    assert logout_resp.status_code == 200
    assert logout_resp.json()["status"] == "SUCCESS"

    # 5. Subsequent request must now be rejected with 401
    auth_client.cookies.clear()
    unauth_resp = auth_client.get("/v1/projects")
    assert unauth_resp.status_code == 401


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
