"""Tests for API Key Lifecycle, Cache Invalidation, Revocation/Rotation, and Auth Brute-Force Throttling."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.core import sha256_hash
from rag_platform.db import Base, DatabaseRepo
from rag_platform.security import ApiKeyRegistry, Role, generate_secure_api_key
from rag_platform.server import app, auth_rate_limiter, get_db


@pytest.fixture
def auth_test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    ApiKeyRegistry.clear()
    auth_rate_limiter.reset()

    def get_test_db():
        with Session(bind=engine) as session:
            yield session

    app.dependency_overrides[get_db] = get_test_db
    yield engine
    app.dependency_overrides.pop(get_db, None)
    ApiKeyRegistry.clear()
    auth_rate_limiter.reset()


def test_api_key_hashed_caching_and_ttl(auth_test_db):
    """ApiKeyRegistry must never cache raw API keys and must honor cache TTL."""
    raw_key = generate_secure_api_key()
    key_hash = sha256_hash(raw_key)

    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        created_key, row = repo.create_api_key(
            client_id="test_client_1",
            api_key=raw_key,
            project_roles={"proj_1": Role.EDITOR},
            is_admin=False,
        )
        sess.commit()
        assert created_key == raw_key
        assert row.key_hash == key_hash

    # Raw key must NOT be stored in memory cache
    assert raw_key not in ApiKeyRegistry._cache
    # Hashed key lookup should succeed
    with Session(auth_test_db) as sess:
        ident = ApiKeyRegistry.get(raw_key, db_session=sess)
        assert ident is not None
        assert ident.client_id == "test_client_1"
        assert ident.project_roles["proj_1"] == Role.EDITOR
        assert not ident.is_admin

    # Cache now contains the hash
    assert key_hash in ApiKeyRegistry._cache

    # Verify TTL expiration
    ident_cached, cached_at = ApiKeyRegistry._cache[key_hash]
    # Artificially age the entry past TTL
    ApiKeyRegistry._cache[key_hash] = (ident_cached, time.monotonic() - 70.0)

    # Next get refreshes from DB
    with Session(auth_test_db) as sess:
        ident_refreshed = ApiKeyRegistry.get(raw_key, db_session=sess)
        assert ident_refreshed is not None
        assert ident_refreshed.client_id == "test_client_1"


def test_api_key_revocation_invalidates_cache_immediately(auth_test_db):
    """Revoking an API key must invalidate cache and reject authentication immediately."""
    raw_key = generate_secure_api_key()
    key_hash = sha256_hash(raw_key)

    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        repo.create_api_key(client_id="revocable_client", api_key=raw_key)
        sess.commit()

    # Pre-populate cache via successful auth
    with Session(auth_test_db) as sess:
        assert ApiKeyRegistry.get(raw_key, db_session=sess) is not None
        assert key_hash in ApiKeyRegistry._cache

    # Revoke key via repo
    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        revoked = repo.revoke_api_key(key_hash)
        sess.commit()
        assert revoked is True

    # Cache must have been purged
    assert key_hash not in ApiKeyRegistry._cache

    # Subsequent auth attempt must fail immediately
    with Session(auth_test_db) as sess:
        ident = ApiKeyRegistry.get(raw_key, db_session=sess)
        assert ident is None


def test_api_key_rotation_lifecycle(auth_test_db):
    """Rotating an API key atomically revokes the old key and issues a new valid credential."""
    raw_old_key = generate_secure_api_key()
    old_hash = sha256_hash(raw_old_key)

    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        repo.create_api_key(
            client_id="rotating_client",
            api_key=raw_old_key,
            project_roles={"proj_sec": Role.ADMIN},
            is_admin=True,
        )
        sess.commit()

    # Warm cache
    with Session(auth_test_db) as sess:
        assert ApiKeyRegistry.get(raw_old_key, db_session=sess) is not None

    # Rotate key
    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        raw_new_key, new_row = repo.rotate_api_key(old_hash)
        sess.commit()
        assert raw_new_key != raw_old_key
        assert new_row.client_id == "rotating_client"
        assert new_row.is_admin is True

    # Old key fails authentication
    with Session(auth_test_db) as sess:
        assert ApiKeyRegistry.get(raw_old_key, db_session=sess) is None

    # New key succeeds with identical identity and roles
    with Session(auth_test_db) as sess:
        new_ident = ApiKeyRegistry.get(raw_new_key, db_session=sess)
        assert new_ident is not None
        assert new_ident.client_id == "rotating_client"
        assert new_ident.is_admin is True
        assert new_ident.project_roles["proj_sec"] == Role.ADMIN


def test_admin_api_key_endpoints(auth_test_db):
    """Admin-only endpoints for issuing, listing, revoking, and rotating credentials."""
    client = TestClient(app)

    # Register admin key and unprivileged key
    admin_key = generate_secure_api_key()
    editor_key = generate_secure_api_key()

    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        repo.create_api_key("admin_user", api_key=admin_key, is_admin=True)
        repo.create_api_key("regular_user", api_key=editor_key, is_admin=False)
        sess.commit()

    # Non-admin receives 403 Forbidden
    res = client.post(
        "/v1/auth/keys",
        headers={"X-API-Key": editor_key},
        json={"client_id": "svc_analytics", "is_admin": False},
    )
    assert res.status_code == 403

    # Admin issues new key
    res = client.post(
        "/v1/auth/keys",
        headers={"X-API-Key": admin_key},
        json={
            "client_id": "svc_analytics",
            "project_roles": {"analytics_proj": "editor"},
            "is_admin": False,
        },
    )
    assert res.status_code == 200
    issued_data = res.json()
    new_raw_key = issued_data["api_key"]
    new_key_hash = issued_data["key_hash"]
    assert issued_data["client_id"] == "svc_analytics"

    # List keys
    list_res = client.get("/v1/auth/keys", headers={"X-API-Key": admin_key})
    assert list_res.status_code == 200
    keys = list_res.json()["keys"]
    assert any(k["key_hash"] == new_key_hash for k in keys)

    # Rotate key
    rotate_res = client.post(
        f"/v1/auth/keys/{new_key_hash}/rotate",
        headers={"X-API-Key": admin_key},
    )
    assert rotate_res.status_code == 200
    rotated_data = rotate_res.json()
    rotated_raw_key = rotated_data["api_key"]
    rotated_key_hash = rotated_data["key_hash"]
    assert rotated_raw_key != new_raw_key

    # Old key is revoked
    rev_test = client.get("/v1/projects", headers={"X-API-Key": new_raw_key})
    assert rev_test.status_code == 401

    # Rotated key is active
    active_test = client.get("/v1/projects", headers={"X-API-Key": rotated_raw_key})
    assert active_test.status_code == 200

    # Revoke rotated key
    revoke_res = client.post(
        f"/v1/auth/keys/{rotated_key_hash}/revoke",
        headers={"X-API-Key": admin_key},
    )
    assert revoke_res.status_code == 200

    final_test = client.get("/v1/projects", headers={"X-API-Key": rotated_raw_key})
    assert final_test.status_code == 401


def test_auth_session_brute_force_throttling(auth_test_db):
    """POST /v1/auth/session must throttle repeated failed authentication attempts with HTTP 429."""
    client = TestClient(app)

    # Register one valid key
    valid_key = generate_secure_api_key()
    with Session(auth_test_db) as sess:
        repo = DatabaseRepo(sess)
        repo.create_api_key("test_user", api_key=valid_key)
        sess.commit()

    # Send 5 failed authentication attempts from IP 198.51.100.42
    for attempt in range(5):
        resp = client.post(
            "/v1/auth/session",
            json={"api_key": f"invalid_key_{attempt}"},
            headers={"X-Forwarded-For": "198.51.100.42"},
        )
        assert resp.status_code == 401, f"Expected 401 on attempt {attempt}, got {resp.status_code}"

    # 6th attempt from same IP must be throttled with HTTP 429
    blocked_resp = client.post(
        "/v1/auth/session",
        json={"api_key": "another_invalid_key"},
        headers={"X-Forwarded-For": "198.51.100.42"},
    )
    assert blocked_resp.status_code == 429
    assert "Too many failed authentication attempts" in blocked_resp.json()["detail"]
    assert "Retry-After" in blocked_resp.headers

    # Another IP should not be throttled
    other_ip_resp = client.post(
        "/v1/auth/session",
        json={"api_key": valid_key},
        headers={"X-Forwarded-For": "203.0.113.99"},
    )
    assert other_ip_resp.status_code == 200
