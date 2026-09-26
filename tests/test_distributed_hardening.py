"""Regression tests for distributed multi-process evaluation caching,
air-gapped SSRF DNS resolution, and worker notification push loop.
"""

import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.db import Base
from rag_platform.evaluators import (
    DatabaseEvaluationCache,
    TwoTierEvaluationCache,
)
from rag_platform.models import (
    MetricFamily,
    MetricResult,
    MetricStatus,
)
from rag_platform.ssrf import (
    SSRFProtectionError,
    validate_url_ssrf,
    register_static_dns,
    clear_static_dns,
)
from rag_platform.worker import (
    DurableRunWorker,
    WorkerNotificationBus,
)


@pytest.fixture
def memory_db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = lambda: Session(engine)
    with session_factory() as sess:
        yield sess, session_factory


class TestMultiProcessSharedEvaluationCache:
    """Verifies that evaluation results are shared across multi-process workers via L2 DB cache."""

    def test_database_evaluation_cache_crud(self, memory_db_session):
        sess, session_factory = memory_db_session
        cache = DatabaseEvaluationCache(capacity=100, session_factory=session_factory)

        result = MetricResult(
            metric_name="faithfulness",
            metric_family=MetricFamily.GENERATION,
            score=0.95,
            status=MetricStatus.PASS,
            reason="All claims supported",
            evaluator_version="1.0.0",
        )

        cache.set("cache_key_1", result)

        # Retrieve and verify
        cached = cache.get("cache_key_1")
        assert cached is not None
        assert cached.score == 0.95
        assert cached.metric_name == "faithfulness"
        assert cache.hits == 1

        # Miss
        assert cache.get("non_existent_key") is None
        assert cache.misses == 1

    def test_database_evaluation_cache_pruning(self, memory_db_session):
        sess, session_factory = memory_db_session
        # Set low capacity of 3
        cache = DatabaseEvaluationCache(capacity=3, session_factory=session_factory)

        for i in range(5):
            res = MetricResult(
                metric_name="faithfulness",
                metric_family=MetricFamily.GENERATION,
                score=float(i) / 10.0,
                status=MetricStatus.PASS,
                reason=f"Test {i}",
            )
            cache.set(f"k_{i}", res)

        stats = cache.stats()
        # Must not exceed capacity of 3
        assert stats["size"] <= 3

    def test_two_tier_cache_multi_process_sharing(self, memory_db_session):
        sess, session_factory = memory_db_session

        # Worker Process 1
        worker_1_cache = TwoTierEvaluationCache(
            l1_capacity=10,
            l2_capacity=100,
            session_factory=session_factory,
        )

        # Worker Process 2 (independent in-memory L1, shared DB L2)
        worker_2_cache = TwoTierEvaluationCache(
            l1_capacity=10,
            l2_capacity=100,
            session_factory=session_factory,
        )

        result = MetricResult(
            metric_name="citation_support",
            metric_family=MetricFamily.CITATION,
            score=1.0,
            status=MetricStatus.PASS,
            reason="Accurate citation reference",
        )

        # Worker 1 evaluates and stores result
        worker_1_cache.set("eval_hash_abc", result)
        assert worker_1_cache.l1.get("eval_hash_abc") is not None

        # Worker 2 has empty L1 memory
        assert "eval_hash_abc" not in worker_2_cache.l1

        # Worker 2 retrieves the key -> hits L2 shared database!
        shared_hit = worker_2_cache.get("eval_hash_abc")
        assert shared_hit is not None
        assert shared_hit.score == 1.0

        # Now Worker 2's L1 cache is populated for future ultra-fast local reads
        assert worker_2_cache.l1.get("eval_hash_abc") is not None


class TestAirGappedSSRFDNSResolution:
    """Verifies that SSRF checks operate in offline/air-gapped environments without failing on DNS lookups."""

    def test_static_dns_map_resolves_offline(self):
        register_static_dns("airgap-rag.example.org", "93.184.216.34")
        try:
            resolved = validate_url_ssrf("https://airgap-rag.example.org/v1/predict")
            assert "93.184.216.34" in resolved
        finally:
            clear_static_dns("airgap-rag.example.org")

    def test_static_dns_map_blocks_private_ip(self):
        # Even with static DNS, private IPs must NEVER bypass security
        register_static_dns("internal-metadata.test", "169.254.169.254")
        try:
            with pytest.raises(SSRFProtectionError) as exc_info:
                validate_url_ssrf("http://internal-metadata.test/computeMetadata/v1")
            assert "disallowed private/reserved IP" in str(exc_info.value)
        finally:
            clear_static_dns("internal-metadata.test")

    def test_custom_dns_resolver_injection(self):
        custom_resolver = lambda host, port: ["93.184.216.34"]
        resolved = validate_url_ssrf(
            "https://custom-dns-target.internal/endpoint",
            dns_resolver=custom_resolver,
        )
        assert resolved == ["93.184.216.34"]

    def test_custom_dns_resolver_rebinding_defense(self):
        # Resolver returns both a public IP and an internal private IP
        malicious_rebinding_resolver = lambda host, port: ["93.184.216.34", "10.240.0.1"]
        with pytest.raises(SSRFProtectionError) as exc_info:
            validate_url_ssrf(
                "https://rebinding-target.internal/endpoint",
                dns_resolver=malicious_rebinding_resolver,
            )
        assert "disallowed private/reserved IP" in str(exc_info.value)

    def test_env_var_static_dns(self, monkeypatch):
        monkeypatch.setenv("SSRF_STATIC_DNS_MAP", '{"airgap-env.internal": ["93.184.216.35"]}')
        resolved = validate_url_ssrf("https://airgap-env.internal/query")
        assert "93.184.216.35" in resolved


class TestWorkerNotificationBusAndAdaptiveLoop:
    """Verifies event-driven push notifications and adaptive backoff worker daemon."""

    @pytest.mark.asyncio
    async def test_worker_notification_bus_broadcast(self):
        ev1 = WorkerNotificationBus.subscribe()
        ev2 = WorkerNotificationBus.subscribe()

        assert not ev1.is_set()
        assert not ev2.is_set()

        WorkerNotificationBus.notify_new_run("run_xyz_123")

        assert ev1.is_set()
        assert ev2.is_set()

        WorkerNotificationBus.unsubscribe(ev1)
        WorkerNotificationBus.unsubscribe(ev2)

    @pytest.mark.asyncio
    async def test_adaptive_worker_loop_wakeup_and_shutdown(self, memory_db_session):
        sess, session_factory = memory_db_session

        stop_event = asyncio.Event()
        # Launch worker loop with very small interval for test speed
        worker_task = asyncio.create_task(
            DurableRunWorker.run_worker_loop(
                stop_event=stop_event,
                min_interval=0.05,
                max_interval=0.2,
                stale_check_interval=100.0,
                session_factory=session_factory,
            )
        )

        # Notify bus and verify it does not error
        WorkerNotificationBus.notify_new_run("dummy_run_id")
        await asyncio.sleep(0.1)

        # Stop worker loop gracefully
        stop_event.set()
        await asyncio.wait_for(worker_task, timeout=2.0)
        assert worker_task.done()
