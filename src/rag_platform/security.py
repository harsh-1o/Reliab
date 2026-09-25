"""Security, privacy, prompt injection defense, and budget guardrails.

# ponytail: single file covers secret redaction, prompt injection encapsulation, and rate limiting.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from rag_platform.core import PolicyViolationError
from rag_platform.models import RunOptions


# --- 1. Secret Redaction ---
class SecretRedactor:
    """Detects and redacts sensitive credentials, API keys, and authorization tokens."""

    PATTERNS = [
        # OpenAI / LLM API keys
        (re.compile(r"sk-[a-zA-Z0-9_-]{20,}"), "sk-***[REDACTED]***"),
        # Bearer tokens
        (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), "Bearer ***[REDACTED]***"),
        # Generic API keys & secrets in query params or headers
        (re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?", re.IGNORECASE), r"\1=***[REDACTED]***"),
        # AWS / Generic Cloud Access Keys
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA***[REDACTED]***"),
    ]

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Sanitize text of known secret formats."""
        if not text:
            return text
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    @classmethod
    def redact_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively redact dictionary values containing sensitive keys or patterns."""
        sensitive_keys = {"authorization", "api_key", "apikey", "password", "secret", "token", "auth"}
        clean = {}
        for k, v in data.items():
            if str(k).lower() in sensitive_keys:
                clean[k] = "***[REDACTED]***"
            elif isinstance(v, str):
                clean[k] = cls.redact_text(v)
            elif isinstance(v, dict):
                clean[k] = cls.redact_dict(v)
            elif isinstance(v, list):
                clean[k] = [cls.redact_dict(item) if isinstance(item, dict) else cls.redact_text(str(item)) if isinstance(item, str) else item for item in v]
            else:
                clean[k] = v
        return clean


# --- 2. Evaluator Prompt Injection Defense ---
class EvaluatorPromptDefense:
    """Neutralizes indirect prompt injection attacks hidden inside retrieved documents."""

    @staticmethod
    def sanitize_untrusted_content(content: str) -> str:
        """Escape XML-style delimiter tags to prevent jailbreak breakout."""
        if not content:
            return ""
        # Neutralize closing tags that attempt to escape prompt delimiters
        escaped = content.replace("</untrusted_evidence>", "&lt;/untrusted_evidence&gt;")
        escaped = escaped.replace("</evidence>", "&lt;/evidence&gt;")
        escaped = escaped.replace("<system>", "&lt;system&gt;").replace("</system>", "&lt;/system&gt;")
        return escaped

    @classmethod
    def wrap_evidence(cls, chunk_text: str, chunk_id: str = "") -> str:
        """Wrap untrusted retrieved document in strict defensive containment tags."""
        clean_text = cls.sanitize_untrusted_content(chunk_text)
        return f'<untrusted_evidence id="{chunk_id}">\n{clean_text}\n</untrusted_evidence>'


# --- 3. Rate Limiting & Concurrency Control ---
class TokenBucketRateLimiter:
    """Async token-bucket rate limiter to protect against provider rate limits."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate  # Tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until requested token quota is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # Need to wait
                needed = tokens - self.tokens
                wait_time = needed / self.rate

            await asyncio.sleep(min(wait_time, 0.1))


# --- 4. Budget & Quota Guard ---
class BudgetGuard:
    """Enforces execution limits to prevent cost amplification attacks."""

    @staticmethod
    def validate_run_bounds(total_cases: int, options: RunOptions, max_allowed_cases: int = 500) -> None:
        """Validate case count against project limits."""
        limit = options.max_cases or max_allowed_cases
        if total_cases > limit:
            raise PolicyViolationError(
                f"Requested evaluation with {total_cases} cases exceeds permitted limit of {limit} cases."
            )
