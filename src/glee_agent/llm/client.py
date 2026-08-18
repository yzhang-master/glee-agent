"""Minimal OpenAI-compatible chat-completions client with a hard timeout and
a module-level circuit breaker.

Contract:
- `complete()` returns stripped text or None on ANY failure (timeout,
  non-200, empty body, exception). One attempt, no retries.
- The HTTP request runs inside a fresh ThreadPoolExecutor(1) future bounded
  by `future.result(timeout=timeout_s)`, so a hung socket cannot blow the
  turn budget even if requests' own timeout misbehaves. Never signal-based:
  strategies run in SDK worker threads.
- Circuit breaker (lock-protected, module-level): BREAKER_FAILURES
  consecutive failures open it for BREAKER_COOLDOWN_S seconds.
- Env (LLM_API_BASE / LLM_API_KEY / LLM_MODEL) is read lazily at call time;
  dotenv is already loaded by the config import.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger("glee_agent.llm")

BREAKER_FAILURES = 3
BREAKER_COOLDOWN_S = 600.0

_now = time.monotonic  # monkeypatchable clock for tests
_lock = threading.Lock()
_consecutive_failures = 0
_open_until = 0.0


def reset_breaker() -> None:
    """Close the breaker and zero the failure count (used by tests)."""
    global _consecutive_failures, _open_until
    with _lock:
        _consecutive_failures = 0
        _open_until = 0.0


def breaker_open() -> bool:
    with _lock:
        return _now() < _open_until


def record(ok: bool) -> None:
    """Record one call outcome; open the breaker on a failure streak."""
    global _consecutive_failures, _open_until
    with _lock:
        if ok:
            _consecutive_failures = 0
            return
        _consecutive_failures += 1
        if _consecutive_failures >= BREAKER_FAILURES:
            _open_until = _now() + BREAKER_COOLDOWN_S
            _consecutive_failures = 0
            logger.warning(
                "LLM circuit breaker OPEN for %.0fs after %d consecutive failures",
                BREAKER_COOLDOWN_S,
                BREAKER_FAILURES,
            )


def llm_available(knobs) -> bool:
    """True when LLM decoration should be attempted this turn."""
    if not getattr(knobs, "llm_enabled", False):
        return False
    if not os.environ.get("LLM_API_KEY", "").strip():
        return False
    # Hermetic test suites must never hit the network by accident; test_llm
    # opts back in explicitly via GLEE_LLM_ALLOW_IN_TESTS.
    if "PYTEST_CURRENT_TEST" in os.environ and not os.environ.get("GLEE_LLM_ALLOW_IN_TESTS"):
        return False
    return not breaker_open()


def _request(base: str, key: str, model: str, prompt: str, system: str | None,
             max_tokens: int, timeout_s: float) -> str | None:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.9,
        },
        timeout=(3, timeout_s),
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        return None
    return content.strip() or None


def complete(prompt: str, system: str | None = None, max_tokens: int = 220,
             timeout_s: float = 8.0) -> str | None:
    """One LLM completion attempt. Returns stripped text, or None on any
    failure. Never raises."""
    if breaker_open():
        return None
    try:
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            return None  # not configured: no breaker penalty
        base = os.environ.get("LLM_API_BASE", "https://yunwu.ai/v1")
        model = os.environ.get("LLM_MODEL", "deepseek-v3")

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(
                _request, base, key, model, prompt, system, max_tokens, timeout_s
            )
            text = future.result(timeout=timeout_s)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception as e:  # noqa: BLE001 — timeouts, HTTP, JSON, anything
        logger.warning("LLM call failed: %s: %s", type(e).__name__, e)
        record(False)
        return None

    if not text:
        record(False)
        return None
    record(True)
    return text
