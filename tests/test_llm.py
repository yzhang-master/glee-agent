"""LLM message layer: fallback safety, wiring, breaker, prompt hygiene."""

from __future__ import annotations

import os

import pytest

from glee_agent.config import Knobs
from glee_agent.families import bargaining, negotiation, persuasion
from glee_agent.guard import guard
from glee_agent.llm import client, messages
from glee_agent.schema import parse_game

from fixtures import bargaining_game, negotiation_decision, negotiation_game, persuasion_game

KNOBS = Knobs()

# Captured at import time, before any monkeypatch touches the env, so the
# live smoke test can still reach the real gateway.
REAL_KEY = os.environ.get("LLM_API_KEY", "").strip()
REAL_BASE = os.environ.get("LLM_API_BASE", "").strip()
REAL_MODEL = os.environ.get("LLM_MODEL", "").strip()


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch):
    """Fresh breaker + fake key + explicit opt-in for wiring under pytest."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("GLEE_LLM_ALLOW_IN_TESTS", "1")
    client.reset_breaker()
    yield
    client.reset_breaker()


# ------------------------------------------------------------ fallback/wiring


def test_persuasion_falls_back_to_template_on_none(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: None)
    view = parse_game(persuasion_game(actor="seller", quality="high"))
    action = persuasion.decide(view, KNOBS)
    assert action["message"] == persuasion._seller_message(True, view)


def test_persuasion_uses_llm_text(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: "Fresh pitch, buy now.")
    view = parse_game(persuasion_game(actor="seller", quality="high"))
    action = persuasion.decide(view, KNOBS)
    assert action["message"] == "Fresh pitch, buy now."


def test_persuasion_llm_disabled_by_knob(monkeypatch):
    called = []
    monkeypatch.setattr(client, "complete", lambda *a, **k: called.append(1) or "x")
    knobs = Knobs(llm_enabled=False)
    view = parse_game(persuasion_game(actor="seller", quality="high"))
    action = persuasion.decide(view, knobs)
    assert not called
    assert action["message"] == persuasion._seller_message(True, view)


def test_persuasion_binary_mode_never_calls_llm(monkeypatch):
    called = []
    monkeypatch.setattr(client, "complete", lambda *a, **k: called.append(1) or "x")
    game = persuasion_game(actor="seller", quality="high")
    game["phase"] = "seller_recommendation"
    game["game_state"]["phase"] = "seller_recommendation"
    game["valid_actions"] = {"type": "seller_recommendation", "fields": {"decision": ["yes", "no"]}}
    action = persuasion.decide(parse_game(game), KNOBS)
    assert action == {"decision": "yes"}
    assert not called


def test_bargaining_offer_message_attached(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: "Fair split, deal now.")
    view = parse_game(bargaining_game())
    action = bargaining.decide(view, KNOBS)
    assert action["message"] == "Fair split, deal now."
    # Numbers unchanged by the message layer.
    assert action["alice_gain"] + action["bob_gain"] == pytest.approx(1000)


def test_bargaining_offer_no_message_on_none(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: None)
    view = parse_game(bargaining_game())
    action = bargaining.decide(view, KNOBS)
    assert "message" not in action


def test_bargaining_no_message_when_messages_not_allowed(monkeypatch):
    called = []
    monkeypatch.setattr(client, "complete", lambda *a, **k: called.append(1) or "x")
    view = parse_game(bargaining_game(game_state={"messages_allowed": False}))
    action = bargaining.decide(view, KNOBS)
    assert "message" not in action
    assert not called


def test_negotiation_offer_and_counter_messages(monkeypatch):
    monkeypatch.setattr(client, "complete", lambda *a, **k: "A fair market price.")
    offer_action = negotiation.decide(parse_game(negotiation_game(role="seller")), KNOBS)
    assert offer_action["message"] == "A fair market price."

    # Lowball offer to the seller forces reject-with-counter.
    decision_view = parse_game(negotiation_decision(role="seller", offer_price=10.0))
    action = negotiation.decide(decision_view, KNOBS)
    assert action["decision"] == "RejectOffer"
    assert action["message"] == "A fair market price."
    assert "product_price" in action


def test_negotiation_decisions_get_no_llm_call_on_accept(monkeypatch):
    called = []
    monkeypatch.setattr(client, "complete", lambda *a, **k: called.append(1) or "x")
    # Buyer offered far below value on the final round -> plain accept.
    game = negotiation_decision(role="buyer", offer_price=10.0)
    game["game_state"]["round"] = 10
    action = negotiation.decide(parse_game(game), KNOBS)
    assert action == {"decision": "AcceptOffer"}
    assert not called


# ------------------------------------------------------------------- guard


def test_guard_still_truncates_long_llm_message():
    view = parse_game(persuasion_game(actor="seller", quality="high"))
    action, corrections = guard({"message": "x" * 3000}, view)
    assert len(action["message"]) == 1990
    assert any("truncated" in c for c in corrections)


def test_postprocess_caps_at_600_strips_quotes_and_newlines():
    out = messages._finish('"line one\nline two"' + " y" * 600)
    assert out is not None
    assert "\n" not in out
    assert not out.startswith('"')
    assert len(out) <= 600


def test_postprocess_rejects_refusals_and_empty():
    assert messages._finish("I cannot help with that request.") is None
    assert messages._finish("As an AI, here is a pitch") is None
    assert messages._finish("   ") is None
    assert messages._finish(None) is None


# ------------------------------------------------------------------ breaker


def test_breaker_opens_after_three_failures_and_cools_down(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(client, "_now", lambda: t[0])
    client.record(False)
    client.record(False)
    assert not client.breaker_open()
    client.record(False)
    assert client.breaker_open()
    assert not client.llm_available(KNOBS)
    t[0] += 599.0
    assert client.breaker_open()
    t[0] += 2.0
    assert not client.breaker_open()
    assert client.llm_available(KNOBS)


def test_success_resets_failure_streak():
    client.record(False)
    client.record(False)
    client.record(True)
    client.record(False)
    client.record(False)
    assert not client.breaker_open()


def test_complete_failures_trip_breaker_and_short_circuit(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ConnectionError("down")

    monkeypatch.setattr(client.requests, "post", boom)
    for _ in range(3):
        assert client.complete("hi", timeout_s=1.0) is None
    assert client.breaker_open()
    assert len(calls) == 3
    # Breaker open: no further HTTP attempts.
    assert client.complete("hi", timeout_s=1.0) is None
    assert len(calls) == 3


# ----------------------------------------------------------- prompt hygiene


def _capture_complete(monkeypatch):
    captured = {}

    def fake(prompt, system=None, **kw):
        captured["prompt"] = prompt
        captured["system"] = system or ""
        return "ok message"

    monkeypatch.setattr(client, "complete", fake)
    return captured


def test_recommend_prompt_never_mentions_quality(monkeypatch):
    captured = _capture_complete(monkeypatch)
    out = messages.persuasion_seller_message(
        recommend=True,
        round_=5,
        history_summary="r2:rec/buy-lo r3:rec/pass-lo r4:norec/pass-hi",
    )
    assert out == "ok message"
    blob = (captured["prompt"] + " " + captured["system"]).lower()
    assert "low" not in blob
    assert "quality is" not in blob
    assert "quality" not in blob


def test_seller_message_generator_cannot_separate_on_quality():
    """The signalling invariant: quality must not be an input at all.

    If the wording of a recommendation could depend on the product's quality,
    the buyer separates lies from honest recommendations by text alone and the
    mixing scheme carries zero information.
    """
    import inspect

    params = inspect.signature(messages.persuasion_seller_message).parameters
    assert "quality" not in " ".join(params)


def test_templates_identical_across_quality():
    """Same signal + same game/round -> byte-identical template, whatever the
    product's quality is."""
    from glee_agent.schema import parse_game

    hi = parse_game(persuasion_game(actor="seller", quality="high"))
    lo = parse_game(persuasion_game(actor="seller", quality="low"))
    assert persuasion._seller_message(True, hi) == persuasion._seller_message(True, lo)
    assert persuasion._seller_message(False, hi) == persuasion._seller_message(False, lo)


def test_every_recommend_template_reads_positive():
    """Both pools must classify as their intended signal under the same
    keyword heuristic opponents use, or our own lie accounting misreads it."""
    for t in persuasion._RECOMMEND_TEMPLATES:
        assert persuasion._msg_polarity(t) == "pos", t
    for t in persuasion._PASS_TEMPLATES:
        assert persuasion._msg_polarity(t) == "neg", t


def test_history_summary_encodes_quality_without_the_word_low():
    game = persuasion_game(actor="seller", quality="low")
    game["game_state"]["history"] = [
        {"round": 1, "quality": "low", "bought": True, "seller_message": "great product"},
        {"round": 2, "quality": "high", "bought": False, "seller_message": "skip this"},
    ]
    summary = persuasion._history_summary(parse_game(game))
    assert "low" not in summary
    assert "lo" in summary and "hi" in summary


def test_bargaining_prompt_never_mentions_delta(monkeypatch):
    captured = _capture_complete(monkeypatch)
    out = messages.bargaining_offer_message(65, 3, human_opponent=True)
    assert out == "ok message"
    blob = (captured["prompt"] + " " + captured["system"]).lower()
    assert "delta" not in blob
    assert "discount" not in blob


def test_negotiation_prompt_never_mentions_reservation(monkeypatch):
    captured = _capture_complete(monkeypatch)
    out = messages.negotiation_offer_message("seller", 175.0, 2)
    assert out == "ok message"
    blob = (captured["prompt"] + " " + captured["system"]).lower()
    assert "reservation" not in blob
    assert "175.00" in captured["prompt"]


# ---------------------------------------------------------------- live smoke


@pytest.mark.skipif(not REAL_KEY, reason="LLM_API_KEY not configured")
def test_live_gateway_smoke(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", REAL_KEY)
    if REAL_BASE:
        monkeypatch.setenv("LLM_API_BASE", REAL_BASE)
    if REAL_MODEL:
        monkeypatch.setenv("LLM_MODEL", REAL_MODEL)
    client.reset_breaker()
    out = client.complete("Reply with the single word: pong", max_tokens=10, timeout_s=8.0)
    if out is None:
        # A live-gateway outage (e.g. "no available channel for model") must
        # not redden the suite; production degrades to templates anyway.
        pytest.skip(f"gateway returned no completion for model {REAL_MODEL!r}")
    assert out.strip()
