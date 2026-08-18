"""Defensive parsing of the server's game dict into typed views.

Never trusts field presence or types: every accessor has a default, every
number is coerced, and anything surprising is recorded on the view for the
logger to pick up. Strategies read ONLY from these views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _num(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


@dataclass
class GameView:
    raw: dict
    game_id: str = ""
    family: str = ""
    your_player: str = ""            # "player_1" | "player_2"
    phase: str = ""
    action_type: str = ""            # valid_actions["type"]
    action_fields: dict = field(default_factory=dict)
    opponent_type: str = "hidden"    # "agent" | "human" | "hidden"
    opponent_name: str | None = None
    state: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    round: int = 1
    max_rounds: int | None = None
    horizon_known: bool = False
    messages_allowed: bool = True
    complete_information: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def my_index(self) -> int:
        """1 or 2."""
        return 2 if self.your_player == "player_2" else 1

    @property
    def opp_player(self) -> str:
        return "player_2" if self.my_index == 1 else "player_1"


def parse_game(game: Any) -> GameView:
    view = GameView(raw=game if isinstance(game, dict) else {})
    g = view.raw
    if not isinstance(game, dict):
        view.warnings.append(f"game is not a dict: {type(game).__name__}")
        return view

    view.game_id = _text(g.get("game_id"))
    view.family = _text(g.get("game_family"))
    view.your_player = _text(g.get("your_player"), "player_1")
    view.phase = _text(g.get("phase"))

    va = g.get("valid_actions")
    if isinstance(va, dict):
        view.action_type = _text(va.get("type"))
        fields_ = va.get("fields")
        view.action_fields = fields_ if isinstance(fields_, dict) else {}
    else:
        view.warnings.append("valid_actions missing or not a dict")

    opp = g.get("opponent")
    if isinstance(opp, dict):
        view.opponent_type = _text(opp.get("type"), "hidden")
        name = opp.get("name")
        view.opponent_name = name if isinstance(name, str) else None

    state = g.get("game_state")
    view.state = state if isinstance(state, dict) else {}
    if not isinstance(state, dict):
        view.warnings.append("game_state missing or not a dict")

    hist = view.state.get("history")
    view.history = hist if isinstance(hist, list) else []

    rnd = _num(view.state.get("round"))
    view.round = int(rnd) if rnd is not None and rnd >= 1 else 1
    mr = _num(view.state.get("max_rounds"))
    view.max_rounds = int(mr) if mr is not None and mr >= 1 else None
    view.horizon_known = bool(view.state.get("horizon_known", view.max_rounds is not None))
    view.messages_allowed = bool(view.state.get("messages_allowed", True))
    view.complete_information = bool(view.state.get("complete_information", False))
    return view


# ---------------------------------------------------------------- bargaining

@dataclass
class BargainingState:
    view: GameView
    money: float = 0.0
    my_delta: float = 1.0
    opp_delta: float | None = None     # None when hidden
    last_offer_my_gain: float | None = None
    last_offer_opp_gain: float | None = None
    i_am_proposer: bool = False


def parse_bargaining(view: GameView) -> BargainingState:
    s = view.state
    b = BargainingState(view=view)
    b.money = _num(s.get("money_to_divide"), 0.0) or 0.0
    my_i, opp_i = view.my_index, 3 - view.my_index
    b.my_delta = _num(s.get(f"delta_{my_i}"), 1.0) or 1.0
    b.opp_delta = _num(s.get(f"delta_{opp_i}"))
    proposer = _text(s.get("proposer")) or _text(s.get("current_player"))
    b.i_am_proposer = proposer == view.your_player

    lo = s.get("last_offer")
    if isinstance(lo, dict):
        b.last_offer_my_gain = _num(lo.get(f"player_{my_i}_gain"))
        b.last_offer_opp_gain = _num(lo.get(f"player_{opp_i}_gain"))
        # Some payloads name gains alice_gain/bob_gain instead.
        if b.last_offer_my_gain is None:
            names = {1: "alice_gain", 2: "bob_gain"}
            b.last_offer_my_gain = _num(lo.get(names[my_i]))
            b.last_offer_opp_gain = _num(lo.get(names[opp_i]))
    return b


# ---------------------------------------------------------------- negotiation

@dataclass
class NegotiationState:
    view: GameView
    my_role: str = "buyer"             # "seller" | "buyer"
    my_value: float | None = None
    opp_value: float | None = None     # None when hidden
    last_offer_price: float | None = None
    last_offer_mine: bool = False


def parse_negotiation(view: GameView) -> NegotiationState:
    s = view.state
    n = NegotiationState(view=view)
    me = view.your_player
    n.my_role = _text(s.get(f"{me}_role"), "seller" if view.my_index == 1 else "buyer")
    n.my_value = _num(s.get(f"{me}_value"))
    n.opp_value = _num(s.get(f"{view.opp_player}_value"))

    lo = s.get("last_offer")
    if isinstance(lo, dict):
        n.last_offer_price = _num(lo.get("price")) or _num(lo.get("product_price"))
        n.last_offer_mine = _text(lo.get("from_player")) == me
    return n


# ---------------------------------------------------------------- persuasion

@dataclass
class PersuasionState:
    view: GameView
    price: float = 0.0
    p: float = 0.5
    v: float | None = None             # seller may not know v
    u: float = 0.0
    total_rounds: int | None = None
    i_am_seller: bool = False
    current_quality: str | None = None  # seller only
    seller_message: str | None = None
    binary_mode: bool = False
    my_total_payoff: float = 0.0


def parse_persuasion(view: GameView) -> PersuasionState:
    s = view.state
    ps = PersuasionState(view=view)
    ps.price = _num(s.get("product_price"), 0.0) or 0.0
    ps.p = _num(s.get("p"), 0.5) or 0.5
    ps.v = _num(s.get("v"))
    ps.u = _num(s.get("u"), 0.0) or 0.0
    tr = _num(s.get("total_rounds"))
    ps.total_rounds = int(tr) if tr else None
    ps.i_am_seller = view.my_index == 1
    q = s.get("current_quality")
    ps.current_quality = q if q in ("high", "low") else None
    sm = s.get("seller_message")
    if isinstance(sm, str):
        ps.seller_message = sm
    elif isinstance(sm, dict):
        ps.seller_message = _text(sm.get("message")) or _text(sm.get("decision")) or None
    ps.binary_mode = _text(s.get("seller_message_type"), "text") == "binary"
    key = "seller_total_payoff" if ps.i_am_seller else "buyer_total_payoff"
    ps.my_total_payoff = _num(s.get(key), 0.0) or 0.0
    return ps
