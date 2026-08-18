"""LLM message layer: free-text "message" fields only.

The deterministic core decides all numbers and accept/reject/buy decisions;
this package only decorates actions with natural-language messages. Any
failure here degrades to template/no-message behavior with zero game risk.
"""

from . import client, messages

__all__ = ["client", "messages"]
