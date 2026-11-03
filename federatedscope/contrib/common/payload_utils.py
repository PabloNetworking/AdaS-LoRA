"""Helpers for the wrapped message payloads exchanged by client.py and server.py."""
from __future__ import annotations

from typing import Any


def is_wrapped_payload(content: Any) -> bool:
    """Return True when content is a dict payload that carries model_para."""
    return isinstance(content, dict) and 'model_para' in content


def extract_model_para(content: Any, default: Any = None) -> Any:
    """Return model_para when the payload is wrapped, else return content."""
    if is_wrapped_payload(content):
        return content.get('model_para', default)
    return content if content is not None else default


