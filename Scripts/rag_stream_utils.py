"""Extract plain text from LangChain / Gemini stream chunks."""

from __future__ import annotations

import ast
import re
from typing import Any

def _looks_like_blocks_repr(s: str) -> bool:
    t = s.strip()
    if not t.startswith("["):
        return False
    return "'text'" in t or '"text"' in t


def _extract_text_from_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _parse_blocks_string(s: str) -> str | None:
    """Parse Gemini/LangChain stringified content-block lists."""
    if not _looks_like_blocks_repr(s):
        return None
    try:
        parsed = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return _extract_text_from_blocks(parsed)
    # Fallback: regex for 'text': '...' or "text": "..."
    parts = re.findall(
        r"['\"]text['\"]\s*:\s*['\"]((?:[^'\"\\]|\\.)*)['\"]",
        s,
    )
    return "".join(parts) if parts else None


def stream_chunk_to_text(chunk: Any) -> str:
    """
    Return only user-visible text from one stream chunk.

    Handles plain strings, block lists, and stringified block reprs from Gemini.
    """
    content = getattr(chunk, "content", None)

    if isinstance(content, list):
        return _extract_text_from_blocks(content)

    if isinstance(content, str):
        if not content:
            return ""
        parsed = _parse_blocks_string(content)
        if parsed is not None:
            return parsed
        if not _looks_like_blocks_repr(content):
            return content
        return ""

    text_attr = getattr(chunk, "text", None)
    if text_attr is None:
        return ""
    s = str(text_attr)
    if not s:
        return ""
    parsed = _parse_blocks_string(s)
    if parsed is not None:
        return parsed
    if _looks_like_blocks_repr(s):
        return ""
    return s
