"""Shared interpretation of raw, titled-JSON, and legacy tool results."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ERROR = re.compile(r'^\[(?P<label>[^:\]]+)(?::\s*(?P<detail>.*))?\]$', re.DOTALL)
_FAILED = {'error', 'failed', 'failure', 'blocked', 'timeout', 'needs_approval',
           'waiting_for_approval', 'conscience_blocked', 'conscience_refuse', 'conscience_caution', 'conscience_unavailable', 'conscience_skip'}


@dataclass(frozen=True)
class Outcome:
    content: str
    ok: bool = True
    error_type: str | None = None
    retryable: bool = False
    detail: str = ''


def tool_outcome(value: Any) -> Outcome:
    """Read status before truncation; JSON arrays are successful data, not errors."""
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = content.strip()
    payload = value if isinstance(value, (dict, list)) else None
    if payload is None:
        candidate = text
        header, sep, body = text.partition('\n')
        if sep and header.startswith('[') and header.endswith(']'):
            candidate = body.strip()
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            pass
    if isinstance(payload, dict):
        status = str(payload.get('status') or '').lower()
        code = payload.get('returncode')
        failed = (payload.get('ok') is False or payload.get('success') is False
                  or status in _FAILED
                  or isinstance(code, (int, float)) and not isinstance(code, bool) and code != 0
                  or bool(payload.get('error')) and payload.get('ok') is not True)
        if failed:
            kind = str(payload.get('error_type') or status or 'tool_error')
            if status in {'waiting_for_approval', 'needs_approval'}:
                kind = 'needs_approval'
            return Outcome(content, False, kind, payload.get('retryable') is True,
                           str(payload.get('error') or payload.get('reason') or kind))
        return Outcome(content)
    if isinstance(payload, list):
        return Outcome(content)
    match = _ERROR.fullmatch(text)
    if match:
        label = match['label'].lower()
        detail = match['detail'] or label
        retryable = (any(word in label for word in ('search failed', 'fetch failed'))
                     or any(word in detail.lower() for word in ('timeout', 'connection', 'empty response')))
        return Outcome(content, False, label.replace(' ', '_'), retryable, detail)
    return Outcome(content)
