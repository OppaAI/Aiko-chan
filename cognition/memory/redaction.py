"""Credential scrubbing for new memory writes; ordinary contact facts stay intact."""
from __future__ import annotations

import re

REDACTED = "[redacted]"
_NAME = r"(?:[a-z][a-z0-9]*_)*(?:api[_-]?key|access[_-]?(?:token|key)|auth[_-]?token|authorization|client[_-]?secret|secret[_-]?(?:access[_-]?)?key|private[_-]?key|credential|password|passwd|token|secret)"
_KEY = re.compile(rf"{_NAME}", re.I)
_ASSIGNMENT = re.compile(
    rf"(?<![\w-])(?P<key>[\"']?{_NAME}[\"']?)(?P<sep>\s*[:=]\s*)"
    r"(?P<value>\[redacted\]|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;\}\]\&]+)", re.I)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----", re.S)
_PLACEHOLDERS = {"", "[redacted]", "<redacted>", "none", "null", "true", "false",
                 "example", "placeholder", "your_api_key", "your_token", "count", "string", "str"}


def _placeholder(value):
    text = str(value).strip().strip("\"'").casefold()
    return (text in _PLACEHOLDERS or text.startswith(("${", "os.getenv(", "os.environ", "getenv(")))


def redact_text(text: str) -> str:
    """Scrub assignments and recognizable credential formats, preserving emails.

    This is deterministic pattern matching, not a guarantee of detecting every
    secret. It applies to new records and does not rewrite historical databases.
    """
    if not isinstance(text, str):
        return text
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _BEARER.sub("Bearer " + REDACTED, text)
    text = _TOKEN.sub(REDACTED, text)
    return _ASSIGNMENT.sub(lambda m: m.group() if _placeholder(m['value'])
                           else m['key'] + m['sep'] + REDACTED, text)


def redact_data(value):
    """Return a scrubbed copy of JSON-like metadata, including secret-key values."""
    if isinstance(value, dict):
        return {key: (REDACTED if _KEY.fullmatch(str(key)) and item is not None
                      and not _placeholder(item) else redact_data(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    return redact_text(value) if isinstance(value, str) else value
