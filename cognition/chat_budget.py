"""Conservative chat-request fitting; estimates are not model token counts."""
from __future__ import annotations

import re

_OPTIONAL = re.compile(
    r'<(?P<tag>memory_context|knowledge_context|codebase_context|wiki_context|'
    r'wiki_knowledge_context|lore|lorebook|search_results)\b[^>]*>.*?</(?P=tag)>', re.S)


class ContextBudgetError(ValueError):
    pass


def estimate_messages(messages):
    total = 0
    for message in messages:
        text = str(message.get('content') or '')
        ascii_chars = sum(ord(c) < 128 for c in text)
        non_ascii_bytes = len(text.encode('utf-8')) - ascii_chars
        total += 8 + (ascii_chars + 2) // 3 + (non_ascii_bytes + 1) // 2
    return total


def fit_chat_request(messages, context_size, max_tokens):
    """Preserve the current task and instructions without mutating history.

    Leave ten percent for estimation/template overhead. Only named retrieved
    blocks may be omitted from system messages; core instructions are retained.
    """
    working = [dict(message) for message in messages]
    ceiling = max(1, int(context_size * 0.9))
    reserve = max(1, int(max_tokens))
    budget = ceiling - reserve
    task = next((m for m in reversed(working) if m.get('role') == 'user'), None)
    if task is None:
        raise ContextBudgetError('No current user request was supplied')
    # Remove complete older turns, including their assistant reply.
    while estimate_messages(working) > budget:
        task_index = next(i for i, m in enumerate(working) if m is task)
        candidates = [i for i, m in enumerate(working)
                      if m.get('role') != 'system' and i < task_index]
        if not candidates:
            break
        start = candidates[0]
        end = start + 1
        while end < len(working) and working[end] is not task and working[end].get('role') == 'assistant':
            end += 1
        del working[start:end]
    # Omit largest optional retrieval blocks first; never truncate the user.
    while estimate_messages(working) > budget:
        blocks = [(len(match.group()), message, match) for message in working
                  if message.get('role') == 'system'
                  for match in _OPTIONAL.finditer(message.get('content') or '')
                  if '[Omitted for context size.' not in match.group()]
        if not blocks:
            break
        _, message, match = max(blocks, key=lambda item: item[0])
        replacement = (f'<{match.group("tag")}>[Omitted for context size. '
                       f'Do not infer facts from missing evidence.]</{match.group("tag")}>')
        original = message['content']
        message['content'] = original[:match.start()] + replacement + original[match.end():]
    available = ceiling - estimate_messages(working)
    if available < min(128, reserve):
        raise ContextBudgetError('The current request and required instructions exceed the context budget; shorten the request or reduce the configured persona')
    return working, min(reserve, available)
