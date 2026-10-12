"""
toolkit/aria_snapshot.py

Accessibility-tree filtering for agent browsing — Python port of
Bright Data's aria_snapshot_filter.js.

Given a Playwright/Crawl4AI accessibility snapshot (mode='ai'),
extract interactive elements (links, buttons, inputs, etc.) with
their names, roles, and URLs. Returns a compact structured list
suitable for LLM context or downstream extraction.

This improves on raw HTML/text extraction by giving the model
semantic structure: "button 'Submit' -> /api/submit", not
"<div class='btn'>Submit</div>".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "combobox",
    "checkbox", "radio", "switch", "slider", "tab",
    "menuitem", "option",
}


@dataclass
class AriaElement:
    ref: str
    role: str
    name: str = ""
    url: Optional[str] = None


def parse_playwright_snapshot(snapshot_text: str) -> list[AriaElement]:
    """Parse a Playwright ariaSnapshot({mode: 'ai'}) text into AriaElements."""
    elements: list[AriaElement] = []
    lines = snapshot_text.split("\n")

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if not trimmed or not trimmed.startswith("-"):
            continue

        # - role [ref=123] "Name"
        ref_match = re.search(r"\[ref=([^\]]+)\]", trimmed)
        if not ref_match:
            continue
        ref = ref_match.group(1)

        role_match = re.match(r"-\s+([a-zA-Z]+)", trimmed)
        if not role_match:
            continue
        role = role_match.group(1).lower()

        if role not in INTERACTIVE_ROLES:
            continue

        name = ""
        name_match = re.search(r'"([^"]*)"', trimmed)
        if name_match:
            name = name_match.group(1)

        # Next line may contain /url: "https://..."
        url = None
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            url_match = re.search(r"/url:\s*(.+)", next_line)
            if url_match:
                url = url_match.group(1).strip().strip("\"'")

        elements.append(AriaElement(ref=ref, role=role, name=name, url=url))

    return elements


def format_compact(elements: list[AriaElement]) -> str:
    """Format elements as compact lines: [ref] role "name" -> url"""
    lines = []
    for el in elements:
        parts = [f"[{el.ref}]", el.role]
        if el.name:
            name = el.name if len(el.name) <= 60 else el.name[:57] + "..."
            parts.append(f'"{name}"')
        if el.url and not el.url.startswith("#"):
            url = el.url if len(el.url) <= 50 else el.url[:47] + "..."
            parts.append(f"-> {url}")
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "No interactive elements found"


def filter_snapshot(snapshot_text: str) -> str:
    """Convenience: parse + format in one call."""
    try:
        elements = parse_playwright_snapshot(snapshot_text)
        return format_compact(elements)
    except Exception as e:
        return f"Error filtering snapshot: {e}"


def format_dom_elements(elements: list[dict]) -> Optional[str]:
    """Format raw DOM element dicts (from page.locator().evaluate_all)."""
    if not elements:
        return None
    lines = []
    for el in elements:
        parts = [f"[{el.get('ref', '?')}]", el.get("role", "unknown")]
        if el.get("name"):
            name = el["name"]
            if len(name) > 60:
                name = name[:57] + "..."
            parts.append(f'"{name}"')
        if el.get("url") and not el["url"].startswith("#"):
            url = el["url"]
            if len(url) > 50:
                url = url[:47] + "..."
            parts.append(f"-> {url}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


async def capture_filtered_snapshot(page) -> dict:
    """
    Capture a filtered ARIA snapshot from a Playwright page.

    Returns dict with url, title, aria_snapshot (filtered), and
    interactive_elements (list of AriaElement).
    """
    full_snapshot = await page.aria_snapshot({"mode": "ai"})
    elements = parse_playwright_snapshot(full_snapshot)
    return {
        "url": page.url,
        "title": await page.title(),
        "aria_snapshot": format_compact(elements),
        "interactive_elements": elements,
    }