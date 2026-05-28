"""
core/tools.py

Aiko's tool belt — starting with web search via SearXNG.
All tools are plain functions that return strings ready for context injection.
"""

import os
import requests


# ── config ────────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")


# ── web search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web via SearXNG and return a compact result string
    ready for injection into Aiko's context.
    """
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q":      query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return f"[search failed: could not reach SearXNG at {SEARXNG_URL}]"
    except requests.exceptions.Timeout:
        return "[search failed: timed out]"
    except requests.exceptions.RequestException as e:
        return f"[search failed: request error: {e}]"
    except ValueError:
        return "[search failed: invalid JSON response]"

    results = data.get("results", [])[:max_results]
    if not results:
        return f"[no results found for: {query}]"

    lines = [f"[Web search results for: {query}]"]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "").strip()
        url     = r.get("url", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    return "\n\n".join(lines)
