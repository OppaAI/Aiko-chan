"""
toolkit/browser_session.py

Browser session management for crawl4ai/Playwright — Python port of
Bright Data's browser_session.js.

Provides domain-scoped browser contexts with:
- Single browser launch per domain (reused across fetches)
- Auto-reconnect on disconnection
- Request/response logging per domain
- Credential-redacted URL logging

Integrates with crawl4ai's AsyncWebCrawler but manages the
Playwright browser/context/page lifecycle explicitly for
better control over concurrent fetches and session state.
"""
from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from system.log import get_logger

log = get_logger(__name__)

CRED_URL_RE = re.compile(r"((?:wss?|https?):\/\/)[^/@\s]+@")


def redact_url_creds(value: str) -> str:
    """Redact credentials from URLs in logs."""
    return CRED_URL_RE.sub(r"\1[REDACTED]@", value)


@dataclass
class DomainSession:
    """Per-domain browser state."""
    browser: Optional[object] = None  # playwright Browser
    context: Optional[object] = None  # playwright BrowserContext
    page: Optional[object] = None     # playwright Page
    browser_closed: bool = True
    requests: dict = field(default_factory=dict)  # request -> response


class BrowserSession:
    """
    Manages Playwright browser sessions scoped by domain.

    Usage:
        session = BrowserSession()
        async with session.page_for(url) as page:
            content = await page.content()
            # or use crawl4ai with this page
    """

    def __init__(self):
        self._domain_sessions: dict[str, DomainSession] = {}
        self._playwright = None
        self._browser = None
        self._current_domain = "default"

    async def _ensure_playwright(self):
        """Lazy-initialize Playwright."""
        if self._playwright is None:
            import importlib
            pw = importlib.import_module("playwright.async_api")
            self._playwright = await pw.async_playwright().start()

    def _get_domain(self, url: str) -> str:
        try:
            return urlparse(url).hostname or "default"
        except Exception:
            return "default"

    def _get_session(self, domain: str) -> DomainSession:
        if domain not in self._domain_sessions:
            self._domain_sessions[domain] = DomainSession()
        return self._domain_sessions[domain]

    async def get_browser(self, domain: str = "default") -> object:
        """Get or create browser for domain (with auto-reconnect)."""
        await self._ensure_playwright()
        session = self._get_session(domain)

        if session.browser and not session.browser_closed:
            try:
                # Test connection
                await session.browser.contexts()
                return session.browser
            except Exception as e:
                log.debug("Browser connection lost for %s (%s), reconnecting...",
                          domain, redact_url_creds(str(e)))
                session.browser = None
                session.context = None
                session.page = None
                session.browser_closed = True

        log.debug("Launching browser for domain %s", domain)
        session.browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        session.browser_closed = False

        # Set up disconnect handler
        session.browser.on("disconnected", lambda: self._on_disconnect(domain))

        return session.browser

    def _on_disconnect(self, domain: str):
        """Handle browser disconnect."""
        log.debug("Browser disconnected for domain %s", domain)
        session = self._domain_sessions.get(domain)
        if session:
            session.browser = None
            session.context = None
            session.page = None
            session.browser_closed = True

    async def get_page(self, url: Optional[str] = None) -> object:
        """Get or create a page for the domain of `url`."""
        if url:
            self._current_domain = self._get_domain(url)
        domain = self._current_domain

        session = self._get_session(domain)
        if session.browser_closed or not session.page:
            browser = await self.get_browser(domain)
            contexts = browser.contexts
            if not contexts:
                session.context = await browser.new_context()
            else:
                session.context = contexts[0]

            session.page = await session.context.new_page()

            # Request/response logging
            session.page.on("request", lambda req: session.requests.__setitem__(req, None))
            session.page.on("response",
                           lambda resp: session.requests.__setitem__(resp.request, resp))

            session.browser_closed = False

            async def on_close():
                session.page = None
            session.page.once("close", lambda: asyncio.create_task(self._on_page_close(domain)))

        return session.page

    async def _on_page_close(self, domain: str):
        session = self._domain_sessions.get(domain)
        if session:
            session.page = None

    async def capture_snapshot(self, url: Optional[str] = None, filtered: bool = True) -> dict:
        """Capture ARIA snapshot for a URL (optionally filtered)."""
        page = await self.get_page(url)
        if not page:
            return {"error": "Failed to get page"}

        try:
            if filtered:
                from agentic.toolkit.aria_snapshot import capture_filtered_snapshot
                return await capture_filtered_snapshot(page)
            else:
                full_snapshot = await page.aria_snapshot({"mode": "ai"})
                return {
                    "url": page.url,
                    "title": await page.title(),
                    "aria_snapshot": full_snapshot,
                }
        except Exception as e:
            return {"error": f"Snapshot failed: {e}"}

    async def close_domain(self, domain: str):
        """Close browser/context/page for a domain."""
        session = self._domain_sessions.get(domain)
        if session:
            if session.page:
                await session.page.close()
            if session.context:
                await session.context.close()
            if session.browser:
                await session.browser.close()
            del self._domain_sessions[domain]

    async def close_all(self):
        """Close all domains."""
        for domain in list(self._domain_sessions.keys()):
            await self.close_domain(domain)
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    @asynccontextmanager
    async def page_for(self, url: str):
        """Context manager: get page, ensure cleanup on exception."""
        page = await self.get_page(url)
        try:
            yield page
        except Exception:
            # On error, mark session as closed so it reconnects next time
            domain = self._get_domain(url)
            session = self._domain_sessions.get(domain)
            if session:
                session.browser_closed = True
                session.page = None
            raise

    def get_session_stats(self) -> dict:
        """Debug: return per-domain session state."""
        return {
            d: {
                "has_browser": s.browser is not None,
                "has_page": s.page is not None,
                "closed": s.browser_closed,
                "requests_tracked": len(s.requests),
            }
            for d, s in self._domain_sessions.items()
        }


# Global singleton for convenience
_global_session: Optional[BrowserSession] = None


def get_browser_session() -> BrowserSession:
    """Get the global browser session (lazy init)."""
    global _global_session
    if _global_session is None:
        _global_session = BrowserSession()
    return _global_session


async def close_global_session():
    """Close the global session (for shutdown)."""
    global _global_session
    if _global_session:
        await _global_session.close_all()
        _global_session = None