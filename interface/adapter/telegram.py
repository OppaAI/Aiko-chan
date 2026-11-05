from __future__ import annotations

import asyncio
import threading

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
except ImportError:
    Application = None


class TelegramAdapter(AdapterBase):
    """Telegram bot adapter using python-telegram-bot."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._token: str = ""
        self._app: Application | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._user_id_map: dict[str, str] = self._parse_user_map(
            self._get_env("TELEGRAM_USER_ID_MAP")
        )

    def _read_config(self) -> dict[str, str]:
        return {
            "token": self._get_env("TELEGRAM_BOT_TOKEN"),
        }

    def _canonical_user_id(self, raw: str) -> str:
        if raw not in self._user_id_map:
            log.debug("[telegram] user %s -> %s (default)", raw, raw)
            return raw
        mapped = self._user_id_map[raw]
        log.debug("[telegram] user %s -> %s (mapped)", raw, mapped)
        return mapped

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if Application is None:
            log.error("[telegram] python-telegram-bot not installed. pip install python-telegram-bot")
            return
        cfg = self._read_config()
        self._token = cfg.get("token", "")
        if not self._token:
            if not getattr(self, "_token_warned", False):
                log.error("[telegram] TELEGRAM_BOT_TOKEN not set in .env")
                self._token_warned = True
            return
        self._token_warned = False

        self._app = Application.builder().token(self._token).build()

        async def handle_text(update: Update, _context) -> None:
            if not update.message or not update.message.text:
                return
            user = update.message.from_user
            if user is None:
                return
            cid = str(update.effective_chat.id) if update.effective_chat else str(user.id)
            uid = self._canonical_user_id(str(user.id))
            display = user.full_name or user.username or uid
            text = update.message.text.strip()
            if not text:
                return
            if text.startswith("/start") or text.startswith("/help"):
                await update.message.reply_text(
                    "Hi! I'm Aiko-chan. Just send me a message and we can chat."
                )
                return
            log.debug("[telegram] msg from %s: %.60s", display, text)
            # Do NOT block on session.wait(): this handler runs on PTB's
            # asyncio event loop; blocking it during inference starves the
            # polling loop. The session runs inference on its own thread
            # and delivers the reply via _on_platform_response -> send_message,
            # which schedules back onto this loop with run_coroutine_threadsafe.
            self.handle_message(cid, uid, display, text)

        if self._app:
            self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
            self._app.add_handler(CommandHandler("start", handle_text))
            self._app.add_handler(CommandHandler("help", handle_text))

            app = self._app

            def poll() -> None:
                loop = asyncio.new_event_loop()
                self._loop = loop
                asyncio.set_event_loop(loop)
                try:
                    if not self._running:
                        return
                    app.run_polling(drop_pending_updates=True, stop_signals=None,
                                    close_loop=False)
                except Exception:
                    log.exception("[telegram] polling failed")
                finally:
                    self._running = False
                    self._loop = None
                    loop.close()

            self._thread = threading.Thread(target=poll, daemon=True)
            self._running = True
            self._thread.start()
            log.info("[telegram] Adapter started")

    def stop(self) -> None:
        self._running = False
        app, loop = self._app, self._loop
        if app is not None and loop is not None and loop.is_running():
            # Let run_polling perform updater/application shutdown in its own loop.
            loop.call_soon_threadsafe(app.stop_running)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)

    def send_message(self, conversation_id: str, text: str) -> None:
        """Send a text message, blocking until Telegram confirms delivery.

        Long texts are split into 4096-char chunks (Telegram's limit),
        sent in order. Raises on delivery failure so callers can report
        it honestly. Must not be called from the adapter's own asyncio
        event loop thread (the blocking wait would deadlock).
        """
        app = self._app
        loop = self._loop
        if not app or loop is None or not loop.is_running() or not self._running:
            raise RuntimeError("telegram adapter not started")
        import asyncio
        try:
            chat_id = int(conversation_id)

            async def send_chunks() -> None:
                for start in range(0, len(text), 4096):
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=text[start:start + 4096],
                    )

            if threading.current_thread() is self._thread:
                raise RuntimeError("cannot synchronously send from Telegram event loop")
            future = asyncio.run_coroutine_threadsafe(send_chunks(), loop)
            try:
                future.result(timeout=30)
            except TimeoutError:
                future.cancel()
                raise
        except Exception as exc:
            log.error("[telegram] Failed to send to %s: %s", conversation_id, exc)
            raise
