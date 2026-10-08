from interface.adapter.base import AdapterBase, ConversationSession
from interface.adapter.discord import DiscordAdapter
from interface.adapter.telegram import TelegramAdapter
from interface.adapter.slack import SlackAdapter
from interface.adapter.matrix import MatrixAdapter
from system.log import get_logger

log = get_logger(__name__)

ADAPTER_REGISTRY: dict[str, type[AdapterBase]] = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "slack": SlackAdapter,
    "matrix": MatrixAdapter,
}

# Live handles of adapters started by start_background_adapters(), keyed by
# lowercase adapter name. Agent tools (e.g. telegram_send) resolve the running
# instance through get_running_adapter() instead of constructing their own.
RUNNING_ADAPTERS: dict[str, AdapterBase] = {}


def get_running_adapter(name: str) -> AdapterBase | None:
    """Return the live adapter instance for `name`, or None if not started."""
    return RUNNING_ADAPTERS.get((name or "").strip().lower())


def start_background_adapters(think, memorize, names: list[str] | None = None) -> list[AdapterBase]:
    """Start configured two-way messenger adapters beside WebUI/CLI sessions.

    These adapters are intentionally headless: inbound messages are routed
    through the same memory and intent pipeline as local chat turns, and
    replies are sent back to the originating messaging service without
    echoing the exchange into the active WebUI or CLI transcript.
    """
    import os

    raw = names if names is not None else [
        part.strip().lower()
        for part in os.getenv("AIKO_MESSENGER_ADAPTERS", "").split(",")
        if part.strip()
    ]
    adapters: list[AdapterBase] = []
    for name in raw:
        cls = ADAPTER_REGISTRY.get(name)
        if cls is None:
            log.warning("[adapter] ignoring unsupported two-way adapter: %s", name)
            continue
        try:
            adapter = cls()
            adapter.boot(think, memorize)
            adapter.start()
            adapters.append(adapter)
            RUNNING_ADAPTERS[name] = adapter
            log.info("[adapter] started %s in background", name)
        except Exception:
            log.exception("[adapter] failed to start %s in background", name)
    return adapters


__all__ = [
    "AdapterBase",
    "ConversationSession",
    "DiscordAdapter",
    "TelegramAdapter",
    "SlackAdapter",
    "MatrixAdapter",
    "ADAPTER_REGISTRY",
    "RUNNING_ADAPTERS",
    "get_running_adapter",
    "start_background_adapters",
]
