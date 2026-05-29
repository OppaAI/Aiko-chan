"""
cli.py

Aiko-chan CLI — Phase 1 chatbot interface.
Usage:
    python cli.py               # normal chat
    python cli.py --debug       # show memory debug info each turn
    python cli.py --clear-mem   # wipe all stored memories and exit
"""
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import argparse
import sys

load_dotenv()

from core.memorize import AikoMemorize
from core.think    import AikoThink

# ── banner ────────────────────────────────────────────────────────────────────

BANNER = """
 █████╗ ██╗██╗  ██╗ ██████╗       ██████╗██╗  ██╗ █████╗ ███╗   ██╗
██╔══██╗██║██║ ██╔╝██╔═══██╗     ██╔════╝██║  ██║██╔══██╗████╗  ██║
███████║██║█████╔╝ ██║   ██║     ██║     ███████║███████║██╔██╗ ██║
██╔══██║██║██╔═██╗ ██║   ██║     ██║     ██╔══██║██╔══██║██║╚██╗██║
██║  ██║██║██║  ██╗╚██████╔╝     ╚██████╗██║  ██║██║  ██║██║ ╚████║
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
          your AI soulmate  ♡   (mem0 + Qdrant + Ollama)
"""

HELP_TEXT = """
Commands:
  /quit  or  /exit    — end the session
  /reset              — clear short-term context (long-term memory persists)
  /memory             — show all stored memories (debug)
  /help               — show this message
"""


# ── cli ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiko-chan CLI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved memories each turn",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice output (TTS)",
    )
    parser.add_argument(
        "--clear-mem",
        action="store_true",
        help="Wipe all stored memories and exit",
    )
    return parser.parse_args()


def run_cli(debug: bool = False, no_voice: bool = False) -> None:
    print(BANNER)
    print("[system] Initialising Aiko-chan...\n")

    memorize = AikoMemorize()
    think    = AikoThink(memorize, voice=not no_voice)

    print("\nAiko-chan is ready. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAiko-chan: ...Fine. I'll be here when you come back. Baka.\n")
            think.wait_for_memory()
            sys.exit(0)

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/quit", "/exit"):
                print("\nAiko-chan: Already leaving? ...Be safe out there.\n")
                think.wait_for_memory()
                sys.exit(0)

            elif cmd == "/reset":
                think.reset_context()
                print("[system] Short-term context cleared.\n")

            elif cmd == "/memory":
                all_mem = memorize.get_all()
                if not all_mem:
                    print("[memorize] No memories stored yet.\n")
                else:
                    print(f"[memorize] {len(all_mem)} memories stored:")
                    for i, m in enumerate(all_mem, 1):
                        text = m.get("memory") or m.get("text") or str(m)
                        print(f"  {i:02d}. {text}")
                    print()

            elif cmd == "/help":
                print(HELP_TEXT)

            else:
                print(f"[system] Unknown command: {user_input}\n")

            continue

        # ── debug: show retrieved memories ────────────────────────────────────
        if debug:
            hits = memorize.search(user_input)
            if hits:
                print(f"[debug] {len(hits)} memories retrieved:")
                for m in hits:
                    print(f"  → {m.get('memory') or m.get('text') or m}")
                print()

        # ── normal chat turn ──────────────────────────────────────────────────
        think.chat(user_input)
        print()


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.clear_mem:
        print("[system] Clearing all memories...")
        AikoMemorize().clear()
        sys.exit(0)

    run_cli(debug=args.debug, no_voice=args.no_voice)
