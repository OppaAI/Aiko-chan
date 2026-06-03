"""
main.py

Aiko-chan CLI — entry point and session orchestrator.

Usage:
    python main.py               # full voice — ASR (faster-whisper) + TTS (MioTTS)
    python main.py --text        # keyboard input + no TTS
    python main.py --debug       # show memory debug info each turn
    python main.py --clear-mem   # wipe all stored memories and exit

Responsibilities:
    - Parse CLI arguments
    - Delegate subsystem boot to core/wakeup.py
    - Drive the TUI init phase and transition to active chat
    - Run the main input → inference → render loop
    - Handle commands (/quit, /reset, /memory, /clear, /remember, /think,
                       /voice, /listen, /web, /help)
    - Clean shutdown on Ctrl-C / Ctrl-D
"""

import argparse
import os
import re
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import curses
import logging
logging.disable(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

from core.silence import silent_stderr
from core.log     import get_logger
from core.wakeup  import AikoWakeup

log = get_logger(__name__)

with silent_stderr():
    from core.memorize import AikoMemorize

from tui.tui import AikoTUI

# ── env ───────────────────────────────────────────────────────────────────────

AI_NAME = os.getenv("AI_NAME", "Aiko")
USER_ID = os.getenv("USER_ID", "")


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan CLI")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + no TTS  (default: ASR + TTS)")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def _run(stdscr, args):
    """
    Orchestrate the full session lifecycle from boot to shutdown inside the
    curses wrapper.

    Stages:
        1. Spawn the TUI and begin the init spin loop.
        2. Delegate all subsystem boot to AikoWakeup, passing TUI callbacks.
        3. Transition the TUI to the active chat phase.
        4. Enter the main input → inference → render loop.
        5. On exit, wait for any background memory writes to complete.
    """
    tui = AikoTUI(stdscr, no_voice=args.text, debug=args.debug)

    # ── init spin ─────────────────────────────────────────────────────────────

    spin_stop = threading.Event()
    spin_t    = threading.Thread(target=tui.spin_loop, args=(spin_stop,), daemon=True)
    spin_t.start()

    # ── boot all subsystems via wakeup ────────────────────────────────────────

    result = AikoWakeup(text_mode=args.text).boot(
        on_loading = tui.step_loading,
        on_done    = tui.step_done,
        on_skip    = tui.step_skip,
    )

    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen

    # ── transition to chat ────────────────────────────────────────────────────

    spin_stop.set()
    spin_t.join()
    tui.status_finish()
    tui._draw()

    tts_enabled = not args.text
    asr_enabled = not args.text

    # ── main loop ─────────────────────────────────────────────────────────────

    while True:
        try:
            if listen and asr_enabled:
                user_input = tui.get_voice_input(
                    listen,
                    wait_fn=speak.wait if speak else None,
                )
            else:
                user_input = tui.get_input()
        except KeyboardInterrupt:
            tui.add_message('sys', "Fine... I'll be here when you come back.")
            tui._draw()
            think.wait_for_memory()
            time.sleep(0.8)
            return

        if not user_input:
            continue

        # ── commands ──────────────────────────────────────────────────────────

        if user_input.startswith('/'):
            cmd = user_input.lower().strip()

            if cmd in ('/quit', '/exit'):
                tui.add_message('sys', 'Already leaving? ...Be safe out there.')
                tui._draw()
                think.wait_for_memory()
                time.sleep(0.8)
                return

            elif cmd == '/reset':
                think.reset_context()
                tui.add_message('sys', 'Short-term context cleared.')

            elif cmd == '/memory':
                all_mem = memorize.get_all()
                if not all_mem:
                    tui.add_message('sys', 'No memories stored yet.')
                else:
                    tui.add_message('sys', f'{len(all_mem)} memories stored:')
                    for i, m in enumerate(all_mem, 1):
                        tui.add_message('sys',
                            f'  {i:02d}. {m.get("memory") or m.get("text") or m}')

            elif cmd == '/clear':
                memorize.clear()
                tui.add_message('sys', 'All persistent memories cleared.')

            elif cmd == '/remember':
                # pin the last user + assistant exchange permanently
                turn = think.last_turn()
                if not turn:
                    tui.add_message('sys', 'Nothing to remember yet — send a message first.')
                else:
                    think.wait_for_memory()
                    user_text, ai_text = turn
                    msgs = [
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": ai_text},
                    ]
                    ok = memorize.pin(msgs)
                    if ok:
                        tui.add_message('sys', "Got it — I'll remember that forever. 📌")
                    else:
                        tui.add_message('sys', 'Failed to pin memory — check logs.')

            elif cmd.startswith('/think'):
                query = user_input[6:].strip()
                if not query:
                    tui.add_message('sys', 'Usage: /think <question>')
                    tui._draw()
                    continue

                think.set_reasoning(True)
                tui.add_message('you', f'[think] {query}')
                tui.turn_start()
                tui._draw()

                raw_chunks     = []
                in_think_block = False
                think_closed   = False

                def _think_token_cb(token):
                    nonlocal in_think_block, think_closed
                    raw_chunks.append(token)
                    assembled = "".join(raw_chunks)

                    if not think_closed:
                        if "<think>" in assembled and not in_think_block:
                            in_think_block = True
                        if in_think_block:
                            if "</think>" in assembled:
                                in_think_block = False
                                think_closed   = True
                            return

                    tui.stream_token(token)
                    tui._draw(buf=[])

                think.chat(query, token_callback=_think_token_cb)

                assembled_full   = "".join(raw_chunks)
                scratchpad_match = re.search(r"<think>(.*?)</think>", assembled_full, re.DOTALL)
                if scratchpad_match:
                    inner = scratchpad_match.group(1).strip()
                    if inner:
                        tui.add_message('sys',
                            f'[scratchpad] {inner[:300]}{"…" if len(inner) > 300 else ""}')

                tui.stream_commit()
                tui._draw()
                continue

            elif cmd == '/voice':
                if speak is None:
                    tui.add_message('sys', 'TTS unavailable — started in --text mode.')
                else:
                    tts_enabled = not tts_enabled
                    think.set_speak(speak if tts_enabled else None)
                    tui._stats['tts_on'] = tts_enabled
                    tui.add_message('sys',
                        f'Voice output (TTS): {"ON  🔊" if tts_enabled else "OFF 🔇"}')

            elif cmd == '/listen':
                if listen is None:
                    tui.add_message('sys', 'ASR unavailable — started in --text mode.')
                else:
                    asr_enabled = not asr_enabled
                    tui._stats['asr_on'] = asr_enabled
                    tui.add_message('sys',
                        f'Voice input  (ASR): {"ON  🎤" if asr_enabled else "OFF ⌨ "}')

            elif cmd == '/help':
                for line in [
                    '/quit /exit        — end session',
                    '/reset             — clear short-term context',
                    '/clear             — wipe long-term memories',
                    '/remember          — pin last turn forever (decay-proof)',
                    '/memory            — show stored memories',
                    '/think <question>  — reason step-by-step (single-shot, 3× token budget)',
                    '/web <query>       — web search',
                    '/voice             — toggle TTS on/off',
                    '/listen            — toggle ASR on/off',
                    '/help              — show this list',
                ]:
                    tui.add_message('sys', line)

            elif cmd.startswith('/web '):
                query = user_input[5:].strip()
                if not query:
                    tui.add_message('sys', 'Usage: /web <query>')
                else:
                    try:
                        from core.tools import web_search
                    except ImportError as e:
                        tui.add_message('sys', f'Web search unavailable: {e}')
                        tui._draw()
                        continue
                    tui.add_message('sys', f'Searching: "{query}"')
                    tui._draw()
                    try:
                        results = web_search(query)
                    except Exception as e:
                        tui.add_message('sys', f'Search failed: {e}')
                        tui._draw()
                        continue
                    think._history.append({"role": "user", "content": results})
                    tui.turn_start()
                    def _web_token_cb(token):
                        tui.stream_token(token)
                        tui._draw(buf=[])
                    think.chat(f"Based on the search results, answer: {query}",
                               token_callback=_web_token_cb)
                    tui.stream_commit()

            else:
                tui.add_message('sys', f'Unknown command: {user_input}')

            tui._draw()
            continue

        # ── normal turn ───────────────────────────────────────────────────────

        if args.debug:
            hits = memorize.search(user_input)
            if hits:
                tui.add_message('sys', f'{len(hits)} memories retrieved:')
                for m in hits:
                    tui.add_message('sys',
                        f'  → {m.get("memory") or m.get("text") or m}')

        tui.add_message('you', user_input)
        tui.turn_start()
        tui._draw()

        def token_cb(token):
            if token.startswith("__SEARCHING__:"):
                query = token.split(":", 1)[1].strip()
                tui.stream_commit()
                tui.add_message('sys', f'Searching the web for: "{query}"...')
                tui._draw(buf=[])
            else:
                tui.stream_token(token)
                tui._draw(buf=[])

        think.chat(user_input, token_callback=token_cb)
        tui.stream_commit()
        tui._draw()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Primary entry point for the Aiko-chan CLI."""
    args = parse_args()
    if args.clear_mem:
        log.info("Clearing all memories...")
        AikoMemorize().clear()
        sys.exit(0)
    curses.wrapper(lambda scr: _run(scr, args))


if __name__ == '__main__':
    main()
