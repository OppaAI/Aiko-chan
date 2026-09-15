"""
main.py

Aiko-chan launcher: single source of truth for version, CLI parsing, and
process-level traps. Entry point is main(); everything else is internal.

Usage:
    python main.py               # browser WebUI (default) — full voice, ASR + TTS
    python main.py --text        # WebUI, keyboard input + TTS/ASR toggled off
    python main.py --no-asr      # WebUI, keyboard input but keep TTS on
    python main.py --cli         # plain no-curses CLI, for local testing only
    # Two-way messenger adapters (Aiko-Lingo etc.) are spawned by the front
    # ends themselves, not by main.py — they run beside WebUI/CLI when
    # MESSENGER_ADAPTERS is set, but this module never spawns them.
    python main.py --debug       # verbose console logging (LOG_CONSOLE=1, LOG_LEVEL=DEBUG) + memory hits per turn; also implies --trace
    python main.py --trace       brain trace per turn (TRACE_BRAIN=1) without DEBUG-level log spam
    python main.py --clear-mem   # wipe all stored memories and exit
    python main.py --logout      # clear stored CLI (GitHub OAuth) auth token and exit
    python main.py --name <name> # set CLI display name (only when GitHub OAuth isn't configured)

This module only parses arguments and dispatches to the right front end:
    - interface/webui/webui.py  -> run_webui(args)   (default)
    - interface/cli/cli.py      -> run_cli(args)     (--cli)

main.py does NOT call AikoWakeup().boot() itself — each front end owns its
 boot timing, because the two have genuinely different requirements:
    - WebUI (interface/webui/webui.py, run_webui()): boot runs to completion
      BEFORE the HTTP/WS server opens (constructed with defer_servers=True),
      so browsers never see a half-booted Aiko. Post-login work (memory
      cleanup, playbook/social seeding) runs in PARALLEL after the first
      authenticated connect via system/prepare.run_post_auth().
    - CLI (interface/cli/cli.py, run_cli()): boot happens inside
      system/orchestrate.py's run_session(ui, args), using AikoSimpleCLI's
      own step_loading/step_done/step_skip methods as the boot callbacks —
      there's no separate browser to keep responsive, so a single blocking
      boot before the prompt appears is the right tradeoff there.
Both paths converge on system/orchestrate.py:run_session(ui, args) for the
actual turn loop (main loop, commands, proactive idle check-ins, karaoke
typewriter, latency/debug accounting) — see that module for details.

Flow:

                                      parse_args()
                                          │
        ┌────────────────┼────────────────┼─────────────────┐
        ▼                ▼                ▼                 ▼
   --clear-mem       --logout          --cli           (default)
        │                │                │                 │
        ▼                ▼                ▼                 ▼
  AikoMemorize()    handle_logout()    run_cli(args)  run_webui(args)
     .clear()            │                │                 │
        │                ▼                ▼                 ▼
        ▼           SystemExit(0)  boot inside      AikoWeb(defer_servers=True)
   SystemExit(0)                   run_session(),    boot runs to completion,
                                    then turn loop    THEN server opens; post-auth
                                                      init via system/prepare.py

Front-end imports are deferred into main() rather than done at module load,
so that --clear-mem and --logout (which don't need FastAPI, uvicorn,
websockets, or any voice subsystem) stay fast and don't require those
dependencies to be installed at all. The heavy AikoMemorize memory stack is
likewise deferred into the --clear-mem branch only, so normal WebUI/CLI
launches never pay for it at import time.

Argument-order and env-var timing notes (why --debug/--trace/logging are
sequenced the way they are in main()) live as inline comments next to that
code, not here — see the code below.
"""

# Comment conventions:
#   untagged = permanent doc (traps, invariants, why)
#   NOTE:    = personal study note, safe to delete

from __future__ import annotations   # annotations become lazy strings — forward refs & newer syntax OK

import argparse                      # CLI argument parsing
import logging                       # logger for all [main] output
import os                            # env gate + hard-exit trap
import traceback                     # logging exit origins
from importlib.metadata import PackageNotFoundError, version
                                     # reads installed dist metadata — version comes from
                                     # pyproject.toml, never hardcoded (single source of truth)

_original_os_exit = os._exit         # capture BEFORE the patch — calling os._exit inside the wrapper would recurse into itself
                                     # NOTE: Python binds the function at assignment time — after
                                     # os._exit = wrapper, the name points at the wrapper, so the
                                     # capture must happen first

_FALLBACK_VERSION = "0.0.0+unknown"  # PEP 440 sentinel when metadata is missing; 0.0.0 sorts below any real release
                                     # NOTE: +unknown is a PEP 440 "local version" — valid, not semver
_CONFIRM_PHRASE = "Clear All Aiko's Memories."       # exact string the user type to arm the wipe


__all__ = ["parse_args", "main"]     # public surface: entry point + CLI parser; _ names are internal


def _resolve_version() -> str:
    """Return the installed package version, or a sentinel if metadata is missing."""
    # Single source of truth: pyproject.toml read via install metadata, so the
    # version never drifts between here and argparse. Re-run `pip install -e .`
    # after bumping, or this falls through to the sentinel below.
    try:
        return version("Aiko-chan")          # must match [project].name in pyproject.toml
    except PackageNotFoundError:             # bare checkout / metadata not installed
        return _FALLBACK_VERSION             # raised when the dist name isn't found — i.e. running without `pip install -e .`


def _install_os_exit_trap(log: logging.Logger, enabled: bool) -> None:
    """Monkeypatch os._exit to log the caller's stack before the hard exit (only when enabled)."""
    if not enabled:                          # trap is opt-in: off unless explicitly enabled
        return

    # os._exit() cannot be caught by try/except, so the only way to observe it
    # is to wrap it: log WHO called it, then perform the real exit.
    def _logged_os_exit(code: int | str | None) -> None:
        try:                                                     # Attempt to log the caller's stack
            log.error("[main] os._exit(%s) called from:\n%s",    # .format_stack() returns list of str, join() makes it one str
                      code, "".join(traceback.format_stack()))
        except Exception:                                        # Exception, not BaseException — a Ctrl+C
            pass                                                 # during logging still exits via finally
        finally:                                                 # finally runs on EVERY path — this is
            _original_os_exit(code)                              # what guarantees the real exit

    os._exit = _logged_os_exit     # patch applied; anything that bound os._exit before this bypasses logging
    # NOTE: Not idempotent — calling this twice double-wraps os._exit (harmless
    # but noisy: two stack logs, still one real exit). Currently called
    # once, unconditionally, in main(). If that ever changes, add a guard.


def _handle_clear_mem(log: logging.Logger) -> int:
    """Handle --clear-mem branch: two-step confirm, wipe all stored memories, exit.

    Two gates before the wipe:
        1. Yes/No prompt
        2. Type the exact confirmation phrase (_CONFIRM_PHRASE)

    Exit codes:
        0 — memories wiped, or aborted at either gate (intentionally
            indistinguishable so scripts don't treat a declined wipe as an error)
        1 — wipe failed (traceback in aiko.log)
    """
    # Gate 1: Yes/No. Abort on Ctrl-C / Ctrl-D. Non-tty stdin (piped/CI) hits
    # EOFError here and aborts safely — --clear-mem never wipes unattended
    # unless a human answered both gates.
    try:
        confirm = input("WARNING: This will PERMANENTLY erase all memories. Continue? [Yes/No]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):             # Ctrl-D raises EOFError, Ctrl-C raises KeyboardInterrupt — both mean "stop, don't wipe"
        print("\nAborted.")
        return 0
    if confirm not in ("y", "yes"):                   # anything except an explicit yes aborts;
                                                      # NOTE: default (empty input/Enter) is also an abort
        print("Aborted memory clear.")
        return 0

    # Gate 2: typed phrase. Guards against fat-finger 'y' on an irreversible
    # op, and against shell-history accidents re-running --clear-mem.
    try:
        typed = input(f'To confirm, type exactly: "{_CONFIRM_PHRASE}"\n> ').strip()
    except (EOFError, KeyboardInterrupt):             # same abort semantics as gate 1
        print("\nAborted.")
        return 0
    if typed != _CONFIRM_PHRASE:                      # exact match — case and punctuation must match;
                                                      # near-misses ('clear all aiko memories') are deliberately rejected
        print("Confirmation phrase did not match. Aborted memory clear.")
        return 0

    log.info("Clearing all memories...")
    print("Clearing all memories...")                 # LOG_CONSOLE is off by default on this path,
                                                      # so log.info alone would be silent in the terminal

    # Deferred heavy import — memory stack (embedding models, vector store)
    # is only paid for on this destructive branch; normal WebUI/CLI launches
    # never touch it. Key Orin win: no torch/vector-store RAM on normal boots.
    from cognition.memory.memorize import AikoMemorize

    try:
        mem = AikoMemorize()                          # may load embedding models — on an 8 GB Orin,
                                                      # check whether clear() needs models at all
                                                      # (storage-layer delete would skip that allocation)
        mem.clear()                                   # NOTE: assumes clear() is atomic or idempotent —
                                                      # if it isn't, a mid-wipe failure can leave
                                                      # partially-cleared storage behind.
        del mem                                       # drop refs so sqlite/faiss/file handles close
                                                      # on GC before exit
    except Exception:                                 # Exception, not BaseException — lets Ctrl+C through.
                                                      # Contain the failure HERE — this branch sits outside
                                                      # main()'s front-end try/except, so a re-raise would
                                                      # escape main() as a raw interpreter traceback.
        log.exception("[main] memory wipe (--clear-mem) failed")
        print("ERROR: memory wipe failed — see aiko.log for details.")
        return 1                                      # failure — distinguishable from 0 == aborted/success

    log.info("Memory cleared.")
    print("Memory cleared.")
    return 0                                          # success


def _handle_logout(log: logging.Logger) -> int:
    """Handle --logout branch: clear stored CLI auth token and exit.

    Exit codes: 0 = token cleared; 1 = handler missing (ImportError) or failed.
    """
    try:
        from interface.cli.cli import handle_logout   # deferred — CLI deps not needed for --clear-mem or default WebUI launch
    except ImportError as e:                          # raised when an import fails
        log.error("Could not load CLI logout handler (missing dependencies?): %s", e)
        return 1
    try:
        handle_logout()                               # same containment fix as --clear-mem — the old
                                                      # log-and-re-raise helper skipped this function's
                                                      # `return 0` and escaped main() raw
    except Exception:                                 # Exception, not BaseException — lets Ctrl+C through
        log.exception("[main] handle_logout() failed")
        return 1
    return 0                                          # success


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Values validated here in ONE place — front ends never re-check."""
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Aiko-chan — local assistant. Front ends: WebUI (default) or CLI (--cli); maintenance: --clear-mem, --logout.",
        epilog="Exit codes: 0 = success or user-declined · 1 = operation failed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,   # keep the epilog's line breaks as written
    )

    # ---- Front end selection -------------------------------------------------
    p.add_argument("--cli",  action="store_true",               # store_true = flag only, no value after it
                   help="terminal chat front end instead of the WebUI")
    p.add_argument("--name", metavar="NAME", default="",        # default "" (not None) — truthiness test below works either way,
                   help="companion name for this session (requires --cli)")   # and consumers get a str, never None
    p.add_argument("--text",   action="store_true",             # quiet-mode preset: ASR + TTS both off
                   help="keyboard input, TTS AND ASR both off (implies --no-asr); subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr", action="store_true",             # narrower preset: only ASR off, TTS stays on
                   help="keyboard input, TTS stays on, ASR off; ASR still loads for /listen")

    # ---- Version -------------------------------------------------------------
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {_resolve_version()}",    # evaluated HERE at parse time, not import — bare checkout
                   help="show installed version and exit")      # gets the sentinel instead of crashing on --help

    # ---- Maintenance (mutually exclusive — each owns the process) ------------
    maintenance = p.add_mutually_exclusive_group()              # argparse enforces: second flag on one cmdline → error+exit 2
    maintenance.add_argument("--clear-mem", action="store_true",
                             help="wipe ALL stored memories (two-gate confirm, then exit)")
    maintenance.add_argument("--logout",    action="store_true",
                             help="clear the stored session and exit")

    # ---- Debug / diagnostics ---------------------------------------------------
    p.add_argument("--debug", action="store_true",
                   help="verbose stderr logging (DEBUG level)")
    p.add_argument("--trace", action="store_true",              # CLI twin of TRACE_BRAIN=1 — main() maps this flag
                   help="per-turn brain tracer (TRACE_BRAIN=1) — what Aiko is thinking, without DEBUG-level log spam")   # onto the env var so the consumer reads only one source

    args = p.parse_args()

    # ---- Post-parse normalization & validation -------------------------------
    # --text is the "quiet mode" preset: it includes ASR-off, so consumers only
    # ever check one flag for ASR. --no-tts is deliberately omitted (never used).
    if args.text:                                               # normalization, not validation: make the implication
        args.no_asr = True                                      # real in the namespace so front ends check one flag

    if args.name and not args.cli:                              # single validation point — front ends never re-check (docstring contract)
        p.error("--name requires --cli")                        # p.error prints usage + msg, exits with code 2

    return args



def main() -> int:
    """Primary entry point for the Aiko-chan application."""
    # Parse args FIRST — --debug needs to set LOG_CONSOLE/LOG_LEVEL in the
    # environment before system.log's root logger is configured below.
    # (system.log resolves its config on the first get_logger() call, not
    # at import time, precisely so this ordering works.)
    args = parse_args()

    # Load config early, before any subsystem init (but after filters, before logging setup)
    from system.config import load_config
    load_config()

    if args.debug:                                      # explicit CLI flag overrides whatever .env set
        os.environ["LOG_CONSOLE"] = "1"
        os.environ["LOG_LEVEL"] = "DEBUG"
        # Background social-listening daemons (Threads/Bluesky/Mastodon) get
        # their urllib3/httpcore debug lines muted automatically; set
        # AIKO_DEBUG_FULL_HTTP=1 to see raw HTTP if you ever need to.

    # --trace enables the per-step brain tracer. Independent of --debug so
    # you can get a clean trace without the DEBUG-level log spam, or
    # combine both for the full picture.
    if args.trace:
        os.environ["TRACE_BRAIN"] = "1"

    # Set up logging and exit tracing
    from system.log import get_logger
    log = get_logger(__name__)
    # Installed before any deferred heavy imports (CLI/WebUI, voice, memory)
    # below, so os._exit is trapped for the whole process lifetime, not
    # just this module's own exit paths.
    _install_os_exit_trap(log, args.debug)

    if args.clear_mem:                                  # if clear memory argument set
        return _handle_clear_mem(log)

    if args.logout:                                     # if logout argument set
        return _handle_logout(log)

    try:                                                # one shared fatal-error trap for both front ends:
        if args.cli:                                    # SystemExit in the main thread exits SILENTLY (no traceback),
            from interface.cli.cli import run_cli       # so log WHO escaped before re-raising; BaseException catch-all
            run_cli(args)                               # covers KeyboardInterrupt and anything else unexpected.
        else:
            from interface.webui.webui import run_webui
            run_webui(args)
    except SystemExit as e:                             # silent-killer trap: SystemExit in the main thread
        if e.code not in (0, None):                     # non-zero exit = abnormal shutdown
            log.exception("[main] SystemExit(%r) escaped the session loop", e.code)
        else:                                           # exit(0) or exit() = clean shutdown
            log.info("[main] clean exit")
        raise                                           # preserve original exit behavior
    except KeyboardInterrupt:                           # graceful interrupt handling
        log.info("[main] KeyboardInterrupt")
        raise
    except Exception:                                   # any other fatal error (not BaseException to avoid catching asyncio cancels)
        log.exception("[main] fatal error escaped the session loop")  # full traceback to aiko.log
        raise                                           # re-raise after logging
    return 0


if __name__ == "__main__":
    raise SystemExit(main())                            # start the entry point
