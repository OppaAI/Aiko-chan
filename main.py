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
    # AIKO_MESSENGER_ADAPTERS is set, but this module never touches them.
    python main.py --debug       # verbose console logging (LOG_CONSOLE=1, LOG_LEVEL=DEBUG) + memory hits per turn; also implies --trace
    python main.py --trace       # brain trace per turn (AIKO_TRACE_BRAIN=1) without DEBUG-level log spam
    python main.py --clear-mem   # wipe all stored memories and exit
    python main.py --logout      # clear stored CLI (GitHub OAuth) auth token and exit
    python main.py --name <name> # set CLI display name (only when GitHub OAuth isn't configured)

This module only parses arguments and dispatches to the right front end:
    - interface/webui/webui.py  -> run_webui(args)   (default)
    - interface/cli/cli.py      -> run_cli(args)     (--cli)

main.py does NOT call AikoWakeup().boot() itself — each front end owns its
own boot timing, because the two have genuinely different requirements:
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
#   NTS:     = personal study note, safe to delete

from __future__ import annotations   # annotations become lazy strings — forward refs & newer syntax OK

# Standard library
import argparse                      # CLI argument parsing
import logging                       # logger for all [main] output
import os                            # env gate (AIKO_TRACE_EXIT) + hard-exit trap
import traceback                     # logging exit origins
from typing import Callable          # callable type annotation
from importlib.metadata import PackageNotFoundError, version
                                     # reads installed dist metadata — version comes from
                                     # pyproject.toml, never hardcoded (single source of truth)

_original_os_exit = os._exit         # capture BEFORE the patch — calling os._exit inside the
                                     # wrapper would recurse into itself
                                     # NTS: Python binds the function at assignment time — after
                                     # os._exit = wrapper, the name points at the wrapper, so the
                                     # capture must happen first

_FALLBACK_VERSION = "0.0.0+unknown"  # PEP 440 sentinel when metadata is missing; 0.0.0 sorts
                                     # below any real release
                                     # NTS: +unknown is a PEP 440 "local version" — valid, not semver

__all__ = ["parse_args", "main"]     # public surface: entry point + CLI parser; _ names are internal


def _install_os_exit_trap(log: logging.Logger) -> None:
    """Monkeypatch os._exit to log the caller's stack before the hard exit (only if AIKO_TRACE_EXIT=1)."""
    if os.environ.get("AIKO_TRACE_EXIT") != "1":                 # trap is opt-in: off unless explicitly enabled
        return

    # os._exit() cannot be caught by try/except, so the only way to observe it
    # is to wrap it: log WHO called it, then perform the real exit.
    def _logged_os_exit(code: int | str | None) -> None:
        try:
            log.error("[main] os._exit(%s) called from:\n%s",
                      code, "".join(traceback.format_stack()))
        except Exception:                                        # NTS: Exception, not BaseException — a Ctrl+C
            pass                                                 # during logging still exits via finally
        finally:                                                 # NTS: finally runs on EVERY path — this is
            _original_os_exit(code)                              # what guarantees the real exit

    os._exit = _logged_os_exit     # patch applied; anything that bound os._exit before this bypasses logging
    # Not idempotent — calling this twice double-wraps os._exit (harmless
    # but noisy: two stack logs, still one real exit). Currently called
    # once, unconditionally, in main(). If that ever changes, add a guard.


def _resolve_version() -> str:
    """Return the installed package version, or a sentinel if metadata is missing."""
    # Single source of truth: pyproject.toml read via install metadata, so the
    # version never drifts between here and argparse. Re-run `pip install -e .`
    # after bumping, or this falls through to the sentinel below.
    try:
        return version("Aiko-chan")          # must match [project].name in pyproject.toml
    except PackageNotFoundError:             # bare checkout / metadata not installed
        return _FALLBACK_VERSION             # NTS: raised when the dist name isn't found —
                                             # i.e. running without `pip install -e .`


def _run_with_error_logging(log: logging.Logger, label: str, fn: Callable[[], None]) -> None:
    """Run fn(); on exception, log traceback under `label`, then re-raise."""
    try:                                                     # NTS: try = run, expect possible failure
        fn()                                                 # NTS: zero-arg, per Callable[[], None]
    except Exception:                                        # NTS: Exception not BaseException — lets Ctrl+C through
        log.exception("[main] fatal error in %s", label)     # NTS: .exception auto-appends traceback; %s = lazy style
        raise                                                # bare raise = re-raise the ORIGINAL exception, traceback intact


def _handle_clear_mem(log) -> int:  # type: ignore[no-untyped-def]
    """Handle --clear-mem branch (extracted to reduce main() complexity C901)."""
    try:
        confirm = input("WARNING: This will permanently erase all memories. Continue? [y/N]: ").strip().lower()  # prompt for user confirm memory wiping
    except (EOFError, KeyboardInterrupt):           # Ctrl-D or Ctrl-C during prompt
        print("\nAborted.")                         # quiet abort message
        return 0                                 # exit code 0
    if confirm != "y":                              # anything other than explicit 'y' aborts
        print("Aborted memory clear.")              # user-facing message
        return 0                                 # exit code 0 (user chose to abort, not an error)
    log.info("Clearing all memories...")            # log success info
    from cognition.memory.memorize import AikoMemorize  # deferred — heavy memory stack, only needed for --clear-mem

    def do_wipe():
        """Initialize memory system and clear all stored memories."""
        mem = AikoMemorize()                        # load memory system
        mem.clear()                                 # wipe out memory

    _run_with_error_logging(log, "memory wipe (--clear-mem)", do_wipe)
    log.info("Memory cleared.")                     # log completion
    return 0                                     # exit code 0


def _handle_logout(log) -> int:  # type: ignore[no-untyped-def]
    """Handle --logout branch (extracted to reduce main() complexity)."""
    try:
        from interface.cli.cli import handle_logout  # load CLI logout handler (may fail if CLI deps missing)
    except ImportError as e:
        log.error("Could not load CLI logout handler (missing dependencies?): %s", e)
        return 1
    _run_with_error_logging(log, "handle_logout()", handle_logout)
    return 0                                     # exit code 0


def parse_args() -> argparse.Namespace:
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan")          # create argument object for declaring arguments
    p.add_argument("--text",      action="store_true",            # text (keyboard) input only
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",            # disable ASR
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",            # debug mode
               help="enable verbose console logging (sets LOG_CONSOLE=1, LOG_LEVEL=DEBUG). Also implies --trace (AIKO_TRACE_BRAIN=1) for backward compat.")
    p.add_argument("--trace",     action="store_true",            # trace Aiko's brain
                   help="enable the per-step brain tracer (AIKO_TRACE_BRAIN=1) without the DEBUG-level log spam. Use this when you only want to see what Aiko is thinking, not every internal HTTP call.")
    p.add_argument("--cli",       action="store_true",            # launch in CLI
                   help="use the plain no-curses CLI instead of the WebUI — for local testing only")
    g = p.add_mutually_exclusive_group()      # prevent conflicting exits (industrial: --clear-mem vs --logout)
    g.add_argument("--clear-mem", action="store_true",            # wipe out all memory and exit
                   help="WARNING: irreversibly wipes all stored memories, then exits")
    g.add_argument("--logout",   action="store_true",             # logout user session
                   help="clear stored CLI auth token and exit")
    p.add_argument("--name",     type=str, default="",            # for use in CLI mode without OAuth setup
                   help="set display name (CLI mode only, ignored with GitHub OAuth)")
    p.add_argument("--version", action="version", version=f"%(prog)s {_resolve_version()}")  # reads pyproject.toml via importlib.metadata
    args = p.parse_args()                                         # return namespace of the arguments
    if args.name and not args.cli:            # validate display name only meaningful in CLI (industrial: early fail)
        p.error("--name requires --cli")
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
    if args.trace or args.debug:                        # --debug still implies --trace for backward compat
        os.environ.setdefault("AIKO_TRACE_BRAIN", "1")

    # Set up logging and exit tracing
    from system.log import get_logger
    log = get_logger(__name__)
    # Installed before any deferred heavy imports (CLI/WebUI, voice, memory)
    # below, so os._exit is trapped for the whole process lifetime, not
    # just this module's own exit paths.
    _install_os_exit_trap(log)
    
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
    raise SystemExit(main())                                              # start the entry point
