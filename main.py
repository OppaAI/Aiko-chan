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
    python main.py --debug       # verbose console logging (LOG_CONSOLE=1, LOG_LEVEL=DEBUG), full firehose
    python main.py --no-console    # silence console logging even with --debug (file log only)
    python main.py --trace       # brain trace per turn (TRACE_BRAIN=1) without DEBUG-level log spam
    python main.py --clear-mem   # wipe learned state (memories, knowledge, experience) and exit
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
                 load_config() + _apply_debug_trace_env()
                 (LOG_CONSOLE/LOG_LEVEL/TRACE_BRAIN set from flags)
                                │
            ┌───────────────────┼───────────────────┼───────────────────┐
            ▼                   ▼                   ▼                   ▼
       --clear-mem           --logout             --cli             (default)
            │                   │                   │                   │
            ▼                   ▼                   ▼                   ▼
     AikoMemorize()       _handle_logout()      run_cli(args)     run_webui(args)
        .clear()            returns 0/1         then turn loop     boot to completion,
            │                                                     THEN server opens
            ▼
       returns 0/1 (becomes the process exit code via SystemExit)

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
                                     # (importlib.metadata is deferred into _resolve_version —
                                     # its dist scan must not run on every boot, only for --version)

_original_os_exit = os._exit         # capture BEFORE the patch — calling os._exit inside the wrapper would recurse into itself
                                     # NOTE: Python binds the function at assignment time — after
                                     # os._exit = wrapper, the name points at the wrapper, so the
                                     # capture must happen first

_FALLBACK_VERSION = "0.0.0+unknown"  # PEP 440 sentinel when metadata is missing; 0.0.0 sorts below any real release
                                      # NOTE: +unknown is a PEP 440 "local version" — valid, not semver
_CONFIRM_PHRASE = "Clear All Aiko's Memories."       # exact string the user types to arm the wipe
_FACTORY_RESET_PHRASE_TMPL = "Factory Reset {uid}."  # formatted with the resolved user id — reset wipes the whole user dir


__all__ = ["parse_args", "main", "_apply_debug_trace_env", "_console_enabled"]     # public surface: entry point + CLI parser; _ names are internal


def _resolve_version() -> str:
    """Return the installed package version, or a sentinel if metadata is missing."""
    # Single source of truth: pyproject.toml read via install metadata, so the
    # version never drifts between here and argparse. Re-run `pip install -e .`
    # after bumping, or this falls through to the sentinel below.
    # Import is deferred (not module-top-level) so the dist-metadata disk scan
    # runs only when --version is actually passed, never on normal boots.
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("Aiko-chan")          # must match [project].name in pyproject.toml
    except PackageNotFoundError:             # bare checkout / metadata not installed
        return _FALLBACK_VERSION             # raised when the dist name isn't found — i.e. running without `pip install -e .`


class _VersionAction(argparse.Action):
    """Print the version and exit — resolves metadata lazily (see _resolve_version)."""
    def __call__(self, parser, namespace, values, option_string=None):
        parser.exit(message=f"{parser.prog} {_resolve_version()}\n")


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
    # but noisy). Currently called once, gated on --debug, in main().


def _console_enabled() -> bool:
    """True when log records already reach the terminal (else print()s fill in)."""
    return os.environ.get("LOG_CONSOLE") == "1"


def _clear_dream_scratch(log: logging.Logger) -> None:
    """Delete deep-study scratch DBs for the active user (transient work files)."""
    from system.userspace import user_state_path
    dream_dir = user_state_path("dream")
    if not dream_dir.is_dir():
        return
    for child in dream_dir.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
        except OSError as e:                            # keep wiping the rest; report at the end via log
            log.warning("[main] could not remove dream scratch %s: %s", child, e)


def _handle_clear_mem(log: logging.Logger) -> int:
    """Handle --clear-mem branch: two-step confirm, wipe learned state, exit.

    Scope (learned state only): episodic memories, learned knowledge,
    agentic experience, and deep-study scratch. Deliberately kept: the
    codebase index (rebuildable cache), profile/identity, skills, workspace,
    mail sessions, and gamification — use a factory reset for those.

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
    if not _console_enabled():                          # log.info alone is silent in the terminal unless
        print("Clearing all memories...")               # console logging is on — print() fills that gap only

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
                                                        # (no explicit close: AikoMemorize owns no documented
                                                        # shutdown hook, so release is left to the interpreter
                                                        # on exit immediately below)
        from cognition.knowledge.schema import delete_all as delete_knowledge
        from agentic.experience.schema import delete_all as delete_experience
        knowledge_counts = delete_knowledge()         # learned docs/chunks (codebase index cache kept)
        experience_counts = delete_experience()       # agentic task outcomes
        _clear_dream_scratch(log)                     # deep-study scratch DBs
        log.info("[main] cleared knowledge=%s experience=%s",
                 knowledge_counts, experience_counts)
    except Exception:                                 # Exception, not BaseException — lets Ctrl+C through.
                                                      # Contain the failure HERE — this branch sits outside
                                                      # main()'s front-end try/except, so a re-raise would
                                                      # escape main() as a raw interpreter traceback.
        log.exception("[main] memory wipe (--clear-mem) failed")
        if not _console_enabled():
            print("ERROR: memory wipe failed — see aiko.log for details.")
        return 1                                      # failure — distinguishable from 0 == aborted/success

    log.info("Memory cleared.")
    if not _console_enabled():
        print("Memory cleared.")
    return 0                                          # success


def _handle_backup(log: logging.Logger, args) -> int:
    """Handle --backup branch: snapshot, verify, transport, exit.

    Non-destructive, so no confirmation gates — but fail-closed: any
    snapshot/verify/transport error returns 1 with the cause in aiko.log.
    Exit codes: 0 = verified backup written; 1 = failed.
    """
    from system.backup import BackupError, run_backup

    dests = [d for d in str(getattr(args, "backup_dest", "usb") or "usb").split(",")]
    try:
        manifest = run_backup(
            getattr(args, "backup_type", "settings") or "settings",
            dests,
            user_id=getattr(args, "user", "") or "",
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    except BackupError as e:
        log.error("[main] backup failed: %s", e)
        if not _console_enabled():
            print(f"ERROR: backup failed — {e}")
        return 1
    except Exception:
        log.exception("[main] backup failed unexpectedly")
        if not _console_enabled():
            print("ERROR: backup failed — see aiko.log for details.")
        return 1
    summary = (f"Backup {manifest.backup_type} for {manifest.user_id}: "
               f"{len(manifest.files)} files, {manifest.total_bytes // 1024} KiB, "
               f"verified={manifest.verified}")
    log.info("[main] %s dests=%s", summary, manifest.dests)
    if not _console_enabled():
        print(summary)
        for line in manifest.dests:
            print(f"  {line}")
    return 0


def _handle_factory_reset(log: logging.Logger, args) -> int:
    """Handle --factory-reset branch: guard on fresh verified backup, two gates, wipe, exit.

    The manifest guard lives in system.backup.perform_factory_reset (fails
    closed without a verified backup <24h old). The human gates mirror
    --clear-mem: explicit Yes plus a typed phrase naming the user, so shell
    history can never re-run a reset unattended.
    Exit codes: 0 = wiped or aborted at a gate; 1 = guard failed or wipe failed.
    """
    from system.backup import BackupError, perform_factory_reset, resolve_user_id

    try:
        uid = resolve_user_id(getattr(args, "user", "") or "")
    except BackupError as e:
        log.error("[main] factory reset refused: %s", e)
        if not _console_enabled():
            print(f"ERROR: {e}")
        return 1
    phrase = _FACTORY_RESET_PHRASE_TMPL.format(uid=uid)
    try:
        confirm = input(f"WARNING: This will PERMANENTLY erase ALL state for '{uid}'. Continue? [Yes/No]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 0
    if confirm not in ("y", "yes"):
        print("Aborted factory reset.")
        return 0
    try:
        typed = input(f'To confirm, type exactly: "{phrase}"\n> ').strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 0
    if typed != phrase:
        print("Confirmation phrase did not match. Aborted factory reset.")
        return 0
    try:
        report = perform_factory_reset(uid)
    except BackupError as e:
        log.error("[main] factory reset refused: %s", e)
        if not _console_enabled():
            print(f"ERROR: {e}")
        return 1
    except Exception:
        log.exception("[main] factory reset failed unexpectedly")
        if not _console_enabled():
            print("ERROR: factory reset failed — see aiko.log for details.")
        return 1
    log.info("[main] factory reset for %s: %s", uid, report)
    if not _console_enabled():
        print(f"Factory reset complete for '{uid}' ({report['removed_files']} files, "
              f"backed by {report['backup_id']} @ {report['backup_utc']}).")
    return 0


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
        description="Aiko-chan — local assistant. Front ends: WebUI (default) or CLI (--cli); maintenance: --clear-mem, --logout, --backup, --factory-reset.",
        epilog=("Exit codes: 0 = success or user-declined · 1 = operation failed.\n"
                "Diagnostics — level and destination are independent:\n"
                "  --trace                 clean brain signal (no DEBUG spam)\n"
                "  --console               INFO and above on the terminal\n"
                "  --debug                 full firehose (DEBUG everywhere)\n"
                "  --debug --no-console    full firehose, file only"),
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
    p.add_argument("--version", action=_VersionAction, nargs=0,
                   help="show installed version and exit")

    # ---- Maintenance (mutually exclusive — each owns the process) ------------
    maintenance = p.add_mutually_exclusive_group()              # argparse enforces: second flag on one cmdline → error+exit 2
    maintenance.add_argument("--clear-mem", action="store_true",
                             help="wipe learned state: memories, knowledge, experience, dream scratch (two-gate confirm, then exit)")
    maintenance.add_argument("--logout",    action="store_true",
                             help="clear the stored CLI auth token and exit")
    maintenance.add_argument("--backup",    action="store_true",
                             help="snapshot user state (--backup-type settings|full) to --backup-dest, verify, then exit")
    maintenance.add_argument("--factory-reset", action="store_true",
                             help="wipe the whole <USER_SPACE_ROOT>/<uid> dir AFTER a verified fresh backup exists (two-gate confirm, then exit)")
    p.add_argument("--backup-type", choices=("settings", "full"), default="settings",
                   help="with --backup: 'settings' snapshots <USER_SPACE_ROOT>/<uid>/ only (~130M, hourly-safe); 'full' adds the codebase minus models/.venv/build/logs/.git (daily)")
    p.add_argument("--backup-dest", default="usb",
                   help="with --backup: comma-separated dests usb|microsd|nas|cloud|pc|dir:<path> (default: usb). microsd takes code only, never DBs.")
    p.add_argument("--user", default="",
                   help="with --backup/--factory-reset: user id (default: owner autodetect, else AIKO_USER_ID)")
    p.add_argument("--dry-run", action="store_true",
                   help="with --backup: plan + snapshot + verify, skip transport, delete staging")

    # ---- Debug / diagnostics ---------------------------------------------------
    p.add_argument("--debug", action="store_true",
                   help="verbose stderr logging (DEBUG level)")
    p.add_argument("--console", action=argparse.BooleanOptionalAction, default=None,   # None = follow the --debug rule below;
                   help="force console logging on (--console) or off (--no-console); default: on with --debug, off otherwise)")
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


def _apply_debug_trace_env(args: argparse.Namespace) -> None:
    """Map --debug/--trace onto their owned env vars (single write source).

    Extracted so the flag→env wiring is unit-testable without running main().
    Direct assignment (not setdefault) so the flag beats any exported shell
    var. Must run before any module that snapshots these vars at import time
    (system.log reads LOG_CONSOLE/LOG_LEVEL, system.brain_trace reads
    TRACE_BRAIN) is imported.
    """
    if args.debug:                                      # --debug: verbose stderr logging. LOG_CONSOLE/LOG_LEVEL
        os.environ["LOG_CONSOLE"] = "1"                 # are OWNED by these flags — never set them in yaml/.env,
        os.environ["LOG_LEVEL"] = "DEBUG"               # main.py is the single write source.
                                                        # --debug is the full firehose (nothing muted, including
                                                        # per-request HTTP chatter); use --trace for clean signal.

    # --console/--no-console explicitly forces console logging either way and
    # beats both --debug and any exported LOG_CONSOLE. Unset (None) keeps the
    # rule above: console follows --debug. "0" (not unset) so an explicit
    # --no-console beats a shell-exported LOG_CONSOLE=1 too.
    if args.console is True:
        os.environ["LOG_CONSOLE"] = "1"
    elif args.console is False:
        os.environ["LOG_CONSOLE"] = "0"

    # --trace enables the per-step brain tracer. Independent of --debug so
    # you can get a clean trace without the DEBUG-level log spam, or
    # combine both for the full picture.
    if args.trace:                                      # --trace: per-turn brain tracer, independent of --debug
        os.environ["TRACE_BRAIN"] = "1"                 # (clean trace without DEBUG spam). Same ownership rule:
                                                         # flag is the only write source; beats any shell export.



def _flush_fly_plasticity(log) -> None:
    """Best-effort write of pending fly MB/CX state on process exit."""
    try:
        from cognition.fly_registry import flush_everything
        result = flush_everything()
        if result:
            log.debug("[main] fly plasticity flushed: %s", result)
    except Exception as exc:
        try:
            log.debug("[main] fly plasticity flush skipped: %s", exc)
        except Exception:
            pass


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

    _apply_debug_trace_env(args)

    # Set up logging and exit tracing
    from system.log import get_logger
    log = get_logger(__name__)
    # Installed before any deferred heavy imports below, so os._exit is trapped
    # for the whole process lifetime. Gated on --debug: the exit-stack dump is
    # heavy-debug territory; a --trace run keeps the console clean.
    _install_os_exit_trap(log, args.debug)

    if args.clear_mem:                                  # if clear memory argument set
        return _handle_clear_mem(log)

    if args.logout:                                     # if logout argument set
        return _handle_logout(log)

    if getattr(args, "backup", False):
        return _handle_backup(log, args)

    if getattr(args, "factory_reset", False):
        return _handle_factory_reset(log, args)

    try:                                                # one shared fatal-error trap for both front ends:
        if args.cli:                                    # SystemExit in the main thread exits SILENTLY (no traceback),
            from interface.cli.cli import run_cli       # so log WHO escaped before re-raising; the Exception handler below
            run_cli(args)                               # covers ordinary fatals — KeyboardInterrupt has its own clause,
        else:                                           # and asyncio cancellations ride BaseException and pass through unlogged.
            from interface.webui.webui import run_webui
            run_webui(args)
    except SystemExit as e:                             # silent-killer trap: SystemExit in the main thread
        if e.code not in (0, None):                     # non-zero exit = abnormal shutdown
            log.exception("[main] SystemExit(%r) escaped the session loop", e.code)
        else:                                           # exit(0) or exit() = clean shutdown
            log.info("[main] clean exit")
        _flush_fly_plasticity(log)
        raise                                           # preserve original exit behavior
    except KeyboardInterrupt:                           # graceful interrupt handling
        _flush_fly_plasticity(log)
        log.info("[main] KeyboardInterrupt")
        raise
    except Exception:                                   # any other fatal error (not BaseException to avoid catching asyncio cancels)
        _flush_fly_plasticity(log)
        log.exception("[main] fatal error escaped the session loop")  # full traceback to aiko.log
        raise                                           # re-raise after logging
    _flush_fly_plasticity(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())                            # start the entry point
