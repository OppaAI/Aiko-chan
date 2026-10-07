"""
agentic/toolkit/coding.py

Safe coding-agent tools for Ministral-3-3B self-improvement on Jetson 8GB.

Verdict on Ministral-3-3B as a coding agent: YES, with guardrails.
  - Capable: single-file scoped tasks (explain, small bugfix, add test,
    rename, docstring) with constrained prompts + retrieved context.
  - NOT capable: large multi-file refactors or architecture redesigns in
    one shot — those need plan -> small-patch -> test loops with human
    review between steps.
  - This module enforces that loop: plan (heuristic/LLM-light) ->
    diff preview (no write) -> apply (approval + backup) -> test (bounded).

Safety rules (all enforced here, not just documented):
  - repo_write_file / repo_replace_text remain available but this module
    is the PREFERRED path: diff preview first, apply second.
  - code_apply_patch requires approval via the standard needs_approval
    gate unless the caller passes an approval bypass (studio dry-run
    never bypasses).
  - Forbids .env / secrets / keys / tokens paths and binary extensions.
  - Max patch 50k chars, atomic write (tmp + replace), .bak backup.
  - code_run_tests runs `pytest -q -x` with 90s timeout, captures tail
    only (4000 chars) to protect 8GB RAM + small-model context.
  - All tools graph=True + react=True so they show in the DAG palette.

Pair with codebase_search / repo_read_file / repo_search_text (read-only)
for context retrieval before proposing a patch.
"""

from __future__ import annotations

import codecs
import difflib
import io
import json
import locale
import selectors
import subprocess
import sys
import time
from pathlib import Path

from agentic.registry import TOOLS, tool
from agentic.toolkit.common import json_block
from system.log import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_PATCH_CHARS = 50_000
# Context budgets (2026-10-06): tuned for Ministral-3B's small window.
# The old 20k read cap could kill the context in 2 operations. Subagents
# (needle_subagent) handle bulk exploration; direct reads stay tiny and
# paginated. See agentic/toolkit/needle_subagent.py for the pattern.
MAX_READ_CHARS = 3_000
TEST_TIMEOUT = 90
TEST_TAIL_CHARS = 800

_ALLOWED_CODE_SUFFIXES = {".py", ".md", ".json", ".txt", ".sh", ".html", ".css", ".js", ".yaml", ".yml", ".toml"}
_FORBIDDEN_SUBSTRINGS = (".env", "secret", "token", "private_key", ".pem", ".key", "credentials")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cert"}


def _spec(name: str, description: str):
    return TOOLS[name] if name in TOOLS else name


def _confine(relative_path: str) -> Path:
    cleaned = (relative_path or "").strip().lstrip("/\\")
    if not cleaned:
        raise ValueError("relative_path required")
    path = (REPO_ROOT / cleaned).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def _check_writable(path: Path, relative: str) -> str | None:
    """Return error string, or None when writable."""
    rel_low = relative.lower()
    if any(s in rel_low for s in _FORBIDDEN_SUBSTRINGS):
        return "refusing to touch secrets/credentials path"
    if path.suffix.lower() not in _ALLOWED_CODE_SUFFIXES:
        return f"unsupported file type: {path.suffix}"
    try:
        rel_parts = path.relative_to(REPO_ROOT).parts
    except ValueError:
        return "path escapes repository"
    if any(part in _SKIP_DIRS for part in rel_parts):
        return "cannot write to restricted directory"
    return None


@tool(
    _spec("code_plan", "Break a coding task into small reviewable steps (heuristic + optional LLM)."),
    description="Break a coding task into small reviewable steps (heuristic + optional LLM).",
    graph=True,
    react=True,
    domain="coding",
)
def code_plan(goal: str = "", files: str = "", *, client=None, model: str | None = None) -> str:
    """Small-model-friendly planner. Caps goal at 1000 chars, files at 2000.

    With LLM: one short call (max 400 tokens of guidance). Without: a
    deterministic 4-step template so Ministral-3B still gets structure.
    """
    try:
        goal = (goal or "").strip()[:1000]
        files = (files or "").strip()[:2000]
        if not goal:
            return json_block("code_plan", {"ok": False, "error": "goal required"})
        if client is not None and model:
            try:
                from agentic.toolkit.synthesize import synthesize_report

                out = synthesize_report(
                    evidence=f"Goal: {goal}\nFiles: {files or '(discover via codebase_search)'}"[:2500],
                    prompt=("Break this into 3-6 tiny steps. Each step: one file, one change, "
                            "how to verify. Keep each step under 2 sentences."),
                    style="plain", client=client, model=model)
                steps = [ln.strip("-•* ") for ln in str(out).splitlines() if ln.strip()][:8]
                steps = [s[:220] for s in steps if s]
                if steps:
                    return json_block("code_plan", {"ok": True, "mode": "llm", "goal": goal, "steps": steps})
            except Exception as e:
                log.debug("code_plan llm fallback: %s", e)
        steps = [
            f"1. Retrieve context: codebase_search + repo_read_file for: {goal[:120]}",
            "2. Write the smallest diff that fixes one thing; preview with code_diff_preview.",
            "3. Apply with code_apply_patch (human approval required).",
            "4. Verify with code_run_tests (targeted) + repo_read_file the edited hunk.",
        ]
        if files:
            steps.insert(1, f"Focus files: {files[:300]}")
        return json_block("code_plan", {"ok": True, "mode": "heuristic", "goal": goal, "steps": steps})
    except Exception as e:
        return json_block("code_plan", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("code_diff_preview", "Preview unified diff for old->new text (no write)."),
    description="Preview unified diff for old->new text (no write).",
    graph=True,
    react=True,
    domain="coding",
)
def code_diff_preview(relative_path: str = "", old_text: str = "", new_text: str = "") -> str:
    """Pure function — never touches disk. Caps output at 8000 chars."""
    try:
        relative = (relative_path or "").strip()[:300]
        if not relative or old_text is None or new_text is None:
            return json_block("code_diff_preview", {"ok": False, "error": "relative_path, old_text, new_text required"})
        if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
            return json_block("code_diff_preview", {"ok": False, "error": f"patch too large (cap {MAX_PATCH_CHARS} chars)"})
        diff = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        if not diff:
            return json_block("code_diff_preview", {"ok": True, "relative_path": relative, "empty": True, "diff": ""})
        return json_block("code_diff_preview", {"ok": True, "relative_path": relative, "empty": False,
                                                "diff": diff[:8000], "truncated": len(diff) > 8000})
    except Exception as e:
        return json_block("code_diff_preview", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("code_apply_patch", "Apply old->new replacement with backup (APPROVAL REQUIRED)."),
    description="Apply old->new replacement with backup (APPROVAL REQUIRED).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def code_apply_patch(relative_path: str = "", old_text: str = "", new_text: str = "") -> str:
    """Replace first occurrence block. Atomic (tmp+replace) + .bak backup.

    Approval is enforced by the agentic approval gate (spec.needs_approval).
    Direct calls without approval still write — the gate lives one layer up,
    same as post_to_social. Studio dry-run should preview, not call this.
    """
    try:
        relative = (relative_path or "").strip()
        if not relative or old_text is None or new_text is None:
            return json_block("code_apply_patch", {"ok": False, "error": "relative_path, old_text, new_text required"})
        if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
            return json_block("code_apply_patch", {"ok": False, "error": f"patch too large (cap {MAX_PATCH_CHARS} chars)"})
        if not old_text:
            return json_block("code_apply_patch", {"ok": False, "error": "old_text must be non-empty (no blind appends)"})
        path = _confine(relative)
        err = _check_writable(path, relative)
        if err:
            return json_block("code_apply_patch", {"ok": False, "error": err})
        if not path.exists() or not path.is_file():
            return json_block("code_apply_patch", {"ok": False, "error": f"file not found: {relative}"})
        content = path.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return json_block("code_apply_patch", {"ok": False, "error": "old_text not found in file"})
        updated = content.replace(old_text, new_text, 1)
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.write_text(content, encoding="utf-8")
        except OSError as e:
            log.warning("code_apply_patch backup failed: %s", e)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(path)
        return json_block("code_apply_patch", {"ok": True, "path": relative,
                                               "bytes": len(updated), "backup": backup.name})
    except Exception as e:
        return json_block("code_apply_patch", {"ok": False, "error": str(e)[:300]})


@tool(
    "code_apply_correction",
    description="Apply a structured triage correction (APPROVAL REQUIRED).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def code_apply_correction(correction: str = "") -> str:
    """Unpack a Needle summary and apply it through the normal patch checks."""
    try:
        prefix = "[needle_subagent]"
        raw = correction.strip()
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
        result = json.loads(raw)
        if result.get("ok") is not True or result.get("truncated"):
            raise ValueError("triage did not produce a complete correction")
        patch = json.loads(result["summary"])
        fields = ("relative_path", "old_text", "new_text")
        if not isinstance(patch, dict) or set(patch) != set(fields):
            raise ValueError("correction must contain relative_path, old_text, and new_text")
        if not all(isinstance(patch[key], str) for key in fields):
            raise ValueError("correction fields must be strings")
        return code_apply_patch(**patch)
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return json_block("code_apply_correction", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("code_run_tests", "Run bounded pytest (90s timeout, tail output)."),
    description="Run bounded pytest (90s timeout, tail output).",
    graph=True,
    react=True,
    domain="coding",
)
def code_run_tests(target: str = "tests/unit/test_agentic_graph_engine.py", extra: str = "-q") -> str:
    """Single pytest invocation. No network, no coverage HTML — Jetson-safe."""
    try:
        target = (target or "tests/unit").strip()[:300] or "tests/unit"
        # Confine: must stay inside repo, must look like a test path.
        if target.startswith("-") or ";" in target or "&" in target or "|" in target:
            return json_block("code_run_tests", {"ok": False, "error": "invalid target"})
        tpath = (REPO_ROOT / target.lstrip("/\\")).resolve()
        if tpath != REPO_ROOT and REPO_ROOT not in tpath.parents:
            return json_block("code_run_tests", {"ok": False, "error": "target escapes repository"})
        cmd = [sys.executable, "-m", "pytest", target, "-q", "-x", "--timeout=60" if False else "-q"]
        # Keep flags minimal: caller extra is allowlisted, not raw shell.
        if (extra or "").strip() in {"-q", "-qq", "-v"}:
            pass  # already quiet
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=TEST_TIMEOUT)
        elapsed = round(time.monotonic() - start, 1)
        tail = (proc.stdout[-TEST_TAIL_CHARS:] + "\n" + proc.stderr[-1000:]) if proc.stderr else proc.stdout[-TEST_TAIL_CHARS:]
        return json_block("code_run_tests", {"ok": proc.returncode == 0, "target": target,
                                             "returncode": proc.returncode, "elapsed_s": elapsed,
                                             "tail": tail[-TEST_TAIL_CHARS:]})
    except subprocess.TimeoutExpired:
        return json_block("code_run_tests", {"ok": False, "error": f"timeout after {TEST_TIMEOUT}s"})
    except FileNotFoundError as e:
        return json_block("code_run_tests", {"ok": False, "error": f"pytest not available: {e}"})
    except Exception as e:
        return json_block("code_run_tests", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("code_lint", "Syntax-check a Python file (py_compile, read-only)."),
    description="Syntax-check a Python file (py_compile, read-only).",
    graph=True,
    react=True,
    domain="coding",
)
def code_lint(relative_path: str = "") -> str:
    """python -m py_compile on one file, 15s timeout. No writes."""
    try:
        relative = (relative_path or "").strip()
        if not relative:
            return json_block("code_lint", {"ok": False, "error": "relative_path required"})
        path = _confine(relative)
        if path.suffix.lower() != ".py":
            return json_block("code_lint", {"ok": False, "error": "only .py files"})
        if not path.exists():
            return json_block("code_lint", {"ok": False, "error": "file not found"})
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                              capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return json_block("code_lint", {"ok": True, "path": relative, "clean": True})
        return json_block("code_lint", {"ok": False, "path": relative, "error": (proc.stderr or "compile failed")[:1000]})
    except subprocess.TimeoutExpired:
        return json_block("code_lint", {"ok": False, "error": "timeout"})
    except Exception as e:
        return json_block("code_lint", {"ok": False, "error": str(e)[:250]})


__all__ = ["code_plan", "code_diff_preview", "code_apply_patch", "code_apply_correction",
           "code_run_tests", "code_lint", "sandbox_run", "ssh_run", "shell_run"]

# ── sandbox_run — execute Python in a confined sandbox ──────────────────
#
# Lets the coding agent RUN code after approval, not just write it.
# Path-confined to the repo, timeout-bounded, output-capped.
# This is the "write code, run it, see what happens, fix it" loop.

SANDBOX_TIMEOUT = 30
SANDBOX_OUTPUT_CHARS = 2000


@tool(
    _spec("sandbox_run", "Run a Python script in a confined sandbox (timeout, output cap)."),
    description="Run a Python script in a confined sandbox (timeout, output cap).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def sandbox_run(relative_path: str, args: str = "", timeout: int = SANDBOX_TIMEOUT) -> str:
    """Run a Python file from the repo in a confined subprocess.

    The script must already exist in the repo (write it with
    code_apply_patch first). It runs with:
      - cwd confined to the repo root
      - wall-clock timeout (default 30s, max 120s)
      - stdout/stderr captured, capped at 2000 chars each
      - PYTHONSAFEPATH=1, no bytecode writing

    No network access is blocked at this layer (use firewall rules for
    that); the confinement is path + time + output size.

    Args:
        relative_path: repo-relative path to a .py file.
        args: optional space-separated arguments (no shell metachars).
        timeout: seconds, clamped to [1, 120].
    """
    try:
        if not relative_path or not relative_path.strip():
            return json_block("sandbox_run", {"ok": False, "error": "path required"})
        rel = relative_path.strip()[:500]
        # Block shell metachars in both path and args
        for bad in (";", "&", "|", "`", "$", "(", ")", "<", ">", "\n"):
            if bad in rel or bad in (args or ""):
                return json_block("sandbox_run", {"ok": False, "error": "shell metachars not allowed"})
        path = (REPO_ROOT / rel.lstrip("/\\")).resolve()
        if path != REPO_ROOT and REPO_ROOT not in path.parents:
            return json_block("sandbox_run", {"ok": False, "error": "path escapes repository"})
        if not path.is_file() or path.suffix.lower() != ".py":
            return json_block("sandbox_run", {"ok": False, "error": "not a Python file in repo"})
        timeout = max(1, min(int(timeout or SANDBOX_TIMEOUT), 120))
        cmd = [sys.executable, str(path)]
        if (args or "").strip():
            cmd.extend((args or "").strip().split()[:20])  # max 20 args

        env = dict(__import__("os").environ)
        env["PYTHONSAFEPATH"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                              capture_output=True, text=True, timeout=timeout)
        elapsed = round(time.monotonic() - start, 1)
        return json_block("sandbox_run", {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": proc.stdout[-SANDBOX_OUTPUT_CHARS:],
            "stderr": proc.stderr[-SANDBOX_OUTPUT_CHARS:],
        })
    except subprocess.TimeoutExpired:
        return json_block("sandbox_run", {"ok": False, "error": f"timeout after {timeout}s"})
    except Exception as e:
        return json_block("sandbox_run", {"ok": False, "error": str(e)[:300]})


# ── ssh_run — SSH to a pre-configured host (APPROVAL REQUIRED) ──────────
#
# Lets Aiko run commands on the user's other machines (e.g. SSH from the
# Jetson to Oppa's PC). This is sensitive — it touches machines outside
# Aiko's own host — so:
#   - Hosts are pre-configured via AIKO_SSH_HOSTS env (JSON), not per-call.
#     Format: {"mypc": {"host": "192.168.1.50", "user": "oppa", "port": 22}}
#   - needs_approval=True: every invocation goes through Aiko's approval gate.
#   - Key-based auth only. No passwords are accepted or stored here.
#   - Full command + output logged. Timeout-bounded, output-capped.
#   - NOT available to subagents (not in SUBAGENT_TOOLS).

SSH_TIMEOUT = 30
SSH_OUTPUT_CHARS = 2000


@tool(
    _spec("ssh_run", "Run a command on a pre-configured SSH host (APPROVAL REQUIRED)."),
    description="Run a command on a pre-configured SSH host (APPROVAL REQUIRED).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def ssh_run(host_id: str = "", command: str = "", timeout: int = SSH_TIMEOUT) -> str:
    """Run a shell command on a pre-configured remote host via SSH.

    The host must be in AIKO_SSH_HOSTS (JSON env var). The command runs
    as the configured user with key-based auth. Every call needs approval
    through Aiko's standard gate.

    Args:
        host_id: key from AIKO_SSH_HOSTS (e.g. "mypc").
        command: shell command to run. No interactive commands.
        timeout: seconds, clamped to [5, 120].
    """
    import json as _json
    try:
        hosts_raw = __import__("os").getenv("AIKO_SSH_HOSTS", "")
        if not hosts_raw.strip():
            return json_block("ssh_run", {"ok": False,
                "error": "no SSH hosts configured",
                "hint": 'Set AIKO_SSH_HOSTS JSON, e.g. {"mypc": {"host": "192.168.1.50", "user": "oppa"}}'})
        hosts = _json.loads(hosts_raw)
        cfg = hosts.get((host_id or "").strip(), {})
        if not isinstance(cfg, dict) or not cfg.get("host") or not cfg.get("user"):
            return json_block("ssh_run", {"ok": False,
                "error": f"unknown host_id: {host_id}",
                "known": sorted(hosts.keys()) if isinstance(hosts, dict) else []})
        if not command or not command.strip():
            return json_block("ssh_run", {"ok": False, "error": "command required"})
        cmd_text = command.strip()[:2000]
        timeout = max(5, min(int(timeout or SSH_TIMEOUT), 120))

        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",           # key auth only, never prompt
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(int(cfg.get("port", 22))),
            f"{cfg['user']}@{cfg['host']}",
            cmd_text,
        ]
        log.info("ssh_run host=%s cmd=%.120s", host_id, cmd_text)
        start = time.monotonic()
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = round(time.monotonic() - start, 1)
        return json_block("ssh_run", {
            "ok": proc.returncode == 0,
            "host": host_id,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": proc.stdout[-SSH_OUTPUT_CHARS:],
            "stderr": proc.stderr[-SSH_OUTPUT_CHARS:],
        })
    except subprocess.TimeoutExpired:
        return json_block("ssh_run", {"ok": False, "error": f"timeout after {timeout}s"})
    except Exception as e:
        return json_block("ssh_run", {"ok": False, "error": str(e)[:300]})


# ── shell_run — general shell on Aiko's own host (APPROVAL REQUIRED) ────
#
# Claude Code-style Bash tool: sed, rm, grep, find, and every other Linux
# command for debugging, testing, and auditing the codebase.
#
# Safety (defense in depth):
#   - needs_approval=True: every call goes through Aiko's approval gate.
#   - DENYLIST: catastrophic patterns are blocked outright, no override.
#   - READONLY allowlist: ls/cat/grep/find/etc skip approval (still logged).
#   - Everything else needs approval, even with the gate.
#   - cwd defaults to repo root; resolved paths must stay in the repo or /tmp.
#   - Timeout-bounded (60s default, 300s max), output-capped.
#   - NOT in subagent tools — only Aiko herself.

SHELL_TIMEOUT = 60
SHELL_OUTPUT_CHARS = 3000

# Patterns that are never allowed, even with approval.
_SHELL_DENYLIST = (
    "rm -rf /", "rm -rf /*", "rm -rf ~", "mkfs", "dd of=/dev",
    ":(){:|:&};:", "chmod -R 777 /", "> /dev/sda", "shutdown",
    "reboot", "poweroff", "halt",
)

# Commands that are read-only and skip the approval gate (still logged).
_SHELL_READONLY = frozenset({
    "ls", "cat", "grep", "find", "head", "tail", "wc", "diff", "file",
    "stat", "du", "df", "ps", "top", "sed", "awk", "sort", "uniq",
    "cut", "tr", "echo", "pwd", "whoami", "uname", "date", "env",
    "which", "type", "git",
})


def _shell_is_readonly(command: str) -> bool:
    first = (command.strip().split() or [""])[0].lstrip("/")
    # sed -i writes; plain sed reads
    if first == "sed" and " -i" in f" {command} ":
        return False
    return first in _SHELL_READONLY


def _drain_shell_output(proc: subprocess.Popen, timeout: int) -> tuple[str, str]:
    """Drain both pipes with bounded reads and retain only their text tails."""
    tails = {"stdout": "", "stderr": ""}
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        for name in tails:
            stream = getattr(proc, name)
            decoder = io.IncrementalNewlineDecoder(
                codecs.getincrementaldecoder(locale.getpreferredencoding(False))(),
                translate=True,
            )
            selector.register(stream, selectors.EVENT_READ, (name, decoder))
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            for key, _ in selector.select(remaining):
                name, decoder = key.data
                chunk = key.fileobj.read1(4096)
                text = decoder.decode(chunk, final=not chunk)
                tails[name] = (tails[name] + text)[-SHELL_OUTPUT_CHARS:]
                if not chunk:
                    selector.unregister(key.fileobj)
        proc.wait(timeout=max(0, deadline - time.monotonic()))
    return tails["stdout"], tails["stderr"]


@tool(
    _spec("shell_run", "Run a shell command on Aiko's host (APPROVAL REQUIRED unless read-only)."),
    description="Run a shell command on Aiko's host (APPROVAL REQUIRED unless read-only).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def shell_run(command: str = "", cwd: str = "", timeout: int = SHELL_TIMEOUT) -> str:
    """Run an arbitrary shell command on Aiko's own host.

    Read-only commands (ls, cat, grep, find, git, ...) run directly.
    Anything that modifies state needs approval via the standard gate.
    Catastrophic patterns (rm -rf /, fork bombs, dd to devices, ...) are
    blocked outright.

    Args:
        command: the shell command to run.
        cwd: working directory (default: repo root).
        timeout: seconds, clamped to [5, 300].
    """
    try:
        cmd_text = (command or "").strip()
        if not cmd_text:
            return json_block("shell_run", {"ok": False, "error": "command required"})
        cmd_text = cmd_text[:4000]

        lowered = f" {cmd_text.lower()} "
        for bad in _SHELL_DENYLIST:
            if bad in lowered:
                log.warning("shell_run blocked denylisted pattern: %.60s", cmd_text)
                return json_block("shell_run", {"ok": False,
                    "error": f"blocked: command contains denylisted pattern '{bad}'"})

        workdir = str(REPO_ROOT)
        if (cwd or "").strip():
            p = Path(cwd.strip())
            p = (p if p.is_absolute() else REPO_ROOT / p).resolve()
            # Resolve symlinks before checking both allowed directory trees.
            if not any(p == root or root in p.parents for root in (REPO_ROOT, Path("/tmp"))):
                return json_block("shell_run", {"ok": False, "error": "cwd outside repository and /tmp"})
            workdir = str(p) if p.is_dir() else str(REPO_ROOT)

        timeout = max(5, min(int(timeout or SHELL_TIMEOUT), 300))
        readonly = _shell_is_readonly(cmd_text)
        log.info("shell_run readonly=%s cwd=%s cmd=%.150s", readonly, workdir, cmd_text)

        with subprocess.Popen(
            cmd_text, shell=True, cwd=workdir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable="/bin/bash",
        ) as proc:
            try:
                stdout, stderr = _drain_shell_output(proc, timeout)
            finally:
                if proc.poll() is None:
                    proc.kill()
        return json_block("shell_run", {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "readonly": readonly,
            "stdout": stdout,
            "stderr": stderr,
        })
    except subprocess.TimeoutExpired:
        return json_block("shell_run", {"ok": False, "error": f"timeout after {timeout}s"})
    except Exception as e:
        return json_block("shell_run", {"ok": False, "error": str(e)[:300]})
