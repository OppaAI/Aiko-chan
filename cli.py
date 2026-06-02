"""
cli.py

Aiko-chan CLI — full-screen curses TUI, cyberpunk edition.
Usage:
    python cli.py               # full voice — ASR (faster-whisper) + TTS (Kokoro)
    python cli.py --text        # keyboard input + no TTS
    python cli.py --debug       # show memory debug info each turn
    python cli.py --clear-mem   # wipe all stored memories and exit

Layout:
    ╔══════════════════════════════════════════════════════════╗
    ║                      BANNER                              ║
    ╠══════════════════╦═══════════════════════════════════════╣
    ║  ASCII ART       ║  INIT LOG  /  ARCH INFO               ║
    ║  (62 wide)       ╠═══════════════════════════════════════╣
    ║                  ║  CHAT MESSAGES                        ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  VITALS BAR                           ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  INPUT                                ║
    ╚══════════════════╩═══════════════════════════════════════╝

Identity data loaded from persona/identity.md.
Hardware info read live from /proc — no hardcoded values.
Vitals bar refreshes every second: RAM, DB size, tokens, tok/s, uptime, mode.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import json
import platform
import subprocess
import curses
import logging
import textwrap
import threading
import time
import uuid
logging.disable(logging.WARNING)

from core.silence import silent_stderr
from dotenv import load_dotenv
import argparse
load_dotenv()

with silent_stderr():
    from core.memorize import AikoMemorize
    from core.speak    import AikoSpeak
    from core.think    import AikoThink

# ── system info (read live at startup) ───────────────────────────────────────

def _read_sys_info() -> dict:
    """
    Sample the host environment at startup, reading raw hardware and OS signals
    from the kernel's exposed interfaces.

    Returns a dict containing:
        cpu          — model name string from /proc/cpuinfo or platform fallback
        ram_total_kb — total physical memory in kilobytes
        ram          — human-readable RAM string (e.g. '7.4 GB')
        storage      — root partition size from df
        os           — JetPack revision string, or PRETTY_NAME from /etc/os-release
    """
    info = {}

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key = line.split(":")[0].strip().lower()
                if key in ("model name", "hardware", "model"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if not info.get("cpu"):
        info["cpu"] = platform.processor() or platform.machine() or "unknown"

    # Total RAM (stored as bytes for vitals calculations)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_total_kb"] = int(line.split()[1])
                    break
    except Exception:
        info["ram_total_kb"] = 0

    ram_kb = info.get("ram_total_kb", 0)
    ram_gb = ram_kb / 1024 / 1024
    info["ram"] = f"{ram_gb:.1f} GB" if ram_gb >= 1 else f"{ram_kb // 1024} MB"

    # Root partition size
    try:
        out = subprocess.check_output(["df", "-h", "/"], text=True).splitlines()[1]
        info["storage"] = out.split()[1]
    except Exception:
        info["storage"] = "unknown"

    # OS / runtime — JetPack first, then /etc/os-release
    try:
        with open("/etc/nv_tegra_release") as f:
            raw = f.readline().strip()
            m   = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', raw)
            info["os"] = f"JetPack R{m.group(1)}.{m.group(2)}" if m else raw[:40]
    except FileNotFoundError:
        try:
            with open("/etc/os-release") as f:
                d = {}
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        d[k] = v.strip('"')
            info["os"] = d.get("PRETTY_NAME", platform.version()[:40])
        except Exception:
            info["os"] = platform.version()[:40] or "unknown"

    return info

_SYS = _read_sys_info()

# ── live vitals (called every tick) ──────────────────────────────────────────

def _ram_used_str() -> str:
    """
    Read current memory pressure from the kernel and return a live usage string.

    Parses MemTotal and MemAvailable from /proc/meminfo each call so the
    vitals bar always reflects the system's actual state.

    Returns a string of the form 'X.X/Y.Y GB', or '? GB' on read failure.
    """
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable")):
                    k, v = line.split(":")
                    vals[k.strip()] = int(v.split()[0])  # KB
                if len(vals) == 2:
                    break
        total = vals.get("MemTotal",     0)
        avail = vals.get("MemAvailable", 0)
        used  = (total - avail) / 1024 / 1024   # GB
        total_gb = total / 1024 / 1024
        return f"{used:.1f}/{total_gb:.1f} GB"  # :.1f instead of :.0f
    except Exception:
        return "? GB"

def _db_size_str() -> str:
    """
    Probe the Qdrant memory store and return the number of living engrams.

    Queries the local Qdrant REST API for the aiko_memory collection's
    points_count, providing a real-time measure of long-term memory depth.

    Returns a string of the form 'N entries', or '? mem' if Qdrant is
    unreachable or the response is malformed.
    """
    try:
        import urllib.request, json
        url = "http://localhost:6333/collections/aiko_memory"
        with urllib.request.urlopen(url, timeout=1) as r:
            data = json.loads(r.read())
        points = data["result"]["points_count"]
        return f"{points} entries"
    except Exception:
        return "? mem"

def _fmt_uptime(seconds: float) -> str:
    """
    Format a raw elapsed-seconds value into a human-readable session age string.

    Args:
        seconds: Elapsed time in seconds since the session awakened.

    Returns:
        A string of the form 'HH:MM:SS'.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ── identity ──────────────────────────────────────────────────────────────────

def _load_identity(path: str = "persona/identity.md") -> dict:
    """
    Awaken Aiko's sense of self by parsing the identity manifest from disk.

    Reads persona/identity.md and extracts three identity primitives:
        - Banner lines: the styled header text rendered across the top of the TUI.
        - ASCII art: the character portrait drawn in the left column, delimited
          by ---ART--- / ---END--- markers.
        - Art color map: a JSON array of per-row palette index lists that drive
          the 256-colour rendering of the ASCII art.

    Art dimensions (art_w, art_h) are derived from the parsed content so the
    layout adapts automatically if the portrait is updated.

    Args:
        path: Path to the identity markdown file. Defaults to
              'persona/identity.md'.

    Returns:
        A dict with keys: banner_lines, art_lines, art_colors, art_w, art_h.
        All values fall back to safe empty defaults if the file is absent or
        a section is malformed.
    """
    result = {
        "banner_lines": [],
        "art_lines":    [],
        "art_colors":   [],
        "art_w":        46,
        "art_h":        44,
    }
    try:
        lines = open(path, encoding="utf-8").readlines()
        text  = "".join(lines)
    except FileNotFoundError:
        return result

    banner_m = re.search(r'## Banner Lines\s*```[^\n]*\n(.*?)```', text, re.DOTALL)
    if banner_m:
        result["banner_lines"] = banner_m.group(1).splitlines()
        while result["banner_lines"] and not result["banner_lines"][-1].strip():
            result["banner_lines"].pop()

    in_art    = False
    art_rows: list[str] = []
    for raw_line in lines:
        stripped = raw_line.rstrip('\n')
        if stripped.strip() == '---ART---':
            in_art = True;  continue
        if stripped.strip() == '---END---':
            in_art = False; continue
        if in_art:
            art_rows.append(stripped)
    if art_rows:
        result["art_lines"] = art_rows
        result["art_h"]     = len(art_rows)
        result["art_w"]     = max((len(l) for l in art_rows), default=46)

    cmap_m = re.search(r'### Art Color Map\s*```json\s*\n(.*?)```', text, re.DOTALL)
    if cmap_m:
        try:
            result["art_colors"] = json.loads(cmap_m.group(1))
        except json.JSONDecodeError:
            pass

    return result


_IDENTITY    = _load_identity()
BANNER_LINES = _IDENTITY["banner_lines"]
BANNER_H     = len(BANNER_LINES)

ANIME_ART_LINES  = _IDENTITY["art_lines"]
ANIME_ART_COLORS = _IDENTITY["art_colors"]
ART_W = _IDENTITY["art_w"]
ART_H = _IDENTITY["art_h"]

# ── session / env ─────────────────────────────────────────────────────────────

SESSION_ID    = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "unknown")
KOKORO_VOICE  = os.getenv("KOKORO_VOICE",  "unknown")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3.5")
SEARXNG_URL   = os.getenv("SEARXNG_URL",   "localhost:8080")

AI_NAME = os.getenv("AI_NAME", "Aiko")
USER_ID = os.getenv("USER_ID", "")

LEFT_W = ART_W + 2

# ── architecture sections ─────────────────────────────────────────────────────

ARCH_SECTIONS = [
    ("MEMORY SYSTEMS", [
        ("Long-term store", "mem0  →  Qdrant vector DB"),
        ("Embedding model", "BGE-base-en-v1.5  (768d)"),
        ("Short-term ctx",  f"Rolling {os.getenv('CONTEXT_WINDOW_TURNS', '20')}-turn window"),
    ]),
    ("COGNITION", [
        ("Inference engine", "Ollama  (local, offline)"),
        ("Active model",     OLLAMA_MODEL),
        ("Web search",       SEARXNG_URL),
    ]),
    ("VOICE ENGINE", [
        ("TTS backend",  "Realtime TTS + Kokoro"),
        ("Voice profile", KOKORO_VOICE),
        ("ASR model",    WHISPER_MODEL),
    ]),
    ("HARDWARE", [
        ("CPU",          _SYS.get("cpu",     "unknown")),
        ("RAM",          _SYS.get("ram",     "unknown")),
        ("Storage",      _SYS.get("storage", "unknown")),
        ("OS / Runtime", _SYS.get("os",      "unknown")),
        ("Session ID",   SESSION_ID),
    ]),
]

ARCH_ROWS = sum(1 + len(items) for _, items in ARCH_SECTIONS) + 2

# ── init steps ────────────────────────────────────────────────────────────────

INIT_STEPS = {
    'think_start':  ('Inference Engine', f'Spawning Ollama worker  ·  {OLLAMA_MODEL}'),
    'think_warmup': ('Model Warm-up',    'Loading weights, running prefill pass …'),
    'mem_qdrant':   ('Vector Database',  'Connecting to Qdrant  ·  localhost:6333'),
    'mem_embed':    ('Embedding Model',  'Loading BGE-base-en-v1.5  ·  768-dim vectors'),
    'mem_cleanup':  ('Memory Lifecycle', 'Pruning decayed memories …'),
    'mem_ready':    ('Memory Cortex',    'mem0 ready  ·  long-term recall online'),
    'speak_kokoro': ('TTS Engine',       f'Initializing Kokoro  ·  {KOKORO_VOICE}'),
    'speak_ready':  ('Voice Output',     'Audio pipeline ready  ·  24 kHz'),
    'speak_skip':   ('Voice Output',     'TTS disabled  (--text mode)'),
    'listen_ready': ('Speech Input',     f'faster-whisper ready  ·  {WHISPER_MODEL}'),
    'listen_skip':  ('Speech Input',     'ASR disabled  (--text mode)'),
}

SPINNER = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

# ── colour pairs ──────────────────────────────────────────────────────────────

CP_PINK    = 1
CP_CYAN    = 2
CP_PURPLE  = 3
CP_MAUVE   = 4
CP_DIM     = 5
CP_WHITE   = 6
CP_SBARBG  = 7
CP_INPUTBG = 8
CP_ART_BASE= 8

ART_PALETTE = [None, 236, 239, 243, 246, 249, 253, 95, 138, 181]

def init_colours() -> None:
    """
    Awaken the terminal's chromatic pathways by registering all colour pairs
    used across the TUI's visual cortex.

    Initialises curses colour mode with transparent backgrounds (-1) for all
    standard UI pairs, plus the art palette block starting at CP_ART_BASE.
    Must be called once inside the curses wrapper before any drawing begins.
    """
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_PINK,    198,  -1)
    curses.init_pair(CP_CYAN,     51,  -1)
    curses.init_pair(CP_PURPLE,  135,  -1)
    curses.init_pair(CP_MAUVE,   177,  -1)
    curses.init_pair(CP_DIM,     245,  -1)
    curses.init_pair(CP_WHITE,    15,  -1)
    curses.init_pair(CP_SBARBG,   16, 198)
    curses.init_pair(CP_INPUTBG,  51,  -1)
    for idx, color256 in enumerate(ART_PALETTE):
        if color256 is not None:
            curses.init_pair(CP_ART_BASE + idx, color256, -1)

def _art_attr(palette_idx: int) -> int:
    """
    Resolve an art palette index to a curses attribute ready for rendering.

    Palette indices 5, 6, and 9 correspond to the brightest tones in the
    portrait and receive A_BOLD to punch them forward on dim terminals.
    Index 0 is the transparency sentinel; it returns -1 to signal the caller
    to render a space with no attribute.

    Args:
        palette_idx: Index into ART_PALETTE (0–9).

    Returns:
        A curses attribute int, or -1 for transparent cells.
    """
    if palette_idx == 0:
        return -1
    attr = curses.color_pair(CP_ART_BASE + palette_idx)
    if palette_idx in (5, 6, 9):
        attr |= curses.A_BOLD
    return attr


# ─────────────────────────────────────────────────────────────────────────────
# AikoTUI
# ─────────────────────────────────────────────────────────────────────────────

class AikoTUI:
    """
    The visual cortex of Aiko-chan's terminal presence.

    Manages the full-screen curses layout, rendering all UI regions — banner,
    left-column portrait, right-column init log / arch panel, chat area, vitals
    bar, and input line — and exposes a thread-safe API for the cognitive
    subsystems to inject messages, stream tokens, and update session metrics.
    """

    INPUT_PROMPT = "  ❯  "

    def __init__(self, stdscr, no_voice: bool = False, debug: bool = False):
        """
        Initialise the TUI's sensory and display apparatus.

        Sets up the curses screen, internal state for messages, streaming
        buffer, init log, and session vitals. Colour pairs are registered
        here before any drawing occurs.

        Args:
            stdscr:   The curses window object provided by curses.wrapper.
            no_voice: When True, ASR and TTS are both dormant (--text mode).
            debug:    When True, memory retrieval hits are surfaced each turn.
        """
        self._scr      = stdscr
        self._no_voice = no_voice
        self._debug    = debug
        self._lock     = threading.Lock()
        self._ts       = time.time()

        self._messages: list[tuple[str, str]] = []
        self._scroll    = 0
        self._streaming = ''

        self._init_log: list[tuple[str, str, str]] = []
        self._frame     = 0
        self._phase     = 'init'
        self._input_buf: list[str] = []

        # vitals — updated externally by _run
        self._stats: dict = {
            'tokens':     0,      # session total tokens received
            'turn_tok':   0,      # tokens this turn
            'turn_start': None,   # time.time() when current turn began
            'tok_s':      0.0,    # tok/s of last completed turn
            'asr_on':     not no_voice,
            'tts_on':     not no_voice,
        }

        init_colours()
        curses.curs_set(0)
        self._scr.nodelay(False)

    # ── layout ────────────────────────────────────────────────────────────────

    @property
    def _pt(self):
        """Row index where the left and right columns begin (below the banner border)."""
        return BANNER_H + 2

    @property
    def _arch_sep_row(self):
        """Row index of the horizontal separator between the arch/init panel and chat area."""
        return self._pt + ARCH_ROWS

    @property
    def _chat_top(self):
        """First row of the chat message region."""
        return self._arch_sep_row + 1

    def _dims(self):
        """Return the current terminal dimensions as (height, width)."""
        return self._scr.getmaxyx()

    def _chat_bot(self, h):
        """
        Return the last row of the chat message region given terminal height h.

        Leaves room for the vitals bar, input separator, input line, and
        bottom border below the chat area.
        """
        return h - 6

    # ── low-level write ───────────────────────────────────────────────────────

    def _wr(self, y, x, text, attr=0):
        """
        Safely emit a string to the terminal at the given coordinates.

        Clips text to the available column width and uses insstr on the final
        cell of the last row to avoid the curses bottom-right corner exception.
        Silently swallows all curses errors so a resize mid-draw never crashes
        the session.

        Args:
            y:    Row coordinate (0-indexed).
            x:    Column coordinate (0-indexed).
            text: String to write; automatically clipped to available width.
            attr: Curses attribute (colour pair + bold flags). Defaults to 0.
        """
        h, w = self._dims()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x
        if avail <= 0:
            return
        try:
            if y == h - 1 and x + len(text) >= w:
                self._scr.addstr(y, x, text[:w - x - 1], attr)
                try:
                    self._scr.insstr(y, w - 1, text[w - x - 1:w - x], attr)
                except curses.error:
                    pass
            else:
                self._scr.addstr(y, x, text[:avail], attr)
        except curses.error:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # BANNER
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_banner(self, h, w):
        """
        Render the top banner region: the decorative header, centred identity
        lines, live clock, and version tag.

        The banner occupies rows 0 through BANNER_H + 1. The bottom row draws
        the T-junction border that separates the banner from the two-column
        body below.

        Args:
            h: Terminal height.
            w: Terminal width.
        """
        pk = curses.color_pair(CP_PINK)  | curses.A_BOLD
        wh = curses.color_pair(CP_WHITE)
        cy = curses.color_pair(CP_CYAN)  | curses.A_BOLD

        self._wr(0, 0, '╔' + '═' * (w - 2) + '╗', pk)

        for i, line in enumerate(BANNER_LINES):
            row = 1 + i
            pad = max(1, (w - len(line)) // 2)
            self._wr(row, 1,   ' ' * (w - 2), 0)
            self._wr(row, 0,   '║', pk)
            self._wr(row, w-1, '║', pk)
            self._wr(row, pad, line, pk)

        clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
        self._wr(1, w - 1 - len(clock_str), clock_str, wh)

        ver_str = f" {AI_NAME} v0.1.5 "
        self._wr(2, w - 1 - len(ver_str), ver_str, cy)

        self._wr(BANNER_H + 1, 0,
            '╠' + '═' * (LEFT_W - 1) + '╦' + '═' * (w - LEFT_W - 2) + '╣', pk)

    def _draw_clock_only(self):
        """
        Refresh only the clock and cursor position without redrawing the full frame.

        Called by the background tick thread during input so the clock stays
        alive while the rest of the layout remains static, avoiding expensive
        full redraws on every second tick.
        """
        with self._lock:
            h, w = self._dims()
            wh = curses.color_pair(CP_WHITE) | curses.A_BOLD
            clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
            self._wr(1, w - 1 - len(clock_str), clock_str, wh)
            rx    = LEFT_W + 1
            inp_r = self._chat_bot(h) + 4
            content = self.INPUT_PROMPT + ''.join(self._input_buf)
            cx = min(rx + len(content), w - 2)
            try:    self._scr.move(inp_r, cx)
            except: pass
            self._scr.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # LEFT COLUMN
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_left_col(self, h, w):
        """
        Paint the left column with Aiko's ASCII art portrait using the
        per-character colour map loaded from identity.md.

        Each cell is resolved through the art palette; transparent cells
        (palette index 0) and spaces are written as plain spaces to preserve
        the terminal's background colour. Rows below the art height are
        padded with spaces to clear any stale content.

        Args:
            h: Terminal height.
            w: Terminal width.
        """
        pk  = curses.color_pair(CP_PINK) | curses.A_BOLD
        pt  = self._pt
        bot = h - 2

        for row in range(pt, bot + 1):
            art_idx = row - pt
            self._wr(row, 0, '║', pk)

            if art_idx < ART_H and art_idx < len(ANIME_ART_LINES):
                art_line = ANIME_ART_LINES[art_idx]
                clr_row  = ANIME_ART_COLORS[art_idx] if art_idx < len(ANIME_ART_COLORS) else []
                x = 1
                for col_idx, ch in enumerate(art_line):
                    if x >= LEFT_W:
                        break
                    pidx = clr_row[col_idx] if col_idx < len(clr_row) else 0
                    attr = _art_attr(pidx)
                    if attr < 0 or ch == ' ':
                        try:    self._scr.addch(row, x, ' ')
                        except: pass
                    else:
                        try:    self._scr.addstr(row, x, ch, attr)
                        except: pass
                    x += 1
                while x < LEFT_W:
                    try:    self._scr.addch(row, x, ' ')
                    except: pass
                    x += 1
            else:
                try:    self._scr.addstr(row, 1, ' ' * (LEFT_W - 1))
                except: pass

        self._wr(h - 1, 0, '╚', pk)

    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT COLUMN
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_right_col(self, h, w, buf):
        """
        Render the entire right column: vertical borders, upper info panel
        (init log or arch), chat area, vitals bar, and input line.

        Delegates sub-region drawing to dedicated methods and stitches the
        separators and bottom border around them. Cursor position is set to
        the end of the current input buffer after each render.

        Args:
            h:   Terminal height.
            w:   Terminal width.
            buf: Current input buffer (list of chars) for the input line.
        """
        pk        = curses.color_pair(CP_PINK)   | curses.A_BOLD
        sbar_attr = curses.color_pair(CP_SBARBG) | curses.A_BOLD
        inp_attr  = curses.color_pair(CP_INPUTBG)

        pt  = self._pt
        sep = self._arch_sep_row
        cb  = self._chat_bot(h)
        rx  = LEFT_W + 1
        rw  = w - LEFT_W - 2

        for row in range(pt, h - 1):
            self._wr(row, LEFT_W, '║', pk)
            self._wr(row, w - 1,  '║', pk)

        for row in range(pt, sep):
            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass

        if self._phase == 'init':
            self._draw_right_init(pt, sep, rx, rw)
        else:
            self._draw_right_arch(pt, sep, rx, rw)

        self._wr(sep, LEFT_W, '╠' + '═' * (w - LEFT_W - 2) + '╣', pk)

        self._draw_chat_area(sep + 1, cb, rx, rw, w)

        # chat / vitals separator
        sbar_sep = cb + 1
        self._wr(sbar_sep, LEFT_W, '╠', pk)
        self._wr(sbar_sep, w - 1,  '╣', pk)
        try:    self._scr.addstr(sbar_sep, LEFT_W + 1, '═' * rw,
                                 curses.color_pair(CP_PINK) | curses.A_BOLD)
        except: pass

        # vitals bar
        self._draw_vitals(sbar_sep + 1, rx, rw, sbar_attr)

        # vitals / input separator
        ir = sbar_sep + 2
        self._wr(ir, LEFT_W, '╠', pk)
        self._wr(ir, w - 1,  '╣', pk)
        try:    self._scr.addstr(ir, LEFT_W + 1, '═' * rw,
                                 curses.color_pair(CP_PINK) | curses.A_BOLD)
        except: pass

        # input line
        inp_r   = ir + 1
        content = self.INPUT_PROMPT + ''.join(buf)
        vis_start = max(0, len(content) - rw + 1)
        line = content[vis_start:].ljust(rw)
        try:    self._scr.addstr(inp_r, rx, line, inp_attr)
        except: pass

        # bottom border
        bot_r = inp_r + 1
        if bot_r < h:
            self._wr(bot_r, LEFT_W, '╩', pk)
            self._wr(bot_r, w - 1,  '╝', pk)
            try:    self._scr.addstr(bot_r, LEFT_W + 1, '═' * rw,
                                     curses.color_pair(CP_PINK) | curses.A_BOLD)
            except: pass
        self._wr(bot_r, 0, '╚' + '═' * (LEFT_W - 1), pk)

        cx = min(rx + len(content), w - 2)
        try:    self._scr.move(inp_r, cx)
        except: pass

    # ── vitals bar ────────────────────────────────────────────────────────────

    def _draw_vitals(self, row: int, rx: int, rw: int, attr):
        """
        Render the live session vitals as a single status line.

        Segments are built left-to-right and silently dropped if the terminal
        is too narrow, so the bar degrades gracefully rather than wrapping.
        Displayed metrics: input/output mode flags, RAM usage, Qdrant entry
        count, cumulative session tokens, tok/s of the last completed turn,
        and session uptime.

        Format (at full width):
            ☊ ASR  │  RAM 5.6/7.4 GB  │  DB 142 entries  │  847 tok  │  23.4 t/s  │  ↑ 0:12:34

        Args:
            row:  Row to render the vitals bar on.
            rx:   Left column start (right panel interior).
            rw:   Available width for the bar content.
            attr: Curses attribute to apply (typically the status-bar colour pair).
        """
        s = self._stats

        mode_parts = []
        mode_parts.append("☊ ASR" if s['asr_on'] else "⌨  TXT")
        mode_parts.append("🎙 TTS" if s['tts_on'] else "🔇 TTS")
        mode_str = "  ".join(mode_parts)

        ram_str   = f"RAM {_ram_used_str()}"
        db_str    = f"DB {_db_size_str()}"
        tok_str   = f"{s['tokens']:,} tokens"
        toks_str  = f"{s['tok_s']:.1f} t/s" if s['tok_s'] > 0 else "— t/s"
        up_str    = f"↑ {_fmt_uptime(time.time() - self._ts)}"

        segments = [mode_str, ram_str, db_str, tok_str, toks_str, up_str]

        # Build bar right-to-left, dropping segments if they don't fit
        sep   = "  │  "
        built = ""
        for seg in segments:
            candidate = (built + sep + seg) if built else ("  " + seg)
            if len(candidate) + 2 <= rw:
                built = candidate
            # if it doesn't fit, skip — don't truncate mid-segment

        bar = built.ljust(rw - 1)
        try:    self._scr.addstr(row, rx, bar[:rw - 1], attr)
        except: pass

    # ── init log ──────────────────────────────────────────────────────────────

    def _draw_right_init(self, pt, pb, rx, rw):
        """
        Render the boot sequence log in the upper right panel during the init phase.

        Each step in _init_log is displayed with a spinner, check, dash, or
        cross glyph depending on its state. When all steps have settled to a
        terminal state (done / skip / error), an 'ALL SYSTEMS ONLINE' banner
        is appended below the log.

        Args:
            pt: Top row of the panel region.
            pb: Bottom row of the panel region (exclusive).
            rx: Left edge column of the right panel interior.
            rw: Available width of the right panel interior.
        """
        pk  = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        dim = curses.color_pair(CP_DIM)
        wh  = curses.color_pair(CP_WHITE)

        self._wr(pt,     rx, " INITIALIZING NEURAL SYSTEMS", pk)
        self._wr(pt + 1, rx, ' ─' * min(14, (rw - 1) // 2), pk)

        row = pt + 2
        for key, state, detail in self._init_log:
            if row >= pb:
                break
            lbl, dflt = INIT_STEPS.get(key, (key, detail))
            txt = detail if detail else dflt
            if state == 'loading':
                sp = SPINNER[self._frame]
                self._wr(row, rx,      f"  {sp} ", wh)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'done':
                self._wr(row, rx,      "  ✓ ", cy)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'skip':
                self._wr(row, rx,      "  – ", dim)
                self._wr(row, rx + 4,  f"{lbl:<20}", dim)
                self._wr(row, rx + 24, txt[:rw - 25], dim)
            elif state == 'error':
                self._wr(row, rx,      "  ✗ ", wh)
                self._wr(row, rx + 4,  f"{lbl:<20}", wh)
                self._wr(row, rx + 24, txt[:rw - 25], curses.color_pair(CP_MAUVE))
            row += 1

        all_fin = (len(self._init_log) > 0 and
                   all(s in ('done', 'skip', 'error') for _, s, _ in self._init_log))
        if all_fin and row < pb - 1:
            self._wr(row + 1, rx, "  [ ALL SYSTEMS ONLINE ]",
                     curses.color_pair(CP_CYAN) | curses.A_BOLD)

    # ── arch panel ────────────────────────────────────────────────────────────

    def _draw_right_arch(self, pt, pb, rx, rw):
        """
        Render the static neural architecture summary in the upper right panel.

        Displays the ARCH_SECTIONS table — memory systems, cognition,
        voice engine, and hardware — once the init phase has completed and
        the TUI has transitioned to chat phase.

        Args:
            pt: Top row of the panel region.
            pb: Bottom row of the panel region (exclusive).
            rx: Left edge column of the right panel interior.
            rw: Available width of the right panel interior.
        """
        pk = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy = curses.color_pair(CP_CYAN)  | curses.A_BOLD
        mv = curses.color_pair(CP_MAUVE)
        wh = curses.color_pair(CP_WHITE)

        self._wr(pt,     rx, " NEURAL ARCHITECTURE", pk)
        self._wr(pt + 1, rx, ' ─' * min(10, (rw - 1) // 2), pk)

        row = pt + 2
        for section, items in ARCH_SECTIONS:
            if row >= pb - 1:
                break
            self._wr(row, rx, f"  {section}", cy)
            row += 1
            for name, val in items:
                if row >= pb:
                    break
                self._wr(row, rx,      f"    {name:<18}", wh)
                self._wr(row, rx + 22, val[:rw - 23],    mv)
                row += 1

    # ── chat area ─────────────────────────────────────────────────────────────

    def _render_lines(self, rw):
        """
        Flatten the message log and active streaming buffer into a list of
        typed display lines suitable for the chat viewport.

        Each message is word-wrapped to fit the available column width.
        Continuation lines are indented to align with the text after the
        sender prefix. A blank separator line is appended after each AI
        response. System messages receive a ◈ prefix glyph.

        The current streaming fragment, if non-empty, is appended as a
        live AI line that updates on each token arrival.

        Args:
            rw: Available column width for the chat region.

        Returns:
            A list of (kind, line) tuples where kind is one of:
                'Y' — user message line
                'A' — AI message line
                'S' — system / separator line
        """
        avail = rw - 2
        out   = []
        for sender, text in self._messages:
            if sender == 'you':
                pre = f" {USER_ID}: "
                ind = " " * len(pre)
                wrp = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('Y', pre + wrp[0]))
                for l in wrp[1:]: out.append(('Y', ind + l))
            elif sender == 'aiko':
                pre = f" {AI_NAME}: "
                ind = " " * len(pre)
                wrp = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('A', pre + wrp[0]))
                for l in wrp[1:]: out.append(('A', ind + l))
                out.append(('S', ''))
            elif sender == 'sys':
                out.append(('S', f"  ◈  {text}"))
        if self._streaming:
            pre = f" {AI_NAME}: "
            ind = " " * len(pre)
            wrp = textwrap.wrap(self._streaming, avail - len(pre)) or [""]
            out.append(('A', pre + wrp[0]))
            for l in wrp[1:]: out.append(('A', ind + l))
        return out

    def _draw_chat_area(self, ct, cb, rx, rw, w):
        """
        Render the visible slice of the chat history into the chat viewport.

        Applies scroll offset so the most recent content is always shown by
        default, and draws a scroll hint glyph in the top-right corner when
        the user has scrolled up. Sender prefixes are bolded relative to
        continuation lines to maintain visual hierarchy.

        Args:
            ct: Top row of the chat region (inclusive).
            cb: Bottom row of the chat region (exclusive).
            rx: Left edge column of the right panel interior.
            rw: Available width of the right panel interior.
            w:  Full terminal width (used for right border placement).
        """
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        mv  = curses.color_pair(CP_MAUVE)
        dim = curses.color_pair(CP_DIM)
        pu  = curses.color_pair(CP_PURPLE)

        ch       = cb - ct
        rendered = self._render_lines(rw)
        total    = len(rendered)
        self._scroll = max(0, min(self._scroll, max(0, total - ch)))
        start    = max(0, total - ch - self._scroll)
        visible  = rendered[start:start + ch]

        for i, (kind, line) in enumerate(visible):
            row = ct + i
            if row >= cb:
                break
            if kind == 'Y':
                prefix    = f" {USER_ID}: "
                bold_attr = cy | curses.A_BOLD
                text_attr = cy
            elif kind == 'A':
                prefix    = f" {AI_NAME}: "
                bold_attr = mv | curses.A_BOLD
                text_attr = mv
            else:
                prefix    = ""
                bold_attr = dim
                text_attr = dim

            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass

            if prefix and line.startswith(prefix):
                self._wr(row, rx, prefix, bold_attr)
                self._wr(row, rx + len(prefix),
                         line[len(prefix):][:rw - len(prefix) - 1], text_attr)
            else:
                self._wr(row, rx, line[:rw - 1], text_attr)
            self._wr(row, w - 1, '║', pk)

        for row in range(ct + len(visible), cb):
            try:    self._scr.addstr(row, rx, ' ' * (rw - 1))
            except: pass
            self._wr(row, w - 1, '║', pk)

        if self._scroll > 0:
            hint = f" ↑{self._scroll}  PgDn↓ "
            self._wr(ct, w - len(hint) - 2, hint, pu)

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER DRAW
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self, buf=None):
        """
        Execute a full repaint of the entire TUI surface under the draw lock.

        Checks minimum terminal dimensions before drawing; renders a plain
        error string if the terminal is too small rather than attempting a
        partial draw that would corrupt the layout. All sub-region draw calls
        are made from here in layout order: banner → left column → right column.

        Args:
            buf: Input buffer (list of chars) to pass to the right-column
                 renderer for the input line. Defaults to an empty list.
        """
        with self._lock:
            h, w = self._dims()
            min_h = BANNER_H + 2 + ARCH_ROWS + 8
            min_w = LEFT_W + 40
            if h < min_h or w < min_w:
                self._scr.clear()
                self._wr(0, 0, f"Terminal too small: {w}x{h}  (need {min_w}x{min_h})", 0)
                self._scr.refresh()
                return
            if buf is None:
                buf = []
            self._draw_banner(h, w)
            self._draw_left_col(h, w)
            self._draw_right_col(h, w, buf)
            self._scr.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    # INIT API
    # ─────────────────────────────────────────────────────────────────────────

    def _upsert_step(self, key, state, detail=''):
        """
        Insert or update a boot step entry in the init log under the draw lock.

        If a step with the given key already exists it is updated in-place,
        preserving display order. New steps are appended.

        Args:
            key:    Step identifier string matching an INIT_STEPS key.
            state:  One of 'loading', 'done', 'skip', or 'error'.
            detail: Optional override text shown next to the step label.
        """
        with self._lock:
            for i, (k, s, d) in enumerate(self._init_log):
                if k == key:
                    self._init_log[i] = (key, state, detail)
                    return
            self._init_log.append((key, state, detail))

    def step_loading(self, key, detail=''):
        """Mark a boot step as actively loading (spinner state)."""
        self._upsert_step(key, 'loading', detail)

    def step_done(self, key, detail=''):
        """Mark a boot step as successfully completed (✓ state)."""
        self._upsert_step(key, 'done',    detail)

    def step_skip(self, key, detail=''):
        """Mark a boot step as intentionally skipped (– state)."""
        self._upsert_step(key, 'skip',    detail)

    def step_error(self, key, detail=''):
        """Mark a boot step as failed (✗ state)."""
        self._upsert_step(key, 'error',   detail)

    def status_finish(self):
        """
        Transition the TUI from the init phase to the active chat phase.

        Switches the upper right panel from the boot log to the static
        architecture summary. Should be called once all init threads have
        joined and the cognitive subsystems are fully online.
        """
        with self._lock:
            self._phase = 'chat'

    # ─────────────────────────────────────────────────────────────────────────
    # CHAT API
    # ─────────────────────────────────────────────────────────────────────────

    def add_message(self, sender, text):
        """
        Commit a completed message to the conversation log.

        Resets the scroll position to the bottom so the new message is
        immediately visible. Thread-safe.

        Args:
            sender: One of 'you', 'aiko', or 'sys'.
            text:   The message content string.
        """
        with self._lock:
            self._messages.append((sender, text))
            self._scroll = 0

    def stream_token(self, token):
        """
        Ingest an incoming token into the live streaming buffer.

        Accumulates the token into _streaming for display before the turn
        completes, and updates session and turn token counters. Records the
        turn start timestamp on the first token of a new turn.

        Args:
            token: A string fragment emitted by the inference engine.
        """
        with self._lock:
            self._streaming += token
            count = len(token)
            self._stats['tokens']   += count
            self._stats['turn_tok'] += count
            if self._stats['turn_start'] is None:
                self._stats['turn_start'] = time.time()

    def stream_commit(self):
        """
        Finalise the active streaming turn and commit the buffered response.

        Moves the accumulated _streaming content into the permanent message
        log, freezes the tok/s metric for the completed turn, and resets
        turn-scoped counters. No-ops gracefully if the buffer is empty.
        Thread-safe.
        """
        with self._lock:
            if self._streaming:
                self._messages.append(('aiko', self._streaming))
                self._streaming = ''
                self._scroll    = 0
            # freeze tok/s for this turn
            if self._stats['turn_start'] is not None:
                elapsed = time.time() - self._stats['turn_start']
                self._stats['tok_s'] = (
                    self._stats['turn_tok'] / elapsed if elapsed > 0 else 0.0)
            self._stats['turn_tok']   = 0
            self._stats['turn_start'] = None

    def turn_start(self):
        """
        Signal the beginning of a new cognitive turn.

        Resets turn-scoped token counters and records the wall-clock start
        time so tok/s can be computed accurately when stream_commit is called.
        Should be called immediately before streaming begins for each new turn.
        """
        with self._lock:
            self._stats['turn_start'] = time.time()
            self._stats['turn_tok']   = 0

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def get_input(self):
        """
        Enter a blocking text input loop, collecting keystrokes until the user
        submits with Enter or interrupts with Ctrl-C / Ctrl-D.

        A background tick thread fires every second to keep the clock alive
        without blocking key reads. Supports backspace, PgUp/PgDn chat
        scrolling, and printable character accumulation. The cursor is shown
        during input and hidden on exit.

        Returns:
            The stripped input string, or an empty string if the user submitted
            without typing.

        Raises:
            KeyboardInterrupt: On Ctrl-C or Ctrl-D, after stopping the tick
                thread and restoring the cursor.
        """
        buf = []
        self._input_buf = buf
        curses.curs_set(1)
        self._scr.nodelay(True)

        stop_tick = threading.Event()
        def _ticker():
            while not stop_tick.is_set():
                self._draw_clock_only()
                stop_tick.wait(1.0)
        tick_t = threading.Thread(target=_ticker, daemon=True)
        tick_t.start()

        try:
            while True:
                try:
                    ch = self._scr.get_wch()
                except curses.error:
                    time.sleep(0.05)
                    continue

                h, w = self._dims()

                if ch in ('\n', '\r', curses.KEY_ENTER):
                    break
                elif ch in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                    if buf: buf.pop()
                elif ch == curses.KEY_PPAGE:
                    with self._lock:
                        rw       = w - LEFT_W - 2
                        rendered = self._render_lines(rw)
                        ch_h     = self._chat_bot(h) - self._chat_top
                        self._scroll = min(
                            self._scroll + max(1, ch_h - 2),
                            max(0, len(rendered) - ch_h))
                elif ch == curses.KEY_NPAGE:
                    with self._lock:
                        ch_h = self._chat_bot(h) - self._chat_top
                        self._scroll = max(0, self._scroll - max(1, ch_h - 2))
                elif ch in ('\x03', '\x04'):
                    curses.curs_set(0)
                    stop_tick.set()
                    tick_t.join()
                    raise KeyboardInterrupt
                elif isinstance(ch, str) and ch.isprintable():
                    buf.append(ch)

                self._draw(buf=buf)
        finally:
            stop_tick.set()
            tick_t.join()
            curses.curs_set(0)

        return ''.join(buf).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # VOICE INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def get_voice_input(self, listen, wait_fn=None):
        """
        Capture a voice utterance via the ASR pipeline and return the
        transcribed text.

        Runs the ASR listener on a daemon thread so the clock tick can
        continue updating via _draw_clock_only. Status tokens emitted by the
        listener ('__LISTENING__', '__TRANSCRIBING__', '__IDLE__') are
        translated into human-readable input-bar labels. Blocks until the
        listener thread signals completion.

        Args:
            listen:  An AikoListen instance with a .listen() method.
            wait_fn: Optional callable passed to listen.listen() so the ASR
                     layer can wait for TTS to finish before listening.

        Returns:
            The transcribed utterance string, or an empty string on failure.
        """
        self._input_buf = []
        result_holder   = [None]
        done_event      = threading.Event()

        def _status_cb(token):
            if token == '__LISTENING__':
                with self._lock:
                    self._input_buf = list("🎤  Listening …")
            elif token == '__TRANSCRIBING__':
                with self._lock:
                    self._input_buf = list("⚙  Transcribing …")
            elif token == '__IDLE__':
                with self._lock:
                    self._input_buf = []
            self._draw(buf=list(self._input_buf))

        def _run():
            result_holder[0] = listen.listen(
                status_callback=_status_cb,
                wait_fn=wait_fn,
            )
            done_event.set()

        threading.Thread(target=_run, daemon=True).start()

        while not done_event.is_set():
            self._draw_clock_only()
            done_event.wait(1.0)

        self._input_buf = []
        self._draw(buf=[])
        return result_holder[0] or ""

    # ─────────────────────────────────────────────────────────────────────────
    # SPIN LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def spin_loop(self, stop_event):
        """
        Drive the init-phase animation loop from a dedicated thread.

        Redraws the full frame at ~12 fps, advancing the spinner frame index
        each tick so loading steps show a live Braille animation. Exits cleanly
        when stop_event is set, performing one final draw to settle the display
        before the caller proceeds.

        Args:
            stop_event: A threading.Event that signals the loop to terminate.
        """
        while not stop_event.is_set():
            self._draw()
            with self._lock:
                self._frame = (self._frame + 1) % len(SPINNER)
            stop_event.wait(0.08)
        self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """
    Parse and return the CLI argument namespace for Aiko-chan's launch options.

    Exposes three flags:
        --text       Disables ASR and TTS; uses keyboard input only.
        --debug      Surfaces memory retrieval hits in the chat area each turn.
        --clear-mem  Wipes all Qdrant-stored memories and exits immediately.

    Returns:
        An argparse.Namespace with attributes: text, debug, clear_mem.
    """
    p = argparse.ArgumentParser(description="Aiko-chan CLI")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + no TTS  (default: ASR + TTS)")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    return p.parse_args()


def _run(stdscr, args):
    """
    Orchestrate the full session lifecycle from boot to shutdown inside the
    curses wrapper.

    Spawns the TUI, concurrently initialises the inference engine (AikoThink)
    and memory cortex (AikoMemorize) on daemon threads, then conditionally
    warms up the voice pipeline. Drives the main interaction loop — routing
    voice or text input, handling slash commands, and managing streaming turns
    — until the user exits or a KeyboardInterrupt is raised.

    Args:
        stdscr: The curses window object provided by curses.wrapper.
        args:   Parsed argument namespace from parse_args().
    """
    tui   = AikoTUI(stdscr, no_voice=args.text, debug=args.debug)
    speak = AikoSpeak(silent=True) if not args.text else None

    memorize  = [None]
    think_ref = [None]
    mem_ready = threading.Event()

    spin_stop = threading.Event()
    spin_t    = threading.Thread(target=tui.spin_loop, args=(spin_stop,), daemon=True)
    spin_t.start()

    def init_think():
        tui.step_loading('think_start')
        think_ref[0] = AikoThink(None, speak=speak)
        tui.step_done('think_start')
        tui.step_loading('think_warmup')
        think_ref[0].join_warmup()
        tui.step_done('think_warmup')
        mem_ready.wait()
        think_ref[0]._memorize = memorize[0]

    def init_memorize():
        tui.step_loading('mem_qdrant')
        memorize[0] = AikoMemorize(silent=True)
        tui.step_done('mem_qdrant')
        tui.step_loading('mem_embed')
        tui.step_done('mem_embed')
        tui.step_loading('mem_cleanup')
        memorize[0].cleanup()
        tui.step_done('mem_cleanup')
        tui.step_loading('mem_ready')
        mem_ready.set()
        tui.step_done('mem_ready')

    t1 = threading.Thread(target=init_think,    daemon=True)
    t2 = threading.Thread(target=init_memorize, daemon=True)
    t1.start(); t2.start()
    t1.join();  t2.join()

    listen = None
    if not args.text:
        tui.step_loading('speak_kokoro')
        speak.warmup()
        tui.step_done('speak_kokoro')
        tui.step_loading('speak_ready')
        tui.step_done('speak_ready')
        tui.step_loading('listen_ready')
        from core.listen import AikoListen
        listen = AikoListen()
        listen.join_warmup()
        tui.step_done('listen_ready')
    else:
        tui.step_skip('speak_skip')
        tui.step_skip('listen_skip')

    spin_stop.set()
    spin_t.join()
    tui.status_finish()
    tui._draw()

    memorize    = memorize[0]
    think       = think_ref[0]
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

            elif cmd == '/voice':
                if speak is None:
                    tui.add_message('sys', 'TTS unavailable — started in --text mode.')
                else:
                    tts_enabled = not tts_enabled
                    think._speak = speak if tts_enabled else None
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
                    '/quit /exit    — end session',
                    '/reset         — clear short-term context',
                    '/clear         — wipe long-term memories',
                    '/memory        — show stored memories',
                    '/web <query>   — web search',
                    '/voice         — toggle TTS on/off',
                    '/listen        — toggle ASR on/off',
                    '/help          — show this list',
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


def main():
    """
    Primary entry point for the Aiko-chan CLI.

    Handles the --clear-mem fast-exit path before handing control to
    curses.wrapper, which sets up the terminal and invokes _run with the
    parsed argument namespace.
    """
    args = parse_args()
    if args.clear_mem:
        print('[system] Clearing all memories...')
        AikoMemorize().clear()
        sys.exit(0)
    curses.wrapper(lambda scr: _run(scr, args))


if __name__ == '__main__':
    main()
