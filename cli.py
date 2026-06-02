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
    info["ram"] = f"{ram_gb:.0f} GB" if ram_gb >= 1 else f"{ram_kb // 1024} MB"

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
    """Return 'X.X/Y GB' string read live from /proc/meminfo."""
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
        return f"{used:.1f}/{total_gb:.0f} GB"
    except Exception:
        return "? GB"

def _db_size_str(db_path: str) -> str:
    """Return human-readable size of Qdrant storage directory."""
    try:
        total = 0
        for dirpath, _, filenames in os.walk(db_path):
            for fname in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
        if total == 0:
            return "0 MB"
        mb = total / 1024 / 1024
        return f"{mb:.0f} MB" if mb >= 1 else f"{total // 1024} KB"
    except Exception:
        return "? MB"

def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ── identity ──────────────────────────────────────────────────────────────────

def _load_identity(path: str = "persona/identity.md") -> dict:
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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
SEARXNG_URL   = os.getenv("SEARXNG_URL",   "localhost:8080")
QDRANT_PATH   = os.getenv("QDRANT_PATH",   os.path.expanduser("~/.qdrant"))

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

    INPUT_PROMPT = "  ❯  "

    def __init__(self, stdscr, no_voice: bool = False, debug: bool = False):
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
        return BANNER_H + 2

    @property
    def _arch_sep_row(self):
        return self._pt + ARCH_ROWS

    @property
    def _chat_top(self):
        return self._arch_sep_row + 1

    def _dims(self):
        return self._scr.getmaxyx()

    def _chat_bot(self, h):
        return h - 6

    # ── low-level write ───────────────────────────────────────────────────────

    def _wr(self, y, x, text, attr=0):
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
        Single status line with live system + session metrics.
        Segments drop off right-to-left if terminal is too narrow.

            🎤 ASR  │  RAM 5.6/8 GB  │  DB 142 MB  │  847 tok  │  23.4 tok/s  │  ↑00:12:34
        """
        s = self._stats

        mode_parts = []
        mode_parts.append("🎤 ASR" if s['asr_on'] else "⌨  TXT")
        mode_parts.append("🔊 TTS" if s['tts_on'] else "🔇 TTS")
        mode_str = "  ".join(mode_parts)

        ram_str   = f"RAM {_ram_used_str()}"
        db_str    = f"DB {_db_size_str(QDRANT_PATH)}"
        tok_str   = f"{s['tokens']:,} tok"
        toks_str  = f"{s['tok_s']:.1f} tok/s" if s['tok_s'] > 0 else "— tok/s"
        up_str    = f"↑{_fmt_uptime(time.time() - self._ts)}"

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
        with self._lock:
            for i, (k, s, d) in enumerate(self._init_log):
                if k == key:
                    self._init_log[i] = (key, state, detail)
                    return
            self._init_log.append((key, state, detail))

    def step_loading(self, key, detail=''): self._upsert_step(key, 'loading', detail)
    def step_done   (self, key, detail=''): self._upsert_step(key, 'done',    detail)
    def step_skip   (self, key, detail=''): self._upsert_step(key, 'skip',    detail)
    def step_error  (self, key, detail=''): self._upsert_step(key, 'error',   detail)

    def status_finish(self):
        with self._lock:
            self._phase = 'chat'

    # ─────────────────────────────────────────────────────────────────────────
    # CHAT API
    # ─────────────────────────────────────────────────────────────────────────

    def add_message(self, sender, text):
        with self._lock:
            self._messages.append((sender, text))
            self._scroll = 0

    def stream_token(self, token):
        with self._lock:
            self._streaming += token
            self._stats['tokens']   += 1
            self._stats['turn_tok'] += 1

    def stream_commit(self):
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
        """Call just before streaming begins for a new turn."""
        with self._lock:
            self._stats['turn_start'] = time.time()
            self._stats['turn_tok']   = 0

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def get_input(self):
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

    def get_voice_input(self, listen):
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
            result_holder[0] = listen.listen(status_callback=_status_cb)
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
    p = argparse.ArgumentParser(description="Aiko-chan CLI")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + no TTS  (default: ASR + TTS)")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    return p.parse_args()


def _run(stdscr, args):
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
                user_input = tui.get_voice_input(listen)
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
    args = parse_args()
    if args.clear_mem:
        print('[system] Clearing all memories...')
        AikoMemorize().clear()
        sys.exit(0)
    curses.wrapper(lambda scr: _run(scr, args))


if __name__ == '__main__':
    main()