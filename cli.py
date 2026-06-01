"""
cli.py

Aiko-chan CLI — full-screen curses TUI, cyberpunk edition.
Usage:
    python cli.py               # normal chat
    python cli.py --no-voice    # disable TTS
    python cli.py --debug       # show memory debug info each turn
    python cli.py --clear-mem   # wipe all stored memories and exit

Layout (all panels confined to right column for chat/status/input):
    ╔══════════════════════════════════════════════════════════╗
    ║                      BANNER                              ║
    ╠══════════════════╦═══════════════════════════════════════╣
    ║  ASCII ART       ║  INIT LOG  /  ARCH INFO               ║
    ║  (62 wide)       ╠═══════════════════════════════════════╣
    ║                  ║  CHAT MESSAGES                        ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  STATUS BAR (right panel only)        ║
    ║                  ╠═══════════════════════════════════════╣
    ║                  ║  INPUT (right panel only)             ║
    ╚══════════════════╩═══════════════════════════════════════╝

Identity data (banner text, ASCII art) is loaded from identity.md at startup.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import json
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

# ── identity (loaded from identity.md) ───────────────────────────────────────

def _load_identity(path: str = "persona/identity.md") -> dict:
    """
    Parse identity.md and return:
      banner_lines  : list[str]
      art_lines     : list[str]   — exact character rows, backticks preserved
      art_colors    : list[list[int]]
      art_w         : int
      art_h         : int

    Art lines are delimited by '---ART---' / '---END---' sentinels (not
    backtick fences) because the art itself contains backtick sequences that
    would prematurely close any markdown code fence.

    Banner and color map still use backtick fences since they contain no
    backtick characters.

    Falls back to empty defaults if the file is missing.
    """
    result = {
        "banner_lines": [],
        "art_lines": [],
        "art_colors": [],
        "art_w": 46,
        "art_h": 44,
    }

    try:
        lines = open(path, encoding="utf-8").readlines()
        text  = "".join(lines)
    except FileNotFoundError:
        return result

    # ── banner: backtick-fenced block after "## Banner Lines" ────────────────
    banner_m = re.search(
        r'## Banner Lines\s*```[^\n]*\n(.*?)```',
        text, re.DOTALL
    )
    if banner_m:
        result["banner_lines"] = banner_m.group(1).splitlines()
        while result["banner_lines"] and not result["banner_lines"][-1].strip():
            result["banner_lines"].pop()

    # ── art lines: sentinel-delimited block ---ART--- … ---END--- ─────────────
    # Read line-by-line to preserve exact whitespace; rstrip('\n') only.
    in_art = False
    art_rows: list[str] = []
    for raw_line in lines:
        stripped = raw_line.rstrip('\n')
        if stripped.strip() == '---ART---':
            in_art = True
            continue
        if stripped.strip() == '---END---':
            in_art = False
            continue
        if in_art:
            art_rows.append(stripped)

    if art_rows:
        result["art_lines"] = art_rows
        result["art_h"] = len(art_rows)
        result["art_w"] = max((len(l) for l in art_rows), default=46)

    # ── color map: backtick-fenced json block after "### Art Color Map" ───────
    cmap_m = re.search(
        r'### Art Color Map\s*```json\s*\n(.*?)```',
        text, re.DOTALL
    )
    if cmap_m:
        try:
            result["art_colors"] = json.loads(cmap_m.group(1))
        except json.JSONDecodeError:
            pass

    return result


_IDENTITY = _load_identity()

BANNER_LINES = _IDENTITY["banner_lines"]
BANNER_H     = len(BANNER_LINES)   # typically 6

ANIME_ART_LINES  = _IDENTITY["art_lines"]
ANIME_ART_COLORS = _IDENTITY["art_colors"]
ART_W = _IDENTITY["art_w"]
ART_H = _IDENTITY["art_h"]

# ── session ───────────────────────────────────────────────────────────────────

SESSION_ID   = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "0.2*af_nicole + 0.8*jf_alpha")

# Identity / owner fields from .env
AI_NAME    = os.getenv("AI_NAME",    "Aiko")
OWNER_NAME = os.getenv("OWNER_NAME", "")
USER_ID    = os.getenv("USER_ID",    "")

LEFT_W = ART_W + 2   # art cols + left/right border chars

# ── architecture sections ─────────────────────────────────────────────────────

ARCH_SECTIONS = [
    ("MEMORY SYSTEMS", [
        ("Long-term store",  "mem0  →  Qdrant vector DB"),
        ("Embedding model",  "BGE-base-en-v1.5  (768d)"),
        ("Short-term ctx",   f"Rolling {os.getenv('CONTEXT_WINDOW_TURNS', '20')}-turn window"),
        ("Recall strategy",  "Semantic + keyword fusion"),
    ]),
    ("COGNITION", [
        ("Inference engine", "Ollama  (local, offline)"),
        ("Active model",     OLLAMA_MODEL),
        ("Web search",       "SearXNG  (google, bing, ddg)"),
        ("Persona source",   "persona/soul.md  (persistent)"),
    ]),
    ("VOICE ENGINE", [
        ("TTS backend",      "Realtime TTS + Kokoro"),
        ("Voice profile",    KOKORO_VOICE),
        ("Voice speed",      f"{os.getenv('KOKORO_SPEED', '1.0')}x"),
        ("Language",         os.getenv("KOKORO_LANG", "en-us")),
    ]),
    ("HARDWARE", [
        ("Compute node",     "Jetson Orin Nano Super 8 GB"),
        ("Storage",          "1 TB NVMe SSD"),
        ("Runtime",          "Ollama  +  JetPack 6.22"),
        ("Session ID",       SESSION_ID),
    ]),
]

ARCH_ROWS = sum(1 + len(items) for _, items in ARCH_SECTIONS) + 2

# ── init step definitions ─────────────────────────────────────────────────────

INIT_STEPS = {
    'think_start':  ('Inference Engine',  f'Spawning Ollama worker  ·  {OLLAMA_MODEL}'),
    'think_warmup': ('Model Warm-up',     'Loading weights, running prefill pass …'),
    'mem_qdrant':   ('Vector Database',   'Connecting to Qdrant  ·  localhost:6333'),
    'mem_embed':    ('Embedding Model',   'Loading BGE-base-en-v1.5  ·  768-dim vectors'),
    'mem_ready':    ('Memory Cortex',     'mem0 ready  ·  long-term recall online'),
    'speak_kokoro': ('TTS Engine',        'Initializing Kokoro Engine  ·  Female voice speaking English with a hint of Japanese ascent'),
    'speak_ready':  ('Voice Output',      'Audio pipeline ready  ·  24 kHz sample-rate'),
    'speak_skip':   ('Voice Output',      'TTS disabled  (--no-voice)'),
}

SPINNER = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

# ── colour pairs ──────────────────────────────────────────────────────────────
# Pairs 1-8: UI chrome
# Pairs 9-17: art palette

CP_PINK    = 1   # 198 hot pink    — borders, banner, accent
CP_CYAN    = 2   # 51  cyan        — You text, init spinner
CP_PURPLE  = 3   # 135 purple      — section headers, dividers
CP_MAUVE   = 4   # 177 mauve       — arch values
CP_DIM     = 5   # 240 dim grey    — detail text
CP_WHITE   = 6   # 15  white       — done ticks
CP_SBARBG  = 7   # black on pink   — status bar bg
CP_INPUTBG = 8   # cyan on dark    — input line

CP_ART_BASE = 8  # art palette pairs start at 9

# Palette: index → terminal 256-color number (or None to skip)
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
    """
    Layout:
      Row 0           : top border
      Rows 1..BANNER_H: banner text
      Row BANNER_H+1  : banner-bottom / panel-top divider  ╠═══╦═══╣
      Rows PT..bottom : left=art | right=init/arch/chat/sbar/input
      Row h-1         : bottom border

    Clock and owner tag are placed in the top-right of banner rows 1–2,
    never overwriting the last banner art line.
    """

    INPUT_PROMPT = "  ❯  "

    def __init__(self, stdscr, no_voice=False, debug=False):
        self._scr       = stdscr
        self._no_voice  = no_voice
        self._debug     = debug
        self._lock      = threading.Lock()
        self._ts        = time.time()

        self._messages: list[tuple[str,str]] = []
        self._scroll    = 0
        self._streaming = ''

        self._init_log: list[tuple[str,str,str]] = []
        self._frame     = 0
        self._phase     = 'init'

        init_colours()
        curses.curs_set(0)
        self._scr.nodelay(False)

    # ── layout ────────────────────────────────────────────────────────────────

    @property
    def _pt(self):
        """First row of the main panel area (below banner divider)."""
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
        """Last chat row (exclusive bottom for chat text)."""
        return h - 6

    # ── low-level ─────────────────────────────────────────────────────────────

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
    # DRAW BANNER
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_banner(self, h, w):
        pk = curses.color_pair(CP_PINK)  | curses.A_BOLD
        wh = curses.color_pair(CP_WHITE)
        cy = curses.color_pair(CP_CYAN)  | curses.A_BOLD

        # Top border
        self._wr(0, 0, '╔' + '═'*(w-2) + '╗', pk)

        # Banner art lines (rows 1 … BANNER_H)
        for i, line in enumerate(BANNER_LINES):
            row = 1 + i
            pad = max(1, (w - len(line)) // 2)
            self._wr(row, 1, ' '*(w-2), 0)
            self._wr(row, 0, '║', pk)
            self._wr(row, w-1, '║', pk)
            self._wr(row, pad, line, pk)

        # ── info overlays — placed in rows 1 and 2, right-aligned ────────────
        # Row 1: live clock (ticks every second)
        clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
        self._wr(1, w - 1 - len(clock_str), clock_str, wh)

        # Row 2: version + owner name  (no elapsed-seconds suffix)
        ver_str = f" {AI_NAME} v0.1.1 "
        self._wr(2, w - 1 - len(ver_str), ver_str, cy)

        # Panel-top divider: ╠═══╦═══╣
        self._wr(BANNER_H+1, 0,
            '╠' + '═'*(LEFT_W-1) + '╦' + '═'*(w-LEFT_W-2) + '╣', pk)
            
    def _draw_clock_only(self):
        """Redraw only the clock cell in row 1 — no full repaint."""
        with self._lock:
            h, w = self._dims()
            wh = curses.color_pair(CP_WHITE) | curses.A_BOLD
            clock_str = f" {time.strftime('%B %d, %Y  %I:%M:%S %p')} "
            self._wr(1, w - 1 - len(clock_str), clock_str, wh)
            rx = LEFT_W + 1
            rw = w - LEFT_W - 2
            inp_r = self._chat_bot(h) + 4
            content = self.INPUT_PROMPT + ''.join(getattr(self, '_input_buf', []))
            cx = min(rx + len(content), w - 2)
            try:
                self._scr.move(inp_r, cx)
            except curses.error:
                pass
            self._scr.refresh()
        
    # ─────────────────────────────────────────────────────────────────────────
    # DRAW LEFT COLUMN — art fills entire height from pt to bottom border
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
                        try:
                            self._scr.addch(row, x, ' ')
                        except curses.error:
                            pass
                    else:
                        try:
                            self._scr.addstr(row, x, ch, attr)
                        except curses.error:
                            pass
                    x += 1
                while x < LEFT_W:
                    try:
                        self._scr.addch(row, x, ' ')
                    except curses.error:
                        pass
                    x += 1
            else:
                try:
                    self._scr.addstr(row, 1, ' ' * (LEFT_W - 1))
                except curses.error:
                    pass

        self._wr(h-1, 0, '╚', pk)

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW RIGHT COLUMN — arch/init at top, chat, sbar, input all inside
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

        # Right-column vertical borders
        for row in range(pt, h - 1):
            self._wr(row, LEFT_W, '║', pk)
            self._wr(row, w - 1,  '║', pk)

        # Clear arch/init area
        for row in range(pt, sep):
            try:
                self._scr.addstr(row, rx, ' ' * (rw - 1))
            except curses.error:
                pass

        if self._phase == 'init':
            self._draw_right_init(pt, sep, rx, rw)
        else:
            self._draw_right_arch(pt, sep, rx, rw)

        # Arch/chat separator
        self._wr(sep, LEFT_W,
            '╠' + '═'*(w-LEFT_W-2) + '╣', pk)

        self._draw_chat_area(sep + 1, cb, rx, rw, w)

        # Chat/sbar separator
        sbar_sep = cb + 1
        self._wr(sbar_sep, LEFT_W,   '╠', pk)
        self._wr(sbar_sep, w - 1,    '╣', pk)
        try:
            self._scr.addstr(sbar_sep, LEFT_W + 1, '═' * rw,
                             curses.color_pair(CP_PINK) | curses.A_BOLD)
        except curses.error:
            pass

        # ── status bar ────────────────────────────────────────────────────────
        sr         = sbar_sep + 1
        left_parts = ["✦", AI_NAME, OLLAMA_MODEL, "mem0", "Kokoro", SESSION_ID]
        if USER_ID:
            left_parts.insert(1, USER_ID)
        right = "  "   # no elapsed seconds here

        left = "  " + "  │  ".join(left_parts) + "  "
        while len(left) + len(right) > rw and left_parts:
            left_parts.pop()
            left = "  " + "  │  ".join(left_parts) + "  "

        bar = left + ' ' * max(0, rw - len(left) - len(right)) + right
        try:
            self._scr.addstr(sr, rx, bar[:rw - 1], sbar_attr)
        except curses.error:
            pass

        # Sbar/input separator
        ir = sr + 1
        self._wr(ir, LEFT_W,   '╠', pk)
        self._wr(ir, w - 1,    '╣', pk)
        try:
            self._scr.addstr(ir, LEFT_W + 1, '═' * rw,
                             curses.color_pair(CP_PINK) | curses.A_BOLD)
        except curses.error:
            pass

        # ── input line ────────────────────────────────────────────────────────
        inp_r   = ir + 1
        content = self.INPUT_PROMPT + ''.join(buf)
        visible_start = max(0, len(content) - rw + 1)
        line = content[visible_start:].ljust(rw)
        try:
            self._scr.addstr(inp_r, rx, line, inp_attr)
        except curses.error:
            pass

        # ── bottom border ─────────────────────────────────────────────────────
        bot_r = inp_r + 1
        if bot_r < h:
            self._wr(bot_r, LEFT_W, '╩', pk)
            self._wr(bot_r, w - 1,  '╝', pk)
            try:
                self._scr.addstr(bot_r, LEFT_W + 1, '═' * rw,
                                 curses.color_pair(CP_PINK) | curses.A_BOLD)
            except curses.error:
                pass

        self._wr(bot_r, 0, '╚' + '═' * (LEFT_W - 1), pk)

        # Reposition cursor to end of input line
        cx = min(rx + len(content), w - 2)
        try:
            self._scr.move(inp_r, cx)
        except curses.error:
            pass

    # ── right panel: init log ─────────────────────────────────────────────────

    def _draw_right_init(self, pt, pb, rx, rw):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        dim = curses.color_pair(CP_DIM)
        wh  = curses.color_pair(CP_WHITE)

        self._wr(pt,   rx, " INITIALIZING NEURAL SYSTEMS", pk)
        self._wr(pt+1, rx, ' ─' * min(14, (rw-1)//2), pk)

        row = pt + 2
        for (key, state, detail) in self._init_log:
            if row >= pb:
                break
            lbl, dflt = INIT_STEPS.get(key, (key, detail))
            txt = detail if detail else dflt

            if state == 'loading':
                sp = SPINNER[self._frame]
                self._wr(row, rx,    f"  {sp} ", wh)
                self._wr(row, rx+4,  f"{lbl:<20}", wh)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'done':
                self._wr(row, rx,    "  ✓ ", cy)
                self._wr(row, rx+4,  f"{lbl:<20}", wh)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'skip':
                self._wr(row, rx,    "  – ", dim)
                self._wr(row, rx+4,  f"{lbl:<20}", dim)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'error':
                self._wr(row, rx,    "  ✗ ", wh)
                self._wr(row, rx+4,  f"{lbl:<20}", wh)
                self._wr(row, rx+24, txt[:rw-25], curses.color_pair(CP_MAUVE))
            row += 1

        all_fin = (len(self._init_log) > 0 and
                   all(s in ('done','skip','error') for (_,s,_) in self._init_log))
        if all_fin and row < pb - 1:
            self._wr(row+1, rx, "  [ ALL SYSTEMS ONLINE ]",
                     curses.color_pair(CP_CYAN) | curses.A_BOLD)

    # ── right panel: arch ─────────────────────────────────────────────────────

    def _draw_right_arch(self, pt, pb, rx, rw):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)   | curses.A_BOLD
        mv  = curses.color_pair(CP_MAUVE)
        wh  = curses.color_pair(CP_WHITE)

        self._wr(pt,   rx, " NEURAL ARCHITECTURE", pk)
        self._wr(pt+1, rx, ' ─' * min(10, (rw-1)//2), pk)

        row = pt + 2
        for section, items in ARCH_SECTIONS:
            if row >= pb - 1:
                break
            self._wr(row, rx, f"  {section}", cy)
            row += 1
            for name, val in items:
                if row >= pb:
                    break
                self._wr(row, rx,    f"    {name:<18}", wh)
                self._wr(row, rx+22, val[:rw-23],       mv)
                row += 1

    # ── chat area ─────────────────────────────────────────────────────────────

    def _render_lines(self, rw):
        avail = rw - 2
        out   = []
        for sender, text in self._messages:
            if sender == 'you':
                pre   = f" {USER_ID}: "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('Y', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('Y', ind + l))
            elif sender == 'aiko':
                pre   = f" {AI_NAME}: "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('A', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('A', ind + l))
                out.append(('S', ''))
            elif sender == 'sys':
                out.append(('S', f"  ◈  {text}"))
        if self._streaming:
            pre   = f" {AI_NAME}: "
            ind   = " " * len(pre)
            lines = textwrap.wrap(self._streaming, avail - len(pre)) or [""]
            out.append(('A', pre + lines[0]))
            for l in lines[1:]:
                out.append(('A', ind + l))
        return out

    def _draw_chat_area(self, ct, cb, rx, rw, w):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)
        mv  = curses.color_pair(CP_MAUVE)
        dim = curses.color_pair(CP_DIM)
        pu  = curses.color_pair(CP_PURPLE)

        ch = cb - ct

        rendered   = self._render_lines(rw)
        total      = len(rendered)
        max_scroll = max(0, total - ch)
        self._scroll = max(0, min(self._scroll, max_scroll))

        start   = max(0, total - ch - self._scroll)
        visible = rendered[start:start+ch]

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

            try:
                self._scr.addstr(row, rx, ' ' * (rw - 1))
            except curses.error:
                pass

            if prefix and line.startswith(prefix):
                self._wr(row, rx, prefix, bold_attr)
                self._wr(row, rx + len(prefix),
                          line[len(prefix):][:rw - len(prefix) - 1], text_attr)
            else:
                self._wr(row, rx, line[:rw - 1], text_attr)

            self._wr(row, w - 1, '║', pk)

        for row in range(ct + len(visible), cb):
            try:
                self._scr.addstr(row, rx, ' ' * (rw - 1))
            except curses.error:
                pass
            self._wr(row, w-1, '║', pk)

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
                self._wr(0, 0,
                    f"Terminal too small: {w}x{h}  (need {min_w}x{min_h})", 0)
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
            for i, (k,s,d) in enumerate(self._init_log):
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

    def stream_commit(self):
        with self._lock:
            if self._streaming:
                self._messages.append(('aiko', self._streaming))
                self._streaming = ''
                self._scroll = 0

    # ─────────────────────────────────────────────────────────────────────────
    # INPUT LOOP — ticker thread keeps clock/uptime alive while idle/typing
    # ─────────────────────────────────────────────────────────────────────────

    def get_input(self):
        buf = []
        self._input_buf = buf
        curses.curs_set(1)
        self._scr.nodelay(True)  # non-blocking so ticker can run

        # Background ticker: redraws every second to update clock
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
                    time.sleep(0.05)  # yield CPU
                    continue

                h, w = self._dims()

                if ch in ('\n', '\r', curses.KEY_ENTER):
                    break
                elif ch in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                    if buf: buf.pop()
                elif ch == curses.KEY_PPAGE:
                    with self._lock:
                        rw = w - LEFT_W - 2
                        rendered = self._render_lines(rw)
                        ch_h = self._chat_bot(h) - self._chat_top
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

                # Redraw immediately on keypress for responsive feel
                self._draw(buf=buf)

        finally:
            stop_tick.set()
            tick_t.join()
            curses.curs_set(0)

        return ''.join(buf).strip()

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
# entry
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Aiko-chan CLI")
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--no-voice",  action="store_true")
    p.add_argument("--clear-mem", action="store_true")
    return p.parse_args()


def _run(stdscr, args):
    tui   = AikoTUI(stdscr, no_voice=args.no_voice, debug=args.debug)
    speak = AikoSpeak(silent=True) if not args.no_voice else None

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

    if speak:
        tui.step_loading('speak_kokoro')
        speak.warmup()
        tui.step_done('speak_kokoro')
        tui.step_loading('speak_ready')
        tui.step_done('speak_ready')
    else:
        tui.step_skip('speak_skip')

    spin_stop.set()
    spin_t.join()
    tui.status_finish()
    tui._draw()

    memorize = memorize[0]
    think    = think_ref[0]

    while True:
        try:
            user_input = tui.get_input()
        except KeyboardInterrupt:
            tui.add_message('sys', "Fine... I'll be here when you come back.")
            tui._draw()
            think.wait_for_memory()
            time.sleep(0.8)
            return

        if not user_input:
            continue

        if user_input.startswith('/'):
            cmd = user_input.lower()
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
                        tui.add_message('sys', f'  {i:02d}. {m.get("memory") or m.get("text") or m}')
            elif cmd == '/voice':
                if think._speak:
                    think._speak = None
                    tui.add_message('sys', 'Voice output: DISABLED')
                else:
                    if not speak:
                        speak = AikoSpeak(silent=True)
                        speak.warmup()
                    think._speak = speak
                    tui.add_message('sys', 'Voice output: ENABLED')
            elif cmd == '/clear':
                memorize.clear()
                tui.add_message('sys', 'All persistent memories cleared from database.')
            elif cmd == '/help':
                tui.add_message('sys', '/quit /exit    — end session')
                tui.add_message('sys', '/reset         — clear context')
                tui.add_message('sys', '/voice         — toggle voice')
                tui.add_message('sys', '/clear         — wipe memories')
                tui.add_message('sys', '/memory        — show memories')
                tui.add_message('sys', '/web <query>   — web search')
                tui.add_message('sys', '/help          — show this list')
            elif cmd.startswith('/web '):
                query = user_input[5:].strip()
                if query:
                    tui.add_message('sys', f'Searching: "{query}"')
                    tui._draw()
                    from core.tools import web_search
                    results = web_search(query)
                    think._history.append({"role": "user", "content": results})
                    def token_cb(token):
                        tui.stream_token(token)
                        tui._draw(buf=[])
                    think.chat(f"Based on the search results, answer: {query}", token_callback=token_cb)
                    tui.stream_commit()
                    tui._draw()
                else:
                    tui.add_message('sys', 'Usage: /web <query>')
            else:
                tui.add_message('sys', f'Unknown command: {user_input}')
            tui._draw()
            continue

        if args.debug:
            hits = memorize.search(user_input)
            if hits:
                tui.add_message('sys', f'{len(hits)} memories retrieved:')
                for m in hits:
                    tui.add_message('sys', f'  → {m.get("memory") or m.get("text") or m}')

        tui.add_message('you', user_input)
        tui._draw()

        def token_cb(token):
            if token.startswith("__SEARCHING__:"):
                query = token.split(":", 1)[1].strip()
                tui.stream_commit()          # discard any partial [SEARCH:...] that leaked
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