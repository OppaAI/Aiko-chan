"""
cli.py

Aiko-chan CLI — full-screen curses TUI, cyberpunk edition.
Usage:
    python cli.py               # normal chat
    python cli.py --no-voice    # disable TTS
    python cli.py --debug       # show memory debug info each turn
    python cli.py --clear-mem   # wipe all stored memories and exit
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
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

# ── session ───────────────────────────────────────────────────────────────────

SESSION_ID   = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")

# ── banner ────────────────────────────────────────────────────────────────────

BANNER_LINES = [
    " █████╗ ██╗██╗  ██╗ ██████╗       ██████╗██╗  ██╗ █████╗ ███╗  ██╗",
    "██╔══██╗██║██║ ██╔╝██╔═══██╗     ██╔════╝██║  ██║██╔══██╗████╗ ██║",
    "███████║██║█████╔╝ ██║   ██║  ─  ██║     ███████║███████║██╔██╗██║",
    "██╔══██║██║██╔═██╗ ██║   ██║     ██║     ██╔══██║██╔══██║██║╚████║",
    "██║  ██║██║██║  ██╗╚██████╔╝     ╚██████╗██║  ██║██║  ██║██║  ███║",
    "╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝       ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝",
]
BANNER_H = len(BANNER_LINES)  # 6

# ── anime art (left panel) ────────────────────────────────────────────────────
# Auto-generated from PNG — 64 chars wide x 50 rows tall
# Color codes per character: K=skip(black), D=dim, M=mauve, P=pink, W=white

ART_W = 64
ART_H = 50

ANIME_ART_LINES = [
    '                                               .<zTCfC)7}|CIIiii',
    "                                              ``:';Ji{C3I1fLJ(v{",
    "                                            '_<!' !(F}iCfi}CJ(}i",
    '                                            .` .-^F}|Jis7|(|(7vF',
    '                                             ,^:/J|)7iJs((vLssLJ',
    '                              `  ``.`        ^r+*F{iz)z?JJ|7|{fv',
    "                        --.      `-'_.``   .'--;sJT?ssL/LJiv{{iF",
    "                       .`    >/ `    `.`.  `'!+cs//*z?z/)vFvsT)?",
    "                      `.   `^'I* _,   `.    ->rr=>!***r+*/;+sr!!",
    "                      -'   -+.(ts*!=. `-`  .- -'<+c//c!<=<c))?+c",
    '                      .`+c  ,Flle}J{`  `.`  .^^_-^=<>c!r/?//c!!+',
    "                       .?lc 'C[t1[5z `....     `   .`,>!++<+,>;;",
    "                       `^|n}^:lutuT-..-`-       ```';+<,,-:,':,!",
    "                        ``>[1rT|v- .```          `'':.::`.,.:-_^",
    "                      -*TL+i[f1)vv_             `.'-``.` `:..._<",
    '                      */r*/it|/|(J<,:               -..` `_`. :_',
    "                      rs?z!sL?*/IFF)z*,            `-.'` `:   ,=",
    "                      <c*z/??Lsc)oIi7sL_           .'.'. `,   ^=",
    "                      :,;+czvvvzzi[Iutv!7ic`       .'`-' .,   _,",
    "                       ->,:;!+><L/7u{u[v*)Fc       -..'- ._   ,=",
    '                        ^,,,,_>zLL/zil|(,:._=>:   `,`--/+.,   ,;',
    "                        >>^:,rTLLsTcsuf*_+>cJLr   .' '_c_._   ^>",
    "                       .+^,+?TLTz=ssc|ec^+>:/LCc  -. __' `^   ^>",
    "                        ^<!sLsLT!=<T?z|v=<><>cfl'.:--'--  ^   ;>",
    "`                      `+r?sLLLL>>=*??)}<;<+<_(ZJ`'---''  ,   =>",
    "-                      -cTsLszss+<+zzvC3F<+<<^+}J'-``.-- `,.` =^",
    ":                      ^Trcz*/zz?z)(7{I{7<=;>>=. -'`-:-.`._`  =_",
    "_`                     -;J{f}JfC{?LF(i7v/*!<:.`  `'.-:-.--,   ;,",
    ":.                      ([l[n{lCuI?ii|CJIn3CT,   .':--'----   ;^",
    "--                     ![Julu11C}nuI}!zi)7C3Cv*' ._'''---'-   ^^",
    "`'`                   ;{(F[ttlI{([ll17zL}1FT[t{3J<``-...`.:   ,,",
    "`-..                 /ivzlult[3CFltluu{|vC}|Jnt{[}?'``   `:   ,_",
    "`-'                 >uJv/}ff}{|I[l1}TcL3{*vsF{(ifF|, ``` `.   ,_",
    "``:.               `7Fzs/c(iiJ*i}{{CfJL()J)7vL7itt}z..-`  -   '=",
    ". ''                =}uzs)n[uo3TvsI[1fLTT;rJ?sC{?TLr`     .   _^",
    ': .:                <}()vcTTT)(tu)*c!!rs/rJ7|(Lr_,.  `  ` -   _=',
    "=` '.               '??cLTvT?rr)(s*)FC}I3;*Lzc<-`---.-    '   ^=",
    ">' ..                 .L1fII}}r ^zI33If1T        .'.`--`  '   =>",
    "=,``.                  7fCCC}f_  z1C}}}f'        .       `'   ;!",
    ':;``.                 .i}C}C3? ` TfC}}3z         .` `.    -   >r',
    "`>'``                 -{f}}fC-   vi}}fC-      `` .`  .   :'   ;r",
    " _,                   ^3}ff1c   `FII}1/        ```'-'`  -+'   =r",
    " .=.                  +1IfIC.   _IIfIf'           .`_^`  `_   =r",
    "  ^'     ``           L1ff1*    ^1If1v           `-.^!=  .:   =r",
    ". :=  `` ` `` ```    _fIf3C.    .{I}1)           `-`;+'  .:   =r",
    "-``=.             ` -FI{fC^      L3}I|            ` :=,  .'   ^+",
    "`. ':               s3}}3! ` ` ``J3}f{.     ` ``   ` `>. ..   ^+",
    '..`.-``            :fffIi.      `{3IIC` ````` ```   ``^- .cl(-^<',
    ":`. `  `           !3CC3!       'Cf}f7         `         .^s+`,>",
    ',.` `    `         ?3CI{-       ;If}1?              ` `  .-   ^>',
]

# Per-character color map: one string per row, same length as art line
# K=transparent/skip, D=dim grey, M=mauve/light-purple, P=hot-pink, W=white
ANIME_ART_COLORS = [
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDDMMMDDMDMMMMDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDDMMMMMMDDDDM',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDKKDDDMDMMMMMDDMD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDMDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDKDDDDDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDDDDMDDDDDDDDDDMMD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDMMMD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKDDKKKKKKKKKKKKKKDDDDDDDDDDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKDKMDKKDKKKKKKKKKKDDDDDDDDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKKKDKDMDDDDKKKKKKKKKKKKDDDDDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDKKDDMMMMDMKKKKKKKKKDDKKDDDDDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDMDKKMMMMMMDKKKKKKKKKKKKKKKKKDDDDDDDDDDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDMMDKMMMMDKKKKKKKKKKKKKKKKKDDDDDKKDKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKKKDMMDDDDKKKKKKKKKKKKKKKKKKKKKKKKKDKKKKD',
    'KKKKKKKKKKKKKKKKKKKKKKKDDDDMMMMDDDKKKKKKKKKKKKKKKKKKKKKKKKKKKKKD',
    'KKKKKKKKKKKKKKKKKKKKKKDDDDDMMDDDDDDKKKKKKKKKKKKKKKKKKKKKKKKKKKKK',
    'KKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDMDDDDDDKKKKKKKKKKKKKKKKKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDMMDDDDKKKKKKKKKKKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDMMMMMDDDMDKKKKKKKKKKKKKKKDKKKKD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDKDDDDDDDDMMMMDDDDDKKKKKKKKKKKKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDKDKKDDDDDDMMDDKKKKDDKKKKKKKKKDDKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDDKKDDDDDDDDMMDKDDDDDDKKKKKKKKDKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDKDDDDDDDDDDDMDDDDKDDMDKKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDDDDDDDDDDMMKKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDDDMDDDDDKDMDKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDDMMDDDDDDDMDKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDDMMMDDDDDDDKKKKKKKKKKKKKKKDK',
    'KKKKKKKKKKKKKKKKKKKKKKKKDDMMMDMMMDDDDDDDDDDDKKKKKKKKKKKKKKDKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKKDMMMMMMMMMDDDDMDMMMMDKKKKKKKKKKKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKKDMDMMMMMMMMMMMDDDDDMMMDDKKKKKKKKKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKKKDMDDMMMMMMDMMMMDDDMMDDMMMMDDKKKKKKKKKKKKKD',
    'KKKKKKKKKKKKKKKKKKKKKDDDDMMMMMMMDMMMMMMDDMMDDMMMMMDKKKKKKKKKKKDK',
    'KKKKKKKKKKKKKKKKKKKKDMDDDMMMMMDMMMMMDDDMMDDDDMDDMDDKKKKKKKKKKKKK',
    'KKKKKKKKKKKKKKKKKKKKDDDDDDDMDDDDMMMMMDDDDDDDDDDDMMMDKKKKKKKKKKKD',
    'KKKKKKKKKKKKKKKKKKKKDMMDDDMMMMMDDDMMMMDDDDDDDDMMDDDDKKKKKKKKKKKD',
    'KKKKKKKKKKKKKKKKKKKKDMDDDDDDDDDMMDDDDDDDDDDDDDDDKKKKKKKKKKKKKKKD',
    'DKKKKKKKKKKKKKKKKKKKKDDDDDDDDDDDDDDDDMMMMDDDDDDKKKKKKKKKKKKKKKDD',
    'DKKKKKKKKKKKKKKKKKKKKKKDMMMMMMDKDDMMMMMMDKKKKKKKKKKKKKKKKKKKKKDD',
    'DKKKKKKKKKKKKKKKKKKKKKKDMMMMMMKKKDMMMMMMKKKKKKKKKKKKKKKKKKKKKKDD',
    'KDKKKKKKKKKKKKKKKKKKKKKMMMMMMDKKKDMMMMMDKKKKKKKKKKKKKKKKKKKKKKDD',
    'KDKKKKKKKKKKKKKKKKKKKKKMMMMMMKKKKDDMMMMKKKKKKKKKKKKKKKKKKKKKKKDD',
    'KKDKKKKKKKKKKKKKKKKKKKDMMMMMDKKKKDMMMMDKKKKKKKKKKKKKKKKKKDKKKKDD',
    'KKDKKKKKKKKKKKKKKKKKKKDMMMMMKKKKKMMMMMKKKKKKKKKKKKKKKDKKKKKKKKDD',
    'KKDKKKKKKKKKKKKKKKKKKKDMMMMDKKKKDMMMMDKKKKKKKKKKKKKKDDDKKKKKKKDD',
    'KKKDKKKKKKKKKKKKKKKKKKMMMMMKKKKKKMMMMDKKKKKKKKKKKKKKDDKKKKKKKKDD',
    'KKKDKKKKKKKKKKKKKKKKKDMMMMDKKKKKKDMMMDKKKKKKKKKKKKKKKDKKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKDMMMMDKKKKKKKDMMMMKKKKKKKKKKKKKKKKDKKKKKKKDD',
    'KKKKKKKKKKKKKKKKKKKKMMMMDKKKKKKKKMMMMMKKKKKKKKKKKKKKKKDKKKDMDKDD',
    'KKKKKKKKKKKKKKKKKKKDMMMMDKKKKKKKKMMMMDKKKKKKKKKKKKKKKKKKKKDDDKDD',
    'DKKKKKKKKKKKKKKKKKKDMMMMKKKKKKKKDMMMMDKKKKKKKKKKKKKKKKKKKKKKKKDD',
]

# Left panel width = art width + 2 border chars (║ on each side)
LEFT_W = ART_W + 2   # = 66

# ── architecture sections ─────────────────────────────────────────────────────

ARCH_SECTIONS = [
    ("MEMORY SYSTEMS", [
        ("Long-term store",  "mem0  →  Qdrant vector DB"),
        ("Embedding model",  "BGE-base-en-v1.5  (768d)"),
        ("Short-term ctx",   "Rolling 20-turn window"),
        ("Recall strategy",  "Semantic + keyword fusion"),
    ]),
    ("COGNITION", [
        ("Inference engine", "Ollama  (local, offline)"),
        ("Active model",     OLLAMA_MODEL),
        ("Web search",       "DuckDuckGo  (on-demand)"),
        ("Persona source",   "soul.md  (persistent)"),
    ]),
    ("VOICE ENGINE", [
        ("TTS backend",      "Kokoro ONNX  (CPU/GPU)"),
        ("Primary voice",    "af_heart  (en-us female)"),
        ("Fallback voice",   "jf_alpha  (en-us soft)"),
        ("Latency target",   "< 300 ms first token"),
    ]),
    ("HARDWARE", [
        ("Compute node",     "Jetson Orin Nano  8 GB"),
        ("Storage",          "1 TB NVMe SSD"),
        ("Runtime",          "vLLM  +  JetPack 6.x"),
        ("Session ID",       SESSION_ID),
    ]),
]

# ── init step definitions ─────────────────────────────────────────────────────

INIT_STEPS = {
    'think_start':  ('Inference Engine',  f'Spawning Ollama worker  ·  {OLLAMA_MODEL}'),
    'think_warmup': ('Model Warm-up',     'Loading weights, running prefill pass …'),
    'mem_qdrant':   ('Vector Database',   'Connecting to Qdrant  ·  localhost:6333'),
    'mem_embed':    ('Embedding Model',   'Loading BGE-base-en-v1.5  ·  768-dim vectors'),
    'mem_ready':    ('Memory Cortex',     'mem0 ready  ·  long-term recall online'),
    'speak_kokoro': ('TTS Engine',        'Initialising Kokoro ONNX  ·  voice: af_heart'),
    'speak_ready':  ('Voice Output',      'Audio pipeline ready  ·  24 kHz sample-rate'),
    'speak_skip':   ('Voice Output',      'TTS disabled  (--no-voice)'),
}

# ── spinner ───────────────────────────────────────────────────────────────────

SPINNER = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
HELP_TEXT = "/quit /exit — end  │  /reset — clear context  │  /memory — show memories  │  /help"

# ── colour pairs ─────────────────────────────────────────────────────────────
#
#  CP_PINK    198  hot pink       — primary accent, borders, banner, art pink
#  CP_CYAN     51  electric cyan  — secondary, values, chat You
#  CP_PURPLE  135  purple         — section headers
#  CP_MAUVE   177  light purple   — art midtones, sub-labels
#  CP_DIM     240  dim grey       — art darks, detail text
#  CP_WHITE    15  bright white   — done ticks, art highlights
#  CP_SBARBG       status bar     — black on pink
#  CP_INPUTBG      input line     — cyan on very dark

CP_PINK    = 1
CP_CYAN    = 2
CP_PURPLE  = 3
CP_MAUVE   = 4
CP_DIM     = 5
CP_WHITE   = 6
CP_SBARBG  = 7
CP_INPUTBG = 8

# Map art color codes to curses pair numbers
_ART_COLOR_MAP = {
    'K': None,    # transparent / skip
    'D': CP_DIM,
    'M': CP_MAUVE,
    'P': CP_PINK,
    'W': CP_WHITE,
}


def init_colours() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_PINK,    198,  -1)
    curses.init_pair(CP_CYAN,     51,  -1)
    curses.init_pair(CP_PURPLE,  135,  -1)
    curses.init_pair(CP_MAUVE,   177,  -1)
    curses.init_pair(CP_DIM,     240,  -1)
    curses.init_pair(CP_WHITE,    15,  -1)
    curses.init_pair(CP_SBARBG,   16, 198)
    curses.init_pair(CP_INPUTBG,  51,  -1)


# ─────────────────────────────────────────────────────────────────────────────
# AikoTUI
# ─────────────────────────────────────────────────────────────────────────────

class AikoTUI:
    """
    Fixed layout:
      Row 0            : top border  ╔═══╗
      Rows 1..BANNER_H : banner text
      Row BANNER_H+1   : banner-bottom / panel-top divider  ╠═══╦═══╣
      Rows BT+1..BT+PH : side panels  (left art | right init/arch)
      Row BT+PH+1      : panel-bottom / chat-top border  ╠═══╩═══╣
      Rows ..h-4       : chat scrollback
      Row h-3          : separator  ╠═══╣
      Row h-2          : status bar (pink bg)
      Row h-1          : separator + input bar
    """

    PANEL_H    = ART_H        # side panels as tall as the art
    INPUT_H    = 1
    SBAR_H     = 1

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

    # ── layout helpers ────────────────────────────────────────────────────────

    @property
    def _panel_row_top(self):
        return BANNER_H + 2

    @property
    def _panel_row_bot(self):
        return self._panel_row_top + self.PANEL_H

    @property
    def _chat_row_top(self):
        return self._panel_row_bot + 1

    def _chat_row_bot(self, h):
        return h - self.SBAR_H - self.INPUT_H - 2

    def _dims(self):
        return self._scr.getmaxyx()

    # ── low-level draw ────────────────────────────────────────────────────────

    def _wr(self, y, x, text, attr=0):
        h, w = self._dims()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x - 1
        if avail <= 0:
            return
        try:
            self._scr.addstr(y, x, text[:avail], attr)
        except curses.error:
            pass

    def _hline(self, y, x, ch, n, attr=0):
        h, w = self._dims()
        if y < 0 or y >= h:
            return
        n = min(n, w - x - 1)
        if n <= 0:
            return
        try:
            self._scr.attron(attr)
            self._scr.hline(y, x, ch, n)
            self._scr.attroff(attr)
        except curses.error:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW BANNER
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_banner(self, h, w):
        pk  = curses.color_pair(CP_PINK) | curses.A_BOLD
        dim = curses.color_pair(CP_DIM)

        self._wr(0, 0, '╔' + '═'*(w-2) + '╗', pk)

        for i, line in enumerate(BANNER_LINES):
            row = 1 + i
            pad = max(1, (w - len(line)) // 2)
            self._wr(row, 1, ' '*(w-2), 0)
            self._wr(row, 0, '║', pk)
            self._wr(row, w-1, '║', pk)
            self._wr(row, pad, line, pk)

        tag = f" Aiko Agent v2.0  [{SESSION_ID}]  {int(time.time()-self._ts)}s "
        self._wr(BANNER_H, w-1-len(tag), tag, dim)

        div_row = BANNER_H + 1
        self._wr(div_row, 0,
            '╠' + '═'*(LEFT_W-1) + '╦' + '═'*(w-LEFT_W-2) + '╣', pk)

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW SIDE PANELS
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_side_panels(self, h, w):
        pk   = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy   = curses.color_pair(CP_CYAN)   | curses.A_BOLD
        pu   = curses.color_pair(CP_PURPLE) | curses.A_BOLD
        mv   = curses.color_pair(CP_MAUVE)
        dim  = curses.color_pair(CP_DIM)
        wh   = curses.color_pair(CP_WHITE)  | curses.A_BOLD

        pt = self._panel_row_top
        pb = self._panel_row_bot

        # outer borders for every panel row
        for r in range(pt, pb):
            self._wr(r, 0,      '║', pk)
            self._wr(r, LEFT_W, '║', pk)
            self._wr(r, w-1,    '║', pk)

        # ── per-character coloured ASCII art ─────────────────────────────────
        for i in range(min(ART_H, pb - pt)):
            row      = pt + i
            art_line = ANIME_ART_LINES[i] if i < len(ANIME_ART_LINES) else ''
            clr_line = ANIME_ART_COLORS[i] if i < len(ANIME_ART_COLORS) else ''
            x        = 1   # start inside left border

            # Draw character by character with per-char color
            for col_idx, ch in enumerate(art_line):
                if x >= LEFT_W:
                    break
                color_code = clr_line[col_idx] if col_idx < len(clr_line) else 'K'
                pair_id    = _ART_COLOR_MAP.get(color_code)
                if pair_id is None or ch == ' ':
                    # transparent / space — just advance
                    try:
                        self._scr.move(row, x)
                        self._scr.addch(' ')
                    except curses.error:
                        pass
                else:
                    attr = curses.color_pair(pair_id)
                    # Bold on brighter tones for contrast
                    if color_code in ('P', 'W'):
                        attr |= curses.A_BOLD
                    try:
                        self._scr.addstr(row, x, ch, attr)
                    except curses.error:
                        pass
                x += 1

            # Pad the rest of the art column with spaces to clear stale chars
            while x < LEFT_W:
                try:
                    self._scr.addch(row, x, ' ')
                except curses.error:
                    pass
                x += 1

        # bottom panel border ╠═══╩═══╣
        self._wr(pb, 0,
            '╠' + '═'*(LEFT_W-1) + '╩' + '═'*(w-LEFT_W-2) + '╣', pk)

        # ── clear right panel and redraw ──────────────────────────────────────
        rx = LEFT_W + 2
        rw = w - LEFT_W - 4
        for r in range(pt, pb):
            try:
                self._scr.move(r, LEFT_W + 1)
                self._scr.clrtoeol()
            except curses.error:
                pass
            self._wr(r, w-1, '║', pk)

        if self._phase == 'init':
            self._draw_right_init(pt, pb, rx, rw, dim, cy, pu, mv, wh, pk)
        else:
            self._draw_right_arch(pt, pb, rx, rw, dim, cy, pu, pk)

    # ── right panel: init loading ─────────────────────────────────────────────

    def _draw_right_init(self, pt, pb, rx, rw, dim, cy, pu, mv, wh, pk):
        self._wr(pt,   rx, "INITIALISING NEURAL SYSTEMS", pk)
        self._wr(pt+1, rx, '─' * min(32, rw), pu)

        row = pt + 2
        for (key, state, detail) in self._init_log:
            if row >= pb:
                break
            lbl, dflt = INIT_STEPS.get(key, (key, detail))
            txt = detail if detail else dflt

            if state == 'loading':
                sp = SPINNER[self._frame]
                self._wr(row, rx,    f" {sp} ", cy)
                self._wr(row, rx+3,  f"{lbl:<20}", curses.color_pair(CP_CYAN)|curses.A_BOLD)
                self._wr(row, rx+23, txt[:rw-24], dim)
            elif state == 'done':
                self._wr(row, rx,    " ✓ ", wh)
                self._wr(row, rx+3,  f"{lbl:<20}", curses.color_pair(CP_WHITE)|curses.A_BOLD)
                self._wr(row, rx+23, txt[:rw-24], dim)
            elif state == 'skip':
                self._wr(row, rx,    " – ", dim)
                self._wr(row, rx+3,  f"{lbl:<20}", dim)
                self._wr(row, rx+23, txt[:rw-24], dim)
            elif state == 'error':
                self._wr(row, rx,    " ✗ ", curses.color_pair(CP_PINK)|curses.A_BOLD)
                self._wr(row, rx+3,  f"{lbl:<20}", curses.color_pair(CP_PINK)|curses.A_BOLD)
                self._wr(row, rx+23, txt[:rw-24], curses.color_pair(CP_MAUVE))
            row += 1

        all_fin = (len(self._init_log) > 0 and
                   all(s in ('done','skip','error') for (_,s,_) in self._init_log))
        if all_fin and row <= pb - 2:
            self._wr(row+1, rx, "[ ALL SYSTEMS ONLINE ]",
                     curses.color_pair(CP_CYAN)|curses.A_BOLD)

    # ── right panel: architecture ─────────────────────────────────────────────

    def _draw_right_arch(self, pt, pb, rx, rw, dim, cy, pu, pk):
        self._wr(pt,   rx, "NEURAL ARCHITECTURE", curses.color_pair(CP_CYAN)|curses.A_BOLD)
        self._wr(pt+1, rx, '─' * min(24, rw), pu)

        row = pt + 2
        for section, items in ARCH_SECTIONS:
            if row >= pb - 1:
                break
            self._wr(row, rx, f" {section}",
                     curses.color_pair(CP_PINK)|curses.A_BOLD)
            row += 1
            for name, val in items:
                if row >= pb:
                    break
                self._wr(row, rx,     f"  {name:<18}", curses.color_pair(CP_DIM))
                self._wr(row, rx+20,  val[:rw-21],     curses.color_pair(CP_MAUVE))
                row += 1

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW CHAT AREA
    # ─────────────────────────────────────────────────────────────────────────

    def _render_lines(self, w):
        avail = w - 4
        out   = []
        for sender, text in self._messages:
            if sender == 'you':
                pre   = "  You ❯  "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('Y', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('Y', ind + l))
            elif sender == 'aiko':
                pre   = "  Aiko ♡  "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('A', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('A', ind + l))
                out.append(('S', ''))
            elif sender == 'sys':
                out.append(('S', f"  ◈  {text}"))
        if self._streaming:
            pre   = "  Aiko ♡  "
            ind   = " " * len(pre)
            lines = textwrap.wrap(self._streaming, avail - len(pre)) or [""]
            out.append(('A', pre + lines[0]))
            for l in lines[1:]:
                out.append(('A', ind + l))
        return out

    def _draw_chat(self, h, w):
        pk  = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)  | curses.A_BOLD
        pk_ = curses.color_pair(CP_PINK)
        dim = curses.color_pair(CP_DIM)

        ct  = self._chat_row_top
        cb  = self._chat_row_bot(h)
        ch  = cb - ct

        for r in range(ct, cb):
            self._wr(r, 0,   '║', pk)
            self._wr(r, w-1, '║', pk)

        rendered   = self._render_lines(w)
        total      = len(rendered)
        max_scroll = max(0, total - ch)
        self._scroll = max(0, min(self._scroll, max_scroll))

        start   = max(0, total - ch - self._scroll)
        visible = rendered[start:start+ch]

        for i, (kind, line) in enumerate(visible):
            r = ct + i
            if r >= cb:
                break
            attr = cy if kind == 'Y' else (pk_ if kind == 'A' else dim)
            try:
                self._scr.move(r, 1)
                self._scr.clrtoeol()
            except curses.error:
                pass
            self._wr(r, 1, line[:w-2], attr)
            self._wr(r, w-1, '║', pk)

        for r in range(ct + len(visible), cb):
            try:
                self._scr.move(r, 1)
                self._scr.clrtoeol()
            except curses.error:
                pass
            self._wr(r, 0, '║', pk)
            self._wr(r, w-1, '║', pk)

        if self._scroll > 0:
            hint = f" ↑ {self._scroll} lines  PgDn to return "
            self._wr(ct, w - len(hint) - 2, hint, curses.color_pair(CP_PURPLE))

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW STATUS BAR + INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_sbar_input(self, h, w, buf):
        pk   = curses.color_pair(CP_PINK)   | curses.A_BOLD
        sbar = curses.color_pair(CP_SBARBG) | curses.A_BOLD
        inp  = curses.color_pair(CP_INPUTBG)

        cb = self._chat_row_bot(h)

        self._wr(cb, 0, '╠' + '═'*(w-2) + '╣', pk)

        sr      = cb + 1
        elapsed = int(time.time() - self._ts)
        left    = f"  ✦ {OLLAMA_MODEL}  │  mem0·Qdrant  │  Kokoro TTS  │  {SESSION_ID}  "
        right   = f"  {elapsed}s  "
        bar     = left + ' '*max(0, w - len(left) - len(right)) + right
        self._wr(sr, 0, bar[:w], sbar)

        ir = sr + 1
        self._wr(ir, 0, '╠' + '═'*(w-2) + '╣', pk)

        inp_r   = ir + 1
        content = self.INPUT_PROMPT + ''.join(buf)
        line    = content[:w-1].ljust(w-1)
        try:
            self._scr.move(inp_r, 0)
            self._scr.clrtoeol()
        except curses.error:
            pass
        self._wr(inp_r, 0, line, inp)

        bot_r = inp_r + 1
        if bot_r < h:
            self._wr(bot_r, 0, '╚' + '═'*(w-2) + '╝', pk)

        cx = min(len(content), w-2)
        try:
            self._scr.move(inp_r, cx)
        except curses.error:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER DRAW
    # ─────────────────────────────────────────────────────────────────────────

    def _draw(self, buf=None):
        with self._lock:
            h, w = self._dims()
            # Need enough room: banner(8) + panels(50) + chat(≥4) + chrome(4) = ~66 min
            if h < 30 or w < 100:
                self._wr(0, 0, f"Terminal too small: {w}x{h} (need 100x30 minimum)", 0)
                self._scr.refresh()
                return
            self._draw_banner(h, w)
            self._draw_side_panels(h, w)
            if self._phase == 'chat':
                self._draw_chat(h, w)
                self._draw_sbar_input(h, w, buf if buf is not None else [])
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

    def step_loading(self, key, detail=''):   self._upsert_step(key, 'loading', detail)
    def step_done(self, key, detail=''):      self._upsert_step(key, 'done',    detail)
    def step_skip(self, key, detail=''):      self._upsert_step(key, 'skip',    detail)
    def step_error(self, key, detail=''):     self._upsert_step(key, 'error',   detail)

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
    # INPUT LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def get_input(self):
        h, w = self._dims()
        buf  = []
        curses.curs_set(1)
        self._scr.nodelay(False)

        while True:
            self._draw(buf=buf)
            try:
                ch = self._scr.get_wch()
            except curses.error:
                continue

            if ch in ('\n', '\r', curses.KEY_ENTER):
                break
            elif ch in (curses.KEY_BACKSPACE, '\x7f', '\b'):
                if buf: buf.pop()
            elif ch == curses.KEY_PPAGE:
                with self._lock:
                    rendered = self._render_lines(w)
                    ch_h = self._chat_row_bot(h) - self._chat_row_top
                    self._scroll = min(
                        self._scroll + max(1, ch_h - 2),
                        max(0, len(rendered) - ch_h))
            elif ch == curses.KEY_NPAGE:
                with self._lock:
                    ch_h = self._chat_row_bot(h) - self._chat_row_top
                    self._scroll = max(0, self._scroll - max(1, ch_h - 2))
            elif ch in ('\x03', '\x04'):
                curses.curs_set(0)
                raise KeyboardInterrupt
            elif isinstance(ch, str) and ch.isprintable():
                buf.append(ch)

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
            elif cmd == '/help':
                tui.add_message('sys', HELP_TEXT)
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

        def token_cb(token):
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