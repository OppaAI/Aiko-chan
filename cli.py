"""
cli.py

Aiko-chan CLI — full-screen curses TUI, cyberpunk edition.
Usage:
    python cli.py               # normal chat
    python cli.py --no-voice    # disable TTS
    python cli.py --debug       # show memory debug info each turn
    python cli.py --clear-mem   # wipe all stored memories and exit

Layout:
    ╔══════════════════════════════════════════════════════════╗
    ║                      BANNER                              ║
    ╠══════════════╦═══════════════════════════════════════════╣
    ║  ASCII ART   ║  INIT LOG  /  ARCH INFO                   ║
    ║  (46 wide)   ╠═══════════════════════════════════════════╣
    ║              ║  CHAT MESSAGES                            ║
    ╠══════════════╩═══════════════════════════════════════════╣
    ║  STATUS BAR (full width)                                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  INPUT (full width)                                      ║
    ╚══════════════════════════════════════════════════════════╝
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

# ── ASCII art (left panel) ────────────────────────────────────────────────────
# Generated from PNG — cropped to remove blank left margin.
# 46 chars wide x 52 rows tall.
# Color palette indices per character:
#   0 = skip (black / transparent)
#   1 = 236  very dark grey
#   2 = 239  dark grey
#   3 = 243  mid-dark grey
#   4 = 246  mid grey
#   5 = 249  light grey
#   6 = 253  near-white
#   7 = 95   warm dark  (muted rose/brown — shadow clothing)
#   8 = 138  warm mid   (dusty rose — clothing/blossoms)
#   9 = 181  warm bright (light rose — bright clothing)

ART_W = 46
ART_H = 52

ANIME_ART_LINES = [
    "                             '<!(}]\\l)})}]]\\{\\",
    '                             `--+/{\\[]]t}?()l{',
    '                         .-:=,..!)\\}{[]{[{I([\\',
    '                         .- `- ^]{/(/?))//))|/',
    "                          .'`'?((|)\\(i/)Ii??i)",
    '           .              `=+~i{{(i|<?(|(||/{i',
    "     ``.       `-_`      `--'~I(|i!ii!l(\\I}}}\\",
    '    .`.   _,     .`. .  .-_,:!??>>??!>I|{II|(i',
    '    `    `_1= ..    ..    ^!!=~=<<<!++i!<=!+>=',
    "   `-   .< ?1,,!`   -.   ` .':^^<<<==,;~^i|>^=",
    "   `.', ._+[rj{^(;  ..   `-:__;~+<<!~=+<?i!<=+",
    '    .>/` _]jrrrj1`   `.   ``_.  ---,^<<>=~^^^,',
    "    .<c1+ =Jj]cc>  `.`..        .-:~^;,',;-:,=",
    "      :/x)`It](; `.`..         ..':::'``'-_-',",
    "    -;;_\\c)/l?: .              .'`` -` `'.-`';",
    '   ,ilI!\\c[[)(\\<              .......  ._..._,',
    "   ><~=+|}|~i{I!~+_              .-.-  ._ . _'",
    "   ~?i?=!!?!</t}}!?!-            ```-  .'   ,,",
    '   ^~=<!!ili>>jj/[|?+_-          -```` .:   ::',
    "   `_,^~<lii>>itJ]J1<l1[~.       ``.-` `'   ''",
    "    `;'_:^;:;??>]}1r]~+??_ .    ._.--- -_   :,",
    "     ::''''^?il?+Ij\\\\~:`-^~=    -_ --|,`'   :;",
    "     ^;:_:+lii!l!<[jI';^^ili,   -` ':~.`_   ,;",
    "    .~:'=?liii,<l<iJ/'^~::ll]- .-..''` .:   ,;",
    '     :~=?i?il<,,?i>l\\;^^^^^ic) -_`--`` .:   ;;',
    '     ~+!iiiil~;,=???[i:^^~:~tc;.-`-`_` .:   ;;',
    "    `+l?i?!?i=^^<!!{[]<^~^^_I]<.` ..-. `'.  ,:",
    "    ,l++!<>>!!!i(|/][\\?,;^;;'..-`.---. `'   ;_",
    "    -;i/{}l}\\{i!{|\\()>=~;:`.   ``.-_`.`-'   ;'",
    '     lJrjc{r[r]?I[)[|\\c1{(~    `--`---`-`   ;:',
    "    ;J(trjt]]{cj[]!+\\|I[t]I<_  _'---`-`-`   ,:",
    "   _\\\\(Jrrr][|jrjt)+!/]/i{c{[);` ````..`-   ':",
    "  ~)(!1jttj]}|rttjc})i}j[(]J\\rJ),..    `_   :'",
    ' _r)l>rtttt]}ttrt{II/r>??l{1}{/(l.     .`   :_',
    " /[>I>>|))I+{j1}/(?</\\I))(/!l]cr)'.``. .`   _,",
    '.<){<!itjrj{ili{ccJ(i|?^li>(r)())+      .   _:',
    " ^t1lI?}{{[1{1Il)??+!!~+|||il>~~:       `   ',",
    ' ^(?!?++<<=+[r(=^!il)\\i!)((I+``.  .     `   :,',
    "  ,^>\\{[}/|<:=>I[]tt1j~.::-. ..-'-`-.  .-   ,;",
    '    |1}}[[t> .^1[[}}1(         `   ..   -   ;^',
    '   ./}}}}}{-  ~1}[[}],         `       .-   ;=',
    '   `{[}}}1<   <}{[}1I          `  ..    `   ^+',
    '   _}[[}[\\.   ?{[[[]:       ...`.  .  -^`   ;+',
    "   ;t[[[1=    (1][t|           .--'.  -:-   ;+",
    '   +t][]{.   .}1[[1;           .. ,:`  ._   ;=',
    '   (1[[t+    .}1[]}.           ```^~:  --   ,=',
    '  ~1[[1\\.     l1}][-           .`.=;-  `-   ,=',
    " '[[{]\\_      =1[[]:              _'_  `-   ,~",
    ' (]}[1~       ?1[[1^        .    . _~. -.`  ,~',
    ",][[]\\.       |1][1;  ...   ..     .'. -!x(`:^",
    "!]}}]=        )]}}}'                   `:!, :;",
    '|]}]{`       -}[][].                   ``   :^',
]

ANIME_ART_COLORS = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7,2,3,3,3,3,2,3,3,3,3,3,3,3,3,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,3,3,3,3,3,3,4,3,2,3,3,2,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,0,0,2,3,3,3,3,3,3,3,3,3,8,8,3,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,3,3,3,3,3,2,3,3,3,3,3,3,8,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2,8,8,8,8,3,3,2,3,8,8,8,8,8,8,3],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,3,3,3,2,3,2,2,8,8,8,8,3,3,3,8],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,2,8,8,8,8,8,8,8,8,8,8,3,2,3,3,3,3],
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,8,8,8,7,7,8,8,7,7,8,3,3,8,8,8,8,8],
    [0,0,0,0,0,0,0,0,0,0,1,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7,7,7,1,7,7,7,7,7,7,7,8,8,2,7,7,7,7,7],
    [0,0,0,0,0,0,0,0,0,2,0,2,4,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,7,7,7,7,7,7,7,7,7,8,8,7,7,7],
    [0,0,0,0,0,1,1,0,0,1,2,3,4,4,3,1,3,1,0,0,0,0,0,0,0,0,1,1,1,1,1,2,7,7,7,7,7,2,7,7,8,8,7,7,7,7],
    [0,0,0,0,0,2,3,0,0,1,3,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,7,7,7,7,7,7,1,7,7,1],
    [0,0,0,0,0,2,4,3,2,0,2,4,4,3,4,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,7],
    [0,0,0,0,0,0,1,3,4,3,0,2,4,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1],
    [0,0,0,0,1,7,7,1,3,4,3,3,8,8,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,7],
    [0,0,0,1,8,8,8,7,3,4,3,3,3,3,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,7,7,7,7,7,3,8,3,7,8,3,2,2,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,7,8,8,8,7,7,7,8,7,7,3,4,3,3,7,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1],
    [0,0,0,7,7,7,7,7,8,8,8,8,7,7,4,4,3,3,8,8,2,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,1,7,7,7,7,8,8,8,7,7,8,4,4,3,4,3,2,2,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,7,1,1,7,7,7,7,7,8,8,7,3,3,3,4,3,7,2,2,2,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,1,7,7,1,1,1,7,8,8,8,8,7,2,4,3,3,7,7,0,1,1,1,2,0,0,0,0,0,1,0,1,0,3,1,0,1,0,0,0,1,1],
    [0,0,0,0,0,1,7,7,1,1,7,8,8,8,7,8,7,7,3,4,2,1,7,7,7,8,2,8,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,7,7,1,7,8,8,8,8,8,7,7,8,7,8,4,3,1,7,7,7,1,2,8,3,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,1,7,7,8,8,8,8,8,7,7,7,8,8,7,8,3,7,7,7,7,7,7,2,4,3,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,7,7,8,8,8,8,8,8,7,7,7,7,8,8,8,3,2,7,7,7,7,7,7,4,4,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,0,7,8,8,8,8,8,8,8,7,7,7,7,7,8,3,3,3,2,7,7,7,7,7,2,3,2,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,1,8,7,7,7,7,7,7,7,7,8,8,8,3,3,3,3,3,2,7,7,7,7,7,1,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,1,7,2,3,3,3,2,3,3,3,2,2,3,3,3,3,3,2,2,1,7,7,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1],
    [0,0,0,0,0,2,4,4,4,4,3,4,3,4,3,2,2,3,3,3,3,3,4,3,3,3,2,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,1,4,3,4,4,4,4,3,3,3,4,4,3,3,2,2,3,3,2,3,4,3,2,2,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,1,3,3,3,4,4,4,4,3,3,3,4,4,4,4,3,2,2,3,3,3,2,3,4,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,1,3,3,2,3,4,4,4,4,3,3,3,4,4,4,4,4,3,3,2,3,4,3,3,3,4,3,4,4,3,1,0,0,0,0,0,0,0,1,0,0,0,1,1],
    [0,1,4,3,2,2,4,4,4,4,4,3,3,4,4,4,4,3,2,2,3,4,2,2,2,2,3,3,3,3,3,3,2,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,3,3,2,2,2,2,3,3,3,2,2,3,4,3,3,3,3,2,2,3,3,2,3,3,3,3,2,2,3,4,4,3,1,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,2,3,3,2,2,2,4,4,4,4,3,2,2,2,3,4,4,4,3,2,3,2,1,2,2,2,3,4,3,3,3,3,2,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,1,4,3,2,2,2,3,3,3,3,3,3,3,2,2,3,8,8,2,2,2,7,2,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,1,3,2,2,2,2,7,7,7,7,2,3,4,3,2,7,7,8,2,3,3,2,2,3,3,3,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,1,1,2,3,3,3,3,3,8,7,1,2,2,2,3,3,4,4,3,4,2,0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1],
    [0,0,0,0,3,4,3,3,3,3,4,2,0,0,1,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,3,3,3,3,3,3,3,0,0,0,1,3,3,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2],
    [0,0,0,0,3,3,3,3,3,3,2,0,0,0,2,3,3,3,3,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2],
    [0,0,0,1,3,3,3,3,3,3,0,0,0,0,2,3,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,2],
    [0,0,0,1,4,3,3,3,4,2,0,0,0,0,3,4,3,3,4,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,2],
    [0,0,0,2,4,3,3,3,3,0,0,0,0,0,3,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,2],
    [0,0,0,3,3,3,3,4,2,0,0,0,0,0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,2],
    [0,0,1,3,3,3,3,3,0,0,0,0,0,0,2,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,1,0,0,0,1,2],
    [0,1,3,3,3,3,3,1,0,0,0,0,0,0,2,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1],
    [0,3,3,3,3,3,1,0,0,0,0,0,0,0,2,4,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1],
    [1,3,3,3,3,3,0,0,0,0,0,0,0,0,3,4,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,2,4,3,0,1,1],
    [2,3,3,3,3,2,0,0,0,0,0,0,0,0,3,3,3,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,0,1,1],
    [3,3,3,3,3,0,0,0,0,0,0,0,0,0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
]

# Palette: index → terminal 256-color number (or None to skip)
ART_PALETTE = [None, 236, 239, 243, 246, 249, 253, 95, 138, 181]

LEFT_W = ART_W + 2   # art + left/right border chars = 48

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

# Precompute how many rows arch needs
ARCH_ROWS = sum(1 + len(items) for _, items in ARCH_SECTIONS) + 2  # header + rule + sections

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

SPINNER   = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
HELP_TEXT = "/quit /exit — end  │  /reset — clear context  │  /memory — show memories  │  /help"

# ── colour pairs ──────────────────────────────────────────────────────────────
# Pairs 1-8: UI chrome
# Pairs 9-17: art palette (one per non-None palette entry)

CP_PINK    = 1   # 198 hot pink    — borders, banner, accent
CP_CYAN    = 2   # 51  cyan        — You text, init spinner
CP_PURPLE  = 3   # 135 purple      — section headers, dividers
CP_MAUVE   = 4   # 177 mauve       — arch values, sub-labels
CP_DIM     = 5   # 240 dim grey    — detail text, timestamps
CP_WHITE   = 6   # 15  white       — done ticks
CP_SBARBG  = 7   # black on pink   — status bar
CP_INPUTBG = 8   # cyan on dark    — input line

# Art palette pairs: index 1..8 of ART_PALETTE → curses pair CP_ART_BASE+idx
CP_ART_BASE = 8   # pairs 9..16

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
    # Art palette (skip index 0 which is None)
    for idx, color256 in enumerate(ART_PALETTE):
        if color256 is not None:
            curses.init_pair(CP_ART_BASE + idx, color256, -1)

def _art_attr(palette_idx: int) -> int:
    """Return curses attr for a palette index (0 = skip → -1)."""
    if palette_idx == 0:
        return -1
    attr = curses.color_pair(CP_ART_BASE + palette_idx)
    # Add bold for brighter/warmer entries
    if palette_idx in (5, 6, 9):  # light grey, near-white, warm bright
        attr |= curses.A_BOLD
    return attr


# ─────────────────────────────────────────────────────────────────────────────
# AikoTUI
# ─────────────────────────────────────────────────────────────────────────────

class AikoTUI:
    """
    Layout rows:
      0               : top border
      1..BANNER_H     : banner
      BANNER_H+1      : banner-bot / panel-top divider   ╠═══╦═══╣
      PT..PT+ART_H-1  : left=art | right=init or arch    ║   ║   ║
      PT+ART_H        : arch-bot / chat-top separator    ║   ╠═══╣
      PT+ART_H+1..CB  : left=art-cont (blank) | right=chat
      CB              : separator                         ╠═══════╣
      CB+1            : status bar (full width, pink bg)
      CB+2            : separator                         ╠═══════╣
      CB+3            : input (full width)
      CB+4            : bottom border
    """

    INPUT_H = 1
    SBAR_H  = 1
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
    def _pt(self):
        """Panel top row (first row inside the art/right-panel area)."""
        return BANNER_H + 2   # row 0 border + BANNER_H rows + 1 divider

    @property
    def _arch_sep_row(self):
        """Row of the horizontal line separating arch from chat."""
        return self._pt + ARCH_ROWS

    @property
    def _chat_top(self):
        return self._arch_sep_row + 1

    def _chat_bot(self, h):
        """Exclusive bottom of chat area."""
        return h - self.SBAR_H - self.INPUT_H - 3  # sep + sbar + sep + input + bot-border

    def _dims(self):
        return self._scr.getmaxyx()

    # ── low-level helpers ─────────────────────────────────────────────────────

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

        # divider with column split
        self._wr(BANNER_H+1, 0,
            '╠' + '═'*(LEFT_W-1) + '╦' + '═'*(w-LEFT_W-2) + '╣', pk)

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW LEFT COLUMN (art, full height from panel top to chat bot)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_left_col(self, h, w):
        pk = curses.color_pair(CP_PINK) | curses.A_BOLD
        pt = self._pt
        cb = self._chat_bot(h)

        # Draw the art rows
        for i in range(ART_H):
            row = pt + i
            if row > cb:
                break
            art_line = ANIME_ART_LINES[i] if i < len(ANIME_ART_LINES) else ''
            clr_row  = ANIME_ART_COLORS[i] if i < len(ANIME_ART_COLORS) else []

            self._wr(row, 0, '║', pk)
            x = 1
            for col_idx, ch in enumerate(art_line):
                if x >= LEFT_W:
                    break
                pidx  = clr_row[col_idx] if col_idx < len(clr_row) else 0
                attr  = _art_attr(pidx)
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
            # Pad remaining art columns
            while x < LEFT_W:
                try:
                    self._scr.addch(row, x, ' ')
                except curses.error:
                    pass
                x += 1

        # Rows below art (blank, still inside left column)
        for row in range(pt + ART_H, cb + 1):
            self._wr(row, 0, '║', pk)
            try:
                self._scr.move(row, 1)
                # clear to the divider
                self._scr.addstr(row, 1, ' ' * (LEFT_W - 1))
            except curses.error:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW RIGHT COLUMN (init/arch on top, chat below)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_right_col(self, h, w):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        pt  = self._pt
        sep = self._arch_sep_row
        cb  = self._chat_bot(h)

        rx  = LEFT_W + 1          # first content col (after ║)
        rw  = w - LEFT_W - 2      # usable width (before right border ║)

        # Right border for the whole column
        for row in range(pt, cb + 1):
            self._wr(row, LEFT_W, '║', pk)
            self._wr(row, w-1,    '║', pk)

        # Clear right panel area
        for row in range(pt, sep):
            try:
                self._scr.move(row, rx)
                self._scr.addstr(row, rx, ' ' * (rw - 1))
            except curses.error:
                pass

        # ── top section: init or arch ─────────────────────────────────────────
        if self._phase == 'init':
            self._draw_right_init(pt, sep, rx, rw)
        else:
            self._draw_right_arch(pt, sep, rx, rw)

        # ── separator between arch and chat ───────────────────────────────────
        self._wr(sep, 0,
            '║' + ' '*(LEFT_W-1) +
            '╠' + '═'*(w-LEFT_W-2) + '╣', pk)

        # ── chat section ──────────────────────────────────────────────────────
        self._draw_chat_area(sep + 1, cb, rx, rw, w)

    # ── right panel: init log ─────────────────────────────────────────────────

    def _draw_right_init(self, pt, pb, rx, rw):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)   | curses.A_BOLD
        pu  = curses.color_pair(CP_PURPLE) | curses.A_BOLD
        dim = curses.color_pair(CP_DIM)
        wh  = curses.color_pair(CP_WHITE)  | curses.A_BOLD

        self._wr(pt,   rx, " INITIALISING NEURAL SYSTEMS", pk)
        self._wr(pt+1, rx, ' ─' * min(16, (rw-1)//2), pu)

        row = pt + 2
        for (key, state, detail) in self._init_log:
            if row >= pb:
                break
            lbl, dflt = INIT_STEPS.get(key, (key, detail))
            txt = detail if detail else dflt

            if state == 'loading':
                sp = SPINNER[self._frame]
                self._wr(row, rx,    f"  {sp} ", cy)
                self._wr(row, rx+4,  f"{lbl:<20}", cy)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'done':
                self._wr(row, rx,    "  ✓ ", wh)
                self._wr(row, rx+4,  f"{lbl:<20}", wh)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'skip':
                self._wr(row, rx,    "  – ", dim)
                self._wr(row, rx+4,  f"{lbl:<20}", dim)
                self._wr(row, rx+24, txt[:rw-25], dim)
            elif state == 'error':
                self._wr(row, rx,    "  ✗ ", pk)
                self._wr(row, rx+4,  f"{lbl:<20}", pk)
                self._wr(row, rx+24, txt[:rw-25], curses.color_pair(CP_MAUVE))
            row += 1

        all_fin = (len(self._init_log) > 0 and
                   all(s in ('done','skip','error') for (_,s,_) in self._init_log))
        if all_fin and row < pb - 1:
            self._wr(row+1, rx, "  [ ALL SYSTEMS ONLINE ]",
                     curses.color_pair(CP_CYAN)|curses.A_BOLD)

    # ── right panel: arch ─────────────────────────────────────────────────────

    def _draw_right_arch(self, pt, pb, rx, rw):
        pk  = curses.color_pair(CP_PINK)   | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)   | curses.A_BOLD
        pu  = curses.color_pair(CP_PURPLE) | curses.A_BOLD
        mv  = curses.color_pair(CP_MAUVE)
        dim = curses.color_pair(CP_DIM)

        self._wr(pt,   rx, " NEURAL ARCHITECTURE", cy)
        self._wr(pt+1, rx, ' ─' * min(16, (rw-1)//2), pu)

        row = pt + 2
        for section, items in ARCH_SECTIONS:
            if row >= pb - 1:
                break
            self._wr(row, rx, f"  {section}", pk)
            row += 1
            for name, val in items:
                if row >= pb:
                    break
                self._wr(row, rx,     f"    {name:<18}", dim)
                self._wr(row, rx+22,  val[:rw-23],       mv)
                row += 1

    # ── chat area ─────────────────────────────────────────────────────────────

    def _render_lines(self, rw):
        avail = rw - 2
        out   = []
        for sender, text in self._messages:
            if sender == 'you':
                pre   = " You ❯  "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('Y', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('Y', ind + l))
            elif sender == 'aiko':
                pre   = " Aiko ♡  "
                ind   = " " * len(pre)
                lines = textwrap.wrap(text, avail - len(pre)) or [""]
                out.append(('A', pre + lines[0]))
                for l in lines[1:]:
                    out.append(('A', ind + l))
                out.append(('S', ''))
            elif sender == 'sys':
                out.append(('S', f"  ◈  {text}"))
        if self._streaming:
            pre   = " Aiko ♡  "
            ind   = " " * len(pre)
            lines = textwrap.wrap(self._streaming, avail - len(pre)) or [""]
            out.append(('A', pre + lines[0]))
            for l in lines[1:]:
                out.append(('A', ind + l))
        return out

    def _draw_chat_area(self, ct, cb, rx, rw, w):
        pk  = curses.color_pair(CP_PINK)  | curses.A_BOLD
        cy  = curses.color_pair(CP_CYAN)  | curses.A_BOLD
        pk_ = curses.color_pair(CP_PINK)
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
            attr = cy if kind == 'Y' else (pk_ if kind == 'A' else dim)
            # Clear the right panel portion of this row
            try:
                self._scr.addstr(row, rx, ' ' * (rw - 1))
            except curses.error:
                pass
            self._wr(row, rx, line[:rw-1], attr)
            self._wr(row, w-1, '║', pk)

        # Clear empty rows
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
    # STATUS BAR + INPUT  (full screen width)
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_chrome(self, h, w, buf):
        pk   = curses.color_pair(CP_PINK)   | curses.A_BOLD
        sbar = curses.color_pair(CP_SBARBG) | curses.A_BOLD
        inp  = curses.color_pair(CP_INPUTBG)

        cb = self._chat_bot(h)

        # chat-area bottom border (full width)
        self._wr(cb, 0, '╠' + '═'*(LEFT_W-1) + '╩' + '═'*(w-LEFT_W-2) + '╣', pk)

        # status bar (full width)
        sr      = cb + 1
        elapsed = int(time.time() - self._ts)
        left    = f"  ✦ {OLLAMA_MODEL}  │  mem0·Qdrant  │  Kokoro TTS  │  {SESSION_ID}  "
        right   = f"  {elapsed}s  "
        bar     = left + ' '*max(0, w - len(left) - len(right)) + right
        self._wr(sr, 0, bar[:w], sbar)

        # separator above input (full width)
        ir = sr + 1
        self._wr(ir, 0, '╠' + '═'*(w-2) + '╣', pk)

        # input line (full width)
        inp_r   = ir + 1
        content = self.INPUT_PROMPT + ''.join(buf)
        line    = content[:w-1].ljust(w-1)
        try:
            self._scr.move(inp_r, 0)
            self._scr.clrtoeol()
        except curses.error:
            pass
        self._wr(inp_r, 0, line, inp)

        # bottom border
        bot_r = inp_r + 1
        if bot_r < h:
            self._wr(bot_r, 0, '╚' + '═'*(w-2) + '╝', pk)

        # cursor
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
            min_h = BANNER_H + 2 + ART_H + 8
            if h < min_h or w < 90:
                self._wr(0, 0, f"Terminal too small: {w}x{h} (need 90x{min_h})", 0)
                self._scr.refresh()
                return
            self._draw_banner(h, w)
            self._draw_left_col(h, w)
            self._draw_right_col(h, w)
            if self._phase == 'chat':
                self._draw_chrome(h, w, buf if buf is not None else [])
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