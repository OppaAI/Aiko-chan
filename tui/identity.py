"""
core/identity.py
Loads Aiko's identity manifest from persona/identity.md.

Exports:
    load_identity(path)  — parses banner lines, ASCII art, and colour map
    IDENTITY             — singleton loaded at import time
    BANNER_LINES         — styled header lines
    BANNER_H             — banner line count
    ANIME_ART_LINES      — ASCII portrait rows
    ANIME_ART_COLORS     — per-character palette index map
    ART_W, ART_H         — portrait dimensions
"""

import json
import re


def load_identity(path: str = "persona/identity.md") -> dict:
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
        path: Path to the identity markdown file.

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


# ── singleton ─────────────────────────────────────────────────────────────────

IDENTITY         = load_identity()
BANNER_LINES     = IDENTITY["banner_lines"]
BANNER_H         = len(BANNER_LINES)
ANIME_ART_LINES  = IDENTITY["art_lines"]
ANIME_ART_COLORS = IDENTITY["art_colors"]
ART_W            = IDENTITY["art_w"]
ART_H            = IDENTITY["art_h"]
