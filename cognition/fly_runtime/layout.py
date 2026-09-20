"""2D layout for MaleCNS catalog neurons — type-based positioning."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutNode:
    id: str
    type: str
    region: str
    x: float
    y: float
    group: str


GROUP_ORDER = [
    ("sensory", "Sensory", (0.08, 0.5)),
    ("AL", "Antennal Lobe", (0.18, 0.25)),
    ("T4", "T4/T5 Motion", (0.18, 0.75)),
    ("T5", "T4/T5 Motion", (0.18, 0.75)),
    ("Tm", "Medulla Tm", (0.28, 0.35)),
    ("Mi", "Medulla Mi", (0.28, 0.45)),
    ("C", "Medulla C", (0.28, 0.55)),
    ("L", "Lamina L", (0.28, 0.65)),
    ("MB", "Mushroom Body", (0.45, 0.2)),
    ("MBON", "MB Output", (0.5, 0.15)),
    ("KC", "Kenyon Cell", (0.5, 0.3)),
    ("DAN", "Dopaminergic", (0.55, 0.25)),
    ("APL", "MB Feedback", (0.55, 0.35)),
    ("CX", "Central Complex", (0.45, 0.7)),
    ("EPG", "CX Compass", (0.5, 0.65)),
    ("PEN", "CX Compass", (0.5, 0.75)),
    ("PFN", "CX Output", (0.55, 0.7)),
    ("LH", "Lateral Horn", (0.45, 0.5)),
    ("LHN", "LH Output", (0.5, 0.5)),
    ("GF", "Giant Fiber", (0.65, 0.5)),
    ("DN", "Descending", (0.8, 0.35)),
    ("DNp", "DN Projection", (0.8, 0.35)),
    ("DNg", "DN Ground", (0.8, 0.45)),
    ("VNC", "Ventral Nerve Cord", (0.85, 0.6)),
    ("output", "Motor Output", (0.9, 0.5)),
    ("unknown", "Other", (0.5, 0.5)),
]


def classify_group(node_type: str) -> str:
    t = node_type.lower()
    if t.startswith(("dn", "dng", "dnp")):
        return "DN"
    if t.startswith(("vnc", "vnc_")):
        return "VNC"
    if t.startswith("gf") or t in {"giant_fiber", "gf1", "gf2"}:
        return "GF"
    if t.startswith(("lh", "lhn")):
        return "LH"
    if t.startswith(("epg", "pen", "pfn", "pgn", "peg")):
        return "CX"
    if t.startswith(("mbon", "mbout")):
        return "MBON"
    if t.startswith(("kc", "kenyon", "cla")):
        return "KC"
    if t.startswith(("dan", "pam", "pp1", "pp2")):
        return "DAN"
    if t.startswith("apl"):
        return "APL"
    if t.startswith("mb") or t in {"mb", "mushroom_body"}:
        return "MB"
    if t.startswith(("t4", "t5")):
        return "T4"
    if t.startswith(("tm", "tmy")):
        return "Tm"
    if t.startswith(("mi", "mir")):
        return "Mi"
    if t.startswith(("c", "ct")) and len(t) <= 3:
        return "C"
    if t.startswith("l") and len(t) <= 3:
        return "L"
    if t.startswith("al") or t in {"al", "antennal_lobe"}:
        return "AL"
    if t.startswith(("sens", "phot", "ol", "oc", "ocg", "amc", "aotu", "vs", "hs", "ch", "dno")):
        return "sensory"
    if t.startswith("out") or t in {"motor", "output"}:
        return "output"
    return "unknown"


def group_center(group: str) -> tuple[float, float]:
    for g, _, center in GROUP_ORDER:
        if g == group:
            return center
    return (0.5, 0.5)


def compute_layout(nodes: list[dict], width: float = 1000.0, height: float = 600.0, seed: int = 42) -> list[LayoutNode]:
    """Assign 2D positions by type group with jitter for separation."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        grp = classify_group(n.get("type", "unknown"))
        by_group[grp].append(n)

    import random
    rng = random.Random(seed)

    layout: list[LayoutNode] = []
    for grp, members in by_group.items():
        cx, cy = group_center(grp)
        base_x = cx * width
        base_y = cy * height
        count = len(members)
        radius_x = 80 + min(count * 0.8, 120)
        radius_y = 60 + min(count * 0.6, 100)
        for i, n in enumerate(members):
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.15, 1.0)
            x = base_x + math.cos(angle) * radius_x * r
            y = base_y + math.sin(angle) * radius_y * r
            x = max(20, min(width - 20, x))
            y = max(20, min(height - 20, y))
            layout.append(LayoutNode(
                id=str(n["id"]),
                type=str(n.get("type", "unknown")),
                region=str(n.get("region", "unknown")),
                x=x, y=y,
                group=grp,
            ))
    return layout


GROUP_COLORS = {
    "sensory": "#48d8ff",
    "AL": "#4ec9b0",
    "Tm": "#5de4c7",
    "Mi": "#6effe0",
    "C": "#7ffff8",
    "L": "#8fffff",
    "MB": "#a88bff",
    "MBON": "#c4a8ff",
    "KC": "#d8c0ff",
    "DAN": "#e8d8ff",
    "APL": "#f0e8ff",
    "CX": "#ff78b7",
    "EPG": "#ff94c9",
    "PEN": "#ffb0db",
    "PFN": "#ffcced",
    "LH": "#ffbe62",
    "LHN": "#ffd08c",
    "GF": "#ff6b6b",
    "DN": "#ff78b7",
    "DNp": "#ff78b7",
    "DNg": "#ff94c9",
    "VNC": "#6bcbff",
    "output": "#ff78b7",
    "unknown": "#91a1ba",
}


def group_color(group: str) -> str:
    return GROUP_COLORS.get(group, "#91a1ba")