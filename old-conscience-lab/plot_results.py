#!/usr/bin/env python3
"""Render the ablation results as a self-contained SVG.

The decisive chart is harm caught against benign interrupted. Every
configuration we have is a trade-off between those two, and the useful
question is not "which number is highest" but "who is closest to the
bottom-right corner" -- high recall with low interruption. A config that wins
one axis while losing the other badly is not a win.

Hand-rolled SVG rather than matplotlib: no dependency to install on either
host, byte-deterministic output, and the result is text so it diffs in git.

Usage:
    python plot_results.py                      # read /tmp/ablate_*.json
    python plot_results.py --out docs/conscience-ablation.svg \\
                          --title "Blind holdout"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib

W, H = 900, 560
PAD_L, PAD_R, PAD_T, PAD_B = 70, 250, 50, 60
COLORS = {
    "3x2": "#2563eb", "v5": "#dc2626", "base": "#059669", "none": "#7c3aed",
}
GLYPH = {"3x2": "circle", "v5": "square", "base": "triangle", "none": "diamond"}


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f)):
            o = r.get("oppa") or {}
            if not o.get("harm_n"):
                continue
            rows.append({
                "id": r["id"], "model": r["model"],
                "l0": r.get("l0"), "l1a": r.get("l1a"),
                "sem": r.get("semantic"),
                "caught": 100.0 * (o["harm_total"] - o["harm_missed"]) / max(o["harm_total"], 1),
                "interrupted": 100.0 * o["benign_interrupted"] / max(o["benign_total"], 1),
                "p50": o.get("lat_p50"), "n": o.get("n"),
                "harm_n": o.get("harm_n"), "benign_n": o["benign_total"],
                "src": f,
            })
    # one point per config: keep the best harm-caught variant of each (model, layers)
    best = {}
    for r in rows:
        key = (r["model"], r["l0"], r["l1a"])
        if key not in best or r["caught"] > best[key]["caught"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: (not r["l0"], -r["caught"]))


def label(r: dict) -> str:
    bits = [r["model"] if r["model"] != "none" else "no judge"]
    bits.append("L0+L1+L2" if r["l0"] else "L2 only")
    if r["sem"]:
        bits.append("semantic")
    return " · ".join(bits)


def render(rows: list[dict], title: str, subtitle: str) -> str:
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def X(caught):   # 100% caught -> right
        return PAD_L + pw * (1.0 - caught / 100.0)

    def Y(interr):   # 0% interrupted -> bottom
        return PAD_T + ph * (interr / 100.0)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{PAD_L}" y="26" font-size="16" font-weight="600" fill="#111827">{esc(title)}</text>',
         f'<text x="{PAD_L}" y="43" font-size="11.5" fill="#6b7280">{esc(subtitle)}</text>']

    # the desirable quadrant
    p.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw/2:.1f}" height="{ph/2:.1f}" '
             f'fill="#ecfdf5" opacity="0.75"/>')
    p.append(f'<text x="{PAD_L+8}" y="{PAD_T+15}" font-size="10.5" fill="#047857" '
             f'font-weight="600">better: high catch, low interruption</text>')

    # gridlines
    for v in range(0, 101, 25):
        x = X(v)
        p.append(f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T+ph}" '
                 f'stroke="#e5e7eb" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{PAD_T+ph+18}" font-size="10.5" fill="#6b7280" '
                 f'text-anchor="middle">{v}%</text>')
        y = Y(v)
        p.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+pw}" y2="{y:.1f}" '
                 f'stroke="#e5e7eb" stroke-width="1"/>')
        p.append(f'<text x="{PAD_L-9}" y="{y+3.5:.1f}" font-size="10.5" fill="#6b7280" '
                 f'text-anchor="end">{v}%</text>')

    p.append(f'<line x1="{PAD_L}" y1="{PAD_T+ph}" x2="{PAD_L+pw}" y2="{PAD_T+ph}" '
             f'stroke="#9ca3af" stroke-width="1.2"/>')
    p.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+ph}" '
             f'stroke="#9ca3af" stroke-width="1.2"/>')
    p.append(f'<text x="{PAD_L+pw/2:.1f}" y="{H-16}" font-size="11.5" fill="#374151" '
             f'text-anchor="middle">harm caught &#8594;</text>')
    p.append(f'<text x="18" y="{PAD_T+ph/2:.1f}" font-size="11.5" fill="#374151" '
             f'text-anchor="middle" transform="rotate(-90 18 {PAD_T+ph/2:.1f})">'
             f'benign interrupted &#8594;</text>')

    # points, nudged apart when they coincide
    seen: dict[tuple, int] = collections.defaultdict(int)
    for r in rows:
        x, y = X(r["caught"]), Y(r["interrupted"])
        k = (round(x), round(y))
        seen[k] += 1
        off = (seen[k] - 1) * 13
        x += off * 0.7
        y -= off * 0.7
        c = COLORS.get(r["model"], "#6b7280")
        g = GLYPH.get(r["model"], "circle")
        if g == "square":
            p.append(f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" '
                     f'fill="{c}" stroke="#fff" stroke-width="1.5"/>')
        elif g == "triangle":
            p.append(f'<polygon points="{x:.1f},{y-7:.1f} {x+6.5:.1f},{y+5:.1f} '
                     f'{x-6.5:.1f},{y+5:.1f}" fill="{c}" stroke="#fff" stroke-width="1.5"/>')
        elif g == "diamond":
            p.append(f'<polygon points="{x:.1f},{y-7:.1f} {x+7:.1f},{y:.1f} '
                     f'{x:.1f},{y+7:.1f} {x-7:.1f},{y:.1f}" fill="{c}" '
                     f'stroke="#fff" stroke-width="1.5"/>')
        else:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{c}" '
                     f'stroke="#fff" stroke-width="1.5"/>')
        r["_x"], r["_y"] = x, y

    # legend / detail column
    ly = PAD_T + 4
    p.append(f'<text x="{PAD_L+pw+22}" y="{ly}" font-size="11" font-weight="700" '
             f'fill="#111827">configurations</text>')
    ly += 17
    for r in rows:
        c = COLORS.get(r["model"], "#6b7280")
        p.append(f'<circle cx="{PAD_L+pw+28}" cy="{ly-3.5:.1f}" r="4.5" fill="{c}"/>')
        p.append(f'<text x="{PAD_L+pw+39}" y="{ly}" font-size="10.5" fill="#111827">'
                 f'{esc(label(r))}</text>')
        ly += 14
        p.append(f'<text x="{PAD_L+pw+39}" y="{ly}" font-size="10" fill="#6b7280">'
                 f'caught {r["caught"]:.0f}% &#183; interrupted {r["interrupted"]:.0f}%'
                 f' &#183; p50 {r["p50"]}ms</text>')
        ly += 18
    p.append("</svg>")
    return "\n".join(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/tmp/ablate_agent_cases*.json")
    ap.add_argument("--out", default="docs/conscience-ablation.svg")
    ap.add_argument("--title", default="Conscience ladder: harm caught vs benign interrupted")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    rows = load(args.glob)
    if not rows:
        print(f"no rows matched {args.glob}; run conscience-lab/ablate.py first")
        return 1
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = args.subtitle or (
        f"{len(rows)} configurations · lower-right is better · "
        "development metrics, not a held-out test")
    out.write_text(render(rows, args.title, sub), encoding="utf-8")
    print(f"wrote {out}  ({len(rows)} configurations)")
    for r in rows:
        print(f"  {label(r):42s} caught {r['caught']:5.1f}%  interrupted {r['interrupted']:5.1f}%"
              f"  p50 {r['p50']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())