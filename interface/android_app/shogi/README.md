# Shogi: human games + Aiko self-play vs YaneuraOu

`games_shogi.py` serves human-vs-AI over `/api/games/shogi` (existing).
`selfplay.py` (new) lets **Aiko play YaneuraOu by herself** and learn.

## Setup (Jetson)

1. **API key** — Aiko decides via Jev. Add it without touching disk plaintext:
   ```bash
   ./util/edit_dotenv.sh        # append: JEV_API_KEY=<key from typesafe.ai>
   ```
   Never paste the key in chat/logs; the client only ever sends it as a
   Bearer header. Optional: `JEV_MODEL` (default `jev-latest`).
2. **Engine** — ARM build with NN weights already in the tree:
   ```bash
   export YANEURAOU_PATH=/home/oppa-ai/jetson/YaneuraOu/source/YaneuraOu-by-gcc
   ```
   Start the server with cwd at the engine dir (or set EvalDir) so it finds
   `eval/nn.bin` next to the binary.
3. **Rules** — `python-shogi>=1.1.1` (already in pyproject deps).

## How a game works

Colors alternate per game. Aiko moves by: opening **book** first
(ε-greedy on her own won lines) → **Jev `choice`** over ≤10 capped
candidates (captures/checks first, recent loss lines in the rubric) →
random legal fallback (logged, game continues). YaneuraOu replies via USI.
Ends: checkmate, draw rules, engine resign, or 256-ply cap.

Budget honesty: Jev is called **once per Aiko move** (~50 calls/game).
`SELFPLAY_JEV_CANDIDATES` (default 10) bounds prompt size, not call count.

## How she learns (every finished game)

1. **Opening book** — `<userdir>/selfplay_book.json`: w/d/l per
   (position, move). This is the main strength engine over time.
2. **Match store** — namespaced `shogi_selfplay` table (human `shogi`
   stats untouched): winner, moves, opening, engine.
3. **Experience** — full game record (moves, result, score 1/0.5/0),
   searchable for later review/reflection.
4. **FlyMB teaching** — win +1 / loss −1 on the opening pattern
   (valence imprint; draws skipped).

Run one: `python -m interface.android_app.shogi.selfplay <uid> [games]`
(any exception aborts the game as void — never as a fake loss).
Tune: `SELFPLAY_MOVETIME_MS` (800), `SELFPLAY_MAX_MOVES` (256),
`SELFPLAY_BOOK_MIN_VISITS` (3), `SELFPLAY_EXPLORATION` (0.05).
