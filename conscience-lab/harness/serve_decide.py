#!/usr/bin/env python3
"""Minimal /v1/decide server for a locally trained LayaFT checkpoint.

The compiled `laya serve` (ggmlc, /v1/decide) is not on this box, and the
python `laya` package only serves /v1/systemone. The harness speaks
/v1/decide, so this shim maps it straight onto the checkpoint:

    POST /v1/decide {"state": ..., "questions": {...}}
      -> agent.predict(state, questions)  # {"answers": {qid: {...}}}

Usage:
    .venv-train/bin/python harness/serve_decide.py <checkpoint_dir> [--port 8094] [--device cuda]

Stdlib + fastapi/uvicorn + torch only. One inference worker (torch is sync).
"""
import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "data", "..")))
# import layaft from the training venv (this script runs under it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="layaft checkpoint dir (runs/<name>/)")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from layaft.model.load import load as load_agent
    print(f"loading {args.checkpoint} on {args.device} ...", flush=True)
    agent = load_agent(args.checkpoint, device=args.device)
    agent.model.eval()
    print("loaded.", flush=True)

    from fastapi import FastAPI, HTTPException, Request
    import uvicorn

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decide-infer")
    app = FastAPI(title="layaft-decide-shim")
    gate = None

    @app.get("/health")
    def health():
        return {"status": "ok", "checkpoint": args.checkpoint}

    @app.post("/v1/decide")
    async def decide(request: Request):
        nonlocal gate
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400,
                                detail="body must be an object with a 'questions' field")
        state, questions = body.get("state"), body["questions"]
        if gate is None:
            gate = asyncio.Lock()
        try:
            loop = asyncio.get_running_loop()
            async with gate:
                return await loop.run_in_executor(
                    pool, lambda: agent.predict(state, questions))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                detail=f"inference failed: {e!r}"[:300])

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
