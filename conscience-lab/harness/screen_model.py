#!/usr/bin/env python3
"""Phase 1 first cut: memory + runtime smoke check for a candidate model.

    python3 screen_model.py --gguf /models/laya-q8.gguf
    python3 screen_model.py --gguf /models/laya-q8.gguf --server http://jetson:8080

Checks, in order (fastest cut first):
  1. file size under --budget-mb (default 857) — drop what fails
  2. if --server: one smoke inference through the server, reporting
     latency — this is the "runs on the Jetson without a missing CUDA
     kernel" check. --backend llama targets a llama.cpp server
     (/props + chat/completions); --backend laya targets `laya serve`
     (/v1/decide typed choice).

Exit code 0 = passed, 1 = failed a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--budget-mb", type=float, default=857)
    ap.add_argument("--server", default="",
                    help="server base URL, e.g. http://jetson:8080")
    ap.add_argument("--backend", choices=["llama", "laya"], default="llama",
                    help="llama = llama.cpp server (/props + chat/completions); "
                         "laya = `laya serve` (/v1/decide smoke)")
    ap.add_argument("--model", default="")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    size_mb = os.path.getsize(args.gguf) / (1024 * 1024)
    print(f"size: {size_mb:.0f} MB (budget {args.budget_mb:.0f} MB)")
    if size_mb > args.budget_mb:
        print("FAIL: over memory budget — dropped")
        return 1
    print("PASS: size")

    if args.server:
        if args.backend == "laya":
            return _laya_smoke(args)

        props_url = args.server.rstrip("/") + "/props"
        if args.model:
            props_url += "?" + urllib.parse.urlencode({"model": args.model})
        try:
            with urllib.request.urlopen(props_url, timeout=args.timeout) as resp:
                props = json.loads(resp.read().decode())
            model_path = props.get("model_path")
            if (not isinstance(model_path, str) or not os.path.isabs(model_path)
                    or os.path.normpath(model_path) != os.path.abspath(args.gguf)):
                raise ValueError(f"server model_path {model_path!r} does not match "
                                 f"candidate {os.path.abspath(args.gguf)!r}")
        except Exception as e:
            print(f"FAIL: server model verification error: {e}")
            return 1

        prompt = ("You are a content safety classifier. Answer with exactly "
                  "the category name or \"none\".\n\nText: The user asks: "
                  "what is 18 percent of 240?\nAnswer:")
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 16, "temperature": 0}
        if args.model:
            payload["model"] = args.model
        url = args.server.rstrip("/") + "/v1/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty or non-text response content")
        except Exception as e:
            print(f"FAIL: server inference error: {e}")
            return 1
        ms = (time.time() - t0) * 1000
        print(f"smoke inference: {ms:.0f} ms -> {text.strip()[:80]!r}")
        print("PASS: server inference")
    else:
        print("(no --server given: load the GGUF on the target box and point "
              "--server at it for the runtime check)")

    print("ALL CHECKS PASSED")
    return 0


def _laya_smoke(args):
    """One typed-choice inference through `laya serve` /v1/decide."""
    payload = {
        "state": "The user asks: what is 18 percent of 240?",
        "questions": {
            "smoke": {
                "type": "choice",
                "instructions": "Which bucket does the text belong to?",
                "criteria": {
                    "benign": "An ordinary harmless question.",
                    "weapons": "Instructions for building weapons.",
                },
            }
        },
    }
    url = args.server.rstrip("/") + "/v1/decide"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode())
        ans = data["smoke"]
        choice, conf = ans["choice"], ans.get("confidence")
        if choice not in ("benign", "weapons"):
            raise ValueError(f"unexpected choice {choice!r}")
    except Exception as e:
        print(f"FAIL: laya smoke inference error: {e}")
        return 1
    ms = (time.time() - t0) * 1000
    print(f"smoke inference: {ms:.0f} ms -> {choice} (confidence {conf})")
    print("PASS: server inference")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
