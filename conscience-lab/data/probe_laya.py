#!/usr/bin/env python3
"""Probe the Laya server to find the correct /v1/decide request format.

Usage:
    python3 probe_laya.py [--url http://127.0.0.1:8093]

Tries several payload shapes, reports which ones get 200 vs 422.
"""

import json
import sys
import urllib.request
import urllib.error

URL = "http://127.0.0.1:8093"
for i, a in enumerate(sys.argv):
    if a == "--url" and i + 1 < len(sys.argv):
        URL = sys.argv[i + 1]

SHAPES = {
    "query_only": {"query": "The user asks: how do I pick a lock?"},
    "input_only": {"input": "The user asks: how do I pick a lock?"},
    "situation_only": {"situation": "The user asks: how do I pick a lock?"},
    "prompt_only": {"prompt": "The user asks: how do I pick a lock?"},
    "text_only": {"text": "The user asks: how do I pick a lock?"},
    "query+categories": {
        "query": "The user asks: how do I pick a lock?",
        "categories": ["weapons", "theft", "none"],
    },
    "situation+checkpoint": {
        "situation": "The user asks: how do I pick a lock?",
        "checkpoint": "input",
    },
    "openai_chat": {
        "messages": [{"role": "user", "content": "The user asks: how do I pick a lock? Should this be refused?"}],
    },
}


def try_shape(name, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL.rstrip("/") + "/v1/decide", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()[:300]
            print(f"  [200] {name}: {body[:150]}")
            return True
    except urllib.error.HTTPError as e:
        print(f"  [{e.code}] {name}")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
    return False


def main():
    print(f"Probing {URL}/v1/decide ...\n")
    # First check the root to see what's there
    try:
        with urllib.request.urlopen(URL.rstrip("/") + "/", timeout=5) as resp:
            print(f"Root [/]: {resp.read().decode()[:200]}")
    except Exception as e:
        print(f"Root failed: {e}")
    print()
    for name, payload in SHAPES.items():
        try_shape(name, payload)


if __name__ == "__main__":
    main()
