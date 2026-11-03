#!/usr/bin/env python3
import json, re, sys, urllib.request, urllib.error

URL = "http://127.0.0.1:8093"

def get(path):
    try:
        with urllib.request.urlopen(URL.rstrip("/") + path, timeout=10) as r:
            return r.status, r.read().decode()[:5000]
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return -1, str(e)

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()[:500]
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return -1, str(e)

print("=== 1. Studio HTML API hints ===")
status, html = get("/")
print(f"GET / -> {status}")
for pat in [r'fetch\([\'"]([^\'"]+)', r'["\'](/api/[^"\']+)', r'["\'](/v1/[^"\']+)']:
    hits = set(re.findall(pat, html, re.IGNORECASE))
    if hits: print(f"  {pat}: {hits}")

print("\n=== 2. Try API paths ===")
for path in ["/api/decide", "/decide", "/v1/complete", "/api/complete",
             "/complete", "/api/v1/decide", "/studio/api/decide"]:
    s, b = post(path, {"query": "test"})
    print(f"  POST {path} -> {s} {b[:80]}")

