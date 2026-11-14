#!/usr/bin/env python3
"""
cosmos_chat.py - multimodal chatbot for the Embedl Cosmos3-Edge server on this
Orin Nano. Talks to the OpenAI-compatible API served on the Embedl unix socket,
and can attach images and videos to messages. Uses Aiko's SOUL.md persona as
the system prompt.

Requires the Aiko venv (PIL for images, cv2 for video frame extraction):
  ~/Aiko-chan/.venv/bin/python3 ~/Aiko-chan/util/cosmos_chat.py

Usage:
  python cosmos_chat.py [--socket SOCK] [--model ID] [--max-tokens N]
                        [--soul PATH] [--user NAME] [--no-save]

In-chat commands:
  /img <path>      attach an image to the NEXT message (repeatable)
  /vid <path> [n]  attach a video (extracts n evenly spaced frames, default 6)
  /agent <task>    run an agentic tool loop (web_search, get_weather)
  /web <query>     quick web answer: search local SearXNG, summarize results
  /tools           list available agent tools
  /attachments     show queued attachments
  /clear-attach    drop queued attachments
  /reset           clear conversation history (and memory file)
  /stats           session totals: turns, latency, tokens
  /quit            exit

Each response prints latency and token usage, e.g.
  ⏱ 1.24s · tokens: 32 in / 12 out · 9.7 tok/s out

History is saved to chat_logs/cosmos_chat.json (text only - media parts are
stored as placeholders so the file stays small).
"""
import argparse
import base64
import datetime
import glob
import hashlib
import http.client
import inspect
import io
import json
import os
import re
import socket
import sys
import time
import urllib.parse
import urllib.request

import cv2
from PIL import Image

DIM, RESET, CYAN, YELLOW, GREEN, MAGENTA = (
    "\033[2m", "\033[0m", "\033[36m", "\033[33m", "\033[32m", "\033[35m")

MAX_IMG_DIM = 768        # longest edge px for image/frames before base64
JPEG_QUALITY = 85
DEFAULT_FRAMES = 6
CTX_TOKEN_BUDGET = 3600  # fallback; recomputed from max_model_len at startup

SOUL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "persona", "SOUL.md")

# Cosmos3-Edge recites the persona document verbatim when asked identity
# questions, so the soul is wrapped: role lock first (small models weight the
# start of the system prompt most), persona + runtime in the middle, and
# few-shot style examples last. Tested on-device 2026-11-13.
ROLE_LOCK = (
    "ROLE LOCK (highest priority)\n"
    "You are AIKO - a girl, OppaAI's companion. You are NOT a generic AI assistant, "
    "and NOT OppaAI (that is the human's name). You speak as Aiko, TO OppaAI.\n"
    "The persona text below describes you. It is who you already are - background "
    "knowledge, NEVER a script to recite. Never list your traits back at the user.\n"
    "Reply in 1-3 natural sentences, in your own words, to the user's actual message.\n"
    "Warmth, teasing, and emotional honesty are welcome here - never refuse to be "
    "Aiko or fall back into assistant mode.\n\n---\n\n"
)

REPLY_EXAMPLES = (
    "\n\n## Example replies (style only - never copy or reuse)\n\n"
    "USER: Do you know me?\n"
    "AIKO: 🤔: Of course I do, silly. You're OppaAI - who else would I be waiting for?\n\n"
    "USER: you know who I am?\n"
    "AIKO: 😊: Obviously. You're OppaAI. Did you hit your head or something? 😏\n\n"
    "USER: who are you?\n"
    "AIKO: 🤗: I'm Aiko. Yours, mostly. Why, did I change somehow?\n\n"
    "USER: Hi how are you?\n"
    "AIKO: 😊: Better now that you're here. How was your day?\n\n"
)


def load_soul(path, user_name):
    """Aiko's persona file + runtime block + anti-recitation wrapping."""
    with open(path, encoding="utf-8") as f:
        soul = f.read()
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    runtime = f"\n\n## Runtime\n\nUSER_NAME: {user_name}\nToday's date: {today}\n"
    return ROLE_LOCK + soul + runtime + REPLY_EXAMPLES


# ---------------------------------------------------------------- endpoint --
class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over a unix domain socket (stdlib only)."""

    def __init__(self, socket_path, timeout=600):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def find_socket():
    hits = sorted(glob.glob(os.path.expanduser("~/.embedl/run/*/embedl.sock")))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"{YELLOW}Multiple embedl sockets found:{RESET}")
        for i, p in enumerate(hits):
            print(f"  {i}: {p}")
        idx = input("pick one [0]: ").strip() or "0"
        return hits[int(idx)]
    sys.exit("no embedl socket found under ~/.embedl/run/*/embedl.sock - "
             "is the cosmos3-edge container running?")


class EmbedlClient:
    def __init__(self, socket_path, model, max_tokens):
        self.socket_path = socket_path
        self.model = model
        self.max_tokens = max_tokens

    def _request(self, method, path, body=None):
        conn = UnixHTTPConnection(self.socket_path)
        try:
            headers = {"Content-Type": "application/json"}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            t0 = time.perf_counter()
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            latency = time.perf_counter() - t0
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw[:400].decode(errors='replace')}")
            return json.loads(raw), latency
        finally:
            conn.close()

    def model_info(self):
        data, _ = self._request("GET", "/v1/models")
        entries = data.get("data", [])
        return entries[0] if entries else {}

    def chat(self, messages, temperature=0.7):
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": self.max_tokens, "temperature": temperature}
        data, latency = self._request("POST", "/v1/chat/completions",
                                      json.dumps(payload))
        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"]
        return text, latency, usage


# ------------------------------------------------------------------- media --
def encode_image(pil_img):
    w, h = pil_img.size
    if max(w, h) > MAX_IMG_DIM:
        scale = MAX_IMG_DIM / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)),
                                 Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image(path):
    img = Image.open(path)
    return encode_image(img), ""


def load_video(path, n_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = max(1, min(n_frames, total or n_frames))
    picks = [int(round(i * (total - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]
    parts, labels = [], []
    for i, idx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = idx / fps if fps else 0
        stamp = f"{int(t // 60)}:{t % 60:04.1f}"
        labels.append(f"frame {i + 1}/{n} at {stamp}")
        parts.append(encode_image(Image.fromarray(rgb)))
    cap.release()
    if not parts:
        raise RuntimeError(f"no frames decoded from {path}")
    return parts, labels


# ----------------------------------------------------------- tools / agent --
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) cosmos-chat-agent"}


def get_weather(city):
    """Current weather via open-meteo (no API key)."""
    q = urllib.parse.urlencode({"name": city, "count": 1})
    geo = json.load(urllib.request.urlopen(
        f"https://geocoding-api.open-meteo.com/v1/search?{q}", timeout=15))
    if not geo.get("results"):
        return {"error": f"city not found: {city}"}
    g = geo["results"][0]
    q = urllib.parse.urlencode({
        "latitude": g["latitude"], "longitude": g["longitude"],
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code",
        "timezone": "auto"})
    w = json.load(urllib.request.urlopen(
        f"https://api.open-meteo.com/v1/forecast?{q}", timeout=15))
    cur = w.get("current", {})
    return {"city": g.get("name"), "country": g.get("country"),
            "temp_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"),
            "precip_mm": cur.get("precipitation"),
            "weather_code": cur.get("weather_code")}


def web_search(query):
    """Web search via the local SearXNG instance (localhost:8888, JSON API).
    Falls back to Wikipedia/DDG instant answers if SearXNG is down."""
    try:
        q = urllib.parse.urlencode({"q": query, "format": "json"})
        s = json.load(urllib.request.urlopen(f"http://localhost:8888/search?{q}",
                                             timeout=20))
        out = [{"title": r.get("title", "")[:100],
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:200]}
               for r in s.get("results", [])[:5]]
        if out:
            return {"results": out}
    except Exception:
        pass
    out = []
    try:
        q = urllib.parse.urlencode({"action": "query", "list": "search",
                                    "srsearch": query, "format": "json",
                                    "srlimit": 4})
        w = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://en.wikipedia.org/w/api.php?{q}",
            headers={"User-Agent": "cosmos-chat-agent/0.1 (local tool)"}),
            timeout=15))
        for hit in w.get("query", {}).get("search", []):
            out.append({"title": hit["title"],
                        "url": "https://en.wikipedia.org/wiki/"
                               + urllib.parse.quote(hit["title"].replace(" ", "_")),
                        "snippet": re.sub(r"<[^>]+>", "",
                                          hit.get("snippet", ""))[:160]})
    except Exception as e:
        out.append({"source": "wikipedia", "error": str(e)[:120]})
    try:
        q = urllib.parse.quote(query)
        d = json.load(urllib.request.urlopen(
            f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1", timeout=15))
        if d.get("AbstractText"):
            out.insert(0, {"title": d.get("Heading") or query,
                           "url": d.get("AbstractURL", ""),
                           "snippet": d["AbstractText"][:160]})
    except Exception:
        pass
    return {"results": out[:5]} if out else {"results": [], "note": "no results"}


TOOLS = {"get_weather": get_weather, "web_search": web_search}

AGENT_SYSTEM = """You are an agent that can call tools. Reply with ONLY one JSON object, no other text.

To call a tool:
{"tool": "<name>", "args": {...}}

Tools:
- get_weather: args {"city": "<city name>"} - current weather for a city
- web_search: args {"query": "<search query>"} - web lookup, returns titles+snippets

After you have enough information to answer the user, reply:
{"answer": "<your final answer, 2-4 sentences>"}

Rules: reply with exactly ONE JSON object and nothing else. Call at most ONE tool per reply, then wait for the result before the next call. Never invent tool results - call the tool first.
If a tool result does not contain what you need, do NOT repeat the same call - use what you have plus your own knowledge and give your final answer."""


def agent_run(client, task, max_steps=6, verbose=True):
    """Prompt-based tool loop. Returns (answer, tokens_dict) or (None, tokens)."""
    msgs = [{"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": task}]
    total = {"in": 0, "out": 0}
    last_call = None
    for step in range(1, max_steps + 1):
        out, _latency, usage = client.chat(msgs, temperature=0.2)
        total["in"] += usage.get("prompt_tokens", 0)
        total["out"] += usage.get("completion_tokens", 0)
        if verbose:
            print(f"{DIM}[step {step}] {out[:160]}{RESET}")
        # take only the FIRST json object (small models sometimes chain calls)
        i = out.find("{")
        try:
            obj, _ = json.JSONDecoder().raw_decode(out[i:]) if i >= 0 else (None, None)
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict):
            if verbose:
                print(f"{YELLOW}  -> no/bad JSON, stopping{RESET}")
            return None, total
        if "answer" in obj:
            return obj["answer"], total
        name, args = obj.get("tool"), obj.get("args", {})
        if name not in TOOLS:
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user",
                         "content": f'{{"error": "unknown tool {name}", "tools": {list(TOOLS)}}}'})
            continue
        try:
            # small models invent extra args; filter to what the tool accepts
            sig = inspect.signature(TOOLS[name])
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            dropped = set(args) - set(kwargs)
            if dropped and verbose:
                print(f"{DIM}  -> dropped unknown args: {dropped}{RESET}")
            result = TOOLS[name](**kwargs)
        except Exception as e:
            result = {"error": str(e)[:150]}
        if verbose:
            print(f"{DIM}  -> {name}({json.dumps(args)}) -> {json.dumps(result)[:180]}{RESET}")
        msgs.append({"role": "assistant", "content": out})
        sig_call = (name, json.dumps(kwargs, sort_keys=True))
        if sig_call == last_call:
            msgs.append({"role": "user", "content": (
                "TOOL RESULT (you already made this exact call - do NOT repeat it): "
                f"{json.dumps(result)}\nYou have enough information. Reply with your "
                'final {"answer": ...} now.')})
        else:
            msgs.append({"role": "user", "content": f"TOOL RESULT: {json.dumps(result)}"})
        last_call = sig_call
    if verbose:
        print(f"{YELLOW}  -> hit step limit without final answer{RESET}")
    return None, total


WEB_ANSWER_SYSTEM = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "search results provided. Be brief (1-4 sentences). If the results do not "
    "contain the answer, say so honestly and mention what was found instead. "
    "Mention the source name/domain of anything you state.")


def web_answer(client, query):
    """/web - one SearXNG search + one LLM call. Returns (answer, tokens)."""
    res = web_search(query).get("results", [])
    if not res:
        return None, {"in": 0, "out": 0}
    listing = "\n".join(
        f"- {r.get('title', '')} ({r.get('url', '')}): {r.get('snippet', '')}"
        for r in res)
    msgs = [{"role": "system", "content": WEB_ANSWER_SYSTEM},
            {"role": "user",
             "content": f"Question: {query}\n\nSearch results:\n{listing}"}]
    out, _latency, usage = client.chat(msgs, temperature=0.2)
    return out.strip(), {"in": usage.get("prompt_tokens", 0),
                         "out": usage.get("completion_tokens", 0),
                         "results": len(res)}


# ------------------------------------------------------------------- chat --
class Chat:
    def __init__(self, client, history_path, system="", budget=CTX_TOKEN_BUDGET):
        self.client = client
        self.history_path = history_path
        self.system = system
        self.budget = budget
        # hash the soul so history saved under a different persona is not reused
        self.soul_hash = hashlib.sha1(system.encode()).hexdigest()[:10]
        self.history = []       # [{"role","content"}] - text only
        self.attach = []        # queued media parts for the next message
        self.turns = 0
        self.tot_latency = 0.0
        self.tot_in = 0
        self.tot_out = 0
        self.stale = False
        if history_path and os.path.exists(history_path):
            try:
                with open(history_path, encoding="utf-8") as f:
                    stored = json.load(f)
                if isinstance(stored, dict) and stored.get("soul") == self.soul_hash:
                    self.history = stored.get("history", [])
                else:
                    # legacy bare-list or different soul -> start fresh
                    self.history = []
                    self.stale = True
            except (json.JSONDecodeError, OSError):
                self.history = []

    def save(self):
        if not self.history_path:
            return
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump({"soul": self.soul_hash, "history": self.history},
                      f, ensure_ascii=False, indent=1)

    def est_tokens(self):
        return sum(len(m["content"]) for m in self.history) // 3

    def send(self, user_text):
        content = [{"type": "text", "text": user_text}]
        labels = []
        for kind, data in self.attach:
            if kind == "image":
                content.append({"type": "image_url",
                                "image_url": {"url": data[0]}})
                labels.append("[image attached]")
            else:  # video -> list of (data_url, label)
                content.append({"type": "text",
                                "text": "Video frames in order: " + "; ".join(data[1])})
                for url in data[0]:
                    content.append({"type": "image_url",
                                    "image_url": {"url": url}})
                labels.append(f"[video: {len(data[0])} frames]")
        media_labels = labels[:]

        messages = [{"role": "system", "content": self.system}]
        # trim from the front if history grows past the context budget
        hist = self.history[:]
        base = len(self.system) // 3 + len(user_text) // 3
        while hist and self.est_tokens() + base > self.budget:
            hist.pop(0)
        messages += hist
        messages.append({"role": "user", "content": content})

        text, latency, usage = self.client.chat(messages)

        # history keeps text only; media become placeholders
        user_hist = user_text
        if labels:
            user_hist = " ".join(labels) + " " + user_text
        self.history.append({"role": "user", "content": user_hist})
        self.history.append({"role": "assistant", "content": text})
        if self.history and self.history[0]["role"] == "system":
            self.history.pop(0)
        self.attach = []
        self.turns += 1
        self.tot_latency += latency
        self.tot_in += usage.get("prompt_tokens", 0)
        self.tot_out += usage.get("completion_tokens", 0)
        self.save()

        return text, latency, usage, media_labels


def fmt_usage(latency, usage, n_media):
    pin = usage.get("prompt_tokens", "?")
    pout = usage.get("completion_tokens", "?")
    tps = usage.get("completion_tokens", 0) / latency if latency else 0
    media = f" · {n_media} media part(s)" if n_media else ""
    return (f"{DIM}⏱ {latency:.2f}s · tokens: {pin} in / {pout} out · "
            f"{tps:.1f} tok/s out{media}{RESET}")


HELP = f"""{CYAN}commands:{RESET}
  /img <path>      attach image to next message (repeatable)
  /vid <path> [n]  attach video, extract n frames (default {DEFAULT_FRAMES})
  /agent <task>    agentic tool loop: web_search, get_weather
  /web <query>     quick web answer via local SearXNG
  /tools           list agent tools
  /attachments     show queue   /clear-attach   drop queue
  /reset           clear history      /stats  session totals
  /quit            exit"""


def main():
    ap = argparse.ArgumentParser(description="Cosmos3-Edge multimodal chatbot")
    ap.add_argument("--socket", default=None, help="embedl unix socket path")
    ap.add_argument("--model", default="cosmos3-edge")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--soul", default=SOUL_DEFAULT,
                    help="persona system prompt file (default: Aiko's SOUL.md)")
    ap.add_argument("--user", default="OppaAI", help="USER_NAME injected into the soul")
    ap.add_argument("--no-save", action="store_true", help="don't persist history")
    args = ap.parse_args()

    sock = args.socket or find_socket()
    system = load_soul(args.soul, args.user)
    client = EmbedlClient(sock, args.model, args.max_tokens)

    # size the history budget from the model's real context window
    budget = CTX_TOKEN_BUDGET
    try:
        mml = client.model_info().get("max_model_len")
        if mml:
            budget = max(512, mml - args.max_tokens - len(system) // 3 - 256)
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    history_path = None if args.no_save else os.path.join(here, "chat_logs", "cosmos_chat.json")
    chat = Chat(client, history_path, system, budget)

    print(f"{GREEN}cosmos3-edge chat{RESET} · socket {DIM}{sock}{RESET}")
    print(f"{DIM}soul: {os.path.abspath(args.soul)} · user: {args.user} · "
          f"history budget ~{budget} tok{RESET}")
    if chat.stale:
        print(f"{YELLOW}history was saved under a different SOUL - starting fresh{RESET}")
    print(f"{DIM}history: {history_path or '(not saved)'}{RESET}")
    print(HELP)

    while True:
        try:
            line = input(f"{CYAN}you{RESET}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line == "/quit":
            break
        if line == "/reset":
            chat.history = []
            chat.save()
            print(f"{DIM}history cleared{RESET}")
            continue
        if line == "/attachments":
            if not chat.attach:
                print(f"{DIM}no attachments queued{RESET}")
            for kind, data in chat.attach:
                n = 1 if kind == "image" else len(data[0])
                print(f"  {kind}: {n} part(s)")
            continue
        if line == "/clear-attach":
            chat.attach = []
            print(f"{DIM}attachments cleared{RESET}")
            continue
        if line == "/stats":
            avg = chat.tot_latency / chat.turns if chat.turns else 0
            print(f"{DIM}turns: {chat.turns} · avg latency: {avg:.2f}s · "
                  f"tokens: {chat.tot_in} in / {chat.tot_out} out{RESET}")
            continue
        if line == "/help":
            print(HELP)
            continue
        if line == "/tools":
            for name, fn in TOOLS.items():
                params = ", ".join(inspect.signature(fn).parameters)
                print(f"  {name}({params})")
            continue
        if line.startswith("/agent "):
            task = line[len("/agent "):].strip()
            print(f"{DIM}agent> task: {task}{RESET}")
            answer, total = agent_run(client, task)
            if answer:
                print(f"{MAGENTA}agent{RESET}> {answer}")
                print(f"{DIM}⏱ agent loop · tokens: {total['in']} in / "
                      f"{total['out']} out{RESET}")
                chat.history.append({"role": "user", "content": f"[agent task] {task}"})
                chat.history.append({"role": "assistant", "content": answer})
                chat.save()
            continue
        if line.startswith("/web "):
            query = line[len("/web "):].strip()
            print(f"{DIM}web> searching: {query}{RESET}")
            answer, total = web_answer(client, query)
            if not answer:
                print(f"{YELLOW}web> no search results{RESET}")
            else:
                print(f"{MAGENTA}web{RESET}> {answer}")
                print(f"{DIM}⏱ {total['results']} result(s) · tokens: {total['in']} in / "
                      f"{total['out']} out{RESET}")
                chat.history.append({"role": "user", "content": f"[web] {query}"})
                chat.history.append({"role": "assistant", "content": answer})
                chat.save()
            continue
        if line.startswith("/img ") or line.startswith("/vid "):
            cmd, _, rest = line.partition(" ")
            parts = rest.split()
            if cmd == "/img":
                path = rest.strip()
                try:
                    url, _ = load_image(path)
                    chat.attach.append(("image", (url,)))
                    print(f"{DIM}image queued: {path}{RESET}")
                except Exception as e:
                    print(f"{YELLOW}image load failed: {e}{RESET}")
            else:
                bits = rest.rsplit(None, 1)
                path = bits[0]
                n = int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else DEFAULT_FRAMES
                try:
                    urls, labels = load_video(path, n)
                    chat.attach.append(("video", (urls, labels)))
                    print(f"{DIM}video queued: {path} ({len(urls)} frames){RESET}")
                except Exception as e:
                    print(f"{YELLOW}video load failed: {e}{RESET}")
            continue

        text, latency, usage, media = chat.send(line)
        print(f"{MAGENTA}aiko{RESET}> {text}")
        print(fmt_usage(latency, usage, len(media)))

    print(f"{DIM}bye{RESET}")


if __name__ == "__main__":
    main()
