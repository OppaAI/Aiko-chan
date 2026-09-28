#!/usr/bin/env python3
"""
test_chat.py - test chatbot for Aiko's SOUL persona against llama-server.

Thinking is OFF by default. Start a message with /think to enable it for that turn.
History is saved per model in chat_logs/<name>.json and reloaded on start.

Usage:
  # Chat with one model
  python aiko_chat.py --name iq4 --url http://localhost:8081

  # Chat with two models at once (needs both servers running)
  python aiko_chat.py --name iq4 --url http://localhost:8081 \
                      --name2 ministral3 --url2 http://localhost:8082

  # Low RAM: chat with model A, then swap servers and replay the same user
  # turns against model B, showing A's reply next to B's
  python aiko_chat.py --name ministral3 --url http://localhost:8081 \
                      --replay chat_logs/iq4.json

In-chat commands:
  /think <msg>   answer with thinking on (this turn only)
  /reset         clear history (memory file too)
  /stats         emoji-format compliance and average tok/s per model
  /quit          exit
"""
import argparse
import datetime
import json
import os
import re
import sys
import requests

FACE_RE = re.compile(r"^[\U0001F300-\U0001FAFF\u2600-\u27BF]\uFE0F?\s*:")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
DIM, RESET, CYAN, YELLOW = "\033[2m", "\033[0m", "\033[36m", "\033[33m"


def load_soul(path, user_name):
    with open(path, encoding="utf-8") as f:
        soul = f.read()
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return (
        soul
        + f"\n\n## Runtime\n\nUSER_NAME: {user_name}\nToday's date: {today}\n"
    )


def est_tokens(text):
    # Conservative: Japanese is ~1 token/char, English ~1 token/4 chars.
    return int(len(text) / 2.5) + 4


class Backend:
    def __init__(self, name, url, system, history_dir, temp, max_tokens, ctx_budget):
        self.name = name
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.system = system
        self.temp = temp
        self.max_tokens = max_tokens
        self.ctx_budget = ctx_budget
        os.makedirs(history_dir, exist_ok=True)
        self.path = os.path.join(history_dir, f"{name}.json")
        self.history = []
        self.turns = 0
        self.face_ok = 0
        self.tps = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.history = json.load(f)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=1)

    def reset(self):
        self.history = []
        self.save()

    def _window(self):
        """Newest messages that fit the token budget (history stays whole on disk)."""
        budget = self.ctx_budget - est_tokens(self.system) - self.max_tokens
        kept, used = [], 0
        for m in reversed(self.history):
            t = est_tokens(m["content"])
            if used + t > budget:
                break
            kept.append(m)
            used += t
        kept.reverse()
        while kept and kept[0]["role"] != "user":  # never start on an assistant turn
            kept.pop(0)
        return kept

    def ask(self, text, think=False):
        self.history.append({"role": "user", "content": text})
        payload = {
            "stream": False,
            "temperature": self.temp,
            "max_tokens": 2048 if think else self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
            "messages": [{"role": "system", "content": self.system}] + self._window(),
        }
        try:
            r = requests.post(self.url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self.history.pop()
            return None, None, f"request failed: {e}"

        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        m = re.search(r"<think>(.*?)</think>", content, re.S)
        if m:  # some templates inline the thinking
            reasoning = reasoning or m.group(1).strip()
        content = THINK_RE.sub("", content).strip()

        # Only the final answer goes into memory, never the thinking.
        self.history.append({"role": "assistant", "content": content})
        self.save()

        self.turns += 1
        self.face_ok += bool(FACE_RE.match(content))
        tps = data.get("timings", {}).get("predicted_per_second")
        if tps:
            self.tps.append(tps)
        return content, reasoning, None

    def stats(self):
        avg = sum(self.tps) / len(self.tps) if self.tps else 0
        return (
            f"{self.name}: {self.face_ok}/{self.turns} replies start with 'emoji:' "
            f"| avg {avg:.1f} tok/s | {len(self.history)} msgs in memory"
        )


def show(b, content, reasoning, err, think):
    if err:
        print(f"{YELLOW}[{b.name}] {err}{RESET}")
        return
    if think and reasoning:
        print(f"{DIM}[{b.name} thinking] {reasoning}{RESET}")
    ok = "" if FACE_RE.match(content) else f" {YELLOW}(no emoji: prefix){RESET}"
    print(f"{CYAN}[{b.name}]{RESET} {content}{ok}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="iq4")
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--name2")
    ap.add_argument("--url2")
    ap.add_argument("--soul", default="../persona/SOUL.md")
    ap.add_argument("--user", default="OppaAI")
    ap.add_argument("--history-dir", default="chat_logs")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--ctx-budget", type=int, default=6000,
                    help="prompt+reply token budget (server -c is 8192)")
    ap.add_argument("--replay", help="history json from another model; replay its user turns")
    args = ap.parse_args()

    system = load_soul(args.soul, args.user)
    backends = [Backend(args.name, args.url, system, args.history_dir,
                        args.temp, args.max_tokens, args.ctx_budget)]
    if args.name2 and args.url2:
        backends.append(Backend(args.name2, args.url2, system, args.history_dir,
                                args.temp, args.max_tokens, args.ctx_budget))

    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            other = json.load(f)
        b = backends[0]
        b.reset()
        pairs = [(other[i]["content"], other[i + 1]["content"] if i + 1 < len(other) else "")
                 for i in range(len(other)) if other[i]["role"] == "user"]
        for user_msg, ref in pairs:
            think = user_msg.startswith("/think")
            text = user_msg[6:].strip() if think else user_msg
            print(f"{args.user}: {text}")
            print(f"{DIM}[{os.path.basename(args.replay)}] {ref}{RESET}")
            show(b, *b.ask(text, think), think)
        print(b.stats())
        return

    for b in backends:
        if b.history:
            print(f"{DIM}[{b.name}] loaded {len(b.history)} messages from {b.path}{RESET}")
    print("Chat started. /think <msg>, /reset, /stats, /quit\n")

    while True:
        try:
            line = input(f"{args.user}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            for b in backends:
                b.reset()
            print("History cleared.\n")
            continue
        if line == "/stats":
            for b in backends:
                print(b.stats())
            print()
            continue

        think = line.startswith("/think")
        text = line[6:].strip() if think else line
        if not text:
            continue
        for b in backends:
            show(b, *b.ask(text, think), think)

    for b in backends:
        print(b.stats())


if __name__ == "__main__":
    sys.exit(main())
