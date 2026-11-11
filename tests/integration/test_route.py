#!/usr/bin/env python3
"""Read-only route diagnostic using Aiko's actual four-way routing policy.

python tests/integration/test_route.py --suite --quiet
python tests/integration/test_route.py "Research the latest ROS release"
Imports are inert: services/models are initialized only by main(). No tools run.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

BUILTIN_SUITE = [("can you ping me when it's 8pm", 'agentic'), ('schedule checking my email every hour and notify me when abc company writes', 'agentic'), ('research carrot cake recipes, plan ingredients and steps, then write a report', 'agentic'), ('put together a message for my team about the deployment delay', 'agentic'), ("pull up what's new in the ROS2 Jazzy release and summarize it", 'agentic'), ('help me map out what I need to do before the hackathon deadline', 'agentic'), ("this keeps segfaulting and I don't know why", 'agentic'), ('refactor the schedule runner to use asyncio instead of threads', 'agentic'), ('open think.py and agentic.py, combine architecture routing into coding, and update tests', 'agentic'), ('should I run the embedder on CPU or offload to GPU for aarch64', 'agentic'), ('pick up where we left off on the memory consolidation refactor', 'agentic'), ('is it weird that I find debugging more satisfying than writing features', 'chat'), ("what's the actual difference between a semaphore and a mutex", 'chat'), ('do you think embeddings are a good long-term foundation for memory', 'chat'), ('walk me through why cosine similarity works for semantic search', 'chat'), ('I feel like my routing thresholds are tuned to my test set', 'chat'), ("what's the Canucks score from last night", 'chat+search'), ('has NVIDIA released any new Jetson hardware this year', 'chat+search'), ("what's ethereum trading at right now", 'chat+search'), ('did llama.cpp merge the Vulkan backend yet', 'chat+search'), ("what's the current onnxruntime stable release", 'chat+search')]


def load_suite(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Prompt file must contain a JSON list')
    suite = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'Entry {index} must be an object')
        prompt = item.get('prompt', item.get('text'))
        expected = item.get('expected')
        if not isinstance(prompt, str) or not prompt.strip() or expected not in ('chat', 'localchat', 'webchat', 'agentic', 'greeting'):
            raise ValueError(f'Entry {index} needs prompt and a valid expected route')
        suite.append((prompt.strip(), 'localchat' if expected == 'chat' else expected))
    return suite


def _make_router():
    # Keep script execution and test collection independent of cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from system.config import load_config
    load_config()
    from cognition.memory.memorize import AikoMemorize
    from cognition.think import AikoThink, LLM_BASE_URL, ROUTER_MODEL, LLM_TIMEOUT
    from openai import OpenAI
    think = AikoThink.__new__(AikoThink)
    think._memorize = AikoMemorize()
    think._memorize_lock = threading.RLock()
    think._semantic_example_cache = {}
    think._semantic_example_cache_lock = threading.RLock()
    think._client = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv('LLM_API_KEY', '') or 'not-needed', timeout=LLM_TIMEOUT)
    think._router_model = ROUTER_MODEL
    return think


def run_suite(router, suite, quiet=False):
    passed = 0
    for prompt, expected in suite:
        expected = 'localchat' if expected == 'chat' else expected
        started = time.monotonic()
        intent, _ = router._route_intent(prompt)
        ok = intent == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {intent} (expected {expected}) {prompt}")
        if not quiet:
            print(f'  {(time.monotonic()-started)*1000:.0f} ms; classification only, no task execution')
    print(f'{passed}/{len(suite)} route expectations passed')
    return 0 if passed == len(suite) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--file')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        suite = load_suite(args.file) if args.file else list(BUILTIN_SUITE) if args.suite else None
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if suite is not None and args.seed is not None:
        random.Random(args.seed).shuffle(suite)
    router = _make_router()
    if suite is not None:
        return run_suite(router, suite, args.quiet)
    if args.prompt:
        print(router._route_intent(args.prompt)[0])
        return 0
    while True:
        try:
            prompt = input('route> ').strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if prompt:
            print(router._route_intent(prompt)[0])


if __name__ == '__main__':
    raise SystemExit(main())
