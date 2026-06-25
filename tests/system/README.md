# System Test Suite

Exercise complete user journeys.

## Core Scenarios

- text-only baseline: `uv run python main.py --text`;
- curses full voice: `uv run python main.py`;
- browser WebUI text: `uv run python main.py --webui --text`;
- browser WebUI voice: `uv run python main.py --webui`;
- agentic planning/workspace/scheduling/photo/repo workflows;
- restart and memory recall journey.

## Acceptance

A scenario passes only if the process exits cleanly, expected artifacts are created in approved locations, and no uncaught traceback appears.
