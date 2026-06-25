# Load Test Suite

Stress queues, services, and resource limits.

## Targets

- sequential and concurrent text turns;
- long prompts near context limits;
- large sqlite memory DB sizes: 100, 1,000, 10,000 memories;
- repeated `/web`, `/memory`, `/reset`, `/voice`, `/listen` commands;
- repeated WebSocket reconnects;
- rapid schedule create/cancel loops;
- repeated ASR/TTS turns and barge-in attempts.

## Metrics

Capture latency percentiles, RSS memory, CPU/GPU usage, DB size, thread count, and error rate.
