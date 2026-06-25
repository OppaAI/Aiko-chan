# Performance Test Suite

Measure latency and throughput against hardware-specific baselines.

## Metrics

- boot to chat-ready;
- boot to voice-ready;
- first-token latency;
- full short-turn latency;
- memory recall latency at multiple DB sizes;
- `/web` latency;
- ASR finalization after silence;
- TTS first-audio latency;
- barge-in reaction time;
- WebUI token-display lag;
- RSS growth over idle and active windows.

## Tools

Use `/usr/bin/time -v`, `ps`, `jtop` on Jetson, `nvidia-smi` on desktop NVIDIA, browser devtools, and Aiko logs.
