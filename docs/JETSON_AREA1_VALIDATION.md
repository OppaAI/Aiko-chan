# Area 1: Jetson Orin Nano 8 GB validation

Code review and cloud tests cannot establish peak unified-memory use or voice
latency on the physical Jetson. Keep the project version at 0.2.0. Run these
checks on the Jetson with the model services and configuration you normally use.

## Capture device telemetry

In a separate terminal, run:

```bash
mkdir -p ~/aiko-measurements
tegrastats --interval 1000 --logfile ~/aiko-measurements/area1-tegrastats.log
```

Stop that foreground command with Ctrl-C after each session. Use a separate log
filename for each run. Record RAM, swap, CPU/GPU activity, and temperature before
launch, at the highest observed startup use, when ready, and after shutdown.
External inference servers remain separate processes; their RAM is part of the
whole-device measurement even though they are not launched by Aiko's Python
process.

## Exercise each startup path

Run these separately, with no other Aiko application instance running:

```bash
uv run python main.py --cli --text
uv run python main.py --text
uv run python main.py
```

For each run, record the time to READY and the slowest displayed startup step.
Send a short message and a longer coding request. For the voice-enabled run,
exercise microphone activation, a spoken reply, and interruption. Keep the
browser connected when checking remote audio. Use `/quit` to end the session;
repeat once with Ctrl-C while a startup step is still pending.

Use `--cli --trace` to inspect turn decisions and `--cli --debug --no-console`
for detailed file logging without console log traffic competing with chat.

## Failure and restart checks

- Start WebUI on an already occupied port: it must report startup failure rather
  than READY, and a subsequent attempt on a free port must start normally.
- Start with an unavailable optional voice service: Aiko must follow its
  documented degraded path rather than claiming the listener failed to bind.
- Repeat normal start/quit cycles and inspect for leftover Aiko application or
  HTTP listener processes. Existing external model servers are expected to stay.
- Restart through the deployment's service manager and confirm shutdown reaches
  the same cleanup path used by the console launcher.

Record the commit, mode, model names/context sizes, telemetry peaks, observed
latency, and any failure messages. Do not mark hardware validation complete until
these results are available. In-flight model calls are allowed to finish, and
live memory is retained when closing it could race an active worker; the runtime
does not forcibly kill Python threads or unload external model services.
