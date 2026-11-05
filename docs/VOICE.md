# Voice playback and self-echo protection

Aiko's default voice path is half duplex: ASR does not consume microphone audio
while Aiko speaks. Capture resumes after a short playback tail. This protects a
raw microphone/speaker pair without loading another model on the 8 GB Jetson.

## Settings

Open WebUI **Settings → Senses & voice**. Save and restart Aiko for file edits.

| Setting | Default | Meaning |
|---|---|---|
| `ASR_ECHO_TAIL_MS` | `400` | Microphone cooldown after playback; range 0–2000 ms. Increase for a speaker/room with a longer acoustic tail. |
| `BARGE_IN_ENABLED` | `0` | Master switch for automatic speech interruption. |
| `BARGE_IN_LOCAL_AEC` | `0` | Assert that the local microphone input is already echo-cancelled by hardware or OS routing. This switch does **not** create an echo canceller. |

Leave local AEC off for a raw Jetson microphone. With barge-in enabled, the
browser requests echo cancellation and requires the audio track to report that
it is enabled before allowing automatic interruption. Without confirmed AEC,
the browser remains half duplex. Local automatic interruption requires the
separate local-AEC setting. Explicit stop requests still cancel playback.

For simultaneous speaking/listening, acoustic separation, headphones, or a
properly configured echo-cancelled input are needed. No energy threshold can
reliably distinguish a raw speaker echo from a human speaking over it. The
software tests do not measure the Jetson's microphone, speaker or room acoustics.

## What the protection covers

- Browser microphone frames, VAD onset events, pre-speech buffers and noise-floor
  updates pause during actual browser playback and its tail. This also applies
  when browser energy gating is disabled. A confirmed AEC barge-in keeps only
  its short onset buffer, including across the server's stop acknowledgment.
- Server ingress rejects incorrectly sized PCM frames and suppresses feedback
  during playback when automatic overlap is disabled. Native capture discards
  overlapping audio if scheduled/proactive speech begins after recording starts.
  The CLI passes its speaker into the listener, so the same coordination applies.
- A small, bounded in-memory cache of recently played TTS text provides a second
  check for close transcript matches during playback overlap. It is scoped to
  the playback user, does not retain microphone audio, and does not suppress
  short commands such as “stop” or “yes”. It is a fallback for residual echo,
  not an acoustic echo canceller.
- Playback workers own separate cancellation events. Stopping an old response
  cannot be undone by a new response, and an old worker cannot clear the new
  response's playing state. Browser decoding has the same cancellation check.
- Local audio control locks are released while audio runs, so stop can interrupt
  playback immediately. Browser stop messages target the cancelled session's
  user. Autoplay/decode timeouts release browser capture suppression.

## Resource and failure handling

The backend speech queue holds at most 32 sentence pieces, with producer
backpressure instead of an unbounded backlog. Closing a full queue does not
block stop; graceful completion still drains its sentences. Synthesis is limited
to two concurrent requests, including prefetch. The browser audio backlog is
also bounded. Cancelling does not forcibly kill an HTTP/native call already in
progress, but its eventual audio and callbacks stay cancelled.

A failed WebUI ASR worker releases its waiter. A foreign user's microphone frame
is skipped rather than terminating the current utterance. CLI/WebUI avoid
starting another capture worker while a timed-out native request is still alive;
typed input remains available.

Automated checks use fake devices, real PCM buffers, and the browser's actual
JavaScript state machines. Run the browser checks independently with:

```sh
node tests/browser/voice_audio.test.js
```

Real speaker/microphone feedback, AEC effectiveness, wake accuracy, speaker-match
quality and RAM/latency measurements still need the physical Jetson.

## Wake sessions and speaker enrollment

Wake windows are per user. Waking Alice does not wake Bob; empty transcripts
and failed speaker matches do not renew a window. Browser capture rechecks the
identity after its first accepted frame, when that identity is actually known.
Fuzzy matching ignores ASR capitalization and punctuation.

Acoustic wake inference runs on complete 80 ms frames with the pinned
`livekit-wakeword` 2.x streaming API. Legacy 0.x classifiers receive a bounded
two-second window. Incomplete frames do not count as failed predictions. The
optional SDK loads only when a wake model is configured; inference uses one CPU
thread. A 2.x runtime needs a streaming 2.x ONNX artifact: older classifier
artifacts must be retrained with that package's training tools. If loading fails,
Aiko retains fuzzy ASR-text wake matching. Real model accuracy still needs testing.

Enroll the identity used by the WebUI from the repository root:

```sh
python -m util.enroll_speak --user OppaAI --model models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx --seconds 6
```

Omitting `--user` uses the bound/configured identity or a uniquely configured
owner. If the identity is ambiguous, the utility asks for `--user` instead of
writing a guest profile. `AIKO_USER_ID` and the legacy `USER_ID` are supported.
`--out` remains available for an explicitly selected output file.

Silent/short recordings, invalid embeddings and failed writes preserve the prior
enrollment. Valid profiles are flushed and atomically replaced with private file
permissions. Enrollment and runtime share vector validation. Re-enrollment is
picked up without rebuilding the extractor; failed extractor initialization is
not cached as a successful profile load.

`SPEAKER_VERIFY_TIMEOUT_S` (default 5 seconds; range 0.1–30) bounds the optional
verifier's wait. Only one verification worker may remain in flight per listener.
It keeps its enrollment snapshot even if another user starts a later request.
Non-finite, zero or dimension-mismatched vectors never produce a successful or
NaN match. When the explicit speaker gate is enabled, failed/timed-out matches
are dropped; otherwise verification remains metadata and text delivery continues.
