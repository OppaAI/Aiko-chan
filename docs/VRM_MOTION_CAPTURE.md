# Capture a video motion and replay it in Aiko

`util/vrm-webcam.html` is an upper-body gesture tester with a simplified skeleton.
Use `util/vrm-video.html` to track a local video on an actual VRM avatar. This is
motion capture and retargeting: it saves animation data without retraining or
modifying the VRM model.

## Capture

From the Aiko checkout, serve the utility on localhost:

```sh
python3 -m http.server 8790 --bind 127.0.0.1
```

Open `http://localhost:8790/util/vrm-video.html`. Select the same `.vrm` file used
by WebUI (`interface/webui/static/assets/Aiko.vrm`), then select your video.
Choose 15, 30, or 60 capture samples per second and click **track video**. The
video is paused and decoded in sequence, allowing slower machines to process
the selected samples instead of dropping them to keep up with real-time video.
Capturing at 60 fps does not create motion detail absent from the source video.

Use **play take**, crop the desired interval, and **save WebUI motion .json**.
A cancelled capture can still export its completed frames; no-body detections
are reported, and a take with no detected body is rejected. The tools process
media in the browser; the tracking libraries/models are downloaded from their
configured public CDNs. Remote videos require CORS; local files are simplest.

## Replay and save

Start Aiko normally and open its WebUI. Expand **Recorded motions**, import the
motion JSON, select it, and press **Play**. **Loop**, **Speed**, **Stop**, and
**Remove** control playback. Imported clips are stored in IndexedDB in this
browser and origin. Keep the exported JSON as a backup; another browser, port,
or device has a separate library. The WebUI does not execute imported JavaScript.

Playback temporarily owns the humanoid skeleton and frames the whole avatar.
When it ends or is stopped, the prior rig and camera are restored and Aiko's
usual animation resumes. Speech still controls mouth and expression weights.
Source face expressions are kept in the JSON for future editing but are not
applied over the WebUI's speech. Normalized VRM rotations and proportional hip
offsets allow reuse across VRMs; different body proportions can change the result.
An already imported clip can also be played by browser code with
`window.aikoPlayMotionClip('clipName', {loop: false, speed: 1})`.

## Accuracy boundaries

This version estimates body and head motion with MediaPipe and Kalidokit. It
holds low-visibility limbs to reduce jitter and limits root drift. It does not
solve detailed fingers, foot orientation/contact, full root travel, or physical
collisions. Lost body detections are reported, but that count is not a joint
accuracy score. Temporal filtering also introduces lag.

For best source footage, use one person, steady lighting, a fixed camera, an
unobstructed full-body view including feet, and a short clip without cuts.
Compare source and preview, especially shoulders, elbows, knees and turns,
before using the take. A single camera cannot uniquely recover hidden limbs,
depth, or every joint twist. Exact matching requires additional views or motion
capture hardware plus calibration, contact constraints, and manual cleanup;
it cannot be promised by changing a VRM file or increasing the capture rate.
