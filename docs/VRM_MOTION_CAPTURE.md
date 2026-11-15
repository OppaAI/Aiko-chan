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
Choose 15, 30, or 60 capture samples per second and click **track video**.
Use **Dance (low lag)** for choreography, **Direct** to remove rig smoothing,
or **Smooth** to reduce jitter. For a fixed camera, optionally enable **Follow
sideways movement and jumps**. This estimates image-plane hip travel using a
hips-to-floor calibration; it does not reconstruct movement toward the camera. The
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
usual animation resumes. Recorded face tracks take priority over speech,
conversational expressions, automatic gaze and blinking while the clip plays;
those controls resume afterward. Clips without facial tracks keep normal WebUI
expression control. Facial capture includes mouth shapes, blinks, gaze expression
weights and a basic smile mapped to the avatar's supported blendshapes. Normalized VRM rotations and proportional hip
offsets allow reuse across VRMs; different body proportions can change the result.
An already imported clip can also be played by browser code with
`window.aikoPlayMotionClip('clipName', {loop: false, speed: 1})`.

## Accuracy boundaries

This version estimates body, head, face and 30 finger joints with MediaPipe and
Kalidokit. Hands are matched to pose wrists; absent hands briefly hold and then
relax. Occluded or blurred fingers cannot be reconstructed. Face and hand
detection counts are shown and exported, so missing detections are visible;
counts measure coverage, never joint accuracy. Pose-guided close-up crops
increase the model input resolution for small faces and hands. The avatar needs the corresponding
finger bones and facial blendshapes. Low-visibility limbs hold to reduce jitter.
Foot orientation/contact, full 3D root travel and physical collisions remain
unsolved. Temporal filtering introduces some lag; Direct removes rig filtering.

For best source footage, use one person, steady lighting, a fixed camera, an
unobstructed full-body view including feet, and a short clip without cuts.
Compare source and preview, especially shoulders, elbows, knees and turns,
before using the take. A single camera cannot uniquely recover hidden limbs,
depth, or every joint twist. Exact matching requires additional views or motion
capture hardware plus calibration, contact constraints, and manual cleanup;
it cannot be promised by changing a VRM file or increasing the capture rate.
