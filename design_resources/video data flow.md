There are actually a larger variety of scales that are ideal for different purposes.
Basically, as soon as we are done with a bigger or higher framerate, we can discard it and keep a lower rate.
The robot must know when it is being teleoperated vs just being controlled with a target queue so it can turn up the framerates.

target resolution and framerate for anchor cameras
 - for apriltag detection: 1920x1080 30fps
 - for teleoperation: 1920x1080 30fps
 - for UI's both local and remote: 960x544 2fps
 - for inference with object recognizer such as dobby: 960x544 2fps

summary of anchor video pipline:
 1. detect aruco markers in frame.
 2. every 15th frame, resize down to 960x544, send to UI,
 3. send to dobby, take heatmap from dobby, also send that to UI

target resolution and framerate for gripper camera
 - before stabilization 960x540 30fps
 - after stabilization, teleoperation: 384x384 30fps
 - after stabilization, inference: 384x384 5fps
 - after stabilization, UI local and remote: 384x384 5fps

summary of gripper video pipeline
 1. start with half res stream from raspi
 2. every 6th frame, stabilize using IMU data, producing square 384x384 image at 5fps
 3. send to both UI and centering model


These are low enough rates that it would probably be fine to drop jpegs on the existing telemetry websocket connection, but I'm strongly skeptical of it
It won't scale, and when teleoperation is enabled, we have to turn the framerates up a lot.

Therefore I am instead considering using an ffmpeg subprocess of observer to send video out.
the ffmpeg command would have one output for sending RTMP packaged video to the cloud,
and another output for mpegts over a udp port on localhost (if a local ursina UI is connected)
from the observer's perspective, it pushes raw frames onto the stdin of this ffmpeg process.
from the ursina UI's perspective, it consumes the stream with pyav in a thread and pushes the frames as textures to entities. 