# nf_robot

Control code for the Stringman household robotic crane from Neufangled Robotics

## [Build Guides and Documentation](https://neufangled.com/docs)

Purchase assembled robots or kits at [neufangled.com](https://neufangled.com)

## Installation of stringman controller (Users)

Linux (python 3.11 or later)

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install "nf_robot[host]"

Start headless robot controller in LAN-only mode.
The particular robot details will be read from/saved to bedroom.conf

    stringman-headless --config=bedroom.conf

The stringman motion controller (stringman-headless) is the program which communicates with the robot components over wifi and acts as the central brain of a single robot. It must be running on the same network as the powered on anchors and gripper in order for the robot to be active and controllable. The main entrypoint is observer.py

It listens on port 4245 for a connection from a UI or local AI policy. The UI can be opened at [neufangled.com/playroom](https://neufangled.com/playroom). Select LAN mode at first.

Refer to the [Usage Guide](https://neufangled.com/docs/usage_guide/) for more detailed instructions on setup and use.

### Arguments to stringman-headless

options:

  --config              A json file where the robot's ID and calibration data are stored. You may use one for a bedroom and one for a playroom for example, even if it is the same hardware being taken
                        down and put back up in another room 
  --telemetry_env {local,staging,production}
                        The cloud telemetry server to connect to (choices: local, staging, production) The default is None, which allows local connections on port 4245 only
                        When production is used if you have already bound the robot to an account at neufangled.com. This is completely optional.
  --auto_start          Automatically unpark and start cleaning when all components connect
  --local_models        Use local models from models/ for the targeting and centering models rather than downloading the production models from huggingface
  --debug               Enable DEBUG level logging

### Minimum system specs

At least 8 cores and 8GB of ram.
In order to perform local inference, some kind of pytorch accelertion is necessary.
Mini PC's or laptops based on the Ryzen 7 7840HS are probably about the cheapest machines that can run stringman's motion controller since it has an NPU that can be used to accelerate pytorch. A mac mini is also a viable option.

Otherwise, any gaming PC is usually more than enough.

### Telemetry stream

stringman-headless listens on port 4245 locally for telemetry connections. This is a websocket sending and receiving protobufs defined in `src/nf_robot/protos`
Every message sent by stringman-headless is a serialized `TelemetryBatchUpdate` and every message received is expected to be a `ControlBatchUpdate`.

Within the telemetry stream, there are `VideoReady` messages containing URIs for connecting to the robot's video streams.

The UI at [neufangled.com/playroom](https://neufangled.com/playroom) sends controls and receives telemetry.
Any AI policy served by `src/nf_robot/ml/stringman_lerobot.py` also sends controls and receives telemetry.
Agents wishing to write code to interface with a stringman robot may also follow this pattern.

The expected inputs are basically marker box velocity and finger and wrist speeds. The gripper hangs 50 cm below the marker box.
Higher level control is achived by having a policy such as DIT or a VLA connected to the robot, and having another client sending `nf.common.EpisodeControl` commands with prompts.

See [Imitation Learning](https://neufangled.com/docs/imitation_learning/) for a more detailed guide.

## Cloud telemetry relay

When stringman-headless is in LAN mode (done by omitting the --telemetry_env argument) it only accepts local telemetry connections and only streams video locally.

If connected to a robot in lan mode from the UI at neufangled.com, you can click "Bind robot" in the run menu, log in with an identity profider, and that robot id (from the config.json file) will be marked as owned by you.
It is then possible to run with `--telemetry_env=production` and stringman will also send telemetry and video to neufangled.com so that you can view and control the robot remotely over the internet. This is accessed from the "My Robots" option when opening neufangled.com/playroom.

No video or telemetry is saved when you use the cloud relay. The only way video gets shared with us is if you record a public lerobot dataset and inform us of it.

## Installation (developers)

    git clone https://github.com/nhnifong/cranebot3-firmware.git

if your python version is 3.13 or above, go for it. otherwise make the venv with a specific version.

    sudo apt install python3-dev python3-virtualenv python3-pip ffmpeg build-essential git-lfs
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install -e ".[host,dev,pi]"

### If you have an RTX 5090

    pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu129

### Run tests

for certain tests, image files are required

    git lfs install
    git lfs pull

run all tests

    pytest tests

### Setting up a component

Robot components that boot from the [`stringman-zero2w.img`](https://storage.googleapis.com/stringman-models/stringman-zero2w.img) (1.6GB) image should begin looking for wifi share codes with their camera immediately. You can produce a code with [qifi.org](htts://qifi.org)

Once the pi sees the code it will connect to the network and remember those settings. It should then be discoverable by the control panel via multicast DNS (Bonjour)

## Starting from a base rpi image

Alternatively the software can be set up from a fresh raspberry pi lite 64 bit image.
After booting any raspberry pi from a fresh image, perform an update

    sudo apt update -y && sudo apt full-upgrade -y -o Dpkg::Options::="--force-confold" && sudo apt install -y git python3-dev python3-virtualenv rpicam-apps i2c-tools

Clone the [cranebot-firmware](https://github.com/nhnifong/cranebot3-firmware) repo

    git clone https://github.com/nhnifong/cranebot3-firmware.git && cd cranebot3-firmware

Set the component type by uncommenting the appropriate line in server.conf

    nano server.conf

Install stringman

    chmod +x install.sh
    sudo ./install.sh

### Additional settings for anchors

Setup for any raspberry pi that will be part of an anchor
Enable uart serial harware interface interactively.

    sudo raspi-config

In interface optoins, select serial port. disable the login shell, but enable hardware serial.

add the following lines lines to to `/boot/firmware/config.txt`  at the end this disables bluetooth, which would otherwise occupy the uart hardware.
Then reboot after this change

    enable_uart=1
    dtoverlay=disable-bt

### Additional settings for gripper

Setup for the raspberry pi in the gripper with the inventor hat mini
Enable i2c

    sudo raspi-config nonint do_i2c 0

Add this line to `/boot/firmware/config.txt` just under `dtparam=i2c_arm=on` and reboot

    dtparam=i2c_baudrate=400000

## Rebuilding the python module

within a venv install the build tools

    python3 -m pip install --upgrade build twine

Bump the version number in pyproject.toml
then at this repo's root, build the module. Artifacts will be in dist/

    python3 -m build

Upload the particular version you just built to PyPi

    python3 -m twine upload dist/nf_robot-4.0.5*

### QA scripts

Note that if you are proceeding to QA scripts right after doing the steps above you must reboot and then stop the service before running those scrips.

    sudo reboot now

log back in

    sudo systemctl stop cranebot.service

Run QA scripts for the specific component type

    /opt/robot/env/bin/qa-gripper-arp
    /opt/robot/env/bin/qa-anchor-arp

These scripts both check whether everything is connected as it should be and in the case of anchors, set whether it is a power anchor or not.

To update to the lastest nf_robot version in a component

    /opt/robot/env/bin/pip install --upgrade "nf_robot[pi]"

## Training models


## Windows

A self contained windows installer can be generated. The exact installation of stringman that ends up in the installer depends on what was in the virtualenv these commands are run from, so make a new one.

    python3 -m venv winvenv
    source winvenv/bin/activate
    pip install nf_robot[host]
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "Stringman" win_main.py

## Support this project

[Donate on Ko-fi](https://ko-fi.com/neufangled)

