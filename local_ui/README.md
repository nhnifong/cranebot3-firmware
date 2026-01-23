# Standalone Local UI for Stringman

To use stringman in a LAN-only mode in which no video or other data leaves the local network

## Install

from this directory (local_ui)

    python3 -m venv venv
    pip install -r requirements.txt

## Run

    python main.py


This local UI is based on ursina and cannot be distributed with the nf_robot python module. It can only be run from source for now.

## Future

This ursina UI currently exists as the only *relatively easy* way to run stringman in a LAN mode but it isn't ideal

The web UI on [neufangled.com](https://neufangled.com/control_panel) is the intended final UI for stringman, but using that in the lan only mode currently requires running a local instance of the website in docker.

I am aiming to slim that down as much as possible so that observer.py could host a static version that it talks to directly without the cloud relay.