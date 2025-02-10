Installation

Since we require the picamera2 module and it can't be installed with pip, you have to do this weired hybrid method

starting in this directory (raspi-anchor)

    sudo apt install python3-picamera2 --no-install-recommends
    python3 -m venv --system-site-packages env
    source env/bin/activate
    pip3 install -r requirements.txt