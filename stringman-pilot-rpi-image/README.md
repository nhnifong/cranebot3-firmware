## Raspberry pi (development)

`stringman-pilot-rpi-image` contains the configuration needed to build an SD card image for various stringman robot components.

Wifi can be discovered after boot but if you want to preconfigure it, edit `preconfigured.nmconnection`

On a raspberry pi with suitable ram, such as a Pi 5,
with [rpi-image-gen](https://github.com/raspberrypi/rpi-image-gen) checked out into a directory which is a sibling of this repo,
Build the pilot anchor image with 

    ./rpi-image-gen build \
      -S ../cranebot3-firmware/stringman-pilot-rpi-image/ \
      -c ../cranebot3-firmware/stringman-pilot-rpi-image/config/stringman.yaml \
      -- IGconf_device_user1pass='Fo0bar!!'

If an SD card reader is attached to the Pi that this was run on, you can immediately write the image with this command
but *make sure you are writing to the correct device*

    sudo rpi-imager --cli ~/rpi-image-gen/work/image-stringman-pilot/stringman-pilot.img /dev/sda

If it does not boot correctly but you can still get into it with ethernet, then check out the logs with

    journalctl -u resize-rootfs.service
    journalctl -u cranebot.service