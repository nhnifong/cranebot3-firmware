import asyncio
import subprocess

REMOTE_HOSTS = [
    "nathan@192.168.1.151",
    "nathan@192.168.1.152",
    "nathan@192.168.1.153",
    "nathan@192.168.1.154",
    "nathan@192.168.1.156",
]
FILES = [
    "spools.py",
    "motor_control.py",
    "anchor_server.py",
    "gripper_server.py",
]
REMOTE_DIR = "cranebot3-firmware"

# blocking version
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
        return False
    return True

async def copy_up(host):
    print(f"Copying files to {host}:{REMOTE_DIR}...")
    run_command(f"scp {" ".join(FILES)} {host}:{REMOTE_DIR}")
    # process = await asyncio.create_subprocess_shell(
    #     f"scp {" ".join(FILES)} {host}:{REMOTE_DIR}",
    #     stdout=asyncio.subprocess.PIPE,  # We still need these to create the process
    #     stderr=asyncio.subprocess.PIPE,  # Even if we don't use the output.
    # )
    # stdout, stderr = await process.communicate() # Get the output
    # if process.returncode != 0:
    #     print(f"Failed to copy to {host}. Command exited with error code {process.returncode}")
    #     print(stderr.decode())
    # return process.returncode  # Return the exit code

async def main():
    for host in REMOTE_HOSTS:
        asyncio.create_task(copy_up(host))

asyncio.run(main())