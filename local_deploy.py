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
    if not run_command(f"scp {" ".join(FILES)} {host}:{REMOTE_DIR}"):
        print(f"Failed to copy to {host}. Skipping restart.")

async def main():
    for host in REMOTE_HOSTS:
        asyncio.create_task(copy_up(host))

asyncio.run(main())