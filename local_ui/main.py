import multiprocessing
import time
import sys
import argparse

from nf_robot.host.observer import start_observation
from ursina_app import start_ui

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CenteringNet System")
    parser.add_argument("--config", type=str, default='configuration.json')
    args = parser.parse_args()

    # Create and start the observer process
    # It runs a websocket server on localhost:4245 which the Ursina UI connects to.
    # because we pass terminate_with_ui=True, it will only run until the UI window closes
    observer_process = multiprocessing.Process(target=start_observation, args=(True,args.config), name='observer')
    # not deamonic so it can have children. (a pool)
    observer_process.daemon = False

    # todo use logging module in these processes.
    observer_process.start()

    # ursina expects this in a global scope in the main file and it won't be called anywhere else.
    def input(key):
        print('If this is called, we failed to hijack the global input function')

    def register_input(cpui):
        global input
        input = cpui.input

    try:
        # allow Ursina to be the main process. it doesn't work as a subprocess.
        ui = start_ui(register_input,args.config)
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminate both UI and observer")
        # TODO it seems to take two ctrl-c to kill this
        ui.end_update_thread()