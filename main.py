import multiprocessing
import numpy as np
import time
import argparse

from data_store import DataStore
from observer import start_observation
from ursina_app import start_ui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane Robot Controller")
    parser.add_argument("--headless", type=bool, help="When true, do not start the UI", default=False)
    args = parser.parse_args()

    # a queue for sending updates to the UI
    # items are dictionaries with 1 or more updates to named values.
    # for example {'gripper_pos_spline_eval': [[t,x,y,z], [t,x,y,z], ...]}
    to_ui_q = multiprocessing.Queue()
    to_ob_q = multiprocessing.Queue()

    # we don't want to block joining the thread that flushes data to these queues.
    # it has a tendency to never finish and I dont know why.
    to_ui_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    # Create and start the observer process
    observer_process = multiprocessing.Process(target=start_observation, args=(to_ui_q, to_ob_q), name='observer')
    # not deamonic so it can have children. (a pool)
    # but it is basically an infinite loop and needs to be killed by sending STOP
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
        if not args.headless:
            # allow Ursina to be the main process. it doesn't work as a subprocess.
            start_ui(to_ui_q, to_ob_q, register_input)
        else:
            # Keep the main process alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        to_ob_q.put({'STOP':None})
        if not args.headless:
            to_ui_q.put({'STOP':None})

        observer_process.join()
        if not args.headless:
            ui_process.join()