import multiprocessing
import numpy as np
import time

from data_store import DataStore
from observer import start_observation

if __name__ == "__main__":
    # a collection of shared arrays for storing measurements
    datastore = DataStore(horizon_s=10, n_cables=3)

    # Create and start the observer process
    observer_process = multiprocessing.Process(target=start_observation, args=(shared_array,))
    observer_process.daemon = True # Set as a daemon process

    # add error minimization process

    # todo add ui process if not in headless mode

    # todo use logging module in these processes.

    observer_process.start()

    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        observer_process.terminate()
        observer_process.join()