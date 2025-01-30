import multiprocessing
import numpy as np
import time

from data_store import DataStore
from observer import start_observation
from position_estimator import start_estimator
from ursina_app import start_ui

if __name__ == "__main__":
    # a collection of shared arrays for storing measurements
    datastore = DataStore(horizon_s=10, n_cables=3)

    # Create and start the observer process
    observer_process = multiprocessing.Process(target=start_observation, args=(shared_array,))
    observer_process.daemon = True

    # error minimization process
    minimizer_process = multiprocessing.Process(target=start_estimator, args=(shared_array,))
    minimizer_process.daemon = True

    # add ui process if not in headless mode
    ui_process = multiprocessing.Process(target=start_ui)
    ui_process.daemon = True

    # todo use logging module in these processes.

    observer_process.start()
    minimizer_process.start()
    ui_process.start()

    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        observer_process.terminate()
        minimizer_process.terminate()
        ui_process.terminate()
        
        observer_process.join()
        minimizer_process.join()
        ui_process.join()