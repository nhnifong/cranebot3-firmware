import multiprocessing
import numpy as np
import time
import argparse

from data_store import DataStore
from observer import start_observation
from position_estimator import start_estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane Robot Controller")

    parser.add_argument("-h", "--headless", type=bool, help="When true, do not start the UI", default=False)
    args = parser.parse_args()

    # try:
    #   params = np.load("calibratrion_data.npz")
    # except IOError:
    #   params = calibrate_all();
    #   np.savez('calibration_data', params**)

    # a collection of shared arrays for storing measurements
    datastore = DataStore(horizon_s=10, n_cables=4)

    # a queue for sending updates from the minimizer to the UI
    # items are dictionaries with 1 or more updates to named values.
    # for example {'gripper_pos_spline_eval': [[t,x,y,z], [t,x,y,z], ...]}
    to_ui_q = multiprocessing.Queue()
    to_pe_q = multiprocessing.Queue()
    to_ob_q = multiprocessing.Queue()

    # Create and start the observer process
    observer_process = multiprocessing.Process(target=start_observation, args=(datastore, to_ui_q, to_pe_q, to_ob_q))
    observer_process.daemon = True

    # error minimization process
    minimizer_process = multiprocessing.Process(target=start_estimator, args=(datastore, to_ui_q, to_pe_q, to_ob_q))
    minimizer_process.daemon = True

    # add ui process if not in headless mode
    if not args.headless:
        from ursina_app import start_ui
        ui_process = multiprocessing.Process(target=start_ui, args=(to_ui_q, ))
        ui_process.daemon = True

    # todo use logging module in these processes.

    observer_process.start()
    minimizer_process.start()
    if not args.headless:
        ui_process.start()

    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        observer_process.terminate()
        minimizer_process.terminate()
        if not args.headless:
            ui_process.terminate()

        observer_process.join()
        minimizer_process.join()
        if not args.headless:
            ui_process.join()

# Other tasks not yet accounted for:
#   triggering the calibration process from the UI
#   Selecting the highest priority target based on images collected from all cameras
#   remembering information about previous targets
#   locating the bins and selecting the best bin for each payload.
#   incorporating grip estimating into the minimizer

# other interprocess communication that is not yet accounted for:
#   In the minimizer process evaluate the position splines over a linear space and send the result to the UI process for visualization
#   The observer process should send information to the UI about bot component connections statuses. And battery info about the gripper
#   The UI should read and visualize the measurements in the datastore.
#   The UI should be able to send manual goto commands as desired positions to the minimizer.
#   The task planner should send desired positions to the minimizer