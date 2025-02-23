from multiprocessing import Pool, Process
from time import sleep

def new_fn(pool):
	result = pool.apply_async(print, ('a pool worker says hello',))
	result.get()

with Pool(processes=1) as pool:
	new_process = Process(target=new_fn, args=(pool,))
	# observer_process.daemon = True
	new_process.start()

	# keep parent process alive for long enough to see the result
	new_process.join()