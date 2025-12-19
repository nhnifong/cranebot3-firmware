import pyarrow as pa
import numpy as np

# Send frames from one process to another using pyarrow

class FrameIPC:
	def __init__(self, identifier: string, num_buffers: int=3, frame_shape, frame_dtype):
		self.identifier = identifier
		self.num_buffers = num_buffers
		self.buffer_index = 0
		self.buffer_ids = [create_shared_memory_id(i) for i in range(num_buffers)]

		# Pre-allocate buffers
	    dummy_frame = np.empty(frame_shape, dtype=frame_dtype)
	    dummy_tensor = pa.Tensor.from_numpy(dummy_frame)
	    frame_size_bytes = dummy_tensor.size

		# List to hold the PyArrow output streams (one for each buffer ID)
	    self.output_streams = {}
	    
	    for buffer_id in self.buffer_ids:
	        # Create the named shared memory block
	        output_streams[buffer_id] = pa.output_stream(buffer_id, size=frame_size_bytes)

	def create_shared_memory_id(self, index):
	    """Generates a unique, OS-level ID for a shared memory segment."""
	    # Using the process ID of the parent ensures uniqueness if multiple instances run
	    # as well as the provided identifier because multiple FrameIPC may send from the same process
	    return f"/video_frame_buf_{os.getpid()}_{self.identifier}_{index}"

	def cleanup_shared_memory(self):
	    """Ensures all OS-level shared memory segments are unlinked (deleted)."""
	    print("\n🧹 Cleaning up shared memory resources...")
	    for buffer_id in self.buffer_ids:
	        try:
	            # pa.os_memory_region().close() is the pyarrow way to unlink the segment
	            pa.os_memory_region(buffer_id).close()
	            print(f"  Unlinked segment: {buffer_id}")
	        except Exception as e:
	            # Ignore errors if the segment was already unlinked or never created
	            # This is important for robust termination.
	            pass
	    print("🧹 Cleanup complete.")

	def send_frame(self, raw_frame: np.ndarray):
		tensor = pa.Tensor.from_numpy(raw_frame)
      	
      	# pick the next bufffer whether or not the UI is done with it.
      	# because we are prioritizing low latency, we would rather the UI's appearance be corrupted than hold up the receipt of new frames.
      	self.buffer_index = (self.buffer_index + 1) % self.num_buffers
		buffer_id = buffer_ids[buffer_index]

        stream = output_streams[buffer_id]
        stream.seek(0) # Reset stream pointer to the beginning of the memory segment
        stream.write(tensor.to_buffer())

        # caller must notify other process via whatever side channel is available to it
        return buffer_id

def retrieve_frame(buffer_id):
	stream = input_streams[buffer_id]
    stream.seek(0)
    received_tensor = pa.read_tensor(stream)
    received_frame = received_tensor.to_numpy()
    return received_frame