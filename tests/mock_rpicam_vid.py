import asyncio
import socket
import io
import time
import os
from PIL import Image, ImageDraw, ImageFont # Pillow for creating dummy image files

# Panda3D imports
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Texture, CardMaker, NodePath, PNMImage, WindowProperties, GraphicsPipe, PNMFileTypeRegistry, StringStream

class RPiCamVidMock:
    """
    A mock class to simulate rpicam-vid for unit testing,
    rendering a 3D scene with a billboard using Panda3D and streaming MJPEG.
    """

    MJPEG_BOUNDARY = b"--frameboundary" # Standard MJPEG stream boundary

    def __init__(self, width: int, height: int, framerate: int, port: int,
                 initial_image_filepath: str,
                 camera_pose: tuple[float, float, float, float, float, float],
                 billboard_initial_pose: tuple[float, float, float, float, float, float]):
        """
        Initializes the RPiCamVidMock server with Panda3D scene setup.

        Args:
            width (int): The width of the video stream in pixels.
            height (int): The height of the video stream in pixels.
            framerate (int): The target framerate (frames per second).
            port (int): The TCP port to listen on for client connections.
            initial_image_filepath (str): Path to the initial image file for the billboard.
            camera_pose (tuple): Initial (x, y, z, h, p, r) for the camera in the 3D scene.
            billboard_initial_pose (tuple): Initial (x, y, z, h, p, r) for the billboard in the 3D scene.
        """
        if not all(isinstance(arg, int) and arg > 0 for arg in [width, height, framerate, port]):
            raise ValueError("Width, height, framerate, and port must be positive integers.")
        if not os.path.exists(initial_image_filepath):
            raise FileNotFoundError(f"Initial image file not found: {initial_image_filepath}")
        if not (isinstance(camera_pose, tuple) and len(camera_pose) == 6 and
                all(isinstance(coord, (int, float)) for coord in camera_pose)):
            raise ValueError("camera_pose must be a tuple of 6 floats (x, y, z, h, p, r).")
        if not (isinstance(billboard_initial_pose, tuple) and len(billboard_initial_pose) == 6 and
                all(isinstance(coord, (int, float)) for coord in billboard_initial_pose)):
            raise ValueError("billboard_initial_pose must be a tuple of 6 floats (x, y, z, h, p, r).")

        self.width = width
        self.height = height
        self.framerate = framerate
        self.port = port
        self.background_color = (128/255, 128/255, 128/255, 1) # Gray background for Panda3D (RGBA 0-1)

        self.initial_image_filepath = initial_image_filepath
        self.camera_pose = camera_pose
        self.billboard_initial_pose = billboard_initial_pose

        self.base: ShowBase = None # Panda3D ShowBase instance
        self._billboard_node: NodePath = None # NodePath for the billboard
        self._current_texture: Texture = None # Current texture applied to billboard
        self._current_image_filepath: str = initial_image_filepath # Track current image path

        self._frame_update_lock = asyncio.Lock() # Protects billboard pose/texture during updates
        self._frame_cb = None # user callback to run every time a frame is sent
        self._server: asyncio.Server = None
        self._running: bool = False

        print(f"RPiCamVidMock initialized: {self.width}x{self.height}@{self.framerate}fps on port {self.port}")

    def _setup_panda3d_scene(self):
        """
        Initializes the Panda3D environment, camera, and the billboard.
        This method is called once when the server starts.
        """
        # Configure window properties for the offscreen buffer
        props = WindowProperties()
        props.setSize(self.width, self.height)
        props.setCursorHidden(True) # Hide cursor in the window
        props.setTitle("RPiCamVidMock Panda3D Scene")

        # Initialize ShowBase in offscreen mode to avoid a visible window.
        # This is suitable for unit testing where a GUI isn't desired.
        self.base = ShowBase(windowType='offscreen')
        self.base.win.setClearColor(self.background_color) # Set background color

        # Set camera pose
        cam_x, cam_y, cam_z, cam_h, cam_p, cam_r = self.camera_pose
        self.base.camera.setPos(cam_x, cam_y, cam_z)
        self.base.camera.setHpr(cam_h, cam_p, cam_r)

        # Create the billboard geometry (a simple quad)
        cm = CardMaker('billboard')
        # Create a unit square from -0.5 to 0.5. The actual size will be controlled by scale.
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        self._billboard_node = NodePath(cm.generate())
        self._billboard_node.reparentTo(self.base.render) # Attach to the main scene graph

        # Load initial texture from the provided file path
        try:
            self._current_texture = self.base.loader.loadTexture(self.initial_image_filepath)
            self._billboard_node.setTexture(self._current_texture)
        except Exception as e:
            print(f"Error loading initial texture from {self.initial_image_filepath}: {e}")
            # Fallback: create a simple solid color texture if image loading fails
            fallback_texture = Texture()
            fallback_texture.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgba)
            fallback_texture.setRamImage(b'\xFF\x00\x00\xFF') # Red pixel
            self._billboard_node.setTexture(fallback_texture)
            print("Using a red fallback texture for the billboard.")

        # Set initial billboard pose (position and HPR)
        bill_x, bill_y, bill_z, bill_h, bill_p, bill_r = self.billboard_initial_pose
        self._billboard_node.setPos(bill_x, bill_y, bill_z)
        self._billboard_node.setHpr(bill_h, bill_p, bill_r)
        # Set a default scale for the billboard, adjust as needed for visibility
        self._billboard_node.setScale(2.0) # Make it 2 units wide/tall

        # Ensure the billboard is not affected by scene lighting or shaders for simplicity
        self._billboard_node.setLightOff()
        self._billboard_node.setShaderOff()

    async def _generate_frame(self) -> bytes:
        """
        Generates a single MJPEG frame by rendering the Panda3D scene.

        Returns:
            bytes: The JPEG image data.
        """
        if not self.base:
            print("Panda3D base not initialized, cannot generate frame.")
            return b''

        # Render a single frame of the Panda3D scene
        self.base.graphicsEngine.renderFrame()
        # Process Panda3D's internal tasks (e.g., updates, events).
        # This is crucial for Panda3D to advance its state.
        self.base.taskMgr.step()

        # Get screenshot from the Panda3D window (offscreen buffer)
        pnm_image = PNMImage()
        # The getScreenshot method captures the current frame buffer content.
        self.base.win.getScreenshot(pnm_image)

        # Convert the PNMImage to JPEG bytes using Panda3D's StringStream
        string_stream = StringStream()
        # Write to the StringStream using the correct PNMFileType
        pnm_image.write(string_stream, 'mock_frame.jpg', PNMFileTypeRegistry.get_global_ptr().get_type_from_extension('jpeg'))
        # Get the bytes from the StringStream
        jpeg_data = string_stream.getData()
        return jpeg_data

    async def update_image_and_transform(self, image_filepath: str = None, billboard_pose: tuple[float, float, float, float, float, float] = None):
        """
        Updates the test image (texture) and its 3D pose (position and HPR).
        This function is thread-safe.

        Args:
            image_filepath (str, optional): New image file path for the billboard texture.
                                            If None, the current image is retained.
            billboard_pose (tuple[float, float, float, float, float, float], optional):
                                            New (x, y, z, h, p, r) for the billboard.
                                            If None, current pose is retained.
        """
        async with self._frame_update_lock:
            if image_filepath is not None and image_filepath != self._current_image_filepath:
                if self.base:
                    try:
                        if not os.path.exists(image_filepath):
                            print(f"Warning: New image file not found: {image_filepath}. Keeping current texture.")
                        else:
                            new_texture = self.base.loader.loadTexture(image_filepath)
                            self._billboard_node.setTexture(new_texture)
                            self._current_texture = new_texture
                            self._current_image_filepath = image_filepath
                            print(f"Billboard texture updated to: {image_filepath}")
                    except Exception as e:
                        print(f"Error loading new texture from {image_filepath}: {e}. Keeping current texture.")
                else:
                    print("Panda3D base not initialized, cannot update texture.")

            if billboard_pose is not None:
                if self.base and self._billboard_node:
                    if not (isinstance(billboard_pose, tuple) and len(billboard_pose) == 6 and
                            all(isinstance(coord, (int, float)) for coord in billboard_pose)):
                        print(f"Warning: Invalid billboard_pose format. Expected (float, float, float, float, float, float). Got {billboard_pose}. Keeping current pose.")
                    else:
                        x, y, z, h, p, r = billboard_pose
                        self._billboard_node.setPos(x, y, z)
                        self._billboard_node.setHpr(h, p, r)
                        print(f"Billboard pose updated to: Pos({x:.2f},{y:.2f},{z:.2f}), Hpr({h:.2f},{p:.2f},{r:.2f})")
                else:
                    print("Panda3D base or billboard node not initialized, cannot update pose.")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handles a single client connection, streaming MJPEG frames.
        This simplified version assumes only one client is expected.
        """
        peername = writer.get_extra_info('peername')
        print(f"Client connected: {peername}")

        # Send initial HTTP headers for MJPEG multipart stream.
        # This is common for browser-based MJPEG viewers.
        try:
            writer.write(b"HTTP/1.0 200 OK\r\n")
            writer.write(b"Content-Type: multipart/x-mixed-replace; boundary=" + self.MJPEG_BOUNDARY + b"\r\n")
            writer.write(b"\r\n")
            await writer.drain()
        except Exception as e:
            print(f"Error sending initial headers to {peername}: {e}")
            writer.close()
            return

        try:
            while self._running:
                start_time = time.monotonic()

                # Generate frame (thread-safe with the lock)
                async with self._frame_update_lock:
                    jpeg_frame = await self._generate_frame()

                # call user function
                self._frame_cb()

                # Send MJPEG part headers and data
                writer.write(self.MJPEG_BOUNDARY + b"\r\n")
                writer.write(b"Content-Type: image/jpeg\r\n")
                writer.write(b"Content-Length: " + str(len(jpeg_frame)).encode() + b"\r\n")
                writer.write(b"\r\n") # End of headers for this part
                writer.write(jpeg_frame)
                writer.write(b"\r\n") # End of this part
                await writer.drain()
                print('sent frame to client')

                # Control framerate by sleeping if necessary
                elapsed_time = time.monotonic() - start_time
                sleep_time = (1.0 / self.framerate) - elapsed_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Client {peername} disconnected: {e}")
        # except Exception as e:
        #     print(f"Error handling client {peername}: {e}")
        finally:
            print(f"Closing connection for client: {peername}")
            writer.close()

    def set_frame_cb(self, frame_cb):
        self._frame_cb = frame_cb

    async def start_server(self):
        """
        Starts the asynchronous TCP server to stream MJPEG video.
        This method should be awaited in an asyncio event loop.
        """
        assert not self._running

        self._running = True
        try:
            # Initialize Panda3D scene *before* starting the server loop
            self._setup_panda3d_scene()

            self._server = await asyncio.start_server(
                self._handle_client, '127.0.0.1', self.port
            )
            addr = self._server.sockets[0].getsockname()
            print(f"RPiCamVidMock server listening on {addr}")

            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            print("Server task cancelled.")
        except Exception as e:
            print(f"Error starting or running server: {e}")
        finally:
            self._running = False
            print("RPiCamVidMock server stopped.")
            if self.base:
                self.base.destroy() # Ensure Panda3D resources are properly cleaned up

    def stop_server(self):
        """
        Stops the MJPEG streaming server and cleans up Panda3D resources.
        """
        if self._server:
            print("Stopping RPiCamVidMock server...")
            self._server.close()
            self._running = False
            print("Server stop initiated.")
        if self.base:
            self.base.destroy() # Explicitly destroy Panda3D base on stop


# --- Example Usage (for testing the class) ---
async def main():
    """
    Example async function to demonstrate the RPiCamVidMock usage.
    """
    # Create a dummy image file for the billboard
    dummy_image_path = "test_billboard_image.jpg"
    if not os.path.exists(dummy_image_path):
        img = Image.new('RGB', (256, 256), color='red')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 50)
        except IOError:
            font = ImageFont.load_default()
        d.text((30, 100), "Panda3D", fill=(255, 255, 255), font=font)
        img.save(dummy_image_path, format='JPEG')
        print(f"Created dummy image: {dummy_image_path}")

    mock_server = RPiCamVidMock(
        width=800, height=600, framerate=15, port=8888,
        initial_image_filepath=dummy_image_path,
        camera_pose=(0, -10, 0, 0, 0, 0), # Camera at (0, -10, 0), looking at origin
        billboard_initial_pose=(0, 0, 0, 0, 0, 0) # Billboard at origin, no rotation
    )

    # Start the server as a background task
    server_task = asyncio.create_task(mock_server.start_server())

    print(f"RPiCamVidMock server started on port {mock_server.port}")
    print("You can now connect to tcp://0.0.0.0:8888 (e.g., using VLC or a browser)")
    print("In a browser, navigate to http://localhost:8888 to see the stream (some browsers might require specific extensions for MJPEG).")
    print("VLC: Media -> Open Network Stream -> http://localhost:8888")

    try:
        # Let the server run with the default image and pose for a few seconds
        print("\nRunning with default image and pose for 5 seconds...")
        await asyncio.sleep(5)

        # --- Scenario 1: Update billboard pose (position) ---
        print("\nMoving billboard to (5, 0, 0) and rotating 45 degrees around Z...")
        await mock_server.update_image_and_transform(billboard_pose=(5, 0, 0, 45, 0, 0))
        await asyncio.sleep(5)

        # --- Scenario 2: Update billboard pose (rotation) ---
        print("\nRotating billboard to 90 degrees around X...")
        await mock_server.update_image_and_transform(billboard_pose=(5, 0, 0, 45, 90, 0))
        await asyncio.sleep(5)

        # --- Scenario 3: Update image content and reset pose ---
        print("\nUpdating billboard image to a green square with 'NEW' text and resetting pose...")
        new_dummy_image_path = "new_test_billboard_image.jpg"
        img = Image.new('RGB', (200, 100), color='green')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        d.text((20, 30), "NEW IMAGE", fill=(255, 255, 255), font=font)
        img.save(new_dummy_image_path, format='JPEG')
        print(f"Created new dummy image: {new_dummy_image_path}")

        while True:
            await mock_server.update_image_and_transform(image_filepath=new_dummy_image_path,
                                                         billboard_pose=(0, 0, 0, 0, 0, 0)) # Reset pose
            await asyncio.sleep(5)

            # --- Scenario 4: Move billboard up and rotate ---
            print("\nMoving billboard up to (0, 0, 2) and rotating 180 degrees around Y...")
            await mock_server.update_image_and_transform(billboard_pose=(0, 0, 2, 0, 180, 0))
            await asyncio.sleep(5)


    except asyncio.CancelledError:
        print("Main task cancelled.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping server...")
    finally:
        # Ensure the server is stopped gracefully
        mock_server.stop_server()
        # Wait for the server task to complete its shutdown
        await server_task
        # Clean up dummy image files
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
        if os.path.exists(new_dummy_image_path):
            os.remove(new_dummy_image_path)
        print("RPiCamVidMock demonstration finished.")

if __name__ == "__main__":
    # To run this, you need to have Panda3D and Pillow installed:
    # pip install panda3d Pillow
    # Run from your terminal: python your_script_name.py
    asyncio.run(main())
