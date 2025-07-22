import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import socket
import io
import time
import os
import numpy as np
from functools import partial
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Texture, CardMaker, NodePath, PNMImage, WindowProperties,
    GraphicsPipe, PNMFileTypeRegistry, StringStream, LColor
)
from cv_common import marker_size, special_sizes

starting_port = 8888
ratio = 588/500 # marker occupies 500 px of the 588 px image width
origin_scale = special_sizes['origin'] * ratio
gantry_scale = marker_size * ratio

class RPiCamVidMock:
    """
    A mock class to simulate rpicam-vid for unit testing,
    rendering a 3D scene with a billboard using Panda3D and streaming MJPEG.
    """

    MJPEG_BOUNDARY = b"--frameboundary" # Standard MJPEG stream boundary

    def __init__(self, width: int, height: int, framerate: int,
                 gantry_initial_pose: tuple[float, float, float, float, float, float]):
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

        self.width = width
        self.height = height
        self.framerate = framerate
        self.background_color = (128/255, 128/255, 128/255, 1) # Gray background for Panda3D (RGBA 0-1)

        self.camera_poses = []
        self.gantry_initial_pose = gantry_initial_pose

        self.base: ShowBase = None # Panda3D ShowBase instance
        self._billboard_node: NodePath = None
        self._gantry_cube_node: NodePath = None

        self._frame_update_lock = asyncio.Lock() # Protects billboard pose/texture during updates
        self._frame_cb = None # user callback to run every time a frame is sent
        self._servers = []
        self._running: bool = False

    def set_camera_poses(self, camera_poses):
        if camera_poses.shape != (4, 6):
            raise ValueError("Expected 4 camera poses, each being 6 floats representing XYZHPR")
        self.camera_poses = camera_poses

    def _create_gantry_cube(self, image_texture_path: str) -> NodePath:
        """
        Creates a cube node showing the gantry texture on it's vertical faces
        """
        cube_root = NodePath("textured_cube_root")

        # Load the image texture for the vertical faces
        gantry_face_tex = self.base.loader.loadTexture(image_texture_path)

        faces = []
        for f in range(5):
            cm = CardMaker(f'{f}')
            cm.setFrame(-0.5, 0.5, -0.5, 0.5) # X, Z plane
            face = NodePath(cm.generate())
            face.setTwoSided(True)
            face.setTexture(gantry_face_tex)
            face.reparentTo(cube_root)
            faces.append(face)

        # Front Face (+Y)
        faces[0].setPos(0, 0.5, 0) # Move to +Y side

        # Back Face (-Y)
        faces[1].setPos(0, -0.5, 0) # Move to -Y side
        faces[1].setHpr(180, 0, 0) # Rotate 180 degrees around Z to face outwards

        # Right Face (+X)
        faces[2].setPos(0.5, 0, 0) # Move to +X side
        faces[2].setHpr(90, 0, 0) # Rotate 90 degrees around Z

        # Left Face (-X)
        faces[3].setPos(-0.5, 0, 0) # Move to -X side
        faces[3].setHpr(-90, 0, 0) # Rotate -90 degrees around Z

        # Top Face (+Z)
        faces[4].setPos(0, 0, 0.5) # Move to +Z side
        faces[4].setHpr(0, -90, 0) # Rotate -90 degrees around X to lie flat
        faces[4].setColor(LColor(1.0, 1.0, 1.0, 1.0)) # Set to blank white

        return cube_root

    def _setup_panda3d_scene(self):
        """
        Initializes the Panda3D environment, camera,
        stationary quad for the aruco board that marks the origin
        and a four sided object representing the gantry
        This method is called once when the server starts.
        """
        # Configure window properties for the offscreen buffer
        if len(self.camera_poses) == 0:
            raise ValueError("Server started before camera poses were set")

        props = WindowProperties()
        props.setSize(self.width, self.height)
        props.setCursorHidden(True) # Hide cursor in the window
        props.setTitle("RPiCamVidMock Panda3D Scene")

        # Initialize ShowBase in offscreen mode to avoid a visible window.
        # This is suitable for unit testing where a GUI isn't desired.
        self.base = ShowBase(windowType='offscreen')
        self.base.win.setClearColor(self.background_color) # Set background color

        # Set camera pose to 0th camera
        cam_x, cam_y, cam_z, cam_h, cam_p, cam_r = self.camera_poses[0]
        self.base.camera.setPos(cam_x, cam_y, cam_z)
        self.base.camera.setHpr(cam_h, cam_p, cam_r)

        # Create a quad to show the origin card
        cm = CardMaker('origin_aruco')
        # Create a unit square from -0.5 to 0.5. The actual size will be controlled by scale.
        cm.setFrame(-0.5, 0.5, -0.5, 0.5)
        self._billboard_node = NodePath(cm.generate())
        self._billboard_node.reparentTo(self.base.render) # Attach to the main scene graph
        self._billboard_node.setTexture(self.base.loader.loadTexture('../boards/origin.png'))
        self._billboard_node.setPos(0, 0, 0)
        self._billboard_node.setHpr(0, 0, 0)
        self._billboard_node.setScale(origin_scale)
        self._billboard_node.setLightOff()
        self._billboard_node.setShaderOff()

        # Create the gantry cube model
        self._gantry_cube_node = self._create_gantry_cube('../boards/gantry_front.png')
        self._gantry_cube_node.reparentTo(self.base.render) # Attach to the main scene graph
        
        # Set initial cube pose
        cube_x, cube_y, cube_z, cube_h, cube_p, cube_r = self.gantry_initial_pose
        self._gantry_cube_node.setPos(cube_x, cube_y, cube_z)
        self._gantry_cube_node.setHpr(cube_h, cube_p, cube_r)
        self._gantry_cube_node.setScale(gantry_scale)
        self._gantry_cube_node.setLightOff()
        self._gantry_cube_node.setShaderOff()

    async def _generate_frame(self, camera_num) -> bytes:
        """
        Generates a single MJPEG frame by rendering the Panda3D scene from a particular camera

        Returns:
            bytes: The JPEG image data.
        """
        if not self.base:
            print("Panda3D base not initialized, cannot generate frame.")
            return b''

        # set camera pose to selected cam
        cam_x, cam_y, cam_z, cam_h, cam_p, cam_r = self.camera_poses[camera_num]
        self.base.camera.setPos(cam_x, cam_y, cam_z)
        self.base.camera.setHpr(cam_h, cam_p, cam_r)

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

    def update_gantry_pose(self, pose: tuple[float, float, float, float, float, float]):
        """
        Updates the gantry 3D pose (position and HPR).

        Args:
            pose (tuple[float, float, float, float, float, float], optional):
                    New (x, y, z, h, p, r) for the billboard.
        """
        self.gantry_initial_pose = pose
        x, y, z, h, p, r = pose
        self._gantry_cube_node.setPos(x, y, z)
        self._gantry_cube_node.setHpr(h, p, r)
        print(f"Gantry pose updated to: Pos({x:.2f},{y:.2f},{z:.2f}), Hpr({h:.2f},{p:.2f},{r:.2f})")

    async def _handle_client(self, cam_num, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handles a single client connection, streaming MJPEG frames.
        
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
                    jpeg_frame = await self._generate_frame(cam_num)

                # Send MJPEG part headers and data
                writer.write(self.MJPEG_BOUNDARY + b"\r\n")
                writer.write(b"Content-Type: image/jpeg\r\n")
                writer.write(b"Content-Length: " + str(len(jpeg_frame)).encode() + b"\r\n")
                writer.write(b"\r\n") # End of headers for this part
                writer.write(jpeg_frame)
                writer.write(b"\r\n") # End of this part
                await writer.drain()

                # Control framerate by sleeping if necessary
                elapsed_time = time.monotonic() - start_time
                sleep_time = (1.0 / self.framerate) - elapsed_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Client {peername} disconnected: {e}")
        finally:
            print(f"Closing connection for client: {peername}")
            writer.close()

    async def start_server(self):
        """
        Starts an asynchronous TCP server to stream MJPEG video for every camera pose
        This method should be awaited in an asyncio event loop.
        """
        assert not self._running

        self._running = True
        try:
            # Initialize Panda3D scene *before* starting the server loop
            self._setup_panda3d_scene()

            # start one server for each camera. the one you connect you determines the view you get.
            for i in range(len(self.camera_poses)):
                self._servers.append(await asyncio.start_server(
                    # this handler always renders from camera i
                    partial(self._handle_client, i), '127.0.0.1', starting_port+i
                ))
                addr = self._servers[-1].sockets[0].getsockname()
                print(f"RPiCamVidMock server listening on {addr} play with ffplay tcp://{addr[0]}:{addr[1]}")

            # Wait for all servers to close.
            await asyncio.gather(*[s.serve_forever() for s in self._servers])

        except asyncio.CancelledError:
            print("RPiCamVidMock server task cancelled.")
        finally:
            self._running = False
            print("RPiCamVidMock server stopped.")
            if self.base:
                self.base.destroy() # Ensure Panda3D resources are properly cleaned up

    def stop_server(self):
        """
        Stops the MJPEG streaming servers and cleans up Panda3D resources.
        """
        if self._servers:
            print("Stopping RPiCamMultiViewServer...")
            for server in self._servers:
                server.close()
            self._running = False
            print("Server stop initiated.")
        if self.base:
            self.base.destroy() # Explicitly destroy Panda3D base on stop


# --- Example Usage (for testing the class) ---
async def main():
    """
    Example async function to demonstrate the RPiCamVidMock usage.
    """

    mock_server = RPiCamVidMock(
        width=800, height=600, framerate=5,
        gantry_initial_pose=(0.5, 0.5, 1, 0, 0, 0) # off center, 1m from floor
    )
    mock_server.set_camera_poses(np.array(
        # [[ 3.01131371e+00,  2.93494618e+00,  3.01700000e+00,
        # -1.18000000e+02,  3.81666562e-14,  1.35000000e+02],
        [(0, -4, 2, 0, -20, 0),
        [ 2.93494618e+00, -3.01131371e+00,  3.01700000e+00,
        -1.18000000e+02,  2.54444375e-14,  4.50000000e+01],
        [-2.93494618e+00,  3.01131371e+00,  3.01700000e+00,
        -1.18000000e+02,  0.00000000e+00, -1.35000000e+02],
        [-3.01131371e+00, -2.93494618e+00,  3.01700000e+00,
        -1.18000000e+02, -1.27222187e-14, -4.50000000e+01]]
    ))

    try:
        await mock_server.start_server()
    except asyncio.CancelledError:
        print("Main task cancelled.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping server...")
    finally:
        # Ensure the server is stopped gracefully
        mock_server.stop_server()
        print("RPiCamVidMock demonstration finished.")

if __name__ == "__main__":
    # To run this, you need to have Panda3D and Pillow installed:
    # pip install panda3d Pillow
    # Run from your terminal: python your_script_name.py
    asyncio.run(main())
