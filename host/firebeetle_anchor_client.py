# this client is designed for the firebeetle based anchor, which is already deprecated
# it isnt self contained because it assumes the present of a few globals, but I dont
# plan on fixing it. just saving it for reference.
class FirebeetleAnchorClient:
    def __init__(self, anchor_num):
        self.origin_detections = []
        self.anchor_num = anchor_num # which anchor are we connected to
        try:
            # read calibration data from file
            saved_info = np.load('anchor_pose_%i' % self.anchor_num)
            self.anchor_pose = tuple(saved_info['pose'])
        except FileNotFoundError:
            self.anchor_pose = (np.array([0,0,0]), np.array([0,0,0]))

        # to help with a loop that does the same thing four times in handle_image
        # name, offset, datastore
        self.arucos = [
            ('gripper_front', model_constants.gripper_aruco_front_inv, datastore.gripper_pose),
            ('gripper_back', model_constants.gripper_aruco_back_inv, datastore.gripper_pose),
            ('gantry_front', model_constants.gantry_aruco_front_inv, datastore.gantry_pose),
            ('gantry_back', model_constants.gantry_aruco_back_inv, datastore.gantry_pose),
        ]

    def calibrate_pose(self):
        global to_ui_q
        global to_pe_q
        # recalculate the pose of the connected anchor from recent origin detections
        anchor_cam_pose = [invert_pose(*average_pose(det)) for det in self.origin_detections]
        self.anchor_pose = compose_poses([anchor_cam_pose, invert_pose(model_constants.gripper_camera)])
        np.savez('anchor_pose_%i' % self.anchor_num, pose = pose)
        to_ui_q.put({'anchor_pose': (self.anchor_num, pose)})
        to_pe_q.put({'anchor_pose': (self.anchor_num, pose)})


    def handle_image(self, headers, buf):
        """
        handle a single image from the stream
        """
        # Decode image
        timestamp = float(headers['X-Timestamp'])
        if buf[:2] != b'\xff\xd8': # start of image marker
            # discard broken frames. they have no header, and a premature end of image marker.
            # I have no idea what causes these but the browser can display them.
            # they must be some kind of intermediate frame, but I have no information of what the encoding is.
            # the bad news is that when the lights are on, almost all frames are of this type.
            print("broken frame")
            return

        frame = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            # within this scope, interpret the symbol calibration_mode as referring to the global calibration_mode
            global calibration_mode
            if calibration_mode:
                for detection in locate_markers(frame):
                    print(f"Found board: {detection.name}")
                    print(f"Timestamp: {timestamp}")
                    print(f"Rotation Vector: {detection.rvec}")
                    print(f"Translation Vector: {detection.tvec}")
                    # sys.stdout.flush()

                    if detection.name == "origin":
                        origin_detections.append(detection)
                        if len(origin_detections) > max_origin_detections:
                            origin_detections.pop(0)
            else:
                for detection in locate_markers(frame):
                    # rotate and translate to where that object's origin would be
                    # given the position and rotation of the camera that made this observation (relative to the origin)
                    # store the time and that position in the appropriate measurement array in observer.

                    for name, offset, dest  in self.arucos:
                        if detection.name == name:
                            # you have the pose of gripper_front relative to a particular anchor camera
                            # Anchor is relative to the origin
                            # anchor camera is relative to anchor
                            # gripper_front is relative to anchor camera
                            # gripper is relative to gripper_front
                            # gripper_grommet is relative to gripper
                            gripper_global_pose = np.array(compose_poses([
                                self.anchor_pose, # obtained from calibration
                                model_constants.anchor_camera, # constant
                                (detection.rotation, detection.translation), # the pose obtained just now
                                offset, # constant
                            ]))
                            dest.insert(np.concatenate([[timestamp], gripper_global_pose.reshape(6)]))


    def handle_json(self, headers, buf):
        """
        handle a single json blob from the stream
        """
        accel = json.loads(buf.decode())
        # datastore.imu_accel.insert(np.array([timestamp, accel['x'], accel['y'], accel['z']]))

    def parse_mixed_replace_stream(self, url):
        """
        Parses a multipart/x-mixed-replace stream using the requests library.
        Blocks until the stream closes.

        Args:
            url: The URL of the stream.

        Prints the content type of each part in the stream.
        """
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            print(response.headers)
            content_type = response.headers.get('Content-Type', '')
            if "multipart/x-mixed-replace" not in content_type:
                print("Error: Not a multipart/x-mixed-replace stream.")
                sys.stdout.flush()
                return
            boundary = content_type[content_type.find('boundary=')+9:]
            boundary_with_newlines = f"\r\n--{boundary}\r\n"
            print(repr(boundary_with_newlines))

            for part in response.iter_lines(chunk_size=2**20, decode_unicode=False, delimiter=boundary_with_newlines.encode()):
                headers = {}
                lines = part.split(b'\r\n')
                for line in lines:
                    line = line.decode()
                    s = line.split(': ')
                    if len(s) == 2:
                        headers[s[0]] = s[1]
                    else:
                        break
                if len(headers) == 0:
                    continue
                # the data begins after the first double newline
                data_start = part.find(b'\r\n\r\n')+4
                if headers['Content-Length'] == '0' or data_start == 3:
                    print('skipping part with zero data size')
                    continue

                if headers['Content-Type'] == 'image/jpeg':
                    self.handle_image(headers, part[data_start:])
                elif headers['Content-Type'] == 'application/json':
                    self.handle_json(headers, part[data_start:])
                else:
                    print(f"Got an unexpected content type {headers['Content-Type']}")
                    sys.stdout.flush()
            print("consumed stream")