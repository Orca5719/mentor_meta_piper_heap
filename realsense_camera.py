"""
RealSense Camera Interface
===========================
Wraps pyrealsense2 to provide a simple capture interface for the
PiPER real robot environment. Supports color + depth streaming,
with optional alignment and cropping.

Dependencies:
    pip install pyrealsense2 opencv-python numpy
"""

import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 is required. Install: pip install pyrealsense2")

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required. Install: pip install opencv-python")


class RealSenseCamera:
    """Intel RealSense camera wrapper with color + depth streaming."""

    def __init__(
        self,
        serial_number: str = None,
        color_resolution: tuple = (640, 480),
        depth_resolution: tuple = (640, 480),
        fps: int = 30,
        align_depth_to_color: bool = True,
        crop_size: tuple = None,  # (h, w) center-crop after capture
        target_size: tuple = (84, 84),  # final resize for RL input
    ):
        self._serial = serial_number
        self._color_res = color_resolution
        self._depth_res = depth_resolution
        self._fps = fps
        self._align_depth = align_depth_to_color
        self._crop_size = crop_size
        self._target_size = target_size
        self._pipeline = None
        self._align = None
        self._connected = False

    def connect(self):
        """Start the RealSense pipeline."""
        self._pipeline = rs.pipeline()
        config = rs.config()

        # Select specific device by serial number if provided
        if self._serial is not None:
            config.enable_device(self._serial)

        config.enable_stream(
            rs.stream.color,
            self._color_res[0], self._color_res[1],
            rs.format.bgr8, self._fps
        )
        config.enable_stream(
            rs.stream.depth,
            self._depth_res[0], self._depth_res[1],
            rs.format.z16, self._fps
        )

        self._profile = self._pipeline.start(config)

        # Alignment object
        if self._align_depth:
            self._align = rs.align(rs.stream.color)

        # Wait for a few frames to stabilize auto-exposure
        print("[RealSense] Warming up camera...")
        for _ in range(30):
            self._pipeline.wait_for_frames()

        self._connected = True
        print("[RealSense] Connected and streaming")

    def disconnect(self):
        """Stop the RealSense pipeline."""
        if self._pipeline is not None and self._connected:
            self._pipeline.stop()
            self._connected = False
            print("[RealSense] Disconnected")

    def capture(self) -> np.ndarray:
        """Capture a single color frame, resized to target_size.

        Returns:
            np.ndarray of shape (target_h, target_w, 3), dtype uint8, RGB order
        """
        return self._capture_resized(self._target_size)

    def capture_full(self) -> np.ndarray:
        """Capture a single color frame at full sensor resolution (no resize).

        Returns:
            np.ndarray of shape (h, w, 3), dtype uint8, RGB order
        """
        return self._capture_resized(None)

    def _capture_resized(self, target_size) -> np.ndarray:
        """Internal: capture a color frame, optionally resized."""
        assert self._connected, "Camera not connected. Call connect() first."
        frames = self._pipeline.wait_for_frames()

        if self._align_depth:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            if target_size is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return np.zeros(target_size + (3,), dtype=np.uint8)

        color_img = np.asanyarray(color_frame.get_data())  # BGR, uint8

        # Center crop if specified
        if self._crop_size is not None:
            h, w = color_img.shape[:2]
            ch, cw = self._crop_size
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            color_img = color_img[y0:y0+ch, x0:x0+cw]

        # Resize to target (if specified)
        if target_size is not None:
            color_img = cv2.resize(color_img, target_size)

        # BGR → RGB
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        return color_img

    def capture_with_depth(self):
        """Capture color + aligned depth frames.

        Returns:
            color: (target_h, target_w, 3), uint8, RGB
            depth: (target_h, target_w), float32, in meters
        """
        assert self._connected, "Camera not connected. Call connect() first."
        frames = self._pipeline.wait_for_frames()

        if self._align_depth:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_img = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame else None

        if color_img is not None:
            if self._crop_size is not None:
                h, w = color_img.shape[:2]
                ch, cw = self._crop_size
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
                color_img = color_img[y0:y0+ch, x0:x0+cw]
            color_img = cv2.resize(color_img, self._target_size)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        if depth_img is not None:
            if self._crop_size is not None:
                h, w = depth_img.shape[:2]
                ch, cw = self._crop_size
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
                depth_img = depth_img[y0:y0+ch, x0:x0+cw]
            depth_img = cv2.resize(depth_img, self._target_size, interpolation=cv2.INTER_NEAREST)
            # Convert from raw depth units (typically 0.001m = 1mm) to meters
            depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
            depth_img = depth_img.astype(np.float32) * depth_scale

        return color_img, depth_img

    def get_depth_scale(self) -> float:
        """Get depth scale factor (raw → meters)."""
        return self._profile.get_device().first_depth_sensor().get_depth_scale()

    def get_intrinsics(self) -> np.ndarray:
        """Get color camera intrinsics as 3x3 matrix.

        Returns:
            K: 3x3 camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        """
        color_stream = self._profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        return K

    def get_extrinsics_depth_to_color(self) -> np.ndarray:
        """Get extrinsics from depth to color as 4x4 homogeneous matrix."""
        depth_stream = self._profile.get_stream(rs.stream.depth)
        color_stream = self._profile.get_stream(rs.stream.color)
        extrinsics = depth_stream.get_extrinsics_to(color_stream)
        R = np.array(extrinsics.rotation).reshape(3, 3)
        t = np.array(extrinsics.translation).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t
        return T


class RealSenseCalibrator:
    """Hand-eye calibration tool for RealSense + PiPER setup.

    Performs eye-on-base (camera fixed, looking at robot) calibration
    using the Tsai-Lenz method. Collects pairs of:
      - Robot end-effector poses (from PiperRobot)
      - Corresponding checkerboard poses (from RealSense)

    Usage:
        calibrator = RealSenseCalibrator(camera, robot)
        calibrator.collect_sample()    # repeat N >= 3 times
        calibrator.calibrate()         # compute X (camera-to-base transform)
        calibrator.save("calibration.npy")
    """

    def __init__(self, camera: RealSenseCamera, robot=None, checkerboard_size=(8, 6), square_size=0.025):
        self._camera = camera
        self._robot = robot
        self._cb_size = checkerboard_size
        self._sq_size = square_size  # meters
        self._robot_poses = []       # list of 4x4 homogeneous matrices
        self._checker_poses = []     # list of 4x4 homogeneous matrices

        # Prepare checkerboard 3D points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:checkerboard_size[0], 0:checkerboard_size[1]
        ].T.reshape(-1, 2) * square_size
        self._objp = objp

    def detect_checkerboard(self, color_img: np.ndarray):
        """Detect checkerboard in a color image.

        Returns:
            rvec, tvec: rotation and translation vectors (OpenCV format), or None if not found
            corners: detected corner points, or None
        """
        gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self._cb_size, None)

        if not found:
            return None, None, None

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Solve PnP
        K = self._camera.get_intrinsics()
        dist = np.zeros(5)  # Assume rectified
        ret, rvec, tvec = cv2.solvePnP(self._objp, corners, K, dist)

        if not ret:
            return None, None, None

        return rvec, tvec, corners

    def collect_sample(self):
        """Collect one calibration sample: detect checkerboard + record robot pose.

        The robot should be holding or positioned near the checkerboard.
        Call this multiple times with the robot in different poses.
        """
        color_img = self._camera.capture()

        # Use full-res for calibration, not the 84x84 resized version
        # Capture a fresh high-res frame
        frames = self._camera._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("[Calibrator] No color frame")
            return False
        color_img = np.asanyarray(color_frame.get_data())
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        rvec, tvec, corners = self.detect_checkerboard(color_img)

        if rvec is None:
            print("[Calibrator] Checkerboard not detected. Try again.")
            return False

        # Convert rvec/tvec to 4x4 homogeneous matrix
        R, _ = cv2.Rodrigues(rvec)
        T_checker = np.eye(4)
        T_checker[:3, :3] = R
        T_checker[:3, 3] = tvec.flatten()

        # Get robot end-effector pose
        if self._robot is None:
            print("[Calibrator] No robot connected")
            return False

        pose = self._robot.get_end_pose()  # [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]
        T_robot = self._pose_to_matrix(pose)

        self._robot_poses.append(T_robot)
        self._checker_poses.append(T_checker)

        print(f"[Calibrator] Sample {len(self._robot_poses)} collected. "
              f"Checker pos: [{tvec[0,0]:.3f}, {tvec[1,0]:.3f}, {tvec[2,0]:.3f}]m, "
              f"Robot pos: [{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}]mm")
        return True

    def calibrate(self) -> np.ndarray:
        """Compute hand-eye calibration (eye-on-base configuration).

        Solves: A * X = X * B  →  X = T_base_to_camera

        Uses cv2.calibrateHandEye with Tsai-Lenz method.

        Returns:
            X: 4x4 homogeneous transformation (base → camera)
        """
        if len(self._robot_poses) < 3:
            raise ValueError(f"Need at least 3 samples, got {len(self._robot_poses)}")

        # Compute relative motions
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_robot, T_checker in zip(self._robot_poses, self._checker_poses):
            R_gripper2base.append(T_robot[:3, :3])
            t_gripper2base.append(T_robot[:3, 3])
            R_target2cam.append(T_checker[:3, :3])
            t_target2cam.append(T_checker[:3, 3])

        # Eye-on-base: camera is fixed, robot moves
        # For eye-on-base, we pass R_gripper2base and R_target2cam directly
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        # Construct X: camera-to-base (or base-to-camera depending on convention)
        X = np.eye(4)
        X[:3, :3] = R_cam2gripper
        X[:3, 3] = t_cam2gripper.flatten()

        print("[Calibrator] Calibration result (cam2base):")
        print(f"  R:\n{X[:3, :3]}")
        print(f"  t: {X[:3, 3]} mm")

        return X

    def save(self, filepath: str):
        """Save calibration result."""
        X = self.calibrate()
        np.save(filepath, X)
        print(f"[Calibrator] Saved to {filepath}")

    def load(self, filepath: str) -> np.ndarray:
        """Load calibration result."""
        X = np.load(filepath)
        print(f"[Calibrator] Loaded from {filepath}")
        return X

    @staticmethod
    def _pose_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
        """Convert [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg] to 4x4 matrix."""
        x, y, z, rx, ry, rz = pose_6d

        # Convert to radians
        rx_rad = np.deg2rad(rx)
        ry_rad = np.deg2rad(ry)
        rz_rad = np.deg2rad(rz)

        # Rotation matrix from Euler angles (XYZ order)
        cx, sx = np.cos(rx_rad), np.sin(rx_rad)
        cy, sy = np.cos(ry_rad), np.sin(ry_rad)
        cz, sz = np.cos(rz_rad), np.sin(rz_rad)

        R = np.array([
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy,   sx*cy,            cx*cy           ],
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]  # mm
        return T

    @property
    def num_samples(self):
        return len(self._robot_poses)
