"""
PiPER Real Robot Environment Wrapper
=====================================
Wraps the PiPER robotic arm (via piper_sdk) to be compatible with the
MENTOR training pipeline (same interface as metaworld_env.py).

Action mapping:
  MetaWorld outputs 4D action in [-1, 1]: [dx, dy, dz, gripper]
  → PiPER end-effector pose increment (MOVEP mode) + gripper control

Key differences from simulation:
  - Uses real camera instead of MuJoCo offscreen rendering
  - Real robot motion has latency; action_repeat controls execution rate
  - Safety limits enforced (workspace bounds, max speed, e-stop)
"""

import os
import time
import threading
import numpy as np
from dm_env import StepType, specs
import dm_env
from gym import spaces
from typing import Any, NamedTuple
from collections import deque

try:
    from piper_sdk import C_PiperInterface_V2, C_PiperForwardKinematics
except ImportError:
    raise ImportError("piper_sdk is required. Install it first: pip install piper_sdk")

# Cross-platform keyboard input
try:
    import msvcrt  # Windows
except ImportError:
    import sys
    import select  # Unix
    import termios
    import tty


class KeyboardRewardListener:
    """Background thread that listens for staged manual reward.

    Staged reward scheme (stacking):
      Key 1 → "到达" (reached):  reward +4.0,  stage=1
      Key 2 → "抓住" (grasped):  reward +1.0,  stage=2
      Key 3 → "对齐" (aligned):  reward +4.0,  stage=3
      Key 4 → "放置" (placed):   reward +1.0,  stage=4, success=True
      Key Q → emergency stop

    Total max reward per episode = 4.0 + 3.0 + 2.0 + 1.0 = 10.0.
    Stages are progressive: pressing a later key implicitly advances the stage.
    """

    # Staged reward constants
    STAGE_REWARDS = {1: 4.0, 2: 1.0, 3: 4.0, 4: 1.0}
    STAGE_NAMES = {0: "none", 1: "reached", 2: "grasped", 3: "aligned", 4: "placed"}

    def __init__(self):
        self._reward_pending = 0.0
        self._success_pending = False
        self._estop = False
        self._current_stage = 0  # tracks highest stage achieved this episode
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start the keyboard listener thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[KeyboardReward] Listener started. "
              "1=reached(+4), 2=grasped(+1), 3=aligned(+4), 4=placed(+1), Q=e-stop")

    def stop(self):
        """Stop the keyboard listener."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def reset_stage(self):
        """Reset stage tracking for a new episode."""
        with self._lock:
            self._current_stage = 0

    @property
    def current_stage(self):
        with self._lock:
            return self._current_stage

    def consume(self):
        """Consume pending reward and success, then reset them.

        Returns:
            (reward, success, estop)
        """
        with self._lock:
            reward = self._reward_pending
            success = self._success_pending
            estop = self._estop
            # Reset after consume (one-shot)
            self._reward_pending = 0.0
            self._success_pending = False
        return reward, success, estop

    def _advance_stage(self, stage):
        """Advance to the given stage if not already reached."""
        with self._lock:
            if stage <= self._current_stage:
                return  # already reached this or a higher stage
            self._current_stage = stage
            reward = self.STAGE_REWARDS[stage]
            self._reward_pending += reward
            success = (stage == 4)
            self._success_pending = self._success_pending or success
            print(f"[KeyboardReward] Stage {stage}: {self.STAGE_NAMES[stage]} | +{reward:.1f}"
                  + (" ★ SUCCESS!" if success else ""))

    def _set_estop(self):
        with self._lock:
            self._estop = True

    def _listen_loop(self):
        """Main keyboard listening loop (runs in background thread)."""
        if os.name == 'nt':
            # Windows
            self._listen_windows()
        else:
            # Unix/Linux
            self._listen_unix()

    def _listen_windows(self):
        while self._running:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'1':
                        self._advance_stage(1)
                    elif ch == b'2':
                        self._advance_stage(2)
                    elif ch == b'3':
                        self._advance_stage(3)
                    elif ch == b'4':
                        self._advance_stage(4)
                    elif ch.lower() == b'q':
                        self._set_estop()
                        print("[KeyboardReward] *** E-STOP triggered ***")
            except Exception:
                pass
            time.sleep(0.01)

    def _listen_unix(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '1':
                        self._advance_stage(1)
                    elif ch == '2':
                        self._advance_stage(2)
                    elif ch == '3':
                        self._advance_stage(3)
                    elif ch == '4':
                        self._advance_stage(4)
                    elif ch.lower() == 'q':
                        self._set_estop()
                        print("[KeyboardReward] *** E-STOP triggered ***")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class PiperRobot:
    """Low-level interface to the PiPER robot arm with safety features."""

    # Unit conversion factors (from piper_sdk demos)
    JOINT_FACTOR = 57295.7795  # rad → SDK int: rad * 1000 * 180 / pi
    POSE_POS_FACTOR = 1000.0   # mm  → SDK int: mm * 1000
    POSE_ROT_FACTOR = 1000.0   # deg → SDK int: deg * 1000
    GRIPPER_FACTOR = 1000000.0 # m   → SDK int: m * 1e6

    def __init__(self, can_port="can0"):
        self.piper = C_PiperInterface_V2(can_port)
        self.fk = C_PiperForwardKinematics()
        self._connected = False
        self._enabled = False
        self._estop = False  # emergency stop flag

    def connect(self):
        """Connect to the robot via CAN bus."""
        self.piper.ConnectPort()
        self._connected = True
        print("[PiperRobot] Connected")

    def enable(self, timeout=10.0):
        """Enable the robot motors."""
        assert self._connected, "Must connect first"
        t0 = time.time()
        while not self.piper.EnablePiper():
            if time.time() - t0 > timeout:
                raise TimeoutError("[PiperRobot] Enable timeout")
            time.sleep(0.01)
        self._enabled = True
        self._estop = False
        # Initialize gripper: set to position mode, close
        self.piper.GripperCtrl(0, 1000, 0x01, 0)
        print("[PiperRobot] Enabled")

    def disable(self):
        """Disable the robot motors."""
        if self._enabled:
            self.piper.DisablePiper()
            self._enabled = False
            print("[PiperRobot] Disabled")

    def emergency_stop(self):
        """Trigger emergency stop: immediately disable motors."""
        print("[PiperRobot] *** EMERGENCY STOP ***")
        self._estop = True
        self.piper.MotionCtrl_2(0x01, 0x01, 0, 0x00)  # zero speed
        self.disable()

    def check_safety(self):
        """Check robot status for errors. Returns True if safe, False if error."""
        if self._estop:
            return False
        try:
            status = self.piper.GetArmStatus()
            arm_status_val = status.arm_status.arm_status
            # 0x00 = Normal, others indicate various errors
            if arm_status_val != 0x00:
                print(f"[PiperRobot] Safety alert: arm_status=0x{arm_status_val:X}")
                if arm_status_val in (0x01, 0x07, 0x09):
                    # Emergency stop / Collision / Joint error → e-stop
                    self.emergency_stop()
                    return False
        except Exception as e:
            print(f"[PiperRobot] Safety check failed: {e}")
            return False
        return True

    def get_end_pose(self):
        """Get current end-effector pose: [X(mm), Y(mm), Z(mm), RX(deg), RY(deg), RZ(deg)]."""
        msgs = self.piper.GetArmEndPoseMsgs()
        pose = msgs.end_pose
        return np.array([
            pose.X_axis * 1e-3,  # 0.001mm → mm
            pose.Y_axis * 1e-3,
            pose.Z_axis * 1e-3,
            pose.RX_axis * 1e-3, # 0.001deg → deg
            pose.RY_axis * 1e-3,
            pose.RZ_axis * 1e-3,
        ])

    def get_joint_angles(self):
        """Get current joint angles in radians."""
        msgs = self.piper.GetArmJointMsgs()
        js = msgs.joint_state
        return np.array([
            js.joint_1 * 1e-3,  # 0.001deg → deg, then convert
            js.joint_2 * 1e-3,
            js.joint_3 * 1e-3,
            js.joint_4 * 1e-3,
            js.joint_5 * 1e-3,
            js.joint_6 * 1e-3,
        ]) * (np.pi / 180.0)  # deg → rad

    def get_gripper_pos(self):
        """Get current gripper position in meters."""
        msgs = self.piper.GetArmGripperMsgs()
        return msgs.gripper_state.grippers_angle * 1e-3  # 0.001mm → mm → m

    def move_to_pose(self, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, speed=100):
        """Move end-effector to target pose (MOVEP mode, cartesian space)."""
        if self._estop:
            print("[PiperRobot] Blocked: emergency stop active")
            return
        X = round(x_mm * self.POSE_POS_FACTOR)
        Y = round(y_mm * self.POSE_POS_FACTOR)
        Z = round(z_mm * self.POSE_POS_FACTOR)
        RX = round(rx_deg * self.POSE_ROT_FACTOR)
        RY = round(ry_deg * self.POSE_ROT_FACTOR)
        RZ = round(rz_deg * self.POSE_ROT_FACTOR)
        self.piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)  # MOVEP mode
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    def move_to_joints(self, joints_rad, speed=100):
        """Move to target joint angles (MOVEJ mode, joint space)."""
        if self._estop:
            print("[PiperRobot] Blocked: emergency stop active")
            return
        j_ints = [round(j * self.JOINT_FACTOR) for j in joints_rad]
        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # MOVEJ mode
        self.piper.JointCtrl(*j_ints)

    def set_gripper(self, pos_m, effort=1000):
        """Set gripper position. pos_m in meters. effort: 0-1000."""
        range_val = round(abs(pos_m * self.GRIPPER_FACTOR))
        self.piper.GripperCtrl(range_val, effort, 0x01, 0)

    def get_arm_status(self):
        """Get robot status for safety checks."""
        return self.piper.GetArmStatus()


class PiperEnv:
    """
    PiPER environment compatible with MENTOR's metaworld_env interface.

    Translates MetaWorld's 4D action space [dx, dy, dz, gripper] to
    PiPER's end-effector pose control + gripper.

    Integrated features:
      - RealSense camera for visual observations
      - Hand-eye calibration support
      - Manual reward via keyboard (SPACE = +10 reward + success)
      - Emergency stop via keyboard (Q key)
    """

    # Default workspace for a table-top task (mm)
    DEFAULT_WORKSPACE = {
        'x_min': -200, 'x_max': 200,
        'y_min': -200, 'y_max': 200,
        'z_min': 50,   'z_max': 400,
    }

    # Default home pose (mm, deg)
    DEFAULT_HOME_POSE = np.array([57.0, 0.0, 215.0, 0.0, 85.0, 0.0])

    def __init__(
        self,
        can_port="can0",
        camera=None,          # camera object or None
        use_realsense=False,  # auto-create RealSense camera
        realsense_serial=None,
        realsense_resolution=(640, 480),
        realsense_fps=30,
        calibration_file=None,  # path to hand-eye calibration .npy
        image_size=(84, 84),
        action_repeat=2,
        action_scale=2.0,     # mm per action unit (MetaWorld action ∈ [-1,1])
        gripper_range=0.08,   # max gripper opening in meters
        workspace=None,
        home_pose=None,
        max_episode_steps=250,
        speed=50,             # robot motion speed (0-100)
        manual_reward=True,   # enable keyboard-based manual reward
    ):
        self._action_repeat = action_repeat
        self._action_scale = action_scale
        self._gripper_range = gripper_range
        self._workspace = workspace or self.DEFAULT_WORKSPACE
        self._home_pose = home_pose if home_pose is not None else self.DEFAULT_HOME_POSE.copy()
        self._max_steps = max_episode_steps
        self._speed = speed
        self._image_size = image_size
        self._step_count = 0

        # Camera setup
        self._camera = camera
        self._owns_camera = False
        if use_realsense and self._camera is None:
            from realsense_camera import RealSenseCamera
            self._camera = RealSenseCamera(
                serial_number=realsense_serial,
                color_resolution=realsense_resolution,
                fps=realsense_fps,
                target_size=self._image_size,
            )
            self._camera.connect()
            self._owns_camera = True
            print("[PiperEnv] RealSense camera connected")

        # Hand-eye calibration
        self._calibration = None
        if calibration_file is not None and os.path.exists(calibration_file):
            self._calibration = np.load(calibration_file)
            print(f"[PiperEnv] Loaded calibration from {calibration_file}")

        # Robot
        self._robot = PiperRobot(can_port)
        self._robot.connect()
        self._robot.enable()

        # Manual reward via keyboard
        self._manual_reward = manual_reward
        self._keyboard = None
        if manual_reward:
            self._keyboard = KeyboardRewardListener()
            self._keyboard.start()

        # Move to home pose
        self._go_home()

    def _go_home(self):
        """Move robot to home pose and reset reward stage."""
        pose = self._home_pose
        self._robot.move_to_pose(
            pose[0], pose[1], pose[2],
            pose[3], pose[4], pose[5],
            speed=30
        )
        time.sleep(1.0)
        self._robot.set_gripper(0.0)
        self._step_count = 0
        # Reset staged reward for new episode
        if self._keyboard is not None:
            self._keyboard.reset_stage()

    def _clip_to_workspace(self, x, y, z):
        """Clip end-effector position to workspace bounds."""
        x = np.clip(x, self._workspace['x_min'], self._workspace['x_max'])
        y = np.clip(y, self._workspace['y_min'], self._workspace['y_max'])
        z = np.clip(z, self._workspace['z_min'], self._workspace['z_max'])
        return x, y, z

    def _capture_image(self):
        """Capture image from camera."""
        if self._camera is not None:
            img = self._camera.capture()
            return img
        else:
            # Placeholder: black image
            return np.zeros(self._image_size + (3,), dtype=np.uint8)

    @property
    def obs_space(self):
        return {
            "image": spaces.Box(0, 255, self._image_size + (3,), dtype=np.uint8),
            "reward": spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": spaces.Box(0, 1, (), dtype=bool),
            "is_last": spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": spaces.Box(0, 1, (), dtype=bool),
            "state": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
            "success": spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        return {"action": spaces.Box(-1, 1, (4,), dtype=np.float32)}

    def step(self, action):
        """Execute one environment step.

        Args:
            action: dict with key 'action', value is np.array of shape (4,)
                    [dx, dy, dz, gripper] all in [-1, 1]

        Reward and success are provided by the human operator:
          Key 1 = reached (+4.0), Key 2 = grasped (+1.0), Key 3 = aligned (+4.0), Key 4 = placed (+1.0).
        """
        action = action["action"]
        assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"

        reward = 0.0
        success = False

        # Check manual reward (keyboard input)
        if self._keyboard is not None:
            kb_reward, kb_success, kb_estop = self._keyboard.consume()
            reward += kb_reward
            success = success or kb_success
            if kb_estop:
                self._robot.emergency_stop()
                print("[PiperEnv] E-stop triggered by keyboard")

        for _ in range(self._action_repeat):
            # Safety check before each motion
            if not self._robot.check_safety():
                print("[PiperEnv] Safety check failed, returning early")
                break

            # Get current pose
            cur_pose = self._robot.get_end_pose()  # [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]

            # Map action to pose delta
            dx = action[0] * self._action_scale  # mm
            dy = action[1] * self._action_scale
            dz = action[2] * self._action_scale

            # Compute target position with workspace clipping
            tgt_x, tgt_y, tgt_z = self._clip_to_workspace(
                cur_pose[0] + dx,
                cur_pose[1] + dy,
                cur_pose[2] + dz,
            )

            # Keep current orientation (no rotation control from MetaWorld)
            tgt_rx = cur_pose[3]
            tgt_ry = cur_pose[4]
            tgt_rz = cur_pose[5]

            # Execute motion
            self._robot.move_to_pose(tgt_x, tgt_y, tgt_z, tgt_rx, tgt_ry, tgt_rz, speed=self._speed)

            # Gripper control: action[3] ∈ [-1, 1] → gripper opening [0, gripper_range]
            gripper_pos = (action[3] + 1.0) / 2.0 * self._gripper_range  # [-1,1] → [0, max]
            self._robot.set_gripper(gripper_pos)

        self._step_count += 1
        is_last = self._step_count >= self._max_steps

        # Check keyboard again after motion (reward may have been given during motion)
        if self._keyboard is not None:
            kb_reward, kb_success, kb_estop = self._keyboard.consume()
            reward += kb_reward
            success = success or kb_success
            if kb_estop:
                self._robot.emergency_stop()

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": is_last,
            "is_terminal": False,
            "image": self._capture_image(),
            "state": cur_pose,
            "success": success,
        }
        return obs

    def reset(self):
        """Reset environment: move to home pose."""
        self._go_home()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._capture_image(),
            "state": self._robot.get_end_pose(),
            "success": False,
        }
        return obs

    def close(self):
        """Clean up: disable robot, stop camera, stop keyboard listener."""
        if self._keyboard is not None:
            self._keyboard.stop()
        if self._owns_camera and self._camera is not None:
            self._camera.disconnect()
        self._robot.disable()


class NormalizeAction:
    """Normalize action space to [-1, 1]. Already normalized for PiperEnv."""

    def __init__(self, env, key="action"):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        return self._env.close()


class TimeLimit:
    """Enforce maximum episode length."""

    def __init__(self, env, duration=250):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

    def close(self):
        return self._env.close()


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class piper_wrapper:
    """Frame-stacking wrapper compatible with MENTOR's metaworld_wrapper interface."""

    def __init__(self, env, nstack=3):
        self._env = env
        self.nstack = nstack
        wos = env.obs_space['image']
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)
        self.observation_space = spaces.Box(
            low=np.transpose(low, (2, 0, 1)),
            high=np.transpose(high, (2, 0, 1)),
            dtype=np.uint8
        )

    def observation_spec(self):
        return specs.BoundedArray(
            self.observation_space.shape,
            np.uint8,
            0,
            255,
            name='observation'
        )

    def action_spec(self):
        act = self._env.act_space['action']
        return specs.BoundedArray(
            act.shape,
            np.float32,
            act.low,
            act.high,
            'action'
        )

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return ExtendedTimeStep(
            observation=np.transpose(self.stackedobs, (2, 0, 1)),
            step_type=StepType.FIRST,
            action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
            reward=0.0,
            discount=1.0,
            success=time_step['success']
        )

    def step(self, action):
        action_dict = {'action': action}
        time_step = self._env.step(action_dict)
        obs = time_step['image']
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs

        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return ExtendedTimeStep(
            observation=np.transpose(self.stackedobs, (2, 0, 1)),
            step_type=step_type,
            action=action,
            reward=time_step['reward'],
            discount=1.0,
            success=time_step['success']
        )

    def close(self):
        self._env.close()


def make(name, frame_stack, action_repeat, seed, **kwargs):
    """
    Factory function compatible with metaworld_env.make() interface.

    Extra kwargs for real robot:
      - can_port: CAN bus port (default: "can0")
      - use_realsense: auto-create RealSense camera (default: False)
      - realsense_serial: RealSense serial number (default: None, auto-detect)
      - realsense_resolution: (width, height) (default: (640, 480))
      - realsense_fps: frame rate (default: 30)
      - calibration_file: path to hand-eye calibration .npy file
      - action_scale: mm per action unit (default: 2.0)
      - gripper_range: max gripper opening in meters (default: 0.08)
      - workspace: dict with x_min/max, y_min/max, z_min/max (mm)
      - home_pose: [X_mm, Y_mm, Z_mm, RX_deg, RY_deg, RZ_deg]
      - speed: robot motion speed 0-100 (default: 50)
      - max_episode_steps: (default: 250)
      - manual_reward: enable keyboard reward (default: True)
    """
    env = PiperEnv(
        action_repeat=action_repeat,
        image_size=(84, 84),
        **kwargs
    )
    env = NormalizeAction(env)
    env = TimeLimit(env, kwargs.get('max_episode_steps', 250))
    env = piper_wrapper(env, frame_stack)
    return env
