"""
Spacemouse Reader for Human Intervention
==========================================
Threaded, non-blocking Spacemouse input reader for real-world
Piper robot training. Continuously reads device state in a
background thread and provides the latest action via get_action().

Usage:
    reader = SpacemouseReader(dead_zone=0.1)
    reader.start()
    action, is_intervening = reader.get_action()
    reader.stop()
"""

import threading
import time
import numpy as np

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None


class SpacemouseReader:
    """Non-blocking Spacemouse reader with intervention detection.

    Runs a background thread that continuously reads the Spacemouse device.
    The training loop can call get_action() to retrieve the latest action
    without blocking.

    Action output: 7D = [dx, dy, dz, droll, dpitch, dyaw, gripper]
      - Position axes (dx, dy, dz) ∈ [-1, 1]: mapped from Spacemouse x, y, z
      - Rotation axes (droll, dpitch, dyaw) ∈ [-1, 1]: mapped from roll, pitch, yaw
      - Gripper ∈ [-1, 1]: button[0] → close (-1), button[1] → open (+1)

    Intervention is detected when any axis exceeds the dead_zone threshold.
    """

    def __init__(self, dead_zone=0.1, action_scale=1.0, device_index=0):
        """
        Args:
            dead_zone: Axes below this threshold are treated as zero.
                       Intervention is detected when any axis exceeds this.
            action_scale: Scale factor applied to raw Spacemouse values.
            device_index: Device index for pyspacemouse (default: 0).
        """
        if pyspacemouse is None:
            raise ImportError(
                "pyspacemouse is required. Install: pip install pyspacemouse"
            )

        self._dead_zone = dead_zone
        self._action_scale = action_scale
        self._device_index = device_index

        # Shared state protected by lock
        self._lock = threading.Lock()
        self._latest_action = np.zeros(7, dtype=np.float32)
        self._is_intervening = False

        # Thread control
        self._running = False
        self._thread = None
        self._device = None

    def start(self):
        """Start the background Spacemouse reading thread."""
        if self._running:
            return

        self._device = pyspacemouse.open(device_index=self._device_index)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[SpacemouseReader] Started. Move the Spacemouse to intervene.")

    def stop(self):
        """Stop the background reading thread and close the device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._device is not None:
            self._device.close()
            self._device = None
        print("[SpacemouseReader] Stopped.")

    def get_action(self):
        """Get the latest Spacemouse action and intervention flag.

        Returns:
            (action, is_intervening):
                action: np.ndarray of shape (7,) or None if not intervening
                        [dx, dy, dz, droll, dpitch, dyaw, gripper]
                is_intervening: bool, True if Spacemouse input exceeds dead_zone
        """
        with self._lock:
            action = self._latest_action.copy()
            is_intervening = self._is_intervening

        if is_intervening:
            return action, True
        else:
            return None, False

    def is_intervening(self):
        """Quick check if the Spacemouse is currently being used."""
        with self._lock:
            return self._is_intervening

    def _read_loop(self):
        """Background thread: continuously read Spacemouse state.

        Handles device disconnection by attempting to reconnect.
        """
        while self._running:
            try:
                state = self._device.read()

                if state is None:
                    # Device returned None, try to reconnect
                    self._try_reconnect()
                    time.sleep(0.1)
                    continue

                # Extract 6-DOF axes
                axes = np.array([
                    state.x,      # dx
                    state.y,      # dy
                    state.z,      # dz
                    state.roll,   # droll
                    state.pitch,  # dpitch
                    state.yaw,    # dyaw
                ], dtype=np.float32)

                # Apply scale
                axes *= self._action_scale

                # Apply dead zone
                is_intervening = bool(np.any(np.abs(axes) > self._dead_zone))
                axes = np.where(np.abs(axes) > self._dead_zone, axes, 0.0)

                # Clip to [-1, 1]
                axes = np.clip(axes, -1.0, 1.0)

                # Gripper from buttons: button[0]=close(-1), button[1]=open(+1)
                gripper = 0.0
                if hasattr(state, 'buttons') and state.buttons is not None:
                    if len(state.buttons) > 0 and state.buttons[0]:
                        gripper = -1.0  # close
                    elif len(state.buttons) > 1 and state.buttons[1]:
                        gripper = 1.0   # open

                    # Buttons also count as intervention
                    if gripper != 0.0:
                        is_intervening = True

                # Build 7D action
                action = np.zeros(7, dtype=np.float32)
                action[:6] = axes
                action[6] = gripper

                # Update shared state
                with self._lock:
                    self._latest_action = action
                    self._is_intervening = is_intervening

                # Clear intervention state if no input
                if not is_intervening:
                    with self._lock:
                        self._is_intervening = False

            except (OSError, IOError) as e:
                # USB disconnect typically raises these
                print(f"[SpacemouseReader] Device error (disconnect?): {e}")
                self._try_reconnect()
                time.sleep(0.5)
            except Exception as e:
                print(f"[SpacemouseReader] Read error: {e}")
                time.sleep(0.05)

            # ~100Hz polling rate
            time.sleep(0.01)

    def _try_reconnect(self):
        """Attempt to reopen the Spacemouse device."""
        print("[SpacemouseReader] Attempting to reconnect...")
        try:
            if self._device is not None:
                try:
                    self._device.close()
                except Exception:
                    pass
            self._device = pyspacemouse.open(device_index=self._device_index)
            if self._device is not None:
                print("[SpacemouseReader] Reconnected successfully.")
            else:
                print("[SpacemouseReader] Reconnect returned None.")
        except Exception as e:
            print(f"[SpacemouseReader] Reconnect failed: {e}")

    @property
    def is_alive(self):
        """Check if the reader thread is still running."""
        return self._thread is not None and self._thread.is_alive()
