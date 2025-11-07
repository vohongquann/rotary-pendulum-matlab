import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from utils import ESP32SerialHelper

class InvertedPendulumSerialEnv(gym.Env):
    def __init__(self, port='COM12', baudrate=921600, mode=None):
        super().__init__()
        # 1) Validate mode
        if mode not in ('swing_up', 'balance'):
            raise ValueError(f"[ERROR] Invalid mode: {mode}. Must be 'swing_up' or 'balance'.")
        self.mode = mode

        # 2) Initialize ESP32 helper
        self.esp = ESP32SerialHelper(port=port, baudrate=baudrate)

        # 3) Define gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32
        )

        # 4) Initial state
        self.last_obs = np.zeros(6, dtype=np.float32)
        self.alpha_hold_time = 0.0

        # 5) Pre-compute angle constants (in radians)
        self._rad170 = np.deg2rad(170.0)   # ~2.967 rad
        self._rad175 = np.deg2rad(175.0)   # ~3.054 rad
        self._rad160 = np.deg2rad(160.0)   # ~2.794 rad
        self._rad90  = np.deg2rad(80.0)    # ~1.571 rad

        # 6) Pre-defined setpoints (float32)
        self._sp_theta = np.float32(0.0)
        self._sp_alpha = np.float32(np.pi)

    def reset(self):
        """
        Send RESET command to ESP32 and wait until it replies with reset_done = 1.
        Then return the initial observation.
        """
        self.esp.send_action("RESET")

        # Wait for {"..., "reset_done":1} response from ESP32
        timeout = 30.0
        t0 = time.time()
        while True:
            data = self.esp.read_json_data()
            if data is not None and data.get("reset_done", 0) == 1:
                break
            if time.time() - t0 > timeout:
                raise RuntimeError("[RESET] ESP32 did not respond in time.")
            # time.sleep(0.001)

        # Get info and observation after reset
        info = self._get_info()
        obs = self._get_obs(info)
        self.last_obs = obs.copy()
        self.alpha_hold_time = 0.0
        return obs, {}

    def send_action(self, action):
        """
        Send action to ESP32.
        If action is a string (e.g., "RESET"), send as-is.
        If action is a scalar or 1-element array, convert to float.
        """
        if isinstance(action, str):
            self.esp.send_action(action)
        else:
            a_scalar = float(action[0]) if hasattr(action, "__len__") else float(action)
            self.esp.send_action(a_scalar)

    def step(self, action):
        """
        1) Send action to ESP32
        2) Read response and construct new observation
        3) Compute reward, terminated, truncated
        """
        # 1) Normalize action to float
        a_scalar = float(action[0]) if hasattr(action, "__len__") else float(action)
        self.esp.send_action(a_scalar)

        # 2) Read info and convert to observation
        info = self._get_info()
        obs  = self._get_obs(info)
        self.last_obs = obs.copy()

        theta, theta_dot, alpha, alpha_dot, sp_theta, sp_alpha = obs

        # 3) Compute reward and termination flags based on mode
        if self.mode == 'swing_up':
            err_alpha = self._wrap_to_pi(alpha - sp_alpha)
            reward = (1.0 / 500.0) * (abs(alpha_dot) - np.log10(err_alpha**2 + 1e-5))
            self.alpha_hold_time += 0.05  # ~50ms per step

            if abs(alpha) > self._rad175:
                time_bonus = self.alpha_hold_time
                reward += 10.0 - 5.0 * np.log10(time_bonus)
                terminated = True
                if terminated == True:
                    reward += 200
                    print(f'Terminated == True')
            else:
                terminated = False

            truncated = (abs(theta) > self._rad90)

        else:  # mode == 'balance'
            if abs(alpha) >= self._rad175:
                self.alpha_hold_time += 0.01  # ~10ms per step
            else:
                self.alpha_hold_time = 0.0

            # Errors
            err_theta = self._wrap_to_pi(theta - sp_theta)
            err_alpha = self._wrap_to_pi(alpha - sp_alpha)

            # Reward components
            r1 = - np.log10(err_theta * err_theta + 1e-5)
            r2 = - 0.1 * np.log10(theta_dot * theta_dot + 1e-5)
            r3 = - np.log10(err_alpha * err_alpha + 1e-5)
            r4 = - 0.5 * np.log10(alpha_dot * alpha_dot + 1e-5)
            reward = r1 + r2 + r3 + r4

            abs_alpha = abs(alpha)
            if self._rad170 <= abs_alpha < self._rad175:
                reward += 20.0
            elif abs_alpha >= self._rad175:
                base_bonus = 50.0
                time_bonus = self.alpha_hold_time * 2.0
                reward += (base_bonus + time_bonus)

            reward = reward / 500.0
            terminated = False
            truncated = (abs(theta) > self._rad90)

        return obs, float(reward), bool(terminated), bool(truncated), {}

    def _wrap_to_pi(self, angle):
        """
        Wrap angle into [-pi, pi] using modulo:
        (angle + pi) % (2*pi) - pi
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def _get_info(self, timeout=5.0):
        """
        Read JSON from ESP32 and parse into float32 array:
        [mode, theta, theta_dot, alpha, alpha_dot, action_prev]
        Raise RuntimeError on timeout.
        """
        start_time = time.time()
        while True:
            data = self.esp.read_json_data()
            if data is not None:
                try:
                    return np.array([
                        data.get("mode", 0),
                        data["theta"],
                        data["theta_dot"],
                        data["alpha"],
                        data["alpha_dot"],
                        data["action_prev"]
                    ], dtype=np.float32)
                except KeyError:
                    pass  # skip if any key is missing

            if time.time() - start_time > timeout:
                raise RuntimeError("[ERROR] ESP32 did not respond in time.")
            time.sleep(0.02)

    def _get_obs(self, info):
        """
        info: numpy float32 array [mode, theta, theta_dot, alpha, alpha_dot, action_prev]
        return: numpy float32 array [theta, theta_dot, alpha, alpha_dot, sp_theta, sp_alpha]
        """
        return np.array([
            self._wrap_to_pi(info[1]),
            info[2],
            self._wrap_to_pi(info[3]),
            info[4],
            self._sp_theta,
            self._sp_alpha
        ], dtype=np.float32)

    def close(self):
        self.esp.close()

