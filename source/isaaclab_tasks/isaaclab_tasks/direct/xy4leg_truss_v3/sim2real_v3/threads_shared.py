from __future__ import annotations

import threading
from typing import Optional
import numpy as np


class SharedData:
    """线程共享的数据结构：仅保留最新观测、最新动作（无历史队列）。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_obs: Optional[np.ndarray] = None  # shape: (obs_dim,)
        self.latest_action: Optional[np.ndarray] = None  # shape: (dof,)

    def set_obs(self, obs: np.ndarray) -> None:
        with self.lock:
            self.latest_obs = obs.copy()

    def get_obs(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_obs is None else self.latest_obs.copy()

    def set_action(self, act: np.ndarray) -> None:
        with self.lock:
            self.latest_action = act.copy()

    def get_action(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_action is None else self.latest_action.copy()

