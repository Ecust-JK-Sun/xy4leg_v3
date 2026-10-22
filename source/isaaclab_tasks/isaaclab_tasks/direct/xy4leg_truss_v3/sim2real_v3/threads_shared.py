from __future__ import annotations

import threading
import time
import numpy as np


class SharedData:
    """Latest samples with monotonic receipt times and thread error propagation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.action_changed = threading.Condition(self.lock)
        self.latest_obs = None
        self.obs_sample_times = None
        self.obs_valid = None
        self.latest_action = None
        self.action_time = None
        self.action_sequence = 0
        self.action_observed_at = None
        self.error = None

    def set_obs(self, obs, sample_times=None, valid=None):
        with self.lock:
            self.latest_obs = np.asarray(obs, dtype=np.float32).copy()
            self.obs_sample_times = None if sample_times is None else np.asarray(sample_times).copy()
            self.obs_valid = None if valid is None else np.asarray(valid, dtype=bool).copy()

    def get_obs(self, max_age_s=None, now=None):
        with self.lock:
            if max_age_s is not None:
                if self.latest_obs is None or self.obs_sample_times is None or self.obs_valid is None:
                    raise RuntimeError("Motor feedback is not ready")
                age = (time.monotonic() if now is None else now) - self.obs_sample_times
                bad = ~self.obs_valid | ~np.isfinite(age) | (age < 0) | (age > max_age_s)
                if np.any(bad):
                    raise RuntimeError(f"Missing, faulted or stale motor feedback at hardware indices {np.flatnonzero(bad).tolist()}")
                if self.latest_obs.shape != (48,) or not np.isfinite(self.latest_obs).all():
                    raise RuntimeError("Invalid motor observation; expected 48 finite values")
            return None if self.latest_obs is None else self.latest_obs.copy()

    def set_action(self, act, observed_at=None):
        action = np.asarray(act, dtype=np.float32)
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Hardware action must contain exactly 24 finite values")
        with self.lock:
            self.latest_action = action.copy()
            self.action_time = time.monotonic()
            self.action_observed_at = observed_at
            self.action_sequence += 1
            self.action_changed.notify_all()

    def get_action(self, max_age_s=None, now=None):
        return self.action_snapshot(max_age_s=max_age_s, now=now)[0]

    def action_snapshot(self, max_age_s=None, now=None):
        with self.lock:
            if self.latest_action is not None and max_age_s is not None:
                age = (time.monotonic() if now is None else now) - self.action_time
                if age < 0 or age > max_age_s:
                    raise RuntimeError(f"Policy command expired ({age:.3f}s old)")
            return (None if self.latest_action is None else self.latest_action.copy(),
                    self.action_time, self.action_sequence, self.action_observed_at)

    def wait_for_action(self, previous_sequence, timeout):
        with self.action_changed:
            if self.action_sequence == previous_sequence and self.error is None:
                self.action_changed.wait(timeout)
            return self.action_sequence != previous_sequence

    def feedback_age(self, now=None):
        with self.lock:
            if self.obs_sample_times is None:
                return float("nan")
            return (time.monotonic() if now is None else now) - float(np.min(self.obs_sample_times))

    def set_error(self, source, error):
        with self.lock:
            if self.error is None:
                self.error = f"{source}: {error}"
            self.action_changed.notify_all()

    def raise_if_failed(self):
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError(error)

