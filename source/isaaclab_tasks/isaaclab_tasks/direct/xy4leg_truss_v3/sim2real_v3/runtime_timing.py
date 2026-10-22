"""Monotonic scheduling and measured timing for the real-policy I/O pipeline."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict


class PeriodicDeadline:
    """Run on an absolute time grid; skip missed ticks instead of replaying them."""

    def __init__(self, hz, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("Frequency must be positive and finite")
        self.period = 1.0 / hz
        self.clock, self.sleep = clock, sleep
        self.next_deadline = clock()
        self.skipped_ticks = 0
        self.last_lateness_s = 0.0

    def wait(self, stop_event=None):
        delay = self.next_deadline - self.clock()
        if stop_event is not None and stop_event.is_set():
            return False
        if delay > 0:
            if stop_event is None:
                self.sleep(delay)
            elif stop_event.wait(delay):
                return False
        now = self.clock()
        self.last_lateness_s = max(now - self.next_deadline, 0.0)
        skipped = int(self.last_lateness_s / self.period)
        self.skipped_ticks += skipped
        self.next_deadline += (skipped + 1) * self.period
        return True


class TimingStats:
    """Thread-safe rates and elapsed milliseconds since the preceding report."""

    def __init__(self):
        self.lock = threading.Lock()
        self.since = time.monotonic()
        self.counts = defaultdict(int)
        self.first_ticks, self.first_counts = {}, {}
        self.samples = defaultdict(list)

    def tick(self, name, count=1):
        if count <= 0:
            return
        now = time.monotonic()
        with self.lock:
            if name not in self.first_ticks:
                self.first_ticks[name], self.first_counts[name] = now, count
            self.counts[name] += count

    def observe(self, name, value):
        if math.isfinite(value):
            with self.lock:
                # Bound storage if reporting is disabled or the consumer stalls.
                if len(self.samples[name]) < 4096:
                    self.samples[name].append(float(value))

    def elapsed_ms(self, name, started):
        self.observe(name, (time.monotonic() - started) * 1000.0)

    def snapshot(self, reset=True):
        now = time.monotonic()
        with self.lock:
            elapsed = max(now - self.since, 1e-9)
            rates = {}
            for name, count in self.counts.items():
                # Include silence after the last tick: a stalled producer must not
                # keep reporting its old rate. Exclude the first sample's interval.
                span = now - self.first_ticks[name]
                rates[name] = (count - self.first_counts[name]) / span if span > 0 else 0.
            result = {"window_s": elapsed, "rates_hz": rates,
                      "counts": dict(self.counts), "samples": {}}
            for name, values in self.samples.items():
                if values:
                    ordered = sorted(values)
                    result["samples"][name] = {
                        "mean": sum(values) / len(values), "max": ordered[-1],
                        "p95": ordered[max(0, math.ceil(.95 * len(values)) - 1)], "n": len(values),
                    }
            if reset:
                self.since = now
                self.counts.clear()
                self.first_ticks.clear()
                self.first_counts.clear()
                self.samples.clear()
            return result


class LatestCamera:
    """One capture owner; policy reads a timestamped latest frame without waiting on the SDK."""

    def __init__(self, camera, img_size, max_age_s=.25, capture_hz=30., stats=None):
        self.camera, self.img_size, self.max_age_s = camera, img_size, max_age_s
        self.capture_hz, self.stats = capture_hz, stats
        self.depth_scale_m = camera.depth_scale_m
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.error = None
        self.sequence = 0
        self.last_read_capture_time = None
        self.last_read_sequence = None
        self.thread = threading.Thread(target=self._capture, name="real-depth-capture", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture(self):
        schedule = PeriodicDeadline(self.capture_hz)
        try:
            while schedule.wait(self.stop_event):
                color, depth = self.camera.read(img_size=(self.img_size, self.img_size))
                received_at = time.monotonic()
                with self.lock:
                    self.sequence += 1
                    # RealSense's read() returns arrays owned by that call.
                    self.latest = (color.copy() if color is not None else None, depth.copy(), received_at, self.sequence)
                if self.stats is not None:
                    self.stats.tick("camera")
        except Exception as exc:
            with self.lock:
                self.error = exc

    def read(self, img_size=None):
        if img_size not in (None, (self.img_size, self.img_size)):
            raise ValueError("Camera cache size must match the estimator input")
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Camera capture failed: {self.error}") from self.error
            if self.latest is None:
                raise RuntimeError("Camera is not ready")
            color, depth, captured_at, sequence = self.latest
            age = time.monotonic() - captured_at
            if age < 0 or age > self.max_age_s:
                raise RuntimeError(f"Camera frame is stale ({age:.3f}s old)")
            self.last_read_capture_time, self.last_read_sequence = captured_at, sequence
            if self.stats is not None:
                self.stats.observe("camera_age_ms", age * 1000.)
            return color.copy() if color is not None else None, depth.copy()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1.)
        self.camera.stop()
