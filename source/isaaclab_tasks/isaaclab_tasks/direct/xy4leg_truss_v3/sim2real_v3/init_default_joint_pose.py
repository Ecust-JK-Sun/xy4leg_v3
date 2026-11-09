from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - runtime fallback for minimal deployments
    yaml = None

from motor_control import DaMiaoMotorController
from initialization_models import DEFAULT_NODE_MASS_FILE, make_compensation_model


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_CATALOG = THIS_DIR / "calibration/directional_candidate_catalog_20261030.json"
LEG_PORTS = {
    "LB": "/dev/ttyUSB0",
    "LF": "/dev/ttyUSB1",
    "RB": "/dev/ttyUSB2",
    "RF": "/dev/ttyUSB3",
}
DEFAULT_LEG_ORDER = ("LB", "LF", "RB", "RF")
MOTOR_IDS = (1, 2, 3, 4, 5)

# Copied from XY4LegTrussV3TeacherEnvCfg.robot.init_state.joint_pos.
INIT_JOINT_POS_DEG = {
    "LB": (65.0, 45.0, -20.0, 24.0, 24.0),
    "LF": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "RB": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "RF": (65.0, 45.0, -20.0, 24.0, 24.0),
}

# Same sim-to-real sign convention as play_real.py. Joint5 is inverted.
SIM_TO_REAL_SIGN = (1.0, 1.0, 1.0, 1.0, -1.0)
DEFAULT_KP = (30.0, 25.0, 20.0, 20.0, 2.0)
DEFAULT_KD = (1.0, 1.0, 1.0, 1.0, 0.5)


@dataclass
class LegController:
    leg: str
    ctrl: DaMiaoMotorController


class ParallelInitialization:
    """Synchronize startup only; running serial loops never wait for peers."""

    def __init__(self, legs):
        self.barrier = threading.Barrier(len(legs))
        self.steps = {}
        self.ready = {leg: False for leg in legs}
        self.timed_out = {leg: False for leg in legs}
        self.legs = tuple(legs)
        self.lock = threading.Lock()
        self.ready_at = {}
        self.done = False

    def wait(self):
        self.barrier.wait()

    def check(self):
        if self.barrier.broken:
            raise threading.BrokenBarrierError("Parallel initialization cancelled by another leg or interruption")

    def ramp_steps(self, leg, steps):
        self.steps[leg] = steps
        self.wait()  # All motors on every selected leg must be enabled first.
        return max(self.steps.values())

    def verify(self, leg, ready, timed_out, feedback_timeout_s=.5):
        now = time.monotonic()
        with self.lock:
            self.ready[leg], self.timed_out[leg] = ready, timed_out
            self.ready_at[leg] = now
            # A paused worker's previous REACHED state must not qualify the group.
            all_ready = all(self.ready.values()) and all(
                motor_leg in self.ready_at and 0 <= now-self.ready_at[motor_leg] <= feedback_timeout_s
                for motor_leg in self.legs)
            self.done = self.done or all_ready or any(self.timed_out.values())
            return self.done

    def abort(self):
        self.barrier.abort()


@dataclass
class IntegralTorqueCompensator:
    """Bounded, rate-limited feedforward learned only from new endpoint feedback."""

    ki: tuple[float, ...] = (8., 8., 8., 8., 1.)
    torque_limit: tuple[float, ...] = (4., 1.5, 1.5, 1.5, .2)
    torque_rate: tuple[float, ...] = (.3, .15, .15, .15, .02)
    deadband_rad: float = math.radians(.25)
    velocity_gate_rad_s: float = math.radians(5.)
    max_dt_s: float = .1
    torque: list[float] = field(default_factory=lambda: [0.] * 5)

    def update(self, updates: dict[int, dict], targets_rad: list[float], dt_s: float) -> None:
        # Cap the integration interval: a scheduling pause must not cause a torque jump.
        dt = min(max(dt_s, 0.), self.max_dt_s)
        for index, motor_id in enumerate(MOTOR_IDS):
            if motor_id not in updates:
                continue
            feedback = updates[motor_id]
            error = targets_rad[index] - feedback["position"]
            if abs(error) <= self.deadband_rad:
                continue  # Retain learned holding torque; do not integrate measurement noise.
            if (abs(feedback["velocity"]) > self.velocity_gate_rad_s
                    and error * self.torque[index] >= 0):
                continue  # Freeze growth during motion, but allow unwinding after an overshoot.
            max_change = self.torque_rate[index] * dt
            delta = max(-max_change, min(max_change, self.ki[index] * error * dt))
            limit = self.torque_limit[index]
            # Store the clipped output itself, so there is no hidden integral windup.
            self.torque[index] = max(-limit, min(limit, self.torque[index] + delta))


def _read_motor_cfg(path: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not path or yaml is None:
        return DEFAULT_KP, DEFAULT_KD
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    kp = tuple(float(x) for x in cfg.get("kp_truss", DEFAULT_KP)[:5])
    kd = tuple(float(x) for x in cfg.get("kd", DEFAULT_KD)[:5])
    if len(kp) != 5 or len(kd) != 5:
        raise ValueError(f"Expected at least five kp_truss/kd values in {path}")
    return kp, kd


def _target_rad_for_leg(leg: str, joint5_sign_flip: bool) -> list[float]:
    signs = list(SIM_TO_REAL_SIGN)
    if not joint5_sign_flip:
        signs[4] = 1.0
    return [math.radians(deg) * sign for deg, sign in zip(INIT_JOINT_POS_DEG[leg], signs)]


def _format_deg(values_rad: list[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(x):7.2f}" for x in values_rad) + "] deg"


def _checked_feedback(feedback: dict | None, motor_id: int | None = None) -> dict | None:
    if not feedback or feedback.get("motor_id") not in MOTOR_IDS:
        return None
    if motor_id is not None and feedback["motor_id"] != motor_id:
        return None
    if feedback.get("error_code") != 1:
        raise RuntimeError(f"Motor {feedback['motor_id']} is not enabled: status={feedback.get('error_code')}")
    if any(not math.isfinite(float(feedback.get(key, math.nan))) for key in ("position", "velocity", "torque")):
        raise RuntimeError(f"Motor {feedback['motor_id']} has invalid position/velocity/torque feedback")
    return feedback


def _read_leg_feedback(ctrl: DaMiaoMotorController, timing=None) -> dict[int, dict]:
    frames, pending, backlog, overflow = ctrl.receive_available()
    if timing is not None:
        timing.received(frames, pending, backlog, len(getattr(ctrl, "_rx_buffer", b"")), overflow)
        timing.observe("rx_backlog_bytes", backlog)
        timing.observe("rx_pending_bytes", pending)
    if overflow:
        raise RuntimeError("Motor feedback queue overflow; cannot verify current joint angles")
    latest = {}
    for frame in frames:
        feedback = _checked_feedback(ctrl.damiao_feedback(frame))
        if feedback is not None:
            latest[feedback["motor_id"]] = feedback
            if timing is not None:
                timing.feedback(feedback["motor_id"])
    return latest


def _feedback_snapshot(received_at, now, started, timeout):
    """Host valid-frame receipt ages; cached values never advance these stamps."""
    return dict(feedback_age_ms={motor: (now-received_at[motor])*1000 if motor in received_at else None
                                 for motor in MOTOR_IDS},
                missing_feedback_motors=[motor for motor in MOTOR_IDS if motor not in received_at],
                stale_feedback_motors=[motor for motor in MOTOR_IDS
                                       if now-received_at.get(motor, started) > timeout],
                feedback_timeout_s=timeout,
                feedback_age_source="host valid-frame processing time; not device sample timestamps")


class InitializationTiming:
    """Measured host cycle/packet rates, with processing time separated from sleep.

    A TX cycle is a completed group of five writes. Feedback counts actual valid
    frames before per-ID deduplication; neither metric is an execution ACK.
    """

    def __init__(self, leg, hz, jsonl_path=None, interval_s=2., source="hardware serial", live_reports=False):
        self.leg, self.hz, self.jsonl_path = leg, hz, jsonl_path
        self.interval_s, self.source = interval_s, source
        self.phases = {}
        self.phase = None
        self.last_report = time.monotonic()
        self.active = None
        self.slept = 0.
        self.error = None
        self.live_reports = live_reports
        self.io_started = time.monotonic()
        self.io_events = deque(maxlen=64)
        self.io_tx_cycles = self.io_rx_polls = self.io_rx_frames = self.io_valid_frames = 0
        self.io_last_valid = {}

    def received(self, frames, pending, backlog, partial, overflow):
        self.io_rx_polls += 1
        self.io_rx_frames += len(frames)
        def raw(frame):
            return bytes(frame).hex() if isinstance(frame, (bytes, bytearray)) else None
        self.io_events.append(dict(kind="rx", elapsed_s=time.monotonic()-self.io_started,
                                   frames=len(frames), pending_bytes=pending, backlog_bytes=backlog,
                                   partial_buffer_bytes=partial, overflow=bool(overflow),
                                   recent_frame_hex=[raw(frame) for frame in frames[-5:]]))

    def io_snapshot(self):
        now = time.monotonic()
        return dict(tx_cycles=self.io_tx_cycles, rx_polls=self.io_rx_polls,
                    rx_complete_frames=self.io_rx_frames, valid_frames=self.io_valid_frames,
                    last_valid_age_ms={motor: (now-self.io_last_valid[motor])*1000
                                       if motor in self.io_last_valid else None for motor in MOTOR_IDS},
                    recent_events=list(self.io_events),
                    interpretation="Bounded host I/O history; TX writes are not device ACKs; raw frames include ignored IDs")

    def begin(self, phase):
        now = time.monotonic()
        self.phase = phase
        data = self.phases.setdefault(phase, dict(start=now, end=now, cycles=0, tx_cycles=0,
                                                rx={motor: 0 for motor in MOTOR_IDS}, samples={},
                                                previous_start=None, previous_tx=None, previous_rx={},
                                                over_budget_cycles=0, skipped_deadlines=0, schedule_mode="relative"))
        if data["previous_start"] is not None:
            self.observe("cycle_start_interval_ms", (now-data["previous_start"])*1000)
        data["previous_start"] = now
        self.active, self.slept = now, 0.

    def observe(self, name, value):
        if self.phase is not None and math.isfinite(value):
            values = self.phases[self.phase]["samples"].setdefault(name, [])
            if len(values) < 4096:
                values.append(value)

    def feedback(self, motor):
        self.io_valid_frames += 1
        self.io_last_valid[motor] = time.monotonic()
        if self.phase is not None:
            data = self.phases[self.phase]
            now = time.monotonic()
            if motor in data["previous_rx"]:
                self.observe(f"rx_motor{motor}_interval_ms", (now-data["previous_rx"][motor])*1000)
            data["previous_rx"][motor] = now
            data["rx"][motor] += 1

    def sent(self, started, ctrl):
        self.io_tx_cycles += 1
        self.io_events.append(dict(kind="tx", elapsed_s=time.monotonic()-self.io_started,
                                   duration_ms=(time.monotonic()-started)*1000))
        if self.phase is not None:
            data = self.phases[self.phase]
            data["tx_cycles"] += 1
            if data["previous_tx"] is not None:
                self.observe("tx_start_interval_ms", (started-data["previous_tx"])*1000)
            data["previous_tx"] = started
            self.observe("tx_ms", (time.monotonic()-started)*1000)
            self.observe("tx_pending_bytes", float(getattr(getattr(ctrl, "ser", None), "out_waiting", 0)))

    def wait_remaining(self, period, started, schedule=None):
        before = time.monotonic()
        if schedule is None:
            time.sleep(max(0., period-(before-started)))
        else:
            previous_skipped = schedule.skipped_ticks
            schedule.wait()
            self.phases[self.phase]["schedule_mode"] = "absolute"
            self.phases[self.phase]["skipped_deadlines"] += schedule.skipped_ticks-previous_skipped
            self.observe("deadline_lateness_ms", schedule.last_lateness_s*1000)
        self.slept += time.monotonic()-before

    def end(self):
        if self.active is None:
            return
        now = time.monotonic()
        work = max(0., now-self.active-self.slept)
        self.observe("work_ms", work*1000)
        self.observe("cycle_ms", (now-self.active)*1000)
        data = self.phases[self.phase]
        data["cycles"] += 1
        data["over_budget_cycles"] += int(work > 1./self.hz)
        data["end"] = now
        self.active = None
        if self.live_reports and now-self.last_report >= self.interval_s:
            self.report()

    def snapshot(self):
        phases = {}
        for name, data in self.phases.items():
            # Include time between cycles (such as blocking diagnostic output).
            end = time.monotonic() if name == self.phase else data["end"]
            elapsed = max(end-data["start"], 1e-9)
            samples = {}
            for key, values in data["samples"].items():
                if values:
                    ordered = sorted(values)
                    samples[key] = dict(n=len(values), mean=sum(values)/len(values), max=ordered[-1],
                                        p95=ordered[max(0, math.ceil(.95*len(values))-1)])
            phases[name] = dict(duration_s=elapsed, cycles=data["cycles"], tx_cycles=data["tx_cycles"],
                                cycle_hz=data["cycles"]/elapsed, tx_cycles_hz=data["tx_cycles"]/elapsed,
                                rx_frames_by_motor=data["rx"].copy(),
                                rx_min_motor_hz=min(data["rx"].values())/elapsed,
                                rx_frames_hz_by_motor={motor: count/elapsed for motor, count in data["rx"].items()},
                                over_budget_cycles=data["over_budget_cycles"], samples=samples)
            phases[name].update(schedule_mode=data["schedule_mode"], skipped_deadlines=data["skipped_deadlines"])
        return dict(leg=self.leg, target_hz=self.hz, source=self.source, phases=phases, error=self.error,
                    io_diagnostics=self.io_snapshot(),
                    interpretation="Measured host writes/receipts, not device execution/independent sampling. "
                                   "Gravity mode uses absolute deadlines; over_budget counts processing longer than the period.")

    def report(self, final=False):
        result = self.snapshot()
        for phase, data in result["phases"].items():
            samples = data["samples"]
            def p95(key):
                return samples.get(key, {}).get("p95", float("nan"))
            print(f"[TIMING] {self.leg}/{phase} cycle={data['cycle_hz']:.2f}/{self.hz:g}Hz "
                  f"tx_group={data['tx_cycles_hz']:.2f}Hz rx_slowest={data['rx_min_motor_hz']:.2f}Hz "
                  f"work_p95={p95('work_ms'):.3f}ms gravity_p95={p95('gravity_ms'):.3f}ms "
                  f"over_budget={data['over_budget_cycles']} missed={data['skipped_deadlines']} final={final}")
        if self.jsonl_path is not None:
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(result, final=final), ensure_ascii=False, allow_nan=False)+"\n")
        self.last_report = time.monotonic()
        return result


class InitializationGravity:
    """Fresh measured pose -> bounded/slewed model feedforward.

    Before all five motors return post-enable command feedback, send zero FF.
    One instance spans ramp and hold so neither its slew state nor receipt ages
    are reset on entry to final verification. J6 is never enabled or commanded.
    """

    def __init__(self, leg, controller, feedback_timeout_s, gravity_b, finger_angle, timing=None):
        self.leg, self.controller = leg, controller
        self.feedback_timeout_s = feedback_timeout_s
        self.gravity_b, self.finger_angle = gravity_b, finger_angle
        self.started = time.monotonic()
        self.latest, self.received_at = {}, {}
        self.last_sent = [0.] * 5
        self.last_report = self.started
        self.timing = timing

    def observe(self, updates, now):
        self.latest.update(updates)
        self.received_at.update({motor: now for motor in updates})

    def prepare(self, ctrl):
        updates = _read_leg_feedback(ctrl, self.timing)
        now = time.monotonic()
        self.observe(updates, now)
        return self.compute(now)

    def compute(self, now):
        """Compute from the shared cache without consuming unlogged serial replies."""
        stale = [motor for motor in MOTOR_IDS
                 if now - self.received_at.get(motor, self.started) > self.feedback_timeout_s]
        if stale:
            ages = _feedback_snapshot(self.received_at, now, self.started, self.feedback_timeout_s)
            raise RuntimeError(f"{self.leg}: missing/stale feedback from motors {stale}; "
                               f"model feedforward stopped; ages_ms={ages['feedback_age_ms']}")
        if all(motor in self.latest for motor in MOTOR_IDS):
            positions = [self.latest[motor]["position"] for motor in MOTOR_IDS]
            if self.timing is not None:
                self.timing.observe("oldest_feedback_age_ms", max(now-stamp for stamp in self.received_at.values())*1000)
            model_started = time.monotonic()
            self.last_sent = self.controller.update_leg(
                self.leg, positions, self.gravity_b, self.finger_angle, now=now).tolist()
            if self.timing is not None:
                self.timing.observe("gravity_ms", (time.monotonic()-model_started)*1000)
        if self.timing is not None and self.timing.live_reports and now - self.last_report >= 2.:
            index = DEFAULT_LEG_ORDER.index(self.leg)
            print(f"[FEEDFORWARD] {self.leg}: raw_ff_nm={self.controller.raw[index::4].round(4).tolist()} "
                  f"sent_ff_nm={[round(value, 4) for value in self.last_sent]}")
            self.last_report = now
        return self.last_sent.copy()

    def diagnostics(self):
        index = DEFAULT_LEG_ORDER.index(self.leg)
        result = dict(gravity_ff_nm=self.last_sent.copy(),
                    gravity_raw_ff_nm=self.controller.raw[index::4].tolist(),
                    gravity_ramp_complete=self.ramp_complete(),
                    gravity_limit_motors=[motor for motor, limited in
                                          zip(MOTOR_IDS, self.controller.saturated[index::4]) if limited])
        if hasattr(self.controller.model, "leg_diagnostics"):
            result.update(self.controller.model.leg_diagnostics(self.leg))
        result.update(_feedback_snapshot(self.received_at, time.monotonic(), self.started, self.feedback_timeout_s))
        return result

    def ramp_complete(self):
        if not all(motor in self.latest for motor in MOTOR_IDS):
            return False
        index = DEFAULT_LEG_ORDER.index(self.leg)
        bounded = [max(-limit, min(limit, raw)) for raw, limit
                   in zip(self.controller.raw[index::4], self.controller.model.torque_limits)]
        return all(abs(sent-target) <= 1e-6 for sent, target in zip(self.last_sent, bounded))


def _enable_and_read_current(
    ctrl: DaMiaoMotorController,
    motor_id: int,
    *, attempts: int = 1, feedback_timeout_s: float = .2,
    diagnostics: dict | None = None, cancel_check=None,
) -> float | None:
    """Bounded enable retries, without sending position/velocity/torque targets.

    The driver's enable call already waits for its reply. The configurable
    timeout below is an additional receive window after that call returns.
    Fault/disabled/nonfinite matching feedback raises immediately, never retries.
    """
    if attempts < 1 or not math.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0:
        raise ValueError("Enable attempts and feedback timeout must be positive")
    data = diagnostics if diagnostics is not None else {}
    data.update(motor_id=motor_id, port=str(getattr(ctrl, "port", "unknown")),
                attempts=[], received_position=False)
    for attempt in range(1, attempts+1):
        if cancel_check is not None:
            cancel_check()
        info = dict(attempt=attempt, observed_motor_ids=[], post_enable_reply_count=0,
                    post_enable_invalid_reply_count=0)
        data["attempts"].append(info)
        def check(reply):
            if reply and "motor_id" in reply and reply["motor_id"] not in info["observed_motor_ids"]:
                info["observed_motor_ids"].append(reply["motor_id"])
            return _checked_feedback(reply, motor_id)

        enable_reply = ctrl.enable_motor(motor_id)
        driver_diagnostics = getattr(ctrl, "last_enable_diagnostics", None)
        if driver_diagnostics is not None:
            info["driver_enable_receive"] = driver_diagnostics
            info["observed_motor_ids"].extend(driver_diagnostics.get("observed_motor_ids", []))
        feedback = check(enable_reply)
        if feedback is not None:
            data["received_position"] = True
            return float(feedback["position"])
        deadline = time.monotonic() + feedback_timeout_s
        while time.monotonic() < deadline:
            if cancel_check is not None:
                cancel_check()
            response = ctrl.receive(8)
            parsed = ctrl.damiao_feedback(response) if response else None
            if hasattr(ctrl, "_rx_buffer"):
                info["post_enable_buffered_partial_bytes"] = len(ctrl._rx_buffer)
            if response:
                info["post_enable_reply_count"] += 1
                if isinstance(response, (bytes, bytearray)):
                    info["last_post_enable_reply_hex"] = bytes(response).hex()
                if parsed is None:
                    info["post_enable_invalid_reply_count"] += 1
            feedback = check(parsed)
            if feedback is not None:
                data["received_position"] = True
                return float(feedback["position"])
            time.sleep(.001)
        if attempt < attempts:
            print(f"[WARN] {data['port']} motor {motor_id}: no matching enable feedback "
                  f"on attempt {attempt}/{attempts}; retrying enable; "
                  f"observed_ids={info['observed_motor_ids']}")
    return None


def _send_position_cycle(item, positions, kp, kd, inter_command_sleep, torque_ff=None, timing=None):
    started = time.monotonic()
    torque_ff = [0.] * 5 if torque_ff is None else torque_ff
    for motor_id, pos, kp_i, kd_i, torque in zip(MOTOR_IDS, positions, kp, kd, torque_ff):
        item.ctrl.send(item.ctrl.damiao_mit_command(
            pos=float(pos), vel=0.0, kp=float(kp_i), kd=float(kd_i), tor=float(torque), motor_id=motor_id,
        ))
        time.sleep(inter_command_sleep)
    if timing is not None:
        timing.sent(started, item.ctrl)


def _cycle_schedule(hz, gravity_compensator):
    if gravity_compensator is None:
        return None
    from runtime_timing import PeriodicDeadline

    schedule = PeriodicDeadline(hz, clock=time.monotonic, sleep=time.sleep)
    # Send the first group immediately, then wait for the end of its first period.
    schedule.next_deadline += schedule.period
    return schedule


def _wait_cycle(period, started, schedule, timing):
    if timing is not None:
        timing.wait_remaining(period, started, schedule)
    elif schedule is not None:
        schedule.wait()
    else:
        time.sleep(max(0., period-(time.monotonic()-started)))


def _send_leg_position_targets(
    item: LegController,
    targets_rad: list[float],
    current_rad: list[float],
    kp: tuple[float, ...],
    kd: tuple[float, ...],
    max_step_rad: float,
    hz: float,
    inter_command_sleep: float,
    gravity_compensator: InitializationGravity | None = None,
    timing: InitializationTiming | None = None,
    parallel: ParallelInitialization | None = None,
) -> None:
    max_delta = max(abs(t - c) for t, c in zip(targets_rad, current_rad))
    steps = max(1, math.ceil(max_delta / max(max_step_rad, 1.0e-6)))
    if parallel is not None:
        steps = parallel.ramp_steps(item.leg, steps)
    period = 1.0 / max(hz, 1.0e-6)
    if parallel is not None:
        # Spread USB writes across one period while retaining each leg's Hz.
        time.sleep(parallel.legs.index(item.leg)*period/len(parallel.legs))
    schedule = _cycle_schedule(hz, gravity_compensator)
    print(
        f"[INFO] {item.leg}: current={_format_deg(current_rad)} "
        f"target={_format_deg(targets_rad)} steps={steps}"
    )

    for step in range(1, steps + 1):
        if parallel is not None:
            parallel.check()
        if timing is not None:
            timing.begin("ramp")
        loop_start = time.monotonic()
        if gravity_compensator is None:
            _read_leg_feedback(item.ctrl, timing)  # Drain ramp replies without waiting for an absent motor.
            torque_ff = None
        else:
            torque_ff = gravity_compensator.prepare(item.ctrl)
        alpha = step / steps
        command_rad = [
            current + alpha * (target - current)
            for current, target in zip(current_rad, targets_rad)
        ]
        _send_position_cycle(item, command_rad, kp, kd, inter_command_sleep, torque_ff, timing)
        _wait_cycle(period, loop_start, schedule, timing)
        if timing is not None:
            timing.end()


def _hold_and_verify_leg(
    item, targets_rad, kp, kd, hz, inter_command_sleep,
    settle_timeout_s, settle_time_s, tolerance_rad, velocity_tolerance_rad_s, feedback_timeout_s,
    integral_compensator: IntegralTorqueCompensator | None = None,
    result_out: dict | None = None,
    gravity_compensator: InitializationGravity | None = None,
    timing: InitializationTiming | None = None,
    parallel: ParallelInitialization | None = None,
    report: bool = True,
) -> bool:
    if integral_compensator is not None and gravity_compensator is not None:
        raise ValueError("Choose integral or gravity compensation; do not add both holding torques")
    # Discard ramp responses; endpoint verification starts with subsequent target commands.
    if gravity_compensator is None:
        _read_leg_feedback(item.ctrl)
    started = time.monotonic()
    stable_since = None
    if gravity_compensator is None:
        latest, received_at = {}, {}
        feedback_started = started
    else:
        # prepare() and the post-send poll both consume serial frames. Model and
        # verification must share the same cache AND per-motor receipt stamps.
        # Aliasing preserves real ages across the ramp/hold boundary; do not
        # copy cached data and give it a new receipt timestamp here.
        latest, received_at = gravity_compensator.latest, gravity_compensator.received_at
        feedback_started = gravity_compensator.started
    endpoint_commanded_at = None
    last_update_at = started
    last_sent_integral = [0.] * 5
    schedule = _cycle_schedule(hz, gravity_compensator)
    reached = False
    while parallel is not None or time.monotonic() - started < settle_timeout_s:
        if parallel is not None:
            parallel.check()
        if timing is not None:
            timing.begin("hold")
        loop_start = time.monotonic()
        if integral_compensator is not None:
            # Do not send learned torque after a long pause with stale feedback.
            stale = [motor_id for motor_id in MOTOR_IDS
                     if loop_start - received_at.get(motor_id, started) > feedback_timeout_s]
            if stale:
                raise RuntimeError(f"{item.leg}: missing/stale feedback from motors {stale}; pose not verified")
            last_sent_integral = integral_compensator.torque.copy()
        torque_ff = (last_sent_integral if gravity_compensator is None
                     else gravity_compensator.prepare(item.ctrl))
        _send_position_cycle(item, targets_rad, kp, kd, inter_command_sleep, torque_ff, timing)
        if endpoint_commanded_at is None:
            endpoint_commanded_at = time.monotonic()
        _wait_cycle(1./hz, loop_start, schedule, timing)
        updates = _read_leg_feedback(item.ctrl, timing)
        now = time.monotonic()
        if gravity_compensator is None:
            latest.update(updates)
            received_at.update({motor_id: now for motor_id in updates})
        else:
            gravity_compensator.observe(updates, now)
        snapshot = _feedback_snapshot(received_at, now, feedback_started, feedback_timeout_s)
        if result_out is not None:
            _record_leg_result(result_out, targets_rad, latest, False, now-started,
                               last_sent_integral, gravity_compensator)
            result_out.update(snapshot)
        stale = snapshot["stale_feedback_motors"]
        if stale:
            raise RuntimeError(f"{item.leg}: missing/stale feedback from motors {stale}; "
                               f"pose not verified; ages_ms={snapshot['feedback_age_ms']}")
        in_tolerance = all(
            motor_id in latest
            and received_at[motor_id] > endpoint_commanded_at
            and abs(latest[motor_id]["position"] - target) <= tolerance_rad
            and abs(latest[motor_id]["velocity"]) <= velocity_tolerance_rad_s
            for motor_id, target in zip(MOTOR_IDS, targets_rad)
        )
        if gravity_compensator is not None:
            # Compare the pose only after the added torque has finished its bounded introduction.
            in_tolerance = in_tolerance and gravity_compensator.ramp_complete()
        if not in_tolerance:
            stable_since = None
            reached = False
        elif stable_since is None:
            stable_since = now
        elif (now - stable_since >= settle_time_s
              and all(received_at[motor_id] > stable_since for motor_id in MOTOR_IDS)):
            reached = True
        if not reached and integral_compensator is not None and all(motor_id in latest for motor_id in MOTOR_IDS):
            integral_compensator.update(updates, targets_rad, now - last_update_at)
        last_update_at = now
        if timing is not None:
            timing.end()
        if parallel is not None:
            if parallel.verify(item.leg, reached, now-started >= settle_timeout_s, feedback_timeout_s):
                break
        elif reached:
            break

    final = [latest.get(motor_id, {}).get("position", math.nan) for motor_id in MOTOR_IDS]
    error = [target - actual for target, actual in zip(targets_rad, final)]
    velocity = [latest.get(motor_id, {}).get("velocity", math.nan) for motor_id in MOTOR_IDS]
    torque = [latest.get(motor_id, {}).get("torque", math.nan) for motor_id in MOTOR_IDS]
    status = "REACHED" if reached else "NOT_REACHED"
    if report:
        print(f"[RESULT] {item.leg}: {status} after {time.monotonic() - started:.2f}s "
              f"final={_format_deg(final)} error(target-actual)={_format_deg(error)}")
        print(f"[RESULT] {item.leg}: velocity={_format_deg(velocity)}/s torque="
              f"[{', '.join(f'{value:.3f}' for value in torque)}] N*m")
    if integral_compensator is not None:
        saturated = [motor_id for index, motor_id in enumerate(MOTOR_IDS)
                     if integral_compensator.torque_limit[index] > 0
                     and abs(last_sent_integral[index]) >= integral_compensator.torque_limit[index] - 1.e-9]
        if result_out is not None:
            result_out["compensation_limit_motors"] = saturated
        if report:
            print(f"[RESULT] {item.leg}: integral_ff="
                  f"[{', '.join(f'{value:.3f}' for value in last_sent_integral)}] N*m "
                  f"compensation_limit_motors={saturated}")
    if result_out is not None:
        _record_leg_result(result_out, targets_rad, latest, reached, time.monotonic()-started,
                           last_sent_integral, gravity_compensator)
        result_out["completion_checks"] = dict(
            position_tolerance_deg=math.degrees(tolerance_rad),
            velocity_tolerance_deg_s=math.degrees(velocity_tolerance_rad_s),
            position_outside_tolerance_motors=[motor for motor, value in zip(MOTOR_IDS, error)
                                               if not math.isfinite(value) or abs(value) > tolerance_rad],
            velocity_outside_tolerance_motors=[motor for motor, value in zip(MOTOR_IDS, velocity)
                                               if not math.isfinite(value) or abs(value) > velocity_tolerance_rad_s],
            stable_duration_s=max(0., time.monotonic()-stable_since) if stable_since is not None else 0.,
            required_stable_duration_s=settle_time_s)
    if gravity_compensator is not None:
        diagnostics = gravity_compensator.diagnostics()
        if report:
            print(f"[RESULT] {item.leg}: gravity_raw_ff={diagnostics['gravity_raw_ff_nm']} N*m "
                  f"gravity_sent_ff={diagnostics['gravity_ff_nm']} N*m "
                  f"gravity_limit_motors={diagnostics['gravity_limit_motors']} "
                  f"gravity_ramp_complete={diagnostics['gravity_ramp_complete']}")
        if result_out is not None:
            result_out.update(diagnostics)
    return reached


def _record_leg_result(result, targets, latest, reached, duration, integral_ff, gravity):
    def values(key, degrees=False):
        data = [latest.get(motor, {}).get(key, math.nan) for motor in MOTOR_IDS]
        return [(math.degrees(v) if degrees else v) if math.isfinite(v) else None for v in data]

    positions = values("position", True)
    result.update(reached=reached, duration_s=duration, position_deg=positions,
                  error_deg=[math.degrees(t)-p if p is not None else None for t, p in zip(targets, positions)],
                  velocity_deg_s=values("velocity", True), torque_nm=values("torque"),
                  integral_ff_nm=integral_ff.copy())
    if gravity is not None:
        result.update(gravity.diagnostics())
        result.update(raw_ff_nm=result["gravity_raw_ff_nm"], sent_ff_nm=result["gravity_ff_nm"])
    else:
        result.update(raw_ff_nm=integral_ff.copy(), sent_ff_nm=integral_ff.copy())


def _print_initialization_summary(legs, results):
    print("\n[SUMMARY] Parallel initialization; joints J1-J5, error = target - actual")
    def vector(result, key):
        return "[" + ", ".join("N/A" if v is None else f"{v:8.3f}"
                                for v in result.get(key, [None]*5)) + "]"

    for leg in legs:
        result = results[leg]
        status = "ERROR" if result.get("error") else "REACHED" if result.get("reached") else "NOT_REACHED"
        print(f"[RESULT] {leg}: {status} after {result.get('duration_s', 0.):.2f}s "
              f"compensation={result.get('compensation', 'unknown')}")
        for label, key, unit in (("initial", "initial_deg", "deg"), ("target", "target_deg", "deg"),
                                 ("final", "position_deg", "deg"),
                                 ("error(target-actual)", "error_deg", "deg"),
                                 ("velocity", "velocity_deg_s", "deg/s"), ("torque", "torque_nm", "N*m"),
                                 ("integral_ff", "integral_ff_nm", "N*m")):
            print(f"  {label}={vector(result, key)} {unit}")
        if "gravity_ff_nm" in result:
            prefix = "table" if result.get("compensation") == "table" else "gravity"
            print(f"  {prefix}_raw_ff={vector(result, 'gravity_raw_ff_nm')} N*m "
                  f"{prefix}_sent_ff={vector(result, 'gravity_ff_nm')} N*m "
                  f"{prefix}_limit_motors={result['gravity_limit_motors']} "
                  f"{prefix}_ramp_complete={result['gravity_ramp_complete']}")
        if "table_direction" in result:
            print(f"  table_direction={result['table_direction']} "
                  f"table_clamped_motors={result['table_clamped_motors']}")
        if "j1_residual_nm" in result:
            print(f"  j1_residual={result['j1_residual_nm']:.4f} N*m "
                  f"direction={result['j1_residual_direction']} "
                  f"limited={result['j1_residual_limited']} "
                  f"angle_clamped={result['j1_residual_angle_clamped']}")
        if "compensation_limit_motors" in result:
            print(f"  compensation_limit_motors={result['compensation_limit_motors']}")
        if result.get("error"):
            print(f"  error={result['error']}")
        if "completion_checks" in result:
            checks = result["completion_checks"]
            print(f"  acceptance: position<={checks['position_tolerance_deg']:g} deg "
                  f"outside={checks['position_outside_tolerance_motors']}; "
                  f"velocity<={checks['velocity_tolerance_deg_s']:g} deg/s "
                  f"outside={checks['velocity_outside_tolerance_motors']}; "
                  f"stable={checks['stable_duration_s']:.2f}/{checks['required_stable_duration_s']:g}s")
        if "feedback_age_ms" in result:
            print(f"  feedback_age_ms={result['feedback_age_ms']} "
                  f"missing={result['missing_feedback_motors']} stale={result['stale_feedback_motors']}")


def _open_leg_controller(leg: str) -> LegController:
    return LegController(leg=leg, ctrl=DaMiaoMotorController(LEG_PORTS[leg]))


def _close_leg_controller(item: LegController) -> None:
    try:
        item.ctrl.close()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="四腿并联初始化；默认 fitted_j1（拟合节点质量＋J1残差）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="对照示例：--compensation nominal --preview；--compensation fitted --preview。"
               "去掉 --preview 执行运动。两种质量模式共用 Kp/Kd、频率、限幅及渐入设置。",
    )
    parser.add_argument("--motor_config", type=str, default=str(THIS_DIR / "damiao.yaml"), help="DaMiao gain yaml.")
    parser.add_argument("--kp", type=float, nargs=5, metavar=("J1", "J2", "J3", "J4", "J5"),
                        help="Override position gains for this initialization only; default comes from motor_config.")
    parser.add_argument("--kd", type=float, nargs=5, metavar=("J1", "J2", "J3", "J4", "J5"),
                        help="Override damping gains for this initialization only; default comes from motor_config.")
    modes = parser.add_argument_group("补偿方案")
    compensation = modes.add_mutually_exclusive_group()
    compensation.add_argument("--compensation", choices=("none", "nominal", "fitted", "fitted_j1", "table"), default="fitted_j1",
                              help="none: 零前馈；nominal: 电机500g/杆50g；fitted: 拟合节点质量；"
                                   "fitted_j1: 拟合质量＋J1残差（默认）；table: 静态双向表。")
    compensation.add_argument("--integral_compensation", action="store_true",
                              help=argparse.SUPPRESS)
    compensation.add_argument("--enable_gravity_compensation", action="store_true",
                              help=argparse.SUPPRESS)
    model_args = parser.add_argument_group("质量模型与静态表（共用控制设置）")
    model_args.add_argument("--node_mass_file", type=str, default=str(DEFAULT_NODE_MASS_FILE),
                           help="fitted 的六节点质量文件；只加载质量，不叠加拟合方向项或零偏。")
    model_args.add_argument("--j1_residual_file", type=str, default=None,
                           help="覆盖J1残差文件；fitted_j1自动加载默认文件，fitted仅显式指定时加载；无积分。")
    model_args.add_argument("--j1_residual_scale", type=float, default=1., help="J1残差比例 [0,1]。")
    model_args.add_argument("--j1_residual_limit_nm", type=float, default=1.5,
                           help="额外J1残差绝对限幅 N·m；总前馈仍受原限幅/变化率约束。")
    model_args.add_argument("--static_catalog", type=str, default=str(DEFAULT_STATIC_CATALOG),
                           help="table 模式的冻结静态双向表目录清单。")
    model_args.add_argument("--gravity_config", type=str, default=str(THIS_DIR / "gravity_compensation.yaml"),
                           help="共用几何/映射/力矩限幅/变化率配置；nominal 的部件质量也来自此文件。")
    model_args.add_argument("--gravity_urdf", type=str, default=None)
    model_args.add_argument("--ff_scale", "--gravity_scale", dest="gravity_scale", metavar="SCALE", type=float, default=1.,
                           help="模型前馈比例 [0,1]；各模型模式共用。")
    model_args.add_argument("--gravity_b", type=float, nargs=3, metavar=("GX", "GY", "GZ"), default=(0., 0., -9.81),
                        help="Fixed gravity in base_link coordinates, m/s^2; default assumes a level base.")
    model_args.add_argument("--gravity_finger_angle_deg", type=float, default=-30.,
                        help="Assumed simulated finger hinge angle; J6 is not read, enabled or commanded.")
    parser.add_argument("--preview", action="store_true", help="仅预览所选模型的目标/原始及限幅前馈，不打开串口。")
    parser.add_argument("--timing_jsonl", type=str, default=None, help="Append measured initialization timing reports.")
    parser.add_argument("--timing_interval_s", type=float, default=2., help="Interval between measured timing reports.")
    parser.add_argument("--live_diagnostics", action="store_true",
                        help="启用控制线程中途打印/时序写盘；默认仅结束后汇总，避免输出阻塞收发。")
    integral_args = parser.add_argument_group("旧积分标定参数（兼容 --integral_compensation）")
    integral_args.add_argument("--ki", type=float, nargs=5, default=(8., 8., 8., 8., 1.),
                        metavar=("J1", "J2", "J3", "J4", "J5"), help=argparse.SUPPRESS)
    integral_args.add_argument("--integral_torque_limit", type=float, nargs=5, default=(4., 1.5, 1.5, 1.5, .2),
                        metavar=("J1", "J2", "J3", "J4", "J5"), help=argparse.SUPPRESS)
    integral_args.add_argument("--integral_torque_rate", type=float, nargs=5, default=(.3, .15, .15, .15, .02),
                        metavar=("J1", "J2", "J3", "J4", "J5"), help=argparse.SUPPRESS)
    integral_args.add_argument("--integral_deadband_deg", type=float, default=.25,
                        help=argparse.SUPPRESS)
    integral_args.add_argument("--integral_velocity_gate_deg_s", type=float, default=5.,
                        help=argparse.SUPPRESS)
    parser.add_argument("--leg_order", nargs="+", default=list(DEFAULT_LEG_ORDER), choices=DEFAULT_LEG_ORDER,
                        help="Legs to run in parallel; order controls the final summary only.")
    parser.add_argument("--result_json", type=str, default=None,
                        help="Save the combined final angle/error/velocity/torque report as JSON.")
    parser.add_argument("--max_step_deg", type=float, default=2.0, help="Largest position change per ramp step.")
    parser.add_argument("--hz", type=float, default=None,
                        help="指令频率 Hz：模型模式默认25；none/旧积分默认10。")
    parser.add_argument("--inter_command_sleep", type=float, default=None,
                        help="同一腿内电机指令间隔 s：模型模式默认0.0002，其余0.02。")
    parser.add_argument("--settle_timeout_s", type=float, default=None,
                        help="最终保持超时 s：模型模式默认20，其余5。")
    parser.add_argument("--settle_time_s", type=float, default=0.5, help="Required continuous time within position/velocity tolerance.")
    parser.add_argument("--tolerance_deg", type=float, default=2.0, help="Maximum absolute error of every joint at completion.")
    parser.add_argument("--velocity_tolerance_deg_s", type=float, default=2.0, help="Maximum absolute joint speed at completion.")
    parser.add_argument("--feedback_timeout_s", type=float, default=0.5, help="Maximum per-motor feedback receipt age during verification.")
    startup_args = parser.add_argument_group("启动反馈（尚未发送位置目标）")
    startup_args.add_argument("--enable_attempts", type=int, default=3,
                              help="每个电机使能反馈缺失时的最大使能请求次数；故障反馈不重试。")
    startup_args.add_argument("--enable_feedback_timeout_s", type=float, default=.5,
                              help="每次驱动使能调用返回后继续等回复的时间 s；独立于运行期反馈超时。")
    parser.add_argument(
        "--fallback_current_deg",
        type=float,
        default=None,
        help="Use this current angle for every joint if feedback is unavailable.",
    )
    parser.add_argument(
        "--no_joint5_sign_flip",
        action="store_true",
        help="Do not apply the Joint5 sim-to-real negative sign used by play_real.py.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.compensation = ("fitted_j1" if args.enable_gravity_compensation else
                         "integral" if args.integral_compensation else args.compensation)
    args.enable_gravity_compensation = args.compensation in ("nominal", "fitted", "fitted_j1", "table")
    if args.j1_residual_file is not None and args.compensation not in ("fitted", "fitted_j1"):
        parser.error("--j1_residual_file requires --compensation fitted or fitted_j1")
    if not math.isfinite(args.j1_residual_scale) or not 0 <= args.j1_residual_scale <= 1:
        parser.error("--j1_residual_scale must be in [0,1]")
    if not math.isfinite(args.j1_residual_limit_nm) or not 0 < args.j1_residual_limit_nm <= 28:
        parser.error("--j1_residual_limit_nm must be in (0,28]")
    if len(set(args.leg_order)) != len(args.leg_order):
        parser.error("--leg_order must not contain duplicate legs/serial ports")
    if not 1 <= args.enable_attempts <= 10:
        parser.error("--enable_attempts must be in [1,10]")
    if args.hz is None:
        args.hz = 25. if args.enable_gravity_compensation else 10.
    if args.inter_command_sleep is None:
        args.inter_command_sleep = .0002 if args.enable_gravity_compensation else .02
    if args.settle_timeout_s is None:
        args.settle_timeout_s = 20. if args.enable_gravity_compensation else 5.

    for key in ("max_step_deg", "hz", "settle_timeout_s", "settle_time_s", "tolerance_deg",
                "velocity_tolerance_deg_s", "feedback_timeout_s", "timing_interval_s", "enable_feedback_timeout_s"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive and finite")
    for key in ("ki", "integral_torque_limit", "integral_torque_rate"):
        if any(not math.isfinite(value) or value < 0 for value in getattr(args, key)):
            parser.error(f"--{key} values must be nonnegative and finite")
    if any(value > 28 for value in args.integral_torque_limit):
        parser.error("--integral_torque_limit must not exceed the protocol's 28 N*m feedforward range")
    if not math.isfinite(args.integral_deadband_deg) or args.integral_deadband_deg < 0:
        parser.error("--integral_deadband_deg must be finite and nonnegative")
    if args.integral_compensation and args.integral_deadband_deg >= args.tolerance_deg:
        parser.error("--integral_deadband_deg must be smaller than --tolerance_deg when compensation is enabled")
    if not math.isfinite(args.integral_velocity_gate_deg_s) or args.integral_velocity_gate_deg_s <= 0:
        parser.error("--integral_velocity_gate_deg_s must be positive and finite")
    if not math.isfinite(args.inter_command_sleep) or args.inter_command_sleep < 0:
        parser.error("--inter_command_sleep must be nonnegative and finite")
    if args.settle_timeout_s <= args.settle_time_s:
        parser.error("--settle_timeout_s must exceed --settle_time_s")
    if args.fallback_current_deg is not None and not math.isfinite(args.fallback_current_deg):
        parser.error("--fallback_current_deg must be finite")
    if not math.isfinite(args.gravity_scale) or not 0 <= args.gravity_scale <= 1:
        parser.error("--gravity_scale must be finite and in [0,1]")
    if not all(math.isfinite(value) for value in (*args.gravity_b, args.gravity_finger_angle_deg)):
        parser.error("Gravity vector and finger angle must be finite")
    if args.enable_gravity_compensation and args.fallback_current_deg is not None:
        parser.error("Gravity compensation requires measured feedback; do not use --fallback_current_deg")
    kp, kd = _read_motor_cfg(args.motor_config)
    if args.kp is not None:
        kp = tuple(args.kp)
    if args.kd is not None:
        kd = tuple(args.kd)
    if any(not math.isfinite(value) or not 0 <= value <= maximum
           for gains, maximum in ((kp, 500), (kd, 5)) for value in gains):
        parser.error("Motor gains must be finite, kp in [0,500] and kd in [0,5]")
    print(f"[INFO] compensation={args.compensation}; kp={kp} kd={kd}; final-target timeout={args.settle_timeout_s:g}s, "
          f"position tolerance={args.tolerance_deg:g} deg, stable time={args.settle_time_s:g}s")
    gravity_model = None
    if args.enable_gravity_compensation:
        from gravity_compensation import GravityFeedforward

        if args.compensation == "table":
            from initialization_models import StaticTableModel

            gravity_model = StaticTableModel(args.gravity_config, args.gravity_urdf, args.static_catalog, args.leg_order)
        else:
            gravity_model = make_compensation_model(
                args.compensation, args.gravity_config, args.gravity_urdf, args.node_mass_file,
                args.j1_residual_file, args.j1_residual_scale, args.j1_residual_limit_nm)
        print("[INFO] Feedforward model:", json.dumps(gravity_model.description(), ensure_ascii=False))
        print(f"[INFO] Feedforward scale={args.gravity_scale:g}, gravity_b={args.gravity_b} m/s^2, "
              f"assumed finger angle={args.gravity_finger_angle_deg:g} deg; "
              f"command rate={args.hz:g} Hz, inter-command delay={args.inter_command_sleep:g}s; "
              "ramp and hold share one zero-start controller, no integral learning")
    if args.integral_compensation:
        print(f"[INFO] Integral compensation: ki={tuple(args.ki)}, "
              f"feedforward limits={tuple(args.integral_torque_limit)} N*m, "
              f"rate limits={tuple(args.integral_torque_rate)} N*m/s; PD torque is additional")
    max_step_rad = math.radians(max(args.max_step_deg, 0.1))

    if args.preview:
        preview = []
        for leg in args.leg_order:
            target = _target_rad_for_leg(leg, joint5_sign_flip=not args.no_joint5_sign_flip)
            entry = dict(leg=leg, compensation=args.compensation,
                         hardware_target_deg=[math.degrees(value) for value in target],
                         raw_ff_nm=[0.]*5, bounded_ff_nm=[0.]*5)
            if gravity_model is not None:
                raw = args.gravity_scale * gravity_model.leg_torque(
                    leg, target, args.gravity_b, math.radians(args.gravity_finger_angle_deg))
                entry.update(gravity_raw_ff_nm=raw.tolist(),
                             gravity_bounded_ff_nm=[max(-limit, min(limit, value)) for value, limit
                                                    in zip(raw, gravity_model.torque_limits)])
                entry.update(raw_ff_nm=entry["gravity_raw_ff_nm"], bounded_ff_nm=entry["gravity_bounded_ff_nm"])
            preview.append(entry)
        print("[PREVIEW] Target-pose calculation only; no feedback, no ports, no runtime slew simulation.")
        print(json.dumps(preview, ensure_ascii=False, indent=2, allow_nan=False))
        return

    parallel = ParallelInitialization(args.leg_order)
    results = {leg: dict(compensation=args.compensation, target_deg=[math.degrees(v) for v in _target_rad_for_leg(
        leg, joint5_sign_flip=not args.no_joint5_sign_flip)], reached=False) for leg in args.leg_order}
    items, timings = {}, {}

    def initialize_leg(leg):
        timing = InitializationTiming(leg, args.hz, args.timing_jsonl, args.timing_interval_s,
                                      live_reports=args.live_diagnostics)
        timings[leg] = timing
        try:
            item = _open_leg_controller(leg)
            items[leg] = item
            targets_rad = _target_rad_for_leg(leg, joint5_sign_flip=not args.no_joint5_sign_flip)
            current_rad: list[float] = []
            results[leg]["startup_feedback"] = {}
            for motor_id in MOTOR_IDS:
                parallel.check()
                diagnostics = results[leg]["startup_feedback"][motor_id] = {}
                pos = _enable_and_read_current(item.ctrl, motor_id, attempts=args.enable_attempts,
                                              feedback_timeout_s=args.enable_feedback_timeout_s,
                                              diagnostics=diagnostics, cancel_check=parallel.check)
                if pos is None:
                    if args.fallback_current_deg is None:
                        raise RuntimeError(
                            f"{leg} motor {motor_id} feedback unavailable on {LEG_PORTS[leg]} "
                            f"after {args.enable_attempts} enable attempts. "
                            "No position ramp started; check port mapping, motor power/ID and serial ownership. "
                            f"Startup diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}"
                        )
                    pos = math.radians(args.fallback_current_deg)
                current_rad.append(pos)
                time.sleep(0.1)
            results[leg]["initial_deg"] = [math.degrees(value) for value in current_rad]

            # Do not start gravity feedback-age timers while another leg is still enabling.
            parallel.wait()
            gravity = None if gravity_model is None else InitializationGravity(
                leg, GravityFeedforward(gravity_model, args.gravity_scale, 1. / args.hz),
                args.feedback_timeout_s, args.gravity_b, math.radians(args.gravity_finger_angle_deg), timing)
            _send_leg_position_targets(
                item=item,
                targets_rad=targets_rad,
                current_rad=current_rad,
                kp=kp,
                kd=kd,
                max_step_rad=max_step_rad,
                hz=args.hz,
                inter_command_sleep=args.inter_command_sleep,
                gravity_compensator=gravity,
                timing=timing,
                parallel=parallel,
            )
            compensator = IntegralTorqueCompensator(
                ki=tuple(args.ki), torque_limit=tuple(args.integral_torque_limit),
                torque_rate=tuple(args.integral_torque_rate), deadband_rad=math.radians(args.integral_deadband_deg),
                velocity_gate_rad_s=math.radians(args.integral_velocity_gate_deg_s), max_dt_s=1. / args.hz,
            ) if args.integral_compensation else None
            reached = _hold_and_verify_leg(
                item, targets_rad, kp, kd, args.hz, args.inter_command_sleep,
                args.settle_timeout_s, args.settle_time_s, math.radians(args.tolerance_deg),
                math.radians(args.velocity_tolerance_deg_s), args.feedback_timeout_s,
                integral_compensator=compensator,
                gravity_compensator=gravity,
                timing=timing,
                result_out=results[leg],
                parallel=parallel,
                report=False,
            )
            results[leg]["reached"] = reached
            results[leg]["io_diagnostics"] = timing.io_snapshot()
        except BaseException as exc:
            timing.error = f"{type(exc).__name__}: {exc}"
            results[leg]["error"] = timing.error
            results[leg]["reached"] = False
            results[leg]["io_diagnostics"] = timing.io_snapshot()
            parallel.abort()
            raise

    print(f"[INFO] Parallel legs={args.leg_order}; common ramp steps, then all-leg stable verification")
    executor = ThreadPoolExecutor(max_workers=len(args.leg_order), thread_name_prefix="init-pose")
    futures = []
    failure = None
    try:
        futures = [executor.submit(initialize_leg, leg) for leg in args.leg_order]
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                parallel.abort()
                # Prefer the actual transport/motor error over a peer's cancelled barrier.
                if failure is None or isinstance(failure, threading.BrokenBarrierError):
                    failure = exc
    except BaseException as exc:
        failure = exc
        parallel.abort()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for leg in args.leg_order:
            if leg in items:
                _close_leg_controller(items[leg])
        for leg in args.leg_order:
            try:
                if leg in timings:
                    timings[leg].report(final=True)
            except Exception as exc:
                print(f"[WARN] Timing report failed: {exc}")
        _print_initialization_summary(args.leg_order, results)
        if args.result_json is not None:
            path = Path(args.result_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            configuration = dict(compensation=args.compensation, kp=kp, kd=kd, hz=args.hz,
                                 control_loop_mode="independent_per_leg",
                                 phase_offset_s={leg: index/args.hz/len(args.leg_order)
                                                 for index, leg in enumerate(args.leg_order)},
                                 feedback_timeout_s=args.feedback_timeout_s,
                                 live_diagnostics=args.live_diagnostics,
                                 inter_command_sleep=args.inter_command_sleep, max_step_deg=args.max_step_deg,
                                 settle_timeout_s=args.settle_timeout_s, settle_time_s=args.settle_time_s,
                                 tolerance_deg=args.tolerance_deg, velocity_tolerance_deg_s=args.velocity_tolerance_deg_s,
                                 enable_attempts=args.enable_attempts,
                                 enable_feedback_timeout_s=args.enable_feedback_timeout_s,
                                 ff_scale=args.gravity_scale, gravity_b=args.gravity_b,
                                 finger_angle_deg=args.gravity_finger_angle_deg,
                                 model=gravity_model.description() if gravity_model is not None else None)
            path.write_text(json.dumps(dict(mode="parallel", configuration=configuration, legs=results), ensure_ascii=False,
                                       indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if failure is not None:
        raise failure
    failed_legs = [leg for leg in args.leg_order if not results[leg]["reached"]]
    if failed_legs:
        raise SystemExit(f"[ERROR] Pose not reached on {failed_legs}; acceptance criteria not met "
                         f"(position<={args.tolerance_deg:g} deg, velocity<={args.velocity_tolerance_deg_s:g} deg/s, "
                         f"stable>={args.settle_time_s:g}s). Inspect acceptance details above; "
                         "this status alone is not a communication fault.")


if __name__ == "__main__":
    main()
