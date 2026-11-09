"""Single-leg zero/random-pose and sine tracking tests. Preview by default.

All angles are hardware angles. Commands/feedback use host timestamps, not
device sampling timestamps. No integral learning and no automatic reconnect.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from gravity_compensation import GravityFeedforward
from initialization_models import DEFAULT_J1_RESIDUAL_FILE, make_compensation_model
from init_default_joint_pose import (
    DEFAULT_LEG_ORDER, DEFAULT_NODE_MASS_FILE, LEG_PORTS, MOTOR_IDS, THIS_DIR,
    InitializationGravity, _checked_feedback, _enable_and_read_current,
    _read_motor_cfg, _target_rad_for_leg,
)
from motor_control import DaMiaoMotorController


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--leg", choices=DEFAULT_LEG_ORDER, default="LB")
    p.add_argument("--mode", choices=("random", "sine"), default="random")
    p.add_argument("--compensation", choices=("nominal", "fitted", "fitted_j1"), default="fitted_j1")
    p.add_argument("--node_mass_file", type=Path, default=DEFAULT_NODE_MASS_FILE)
    p.add_argument("--j1_residual_file", type=Path,
                   default=DEFAULT_J1_RESIDUAL_FILE)
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--j1_range_deg", type=float, default=35.)
    p.add_argument("--other_range_deg", type=float, default=20.)
    p.add_argument("--max_speed_deg_s", type=float, default=15.)
    p.add_argument("--max_accel_deg_s2", type=float, default=30.)
    p.add_argument("--frequency_hz", type=float, default=.05)
    p.add_argument("--cycles", type=int, default=5, help="First/last cycle fade in/out; default leaves three steady cycles")
    p.add_argument("--hold_s", type=float, default=3.)
    p.add_argument("--zero_timeout_s", type=float, default=25., help="Maximum zero/center verification wait; record failure and continue on timeout")
    p.add_argument("--tolerance_deg", type=float, default=2.)
    p.add_argument("--stable_s", type=float, default=.5)
    p.add_argument("--velocity_tolerance_deg_s", type=float, default=2.)
    p.add_argument("--monitor_margin_deg", type=float, default=5.)
    p.add_argument("--feedback_timeout_s", type=float, default=.5)
    p.add_argument("--hz", type=float, default=25.)
    p.add_argument("--poll_hz", type=float, default=200.)
    p.add_argument("--kp", type=float, nargs=5)
    p.add_argument("--kd", type=float, nargs=5)
    p.add_argument("--port", help="Override the selected leg's serial path")
    p.add_argument("--enable_motors", action="store_true", help="Enable this leg and execute; default only saves plan")
    p.add_argument("--output_dir", type=Path)
    return p


def validate(a):
    positive = ("j1_range_deg", "other_range_deg", "max_speed_deg_s", "max_accel_deg_s2", "frequency_hz",
                "hold_s", "zero_timeout_s", "tolerance_deg", "stable_s", "velocity_tolerance_deg_s",
                "feedback_timeout_s", "hz", "poll_hz", "monitor_margin_deg")
    if any(not math.isfinite(getattr(a, k)) or getattr(a, k) <= 0 for k in positive):
        raise ValueError("Rates, ranges, times and tolerances must be finite and positive")
    if not 1 <= a.points <= 100 or a.cycles < 3 or a.hz != 25 or a.poll_hz < a.hz:
        raise ValueError("Require 1..100 points, >=3 sine cycles, hz=25 and poll_hz>=hz")
    if a.zero_timeout_s <= a.stable_s or a.hold_s <= a.stable_s:
        raise ValueError("Hold and zero timeout must exceed stable_s")
    if a.j1_range_deg > 35 or a.other_range_deg > 20:
        raise ValueError("Requested ranges may not exceed J1 +/-35 deg and other joints +/-20 deg")
    kp, kd = _read_motor_cfg(str(THIS_DIR/"damiao.yaml"))
    a.kp, a.kd = list(a.kp or kp), list(a.kd or kd)
    if any(not math.isfinite(v) or not 0 <= v <= bound
           for values, bound in ((a.kp, 500), (a.kd, 5)) for v in values):
        raise ValueError("Expected kp in [0,500], kd in [0,5]")
    amplitudes = np.array([a.j1_range_deg]+[a.other_range_deg]*4)
    if a.mode == "sine":
        omega = 2*math.pi*a.frequency_hz
        fade = 1/a.frequency_hz
        peak_v = max(amplitudes)*(omega+1.875/fade)
        peak_a = max(amplitudes)*(omega**2+2*omega*1.875/fade+(10/math.sqrt(3))/fade**2)
        if peak_v > a.max_speed_deg_s or peak_a > a.max_accel_deg_s2:
            raise ValueError("Sine amplitude/frequency exceeds max speed or acceleration; reduce frequency/range")
        if a.hz/a.frequency_hz < 20:
            raise ValueError("Sine requires at least 20 target updates per cycle")


def make_model(a):
    return make_compensation_model(a.compensation, node_mass_file=a.node_mass_file,
                                   j1_residual_file=a.j1_residual_file if a.compensation == "fitted_j1" else None)


def move_duration(start, end, speed, accel, hz=25.):
    delta = float(np.max(np.abs(np.asarray(end)-start)))
    # Quintic smootherstep: max |s'|=1.875, max |s''|=10/sqrt(3).
    return max(2/hz, 1.875*delta/speed, math.sqrt((10/math.sqrt(3))*delta/accel))


def smooth_pose(start, end, fraction):
    u = float(np.clip(fraction, 0, 1))
    return np.asarray(start)+(10*u**3-15*u**4+6*u**5)*(np.asarray(end)-start)


def sine_pose(center, amplitude, elapsed, frequency, cycles):
    duration = cycles/frequency
    fraction = min(1., max(0., elapsed*frequency), max(0., (duration-elapsed)*frequency))
    envelope = float(smooth_pose(0.,1.,fraction))
    return np.asarray(center)+np.asarray(amplitude)*envelope*math.sin(2*math.pi*frequency*elapsed)


def make_plan(a):
    center = np.rad2deg(_target_rad_for_leg(a.leg, True))
    ranges = np.array([a.j1_range_deg]+[a.other_range_deg]*4)
    rng = np.random.default_rng(a.seed)
    targets = rng.uniform(center-ranges, center+ranges, (a.points, 5))
    return dict(leg=a.leg, mode=a.mode, zero_pose_deg=[0.]*5, center_deg=center.tolist(),
                target_min_deg=(center-ranges).tolist(), target_max_deg=(center+ranges).tolist(),
                monitor_min_deg=(np.minimum(0, center-ranges)-a.monitor_margin_deg).tolist(),
                monitor_max_deg=(np.maximum(0, center+ranges)+a.monitor_margin_deg).tolist(),
                targets_deg=targets.tolist() if a.mode == "random" else [],
                zero_to_target_duration_s=[move_duration(np.zeros(5), q, a.max_speed_deg_s,
                                                         a.max_accel_deg_s2, a.hz) for q in targets]
                    if a.mode == "random" else [],
                sine_amplitude_deg=ranges.tolist(), frequency_hz=a.frequency_hz, cycles=a.cycles,
                sine_peak_speed_bound_deg_s=(ranges*a.frequency_hz*(2*math.pi+1.875)).tolist(),
                sine_peak_accel_bound_deg_s2=(ranges*a.frequency_hz**2*((2*math.pi)**2+4*math.pi*1.875+10/math.sqrt(3))).tolist(),
                note="Single leg. All five joints are commanded to hardware zero before every random pose; "
                     "unmet accuracy/stability checks are recorded and do not stop the plan. "
                     "Measured-start-to-zero duration is computed after enabling. No integral learning.")


class Session:
    def __init__(self, a, ctrl, model, current, plan):
        self.a, self.ctrl, self.plan = a, ctrl, plan
        self.started = time.monotonic()
        self.last_tx = None
        self.command = np.asarray(current, dtype=float)
        self.comp = InitializationGravity(a.leg, GravityFeedforward(model, max_dt_s=1/a.hz),
                                          a.feedback_timeout_s, [0.,0.,-9.81], -math.pi/6)
        self.commands, self.feedback, self.holds = [], [], []
        self.target_clips = 0
        self.previous_delta = np.zeros(5)

    def poll(self, phase, point):
        frames, pending, backlog, overflow = self.ctrl.receive_available()
        if overflow:
            raise RuntimeError("Feedback queue overflow")
        for frame in frames:
            f = _checked_feedback(self.ctrl.damiao_feedback(frame))
            if f is None:
                continue
            now = time.monotonic()
            position = math.degrees(f["position"])
            index = f["motor_id"]-1
            if not self.plan["monitor_min_deg"][index] <= position <= self.plan["monitor_max_deg"][index]:
                raise RuntimeError(f"J{index+1} measured angle {position:.3f} outside monitoring envelope")
            self.comp.observe({f["motor_id"]: f}, now)
            self.feedback.append(dict(time_s=now-self.started, phase=phase, point=point, joint=index+1,
                                      position_deg=position, velocity_deg_s=math.degrees(f["velocity"]),
                                      torque_nm=f["torque"], backlog_bytes=backlog))
        now = time.monotonic()
        stale = [m for m in MOTOR_IDS if now-self.comp.received_at.get(m, self.comp.started) > self.a.feedback_timeout_s]
        if stale:
            raise RuntimeError(f"Missing/stale feedback from motors {stale}; no reconnect or automatic re-enable")

    def tick(self, requested, phase, point, logical_s):
        self.poll(phase, point)
        now = time.monotonic()
        # Use the same absolute deadline as pause(): subtraction can round an
        # exactly reached deadline below one period and produce a zero-sleep loop.
        if self.last_tx is not None and now < self.last_tx+1/self.a.hz:
            return False
        dt = 1/self.a.hz if self.last_tx is None else min(now-self.last_tx, 1/self.a.hz)
        desired = np.asarray(requested, dtype=float)
        # No catch-up jump after an OS pause; the logical path advances once per TX.
        delta = np.clip(desired-self.command, -self.a.max_speed_deg_s*dt, self.a.max_speed_deg_s*dt)
        delta = np.clip(delta, self.previous_delta-self.a.max_accel_deg_s2*dt**2,
                         self.previous_delta+self.a.max_accel_deg_s2*dt**2)
        sent = self.command+delta
        if np.any(sent < self.plan["monitor_min_deg"]) or np.any(sent > self.plan["monitor_max_deg"]):
            raise RuntimeError("Rate-limited command outside monitoring envelope")
        clipped = bool(np.max(abs(sent-desired)) > 1e-7)
        self.target_clips += int(clipped)
        torques = self.comp.compute(now)
        model = self.comp.controller.model
        residual = model.leg_diagnostics(self.a.leg) if hasattr(model,"leg_diagnostics") else {}
        self.last_tx = now
        self.command = sent
        self.previous_delta = delta
        index = DEFAULT_LEG_ORDER.index(self.a.leg)
        for j, motor in enumerate(MOTOR_IDS):
            stamp = time.monotonic()-self.started
            self.ctrl.send(self.ctrl.damiao_mit_command(pos=math.radians(sent[j]), vel=0.,
                           kp=self.a.kp[j], kd=self.a.kd[j], tor=torques[j], motor_id=motor))
            self.commands.append(dict(time_s=stamp, phase=phase, point=point, logical_s=logical_s, joint=motor,
                                      requested_deg=float(desired[j]), target_deg=float(sent[j]),
                                      ff_nm=torques[j], raw_ff_nm=float(self.comp.controller.raw[index+4*j]),
                                      j1_residual_nm=residual.get("j1_residual_nm",0.),
                                      j1_residual_direction=residual.get("j1_residual_direction",0),
                                      j1_residual_angle_clamped=residual.get("j1_residual_angle_clamped",False),
                                      target_rate_limited=clipped))
            time.sleep(.0002)
        return True

    def segment(self, duration, reference, phase, point=-1):
        steps = max(1, math.ceil(duration*self.a.hz))
        k = 0
        while k <= steps:
            logical = min(k/self.a.hz, duration)
            if self.tick(reference(logical), phase, point, logical):
                k += 1
            self.pause()

    def pause(self):
        remaining = (self.last_tx+1/self.a.hz-time.monotonic()) if self.last_tx is not None else 1/self.a.poll_hz
        time.sleep(min(1/self.a.poll_hz,max(0.,remaining)))

    def move(self, target, phase, point=-1):
        start = self.command.copy()
        duration = move_duration(start, target, self.a.max_speed_deg_s, self.a.max_accel_deg_s2, self.a.hz)
        self.segment(duration, lambda t: smooth_pose(start, target, t/duration), phase, point)

    def hold(self, target, phase, point=-1, verify=False):
        started, stable, first_ready, ready = time.monotonic(), None, None, False
        limit = self.a.zero_timeout_s if verify else self.a.hold_s
        target = np.asarray(target)
        while time.monotonic()-started < limit:
            self.tick(target, phase, point, time.monotonic()-started)
            now = time.monotonic()
            valid = all(m in self.comp.latest and
                        abs(math.degrees(self.comp.latest[m]["position"])-target[m-1]) <= self.a.tolerance_deg and
                        abs(math.degrees(self.comp.latest[m]["velocity"])) <= self.a.velocity_tolerance_deg_s
                        for m in MOTOR_IDS) and self.comp.ramp_complete()
            stable = (stable if stable is not None else now) if valid else None
            ready = (stable is not None and now-stable >= self.a.stable_s and
                     all(self.comp.received_at[m] > stable for m in MOTOR_IDS))
            if ready and first_ready is None:
                first_ready = now-started-self.a.stable_s
            if verify and ready:
                break
            self.pause()
        latest = [self.comp.latest.get(m, {}).get("position") for m in MOTOR_IDS]
        errors = [float(target[i]-math.degrees(q)) if q is not None else None for i,q in enumerate(latest)]
        outcome = dict(phase=phase, point=point, target_deg=target.tolist(), error_deg=errors,
                       accepted=bool(ready), elapsed_s=time.monotonic()-started,
                       observed_settling_ms=max(0.,first_ready)*1000 if first_ready is not None else None,
                       stable_duration_s=time.monotonic()-stable if stable is not None else 0.)
        outcome["position_outside_tolerance_joints"] = [j for j, error in zip(MOTOR_IDS, errors)
                                                        if error is None or abs(error) > self.a.tolerance_deg]
        outcome["velocity_deg_s"] = [math.degrees(self.comp.latest[m]["velocity"]) if m in self.comp.latest else None
                                     for m in MOTOR_IDS]
        outcome["velocity_outside_tolerance_joints"] = [j for j, speed in zip(MOTOR_IDS, outcome["velocity_deg_s"])
                                                        if speed is None or abs(speed) > self.a.velocity_tolerance_deg_s]
        outcome["gravity_ramp_complete"] = self.comp.ramp_complete()
        tail = [f for f in self.feedback if f["phase"] == phase and f["point"] == point
                and f["time_s"] >= max(started-self.started,time.monotonic()-self.started-1.)]
        outcome["steady_error_median_deg"] = [float(np.median([target[j-1]-f["position_deg"] for f in tail if f["joint"]==j]))
                                               if any(f["joint"]==j for f in tail) else None for j in MOTOR_IDS]
        self.holds.append(outcome)
        return outcome


def analyze(a, commands, feedback, holds):
    groups = []
    for phase in ("outbound", "target_hold", "return_zero", "sine"):
        for joint in MOTOR_IDS:
            c = [r for r in commands if r["joint"] == joint]
            samples = [r for r in feedback if r["joint"] == joint and r["phase"] == phase]
            times = [r["time_s"] for r in c]
            paired = [(f, bisect.bisect_right(times, f["time_s"])-1) for f in samples]
            paired = [(f,i) for f,i in paired if i >= 0 and c[i]["phase"] == phase]
            if not paired:
                continue
            errors = np.array([c[i]["target_deg"]-f["position_deg"] for f,i in paired])
            requested = np.array([c[i]["requested_deg"]-f["position_deg"] for f,i in paired])
            item = dict(phase=phase, joint=joint, samples=len(paired), mae_deg=float(abs(errors).mean()),
                        rmse_deg=float(np.sqrt(np.mean(errors**2))), p95_abs_deg=float(np.percentile(abs(errors),95)),
                        max_abs_deg=float(abs(errors).max()), within_tolerance_fraction=float(np.mean(abs(errors)<=a.tolerance_deg)),
                        requested_rmse_deg=float(np.sqrt(np.mean(requested**2))))
            if phase == "sine":
                usable = [(f,i) for f,i in paired if 1/a.frequency_hz <= c[i]["logical_s"] <= (a.cycles-1)/a.frequency_hz]
                if len(usable) >= 12:
                    t = np.array([c[i]["logical_s"] for f,i in usable])
                    matrix = np.column_stack((np.ones(len(t)),np.sin(2*np.pi*a.frequency_hz*t),np.cos(2*np.pi*a.frequency_hz*t)))
                    if np.ptp(t) < .9/a.frequency_hz or np.linalg.matrix_rank(matrix) < 3:
                        item["phase_note"] = "Insufficient steady-cycle coverage to identify phase delay"
                        groups.append(item)
                        continue
                    ref = np.linalg.lstsq(matrix,[c[i]["target_deg"] for f,i in usable],rcond=None)[0]
                    actual = np.linalg.lstsq(matrix,[f["position_deg"] for f,i in usable],rcond=None)[0]
                    ref_amp, amp = float(np.hypot(*ref[1:])), float(np.hypot(*actual[1:]))
                    identifiable = ref_amp > .02 and amp > max(.02,.01*ref_amp)
                    phase_diff = (math.atan2(ref[2],ref[1])-math.atan2(actual[2],actual[1])+math.pi)%(2*math.pi)-math.pi
                    effective_hz = a.frequency_hz*(t[-1]-t[0])/(usable[-1][0]["time_s"]-usable[0][0]["time_s"])
                    item.update(amplitude_gain=amp/ref_amp if ref_amp>.02 else None,
                                observed_phase_delay_ms=phase_diff/(2*math.pi*effective_hz)*1000 if identifiable else None,
                                effective_frequency_hz=effective_hz,
                                phase_note="Host-receipt fit versus transmitted reference; not device ACK latency")
            groups.append(item)
    cadence = {}
    for joint in MOTOR_IDS:
        times = np.array([r["time_s"] for r in commands if r["joint"] == joint])
        if len(times) > 1:
            gaps = np.diff(times)
            positions = np.array([r["target_deg"] for r in commands if r["joint"] == joint])
            cadence[str(joint)] = dict(actual_tx_hz=float((len(times)-1)/(times[-1]-times[0])),
                                       min_interval_ms=float(gaps.min()*1000),max_interval_ms=float(gaps.max()*1000),
                                       max_observed_target_rate_deg_s=float(np.max(abs(np.diff(positions)/gaps))))
    tested = [h for h in holds if h["phase"] == "target_hold"]
    return dict(metrics=groups, holds=holds, tolerance_deg=a.tolerance_deg, command_cadence_by_joint=cadence,
                accepted_random_points=sum(h["accepted"] for h in tested), random_points_tested=len(tested),
                interpretation="Absolute tracking to transmitted ZOH references; requested-reference RMSE also reported. "
                               "Zero-based static accuracy comes from target_hold, sine response from amplitude/phase. "
                               "No independent device sample timestamps; no 40ms settling certification.")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_report(out, a, summary, commands, feedback):
    lines = [f"# {a.leg} {a.mode} 全腿模型跟踪测试", "",
             f"运行完成：{summary['complete']}；随机目标通过：{summary['accepted_random_points']}/{summary['random_points_tested']}。",
             f"角度容差 {a.tolerance_deg:g}°；位置目标限速 {a.max_speed_deg_s:g}°/s，名义离散加速度限幅 {a.max_accel_deg_s2:g}°/s²。",
             f"KP（J1–J5）：{a.kp}；KD：{a.kd}。",
             "精度或稳定性未达标只记录，不中止后续阶段；完成测试不代表验收通过。",
             "", "| 阶段 | 关节 | MAE° | RMSE° | P95绝对误差° | 最大误差° | 容差内比例 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in summary["metrics"]:
        lines.append(f"| {m['phase']} | J{m['joint']} | {m['mae_deg']:.3f} | {m['rmse_deg']:.3f} | "
                     f"{m['p95_abs_deg']:.3f} | {m['max_abs_deg']:.3f} | {m['within_tolerance_fraction']:.1%} |")
        if "amplitude_gain" in m:
            lines.append(f"\nJ{m['joint']} 正弦幅值比：{m['amplitude_gain']}；观测相位延迟：{m['observed_phase_delay_ms']} ms；"
                         f"实际频率约 {m['effective_frequency_hz']:.4f} Hz。\n")
    lines += ["", "| 关节 | 实际发送Hz | 最大间隔ms | 最大目标变化率°/s |",
              "| --- | --- | --- | --- |"]
    for joint, cadence in summary["command_cadence_by_joint"].items():
        lines.append(f"| J{joint} | {cadence['actual_tx_hz']:.3f} | {cadence['max_interval_ms']:.3f} | "
                     f"{cadence['max_observed_target_rate_deg_s']:.3f} |")
    lines += ["", "| 保持阶段 | 点号 | 验收通过 | 最终误差J1–J5（°） |",
              "| --- | --- | --- | --- |"]
    for hold in summary["holds"]:
        errors = ", ".join("null" if v is None else f"{v:.3f}" for v in hold["error_deg"])
        lines.append(f"| {hold['phase']} | {hold['point']} | {hold['accepted']} | {errors} |")
    lines += ["", "每个目标最终误差、观察到的稳定时间及通过状态见summary.json的holds。",
              "commands.csv保存实际发送目标、未限速请求和前馈；feedback.csv保存五电机反馈角度/速度/力矩。",
              "响应指标基于主机收发时间；相位估计剔除首尾渐入/渐出周期，不等于设备执行ACK延迟。",
              "轨迹过渡采用五次曲线；正弦首尾幅值平滑渐变。位置目标最大单周期变化和离散加速度再次限幅，",
              "延迟后不追发跳变；实际电机加速度由反馈检查，不由名义目标限幅保证。",
              "前馈仍受原模型绝对限幅及变化率约束，不使用积分；全零位和跨姿态属于待验证区域。", ""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    if not commands or not feedback:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # CSV/JSON/Markdown remain usable on minimal motor deployments.
    fig, axes = plt.subplots(5,2,figsize=(13,15))
    for joint in MOTOR_IDS:
        c = [r for r in commands if r["joint"] == joint]
        f = [r for r in feedback if r["joint"] == joint]
        axes[joint-1,0].plot([r['time_s'] for r in c],[r['target_deg'] for r in c],label='sent target')
        axes[joint-1,0].plot([r['time_s'] for r in f],[r['position_deg'] for r in f],label='measured')
        times = [r['time_s'] for r in c]
        pairs = [(r,bisect.bisect_right(times,r['time_s'])-1) for r in f]
        pairs = [(r,i) for r,i in pairs if i>=0]
        axes[joint-1,1].plot([r['time_s'] for r,i in pairs],[c[i]['target_deg']-r['position_deg'] for r,i in pairs])
        for limit in (-a.tolerance_deg,a.tolerance_deg):axes[joint-1,1].axhline(limit,color='gray',ls='--')
        axes[joint-1,0].set_ylabel(f'J{joint} angle (deg)')
        axes[joint-1,1].set_ylabel(f'J{joint} error (deg)')
        for ax in axes[joint-1]:ax.set_xlabel('Host time (s)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle(f'{a.leg}: {a.mode}, {a.compensation}')
    fig.tight_layout();fig.savefig(out/'tracking.png',dpi=150);plt.close(fig)


def run(a):
    validate(a)
    model = make_model(a)
    plan = make_plan(a)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    out = a.output_dir or Path(f"logs/real_v3/model_tracking_{a.leg}_{a.mode}_{stamp}")
    out.mkdir(parents=True, exist_ok=False)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    meta = dict(settings=settings, plan=plan, model=model.description(), complete=False,
                run_type="hardware" if a.enable_motors else "preview", error=None,
                tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    if not a.enable_motors:
        print(json.dumps(dict(preview=True, output_dir=str(out), plan=plan,
                              max_speed_deg_s=a.max_speed_deg_s, max_accel_deg_s2=a.max_accel_deg_s2,
                              torque_limits_nm=model.torque_limits.tolist(), torque_slew_nm_s=model.slew_rates.tolist()),indent=2))
        return out
    ctrl, session = None, None
    try:
        ctrl = DaMiaoMotorController(a.port or LEG_PORTS[a.leg])
        current = [_enable_and_read_current(ctrl,m,attempts=3,feedback_timeout_s=.5) for m in MOTOR_IDS]
        if any(v is None for v in current):
            raise RuntimeError("Initial feedback missing; no position targets sent")
        current_deg = np.rad2deg(current)
        meta["measured_start_deg"] = current_deg.tolist()
        if np.any(current_deg < plan["monitor_min_deg"]) or np.any(current_deg > plan["monitor_max_deg"]):
            raise RuntimeError("Measured start outside monitor envelope; no position targets sent")
        session = Session(a,ctrl,model,current_deg,plan)
        zero, center = np.zeros(5), np.array(plan["center_deg"])
        session.move(zero,"initialize_zero")
        session.hold(zero,"verify_zero",verify=True)
        if a.mode == "random":
            for point, target in enumerate(plan["targets_deg"]):
                session.move(target,"outbound",point)
                session.hold(target,"target_hold",point)
                session.move(zero,"return_zero",point)
                session.hold(zero,"verify_zero",point,verify=True)
        else:
            session.move(center,"sine_entry")
            session.hold(center,"verify_center",verify=True)
            amplitude = np.array(plan["sine_amplitude_deg"])
            session.segment(a.cycles/a.frequency_hz,
                            lambda t:sine_pose(center,amplitude,t,a.frequency_hz,a.cycles),"sine")
            session.move(zero,"return_zero")
            session.hold(zero,"verify_zero",verify=True)
        meta["complete"] = True
    except BaseException as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if ctrl is not None:
            ctrl.close()  # Preserve existing close-only semantics; no extra motor command on faults.
        if session is not None:
            write_csv(out/"commands.csv",session.commands)
            write_csv(out/"feedback.csv",session.feedback)
            summary = analyze(a,session.commands,session.feedback,session.holds)
            summary.update(complete=meta["complete"],error=meta["error"],target_rate_limited_cycles=session.target_clips)
            (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
            try:
                write_report(out,a,summary,session.commands,session.feedback)
            except Exception as report_error:
                meta["report_error"] = str(report_error)
            print(json.dumps(dict(output_dir=str(out),complete=meta["complete"],holds=session.holds),indent=2))
        (out/"run.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    return out


if __name__ == "__main__":
    run(build_parser().parse_args())
