"""Experimental angle-table load and directional-friction feedforward.

The table is conditional on the other joints/support posture used for calibration;
it is not a whole-robot gravity model. No hardware is opened by this module.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from statistics import median


class LoadFrictionModel:
    def __init__(self, data):
        self.data = data
        if data.get("version") != 1 or data.get("leg") not in ("LB", "LF", "RB", "RF"):
            raise ValueError("Unsupported compensation model version/leg")
        if data.get("joint") not in (1, 2, 3, 4, 5):
            raise ValueError("Compensation model joint must be Joint1-5")
        self.angles = tuple(float(value) for value in data["load_angle_deg"])
        self.loads = tuple(float(value) for value in data["load_torque_nm"])
        if len(self.angles) < 2 or len(self.angles) != len(self.loads):
            raise ValueError("Load table needs at least two paired angle/torque entries")
        if (not all(math.isfinite(value) for value in self.angles + self.loads)
                or any(b <= a for a, b in zip(self.angles, self.angles[1:]))):
            raise ValueError("Load table angles must be finite and strictly increasing")
        for key in ("friction_torque_nm", "velocity_soft_deg_s", "filter_tau_s",
                    "slew_rate_nm_s", "feedforward_limit_nm", "max_interpolation_gap_deg"):
            value = float(data[key])
            if not math.isfinite(value) or (value < 0 if key == "friction_torque_nm" else value <= 0):
                raise ValueError(f"{key} must be finite and {'nonnegative' if key == 'friction_torque_nm' else 'positive'}")
            setattr(self, key, value)
        if (self.feedforward_limit_nm > 28 or self.friction_torque_nm > self.feedforward_limit_nm
                or any(abs(value) > self.feedforward_limit_nm for value in self.loads)):
            raise ValueError("Compensation torque exceeds its configured/protocol limit")

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_identity(self, leg, joint):
        if (leg, joint) != (self.data["leg"], self.data["joint"]):
            raise ValueError(f"Compensation model belongs to {self.data['leg']} Joint{self.data['joint']}")

    def coverage_intervals(self):
        intervals = [[self.angles[0], self.angles[0]]]
        for previous, angle in zip(self.angles, self.angles[1:]):
            if angle - previous > self.max_interpolation_gap_deg:
                intervals.append([angle, angle])
            else:
                intervals[-1][1] = angle
        return intervals

    def load_at(self, angle_deg):
        if not math.isfinite(angle_deg):
            raise ValueError("Nonfinite joint position in compensation")
        if angle_deg < self.angles[0]:
            return self.loads[0], True
        if angle_deg > self.angles[-1]:
            return self.loads[-1], True
        index = bisect.bisect_left(self.angles, angle_deg)
        if self.angles[index] == angle_deg:
            return self.loads[index], False
        low, high = self.angles[index - 1], self.angles[index]
        if high - low > self.max_interpolation_gap_deg:
            nearest = index - 1 if angle_deg - low <= high - angle_deg else index
            return self.loads[nearest], True  # Never invent coverage across an unmeasured gap.
        alpha = (angle_deg - low) / (high - low)
        return self.loads[index - 1] + alpha * (self.loads[index] - self.loads[index - 1]), False

    def span_coverage(self, low_deg, high_deg):
        if not math.isfinite(low_deg) or not math.isfinite(high_deg) or high_deg <= low_deg:
            raise ValueError("Coverage interval must be finite and increasing")
        covered = sum(max(0., min(high_deg, high) - max(low_deg, low))
                      for low, high in self.coverage_intervals())
        return covered / (high_deg - low_deg)


class LoadFrictionController:
    def __init__(self, model, initial_torque, max_dt_s, friction_scale=1.):
        if not math.isfinite(initial_torque) or abs(initial_torque) > model.feedforward_limit_nm:
            raise ValueError("Warmup feedforward exceeds model's output limit")
        self.model, self.output, self.max_dt_s = model, initial_torque, max_dt_s
        if not math.isfinite(friction_scale) or not 0 <= friction_scale <= 1:
            raise ValueError("friction_scale must be finite and in [0,1]")
        self.friction_scale = friction_scale

    def update(self, position_deg, direction, dt_s):
        if not math.isfinite(direction) or not math.isfinite(dt_s):
            raise ValueError("Nonfinite motion direction or compensation interval")
        direction = max(-1., min(1., direction))
        dt = min(max(0., dt_s), self.max_dt_s)
        load, clamped = self.model.load_at(position_deg)
        friction = self.model.friction_torque_nm * self.friction_scale * direction
        requested = load + friction
        bounded = max(-self.model.feedforward_limit_nm, min(self.model.feedforward_limit_nm, requested))
        filtered = self.output + (1 - math.exp(-dt / self.model.filter_tau_s)) * (bounded - self.output)
        change = self.model.slew_rate_nm_s * dt
        self.output += max(-change, min(change, filtered - self.output))
        return dict(ff_nm=self.output, requested_load_nm=load, requested_friction_nm=friction,
                    requested_ff_nm=requested, load_clamped=int(clamped),
                    ff_saturated=int(requested != bounded), motion_direction=direction, dynamic_active=1)


def fit_ramps(directories, bin_deg=1., min_samples=3):
    """Pool raw moving samples from compatible completed ramps, in both directions."""
    if not math.isfinite(bin_deg) or bin_deg <= 0 or min_samples < 3:
        raise ValueError("bin_deg must be positive and min_samples >=3")
    buckets, sources, identity, gains, contexts = {}, [], None, None, set()
    unrecorded_contexts = 0
    for directory in map(Path, directories):
        meta = json.loads((directory / "run.json").read_text())
        settings = meta["settings"]
        if not meta["complete"] or meta["run_type"] != "hardware" or settings["mode"] != "ramp":
            raise ValueError(f"Only completed hardware ramps can be fitted: {directory}")
        current_identity = (settings["leg"], settings["joint"])
        current_gains = (settings["kp"], settings["kd"])
        if identity is not None and (identity != current_identity or gains != current_gains):
            raise ValueError("Do not mix joints or control gains in one calibration")
        identity, gains = current_identity, current_gains
        posture = meta.get("other_joint_targets_deg")
        if posture is None:
            unrecorded_contexts += 1
        else:
            contexts.add(json.dumps(posture))
        if len(contexts) > 1:
            raise ValueError("Do not merge different other-joint target postures into one 1-D load table")
        with (directory / "commands.csv").open() as stream:
            commands = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        updates, seen = [], set()
        for index, row in enumerate(commands):
            sequence = row.get("target_sequence", index)
            if sequence not in seen:
                seen.add(sequence)
                updates.append(row)
        times = [row["time_s"] for row in updates]
        directions = [0] + [(1 if updates[i]["target_deg"] > updates[i-1]["target_deg"] else
                            -1 if updates[i]["target_deg"] < updates[i-1]["target_deg"] else 0)
                           for i in range(1, len(updates))]
        with (directory / "feedback.csv").open() as stream:
            for raw in csv.DictReader(stream):
                row = {key: float(value) for key, value in raw.items()}
                if row["motor_id"] != identity[1] or row["time_s"] < times[0]:
                    continue
                index = bisect.bisect_right(times, row["time_s"]) - 1
                direction = directions[index]
                if direction and direction * row["velocity_deg_s"] >= .5:
                    bucket = math.floor(row["position_deg"] / bin_deg)
                    buckets.setdefault(bucket, {1: [], -1: []})[direction].append(row["torque_nm"])
        sources.append(str(directory))
    paired = [(bucket, median(values[1]), median(values[-1]), len(values[1]), len(values[-1]))
              for bucket, values in sorted(buckets.items()) if min(len(values[1]), len(values[-1])) >= min_samples]
    if len(paired) < 3:
        raise ValueError("Need at least three angle bins with enough samples in both directions")
    half_gaps = [(up-down)/2 for _, up, down, _, _ in paired]
    if median(half_gaps) <= 0:
        raise ValueError("Directional torque data does not support positive friction compensation")
    data = dict(version=1, leg=identity[0], joint=identity[1],
                load_angle_deg=[(bucket+.5)*bin_deg for bucket, *_ in paired],
                load_torque_nm=[(up+down)/2 for _, up, down, _, _ in paired],
                friction_torque_nm=median(half_gaps), velocity_soft_deg_s=.5,
                filter_tau_s=.05, slew_rate_nm_s=10., feedforward_limit_nm=4.,
                max_interpolation_gap_deg=2*bin_deg,
                calibration=dict(source_runs=sources, bin_deg=bin_deg, kp=gains[0], kd=gains[1],
                                 other_joint_targets_deg=json.loads(next(iter(contexts))) if contexts else None,
                                 runs_with_unrecorded_posture=unrecorded_contexts,
                                 increasing_samples=[row[3] for row in paired], decreasing_samples=[row[4] for row in paired],
                                 context="Conditional on calibration support and other-joint posture; verify before reuse",
                                 status="Experimental local table; no whole-robot or full-range validation"))
    model = LoadFrictionModel(data)
    data["calibration"]["coverage_intervals_deg"] = model.coverage_intervals()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Completed ramp directories; combine to extend coverage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin_deg", type=float, default=1.)
    args = parser.parse_args()
    data = fit_ramps(args.runs, args.bin_deg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), coverage=data["calibration"]["coverage_intervals_deg"],
                          friction_nm=data["friction_torque_nm"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
