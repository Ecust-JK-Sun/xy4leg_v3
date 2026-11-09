"""Explicit experimental reader/controller for offline directional candidates.

No serial access; this does not change the legacy compensation or policy runner.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path


class DirectionalTable:
    def __init__(self, data):
        if (data.get("schema_version") != 1 or
                data.get("model_type") != "directional_static_holding_torque_candidate" or
                data.get("leg") not in ("LB", "LF", "RB", "RF") or data.get("joint") not in range(1, 6)):
            raise ValueError("Unsupported directional candidate identity/schema")
        self.data = data
        self.angles = tuple(float(q) for q in data["angle_deg"])
        self.up = tuple(float(t) for t in data["increasing_torque_nm"])
        self.down = tuple(float(t) for t in data["decreasing_torque_nm"])
        self.limit = float(data["feedforward_limit_nm"])
        if (len(self.angles) < 2 or len(self.angles) != len(self.up) or len(self.up) != len(self.down)
                or not all(math.isfinite(x) for x in self.angles + self.up + self.down)
                or any(b <= a for a, b in zip(self.angles, self.angles[1:]))
                or not math.isfinite(self.limit) or not 0 < self.limit <= 28
                or any(abs(x) > self.limit for x in self.up + self.down)):
            raise ValueError("Invalid angle grid, torque values or feedforward limit")
        for key, expected in (("midpoint_torque_nm", [(u+d)/2 for u,d in zip(self.up,self.down)]),
                              ("signed_half_difference_nm", [(u-d)/2 for u,d in zip(self.up,self.down)])):
            supplied = data[key]
            if len(supplied) != len(expected) or any(not math.isfinite(x) or abs(x-y)>1.e-8 for x,y in zip(supplied,expected)):
                raise ValueError("Candidate midpoint/half-difference do not reconstruct both branches")
        support = data.get("support", [])
        if support and (len(support) != len(self.angles) or any(abs(row["angle_deg"]-q)>1.e-8 for row,q in zip(support,self.angles))):
            raise ValueError("Candidate support grid does not match torque table")
        self.support = tuple(row.get("at_least_three_rounds_each_direction_gap_le_4deg", False) for row in support)

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_identity(self, leg, joint):
        if (self.data["leg"], self.data["joint"]) != (leg, joint):
            raise ValueError("Candidate belongs to a different leg/joint")

    def at(self, angle):
        if not math.isfinite(angle):
            raise ValueError("Nonfinite measured angle")
        q = max(self.angles[0], min(self.angles[-1], angle))
        index = bisect.bisect_left(self.angles, q)
        if self.angles[index] == q:
            up, down = self.up[index], self.down[index]
            supported = self.support[index] if self.support else False
        else:
            weight = (q-self.angles[index-1])/(self.angles[index]-self.angles[index-1])
            up = self.up[index-1]+weight*(self.up[index]-self.up[index-1])
            down = self.down[index-1]+weight*(self.down[index]-self.down[index-1])
            supported = bool(self.support and self.support[index-1] and self.support[index])
        return dict(midpoint_nm=(up+down)/2, half_difference_nm=(up-down)/2,
                    clamped=int(q != angle), low_support=int(not supported))


class DirectionalController:
    def __init__(self, table, variant, initial_torque, filter_tau_s=.05, slew_rate_nm_s=10., max_dt_s=.01):
        if variant not in ("fixed", "midpoint", "directional"):
            raise ValueError("Unknown comparison variant")
        if not math.isfinite(initial_torque) or abs(initial_torque) > table.limit:
            raise ValueError("Initial feedforward exceeds candidate limit")
        if any(not math.isfinite(x) or x <= 0 for x in (filter_tau_s, slew_rate_nm_s, max_dt_s)):
            raise ValueError("Filter, slew rate and time cap must be positive")
        self.table, self.variant = table, variant
        self.output = self.initial = initial_torque
        self.tau, self.slew, self.max_dt = filter_tau_s, slew_rate_nm_s, max_dt_s

    def update(self, angle, direction, dt_s):
        if direction not in (-1, 0, 1) or not math.isfinite(dt_s):
            raise ValueError("Invalid latched direction/time interval")
        components = self.table.at(angle)
        fixed = self.variant == "fixed" or (self.variant == "directional" and not direction)
        load = 0. if fixed else components["midpoint_nm"]
        directional = components["half_difference_nm"]*direction if self.variant == "directional" and not fixed else 0.
        initial = self.initial if fixed else 0.
        requested = initial+load+directional
        bounded = max(-self.table.limit, min(self.table.limit, requested))
        if self.variant == "fixed":
            self.output = self.initial
        else:
            dt = min(max(0., dt_s), self.max_dt)
            filtered = self.output+(1-math.exp(-dt/self.tau))*(bounded-self.output)
            self.output += max(-self.slew*dt, min(self.slew*dt, filtered-self.output))
        return dict(ff_nm=self.output, requested_fixed_nm=initial, requested_midpoint_nm=load, requested_directional_nm=directional,
                    requested_ff_nm=requested, ff_saturated=int(requested != bounded),
                    model_clamped=components["clamped"], model_low_support=components["low_support"],
                    applied_direction=direction)
