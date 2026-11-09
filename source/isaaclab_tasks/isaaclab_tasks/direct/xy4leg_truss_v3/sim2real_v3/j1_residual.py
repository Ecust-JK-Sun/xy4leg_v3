"""Explicit J1 pose/direction residual wrapper; never integrates tracking error."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np


class J1ResidualModel:
    def __init__(self, base, path, scale=1., limit_nm=1.5):
        if not math.isfinite(scale) or not 0 <= scale <= 1 or not math.isfinite(limit_nm) or not 0 < limit_nm <= 28:
            raise ValueError("Residual scale must be in [0,1], limit in (0,28]")
        self.base, self.scale, self.limit_nm = base, scale, limit_nm
        self.path = Path(path).resolve()
        contents = self.path.read_bytes()
        self.sha256 = hashlib.sha256(contents).hexdigest()
        self.data = json.loads(contents)
        if self.data.get("schema_version") != 1 or self.data.get("model_type") != "j1_pose_direction_residual":
            raise ValueError("Unsupported J1 residual schema")
        identity = base.description()
        for key in ("urdf_sha256", "joint_signs", "joint_offsets_deg", "finger_mass_kg", "node_mass_sha256"):
            if identity.get(key) != self.data["base_model"].get(key):
                raise ValueError(f"J1 residual base model differs: {key}")
        self.deadband = float(self.data["direction_deadband_deg"])
        if not math.isfinite(self.deadband) or self.deadband <= 0:
            raise ValueError("Direction deadband must be positive")
        for leg in ("LB", "LF", "RB", "RF"):
            p = self.data["legs"][leg]
            values = [p[k] for k in ("gain", "bias_nm", "gap_nm", "gap_slope_nm", "center_deg", "scale_deg")]
            bounds = p["angle_range_deg"]
            if (not np.isfinite(values).all() or not 0 < p["gain"] < 2 or p["scale_deg"] <= 0
                    or len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
                raise ValueError(f"Invalid J1 residual parameters: {leg}")
        self.torque_limits, self.slew_rates = base.torque_limits, base.slew_rates
        self.anchors, self.directions, self.last = {}, {}, {}

    def leg_torque(self, leg, hardware_angles, gravity_b=(0., 0., -9.81), finger_angle=-math.pi/6):
        result = self.base.leg_torque(leg, hardware_angles, gravity_b, finger_angle).copy()
        q = math.degrees(hardware_angles[0])
        anchor = self.anchors.setdefault(leg, q)
        delta = q-anchor
        if abs(delta) >= self.deadband:
            self.directions[leg] = 1 if delta > 0 else -1
            self.anchors[leg] = q
        direction = self.directions.get(leg, 0)
        p = self.data["legs"][leg]
        clamped = float(np.clip(q, *p["angle_range_deg"]))
        common = (p["gain"]-1)*result[0]+p["bias_nm"]
        gap = p["gap_nm"]+p["gap_slope_nm"]*(clamped-p["center_deg"])/p["scale_deg"]
        requested = common+direction*gap
        correction = self.scale*float(np.clip(requested, -self.limit_nm, self.limit_nm))
        result[0] += correction
        self.last[leg] = dict(j1_residual_nm=correction, j1_residual_requested_nm=float(requested),
                              j1_residual_direction=direction, j1_residual_angle_clamped=clamped != q,
                              j1_residual_limited=bool(abs(requested) > self.limit_nm))
        return result

    def description(self):
        description = self.base.description()
        description["j1_residual"] = dict(path=str(self.path), sha256=self.sha256, scale=self.scale,
                                          limit_nm=self.limit_nm, parameters=self.data["legs"],
                                          direction="measured displacement latch; retained at rest; no error integration",
                                          scope=self.data["scope"])
        return description

    def torque(self, hardware_angles, gravity_b=(0., 0., -9.81), finger_angles=None):
        q = np.asarray(hardware_angles, dtype=float)
        fingers = np.full(4, -math.pi/6) if finger_angles is None else np.asarray(finger_angles, dtype=float)
        if q.shape != (20,) or fingers.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(fingers).all():
            raise ValueError("Expected 20 finite joint-major angles and four finger angles")
        result = np.empty(20)
        for index, leg in enumerate(("LB", "LF", "RB", "RF")):
            result[index::4] = self.leg_torque(leg, q[index::4], gravity_b, fingers[index])
        return result

    def leg_diagnostics(self, leg):
        return self.last.get(leg, {})
