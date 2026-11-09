"""URDF geometry with lumped motor/rod masses; no hardware or simulator imports.

Each leg has five main joints and a sixth motor at the midpoint of the two
finger hinges (fixed in Link5). Five straight uniform rods join these six
nodes. Each rod's mass is split equally between its endpoints. Finger masses
use the URDF inertial origins as approximate geometric centroids, transformed
by their opening angles. CAD masses/inertias and base mass are not reused.

Only leg self-weight is compensated. Contact loads, friction and acceleration
torques are excluded. All returned torques follow the hardware joint-major
order [LB1, LF1, RB1, RF1, ..., LB5, LF5, RB5, RF5].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[5]
DEFAULT_CONFIG = THIS_DIR / "gravity_compensation.yaml"
LEGS = ("LB", "LF", "RB", "RF")
DEFAULT_POSE_DEG = np.array([
    [65., 45., -20., 24., -24.], [-65., -85., -20., 24., -24.],
    [-65., -85., -20., 24., -24.], [65., 45., -20., 24., -24.],
])


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _xyz(element, field, default="0 0 0"):
    return _vector([float(v) for v in element.get(field, default).split()], 3, field)


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        roll, pitch, yaw = _xyz(element, "rpy")
        transform[:3, :3] = (_rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch)
                            @ _rotation((1, 0, 0), roll))
        transform[:3, 3] = _xyz(element, "xyz")
    return transform


@dataclass
class MassState:
    positions: np.ndarray
    masses: np.ndarray
    affected: np.ndarray
    joint_positions: np.ndarray
    joint_axes: np.ndarray
    motor_positions: np.ndarray


class GravityModel:
    def __init__(self, config_path=DEFAULT_CONFIG, urdf_path=None):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(self.config, dict):
            raise ValueError("Gravity configuration must be a mapping")
        cfg = self.config
        path = Path(urdf_path if urdf_path is not None else cfg["urdf_path"])
        self.urdf_path = (path if path.is_absolute() else REPO_ROOT / path).resolve()
        contents = self.urdf_path.read_bytes()
        self.urdf_sha256 = hashlib.sha256(contents).hexdigest()
        root = ET.fromstring(contents)
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        links = {link.attrib["name"]: link for link in root.findall("link")}
        for name in ("motor_mass_kg", "link_mass_kg", "finger_mass_kg"):
            value = float(cfg[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)
        self.signs = _vector(cfg["joint_signs"], 5, "joint_signs")
        if not np.isin(self.signs, [-1., 1.]).all():
            raise ValueError("joint_signs must be +/-1")
        self.offsets = np.array([np.deg2rad(_vector(cfg["joint_offsets_deg"][leg], 5, leg)) for leg in LEGS])
        self.torque_limits = _vector(cfg["torque_limit_nm"], 5, "torque_limit_nm")
        self.slew_rates = _vector(cfg["slew_rate_nm_s"], 5, "slew_rate_nm_s")
        if (np.any(self.torque_limits <= 0) or np.any(self.torque_limits > 28)
                or np.any(self.slew_rates <= 0)):
            raise ValueError("Positive slew rates and torque limits in (0,28] are required")
        self.chains, self.fingers = {}, {}
        for leg in LEGS:
            parent = "base_link"
            chain = []
            for index in range(1, 6):
                joint = joints[f"{leg}_Joint{index}"]
                if joint.find("parent").attrib["link"] != parent:
                    raise ValueError(f"Unexpected {leg} main-joint topology")
                chain.append(self._joint(joint))
                parent = joint.find("child").attrib["link"]
            fingers = []
            for index in (1, 2):
                joint = joints[f"{leg}_{index}out_Joint"]
                if joint.find("parent").attrib["link"] != parent:
                    raise ValueError(f"Unexpected {leg} finger topology")
                link = links[joint.find("child").attrib["link"]]
                centroid = link.find("inertial/origin")
                if centroid is None:
                    raise ValueError(f"Missing finger centroid: {leg} finger {index}")
                fingers.append((*self._joint(joint), _xyz(centroid, "xyz")))
            self.chains[leg], self.fingers[leg] = chain, fingers

    @staticmethod
    def _joint(joint):
        if joint.attrib["type"] not in ("revolute", "continuous"):
            raise ValueError("Gravity model expects revolute main/finger joints")
        element = joint.find("axis")
        axis = np.array([1., 0., 0.]) if element is None else _xyz(element, "xyz")
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("Joint axis cannot be zero")
        return _origin(joint.find("origin")), axis / norm

    def mass_state(self, leg, hardware_angles, finger_angle=-math.pi / 6, representation="endpoints"):
        """Return points for one leg; `midpoints` is an independent equivalence check."""
        if leg not in LEGS or representation not in ("endpoints", "midpoints"):
            raise ValueError("Unknown leg or mass representation")
        q_hw = _vector(hardware_angles, 5, "hardware_angles")
        if not math.isfinite(finger_angle):
            raise ValueError("finger_angle must be finite")
        q = self.signs * q_hw + self.offsets[LEGS.index(leg)]
        transform = np.eye(4)
        origins, axes = [], []
        for angle, (origin, axis) in zip(q, self.chains[leg]):
            frame = transform @ origin
            origins.append(frame[:3, 3].copy())
            axes.append(frame[:3, :3] @ axis)
            motion = np.eye(4)
            motion[:3, :3] = _rotation(axis, angle)
            transform = frame @ motion
        hinges = [transform @ finger[0] for finger in self.fingers[leg]]
        j6 = (hinges[0][:3, 3] + hinges[1][:3, 3]) / 2
        nodes = np.vstack((origins, j6))
        positions, masses, affected = [], [], []

        def add(position, mass, last_joint):
            positions.append(position)
            masses.append(mass)
            affected.append(np.arange(5) <= min(last_joint, 4))

        if representation == "endpoints":
            node_masses = np.full(6, self.motor_mass_kg + self.link_mass_kg)
            node_masses[[0, -1]] -= self.link_mass_kg / 2
            for index, (point, mass) in enumerate(zip(nodes, node_masses)):
                add(point, mass, index)
        else:
            for index, point in enumerate(nodes):
                add(point, self.motor_mass_kg, index)
            for index in range(5):
                add((nodes[index] + nodes[index + 1]) / 2, self.link_mass_kg, index)
        for frame, (_, axis, centroid) in zip(hinges, self.fingers[leg]):
            point = frame[:3, 3] + frame[:3, :3] @ _rotation(axis, finger_angle) @ centroid
            add(point, self.finger_mass_kg, 4)
        return MassState(np.asarray(positions), np.asarray(masses), np.asarray(affected),
                         np.asarray(origins), np.asarray(axes), nodes)

    def leg_torque(self, leg, hardware_angles, gravity_b=(0., 0., -9.81),
                   finger_angle=-math.pi / 6, representation="endpoints"):
        """Unbounded hardware torque that cancels gravity, in N*m; base is held fixed."""
        gravity = _vector(gravity_b, 3, "gravity_b")
        state = self.mass_state(leg, hardware_angles, finger_angle, representation)
        levers = state.positions[:, None, :] - state.joint_positions[None, :, :]
        forces = state.masses[:, None] * gravity
        moments = np.cross(levers, forces[:, None, :])
        generalized = np.einsum("nij,ij->ni", moments, state.joint_axes)
        return -np.sum(generalized * state.affected, axis=0) * self.signs

    def potential_energy(self, leg, hardware_angles, gravity_b=(0., 0., -9.81),
                         finger_angle=-math.pi / 6, representation="endpoints"):
        state = self.mass_state(leg, hardware_angles, finger_angle, representation)
        gravity = _vector(gravity_b, 3, "gravity_b")
        return -float(np.sum(state.masses * (state.positions @ gravity)))

    def torque(self, hardware_angles, gravity_b=(0., 0., -9.81), finger_angles=None):
        q = _vector(hardware_angles, 20, "hardware_angles")
        fingers = np.full(4, -math.pi / 6) if finger_angles is None else _vector(finger_angles, 4, "finger_angles")
        result = np.empty(20)
        for index, leg in enumerate(LEGS):
            result[index::4] = self.leg_torque(leg, q[index::4], gravity_b, fingers[index])
        return result

    def description(self):
        return dict(model_version=1, urdf_path=str(self.urdf_path), urdf_sha256=self.urdf_sha256,
                    motor_mass_kg=self.motor_mass_kg, link_mass_kg=self.link_mass_kg,
                    finger_mass_kg=self.finger_mass_kg,
                    mass_per_leg_kg=6*self.motor_mass_kg + 5*self.link_mass_kg + 2*self.finger_mass_kg,
                    j6_location="midpoint of the two finger hinge origins, fixed in Link5",
                    finger_centroids="URDF inertial origins, approximate geometric centroids",
                    joint_signs=self.signs.tolist(), joint_offsets_deg=np.rad2deg(self.offsets).tolist(),
                    torque_limit_nm=self.torque_limits.tolist(), slew_rate_nm_s=self.slew_rates.tolist(),
                    scope="Fixed-base leg self-weight only; no contact load, friction or dynamic compensation")


def compare_static_points(model, path, gravity_b=(0., 0., -9.81), finger_angle=-math.pi / 6):
    """Compare raw predictions with recorded q1..q5 and selected-joint holding torque.

    No parameters are fitted. Base orientation and finger opening are assumed,
    since the existing static-points CSV does not record them.
    """
    path = Path(path).resolve()
    groups = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            leg, joint = row["leg"], int(row["joint"])
            if leg not in LEGS or not 1 <= joint <= 5:
                raise ValueError("Static CSV contains an invalid leg/joint")
            q = np.deg2rad([float(row[f"q{index}_deg"]) for index in range(1, 6)])
            measured = float(row["torque_nm"])
            if not math.isfinite(measured):
                raise ValueError("Static CSV contains nonfinite measured torque")
            predicted = model.leg_torque(leg, q, gravity_b, finger_angle)[joint-1]
            groups.setdefault((leg, joint), []).append(predicted-measured)
    if not groups:
        raise ValueError("Static CSV contains no points")
    results = []
    for (leg, joint), values in sorted(groups.items()):
        error = np.asarray(values)
        results.append(dict(leg=leg, joint=joint, samples=len(values),
                            mae_nm=float(np.abs(error).mean()), bias_nm=float(error.mean()),
                            rmse_nm=float(np.sqrt(np.mean(error**2))),
                            abs_error_p95_nm=float(np.percentile(np.abs(error), 95))))
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                samples=sum(len(values) for values in groups.values()), groups=results,
                interpretation="Raw torque prediction errors, not compensated position accuracy; "
                               "includes baseline points. No fitting. Gravity direction and finger opening are assumed.")


class GravityFeedforward:
    """Scaled, bounded feedforward with a zero-start and a capped slew interval."""

    def __init__(self, model, scale=1., max_dt_s=.04):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("Gravity scale must be finite and in [0,1]")
        if not math.isfinite(max_dt_s) or max_dt_s <= 0:
            raise ValueError("Gravity max_dt_s must be finite and positive")
        self.model, self.scale, self.max_dt_s = model, scale, max_dt_s
        self.output = np.zeros(20)
        self.raw = np.zeros(20)
        self.saturated = np.zeros(20, dtype=bool)
        self.last_time = None

    def update(self, hardware_angles, gravity_b, finger_angles=None, now=None):
        raw = self.scale * self.model.torque(hardware_angles, gravity_b, finger_angles)
        return self._apply_limits(raw, now)

    def update_leg(self, leg, hardware_angles, gravity_b=(0., 0., -9.81),
                   finger_angle=-math.pi / 6, now=None):
        """Same model/limits for a single-leg initialization; create one controller per leg."""
        index = LEGS.index(leg)
        raw = np.zeros(20)
        raw[index::4] = self.scale * self.model.leg_torque(leg, hardware_angles, gravity_b, finger_angle)
        return self._apply_limits(raw, now)[index::4]

    def _apply_limits(self, raw, now):
        if now is None:
            import time
            now = time.monotonic()
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError("Gravity update time must be finite and monotonic")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite calculated gravity torque")
        limit = np.repeat(self.model.torque_limits, 4)
        bounded = np.clip(raw, -limit, limit)
        dt = 0. if self.last_time is None else min(now-self.last_time, self.max_dt_s)
        step = np.repeat(self.model.slew_rates, 4) * dt
        self.output += np.clip(bounded-self.output, -step, step)
        self.raw, self.saturated, self.last_time = raw, raw != bounded, now
        return self.output.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--leg", choices=LEGS, default="LB")
    parser.add_argument("--angles_deg", type=float, nargs=5, help="Hardware J1-5 angles; defaults to V3 initial pose")
    parser.add_argument("--finger_angle_deg", type=float, default=-30.)
    parser.add_argument("--gravity_b", type=float, nargs=3, default=[0., 0., -9.81])
    parser.add_argument("--static_points_csv", type=Path,
                        help="Also compare against the existing points_all_legs.csv schema without fitting")
    args = parser.parse_args()
    model = GravityModel(args.config, args.urdf)
    angles = (DEFAULT_POSE_DEG[LEGS.index(args.leg)] if args.angles_deg is None
              else _vector(args.angles_deg, 5, "angles_deg"))
    q, finger = np.deg2rad(angles), math.radians(args.finger_angle_deg)
    raw = model.leg_torque(args.leg, q, args.gravity_b, finger)
    mid = model.leg_torque(args.leg, q, args.gravity_b, finger, "midpoints")
    report = dict(model=model.description(), leg=args.leg, hardware_angles_deg=angles.tolist(),
                  gravity_b_m_s2=list(args.gravity_b), finger_angle_deg=args.finger_angle_deg,
                  raw_feedforward_nm=raw.tolist(),
                  bounded_feedforward_nm=np.clip(raw, -model.torque_limits, model.torque_limits).tolist(),
                  midpoint_equivalence_max_error_nm=float(np.max(np.abs(raw-mid))),
                  mode="OFFLINE preview; raw/bounded values exclude runtime slew limiting")
    if args.static_points_csv is not None:
        report["static_comparison"] = compare_static_points(model, args.static_points_csv, args.gravity_b, finger)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
