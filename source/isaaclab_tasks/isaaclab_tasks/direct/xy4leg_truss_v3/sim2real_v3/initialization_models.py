"""Offline-loadable feedforward models for explicit initialization comparisons."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gravity_compensation import DEFAULT_CONFIG, GravityModel, LEGS, REPO_ROOT


DEFAULT_COMPENSATION = "fitted_j1"
DEFAULT_NODE_MASS_FILE = Path(__file__).resolve().parent / "calibration/node_masses_static_20261102.json"
DEFAULT_J1_RESIDUAL_FILE = Path(__file__).resolve().parent / "calibration/j1_residual_20261103.json"


def make_compensation_model(mode=DEFAULT_COMPENSATION, config_path=DEFAULT_CONFIG, urdf_path=None,
                            node_mass_file=None, j1_residual_file=None, residual_scale=1., residual_limit_nm=1.5):
    """Load the same frozen model for initialization, policy play and replay.

    Missing or incompatible calibration files raise before any motor is opened.
    Explicit fitted mode without a residual remains available for comparisons.
    """
    if mode not in ("none", "nominal", "fitted", "fitted_j1"):
        raise ValueError(f"Unsupported compensation mode: {mode}")
    if j1_residual_file is not None and mode not in ("fitted", "fitted_j1"):
        raise ValueError("J1 residual requires fitted or fitted_j1 compensation")
    if mode == "none":
        return None
    if mode == "nominal":
        return GravityModel(config_path, urdf_path)
    model = FittedNodeGravityModel(config_path, urdf_path, node_mass_file or DEFAULT_NODE_MASS_FILE)
    if mode == "fitted_j1" or j1_residual_file is not None:
        from j1_residual import J1ResidualModel

        model = J1ResidualModel(model, j1_residual_file or DEFAULT_J1_RESIDUAL_FILE,
                                residual_scale, residual_limit_nm)
    return model


class FittedNodeGravityModel(GravityModel):
    """Same geometry/fingers/limits as nominal; replace only six node masses."""

    def __init__(self, config_path, urdf_path, mass_file):
        super().__init__(config_path, urdf_path)
        self.mass_file = Path(mass_file).resolve()
        payload = self.mass_file.read_bytes()
        self.mass_sha256 = hashlib.sha256(payload).hexdigest()
        self.fit = json.loads(payload)
        if self.fit.get("schema_version") != 1 or self.fit.get("model_type") != "fitted_node_gravity":
            raise ValueError("Unsupported fitted node mass file")
        reference = self.fit["geometry"]
        if (reference["urdf_sha256"] != self.urdf_sha256
                or not np.array_equal(reference["joint_signs"], self.signs)
                or not np.allclose(reference["joint_offsets_deg"], np.rad2deg(self.offsets), atol=1e-10, rtol=0)
                or reference["finger_mass_kg"] != self.finger_mass_kg):
            raise ValueError("Fitted masses require the calibration URDF, encoder mapping and finger masses")
        self.node_masses = {}
        for leg in LEGS:
            values = np.asarray(self.fit["node_masses_kg"][leg], dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{leg}: need six finite nonnegative node masses")
            self.node_masses[leg] = values

    def mass_state(self, leg, hardware_angles, finger_angle=-math.pi/6, representation="endpoints"):
        if representation != "endpoints":
            raise ValueError("Fitted nodes have no identified motor/rod midpoint decomposition")
        state = super().mass_state(leg, hardware_angles, finger_angle, representation)
        state.masses[:6] = self.node_masses[leg]
        return state

    def description(self):
        description = super().description()
        # Nominal component masses are not the active fitted mass distribution.
        for key in ("motor_mass_kg", "link_mass_kg", "mass_per_leg_kg"):
            description.pop(key)
        description.update(model_type="fitted_node_gravity", node_mass_file=str(self.mass_file),
                           node_mass_sha256=self.mass_sha256,
                           node_masses_kg={leg: values.tolist() for leg, values in self.node_masses.items()},
                           mass_per_leg_kg={leg: float(values.sum()+2*self.finger_mass_kg)
                                            for leg, values in self.node_masses.items()},
                           fit_source=self.fit["source"],
                           nuisance_terms_applied=False)
        return description


class StaticTableModel(GravityModel):
    """Frozen directional static tables, with per-leg measured-motion latches.

    Reuses the same bounded/slewed feedforward path, not the mass calculation.
    Table midpoint is used until a measured motion direction is available.
    """

    def __init__(self, config_path, urdf_path, catalog_path, legs):
        super().__init__(config_path, urdf_path)
        from directional_compensation import DirectionalTable

        self.catalog_path = Path(catalog_path).resolve()
        contents = self.catalog_path.read_bytes()
        self.catalog_sha256 = hashlib.sha256(contents).hexdigest()
        catalog = json.loads(contents)
        if catalog.get("version") != 1:
            raise ValueError("Unsupported static table catalog")
        self.tables, self.anchors, self.directions, self.clamped = {}, {}, {}, {}
        for leg in legs:
            tables = []
            for joint in range(1, 6):
                matches = [p for p in catalog["profiles"] if (p["leg"], p["joint"]) == (leg, joint)]
                if len(matches) != 1:
                    raise ValueError(f"Need exactly one static table for {leg} J{joint}")
                entry = matches[0]
                path = Path(entry["path"])
                path = path if path.is_absolute() else REPO_ROOT / path
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Static table differs from frozen catalog: {path}")
                table = DirectionalTable(json.loads(payload))
                table.validate_identity(leg, joint)
                tables.append(table)
            self.tables[leg] = tables
            self.directions[leg] = np.zeros(5, dtype=int)
            self.clamped[leg] = []

    def leg_torque(self, leg, hardware_angles, gravity_b=(0., 0., -9.81),
                   finger_angle=-math.pi/6, representation="endpoints"):
        q = np.rad2deg(np.asarray(hardware_angles, dtype=float))
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("Static table requires five finite measured angles")
        anchors = self.anchors.setdefault(leg, q.copy())
        delta = q-anchors
        moving = np.abs(delta) >= .25
        self.directions[leg][moving] = np.sign(delta[moving]).astype(int)
        anchors[moving] = q[moving]
        components = [table.at(angle) for table, angle in zip(self.tables[leg], q)]
        self.clamped[leg] = [i+1 for i, c in enumerate(components) if c["clamped"]]
        return np.array([max(-table.limit, min(table.limit, c["midpoint_nm"]+d*c["half_difference_nm"]))
                         for table, c, d in zip(self.tables[leg], components, self.directions[leg])])

    def description(self):
        return dict(model_type="directional_static_table", catalog=str(self.catalog_path),
                    catalog_sha256=self.catalog_sha256, legs=list(self.tables),
                    direction="measured angle latch, 0.25 deg threshold; midpoint before first motion",
                    torque_limit_nm=self.torque_limits.tolist(), slew_rate_nm_s=self.slew_rates.tolist(),
                    scope="Tables conditional on calibration posture; no gravity mass calculation")

    def leg_diagnostics(self, leg):
        return dict(table_direction=self.directions[leg].tolist(), table_clamped_motors=self.clamped[leg])
