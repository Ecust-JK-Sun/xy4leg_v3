"""CPU checks for physical torque basis and constrained mass recovery."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REAL = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gravity = load("gravity_compensation", REAL / "gravity_compensation.py")
fitting = load("node_mass_fit_test", REAL / "fit_static_node_masses.py")


@pytest.mark.parametrize("leg", gravity.LEGS)
def test_basis_matches_existing_model_and_j1_is_unobservable(leg):
    model = gravity.GravityModel()
    rng = np.random.default_rng(24)
    for _ in range(5):
        angles, g = rng.uniform(-2, 2, 5), rng.uniform(-9, 9, 3)
        basis, fixed = fitting.torque_basis(model, leg, angles, g, -.4)
        np.testing.assert_allclose(basis @ fitting.nominal_masses(model) + fixed,
                                   model.leg_torque(leg, angles, g, -.4), atol=1e-12)
        np.testing.assert_array_equal(basis[:, 0], 0)


@pytest.mark.parametrize("mode", ["mass_only", "mass_direction", "mass_offset_direction"])
def test_recovers_known_masses_with_direction_and_offset(mode):
    model = gravity.GravityModel()
    rng = np.random.default_rng(12)
    rows, matrices, fixed = [], [], []
    for repeat in range(1, 4):
        for joint in range(1, 6):
            for direction in (-1, 1):
                for _ in range(10):
                    basis, fingers = fitting.torque_basis(model, "LB", rng.uniform(-2, 2, 5),
                                                         [0, 0, -9.81], -.5)
                    matrices.append(basis[joint-1])
                    fixed.append(fingers[joint-1])
                    rows.append(dict(joint=joint, round=repeat, direction=direction))
    matrices, fixed = np.array(matrices), np.array(fixed)
    nominal = fitting.nominal_masses(model)
    # Deliberately set an arbitrary J1: observations cannot determine it.
    truth = np.array([12., .4, .8, .3, .65, .7])
    nuisance = fitting.nuisance_matrix(rows, mode)
    coefficients = np.linspace(-.2, .3, nuisance.shape[1])
    target = matrices @ truth + fixed + nuisance @ coefficients
    for row, torque in zip(rows, target):
        row["torque"] = torque
    fitted = fitting.fit(rows, matrices, fixed, nominal, mode)
    assert fitted["mass_rank_after_nuisance"] == 5
    assert fitted["node_masses_kg"][0] == nominal[0]
    np.testing.assert_allclose(fitted["node_masses_kg"][1:], truth[1:], atol=1e-8)
    np.testing.assert_allclose(fitting.predict(rows, matrices, fixed, fitted, mode), target, atol=1e-8)


def test_negative_mass_solution_hits_boundary_instead_of_becoming_physical_claim():
    rows = [dict(joint=i+1, round=1, direction=1, torque=t)
            for i, t in enumerate([-1., .5, .6, .7, .8])]
    matrix = np.column_stack((np.zeros(5), np.eye(5)))
    fitted = fitting.fit(rows, matrix, np.zeros(5), np.ones(6), "mass_only")
    assert "J2" in fitted["bound_nodes"]
    assert fitted["node_masses_kg"][1] >= 0
