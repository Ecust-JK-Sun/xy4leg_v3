"""Offline nonnegative node-mass identification from full-pose static points.

Only the scanned axis is used: other axes' arrival directions are not recorded.
J1 is fixed to its nominal mass because its own origin produces zero moment.
Finger masses, gravity, geometry, encoder signs and zero offsets remain fixed.
No hardware imports, runtime configuration changes, or hidden regularization.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from gravity_compensation import DEFAULT_CONFIG, LEGS, GravityModel


def torque_basis(model, leg, angles, gravity_b, finger_angle):
    """Return hardware torque per kg (5 x 6), and fixed finger torque (5)."""
    state = model.mass_state(leg, angles, finger_angle)
    moments = np.cross(state.positions[:, None, :] - state.joint_positions[None, :, :],
                       np.asarray(gravity_b)[None, None, :])
    basis = -np.einsum("nij,ij->ni", moments, state.joint_axes) * state.affected * model.signs
    return basis[:6].T, state.masses[6:] @ basis[6:]


def nominal_masses(model):
    result = np.full(6, model.motor_mass_kg + model.link_mass_kg)
    result[[0, -1]] -= model.link_mass_kg / 2
    return result


def load_points(path, saturation_path=None):
    events = set()
    if saturation_path:
        with Path(saturation_path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["phase"] == "settle":
                    events.add((row["source_id"], int(row["visit"]), int(row["motor"])))
    accepted, excluded, seen = [], [], set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            identifier = raw["global_point_id"]
            if identifier in seen:
                raise ValueError(f"Duplicate point: {identifier}")
            seen.add(identifier)
            leg, joint = raw["leg"], int(raw["joint"])
            if leg not in LEGS or not 1 <= joint <= 5:
                raise ValueError("Invalid leg/joint")
            direction = int(raw["observed_direction"])
            if direction not in (-1, 0, 1):
                raise ValueError("Invalid observed direction")
            angles = np.deg2rad([float(raw[f"q{i}_deg"]) for i in range(1, 6)])
            torque = float(raw["torque_nm"])
            if not np.isfinite(angles).all() or not np.isfinite(torque):
                raise ValueError(f"Nonfinite point: {identifier}")
            reason = None
            if int(raw["direction"]) == 0 or direction == 0:
                reason = "baseline_or_unknown_direction"
            elif (raw["source_id"], int(raw["visit"]), joint) in events:
                reason = "selected_axis_settle_integral_saturation"
            if reason:
                excluded.append(dict(point_id=identifier, reason=reason))
                continue
            accepted.append(dict(point_id=identifier, leg=leg, joint=joint,
                                 round=int(raw["round"]), direction=direction,
                                 angles=angles, torque=torque))
    if not accepted:
        raise ValueError("No usable static points")
    return accepted, excluded


def nuisance_matrix(rows, mode):
    indicator = np.eye(5)[[r["joint"] - 1 for r in rows]]
    direction = indicator * np.array([r["direction"] for r in rows])[:, None]
    if mode == "mass_only":
        return np.empty((len(rows), 0))
    if mode == "mass_direction":
        return direction
    if mode == "mass_offset_direction":
        return np.column_stack((indicator, direction))
    raise ValueError(f"Unknown fit mode: {mode}")


def weights(rows):
    counts = collections.Counter((r["joint"], r["round"], r["direction"]) for r in rows)
    w = np.array([1 / counts[r["joint"], r["round"], r["direction"]] for r in rows])
    return w / np.mean(w)


def fit(rows, basis, fixed, nominal, mode):
    # Exclude exactly unobservable J1, never report its prior as a measurement.
    nuisance = nuisance_matrix(rows, mode)
    matrix = np.column_stack((basis[:, 1:], nuisance))
    root = np.sqrt(weights(rows))
    target = np.array([r["torque"] for r in rows]) - fixed - basis[:, 0] * nominal[0]
    lower = np.r_[np.zeros(5), np.full(nuisance.shape[1], -np.inf)]
    result = lsq_linear(matrix * root[:, None], target * root, bounds=(lower, np.inf),
                        tol=1e-12, max_iter=1000)
    if not result.success:
        raise ValueError(f"Mass fit failed: {result.message}")
    # Project nuisance terms away before evaluating mass identifiability.
    mass_matrix = basis[:, 1:] * root[:, None]
    if nuisance.shape[1]:
        n = nuisance * root[:, None]
        mass_matrix -= n @ np.linalg.lstsq(n, mass_matrix, rcond=None)[0]
    singular = np.linalg.svd(mass_matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(mass_matrix))
    return dict(node_masses_kg=np.r_[nominal[0], result.x[:5]].tolist(),
                nuisance_nm=result.x[5:].tolist(),
                mass_rank_after_nuisance=rank, mass_singular_values=singular.tolist(),
                mass_condition_number=float(singular[0] / singular[-1]) if rank == 5 else None,
                bound_nodes=[f"J{i+2}" for i, v in enumerate(result.x[:5]) if v < 1e-7],
                optimality=float(result.optimality))


def predict(rows, basis, fixed, fitted, mode, include_nuisance=True):
    value = basis @ fitted["node_masses_kg"] + fixed
    if include_nuisance:
        value += nuisance_matrix(rows, mode) @ fitted["nuisance_nm"]
    return value


def metrics(rows, predictions):
    error = np.asarray(predictions) - [r["torque"] for r in rows]
    return dict(samples=len(rows), mae_nm=float(np.mean(np.abs(error))),
                rmse_nm=float(np.sqrt(np.mean(error ** 2))), bias_nm=float(np.mean(error)),
                balanced_mae_nm=float(np.average(np.abs(error), weights=weights(rows))))


def run(args):
    model = GravityModel(args.config, args.urdf)
    rows, excluded = load_points(args.points_csv, args.saturation_annotations)
    nominal = nominal_masses(model)
    report = dict(model=model.description(), assumptions=dict(
        gravity_b=args.gravity_b, finger_angle_deg=args.finger_angle_deg,
        measured_torque="scanned-axis feedback torque, not integral command",
        weighting="equal joint/round/observed-direction groups within each leg",
        loss="nonnegative weighted least squares; no mass regularization or upper bound",
        primary_mode="mass_direction",
        J1="unobservable: fixed to nominal, excluded from optimization",
        fingers="fixed nominal masses, excluded from node estimates",
        validation="leave one complete round out; all its joint scans held out together",
        sensitivity="mass_offset_direction adds free per-axis offsets; inspect mass stability",
        limitations="assumed gravity and opening; static friction/history and geometry errors remain; "
                    "node values are effective gravity parameters, not measured component masses"),
        inputs={}, excluded=excluded, legs={})
    for key, path in (("points_csv", args.points_csv), ("config", args.config),
                      ("saturation_annotations", args.saturation_annotations)):
        if path:
            p = Path(path).resolve()
            report["inputs"][key] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    predictions = []
    for leg in LEGS:
        selected = [r for r in rows if r["leg"] == leg]
        if not selected:
            continue
        rounds = sorted({r["round"] for r in selected})
        if len(rounds) < 2 or any({r["joint"] for r in selected if r["round"] == repeat} != set(range(1, 6))
                                  for repeat in rounds):
            raise ValueError(f"{leg}: need at least two complete five-axis rounds")
        bases, fingers = zip(*(torque_basis(model, leg, r["angles"], args.gravity_b,
                                            np.deg2rad(args.finger_angle_deg)) for r in selected))
        basis = np.array([b[r["joint"] - 1] for r, b in zip(selected, bases)])
        fixed = np.array([f[r["joint"] - 1] for r, f in zip(selected, fingers)])
        baseline = basis @ nominal + fixed
        entry = dict(samples=len(selected), rounds=rounds, nominal_node_masses_kg=nominal.tolist(),
                     nominal_metrics=metrics(selected, baseline), fits={})
        for mode in ("mass_only", "mass_direction", "mass_offset_direction"):
            fitted = fit(selected, basis, fixed, nominal, mode)
            prediction = predict(selected, basis, fixed, fitted, mode)
            gravity_only = predict(selected, basis, fixed, fitted, mode, False)
            cv, folds = np.empty(len(selected)), []
            for repeat in rounds:
                train = np.array([r["round"] != repeat for r in selected])
                test = ~train
                training = [r for r, keep in zip(selected, train) if keep]
                testing = [r for r, keep in zip(selected, test) if keep]
                candidate = fit(training, basis[train], fixed[train], nominal, mode)
                cv[test] = predict(testing, basis[test], fixed[test], candidate, mode)
                folds.append(dict(held_out_round=repeat, fit=candidate,
                                  metrics=metrics(testing, cv[test])))
            fitted.update(train_metrics=metrics(selected, prediction),
                          gravity_only_metrics=metrics(selected, gravity_only),
                          held_out_metrics=metrics(selected, cv), folds=folds,
                          joint_metrics=[dict(joint=j, nominal=metrics(
                              [r for r in selected if r["joint"] == j], baseline[[r["joint"] == j for r in selected]]),
                              held_out=metrics([r for r in selected if r["joint"] == j],
                                               cv[[r["joint"] == j for r in selected]])) for j in range(1, 6)])
            entry["fits"][mode] = fitted
            for i, row in enumerate(selected):
                predictions.append(dict(point_id=row["point_id"], leg=leg, joint=row["joint"],
                                        round=row["round"], direction=row["direction"], mode=mode,
                                        measured_nm=row["torque"], nominal_nm=baseline[i],
                                        fitted_gravity_nm=gravity_only[i], fitted_total_nm=prediction[i],
                                        held_out_total_nm=cv[i]))
        report["legs"][leg] = entry
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    mass_rows = []
    for leg, entry in report["legs"].items():
        for mode, fitted in entry["fits"].items():
            fold_masses = np.array([fold["fit"]["node_masses_kg"] for fold in fitted["folds"]])
            for index, mass in enumerate(fitted["node_masses_kg"]):
                mass_rows.append(dict(leg=leg, mode=mode, node=f"J{index+1}", mass_kg=mass,
                                      status="fixed_unobservable" if index == 0 else
                                      "nonnegative_boundary" if mass < 1e-7 else "fitted",
                                      leave_round_min_kg=float(fold_masses[:, index].min()),
                                      leave_round_max_kg=float(fold_masses[:, index].max())))
    with (output / "node_masses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mass_rows[0]))
        writer.writeheader()
        writer.writerows(mass_rows)
    lines = ["# 静态关节节点等效质量拟合", "",
             f"纳入 {len(rows)} 点，排除 {len(excluded)} 点；仅使用被测轴反馈力矩和五轴实测姿态。",
             "主模型：τ = A(q) m + 固定夹爪重力 + 实测到达方向 × 每轴方向项。",
             "J1 位于自身转轴且不作用于下游轴，其质量不可辨识；下表 J1 固定为名义值。",
             "质量非负、无上界、无质量先验正则；每腿独立拟合。", "",
             "| 腿 | 模型 | J1固定 kg | J2 kg | J3 kg | J4 kg | J5 kg | J6 kg | 名义 MAE | 留轮 MAE N·m | 条件数 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for leg, entry in report["legs"].items():
        for mode, f in entry["fits"].items():
            masses = " | ".join(f"{v:.4f}" for v in f["node_masses_kg"])
            condition = f["mass_condition_number"]
            condition_text = f"{condition:.1f}" if condition is not None else "秩不足"
            lines.append(f"| {leg} | {mode} | {masses} | {entry['nominal_metrics']['mae_nm']:.4f} | "
                         f"{f['held_out_metrics']['mae_nm']:.4f} | {condition_text} |")
    lines += ["", "## 诊断结论", ""]
    for leg, entry in report["legs"].items():
        primary = entry["fits"]["mass_direction"]
        sensitivity = entry["fits"]["mass_offset_direction"]
        shift = np.max(np.abs(np.array(primary["node_masses_kg"])[1:]
                              - np.array(sensitivity["node_masses_kg"])[1:]))
        bounds = ", ".join(primary["bound_nodes"]) or "无"
        lines.append(f"- {leg}：主模型零质量边界 {bounds}；加入自由零偏后，单节点质量最大变化 {shift:.3f} kg。")
    lines += ["", "J1 无法由这批数据识别；J2–J6 在固定几何/重力假设下矩阵满秩，",
              "但边界解与零偏敏感性说明满秩不等于获得可信的实物质量。",
              "建议将结果保留为离线候选，不用本次零边界值替换真实电机质量。",
              "", "mass_only 为直接质量拟合对照；mass_direction 为主候选；",
              "mass_offset_direction 用于检查自由零偏对质量的影响，不据此宣称识别出真实质量。",
              "留轮 MAE 含对应模型的方向/零偏项，不等于纯重力误差或补偿后的角度精度。",
              "fit.json 另存纯重力误差、逐轴误差、奇异值、零质量边界和每个留出轮次的质量。",
              "若质量落到零边界、跨轮漂移或加入零偏后大幅改变，应视作模型不匹配或辨识不稳。",
              "", "假定机体水平、夹爪开度由命令指定（默认 −30°）；CSV 未记录这两者。",
              "夹爪两活动件各保留配置中的固定质量；节点质量包含杆端等效质量，不是单个电机称重。",
              "结果只适用于已有静态姿态范围；未验证接触承重和动态轨迹。",
              "本工具不更改运行时配置，不连接电机。所有输入哈希及排除点见 fit.json。", ""]
    (output / "FIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points_csv", type=Path, required=True)
    parser.add_argument("--saturation_annotations", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--gravity_b", type=float, nargs=3, default=[0., 0., -9.81])
    parser.add_argument("--finger_angle_deg", type=float, default=-30.)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.gravity_b).all() or np.linalg.norm(args.gravity_b) < 1e-9:
        parser.error("gravity must be finite and nonzero")
    run(args)


if __name__ == "__main__":
    main()
