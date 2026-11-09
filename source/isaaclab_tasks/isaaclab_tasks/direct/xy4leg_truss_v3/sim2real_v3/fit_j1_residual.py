"""Fit low-dimensional J1 residuals; no simulator, serial, or hardware access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gravity_compensation import DEFAULT_CONFIG, LEGS
from initialization_models import FittedNodeGravityModel
from fit_static_node_masses import load_points, weights


def design(rows, base, linear):
    direction = np.array([r["direction"] for r in rows])
    center = 65. if rows[0]["leg"] in ("LB", "RF") else -65.
    z = (np.rad2deg([r["angles"][0] for r in rows])-center)/30.
    matrix = np.column_stack((base, np.ones(len(rows)), direction))
    return np.column_stack((matrix, direction*z)) if linear else matrix


def fit(rows, base, linear):
    matrix = design(rows, base, linear)
    w = weights(rows)
    y = np.array([r["torque"] for r in rows])
    root = np.sqrt(w)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("Insufficient J1 excitation")
    c = np.linalg.lstsq(matrix*root[:, None], y*root, rcond=None)[0]
    residual = y-matrix@c
    cutoff = 1.5*max(56/4095, 1.4826*np.median(abs(residual-np.median(residual))))
    for _ in range(300):
        root = np.sqrt(w*np.minimum(1., cutoff/np.maximum(abs(y-matrix@c), 1e-12)))
        update = np.linalg.lstsq(matrix*root[:, None], y*root, rcond=None)[0]
        if max(abs(update-c)) < 1e-9:
            return update
        c = update
    raise ValueError("J1 robust fit did not converge")


def mae(rows, prediction):
    return float(np.average(abs(np.array([r["torque"] for r in rows])-prediction), weights=weights(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points_csv", type=Path, required=True)
    parser.add_argument("--saturation_annotations", type=Path)
    parser.add_argument("--node_mass_file", type=Path, default=Path(DEFAULT_CONFIG).parent/"calibration/node_masses_static_20261102.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    model = FittedNodeGravityModel(DEFAULT_CONFIG, None, args.node_mass_file)
    rows, excluded = load_points(args.points_csv, args.saturation_annotations)
    model_identity = model.description()
    candidate = dict(schema_version=1, model_type="j1_pose_direction_residual", base_model=model_identity,
                     gravity_b=[0., 0., -9.81], finger_angle_deg=-30., direction_deadband_deg=.25,
                     source_sha256=hashlib.sha256(args.points_csv.read_bytes()).hexdigest(), legs={},
                     scope="Static-trained experimental residual; dynamic friction/trajectory accuracy unvalidated")
    report = dict(excluded=excluded, legs={})
    for leg in LEGS:
        selected = [r for r in rows if r["leg"] == leg and r["joint"] == 1]
        rounds = sorted({r["round"] for r in selected})
        if len(rounds) < 3:
            raise ValueError("Need at least three complete rounds per leg")
        base = np.array([model.leg_torque(leg, r["angles"])[0] for r in selected])
        trials = []
        for linear in (False, True):
            prediction = np.empty(len(selected))
            folds = []
            for repeat in rounds:
                train = np.array([r["round"] != repeat for r in selected])
                training = [r for r, keep in zip(selected, train) if keep]
                testing = [r for r, keep in zip(selected, ~train) if keep]
                c = fit(training, base[train], linear)
                prediction[~train] = design(testing, base[~train], linear)@c
                folds.append(dict(held_out_round=repeat, coefficients=c.tolist(),
                                  mae_nm=mae(testing, prediction[~train])))
            c = fit(selected, base, linear)
            trials.append(dict(linear_direction_gap=linear, coefficients=c.tolist(), folds=folds,
                               held_out_mae_nm=mae(selected, prediction)))
        best = min(trials, key=lambda v: v["held_out_mae_nm"])
        gain, bias, gap, *slope = best["coefficients"]
        if not 0 < gain < 2:
            raise ValueError("Invalid extrapolating gravity gain")
        q = np.rad2deg([r["angles"][0] for r in selected])
        candidate["legs"][leg] = dict(gain=gain, bias_nm=bias, gap_nm=gap,
                                     gap_slope_nm=slope[0] if slope else 0.,
                                     center_deg=65. if leg in ("LB", "RF") else -65., scale_deg=30.,
                                     angle_range_deg=[float(min(q)), float(max(q))])
        report["legs"][leg] = dict(base_mae_nm=mae(selected, base), candidates=trials,
                                  selected=best, samples=len(selected))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"j1_residual.json").write_text(json.dumps(candidate, indent=2)+"\n")
    (args.output_dir/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# J1姿态与方向残差候选", "",
             "τ1 = gain × τ1_base(q1..q5) + bias + direction × [gap + slope × (q1−center)/30°]。",
             "J2–J5不变，固定拟合节点质量；实测到达方向用于离线拟合。无误差积分。",
             "运行时方向由实测J1角度变化锁存，初次无方向时为0；方向模型不是静态表插值。",
             "", "| 腿 | 基础MAE N·m | 留轮候选MAE N·m | gain | bias N·m | gap N·m | slope N·m |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for leg, p in candidate["legs"].items():
        r = report["legs"][leg]
        lines.append(f"| {leg} | {r['base_mae_nm']:.3f} | {r['selected']['held_out_mae_nm']:.3f} | "
                     f"{p['gain']:.3f} | {p['bias_nm']:.3f} | {p['gap_nm']:.3f} | {p['gap_slope_nm']:.3f} |")
    lines += ["", "留轮指标用于二选一选型，不是独立动态测试；同一静态姿态族的跨轮验证也不能证明跨姿态泛化。",
              "纯重力比例接近1时，剩余误差主要与方向/保持历史有关，不能靠整体放大质量消除。",
              "方向项来自静态到达历史，不宣称已识别出动态Coulomb摩擦。",
              "候选只在显式指定残差文件时启用；残差另限幅，最终总前馈仍受原限幅与变化率约束。",
              "原变化率可能限制快速换向时的跟随，需用动态日志验证，不能据本报告提高变化率。",
              "角度范围外截断方向项角度自变量；基础重力仍使用完整实测姿态。", ""]
    (args.output_dir/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
