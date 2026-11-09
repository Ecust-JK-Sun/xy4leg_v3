"""Fit offline, direction-dependent static holding-torque candidates.

Fits actual encoder angles and observed arrival directions from complete rounds.
No serial imports or hardware access. Existing runtime compensation is not changed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODES = ("angle_only", "constant_direction_gap", "directional_curves")
DEGREES = (1, 2, 3)
TORQUE_QUANTUM_NM = 56 / 4095


def load_rows(merged_directory, saturation_annotations=None):
    root = Path(merged_directory).resolve()
    manifest = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("A completed merged batch is required")
    expected_leg = manifest["procedure"]["leg"]
    if expected_leg not in ("LB", "LF", "RB", "RF"):
        raise ValueError("Unknown leg in merged batch")
    events = collections.defaultdict(list)
    if saturation_annotations:
        with Path(saturation_annotations).open() as stream:
            for event in csv.DictReader(stream):
                if event["phase"] == "settle":
                    events[Path(event["source_directory"]).resolve(), int(event["visit"]), int(event["motor"])].append(event)
    rows, settings, sources, seen = [], {}, [], set()
    for entry in manifest["runs"]:
        directory = (root / entry["directory"]).resolve()
        if directory in seen:
            raise ValueError(f"Duplicate source scan: {directory}")
        seen.add(directory)
        path = directory / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        if (not run.get("complete") or run["return_to_initial"]["status"] != "complete" or
                len(run["points"]) != len(run["procedure"]["plan"]) or
                not all(point.get("valid") for point in run["points"])):
            raise ValueError(f"Incomplete scan or full-leg return: {directory}")
        joint, repeat = entry["joint"], entry["round"]
        if (repeat, joint) in {(row["round"], row["joint"]) for row in rows}:
            raise ValueError("Duplicate merged round/joint")
        configuration = {key: value for key, value in run["settings"].items()
                         if key not in {"point_timeout_s", "output_dir", "enable_motors", "port", "motor_config"}}
        if configuration["leg"] != expected_leg or configuration["joint"] != joint:
            raise ValueError("Merged scans must belong to the same leg and declared joint")
        if joint in settings and settings[joint] != configuration:
            raise ValueError(f"Incompatible settings for J{joint}")
        settings[joint] = configuration
        sources.append(dict(leg=expected_leg, joint=joint, round=repeat, path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            default_pose_deg=run["procedure"]["default_pose_deg"],
                            timeout_s=run["settings"]["point_timeout_s"]))
        for point in run["points"]:
            direction = point.get("observed_approach_direction", 0)
            if not point["direction"] or direction not in (-1, 1):
                continue
            saturated = events[directory, point["visit"], joint]
            rows.append(dict(leg=expected_leg, joint=joint, round=repeat, source_directory=str(directory),
                             source_visit=point["visit"], q=point["selected_position_deg"]["median"],
                             torque=point["selected_torque_nm"]["median"], direction=direction,
                             command_direction=point["direction"], reversals=len(point["motion_reversals"]),
                             settle_integral_saturated=bool(saturated),
                             settle_saturation_packets=sum(int(event["packets"]) for event in saturated),
                             **{f"q{motor}_deg": point["position_deg_by_motor"][str(motor)]["median"]
                                for motor in range(1, 6)}))
    if set(settings) != {1, 2, 3, 4, 5}:
        raise ValueError("A complete five-joint merged batch is required")
    repeats = {row["round"] for row in rows}
    if any({row["round"] for row in rows if row["joint"] == joint} != repeats for joint in settings):
        raise ValueError("Every joint must have the same complete rounds")
    if any(not np.allclose(source["default_pose_deg"], sources[0]["default_pose_deg"], atol=1.e-8)
           for source in sources):
        raise ValueError("Do not mix different default target postures")
    return rows, settings, sources


def design(q, direction, degree, mode, center, scale):
    polynomial = np.polynomial.polynomial.polyvander((np.asarray(q)-center)/scale, degree)
    if mode == "angle_only":
        return polynomial
    if mode == "constant_direction_gap":
        return np.column_stack((polynomial, direction))
    if mode == "directional_curves":
        return np.column_stack((polynomial, np.asarray(direction)[:, None]*polynomial))
    raise ValueError(f"Unknown model: {mode}")


def fit(rows, degree, mode, center, scale):
    if not rows or degree not in DEGREES or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Need rows, a supported degree, and positive scale")
    if any(row["direction"] not in (-1, 1) for row in rows):
        raise ValueError("Training requires observed increasing/decreasing arrival directions")
    q = np.array([row["q"] for row in rows])
    torque = np.array([row["torque"] for row in rows])
    direction = np.array([row["direction"] for row in rows])
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(torque)):
        raise ValueError("Nonfinite training data")
    matrix = design(q, direction, degree, mode, center, scale)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("Training data cannot identify requested model")
    counts = collections.Counter((row["round"], row["direction"]) for row in rows)
    # Equal total weight for each round/direction, never each feedback packet.
    base = np.array([1 / counts[row["round"], row["direction"]] for row in rows])
    root = np.sqrt(base)
    coefficients = np.linalg.lstsq(matrix*root[:, None], torque*root, rcond=None)[0]
    residual = torque - matrix@coefficients
    noise = max(TORQUE_QUANTUM_NM, 1.4826*np.median(np.abs(residual-np.median(residual))))
    delta = 1.5*noise  # Fixed cutoff per fit, estimated only from training data.
    converged = False
    for iteration in range(200):
        residual = torque - matrix@coefficients
        robust = np.minimum(1., delta/np.maximum(np.abs(residual), 1.e-12))
        root = np.sqrt(base*robust)
        update = np.linalg.lstsq(matrix*root[:, None], torque*root, rcond=None)[0]
        if np.max(np.abs(update-coefficients)) < 1.e-8:
            coefficients, converged = update, True
            break
        coefficients = update
    residual = torque - matrix@coefficients
    robust = np.minimum(1., delta/np.maximum(np.abs(residual), 1.e-12))
    return dict(degree=degree, mode=mode, center=center, scale=scale,
                coefficients=coefficients.tolist(), huber_delta_nm=float(delta), iterations=iteration+1,
                converged=converged, robust_weights=robust.tolist())


def predict(model, q, direction):
    q, direction = np.asarray(q, dtype=float), np.asarray(direction, dtype=float)
    return design(q, direction, model["degree"], model["mode"], model["center"], model["scale"])@model["coefficients"]


def table_predict(model, rows, grid):
    up = predict(model, grid, np.ones(len(grid)))
    down = predict(model, grid, -np.ones(len(grid)))
    if any(row["q"] < grid[0] or row["q"] > grid[-1] for row in rows):
        raise ValueError("Validation must not extrapolate beyond the exported table")
    return np.array([np.interp(row["q"], grid, up if row["direction"] > 0 else down) for row in rows])


def metrics(rows, prediction):
    errors = np.asarray([row["torque"] for row in rows])-prediction
    grouped = collections.defaultdict(list)
    for row, error in zip(rows, errors):
        grouped[row["round"], row["direction"]].append(abs(float(error)))
    return dict(points=len(rows), mae_nm=float(np.mean(np.abs(errors))),
                group_balanced_mae_nm=float(np.mean([np.mean(value) for value in grouped.values()])),
                rmse_nm=float(np.sqrt(np.mean(errors**2))),
                p95_abs_error_nm=float(np.percentile(np.abs(errors), 95)),
                max_abs_error_nm=float(np.max(np.abs(errors))))


def inside(rows, grid):
    return [row for row in rows if grid[0] <= row["q"] <= grid[-1]]


def select_degree(rows, mode, center, scale, grid, degrees=DEGREES):
    repeats = sorted({row["round"] for row in rows})
    if len(repeats) < 2:
        raise ValueError("Need at least two training rounds for degree selection")
    candidates = []
    for degree in degrees:
        scores = []
        for repeat in repeats:
            train = [row for row in rows if row["round"] != repeat]
            test = inside([row for row in rows if row["round"] == repeat], grid)
            model = fit(train, degree, mode, center, scale)
            scores.append(metrics(test, table_predict(model, test, grid))["group_balanced_mae_nm"])
        candidates.append(dict(degree=degree, mean_mae_nm=float(np.mean(scores)),
                               standard_error_nm=float(np.std(scores, ddof=1)/math.sqrt(len(scores))),
                               fold_mae_nm=scores))
    best = min(candidates, key=lambda item:item["mean_mae_nm"])
    cutoff = best["mean_mae_nm"]+best["standard_error_nm"]
    selected = min(item["degree"] for item in candidates if item["mean_mae_nm"] <= cutoff+1.e-12)
    return selected, candidates


def nested_validation(rows, mode, center, scale, grid, degrees=DEGREES):
    folds, records = [], []
    repeats = sorted({row["round"] for row in rows})
    if len(repeats) < 3:
        raise ValueError("Nested validation requires at least three distinct rounds")
    for held_out in repeats:
        train = [row for row in rows if row["round"] != held_out]
        test = inside([row for row in rows if row["round"] == held_out], grid)
        degree, selection = select_degree(train, mode, center, scale, grid, degrees)
        model = fit(train, degree, mode, center, scale)
        prediction = table_predict(model, test, grid)
        folds.append(dict(held_out_round=held_out, training_rounds=sorted({r["round"] for r in train}),
                          selected_degree=degree, inner_selection=selection, **metrics(test, prediction)))
        for row, estimate in zip(test, prediction):
            records.append(dict(row, model=mode, predicted_nm=float(estimate),
                                residual_nm=row["torque"]-float(estimate), selected_degree=degree))
    errors = np.array([row["residual_nm"] for row in records])
    return dict(mean_fold_balanced_mae_nm=float(np.mean([fold["group_balanced_mae_nm"] for fold in folds])),
                pooled_mae_nm=float(np.mean(np.abs(errors))), pooled_rmse_nm=float(np.sqrt(np.mean(errors**2))),
                pooled_p95_abs_error_nm=float(np.percentile(np.abs(errors), 95)),
                evaluated_points=len(records), excluded_boundary_points=len(rows)-len(records), folds=folds), records


def support_at(rows, angle, max_gap=4.):
    result = {}
    for direction, label in ((1, "increasing"), (-1, "decreasing")):
        gaps = []
        for repeat in sorted({row["round"] for row in rows}):
            angles = sorted({row["q"] for row in rows if row["round"] == repeat and row["direction"] == direction})
            if not angles or angle < angles[0] or angle > angles[-1]:
                continue
            index = int(np.searchsorted(angles, angle))
            gap = 0. if angles[index] == angle else angles[index]-angles[index-1]
            gaps.append(gap)
        result[f"{label}_bracketed_rounds"] = len(gaps)
        result[f"{label}_rounds_gap_le_4deg"] = sum(gap <= max_gap for gap in gaps)
        result[f"{label}_max_gap_deg"] = max(gaps) if gaps else None
    result["at_least_three_rounds_each_direction_gap_le_4deg"] = min(
        result["increasing_rounds_gap_le_4deg"], result["decreasing_rounds_gap_le_4deg"]) >= 3
    return result


def bootstrap_curves(rows, degree, center, scale, grid, count, seed):
    rng = np.random.default_rng(seed)
    repeats = sorted({row["round"] for row in rows})
    up, down = [], []
    for _ in range(count):
        sampled = []
        for replica, repeat in enumerate(rng.choice(repeats, len(repeats), replace=True), 1):
            sampled.extend(dict(row, round=replica) for row in rows if row["round"] == repeat)
        model = fit(sampled, degree, "directional_curves", center, scale)
        up.append(predict(model, grid, np.ones(len(grid))))
        down.append(predict(model, grid, -np.ones(len(grid))))
    return {label: np.percentile(values, [10, 90], axis=0).tolist() for label, values in (("increasing", up), ("decreasing", down))}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader(); writer.writerows(rows)


def run(args):
    if not math.isfinite(args.grid_deg) or not 0 < args.grid_deg <= 2 or args.bootstrap < 32:
        raise ValueError("Require grid_deg in (0,2] and at least 32 bootstrap resamples")
    rows, settings, sources = load_rows(args.merged_dir, getattr(args, "saturation_annotations", None))
    leg = sources[0]["leg"]
    repeats = sorted({row["round"] for row in rows})
    if len(repeats) < 3:
        raise ValueError("Nested round validation requires at least three complete rounds")
    degrees = DEGREES if len(repeats) >= 5 else (1, 2)
    protocol = dict(method="nested_leave_one_round_out", rounds=len(repeats),
                    outer_folds=len(repeats), outer_training_rounds=len(repeats)-1,
                    inner_folds=len(repeats)-1, inner_training_rounds=len(repeats)-2,
                    allowed_degrees=list(degrees),
                    limitation="With three rounds, inner fits use one round and degree selection is less stable than five-round validation"
                    if len(repeats)==3 else "Conditional on the recorded rounds and postures")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    all_table, all_cv, all_residuals, index = [], [], [], []
    for joint in sorted(settings):
        data = [row for row in rows if row["joint"] == joint]
        limits = []
        for repeat in repeats:
            for direction in (-1, 1):
                angles = [row["q"] for row in data if row["round"] == repeat and row["direction"] == direction]
                if len(angles) < 4:
                    raise ValueError("Every round must support both directions")
                limits.append(dict(round=repeat, direction=direction, low_deg=min(angles), high_deg=max(angles)))
        low, high = max(item["low_deg"] for item in limits), min(item["high_deg"] for item in limits)
        first, last = math.ceil(low/args.grid_deg), math.floor(high/args.grid_deg)
        grid = np.arange(first, last+1)*args.grid_deg
        if len(grid) < 4:
            raise ValueError("Insufficient common observed-angle support")
        center, scale = float((grid[0]+grid[-1])/2), float((grid[-1]-grid[0])/2)
        validation, fitted = {}, {}
        for mode in MODES:
            print(f"[FIT] {leg} J{joint} {mode}: {len(repeats)}-round nested validation", flush=True)
            validation[mode], predictions = nested_validation(data, mode, center, scale, grid, degrees)
            all_cv.extend(predictions)
            degree, selection = select_degree(data, mode, center, scale, grid, degrees)
            model = fit(data, degree, mode, center, scale)
            model["degree_selection"] = selection
            fitted[mode] = model
        model = fitted["directional_curves"]
        up = predict(model, grid, np.ones(len(grid)))
        down = predict(model, grid, -np.ones(len(grid)))
        midpoint, half = (up+down)/2, (up-down)/2
        torque_limit = settings[joint]["integral_torque_limit"][joint-1]
        if max(np.max(np.abs(up)), np.max(np.abs(down))) > torque_limit:
            raise ValueError(f"J{joint} candidate exceeds the existing feedforward limit; not clipped silently")
        bands = bootstrap_curves(data, model["degree"], center, scale, grid, args.bootstrap, args.seed+joint)
        support = [dict(angle_deg=float(angle), **support_at(data, angle)) for angle in grid]
        for n, angle in enumerate(grid):
            all_table.append(dict(joint=joint, angle_deg=float(angle), increasing_nm=float(up[n]),
                                  decreasing_nm=float(down[n]), midpoint_nm=float(midpoint[n]), half_difference_nm=float(half[n]),
                                  increasing_bootstrap_p10_nm=bands["increasing"][0][n],
                                  increasing_bootstrap_p90_nm=bands["increasing"][1][n],
                                  decreasing_bootstrap_p10_nm=bands["decreasing"][0][n],
                                  decreasing_bootstrap_p90_nm=bands["decreasing"][1][n], **{k:v for k,v in support[n].items() if k!="angle_deg"}))
        prediction = predict(model, [row["q"] for row in data], [row["direction"] for row in data])
        for row, estimate, weight in zip(data, prediction, model["robust_weights"]):
            all_residuals.append(dict(row, fitted_nm=float(estimate), residual_nm=row["torque"]-float(estimate),
                                      huber_weight=weight, strongly_downweighted=weight < .5))
        batch_holdout = {}
        for mode in MODES if len(repeats) >= 5 else ():
            train, test = [r for r in data if r["round"] <= 3], inside([r for r in data if r["round"] > 3], grid)
            degree, _ = select_degree(train, mode, center, scale, grid)
            other = fit(train, degree, mode, center, scale)
            batch_holdout[mode] = dict(selected_degree=degree, training_rounds=[1, 2, 3], test_rounds=[4, 5],
                                      **metrics(test, table_predict(other, test, grid)))
        review_flags = ["limited_three_round_validation"] if len(repeats)==3 else []
        dense = sum(item["at_least_three_rounds_each_direction_gap_le_4deg"] for item in support)
        if dense/len(support) < .75:
            review_flags.append("sparse_directional_support")
        if np.min(half) < 0 < np.max(half):
            review_flags.append("directional_half_gap_changes_sign")
        if validation["directional_curves"]["mean_fold_balanced_mae_nm"] >= validation["angle_only"]["mean_fold_balanced_mae_nm"]:
            review_flags.append("directional_curves_not_better_than_angle_only_on_holdout")
        saturation_sensitivity = dict(annotated_points=sum(row["settle_integral_saturated"] for row in data),
                                      primary_fit_retains_annotated_points=True,
                                      annotated_records=[{key:row[key] for key in ("round", "source_visit", "q", "torque", "settle_saturation_packets")}
                                                         for row in data if row["settle_integral_saturated"]])
        if saturation_sensitivity["annotated_points"]:
            review_flags.append("settle_integral_saturation_in_source")
            unsaturated = [row for row in data if not row["settle_integral_saturated"]]
            diagnostic = fit(unsaturated, model["degree"], "directional_curves", center, scale)
            saturation_sensitivity.update(
                comparison="Same degree, diagnostic refit without flagged points; primary table is unchanged",
                increasing_max_change_nm=float(np.max(np.abs(up-predict(diagnostic, grid, np.ones(len(grid)))))),
                decreasing_max_change_nm=float(np.max(np.abs(down-predict(diagnostic, grid, -np.ones(len(grid)))))))
        profile = dict(schema_version=1, model_type="directional_static_holding_torque_candidate", leg=leg, joint=joint,
                       status="offline_candidate_not_dynamically_validated", active_in_runtime=False,
                       angle_coordinate="hardware encoder feedback degrees; Joint5 sign already transformed",
                       angle_deg=grid.tolist(), increasing_torque_nm=up.tolist(), decreasing_torque_nm=down.tolist(),
                       midpoint_torque_nm=midpoint.tolist(), signed_half_difference_nm=half.tolist(),
                       formula="tau(q,s)=midpoint(q)+s*half_difference(q); s=+1 increasing arrival, -1 decreasing arrival",
                       stationary_direction="Retain approach history when known; no unique stationary branch identified from angle alone",
                       lookup="Piecewise linear in angle within the table; outside clamp to the nearest endpoint and flag out-of-domain",
                       candidate_domain_deg=[float(grid[0]), float(grid[-1])], feedforward_limit_nm=torque_limit,
                       prior_default_pose_deg=sources[0]["default_pose_deg"], kp=settings[joint]["kp"], kd=settings[joint]["kd"],
                       fitting=dict(method="Huber IRLS polynomial in normalized angle, equal round/direction total weights",
                                    degree_selection="Nested round validation; smallest degree within one standard error of the best inner CV score",
                                    selected_model={k:v for k,v in model.items() if k!="robust_weights"},
                                    points=len(data), baseline_holds_excluded=len(repeats),
                                    torque_feedback_quantum_nm=TORQUE_QUANTUM_NM,
                                    no_automatic_deletion=True, observed_direction_required=True),
                       validation=validation, validation_protocol=protocol, review_flags=review_flags,
                       settle_saturation_sensitivity=saturation_sensitivity,
                       supplemental_batch_holdout=batch_holdout,
                       alternative_fits={mode:{k:v for k,v in value.items() if k!="robust_weights"} for mode,value in fitted.items()},
                       round_bootstrap=dict(resamples=args.bootstrap, seed=args.seed+joint, percentile=[10,90], bands=bands,
                                            interpretation=f"Sensitivity to {len(repeats)} recorded rounds, not sensor accuracy or dynamic performance confidence"),
                       support=support, raw_branch_bounds=limits,
                       sources=[source for source in sources if source["joint"]==joint],
                       limitations=["Conditional on full-leg posture/support used in calibration, not a complete robot gravity model",
                                    "Directional residual includes friction, hysteresis and holding/control history; not identified pure friction",
                                    "Static observed arrival direction is not validated as a dynamic branch-switching signal",
                                    "No runtime controller, direction switching, filter, slew limit or hardware is enabled by this file",
                                    "Current version-1 LoadFrictionModel loader does not accept this candidate schema"])
        filename = f"{leg.lower()}_j{joint}_directional_static_candidate_v1.json"
        write_json(out/filename, profile)
        index.append(dict(joint=joint, file=filename, domain_deg=profile["candidate_domain_deg"], degree=model["degree"],
                          direction_gap_range_nm=[float(np.min(up-down)),float(np.max(up-down))],
                          nested_mae_nm={mode:validation[mode]["mean_fold_balanced_mae_nm"] for mode in MODES},
                          strongly_downweighted_points=sum(row["strongly_downweighted"] for row in all_residuals if row["joint"]==joint),
                          supported_nodes=dense, total_nodes=len(support), review_flags=review_flags,
                          coefficients_converged=model["converged"]))
        print(f"[CANDIDATE] {leg} J{joint}: {profile['candidate_domain_deg']}, degree={model['degree']}, CV={index[-1]['nested_mae_nm']}", flush=True)
    write_csv(out/"candidate_tables.csv", all_table)
    write_csv(out/"nested_heldout_predictions.csv", all_cv)
    write_csv(out/"fit_residuals.csv", all_residuals)
    annotations = getattr(args, "saturation_annotations", None)
    annotation_source = dict(path=str(Path(annotations).resolve()), sha256=hashlib.sha256(Path(annotations).read_bytes()).hexdigest()) if annotations else None
    write_json(out/"index.json", dict(status="offline_candidates", leg=leg, rounds=len(repeats), validation_protocol=protocol,
                                     merged_directory=str(Path(args.merged_dir).resolve()),
                                     saturation_annotation_source=annotation_source,
                                     seed=args.seed, bootstrap=args.bootstrap, profiles=index,
                                     validation="Nested leave-one-round-out; inner round splits choose polynomial degree independently in each outer fold"))
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("Source changed during fitting")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="New directory; never overwrites a prior candidate")
    parser.add_argument("--grid_deg", type=float, default=1.)
    parser.add_argument("--bootstrap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--saturation_annotations", help="Optional saturation_windows.csv from raw-command analysis; flags retained points")
    args = parser.parse_args()
    try:
        output = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[DONE] {output.resolve()}")


if __name__ == "__main__":
    main()
