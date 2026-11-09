"""Scan J1 through J5 in round order, verifying the full default pose between scans.

Defaults: five rounds, one initial->min->max->initial sweep per joint per round,
default pose +/-30 degrees, 2 degree steps. Preview unless --enable_motors.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import calibrate_joint_load as calibration


def build_parser():
    parser = calibration.build_parser(batch=True)
    parser.description = __doc__
    parser.add_argument("--resume_batch", help="Resume an existing batch using its saved settings; preview unless --enable_motors")
    return parser


def make_runs(args):
    """Validate the whole batch before creating logs or opening a serial port."""
    if args.rounds < 1:
        raise ValueError("rounds must be >= 1")
    if not math.isfinite(args.range_deg) or args.range_deg <= 0:
        raise ValueError("range_deg must be positive and finite")
    if args.return_tolerance_deg is None:
        raise ValueError("Batch runs require full-leg return verification")
    defaults = calibration._target_rad_for_leg(args.leg, True)
    runs = []
    for round_number in range(1, args.rounds + 1):
        for joint in calibration.MOTOR_IDS:
            settings = vars(args).copy()
            for key in ("rounds", "range_deg", "resume_batch"):
                settings.pop(key, None)
            initial = round(math.degrees(defaults[joint - 1]), 10)
            settings.update(joint=joint, min_deg=initial - args.range_deg,
                            max_deg=initial + args.range_deg, cycles=1)
            child = argparse.Namespace(**settings)
            plan = calibration.validate(child)
            runs.append((dict(round=round_number, joint=joint, min_deg=child.min_deg,
                              max_deg=child.max_deg, planned_visits=len(plan),
                              directory=f"round_{round_number:02d}_j{joint}"), child))
    return runs


def run(args):
    if args.resume_batch:
        return resume(args)
    runs = make_runs(args)
    visits = sum(entry["planned_visits"] for entry, _ in runs)
    description = dict(leg=args.leg, rounds=args.rounds, joint_order=list(calibration.MOTOR_IDS),
                       settings=vars(args), scans=len(runs), planned_visits=visits,
                       default_pose_deg=[math.degrees(x) for x in calibration._target_rad_for_leg(args.leg, True)],
                       all_joint_return_tolerance_deg=args.return_tolerance_deg,
                       plan=[entry for entry, _ in runs],
                       minimum_hold_total_s=(visits + 2 * len(runs)) *
                       (args.stable_s + args.discard_s + args.sample_s),
                       note="Hold minimum includes initialization and return checks; excludes movement and settling delays.")
    print(json.dumps(description, ensure_ascii=False, indent=2))
    if not args.enable_motors:
        print("[PREVIEW] No serial port opened or logs created. Add --enable_motors to execute.")
        return description
    root = calibration.THIS_DIR.parents[5] / "logs/real_v3"
    directory = Path(args.output_dir) if args.output_dir else root / (
        f"static_load_{args.leg}_j1_j5_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(procedure=description, complete=False, error=None, runs=[])

    def save():
        (directory / "batch.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        for entry, child in runs:
            record = dict(entry, status="running")
            metadata["runs"].append(record)
            save()
            child.output_dir = str(directory / entry["directory"])
            print(f"[BATCH] Round {entry['round']}/{args.rounds}, {args.leg} J{entry['joint']}")
            calibration.run(child)
            # Do not advance merely because a runner returned without raising.
            summary = json.loads((Path(child.output_dir) / "summary.json").read_text(encoding="utf-8"))
            if not summary["complete"] or summary.get("return_to_initial_status") != "complete":
                raise RuntimeError("Scan or full-leg return verification incomplete; batch stopped")
            record["status"] = "complete"
            save()
        metadata["complete"] = True
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if metadata["runs"] and metadata["runs"][-1]["status"] == "running":
            metadata["runs"][-1].update(status="failed", error=metadata["error"])
        raise
    finally:
        save()
        print(f"[BATCH] Results: {directory.resolve()}")
    return directory


def resume(args):
    """Skip verified scans, retry failed scans in new directories, preserve all attempts."""
    directory = Path(args.resume_batch).resolve()
    batch_path = directory / "batch.json"
    metadata = json.loads(batch_path.read_text(encoding="utf-8"))
    settings = dict(metadata["procedure"]["settings"], resume_batch=None, enable_motors=args.enable_motors)
    records = {}
    for record in metadata["runs"]:
        key = (record["round"], record["joint"])
        if key in records:
            raise ValueError(f"Duplicate batch record: {key}")
        records[key] = record
    # Freeze the actual resolved gains used by this batch, even if its initial
    # command omitted gains and code/config defaults have changed since then.
    for record in metadata["runs"]:
        if record["status"] == "complete":
            previous = json.loads((directory / record["directory"] / "run.json").read_text(encoding="utf-8"))
            for key in ("kp", "kd"):
                if settings.get(key) is None:
                    settings[key] = previous["settings"][key]
            break
    runs = make_runs(argparse.Namespace(**settings))
    pending, skipped = [], []
    for entry, child in runs:
        key = (entry["round"], entry["joint"])
        record = records.get(key)
        if record and record["status"] == "complete":
            path = directory / record["directory"]
            previous = json.loads((path / "run.json").read_text(encoding="utf-8"))
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if (not previous["complete"] or not summary["complete"] or
                    previous["return_to_initial"]["status"] != "complete" or
                    summary.get("return_to_initial_status") != "complete" or
                    len(previous["points"]) != entry["planned_visits"] or
                    not all(point.get("valid") for point in previous["points"])):
                raise ValueError(f"Cannot skip unverified completed scan: {path}")
            ignored = {"output_dir", "enable_motors"}
            actual = {k: v for k, v in previous["settings"].items() if k not in ignored}
            expected = {k: v for k, v in vars(child).items() if k not in ignored}
            if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError(f"Saved settings do not match completed scan: {path}")
            skipped.append(entry)
            continue
        name, attempt = entry["directory"], 0
        while (directory / name).exists():
            attempt += 1
            name = f"{entry['directory']}_retry_{attempt:02d}"
        pending.append((dict(entry, directory=name), child))
    description = dict(batch=str(directory), skipped_complete_scans=len(skipped), pending_scans=len(pending),
                       settings=settings, plan=[entry for entry, _ in pending])
    print(json.dumps(description, ensure_ascii=False, indent=2))
    if not args.enable_motors or not pending:
        print("[RESUME PREVIEW] No serial port opened or files changed." if pending else "[RESUME] All scans already complete.")
        return description
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    (directory / f"batch_before_resume_{stamp}.json").write_text(batch_path.read_text(encoding="utf-8"), encoding="utf-8")
    history = dict(started_at=stamp, skipped_complete_scans=len(skipped), pending_scans=len(pending), status="running")
    metadata.setdefault("resume_history", []).append(history)
    metadata.update(complete=False, error=None)

    def save():
        temporary = directory / "batch.json.tmp"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(batch_path)

    current = None
    try:
        for entry, child in pending:
            prior = records.get((entry["round"], entry["joint"]))
            current = dict(entry, status="running")
            if prior:
                current["previous_attempts"] = list(prior.get("previous_attempts", [])) + [
                    {k: v for k, v in prior.items() if k != "previous_attempts"}]
                metadata["runs"][metadata["runs"].index(prior)] = current
            else:
                metadata["runs"].append(current)
            save()
            child.output_dir = str(directory / entry["directory"])
            print(f"[RESUME] Round {entry['round']}/{settings['rounds']}, {child.leg} J{entry['joint']}")
            calibration.run(child)
            summary = json.loads((Path(child.output_dir) / "summary.json").read_text(encoding="utf-8"))
            if not summary["complete"] or summary.get("return_to_initial_status") != "complete":
                raise RuntimeError("Scan or full-leg return verification incomplete; batch stopped")
            current["status"] = "complete"
            save()
        metadata["complete"] = True
        history["status"] = "complete"
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        history.update(status="failed", error=metadata["error"])
        if current is not None and current["status"] == "running":
            current.update(status="failed", error=metadata["error"])
        raise
    finally:
        save()
        print(f"[RESUME] Results: {directory}")
    return directory


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
