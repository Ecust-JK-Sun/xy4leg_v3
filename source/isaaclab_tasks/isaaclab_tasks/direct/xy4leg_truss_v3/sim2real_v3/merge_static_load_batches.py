"""Merge complete J1-J5 rounds offline, retaining source logs and visit provenance.

Raw per-scan directories are linked into the result; keep the source directories.
Incomplete rounds are excluded. No serial port is opened and no motor is enabled.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import calibrate_joint_load as calibration


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rounds(batches):
    rounds, excluded, seen_batches, seen_scans = [], [], set(), set()
    for batch_path in batches:
        root = Path(batch_path).resolve()
        if root in seen_batches:
            raise ValueError(f"Duplicate source batch: {root}")
        seen_batches.add(root)
        batch = read_json(root / "batch.json")
        records = {}
        for entry in batch["runs"]:
            key = (entry["round"], entry["joint"])
            if key in records:
                raise ValueError(f"Duplicate source round/joint {key}: {root}")
            records[key] = entry
        planned_rounds = {entry["round"] for entry in batch["procedure"].get("plan", [])}
        for round_number in sorted(planned_rounds | {key[0] for key in records}):
            entries = [records.get((round_number, joint)) for joint in calibration.MOTOR_IDS]
            if any(entry is None or entry["status"] != "complete" for entry in entries):
                excluded.append(dict(source_batch=str(root), source_round=round_number,
                                     reason="Not all five scans completed; entire round excluded"))
                continue
            scans = []
            for joint, entry in zip(calibration.MOTOR_IDS, entries):
                directory = (root / entry["directory"]).resolve()
                if directory in seen_scans:
                    raise ValueError(f"Duplicate source scan: {directory}")
                seen_scans.add(directory)
                run = read_json(directory / "run.json")
                summary = read_json(directory / "summary.json")
                points = run["points"]
                expected = len(run["procedure"]["plan"])
                if (not run["complete"] or not summary["complete"] or
                        run["return_to_initial"]["status"] != "complete" or
                        summary.get("return_to_initial_status") != "complete" or
                        len(points) != expected or not all(point.get("valid") for point in points)):
                    raise ValueError(f"Source marked complete has incomplete data or return verification: {directory}")
                if run["settings"]["joint"] != joint:
                    raise ValueError(f"Source joint does not match batch manifest: {directory}")
                scans.append(dict(directory=directory, run=run))
            rounds.append(dict(source_batch=str(root), source_round=round_number, scans=scans))
    return rounds, excluded


def merge(batches, output_dir, expected_rounds=5):
    if expected_rounds < 1:
        raise ValueError("expected_rounds must be >= 1")
    rounds, excluded = collect_rounds(batches)
    if len(rounds) != expected_rounds:
        raise ValueError(f"Expected {expected_rounds} complete rounds, found {len(rounds)}; no output created")
    # A longer timeout permits additional settling; all other acquisition settings
    # must agree for each joint. Resolved gains are compared, not YAML filenames.
    ignored_settings = {"output_dir", "point_timeout_s", "motor_config", "port", "enable_motors"}
    reference, seen_legs = {}, set()
    for repeat in rounds:
        for scan in repeat["scans"]:
            settings = scan["run"]["settings"]
            joint = settings["joint"]
            seen_legs.add(settings["leg"])
            comparable = {key: value for key, value in settings.items() if key not in ignored_settings}
            if joint in reference and reference[joint] != comparable:
                raise ValueError(f"Incompatible acquisition settings for J{joint}: {scan['directory']}")
            reference[joint] = comparable
    if len(seen_legs) != 1:
        raise ValueError("All source rounds must measure the same leg")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    records, rows = [], []
    points_by_joint = {joint: [] for joint in calibration.MOTOR_IDS}
    for merged_round, repeat in enumerate(rounds, 1):
        for scan in repeat["scans"]:
            run = scan["run"]
            joint = run["settings"]["joint"]
            name = f"round_{merged_round:02d}_j{joint}"
            (directory / name).symlink_to(scan["directory"], target_is_directory=True)
            records.append(dict(round=merged_round, joint=joint, directory=name, status="complete",
                                source_batch=repeat["source_batch"], source_round=repeat["source_round"],
                                source_directory=str(scan["directory"]), planned_visits=len(run["points"]),
                                point_timeout_s=run["settings"]["point_timeout_s"]))
            for original in run["points"]:
                point = dict(original, visit=len(points_by_joint[joint]))
                points_by_joint[joint].append(point)
                rows.append(dict(joint=joint, round=merged_round, visit=point["visit"],
                                 source_batch=repeat["source_batch"], source_round=repeat["source_round"],
                                 source_directory=str(scan["directory"]), source_visit=original["visit"],
                                 target_deg=point["target_deg"], command_direction=point["direction"],
                                 observed_approach_direction=point.get("observed_approach_direction"),
                                 actual_deg=point["selected_position_deg"]["median"],
                                 torque_nm=point["selected_torque_nm"]["median"]))
    for joint, points in points_by_joint.items():
        summary = dict(joint=joint, complete=True, rounds=len(rounds), valid_visits=len(points),
                       angle_summary=calibration.summarize_points(points),
                       note="Visit IDs are unique within this joint; points.csv maps each visit to its original scan.")
        (directory / f"summary_j{joint}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "points.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(run_type="offline_merge", complete=True, error=None,
                    procedure=dict(leg=next(iter(seen_legs)), rounds=len(rounds), scans=len(records),
                                   joint_order=list(calibration.MOTOR_IDS), planned_visits=len(rows),
                                   plan=records),
                    runs=records, excluded_rounds=excluded,
                    note="Raw scan directories are symbolic links to sources. Keep source directories. "
                    "Host timestamps remain local to each source scan; source logs are not edited. "
                    "Point timeouts may differ and are recorded per scan. Direction grouping uses observed arrival motion.")
    (directory / "batch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MERGED] {len(rounds)} complete rounds, {len(records)} scans, {len(rows)} valid visits: {directory.resolve()}")
    for repeat in excluded:
        print(f"[EXCLUDED] {repeat['source_batch']} round {repeat['source_round']}: {repeat['reason']}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", required=True, help="Source batch directories, in desired round order")
    parser.add_argument("--output_dir", required=True, help="New directory; existing directories are never overwritten")
    parser.add_argument("--expected_rounds", type=int, default=5)
    args = parser.parse_args()
    try:
        merge(args.batches, args.output_dir, args.expected_rounds)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
