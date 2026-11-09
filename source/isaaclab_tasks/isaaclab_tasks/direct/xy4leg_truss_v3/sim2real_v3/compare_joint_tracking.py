"""Pair completed fixed-bias and load/friction runs with identical test settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def signature(settings):
    fields = ("leg", "joint", "mode", "center_deg", "amplitude_deg", "min_deg", "max_deg", "duration_s",
              "hz", "poll_hz", "tolerance_deg", "integral_compensation")
    values = tuple(settings[key] for key in fields)
    mode = settings["mode"]
    detail = (settings["frequency_hz"],) if mode == "sine" else (settings["hold_s"],)
    if mode == "ramp":
        detail += (settings["ramp_speed_deg_s"],)
    return values + detail + (settings.get("tx_hz") or settings["hz"],
                              settings.get("warmup_tolerance_deg", 1.),
                              settings["warmup_timeout_s"], tuple(settings["kp"]), tuple(settings["kd"]))


def completed_runs(root, kind):
    result = []
    for path in sorted(Path(root).rglob("run.json"), key=lambda p: p.stat().st_mtime):
        metadata = json.loads(path.read_text())
        model = bool(metadata["settings"].get("compensation_model"))
        scale = metadata["settings"].get("friction_scale", 1. if model else 0.)
        selected = (not model if kind == "fixed" else model and scale == 0. if kind == "load_only" else model)
        if metadata.get("complete") and metadata.get("run_type") == "hardware" and selected:
            summary_path = path.parent / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("complete") and summary.get("tracking") is not None:
                    result.append((path.parent, metadata, summary))
    return result


def _model_key(metadata):
    snapshot = metadata.get("compensation_model_snapshot")
    if snapshot is not None:
        return json.dumps(snapshot, sort_keys=True)
    return metadata.get("model_snapshot_sha256")


def compare(baseline_root, candidate_root, baseline_mode="fixed"):
    if baseline_mode not in ("fixed", "load_only"):
        raise ValueError("baseline_mode must be fixed or load_only")
    baselines = {(signature(meta["settings"]), _model_key(meta) if baseline_mode == "load_only" else None):
                 (path, meta, summary) for path, meta, summary in completed_runs(baseline_root, baseline_mode)}
    # Retain each friction scale/model variant; do not overwrite it with the latest different scale.
    candidates = {(signature(meta["settings"]), _model_key(meta), meta["settings"].get("friction_scale", 1.)):
                  (path, meta, summary) for path, meta, summary in completed_runs(candidate_root, "model")}
    if not candidates:
        raise ValueError("No completed hardware load/friction runs found; previews/aborted runs cannot be compared")
    pairs, unmatched = [], []
    for (trajectory, model_key, scale), (path, meta, after) in candidates.items():
        key = (trajectory, model_key if baseline_mode == "load_only" else None)
        if key not in baselines or (baseline_mode == "load_only" and model_key is None):
            unmatched.append(dict(candidate=str(path), reason="No baseline with identical settings and required fixed model snapshot"))
            continue
        before_path, before_meta, before = baselines[key]
        changes = {}
        for name in ("rmse_deg", "abs_error_p95_deg", "abs_error_max_deg", "bias_deg"):
            a, b = before["tracking"][name], after["tracking"][name]
            changes[name] = dict(before=a, after=b, difference=b-a,
                                 reduction_percent=100*(a-b)/a if a > 0 and name != "bias_deg" else None)
        pairs.append(dict(mode=meta["settings"]["mode"], frequency_hz=meta["settings"]["frequency_hz"],
                          baseline_mode=baseline_mode, friction_scale=scale,
                          baseline=str(before_path), candidate=str(path), metrics=changes,
                          before_sine_fit=before.get("sine_fit"), after_sine_fit=after.get("sine_fit"),
                          before_settled_steps=sum(row["observed_settling_ms"] is not None for row in before.get("steps", [])),
                          after_settled_steps=sum(row["observed_settling_ms"] is not None for row in after.get("steps", [])),
                          before_warmup=before_meta.get("warmup"), after_warmup=meta.get("warmup"),
                          target_rate_hz=after["target_update_cadence"]["hz"],
                          tx_rate_hz=after["command_cadence"]["hz"], rx_rate_hz=after["feedback_cadence"]["hz"],
                          compensation=after.get("compensation"),
                          coverage=after.get("coverage"),
                          before_calibrated_tracking=before.get("tracking_calibrated_only"),
                          after_calibrated_tracking=after.get("tracking_calibrated_only"),
                          model_snapshot=meta.get("compensation_model_snapshot")))
    return dict(pairs=pairs, unmatched=unmatched, baseline_mode=baseline_mode,
                interpretation="Matched settings; startup history and learned warmup torques can still differ. "
                               "Positive reduction means smaller error, not automatic task-level acceptance.")


def write_report(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    lines = ["# 负载＋摩擦补偿前后对照", "", "只比较完整实机记录；轨迹、监测范围、增益和频率必须一致。", "",
             "| 测试 | 摩擦比例 | RMSE 前→后 | p95 前→后 | p95 降低 | 未标定查询帧占比 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for pair in result["pairs"]:
        metrics = pair["metrics"]
        rmse, p95 = metrics["rmse_deg"], metrics["abs_error_p95_deg"]
        label = pair["mode"] + (f" {pair['frequency_hz']:g}Hz" if pair["mode"] == "sine" else "")
        reduction = p95["reduction_percent"]
        coverage = (pair.get("compensation") or {}).get("clamped_load_fraction")
        lines.append(f"| {label} | {pair['friction_scale']:g} | {rmse['before']:.3f}° → {rmse['after']:.3f}° | "
                     f"{p95['before']:.3f}° → {p95['after']:.3f}° | "
                     f"{f'{reduction:.1f}%' if reduction is not None else 'n/a'} | "
                     f"{f'{coverage:.1%}' if coverage is not None else 'n/a'} |")
    lines += ["", "未标定区域使用端点夹紧，不能将其视为完整角度范围验证。", "",
              "启动历史和预热补偿仍可能不同；误差变小也不等于满足任务要求。", ""]
    for pair in result["pairs"]:
        lines += [f"- 前：`{pair['baseline']}`；后：`{pair['candidate']}`"]
    if result["unmatched"]:
        lines += ["", "无法配对：", json.dumps(result["unmatched"], ensure_ascii=False, indent=2)]
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_root", type=Path, default=Path("logs/real_v3"))
    parser.add_argument("--baseline_mode", choices=("fixed", "load_only"), default="fixed",
                        help="Use fixed-bias runs or the same fixed load table with friction disabled as the baseline.")
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_root, args.candidate_root, args.baseline_mode)
    write_report(result, args.output_dir)
    print(json.dumps(dict(paired=len(result["pairs"]), unmatched=len(result["unmatched"]),
                          output=str(args.output_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
