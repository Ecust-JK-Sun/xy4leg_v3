"""Paired random-position experiments: frozen initial FF, midpoint, directional.

Preview by default. Identical bounded random targets, gains and holds per trial.
Only one joint moves; all five joints return to default after every variant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics as st
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import calibrate_joint_load as static
from directional_compensation import DirectionalController, DirectionalTable

REPO = static.THIS_DIR.parents[5]
CATALOG = static.THIS_DIR / "calibration/directional_candidate_catalog_20261030.json"
VARIANTS = ("fixed", "midpoint", "directional")
CONTROL_FIELDS = ("variant", "applied_direction", "observed_direction", "requested_fixed_nm", "requested_midpoint_nm",
                  "requested_directional_nm", "requested_ff_nm", "ff_saturated", "model_clamped", "model_low_support",
                  "model_angle_deg", "model_feedback_time_s")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def make_plan(initial, low, high, points, minimum_step, maximum_step, seed):
    """Uniform on feasible left/right intervals; a bounded random walk, not IID targets."""
    if (not all(math.isfinite(x) for x in (initial, low, high, minimum_step, maximum_step)) or
            not low <= initial <= high or low >= high or not 0 < minimum_step < maximum_step or
            high-low < 2*minimum_step or points < 1):
        raise ValueError("Invalid random target range/step/count")
    rng, previous, plan = random.Random(seed), initial, []
    for step in range(points):
        intervals = [(max(low, previous-maximum_step), min(high, previous-minimum_step)),
                     (max(low, previous+minimum_step), min(high, previous+maximum_step))]
        intervals = [(a,b) for a,b in intervals if b>a]
        if not intervals:
            raise ValueError("No feasible next random target")
        offset = rng.random()*sum(b-a for a,b in intervals)
        for a,b in intervals:
            if offset <= b-a:
                target = a+offset
                break
            offset -= b-a
        plan.append(dict(step=step, previous_deg=previous, target_deg=target,
                         direction=1 if target>previous else -1, delta_deg=target-previous))
        previous = target
    return plan


def resolve_model(args):
    if args.model:
        path = Path(args.model).resolve()
        payload = path.read_bytes()
    else:
        catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
        matches = [entry for entry in catalog["profiles"] if (entry["leg"],entry["joint"]) == (args.leg,args.joint)]
        if len(matches) != 1:
            raise ValueError("Candidate catalog must contain exactly one match for this leg/joint")
        path = Path(matches[0]["path"])
        path = path if path.is_absolute() else REPO/path
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != matches[0]["sha256"]:
            raise ValueError("Candidate differs from the frozen catalog; update the catalog explicitly")
    model = DirectionalTable(json.loads(payload))
    model.validate_identity(args.leg,args.joint)
    return model, dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest())


def prepare(args, model):
    positive = ("hold_s", "initial_hold_s", "steady_s", "settle_s", "tolerance_deg", "tx_hz", "poll_hz",
                "point_timeout_s", "return_tolerance_deg", "filter_tau_s", "slew_rate_nm_s", "feedback_timeout_s",
                "init_speed_deg_s", "init_range_deg", "direction_resolution_deg")
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in positive):
        raise ValueError("Durations, rates and tolerances must be finite and positive")
    if (args.trials<1 or args.points<2 or not 10<=args.tx_hz<=100 or args.poll_hz<args.tx_hz or
            args.steady_s>=args.hold_s or args.settle_s>=args.hold_s or args.steady_s<5/args.tx_hz or
            args.points*args.hold_s+args.initial_hold_s>1800 or args.return_tolerance_deg<=.25):
        raise ValueError("Invalid trial count, hold/sample timing, return tolerance or send/poll rate")
    if not math.isfinite(args.range_margin_deg) or args.range_margin_deg<0:
        raise ValueError("range_margin_deg must be finite and nonnegative")
    if not math.isfinite(args.monitor_margin_deg) or args.monitor_margin_deg<0:
        raise ValueError("monitor_margin_deg must be finite and nonnegative")
    if args.min_step_deg <= args.tolerance_deg:
        raise ValueError("Minimum target step must exceed the accuracy tolerance")
    base = static._target_rad_for_leg(args.leg,True)
    defaults = [math.degrees(x) for x in base]
    pose=model.data.get("prior_default_pose_deg",[])
    if len(pose)!=5 or any(not math.isfinite(q) for q in pose) or any(abs(a-b)>1.e-7 for a,b in zip(defaults,pose)):
        raise ValueError("Candidate default posture differs from this leg's initialization")
    initial = defaults[args.joint-1]
    low = args.min_deg if args.min_deg is not None else model.angles[0]+args.range_margin_deg
    high = args.max_deg if args.max_deg is not None else model.angles[-1]-args.range_margin_deg
    if not model.angles[0]<=low<high<=model.angles[-1] or not initial-30<=low<high<=initial+30:
        raise ValueError("Targets must lie inside candidate support and the previously tested +/-30 deg range")
    args.kp = tuple(args.kp or model.data["kp"])
    args.kd = tuple(args.kd or model.data["kd"])
    if any(len(values)!=5 for values in (args.kp,args.kd)):
        raise ValueError("Exactly five kp/kd values are required before opening hardware")
    static_args = static.build_parser().parse_args([
        "--leg",args.leg,"--joint",str(args.joint),"--min_deg",str(initial-30),"--max_deg",str(initial+30),"--cycles","1"])
    for key in ("kp","kd","tx_hz","poll_hz","point_timeout_s","return_tolerance_deg","feedback_timeout_s",
                "init_speed_deg_s","init_range_deg","monitor_margin_deg","integral_torque_limit"):
        setattr(static_args,key,getattr(args,key))
    static.validate(static_args)
    static_args.other_tolerance_deg=args.return_tolerance_deg
    if abs(static_args.integral_torque_limit[args.joint-1]-model.limit)>1.e-9:
        raise ValueError("Warmup and candidate feedforward limits must match")
    cases = []
    for trial in range(1,args.trials+1):
        plan = make_plan(initial,low,high,args.points,args.min_step_deg,args.max_step_deg,args.seed+trial-1)
        order = VARIANTS[(trial-1)%3:]+VARIANTS[:(trial-1)%3]
        for variant in order:
            cases.append(dict(trial=trial,variant=variant,plan=plan,plan_sha256=digest(plan),
                              directory=f"trial_{trial:02d}_{variant}",status="planned"))
    description = dict(leg=args.leg,joint=args.joint,trials=args.trials,variants=list(VARIANTS),
                       points_per_trial=args.points,seed=args.seed,min_step_deg=args.min_step_deg,max_step_deg=args.max_step_deg,
                       hold_s=args.hold_s,kp=list(args.kp),kd=list(args.kd),feedforward_limit_nm=model.limit,
                       filter_tau_s=args.filter_tau_s,slew_rate_nm_s=args.slew_rate_nm_s,
                       target_bounds_deg=[low,high],monitor_bounds_deg=[initial-30-args.monitor_margin_deg,
                                                                     initial+30+args.monitor_margin_deg],
                       initial_deg=initial,default_pose_deg=defaults,groups=len(cases),
                       random_hold_total_s=len(cases)*(args.initial_hold_s+args.points*args.hold_s),
                       variant_order=[dict(trial=t,order=[case["variant"] for case in cases if case["trial"]==t])
                                      for t in range(1,args.trials+1)],
                       target_spans=[dict(trial=t,minimum_deg=min(p["target_deg"] for p in cases[(t-1)*3]["plan"]),
                                          maximum_deg=max(p["target_deg"] for p in cases[(t-1)*3]["plan"])) for t in range(1,args.trials+1)],
                       direction_source=args.direction_source,model_review_flags=model.data.get("review_flags",[]),
                       note="Bounded random walk, same targets and hold times in all variants of a trial. "
                            "Fixed baseline freezes initial-pose learned feedforward; motion-phase integration is disabled. "
                            "Hold totals exclude initialization/return. Commanded direction is not proven equal to static approach history.")
    return static_args,cases,description


class CommandWriter:
    def __init__(self, writer, selected, variant):
        self.writer,self.selected,self.variant = writer,selected,variant
        self.context,self.selected_rows = {},[]

    def writerow(self, row):
        extras = {field:0 for field in CONTROL_FIELDS}
        extras["variant"] = self.variant
        if row["motor_id"]==self.selected:
            extras.update(self.context)
        value = dict(row,**extras)
        self.writer.writerow(value)
        if row["motor_id"]==self.selected:
            self.selected_rows.append(value)


class FeedbackWriter:
    def __init__(self, writer, selected):
        self.writer,self.selected,self.selected_rows=writer,selected,[]

    def writerow(self, row):
        self.writer.writerow(row)
        if row["motor_id"]==self.selected:
            self.selected_rows.append(dict(row))


def ramp_to_default(session, current, targets, visit, phase):
    duration = max(abs(math.degrees(b-a)) for a,b in zip(current,targets))/session.args.init_speed_deg_s
    started=time.monotonic()
    while time.monotonic()-started<duration:
        fraction=min(1.,(time.monotonic()-started)/duration)
        session.tick([a+fraction*(b-a) for a,b in zip(current,targets)],visit,phase,False)


def analyze_steps(args, plan, commands, feedback, capture_end_s):
    selected = sorted(feedback,key=lambda row:row["time_s"])
    starts = {point["step"]:next((row["time_s"] for row in commands if row["phase"]=="random_hold" and row["visit"]==point["step"]),None)
              for point in plan}
    result=[]
    for point in plan:
        step=point["step"]
        begin=starts[step]
        if begin is None:
            result.append(dict(step=step,target_deg=point["target_deg"],valid=False,reason="target_not_sent"))
            continue
        end=starts.get(step+1) or capture_end_s
        window=[row for row in selected if begin<=row["time_s"]<end]
        tail=[row for row in window if row["time_s"]>=end-args.steady_s]
        adequate=(len(tail)>=max(5,int(.5*args.tx_hz*args.steady_s)) and
                  tail[-1]["time_s"]-tail[0]["time_s"]>=.8*args.steady_s)
        errors=[point["target_deg"]-row["position_deg"] for row in window]
        outside=[i for i,error in enumerate(errors) if abs(error)>args.tolerance_deg]
        candidate=outside[-1]+1 if outside else 0
        settled=(candidate<len(window) and window[-1]["time_s"]-window[candidate]["time_s"]>=args.settle_s and
                 max((b["time_s"]-a["time_s"] for a,b in zip(window[candidate:],window[candidate+1:])),default=0)<=args.feedback_timeout_s)
        arrival=next((row["time_s"] for row,error in zip(window,errors) if abs(error)<=args.tolerance_deg),None)
        preceding=[sample for sample in selected if sample["time_s"]<begin]
        initial_position=preceding[-1]["position_deg"] if preceding else (window[0]["position_deg"] if window else None)
        tracker=static.ApproachTracker(initial_position,args.direction_resolution_deg)
        for row in window:tracker.observe(row["position_deg"],row["time_s"])
        row=dict(step=step,target_deg=point["target_deg"],previous_deg=point["previous_deg"],direction=point["direction"],
                 valid=adequate,reason=None if adequate else "insufficient_steady_feedback",first_command_s=begin,end_s=end,
                 initial_position_deg=initial_position,observed_direction_resolution_deg=args.direction_resolution_deg,
                 feedback_samples=len(window),steady_samples=len(tail),
                 first_arrival_ms=(arrival-begin)*1000 if arrival is not None else None,
                 sustained_settling_ms=(window[candidate]["time_s"]-begin)*1000 if settled else None,
                 observed_approach_direction=tracker.direction,observed_reversals=len(tracker.reversals),
                 overshoot_deg=max([0.]+[point["direction"]*(sample["position_deg"]-point["target_deg"]) for sample in window]))
        if adequate:
            tail_errors=[point["target_deg"]-sample["position_deg"] for sample in tail]
            row.update(steady_error_median_deg=st.median(tail_errors),steady_mae_deg=st.mean(map(abs,tail_errors)),
                       steady_rmse_deg=math.sqrt(st.mean(e*e for e in tail_errors)),
                       steady_position_span_deg=max(s["position_deg"] for s in tail)-min(s["position_deg"] for s in tail),
                       steady_all_within_tolerance=all(abs(e)<=args.tolerance_deg for e in tail_errors))
        for ms in (40,100,200):
            sample=next((s for s in window if s["time_s"]>=begin+ms/1000),None)
            if sample and sample["time_s"]<=begin+ms/1000+.025:
                row[f"error_at_{ms}ms_deg"]=point["target_deg"]-sample["position_deg"]
                row[f"observed_time_at_{ms}ms_ms"]=(sample["time_s"]-begin)*1000
            else:
                row[f"error_at_{ms}ms_deg"]=None
                row[f"observed_time_at_{ms}ms_ms"]=None
        result.append(row)
    return result


def aggregate_steps(points):
    valid=[p for p in points if p.get("valid")]
    settled=[p["sustained_settling_ms"] for p in valid if p["sustained_settling_ms"] is not None]
    return dict(planned_steps=len(points),valid_steps=len(valid),
                steady_mae_deg=st.mean(p["steady_mae_deg"] for p in valid) if valid else None,
                steady_abs_median_error_deg=st.mean(abs(p["steady_error_median_deg"]) for p in valid) if valid else None,
                steady_pass_fraction=st.mean(p["steady_all_within_tolerance"] for p in valid) if valid else None,
                settled_fraction=len(settled)/len(valid) if valid else None,
                settling_median_ms=st.median(settled) if settled else None,
                max_overshoot_deg=max((p["overshoot_deg"] for p in valid),default=None),
                reversed_arrival_steps=sum(p["observed_approach_direction"] not in (0,p["direction"]) for p in valid),
                settling_interpretation="Median only over settled steps; use settled_fraction to account for failures. No device 40ms certification.")


def run_group(args, static_args, case, model, directory):
    directory.mkdir(parents=True,exist_ok=False)
    metadata=dict(settings=vars(args),case={k:v for k,v in case.items() if k not in ("status","error","previous_attempts")},model_snapshot=model.data,complete=False,error=None,
                  experimental_model_application=case["variant"]!="fixed",policy_runtime_modified=False,
                  initialization=dict(status="not_started"),return_to_initial=dict(status="not_started"),
                  model_snapshot_sha256=digest(model.data),plan_sha256=case["plan_sha256"],
                  certified_40ms=False,baseline_definition="Initial-pose learned feedforward frozen throughout random holds")
    ctrl=session=None
    command_writer=feedback_writer=None
    capture_end=None
    base=static._target_rad_for_leg(args.leg,True)
    bounds={motor:(math.degrees(target)-args.init_range_deg-args.monitor_margin_deg,
                   math.degrees(target)+args.init_range_deg+args.monitor_margin_deg) for motor,target in zip(static.MOTOR_IDS,base)}
    try:
        with (directory/"commands.csv").open("x",newline="") as command_stream, (directory/"feedback.csv").open("x",newline="") as feedback_stream:
            commands=csv.DictWriter(command_stream,static.COMMAND_FIELDS+CONTROL_FIELDS)
            feedback=csv.DictWriter(feedback_stream,static.FEEDBACK_FIELDS)
            commands.writeheader();feedback.writeheader()
            command_writer=CommandWriter(commands,args.joint,case["variant"])
            feedback_writer=FeedbackWriter(feedback,args.joint)
            save_json(directory/"run.json",metadata)
            ctrl=static.DaMiaoMotorController(args.port or static.LEG_PORTS[args.leg])
            current=[static._enable_and_read_current(ctrl,motor) for motor in static.MOTOR_IDS]
            if any(value is None for value in current):raise RuntimeError("Missing initial joint positions")
            metadata["initialization"]["measured_start_deg"]=[math.degrees(x) for x in current]
            if any(not bounds[m][0]<=math.degrees(q)<=bounds[m][1] for m,q in zip(static.MOTOR_IDS,current)):
                raise RuntimeError("Initial position outside initialization bounds; no pose command sent")
            session=static.Session(static_args,ctrl,command_writer,feedback_writer)
            session.monitor_bounds_deg=bounds
            ramp_to_default(session,current,base,-1,"warmup_ramp")
            warmup=dict(visit=-1,target_deg=math.degrees(base[args.joint-1]),direction=0)
            metadata["initialization"].update(status="verifying",pose_verification=warmup)
            static.measure_point(session,warmup,base,phase_prefix="warmup_")
            metadata["initialization"].update(status="complete",initial_ff_nm=session.integral.torque.copy(),
                                              measured_pose_deg=[math.degrees(session.latest[m]["position"]) for m in static.MOTOR_IDS])
            # Monitor the prior confirmed scan range, independently of random target limits.
            defaults=[math.degrees(x) for x in base]
            session.monitor_bounds_deg={m:(defaults[m-1]-30-args.monitor_margin_deg,defaults[m-1]+30+args.monitor_margin_deg)
                                        for m in static.MOTOR_IDS}
            initial_ff=session.integral.torque.copy()
            controller=DirectionalController(model,case["variant"],initial_ff[args.joint-1],
                                             args.filter_tau_s,args.slew_rate_nm_s,1/args.tx_hz)
            started=previous_update=time.monotonic()
            total=args.initial_hold_s+len(case["plan"])*args.hold_s
            previous_step=-1
            approach=static.ApproachTracker(math.degrees(session.latest[args.joint]["position"]),args.direction_resolution_deg)
            while time.monotonic()-started<total:
                now=time.monotonic()
                elapsed=now-started
                step=-1 if elapsed<args.initial_hold_s else int((elapsed-args.initial_hold_s)/args.hold_s)
                if step>=len(case["plan"]):break
                targets=list(base)
                if step>=0:
                    if step>previous_step+1:raise RuntimeError("Skipped an entire random target hold")
                    point=case["plan"][step]
                    if step!=previous_step:
                        approach=static.ApproachTracker(math.degrees(session.latest[args.joint]["position"]),args.direction_resolution_deg)
                        print(f"[RANDOM] {case['variant']} {step+1}/{len(case['plan'])} target={point['target_deg']:.3f} dir={point['direction']:+d}",flush=True)
                    direction=approach.direction if args.direction_source=="observed" and approach.direction else point["direction"]
                    model_angle=math.degrees(session.latest[args.joint]["position"])
                    state=controller.update(model_angle,direction,now-previous_update)
                    session.integral.torque[args.joint-1]=state.pop("ff_nm")
                    command_writer.context=dict(state,observed_direction=approach.direction,model_angle_deg=model_angle,
                                                model_feedback_time_s=session.received_at[args.joint]-session.started)
                    targets[args.joint-1]=math.radians(point["target_deg"])
                    _,received=session.tick(targets,step,"random_hold",False)
                else:
                    command_writer.context={}
                    _,received=session.tick(targets,-3,"random_initial_hold",False)
                for row in received:
                    if row["motor_id"]==args.joint:approach.observe(row["position_deg"],row["time_s"])
                    elif abs(row["position_deg"]-defaults[row["motor_id"]-1])>5:
                        raise RuntimeError(f"Other joint J{row['motor_id']} deviated more than 5 deg from its default target")
                previous_step=step
                previous_update=now
            capture_end=time.monotonic()-session.started
            metadata["random_capture_end_s"]=capture_end
            command_writer.context={}
            current=[session.latest[m]["position"] for m in static.MOTOR_IDS]
            ramp_to_default(session,current,base,-2,"return_ramp")
            returned=dict(visit=-2,target_deg=math.degrees(base[args.joint-1]),direction=0)
            metadata["return_to_initial"].update(status="verifying",pose_verification=returned)
            static.measure_point(session,returned,base,phase_prefix="return_",tolerance_deg=args.return_tolerance_deg)
            metadata["return_to_initial"]["status"]="complete"
            metadata["complete"]=True
    except BaseException as error:
        metadata["error"]=f"{type(error).__name__}: {error}"
        if metadata["initialization"]["status"]!="complete":
            metadata["initialization"].update(status="failed",error=metadata["error"])
        if metadata["return_to_initial"]["status"]=="verifying":metadata["return_to_initial"]["status"]="failed"
        raise
    finally:
        try:
            if ctrl is not None:ctrl.close()
        finally:
            end=capture_end if capture_end is not None else (time.monotonic()-session.started if session else 0)
            points=analyze_steps(args,case["plan"],command_writer.selected_rows if command_writer else [],
                                 feedback_writer.selected_rows if feedback_writer else [],end)
            metadata["steps"]=points
            summary=dict(complete=metadata["complete"],error=metadata["error"],
                         return_to_initial_status=metadata["return_to_initial"]["status"],
                         plan_sha256=case["plan_sha256"],model_snapshot_sha256=digest(model.data),
                         variant=case["variant"],trial=case["trial"],**aggregate_steps(points),steps=points)
            if session:
                summary.update(command_cadence=static.cadence(session.tx_times),skipped_command_ticks=session.skipped_tx,
                               feedback_cadence={str(m):static.cadence(values) for m,values in session.packet_times.items()})
            active=[row for row in (command_writer.selected_rows if command_writer else []) if row["phase"]=="random_hold"]
            summary["compensation_diagnostics"]=dict(
                packets=len(active),model_clamped_packets=sum(row["model_clamped"] for row in active),
                model_low_support_packets=sum(row["model_low_support"] for row in active),
                saturated_packets=sum(row["ff_saturated"] for row in active),
                initial_ff_nm=metadata["initialization"].get("initial_ff_nm"),
                initial_pose_deg=metadata["initialization"].get("measured_pose_deg"))
            save_json(directory/"run.json",metadata)
            save_json(directory/"summary.json",summary)
            print(f"[RESULT] {directory}",flush=True)
    return summary


def write_comparison(root, metadata):
    matched=[]
    for trial in sorted({case["trial"] for case in metadata["cases"]}):
        group=[case for case in metadata["cases"] if case["trial"]==trial]
        if len(group)!=3 or any(case["status"]!="complete" for case in group):continue
        summaries={case["variant"]:json.loads((root/case["directory"]/"summary.json").read_text()) for case in group}
        if any(not s["complete"] or s["return_to_initial_status"]!="complete" or
               s["model_snapshot_sha256"]!=metadata["model_snapshot_sha256"] for s in summaries.values()):
            raise ValueError("A matched group has incomplete return or a different model snapshot")
        hashes={summary["plan_sha256"] for summary in summaries.values()}
        if len(hashes)!=1:raise ValueError("Cannot compare different random target sequences")
        for step in range(metadata["settings"]["points"]):
            values={variant:summaries[variant]["steps"][step] for variant in VARIANTS}
            if not all(value["valid"] for value in values.values()):continue
            matched.append(dict(trial=trial,step=step,**{variant:values[variant] for variant in VARIANTS}))
    modes={variant:aggregate_steps([row[variant] for row in matched]) for variant in VARIANTS}
    baseline=modes["fixed"]["steady_mae_deg"]
    improvements={variant:dict(steady_mae_reduction_deg=baseline-modes[variant]["steady_mae_deg"],
                                steady_mae_reduction_fraction=1-modes[variant]["steady_mae_deg"]/baseline if baseline else None)
                  for variant in ("midpoint","directional") if baseline is not None}
    result=dict(complete=metadata["complete"],matched_steps=len(matched),variants=modes,improvements_vs_fixed=improvements,
                baseline="Frozen initial-pose feedforward; not zero-torque PD-only",
                certified_40ms=False,note="Use identical matched steps only. Settling medians exclude unsettled steps; inspect pass fractions.")
    save_json(root/"comparison.json",result)
    with (root/"paired_steps.csv").open("w",newline="") as stream:
        fields=["trial","step","target_deg"]+[f"{v}_{k}" for v in VARIANTS for k in ("steady_mae_deg","steady_error_median_deg","sustained_settling_ms","overshoot_deg")]
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for row in matched:
            writer.writerow(dict(trial=row["trial"],step=row["step"],target_deg=row["fixed"]["target_deg"],
                                 **{f"{v}_{k}":row[v][k] for v in VARIANTS for k in ("steady_mae_deg","steady_error_median_deg","sustained_settling_ms","overshoot_deg")}))
    def number(value, precision=3):
        return f"{value:.{precision}f}" if value is not None else "—"
    lines="\n".join(
        f"| {variant} | {value['valid_steps']} | {number(value['steady_mae_deg'])} | "
        f"{number(value['steady_pass_fraction'])} | {number(value['settled_fraction'])} | "
        f"{number(value['settling_median_ms'],1)} | {number(value['max_overshoot_deg'])} |"
        for variant,value in modes.items())
    changes="\n".join(f"- {variant}：稳态MAE减少 {number(value['steady_mae_reduction_deg'])}°，"
                       f"相对减少 {number(100*value['steady_mae_reduction_fraction'],1) if value['steady_mae_reduction_fraction'] is not None else '—'}%。"
                       for variant,value in improvements.items())
    text=f"""# 随机位置控制对照

{metadata['settings']['leg']} J{metadata['settings']['joint']}，完整匹配停点 {len(matched)} 个，实验完成状态 `{metadata['complete']}`。
三个模式每轮重放相同随机目标、保持时间和增益。仅纳入三个模式均完成并复位、且反馈窗口有效的配对停点。

- `fixed`：初始姿态预热得到的前馈随后冻结，不是零前馈PD。
- `midpoint`：只查候选中点L(q)，替换所测关节的预热前馈。
- `directional`：查L(q)+sH(q)，替换所测关节的预热前馈，不重复叠加。
- 其他四个关节保持默认目标和各自冻结前馈；每组前重新预热，每组后整腿回默认并在{metadata['settings']['return_tolerance_deg']}°内验证。

| 模式 | 配对有效点 | 稳态MAE° | 稳态全部包在容差内比例 | 持续到位比例 | 持续到位中位ms | 最大过冲° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{lines}

{changes or '尚无完整三模式配对，不计算改进。'}

稳态统计取每点末尾 `{metadata['settings']['steady_s']}` 秒，逐点等权；持续到位要求最后一次离开误差带后至少保持 `{metadata['settings']['settle_s']}` 秒。
到位时间中位数仅计算已到位点，必须结合到位比例解读；负的“减少”表示退步，未把失败点强行赋值为成功。
误差带为±{metadata['settings']['tolerance_deg']}°，时间以实际首个新目标发送时间和主机收包时间计算，不能认证设备40 ms到位。
目标采用有最小/最大步幅限制的随机游走，不是全范围独立均匀采样；目标实际覆盖跨度见 `experiment.json.procedure.target_spans`。
默认方向为目标阶跃方向并在保持期间锁存；`--direction_source observed`改为角度历史识别，运动尚未识别时用目标方向提示。
目标方向不保证等于静态标定的最终实测到达方向，日志同时保存实测方向与反转；方向识别和滤波/限速本身也是实验配置。
模型范围外端点夹紧、低支持和限幅包数保存在各组 `summary.json.compensation_diagnostics`；初始化姿态与冻结前馈也单独记录。

组顺序每轮轮换，减少固定先后顺序影响；每组独立预热的实际姿态和前馈仍可能不同，需结合日志判断因果限制。
发生反馈失效/边界中止时保存部分数据并关闭串口，不自动回程；未完成组不计入配对。
本报告不调整候选表、PD增益或误差门槛。

原始数据：每组 `commands.csv` / `feedback.csv` / `run.json`；逐点结果：`summary.json.steps`；配对结果：`paired_steps.csv` / `comparison.json`。
"""
    (root/"COMPARISON.md").write_text(text,encoding="utf-8")
    if matched:
        plot_comparison(root,metadata)
        print("[COMPARE] "+", ".join(f"{variant} steady_MAE={number(value['steady_mae_deg'])} deg" for variant,value in modes.items()),flush=True)
    return result


def plot_comparison(root,metadata):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    trials=[trial for trial in sorted({case["trial"] for case in metadata["cases"]})
            if all(case["status"]=="complete" for case in metadata["cases"] if case["trial"]==trial)]
    fig,axes=plt.subplots(len(trials),2,figsize=(14,4*len(trials)),squeeze=False,constrained_layout=True)
    for line,trial in enumerate(trials):
        for variant,color in zip(VARIANTS,("#777777","#cb7d29","#227fa3")):
            case=next(case for case in metadata["cases"] if case["trial"]==trial and case["variant"]==variant)
            directory=root/case["directory"]
            with (directory/"commands.csv").open() as stream:
                commands=[row for row in csv.DictReader(stream) if row["phase"]=="random_hold" and int(row["motor_id"])==metadata["settings"]["joint"]]
            start=float(commands[0]["time_s"])
            with (directory/"feedback.csv").open() as stream:
                feedback=[row for row in csv.DictReader(stream) if row["phase"]=="random_hold" and int(row["motor_id"])==metadata["settings"]["joint"]]
            times=[float(row["time_s"])-start for row in feedback]
            axes[line,0].plot(times,[float(row["position_deg"]) for row in feedback],color=color,label=variant,lw=1)
            axes[line,1].plot(times,[float(row["target_deg"])-float(row["position_deg"]) for row in feedback],color=color,label=variant,lw=.8)
            if variant=="fixed":
                axes[line,0].step([float(row["time_s"])-start for row in commands],[float(row["target_deg"]) for row in commands],where="post",color="black",lw=.7,label="Target")
        for ax in axes[line]:
            ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel("Time since first random target (s)")
        axes[line,0].set(title=f"Trial {trial}: identical random targets",ylabel="Hardware joint angle (deg)")
        axes[line,1].set(title=f"Trial {trial}: observed tracking errors",ylabel="Target - measured (deg)")
        axes[line,1].axhline(metadata["settings"]["tolerance_deg"],color="red",ls="--",lw=.6)
        axes[line,1].axhline(-metadata["settings"]["tolerance_deg"],color="red",ls="--",lw=.6)
    fig.savefig(root/"comparison.png",dpi=160)
    fig.savefig(root/"comparison.pdf")
    plt.close(fig)


def run(args, adapter=None):
    resolver=adapter.resolve_model if adapter else resolve_model
    preparer=adapter.prepare if adapter else prepare
    group_runner=adapter.run_group if adapter else run_group
    comparer=adapter.write_comparison if adapter else write_comparison
    restore=adapter.from_snapshot if adapter else DirectionalTable
    if args.analyze_dir:
        root=Path(args.analyze_dir)
        return comparer(root,json.loads((root/"experiment.json").read_text()))
    resuming=bool(args.resume_dir)
    if resuming:
        root=Path(args.resume_dir).resolve()
        metadata=json.loads((root/"experiment.json").read_text())
        enabled=args.enable_motors
        args=argparse.Namespace(**metadata["settings"])
        args.enable_motors=enabled
        model=restore(metadata["model_snapshot"])
        model.validate_identity(args.leg,args.joint)
        static_args,cases,description=preparer(args,model)
        if digest([c["plan_sha256"] for c in cases])!=metadata["all_plans_sha256"]:
            raise ValueError("Random plans changed; refusing to resume a different comparison")
        cases=metadata["cases"]
    else:
        model,source=resolver(args)
        static_args,cases,description=preparer(args,model)
        stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        prefix=adapter.output_prefix(args) if adapter else f"random_compare_{args.leg}_j{args.joint}"
        root=Path(args.output_dir) if args.output_dir else REPO/"logs/real_v3"/f"{prefix}_{stamp}"
        metadata=dict(settings=vars(args).copy(),procedure=description,cases=cases,complete=False,error=None,
                      model_snapshot=model.data,model_source=source,
                      model_snapshot_sha256=digest(model.data),all_plans_sha256=digest([c["plan_sha256"] for c in cases]))
    pending=[case for case in cases if case["status"]!="complete"]
    for case in cases:
        if case["status"]!="complete":continue
        summary=json.loads((root/case["directory"]/"summary.json").read_text())
        if (not summary["complete"] or summary["return_to_initial_status"]!="complete" or
                summary["plan_sha256"]!=case["plan_sha256"] or summary["model_snapshot_sha256"]!=digest(model.data)):
            raise ValueError("Previously completed group is not verified")
    print(json.dumps(dict(description,pending_groups=len(pending),
                          plans=[dict(trial=case["trial"],plan_sha256=case["plan_sha256"],targets_deg=[p["target_deg"] for p in case["plan"]])
                                 for case in cases if case["variant"]=="fixed"]),ensure_ascii=False,indent=2))
    if not args.enable_motors:
        print("[PREVIEW] No serial port opened and no logs created.")
        return description
    if not pending:return comparer(root,metadata)
    if not resuming:
        # Stored settings use resume_dir=None; output and source snapshots remain fixed.
        root.mkdir(parents=True,exist_ok=False)
    else:
        stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        (root/f"experiment_before_resume_{stamp}.json").write_text((root/"experiment.json").read_text(),encoding="utf-8")
    metadata.update(complete=False,error=None)
    save_json(root/"experiment.json",metadata)
    current=None
    try:
        for current in pending:
            original=current["directory"]
            name,attempt=original,0
            while (root/name).exists():
                attempt+=1;name=f"{original}_retry_{attempt:02d}"
            if name!=original:
                current.setdefault("previous_attempts",[]).append(dict(directory=original,status=current["status"],error=current.get("error")))
            current.pop("error",None)
            current.update(directory=name,status="running")
            save_json(root/"experiment.json",metadata)
            print(f"[GROUP] {args.leg} {description.get('scope_label',f'J{args.joint}')} trial {current['trial']} {current['variant']}",flush=True)
            summary=group_runner(args,static_args,current,model,root/name)
            if not summary["complete"] or summary["return_to_initial_status"]!="complete":
                raise RuntimeError("Group or full-leg return did not complete")
            current["status"]="complete"
            save_json(root/"experiment.json",metadata)
        metadata["complete"]=True
    except BaseException as error:
        metadata["error"]=f"{type(error).__name__}: {error}"
        if current and current["status"]=="running":current.update(status="failed",error=metadata["error"])
        raise
    finally:
        save_json(root/"experiment.json",metadata)
        comparer(root,metadata)
        print(f"[EXPERIMENT] {root.resolve()}",flush=True)
    return root


def build_parser(*,whole_leg=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg",choices=tuple(static.LEG_PORTS),default="LB")
    if whole_leg:
        parser.set_defaults(joint=1,model=None,min_deg=None,max_deg=None)
    else:
        parser.add_argument("--joint",type=int,choices=static.MOTOR_IDS,default=1)
    parser.add_argument("--catalog",default=str(CATALOG))
    if not whole_leg:
        parser.add_argument("--model",help="Optional explicit candidate; otherwise use frozen catalog")
    parser.add_argument("--port")
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--points",type=int,default=20)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--hold_s",type=float,default=2.)
    parser.add_argument("--initial_hold_s",type=float,default=2.)
    parser.add_argument("--steady_s",type=float,default=.5)
    parser.add_argument("--settle_s",type=float,default=.2)
    parser.add_argument("--tolerance_deg",type=float,default=.5)
    parser.add_argument("--min_step_deg",type=float,default=2.)
    parser.add_argument("--max_step_deg",type=float,default=10.)
    if not whole_leg:
        parser.add_argument("--min_deg",type=float)
        parser.add_argument("--max_deg",type=float)
    parser.add_argument("--range_margin_deg",type=float,default=2.)
    parser.add_argument("--direction_source",choices=("target","observed"),default="target",
                        help="Target: hold latest target-step direction; observed: encoder history with target hint before motion")
    parser.add_argument("--direction_resolution_deg",type=float,default=.2)
    parser.add_argument("--filter_tau_s",type=float,default=.05)
    parser.add_argument("--slew_rate_nm_s",type=float,default=10.)
    parser.add_argument("--tx_hz",type=float,default=100.)
    parser.add_argument("--poll_hz",type=float,default=200.)
    parser.add_argument("--feedback_timeout_s",type=float,default=.2)
    parser.add_argument("--point_timeout_s",type=float,default=90.)
    parser.add_argument("--return_tolerance_deg",type=float,default=2.)
    parser.add_argument("--init_speed_deg_s",type=float,default=5.)
    parser.add_argument("--init_range_deg",type=float,default=30.)
    parser.add_argument("--monitor_margin_deg",type=float,default=1.)
    parser.add_argument("--integral_torque_limit",type=float,nargs=5,default=static.STATIC_DEFAULT_INTEGRAL_LIMIT)
    parser.add_argument("--kp",type=float,nargs=5)
    parser.add_argument("--kd",type=float,nargs=5)
    parser.add_argument("--output_dir")
    continuation=parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume_dir",help="Resume saved experiment, reusing its plans, settings and model snapshot")
    continuation.add_argument("--analyze_dir",help="Rebuild paired comparison from completed group summaries without hardware")
    parser.add_argument("--enable_motors",action="store_true")
    return parser


def main():
    parser=build_parser()
    try:run(parser.parse_args())
    except ValueError as error:parser.error(str(error))


if __name__=="__main__":
    main()
