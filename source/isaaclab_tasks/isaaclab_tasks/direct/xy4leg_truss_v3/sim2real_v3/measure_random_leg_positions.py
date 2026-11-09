"""Five-joint random-pose comparison on one leg; preview unless --enable_motors.

Five targets change in one control cycle; serial motor packets remain sequential.
Default poses stay within each default joint angle +/-10 deg and candidate support.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import time
import types

import measure_random_joint_positions as single
from directional_compensation import DirectionalController, DirectionalTable

static=single.static


class LegBundle:
    def __init__(self,data):
        if data.get("model_type")!="leg_directional_candidate_bundle" or data.get("schema_version")!=1:
            raise ValueError("Expected a whole-leg candidate bundle, not a single-joint experiment")
        if set(data["models"])!={str(m) for m in static.MOTOR_IDS}:
            raise ValueError("Exactly five candidates required")
        self.data=data
        self.tables={m:DirectionalTable(data["models"][str(m)]) for m in static.MOTOR_IDS}
        self.validate_identity(data["leg"])

    def validate_identity(self,leg,joint=None):
        if leg!=self.data["leg"]:raise ValueError("Candidate bundle belongs to another leg")
        for motor,table in self.tables.items():table.validate_identity(leg,motor)


def resolve_model(args):
    models,sources={},{}
    for motor in static.MOTOR_IDS:
        local=argparse.Namespace(**vars(args));local.joint=motor;local.model=None
        table,source=single.resolve_model(local)
        models[str(motor)],sources[str(motor)]=table.data,source
    return LegBundle(dict(schema_version=1,model_type="leg_directional_candidate_bundle",leg=args.leg,models=models)),sources


def prepare(args,bundle):
    if not math.isfinite(args.pose_range_deg) or not 0<args.pose_range_deg<=30:
        raise ValueError("pose_range_deg must be in (0,30] deg")
    defaults=[math.degrees(x) for x in static._target_rad_for_leg(args.leg,True)]
    individual,bounds={},{}
    static_args=None
    for motor,table in bundle.tables.items():
        local=argparse.Namespace(**vars(args))
        local.joint=motor
        local.seed=args.seed+1009*(motor-1)
        local.min_deg=max(defaults[motor-1]-args.pose_range_deg,table.angles[0]+args.range_margin_deg)
        local.max_deg=min(defaults[motor-1]+args.pose_range_deg,table.angles[-1]-args.range_margin_deg)
        prepared,cases,description=single.prepare(local,table)
        if static_args is None:
            static_args=prepared
            args.kp,args.kd=local.kp,local.kd
        elif local.kp!=args.kp or local.kd!=args.kd:
            raise ValueError("All five candidates must use the same gain vectors")
        individual[motor]={case["trial"]:case["plan"] for case in cases if case["variant"]=="fixed"}
        bounds[str(motor)]=description["target_bounds_deg"]
    cases=[]
    for trial in range(1,args.trials+1):
        plan=[]
        for step in range(args.points):
            plan.append(dict(step=step,**{key:[individual[m][trial][step][key] for m in static.MOTOR_IDS]
                                           for key in ("previous_deg","target_deg","direction","delta_deg")}))
        order=single.VARIANTS[(trial-1)%3:]+single.VARIANTS[:(trial-1)%3]
        for variant in order:
            cases.append(dict(trial=trial,variant=variant,plan=plan,plan_sha256=single.digest(plan),
                              directory=f"trial_{trial:02d}_{variant}",status="planned"))
    description=dict(scope="whole_leg",scope_label="J1-J5",leg=args.leg,trials=args.trials,groups=len(cases),
                     points_per_trial=args.points,pose_range_deg=args.pose_range_deg,
                     min_step_deg=args.min_step_deg,max_step_deg=args.max_step_deg,
                     hold_s=args.hold_s,default_pose_deg=defaults,target_bounds_deg=bounds,
                     kp=list(args.kp),kd=list(args.kd),variants=list(single.VARIANTS),
                     feedforward_limit_nm=[bundle.tables[m].limit for m in static.MOTOR_IDS],
                     filter_tau_s=args.filter_tau_s,slew_rate_nm_s=args.slew_rate_nm_s,
                     random_hold_total_s=len(cases)*(args.initial_hold_s+args.points*args.hold_s),
                     seed_by_joint={str(m):args.seed+1009*(m-1) for m in static.MOTOR_IDS},
                     direction_source=args.direction_source,
                     model_review_flags={str(m):table.data.get("review_flags",[]) for m,table in bundle.tables.items()},
                     note="Independent bounded random walk per joint; same five-dimensional targets across variants. "
                          "Targets update in one host cycle, motor packets are sequential. "
                          "Independent one-dimensional tables do not identify full-pose coupling.")
    return static_args,cases,description


class CommandWriter:
    def __init__(self,writer,variant):
        self.writer,self.variant=writer,variant
        self.contexts={}
        self.rows={m:[] for m in static.MOTOR_IDS}

    def writerow(self,row):
        context={field:0 for field in single.CONTROL_FIELDS}
        context["variant"]=self.variant
        context.update(self.contexts.get(row["motor_id"],{}))
        value=dict(row,**context)
        self.writer.writerow(value)
        self.rows[row["motor_id"]].append(value)


class FeedbackWriter:
    def __init__(self,writer):
        self.writer=writer
        self.rows={m:[] for m in static.MOTOR_IDS}

    def writerow(self,row):
        self.writer.writerow(row)
        self.rows[row["motor_id"]].append(dict(row))


def pose_steps(joints,args):
    poses=[]
    for step in range(args.points):
        values=[joints[str(m)]["steps"][step] for m in static.MOTOR_IDS]
        if not all(value.get("valid") for value in values):
            poses.append(dict(step=step,valid=False,reason="At least one joint has incomplete feedback"))
            continue
        sent=max(value["first_command_s"] for value in values)
        end=min(value["end_s"] for value in values)
        settling=None
        if all(value["sustained_settling_ms"] is not None for value in values):
            last=max(value["first_command_s"]+value["sustained_settling_ms"]/1000 for value in values)
            if end-last>=args.settle_s:
                settling=max(0.,last-sent)*1000
        poses.append(dict(step=step,valid=True,target_deg=[value["target_deg"] for value in values],
                          all_five_steady_within=all(value["steady_all_within_tolerance"] for value in values),
                          mean_axis_steady_mae_deg=st.mean(value["steady_mae_deg"] for value in values),
                          worst_axis_steady_mae_deg=max(value["steady_mae_deg"] for value in values),
                          sustained_settling_ms=settling,
                          first_axis_send_skew_ms=(sent-min(value["first_command_s"] for value in values))*1000))
    return poses


def aggregate_poses(poses):
    valid=[pose for pose in poses if pose.get("valid")]
    settled=[pose["sustained_settling_ms"] for pose in valid if pose["sustained_settling_ms"] is not None]
    return dict(valid_poses=len(valid),mean_axis_steady_mae_deg=st.mean(p["mean_axis_steady_mae_deg"] for p in valid) if valid else None,
                all_five_steady_pass_fraction=st.mean(p["all_five_steady_within"] for p in valid) if valid else None,
                all_five_settled_fraction=len(settled)/len(valid) if valid else None,
                all_five_settling_median_ms=st.median(settled) if settled else None,
                max_first_axis_send_skew_ms=max((p["first_axis_send_skew_ms"] for p in valid),default=None))


def run_group(args,static_args,case,bundle,directory):
    directory.mkdir(parents=True,exist_ok=False)
    metadata=dict(scope="whole_leg",settings=vars(args),case={k:v for k,v in case.items() if k not in ("status","error","previous_attempts")},
                  model_snapshot=bundle.data,model_snapshot_sha256=single.digest(bundle.data),
                  plan_sha256=case["plan_sha256"],complete=False,error=None,
                  initialization=dict(status="not_started"),return_to_initial=dict(status="not_started"),
                  certified_40ms=False,experimental_model_application=case["variant"]!="fixed")
    ctrl=session=None
    command_writer=feedback_writer=None
    capture_end=None
    base=static._target_rad_for_leg(args.leg,True)
    defaults=[math.degrees(x) for x in base]
    bounds={m:(q-args.init_range_deg-args.monitor_margin_deg,q+args.init_range_deg+args.monitor_margin_deg)
            for m,q in enumerate(defaults,1)}
    try:
        with (directory/"commands.csv").open("x",newline="") as command_stream,(directory/"feedback.csv").open("x",newline="") as feedback_stream:
            commands=csv.DictWriter(command_stream,static.COMMAND_FIELDS+single.CONTROL_FIELDS)
            feedback=csv.DictWriter(feedback_stream,static.FEEDBACK_FIELDS)
            commands.writeheader();feedback.writeheader()
            command_writer,feedback_writer=CommandWriter(commands,case["variant"]),FeedbackWriter(feedback)
            single.save_json(directory/"run.json",metadata)
            ctrl=static.DaMiaoMotorController(args.port or static.LEG_PORTS[args.leg])
            current=[static._enable_and_read_current(ctrl,m) for m in static.MOTOR_IDS]
            if any(x is None for x in current):raise RuntimeError("Missing startup position")
            metadata["initialization"]["measured_start_deg"]=[math.degrees(x) for x in current]
            if any(not bounds[m][0]<=math.degrees(q)<=bounds[m][1] for m,q in zip(static.MOTOR_IDS,current)):
                raise RuntimeError("Startup outside initialization monitoring bounds; no pose command sent")
            session=static.Session(static_args,ctrl,command_writer,feedback_writer)
            session.monitor_bounds_deg=bounds
            single.ramp_to_default(session,current,base,-1,"warmup_ramp")
            warmup=dict(visit=-1,target_deg=defaults[0],direction=0)
            metadata["initialization"].update(status="verifying",pose_verification=warmup)
            static.measure_point(session,warmup,base,phase_prefix="warmup_",tolerance_deg=.8)
            initial=session.integral.torque.copy()
            metadata["initialization"].update(status="complete",initial_ff_nm=initial,
                                              measured_pose_deg=[math.degrees(session.latest[m]["position"]) for m in static.MOTOR_IDS])
            session.monitor_bounds_deg={m:(q-30-args.monitor_margin_deg,q+30+args.monitor_margin_deg) for m,q in enumerate(defaults,1)}
            controllers={m:DirectionalController(bundle.tables[m],case["variant"],initial[m-1],args.filter_tau_s,
                                                  args.slew_rate_nm_s,1/args.tx_hz) for m in static.MOTOR_IDS}
            trackers={m:static.ApproachTracker(math.degrees(session.latest[m]["position"]),args.direction_resolution_deg) for m in static.MOTOR_IDS}
            started=previous_update=time.monotonic()
            duration=args.initial_hold_s+args.points*args.hold_s
            previous_step=-1
            while time.monotonic()-started<duration:
                now=time.monotonic()
                elapsed=now-started
                step=-1 if elapsed<args.initial_hold_s else int((elapsed-args.initial_hold_s)/args.hold_s)
                if step>=args.points:break
                targets=list(base)
                if step>=0:
                    if step>previous_step+1:raise RuntimeError("Skipped an entire random pose")
                    point=case["plan"][step]
                    if step!=previous_step:
                        trackers={m:static.ApproachTracker(math.degrees(session.latest[m]["position"]),args.direction_resolution_deg) for m in static.MOTOR_IDS}
                        print(f"[POSE] {case['variant']} {step+1}/{args.points}: {[round(q,3) for q in point['target_deg']]}",flush=True)
                    for motor in static.MOTOR_IDS:
                        direction=trackers[motor].direction if args.direction_source=="observed" and trackers[motor].direction else point["direction"][motor-1]
                        angle=math.degrees(session.latest[motor]["position"])
                        state=controllers[motor].update(angle,direction,now-previous_update)
                        session.integral.torque[motor-1]=state.pop("ff_nm")
                        command_writer.contexts[motor]=dict(state,observed_direction=trackers[motor].direction,
                                                            model_angle_deg=angle,model_feedback_time_s=session.received_at[motor]-session.started)
                    targets=[math.radians(q) for q in point["target_deg"]]
                    _,received=session.tick(targets,step,"random_hold",False)
                else:
                    command_writer.contexts={}
                    _,received=session.tick(targets,-3,"random_initial_hold",False)
                for row in received:trackers[row["motor_id"]].observe(row["position_deg"],row["time_s"])
                previous_update,previous_step=now,step
            capture_end=time.monotonic()-session.started
            metadata["random_capture_end_s"]=capture_end
            command_writer.contexts={}
            single.ramp_to_default(session,[session.latest[m]["position"] for m in static.MOTOR_IDS],base,-2,"return_ramp")
            returned=dict(visit=-2,target_deg=defaults[0],direction=0)
            metadata["return_to_initial"].update(status="verifying",pose_verification=returned)
            static.measure_point(session,returned,base,phase_prefix="return_",tolerance_deg=args.return_tolerance_deg)
            metadata["return_to_initial"]["status"]="complete"
            metadata["complete"]=True
    except BaseException as error:
        metadata["error"]=f"{type(error).__name__}: {error}"
        if metadata["initialization"]["status"]!="complete":metadata["initialization"]["status"]="failed"
        if metadata["return_to_initial"]["status"]=="verifying":metadata["return_to_initial"]["status"]="failed"
        raise
    finally:
        try:
            if ctrl is not None:ctrl.close()
        finally:
            end=capture_end if capture_end is not None else (time.monotonic()-session.started if session else 0)
            joints={}
            for motor in static.MOTOR_IDS:
                plan=[dict(step=p["step"],**{k:p[k][motor-1] for k in ("target_deg","previous_deg","direction")}) for p in case["plan"]]
                rows=single.analyze_steps(args,plan,command_writer.rows[motor] if command_writer else [],feedback_writer.rows[motor] if feedback_writer else [],end)
                joints[str(motor)]=dict(**single.aggregate_steps(rows),steps=rows)
            poses=pose_steps(joints,args)
            metadata["joint_steps"],metadata["poses"]=joints,poses
            summary=dict(scope="whole_leg",complete=metadata["complete"],error=metadata["error"],variant=case["variant"],trial=case["trial"],
                         return_to_initial_status=metadata["return_to_initial"]["status"],plan_sha256=case["plan_sha256"],
                         model_snapshot_sha256=single.digest(bundle.data),joints=joints,poses=poses,whole_leg=aggregate_poses(poses))
            summary["compensation_diagnostics"]={}
            for motor in static.MOTOR_IDS:
                active=[r for r in command_writer.rows[motor] if r["phase"]=="random_hold"] if command_writer else []
                summary["compensation_diagnostics"][str(motor)]=dict(
                    packets=len(active),model_clamped_packets=sum(r["model_clamped"] for r in active),
                    model_low_support_packets=sum(r["model_low_support"] for r in active),
                    saturated_packets=sum(r["ff_saturated"] for r in active))
            if session:summary.update(command_cadence=static.cadence(session.tx_times),skipped_command_ticks=session.skipped_tx)
            single.save_json(directory/"run.json",metadata)
            single.save_json(directory/"summary.json",summary)
            print(f"[RESULT] {directory}",flush=True)
    return summary


def write_comparison(root,metadata):
    matched=[]
    for trial in sorted({case["trial"] for case in metadata["cases"]}):
        cases=[case for case in metadata["cases"] if case["trial"]==trial]
        if len(cases)!=3 or any(case["status"]!="complete" for case in cases):continue
        summaries={case["variant"]:json.loads((root/case["directory"]/"summary.json").read_text()) for case in cases}
        if len({s["plan_sha256"] for s in summaries.values()})!=1 or any(
                not s["complete"] or s["scope"]!="whole_leg" or s["return_to_initial_status"]!="complete" or
                s["model_snapshot_sha256"]!=metadata["model_snapshot_sha256"] for s in summaries.values()):
            raise ValueError("Cannot compare incomplete whole-leg groups or different plans/models")
        for step in range(metadata["settings"]["points"]):
            if all(s["poses"][step]["valid"] for s in summaries.values()):
                matched.append(dict(trial=trial,step=step,summaries=summaries))
    modes={}
    for variant in single.VARIANTS:
        poses=[row["summaries"][variant]["poses"][row["step"]] for row in matched]
        joints={str(m):single.aggregate_steps([row["summaries"][variant]["joints"][str(m)]["steps"][row["step"]] for row in matched]) for m in static.MOTOR_IDS}
        modes[variant]=dict(joints=joints,whole_leg=aggregate_poses(poses))
    result=dict(scope="whole_leg",complete=metadata["complete"],matched_poses=len(matched),variants=modes,certified_40ms=False,
                note="Each joint is scored separately; pose success requires all five. Independent 1D tables do not model full-pose coupling.")
    single.save_json(root/"comparison.json",result)
    with (root/"paired_joint_steps.csv").open("w",newline="") as stream:
        keys=["trial","pose","joint","target_deg"]+[f"{v}_{k}" for v in single.VARIANTS for k in ("steady_mae_deg","steady_error_median_deg","sustained_settling_ms","overshoot_deg")]
        writer=csv.DictWriter(stream,keys);writer.writeheader()
        for row in matched:
            for motor in static.MOTOR_IDS:
                values={v:row["summaries"][v]["joints"][str(motor)]["steps"][row["step"]] for v in single.VARIANTS}
                writer.writerow(dict(trial=row["trial"],pose=row["step"],joint=motor,target_deg=values["fixed"]["target_deg"],
                                     **{f"{v}_{k}":values[v][k] for v in single.VARIANTS for k in ("steady_mae_deg","steady_error_median_deg","sustained_settling_ms","overshoot_deg")}))
    def n(value):return f"{value:.3f}" if value is not None else "—"
    table="\n".join(f"| {variant} | "+" | ".join(n(value["joints"][str(m)]["steady_mae_deg"]) for m in static.MOTOR_IDS)+
                    f" | {n(value['whole_leg']['all_five_steady_pass_fraction'])} | {n(value['whole_leg']['all_five_settled_fraction'])} |"
                    for variant,value in modes.items())
    text=f"""# 一条腿五轴同步随机姿态对照

{metadata['settings']['leg']}，完整三模式匹配姿态数{len(matched)}，每姿态包含五轴独立目标。
`fixed`冻结全部五轴初始预热前馈；`midpoint`五轴各自应用L(q)；`directional`五轴各自应用L(q)+sH(q)。
随机运动期间全部位置积分冻结，各组后整腿回默认并验证复位。

| 模式 | J1稳态MAE° | J2 | J3 | J4 | J5 | 整姿态五轴均达标比例 | 五轴持续到位比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

各轴误差带±{metadata['settings']['tolerance_deg']}°，尾窗{metadata['settings']['steady_s']}秒；整姿态要求五轴全部满足。
所有指标只纳入三模式都完整并复位、且五轴尾窗有效的同一姿态，不能用平均误差掩盖个别轴失败。
五个目标在同一主机控制周期更新，串口包仍按电机顺序发送；每姿态实际首包时间差保存在 `summary.json.poses.first_axis_send_skew_ms`，不声称硬件严格同时到达。
初始化采用五轴均0.8°预热验证；返回容差{metadata['settings']['return_tolerance_deg']}°。
本测试的目标组合偏离单轴标定时其他关节默认姿态，五条独立表没有识别完整姿态耦合，补偿效果需要本次数据验证。
只记录关节角精度，未用这些指标替代末端笛卡尔精度，不能认证设备40ms到位。
日志保留每轴的方向、实际查表角、前馈分量、夹紧/低支持/限幅标记和实际位置。
每组单独预热的实际完整姿态和力矩仍可能不同；查看 `run.json.initialization`。
`comparison.json`、`paired_joint_steps.csv`给出逐轴和整腿配对结果，原始CSV包含全部五电机。
"""
    (root/"COMPARISON.md").write_text(text,encoding="utf-8")
    if matched:plot_comparison(root,metadata)
    return result


def plot_comparison(root,metadata):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:return
    trials=[t for t in sorted({c["trial"] for c in metadata["cases"]}) if all(c["status"]=="complete" for c in metadata["cases"] if c["trial"]==t)]
    for trial in trials:
        fig,axes=plt.subplots(5,2,figsize=(15,16),constrained_layout=True)
        for variant,color in zip(single.VARIANTS,("#777777","#cf842b","#247fa3")):
            case=next(c for c in metadata["cases"] if c["trial"]==trial and c["variant"]==variant)
            with (root/case["directory"]/"commands.csv").open() as stream:
                commands=[r for r in csv.DictReader(stream) if r["phase"]=="random_hold"]
            with (root/case["directory"]/"feedback.csv").open() as stream:
                feedback=[r for r in csv.DictReader(stream) if r["phase"]=="random_hold"]
            start=min(float(r["time_s"]) for r in commands)
            for motor in static.MOTOR_IDS:
                samples=[r for r in feedback if int(r["motor_id"])==motor]
                x=[float(r["time_s"])-start for r in samples]
                axes[motor-1,0].plot(x,[float(r["position_deg"]) for r in samples],color=color,lw=.9,label=variant)
                axes[motor-1,1].plot(x,[float(r["target_deg"])-float(r["position_deg"]) for r in samples],color=color,lw=.8,label=variant)
                if variant=="fixed":
                    target=[r for r in commands if int(r["motor_id"])==motor]
                    axes[motor-1,0].step([float(r["time_s"])-start for r in target],[float(r["target_deg"]) for r in target],where="post",color="black",lw=.7,label="Target")
        for motor in static.MOTOR_IDS:
            axes[motor-1,0].set(title=f"J{motor}: simultaneous five-joint trial {trial}",xlabel="Time (s)",ylabel="Joint angle (deg)")
            axes[motor-1,1].set(title=f"J{motor}: tracking error",xlabel="Time (s)",ylabel="Target - actual (deg)")
            for ax in axes[motor-1]:ax.grid(alpha=.2);ax.legend(fontsize=7)
            for sign in (-1,1):axes[motor-1,1].axhline(sign*metadata["settings"]["tolerance_deg"],color="red",ls="--",lw=.7)
        fig.savefig(root/f"comparison_trial_{trial:02d}.png",dpi=150)
        fig.savefig(root/f"comparison_trial_{trial:02d}.pdf")
        plt.close(fig)


ADAPTER=types.SimpleNamespace(resolve_model=resolve_model,prepare=prepare,run_group=run_group,write_comparison=write_comparison,
                              from_snapshot=LegBundle,output_prefix=lambda args:f"random_leg_compare_{args.leg}")


def run(args):return single.run(args,adapter=ADAPTER)


def build_parser():
    parser=single.build_parser(whole_leg=True)
    parser.description=__doc__
    parser.add_argument("--pose_range_deg",type=float,default=10.,help="Each joint target within default +/- this half-range, intersected with candidate support")
    return parser


def main():
    parser=build_parser()
    try:run(parser.parse_args())
    except ValueError as error:parser.error(str(error))


if __name__=="__main__":main()
