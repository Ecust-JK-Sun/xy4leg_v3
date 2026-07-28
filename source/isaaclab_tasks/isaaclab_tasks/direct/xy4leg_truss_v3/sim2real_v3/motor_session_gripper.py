from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
import numpy as np

from motor_control import DaMiaoMotorController


@dataclass
class MotorSession:
    ctrl_lb: DaMiaoMotorController
    ctrl_lf: DaMiaoMotorController
    ctrl_rb: DaMiaoMotorController
    ctrl_rf: DaMiaoMotorController
    lb_ids: List[int]
    lf_ids: List[int]
    rb_ids: List[int]
    rf_ids: List[int]
    kp_list: List[float]
    kd_list: List[float]
    pos_buf: np.ndarray
    vel_buf: np.ndarray
    enabled: bool = False

    def enable_all(self):
        """显式使能四路电机。"""
        try:
            for mid in self.lb_ids:
                self.ctrl_lb.enable_motor(mid)
                time.sleep(0.1)
            for mid in self.lf_ids:
                self.ctrl_lf.enable_motor(mid)
                time.sleep(0.1)
            for mid in self.rb_ids:
                self.ctrl_rb.enable_motor(mid)
                time.sleep(0.1)
            for mid in self.rf_ids:
                self.ctrl_rf.enable_motor(mid)
                time.sleep(0.1)
            self.enabled = True
        except Exception as e:
            print(f"[ERROR] Failed to enable motors: {e}")
            self.enabled = False

    def send_targets(self, target_pos: np.ndarray):
        """使用 MIT 模式发送目标位置（需事先 enable)。
        支持 24 DOF (6 关节/腿),第6个关节为力矩控制。
        """
        if target_pos.shape[0] < 24:
            return
        # NPZ 顺序: LB, LF, RB, RF (每腿6个关节)
        lb_target = target_pos[0::4]   # [0, 4, 8, 12, 16, 20]
        lf_target = target_pos[1::4]   # [1, 5, 9, 13, 17, 21]
        rb_target = target_pos[2::4]   # [2, 6, 10, 14, 18, 22]
        rf_target = target_pos[3::4]   # [3, 7, 11, 15, 19, 23]
        self._send_leg(self.ctrl_lb, lb_target, self.lb_ids, self.kp_list, self.kd_list)
        self._send_leg(self.ctrl_lf, lf_target, self.lf_ids, self.kp_list, self.kd_list)
        self._send_leg(self.ctrl_rb, rb_target, self.rb_ids, self.kp_list, self.kd_list)
        self._send_leg(self.ctrl_rf, rf_target, self.rf_ids, self.kp_list, self.kd_list)

    def recv_feedback(self):
        """接收各路反馈并写入缓冲（仅接收，不发送）。"""
        self._recv_leg(self.ctrl_lb, self.lb_ids, self.pos_buf, self.vel_buf, start_idx=0)
        self._recv_leg(self.ctrl_lf, self.lf_ids, self.pos_buf, self.vel_buf, start_idx=1)
        self._recv_leg(self.ctrl_rb, self.rb_ids, self.pos_buf, self.vel_buf, start_idx=2)
        self._recv_leg(self.ctrl_rf, self.rf_ids, self.pos_buf, self.vel_buf, start_idx=3)

    def _send_leg(self, ctrl, targets, ids, kp_list, kd_list):
        """发送单腿的目标位置。
        对于 Joint6(索引5),使用力矩控制(pos=0, vel=0, tor=targets[5]）。
        """
        for i, (mid, val, kp_i, kd_i) in enumerate(zip(ids, targets, kp_list, kd_list)):
            try:
                if i == 5:  # Joint6: 力矩控制
                    cmd = ctrl.damiao_mit_command(pos=0.0, vel=0.0, kp=0.0, kd=0.0, tor=float(val), motor_id=mid)
                else:  # Joint1-5: 位置控制
                    cmd = ctrl.damiao_mit_command(pos=float(val), vel=0.0, kp=kp_i, kd=kd_i, tor=0.0, motor_id=mid)
                ctrl.send(cmd)
                time.sleep(0.0002)
            except Exception as e:
                print(f"[WARN] motor {mid} send error: {e}")
                continue

    def _recv_leg(self, ctrl, ids, pos_buf, vel_buf, start_idx: int):
        # indices interleaved: 0,4,8,... with offset start_idx
        id_to_idx = {mid: start_idx + 4 * i for i, mid in enumerate(ids)}
        for mid in ids:
            try:
                resp = ctrl.receive(8)
                time.sleep(0.0002)
                if resp and len(resp) == 8:
                    fb = ctrl.damiao_feedback(resp)
                    if fb:
                        rec_id = fb["motor_id"]
                        if rec_id in id_to_idx:
                            g_idx = id_to_idx[rec_id]
                            pos_buf[g_idx] = fb["position"]
                            vel_buf[g_idx] = fb["velocity"]
            except Exception as e:
                print(f"[WARN] motor {mid} recv error: {e}")
                continue

    def close(self):
        for ctrl in [self.ctrl_lb, self.ctrl_lf, self.ctrl_rb, self.ctrl_rf]:
            try:
                ctrl.close()
            except Exception:
                pass


def create_motor_session(kp_list, kd_list):
    """初始化电机控制，会尝试打开四路串口并返回 session。kp_list/kd_list 为每关节增益。"""
    ctrl_lb = DaMiaoMotorController("/dev/ttyUSB0")
    ctrl_lf = DaMiaoMotorController("/dev/ttyUSB1")
    ctrl_rb = DaMiaoMotorController("/dev/ttyUSB2")
    ctrl_rf = DaMiaoMotorController("/dev/ttyUSB3")

    lb_ids = [1, 2, 3, 4, 5, 6]
    lf_ids = [1, 2, 3, 4, 5, 6]
    rb_ids = [1, 2, 3, 4, 5, 6]
    rf_ids = [1, 2, 3, 4, 5, 6]

    motor_enabled = all([ctrl_lb.ser, ctrl_lf.ser, ctrl_rb.ser, ctrl_rf.ser])
    if not motor_enabled:
        print("[WARN] Some motor controllers failed to initialize. Motor control disabled.")

    pos_buf = np.zeros(24, dtype=np.float32)
    vel_buf = np.zeros(24, dtype=np.float32)

    return motor_enabled, MotorSession(
        ctrl_lb=ctrl_lb,
        ctrl_lf=ctrl_lf,
        ctrl_rb=ctrl_rb,
        ctrl_rf=ctrl_rf,
        lb_ids=lb_ids,
        lf_ids=lf_ids,
        rb_ids=rb_ids,
        rf_ids=rf_ids,
        kp_list=kp_list,
        kd_list=kd_list,
        pos_buf=pos_buf,
        vel_buf=vel_buf,
    )

