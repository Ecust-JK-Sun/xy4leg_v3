from __future__ import annotations

import threading
import time
import numpy as np
import torch

from threads_shared import SharedData


def start_motor_obs_thread(
    shared: SharedData,
    session,
    device,
    num_envs: int,
    loop_hz: float = 25.0,
    stop_event: threading.Event | None = None,
):
    """单线程：先收电机反馈，再写关节角/速到 shared.obs。"""

    def loop():
        period = 1.0 / loop_hz
        while stop_event is None or not stop_event.is_set():
            start = time.time()
            session.recv_feedback()
            dof_pos_tensor = (
                torch.tensor(session.pos_buf.copy(), device=device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(num_envs, 1)
            )
            dof_vel_tensor = (
                torch.tensor(session.vel_buf.copy(), device=device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(num_envs, 1)
            )
            obs_tensor = torch.cat((dof_pos_tensor, dof_vel_tensor), dim=-1)
            shared.set_obs(obs_tensor[0].cpu().numpy())
            sleep_till = start + period
            remaining = sleep_till - time.time()
            if remaining > 0:
                time.sleep(remaining)
            else:
                print("recv feedback time out")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_serial_tx_thread(
    shared: SharedData,
    motor_enabled: bool,
    session,
    loop_hz: float = 25.0,
    stop_event: threading.Event | None = None,
):
    """发送线程：获取最新动作并直接下发到串口（无历史队列，限幅已在外部完成）。"""

    def loop():
        period = 1.0 / loop_hz
        while stop_event is None or not stop_event.is_set():
            start = time.time()
            act = shared.get_action()
            if motor_enabled and act is not None:
                session.send_targets(act)
            sleep_till = start + period
            remaining = sleep_till - time.time()
            if remaining > 0:
                time.sleep(remaining)
            else:
                print("send_targets time out")

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
