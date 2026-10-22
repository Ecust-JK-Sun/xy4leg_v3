from __future__ import annotations

import threading
import time

from threads_shared import SharedData
from runtime_timing import PeriodicDeadline


def start_motor_obs_thread(
    shared: SharedData,
    session,
    device,
    num_envs: int,
    loop_hz: float = 25.0,
    stop_event: threading.Event | None = None,
    stats=None,
):
    """Poll all four ports without missing-packet waits; publish the latest per-motor state."""

    def run():
        schedule = PeriodicDeadline(loop_hz)
        previous_frames = previous_complete = previous_invalid = 0
        while schedule.wait(stop_event):
            start = time.monotonic()
            session.recv_feedback()
            obs, timestamps, valid = session.feedback_snapshot()
            shared.set_obs(obs, sample_times=timestamps, valid=valid)
            if stats is not None:
                communication = session.communication_snapshot()
                stats.tick("rx_poll")
                stats.tick("rx_joint_frames", communication["rx_frames"] - previous_frames)
                stats.tick("rx_complete", communication["rx_complete"] - previous_complete)
                stats.tick("rx_invalid_frames", communication["rx_invalid_frames"] - previous_invalid)
                previous_frames, previous_complete = communication["rx_frames"], communication["rx_complete"]
                previous_invalid = communication["rx_invalid_frames"]
                stats.observe("rx_backlog_bytes", communication["rx_backlog_bytes"])
                stats.observe("rx_pending_bytes", communication["rx_pending_bytes"])
                stats.elapsed_ms("rx_ms", start)

    def loop():
        try:
            run()
        except Exception as exc:
            shared.set_error("motor feedback thread", exc)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_serial_tx_thread(
    shared: SharedData,
    motor_enabled: bool,
    session,
    loop_hz: float = 25.0,
    stop_event: threading.Event | None = None,
    command_timeout_s: float = 0.5,
    feedback_ready_event: threading.Event | None = None,
    feedback_timeout_s: float = 0.5,
    stats=None,
    mode: str = "periodic",
):
    """Send the latest action periodically, or once per publication with mode='on_update'."""

    if mode not in ("periodic", "on_update"):
        raise ValueError(f"Unsupported transmit mode: {mode}")

    def run():
        period = 1.0 / loop_hz
        schedule = PeriodicDeadline(loop_hz)
        previous_sequence = 0
        while stop_event is None or not stop_event.is_set():
            if mode == "periodic":
                if not schedule.wait(stop_event):
                    break
            else:
                shared.wait_for_action(previous_sequence, timeout=period)
            shared.raise_if_failed()
            act, published_at, sequence, observed_at = shared.action_snapshot(max_age_s=command_timeout_s)
            if stop_event is not None and stop_event.is_set():
                break
            if mode == "on_update" and sequence == previous_sequence:
                continue
            if motor_enabled and act is not None:
                if feedback_ready_event is not None and feedback_ready_event.is_set():
                    shared.get_obs(max_age_s=feedback_timeout_s)
                send_started = time.monotonic()
                session.send_targets(act)
                if stats is not None:
                    stats.tick("tx")
                    stats.tick("tx_repeat" if sequence == previous_sequence else "tx_new")
                    stats.observe("command_age_ms", (send_started - published_at) * 1000.)
                    if observed_at is not None:
                        stats.observe("observation_to_tx_ms", (send_started - observed_at) * 1000.)
                    stats.elapsed_ms("tx_ms", send_started)
                    if hasattr(session, "transmit_queue_bytes"):
                        stats.observe("tx_pending_bytes", session.transmit_queue_bytes())
                previous_sequence = sequence

    def loop():
        try:
            run()
        except Exception as exc:
            # Stop software transmission and report; do not send a shutdown action.
            shared.set_error("motor command thread", exc)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
