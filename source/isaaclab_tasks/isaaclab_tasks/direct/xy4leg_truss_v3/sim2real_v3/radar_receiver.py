import socket
import json
import numpy as np
import threading
import time

def quaternion_to_euler(qx, qy, qz, qw):
    try:
        from scipy.spatial.transform import Rotation as R

        r = R.from_quat([qx, qy, qz, qw])
        return r.as_euler('xyz', degrees=True)
    except:
        return [0, 0, 0]

class RadarReceiver:
    def __init__(self, ip='0.0.0.0', port=8888, source_ip=None, stats=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        # 设置为非阻塞模式，方便线程退出
        self.sock.settimeout(0.05)
        self.source_ip = source_ip
        self.stats = stats
        self.last_received_at = None
        self.last_sender_timestamp = None
        self.error = None
        
        self.running = False
        self.lock = threading.Lock()
        
        # 共享数据区 (默认值)
        self.latest_data = {
            "timestamp": 0.0,
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([0.0, 0.0, 0.0, 1.0]), # x, y, z, w
            "lin_vel": np.array([0.0, 0.0, 0.0]),
            "ang_vel": np.array([0.0, 0.0, 0.0])
        }
        
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()
            print(f"[INFO] Radar Receiver started on port {self.sock.getsockname()[1]}")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.sock.close()
        print("[INFO] Radar Receiver stopped")

    def _receive_loop(self):
        while self.running:
            try:
                # 接收数据 (缓冲区大小 1024 字节足够了)
                data_bytes, addr = self.sock.recvfrom(4096)
                if self.source_ip is not None and addr[0] != self.source_ip:
                    continue
                data = self._decode_packet(data_bytes)
                with self.lock:
                    stamp = data["timestamp"]
                    if stamp is not None:
                        if self.last_sender_timestamp is not None and stamp <= self.last_sender_timestamp:
                            if self.stats is not None:
                                self.stats.tick("radar_out_of_order")
                            continue
                        self.last_sender_timestamp = stamp
                    self.latest_data = data
                    self.last_received_at = time.monotonic()
                if self.stats is not None:
                    self.stats.tick("radar")

            except socket.timeout:
                continue
            except (ValueError, TypeError, UnicodeDecodeError, KeyError):
                # Malformed packets never refresh the last valid observation's age.
                if self.stats is not None:
                    self.stats.tick("radar_invalid")
                continue
            except OSError as exc:
                if self.running:
                    self.error = exc
                break

    @staticmethod
    def _decode_packet(payload):
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict) or "q" not in data or "w" not in data:
            raise ValueError("Radar packet must include quaternion q and angular velocity w")
        result = {}
        for key, output, count, default in (("q", "quat", 4, None), ("w", "ang_vel", 3, None),
                                             ("v", "lin_vel", 3, [0, 0, 0]), ("p", "pos", 3, [0, 0, 0])):
            value = np.asarray(data.get(key, default), dtype=np.float32)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid radar field {key}")
            result[output] = value
        norm = np.linalg.norm(result["quat"])
        if norm < 1e-6 or not np.isfinite(norm):
            raise ValueError("Invalid radar quaternion")
        result["quat"] /= norm
        stamp = float(data["t"]) if "t" in data else None
        if stamp is not None and not np.isfinite(stamp):
            raise ValueError("Invalid radar timestamp")
        result["timestamp"] = stamp
        return result

    def get_data(self, max_age_s=None, now=None):
        """获取最新的数据副本"""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Radar receiver failed: {self.error}")
            if max_age_s is not None:
                if self.last_received_at is None:
                    raise RuntimeError("Radar has not received a valid packet")
                age = (time.monotonic() if now is None else now) - self.last_received_at
                if age < 0 or age > max_age_s:
                    raise RuntimeError(f"Radar observation is stale ({age:.3f}s old)")
                if getattr(self, "stats", None) is not None:
                    self.stats.observe("radar_age_ms", age * 1000.)
            return {key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in self.latest_data.items()}

# 保留脚本直接运行的功能用于测试
def main():
    receiver = RadarReceiver()
    receiver.start()
    
    try:
        while True:
            data = receiver.get_data()
            pos = data['pos']
            quat = data['quat']
            vel = data['lin_vel']
            ang = data['ang_vel']
            
            roll, pitch, yaw = quaternion_to_euler(*quat)
            
            print("-" * 60)
            print(f"时间戳: {data['timestamp']}")
            print(f"【位置】 X: {pos[0]:.3f}, Y: {pos[1]:.3f}, Z: {pos[2]:.3f}")
            print(f"【四元数】: {quat}")
            print(f"【姿态】 R: {roll:.2f}, P: {pitch:.2f}, Y: {yaw:.2f}")
            print(f"【线速】 X: {vel[0]:.3f}, Y: {vel[1]:.3f}, Z: {vel[2]:.3f}")
            print(f"【角速】 X: {ang[0]:.3f}, Y: {ang[1]:.3f}, Z: {ang[2]:.3f}")
            
            time.sleep(0.1) # 打印频率限制
            
    except KeyboardInterrupt:
        print("\n停止接收...")
    finally:
        receiver.stop()

if __name__ == '__main__':
    main()
