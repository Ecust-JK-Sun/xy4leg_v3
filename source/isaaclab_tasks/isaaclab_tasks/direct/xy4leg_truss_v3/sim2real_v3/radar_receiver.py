import socket
import json
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(qx, qy, qz, qw):
    try:
        r = R.from_quat([qx, qy, qz, qw])
        return r.as_euler('xyz', degrees=True)
    except:
        return [0, 0, 0]

class RadarReceiver:
    def __init__(self, ip='0.0.0.0', port=8888):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        # 设置为非阻塞模式，方便线程退出
        self.sock.settimeout(0.0001)
        
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
            self.thread.join()
        self.sock.close()
        print("[INFO] Radar Receiver stopped")

    def _receive_loop(self):
        while self.running:
            try:
                # 接收数据 (缓冲区大小 1024 字节足够了)
                data_bytes, addr = self.sock.recvfrom(1024)
                
                json_str = data_bytes.decode('utf-8')
                data = json.loads(json_str)

                # 解析并更新
                with self.lock:
                    self.latest_data["timestamp"] = data.get('t', 0.0)
                    self.latest_data["pos"] = np.array(data.get("p", [0,0,0]))
                    self.latest_data["quat"] = np.array(data.get("q", [0,0,0,1]))
                    self.latest_data["lin_vel"] = np.array(data.get("v", [0,0,0]))
                    self.latest_data["ang_vel"] = np.array(data.get("w", [0,0,0]))

            except socket.timeout:
                continue
            except (json.JSONDecodeError, Exception) as e:
                # print(f"[WARN] Receive error: {e}")
                pass

    def get_data(self):
        """获取最新的数据副本"""
        with self.lock:
            return self.latest_data.copy()

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
