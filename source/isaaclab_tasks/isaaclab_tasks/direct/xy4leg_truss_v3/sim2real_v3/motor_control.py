import serial
import struct
import math
import time
from time import sleep


class DaMiaoMotorController:
    def __init__(self, port, baudrate=115200, timeout=0.005):
        """初始化串口"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._connect()

    def _connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            if self.ser.isOpen():  # type: ignore[attr-defined]
                print(f"{self.port} 打开成功，波特率: {self.baudrate}")
        except serial.SerialException as e:
            print(f"无法打开串口 {self.port}: {e}")
            self.ser = None

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.isOpen():  # type: ignore[attr-defined]
            self.ser.close()
            print(f"{self.port} 已关闭")
    
    def uint_to_float(self, x_int: int, x_min: float, x_max: float, bits: int) -> float:
        """Converts unsigned int to float, given range and number of bits."""
        span = x_max - x_min
        offset = x_min
        return float(x_int) * span / float((1 << bits) - 1) + offset

    def float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """Converts a float to an unsigned int, given range and number of bits."""
        span = x_max - x_min
        offset = x_min
        return int((x - offset) * float((1 << bits) - 1) / span)

    def send(self, send_data):
        """发送16进制数据"""
        try:
            if self.ser and self.ser.isOpen():  # type: ignore[attr-defined]
                self.ser.write(send_data)
                # print(f"发送成功: {send_data.hex().upper()}")
            else:
                print("发送失败，串口未打开！")
        except ValueError as e:
            print(f"输入数据格式错误: {e}")

    def receive(self, num_bytes=8):
        """接收16进制数据"""
        try:
            if self.ser and self.ser.isOpen():  # type: ignore[attr-defined]
                response = self.ser.read(num_bytes)
                print(f"接收成功: {response.hex().upper()}")
                return response
            else:
                print("接收失败，串口未打开！")
        except Exception as e:
            print(f"接收数据时发生错误: {e}")
        return None

    def enable_motor(self, motor_id):
        """使能电机"""
        special_command = bytes.fromhex(f"{motor_id:02X} FF FF FF FF FF FF FF FF FC")
        self.send(special_command)
        response_enable = self.receive()
        feedback = self.damiao_feedback(response_enable)
        print(f"使能电机 {motor_id}")
        return feedback
    
    def disable_motor(self, motor_id):
        """失能电机"""
        special_command = bytes.fromhex(f"{motor_id:02X} FF FF FF FF FF FF FF FF FD")
        self.send(special_command)
        response_disable = self.receive()
        return response_disable
    
    def set_zero(self, motor_id):
        """设置电机当前位置为零点"""
        zero_command = bytes.fromhex(f"{motor_id:02X} FF FF FF FF FF FF FF FF FE")
        self.send(zero_command)
        sleep(0.1)
        response_zero = self.receive()
        return response_zero

    def damiao_feedback(self,data):
        """ID、错误码、位置、速度、力矩、温度"""

        if len(data) < 8:
           print(f"错误：数据长度不足，收到 {len(data)} 字节 -> {data}")
           return None
        
        motor_id = data[0] & 0x0F
        error_code = (data[0] >> 4) & 0x0F
        pos_int = (data[1] << 8) | data[2]          # 位置（16-bit）
        vel_int = (data[3] << 4) | (data[4] >> 4)   # 速度（12-bit）
        tor_int = ((data[4] & 0x0F) << 8) | data[5] # 力矩（12-bit）
        t_mos = data[6]                             # MOS温度
        t_rotor = data[7]                           # 线圈温度

        position = self.uint_to_float(pos_int, -12.5, 12.5, 16)
        velocity = self.uint_to_float(vel_int, -10.0, 10.0, 12)
        torque = self.uint_to_float(tor_int, -28.0, 28.0, 12)

        return {
        "motor_id": motor_id,
        "error_code": error_code,
        "position": position,
        "velocity": velocity,
        "torque": torque,
        "t_mos": t_mos,
        "t_rotor": t_rotor
        }

    
    def damiao_mit_command(self, pos, vel, kp, kd, tor, motor_id):
        """生成 MIT 模式的电机命令(10 字节）"""
        cmd = bytearray(10)

        pos_tmp = self.float_to_uint(pos, -12.5, 12.5, 16) # 16-bit
        vel_tmp = self.float_to_uint(vel, -10, 10, 12) # 12-bit
        kp_tmp = self.float_to_uint(kp, 0, 500, 12) # 12-bit
        kd_tmp = self.float_to_uint(kd, 0, 5, 12) # 12-bit
        tor_tmp = self.float_to_uint(tor, -28, 28, 12) # 12-bit

        cmd[0] = motor_id
        cmd[1] = 0xF2
        cmd[2] = (pos_tmp >> 8) & 0xFF
        cmd[3] = pos_tmp & 0xFF
        cmd[4] = (vel_tmp >> 4) & 0xFF
        cmd[5] = ((vel_tmp & 0xF) << 4) | ((kp_tmp >> 8) & 0xF)
        cmd[6] = kp_tmp & 0xFF
        cmd[7] = (kd_tmp >> 4) & 0xFF
        cmd[8] = ((kd_tmp & 0xF) << 4) | ((tor_tmp >> 8) & 0xF)
        cmd[9] = tor_tmp & 0xFF
        return bytes(cmd)

    @staticmethod
    def damiao_command(pos_hex, vel_hex, motor_id):
        """生成电机命令"""
        cmd = bytearray(10)
        pos = int(pos_hex, 16)
        vel = int(vel_hex, 16)
        cmd[0] = motor_id
        cmd[1] = 0xF1
        cmd[2] = (pos >> 24) & 0xFF
        cmd[3] = (pos >> 16) & 0xFF
        cmd[4] = (pos >> 8) & 0xFF
        cmd[5] = pos & 0xFF
        cmd[6] = (vel >> 24) & 0xFF
        cmd[7] = (vel >> 16) & 0xFF
        cmd[8] = (vel >> 8) & 0xFF
        cmd[9] = vel & 0xFF
        return bytes(cmd)
    
    def send_positions(self, positions, motor_ids):

        if len(positions) != len(motor_ids):
            print("错误:positions 数量应与 motor_ids 数量相同")
            return

        vel = 0.8  # 固定速度
        vel_bytes = struct.pack("<f", vel)
        vel_hex = vel_bytes.hex()

        for motor_id, pos in zip(motor_ids, positions):
            pos_bytes = struct.pack("<f", pos)
            pos_hex = pos_bytes.hex()
            command = self.damiao_command(pos_hex, vel_hex, motor_id)
            self.send(command)
            response = self.receive()
            time.sleep(0.0001)

    def process_commands(self):
        """处理用户输入的电机控制命令"""
        if self.ser is None:
            print("无法操作：串口未打开")
            return

        while True:
            try:
                user_input = input("输入多个电机信息（格式：'ID 角度 速度;ID 角度 速度'，回车跳过）：")
                if not user_input.strip():
                    break

                motor_commands = user_input.split(";")
                for motor_command in motor_commands:
                    motor_params = list(map(float, motor_command.strip().split()))
                    if len(motor_params) < 2:
                        print("输入格式错误，请检查输入！")
                        continue

                    motor_id = int(motor_params[0])
                    self.enable_motor(motor_id)

                    pos = motor_params[1]
                    pos_radians = math.radians(pos)
                    pos_bytes = struct.pack("<f", pos_radians)
                    pos_hex = pos_bytes.hex()

                    vel = 5.0 if len(motor_params) == 2 else motor_params[2]
                    vel_bytes = struct.pack("<f", vel)
                    vel_hex = vel_bytes.hex()

                    command = self.damiao_command(pos_hex, vel_hex, motor_id)
                    sleep(0.0001)
                    self.send(command)
                    response = self.receive()
                    feedback = self.damiao_feedback(response)
                    print(feedback)
            except Exception as e:
                print(f"处理命令时发生错误: {e}")
                break

    def process_commands_mit(self):
        """
        MIT 模式控制：
        - 1-5号电机(位置模式): ID 角度(度) [kp kd]，默认 kp=10.0, kd=0.5, vel=0, tor=0
        - 6号电机(力矩模式): ID tor,kp=0, kd=0, pos=0, vel=0
        分号分隔多条命令
        """
        if self.ser is None:
            print("无法操作：串口未打开")
            return

        while True:
            try:
                user_input = input("输入电机信息（位置模式:'ID 角度 [kp kd]' | 力矩模式(6号):'ID tor'，分号分隔，回车结束）：")
                if not user_input.strip():
                    break

                motor_commands = user_input.split(";")
                for motor_command in motor_commands:
                    motor_params = list(map(float, motor_command.strip().split()))
                    if len(motor_params) < 2:
                        print("输入格式错误，请检查输入！")
                        continue

                    motor_id = int(motor_params[0])
                    
                    self.enable_motor(motor_id)

                    # 判断是否为六号电机（夹爪电机）
                    if motor_id == 6:
                        # 六号电机使用力矩模式
                        tor = motor_params[1]  # 第二个参数是力矩
                        cmd = self.damiao_mit_command(pos=0.0, vel=0.0, kp=0.0, kd=0.0, tor=tor, motor_id=motor_id)
                        print(f"[力矩模式] 电机 {motor_id}: tor={tor}N·m")
                    else:
                        # 1-5号电机使用位置模式
                        angle_deg = motor_params[1]
                        kp = motor_params[2] if len(motor_params) > 2 else 20.0
                        kd = motor_params[3] if len(motor_params) > 3 else 1
                        
                        pos_radians = math.radians(angle_deg)
                        cmd = self.damiao_mit_command(pos=pos_radians, vel=0.0, kp=kp, kd=kd, tor=0.0, motor_id=motor_id)
                        print(f"[位置模式] 电机 {motor_id}: pos={angle_deg}°, kp={kp}, kd={kd}")
                    
                    sleep(0.0001)
                    self.send(cmd)
                    response = self.receive()
                    feedback = self.damiao_feedback(response) if response else None
                    print(f"MIT反馈: {feedback}")
            except Exception as e:
                print(f"处理 MIT 命令时发生错误: {e}")
                break


if __name__ == '__main__':
    controller = DaMiaoMotorController('/dev/ttyUSB3')
    if controller.ser is None:
        print("程序终止：串口未能成功打开")
    else:
        try:
            # # controller.process_commands()
            controller.process_commands_mit()
            # controller.disable_motor(1)
            # controller.enable_motor(1)
            # controller.set_zero(6)
        finally:
            controller.close()

