import numpy as np
from utils.geometry import quat_to_euler, euler_to_quat, normalize

class FlightDynamics:
    def __init__(self, initial_pos, initial_vel, initial_quat):
        """
        简化版动力学：基于运动学 (Kinematics) 更新
        """
        self.pos = np.array(initial_pos, dtype=np.float64)
        self.vel = np.array(initial_vel, dtype=np.float64)
        self.quat = np.array(initial_quat, dtype=np.float64)
        
        # 基础性能参数
        self.min_speed = 50.0   # m/s
        self.max_speed = 400.0  # m/s (约1.2马赫)
        self.acc_rate = 15.0    # m/s^2 (加速率)
        self.dec_rate = 10.0    # m/s^2 (减速率)
        self.g0 = 9.81
        
        # 用于 ObservationParser 的兼容属性
        self.mass = 10000.0 # 虚拟质量，仅用于兼容接口

    def step(self, cmd, dt=0.05):
        """
        根据 maneuver_lib 的指令更新状态
        cmd: {'acc_flag', 'target_g', 'target_roll', 'flag'}
        """
        # 1. 提取指令
        acc_flag = cmd['acc_flag']
        target_g = cmd['target_g']
        target_roll = cmd['target_roll']
        
        # 2. 更新速度大小 (Speed)
        current_speed = np.linalg.norm(self.vel)
        if current_speed < 1e-3: current_speed = 1.0 # 防止除零
        
        if acc_flag == 2:
            current_speed += self.acc_rate * dt
        elif acc_flag == -2:
            current_speed -= self.dec_rate * dt
        # acc_flag == 0 时，速度保持不变 (简化处理，忽略阻力减速)
            
        current_speed = np.clip(current_speed, self.min_speed, self.max_speed)
        
        # 3. 更新速度方向 (Velocity Direction)
        # 核心逻辑：计算升力矢量带来的向心加速度
        
        # A. 构建坐标系
        v_norm = self.vel / np.linalg.norm(self.vel) # 前向矢量
        world_up = np.array([0, 0, 1])
        
        # 右矢量 (水平面的右)
        right = np.cross(v_norm, world_up)
        if np.linalg.norm(right) < 1e-3: 
            # 特殊情况：垂直爬升/俯冲时，右矢量暂定为 X 轴
            right = np.array([1, 0, 0])
        right = normalize(right)
        
        # 机体上矢量 (不考虑滚转时的"上")
        body_up = np.cross(right, v_norm)
        
        # B. 计算升力方向 (Lift Vector)
        # 将 body_up 绕 v_norm 旋转 target_roll 角度
        # Rodrigues' rotation formula 简化版
        # lift_dir = body_up * cos(roll) + right * sin(roll)
        sin_r = np.sin(target_roll)
        cos_r = np.cos(target_roll)
        lift_dir = normalize(body_up * cos_r + right * sin_r)
        
        # C. 计算合加速度矢量
        # a_total = a_lift + a_gravity
        # a_lift 大小 = n * g0
        acc_lift = lift_dir * (target_g * self.g0)
        acc_gravity = np.array([0, 0, -self.g0])
        
        # 总加速度 (改变方向的部分)
        # 注意：这里我们只用横向加速度来改变方向，切向加速度(加减速)已经在上面独立处理了
        acc_turn = acc_lift + acc_gravity
        
        # 去除切向分量，只保留垂直于速度方向的分量 (纯改变方向)
        # 这一步是为了防止重力导致额外的加速/减速，让 speed 控制权完全在 acc_flag 手里
        acc_turn_perp = acc_turn - v_norm * np.dot(acc_turn, v_norm)
        
        # 4. 更新速度矢量
        # v_new_dir = v_old_dir + (a_perp * dt) / speed
        v_new_vec = self.vel + acc_turn_perp * dt
        self.vel = normalize(v_new_vec) * current_speed
        
        # 5. 更新位置
        self.pos += self.vel * dt
        
        # 6. 更新姿态 (Quaternion)
        # 简化处理：姿态直接由速度矢量和滚转角决定
        # Pitch/Yaw 来自速度矢量
        pitch = np.arcsin(np.clip(self.vel[2] / current_speed, -1.0, 1.0))
        yaw = np.arctan2(self.vel[1], self.vel[0])
        
        # Roll 直接使用指令值 (假设瞬间响应，或者您可以加一个平滑插值)
        # 这里为了响应迅速，直接设为 target_roll
        self.quat = euler_to_quat(target_roll, pitch, yaw)
        
        # 返回兼容旧接口的数据
        # n (过载) 在这里就是 target_g
        return self.pos, self.vel, current_speed, target_g