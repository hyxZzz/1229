import numpy as np
from utils.geometry import quat_to_euler, body_to_earth, euler_to_quat, normalize

# 常量定义
G0 = 9.81              # 重力加速度 (m/s^2)
R_EARTH = 6371000.0    # 地球半径 (m)
RHO_SL = 1.225         # 海平面空气密度 (kg/m^3)
SOUND_SPEED = 340.0    # 海平面音速 (m/s)

class FlightDynamics:
    def __init__(self, initial_pos, initial_vel, initial_quat):
        """
        :param initial_pos: np.array [x, y, z] (z up)
        :param initial_vel: np.array [vx, vy, vz] (Earth frame)
        :param initial_quat: np.array [w, x, y, z]
        """
        self.pos = np.array(initial_pos, dtype=np.float64)
        self.vel = np.array(initial_vel, dtype=np.float64)
        self.quat = np.array(initial_quat, dtype=np.float64) # Attitude quaternion
        self.mass = 12000.0  # 假设质量 kg (F-16 class)
        self.fuel = 3000.0   # 燃油 kg
        
        # 气动系数 (Parametric Model)
        self.S_wing = 27.87  # 机翼面积 m^2
        self.CD0 = 0.01      # 零升阻力系数
        self.K = 0.15        # 诱导阻力系数 (CD = CD0 + K * CL^2)
        self.max_thrust_sl = 120000.0 # 海平面最大推力 (N)

    def _rotate_vector(self, vec, axis, angle):
        axis = normalize(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return (vec * cos_a +
                np.cross(axis, vec) * sin_a +
                axis * np.dot(axis, vec) * (1 - cos_a))

    def _g_limits(self, speed, mach):
        max_g_struct = 9.0
        min_g_struct = -3.0

        if speed < 120.0:
            max_g_struct = 5.0
        if speed < 80.0:
            max_g_struct = 3.0
            min_g_struct = 0.0

        if mach > 1.2:
            max_g_struct = max(7.0, max_g_struct - 2.0 * (mach - 1.2))

        return min_g_struct, max_g_struct

    def get_atmos(self, altitude):
        """简化的标准大气模型"""
        if altitude < 0: altitude = 0
        rho = RHO_SL * np.exp(-altitude / 9300.0) 
        return rho

    def get_mach(self, v_mag, altitude):
        a = SOUND_SPEED * max(0.8, (1 - 2.25e-5 * altitude)) 
        return v_mag / a

    def step(self, action_cmd, dt=0.01):
        """
        核心积分步进
        """
        # 1. 提取状态
        speed = np.linalg.norm(self.vel)
        if speed < 1.0: speed = 1.0 
        
        # [核心修正] 失速判定 (Stall Logic)
        # 防止 RL 利用低速物理漏洞悬停
        is_stalled = False
        if speed < 50.0: # 50 m/s (180 km/h) 为硬性失速线
            is_stalled = True

        altitude = self.pos[2]
        rho = self.get_atmos(altitude)
        mach = self.get_mach(speed, altitude)
        
        curr_rpy = quat_to_euler(self.quat)
        curr_roll, curr_pitch, curr_yaw = curr_rpy

        # 2. 控制响应
        target_roll = np.clip(action_cmd['target_roll'], -np.pi, np.pi)
        roll_rate = 3.0 
        roll_diff = target_roll - curr_roll
        d_roll = np.clip(roll_diff, -roll_rate * dt, roll_rate * dt)
        new_roll = curr_roll + d_roll

        target_pitch_rate = action_cmd.get('target_pitch_rate', 0.0)
        pitch_rate_limit = np.deg2rad(20)
        d_pitch = np.clip(target_pitch_rate, -pitch_rate_limit, pitch_rate_limit) * dt

        # 3. 计算升力与过载
        Q = 0.5 * rho * speed**2
        
        target_n = action_cmd['target_g']
        
        CL_max = 1.8 
        # 如果失速，强制丧失升力
        if is_stalled:
            CL_max = 0.0
            
        max_lift = CL_max * Q * self.S_wing
        max_g_aero = max_lift / (self.mass * G0)

        min_g_struct, max_g_struct = self._g_limits(speed, mach)
        max_g_allowed = min(max_g_struct, max_g_aero)
        n = np.clip(target_n, min_g_struct, max_g_allowed)
        
        lift = n * self.mass * G0
        CL = lift / (Q * self.S_wing + 1e-6)

        # 4. 计算阻力
        wave_drag_factor = 1.0
        if 0.9 < mach < 1.2:
            wave_drag_factor = 1.0 + 2.0 * (mach - 0.9) 
        
        CD = (self.CD0 + self.K * (CL**2)) * wave_drag_factor
        
        # 失速时阻力激增
        if is_stalled:
            CD = 2.0 # 平板阻力
            
        drag = CD * Q * self.S_wing

        # 5. 计算推力
        throttle = np.clip(action_cmd['throttle'], 0.0, 1.0)
        thrust_avail = self.max_thrust_sl * (rho / RHO_SL) * throttle
        
        # 6. 力学方程
        v_norm = self.vel / speed
        
        # 简化的姿态处理
        vel_pitch = np.arcsin(v_norm[2])
        vel_yaw = np.arctan2(v_norm[1], v_norm[0])
        
        # 重力
        fg = np.array([0, 0, -self.mass * G0])
        
        # 推力与阻力
        f_thrust = thrust_avail * v_norm 
        f_drag = -drag * v_norm
        
        # 升力方向计算
        world_up = np.array([0, 0, 1])
        right_vec = np.cross(v_norm, world_up)
        if np.linalg.norm(right_vec) < 1e-3: 
            right_vec = np.array([1, 0, 0])
        right_vec = normalize(right_vec)

        v_norm_cmd = self._rotate_vector(v_norm, right_vec, d_pitch)
        
        lift_up_vec = np.cross(right_vec, v_norm_cmd)
        
        lift_dir = lift_up_vec * np.cos(new_roll) + \
                   np.cross(v_norm_cmd, lift_up_vec) * np.sin(new_roll) + \
                   v_norm_cmd * np.dot(v_norm_cmd, lift_up_vec) * (1 - np.cos(new_roll))
                   
        f_lift = lift * lift_dir

        f_total = f_thrust + f_drag + f_lift + fg
        acc = f_total / self.mass
        
        # 7. 积分
        self.pos += self.vel * dt + 0.5 * acc * dt**2
        self.vel += acc * dt
        
        final_pitch = np.arcsin(normalize(self.vel)[2])
        final_yaw = np.arctan2(self.vel[1], self.vel[0])
        self.quat = euler_to_quat(new_roll, final_pitch, final_yaw)
        
        return self.pos, self.vel, speed, n
