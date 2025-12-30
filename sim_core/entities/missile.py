import numpy as np
from sim_core.entities.base import Entity
from utils.geometry import get_distance, normalize

class Missile(Entity):
    def __init__(self, uid, team, launcher, target):
        super().__init__(uid, team)
        self.launcher_uid = launcher.uid
        self.target = target 
        
        # 初始状态继承自载机
        self.pos = launcher.pos.copy()
        launch_dir = normalize(launcher.vel)
        # 初始速度：载机速度 + 弹射初速
        self.vel = launcher.vel + launch_dir * 50.0 
        
        # --- 1. 动力阶段参数 ---
        self.time_alive = 0.0
        self.boost_time = 3.0       # 助推阶段 (s)
        self.sustain_time = 10.0    # 续航阶段 (s) (结束时间 T=13s)
        
        self.boost_thrust = 150.0   # 助推加速度 (m/s^2) ~15G
        self.sustain_thrust = 50.0  # 续航加速度 (m/s^2) ~5G
        
        self.current_stage = "Boost" 
        
        # --- 2. 导引律参数 ---
        self.max_g = 40.0           # 最大过载 (G)
        self.N_pn = 4.0             # 比例导引系数
        self.fov = np.deg2rad(60.0) # 导引头视场角
        
        # --- 3. 状态标志 ---
        self.lost_lock = False      # 是否丢失目标（供 Env 识别）
        self.max_speed = 1200.0     # 极速限制 (m/s) ~ Mach 4
        self.min_speed = 100.0      # 失速速度
        self.seeker_memory = 2.0   # 目标丢失后的惯导记忆时间 (s)
        self.terminal_range = 5000.0  # 末段强制追踪距离 (m)
        self.last_seen_pos = target.pos.copy()
        self.last_seen_vel = target.vel.copy()
        self.last_lock_time = 0.0

    def update(self, dt, enemies=None):
        """
        :param enemies: 传入是为了兼容接口，实际不再内部自动重规划
        """
        if not self.is_active: 
            return False, None
            
        self.time_alive += dt
        
        # 记录上一帧位置，用于防穿模检测
        prev_pos = self.pos.copy()
        
        # 1. 目标有效性检查
        target_valid = self.target is not None and self.target.is_active
        dist_to_target = None
        if not target_valid:
            self.lost_lock = True
        else:
            r_vec = self.target.pos - self.pos
            dist_to_target = np.linalg.norm(r_vec)
            # 检查视场角 (FOV)
            my_dir = normalize(self.vel)
            vec_to_target = normalize(r_vec)
            angle = np.arccos(np.clip(np.dot(my_dir, vec_to_target), -1, 1))
            if angle <= self.fov or (dist_to_target is not None and dist_to_target <= self.terminal_range):
                self.lost_lock = False
                self.last_seen_pos = self.target.pos.copy()
                self.last_seen_vel = self.target.vel.copy()
                self.last_lock_time = self.time_alive
            else:
                self.lost_lock = True

        # 2. 计算动力 (Thrust)
        acc_thrust = 0.0
        if self.time_alive < self.boost_time:
            self.current_stage = "Boost"
            acc_thrust = self.boost_thrust
        elif self.time_alive < (self.boost_time + self.sustain_time):
            self.current_stage = "Sustain"
            acc_thrust = self.sustain_thrust
        else:
            self.current_stage = "Coast"
            acc_thrust = 0.0 
            
        # 3. 计算导引过载 (Guidance Load)
        acc_guide = np.zeros(3)
        
        if target_valid:
            # === 比例导引 (PN) ===
            if self.lost_lock and (self.time_alive - self.last_lock_time) <= self.seeker_memory:
                time_since_lock = self.time_alive - self.last_lock_time
                guide_pos = self.last_seen_pos + self.last_seen_vel * time_since_lock
                guide_vel = self.last_seen_vel
            elif not self.lost_lock:
                guide_pos = self.target.pos
                guide_vel = self.target.vel
            else:
                guide_pos = None
                guide_vel = None

            if guide_pos is not None:
                r_vec = guide_pos - self.pos
                dist = np.linalg.norm(r_vec)
                r_dir = r_vec / (dist + 1e-6)
                
                v_rel = guide_vel - self.vel
                vc = -np.dot(v_rel, r_dir)
                omega = np.cross(r_vec, v_rel) / (dist**2 + 1e-6)
                
                acc_magnitude = self.N_pn * vc * np.linalg.norm(omega)
                acc_magnitude = np.clip(acc_magnitude, -self.max_g * 9.81, self.max_g * 9.81)
                
                cmd_vec = np.cross(omega, self.vel)
                if np.linalg.norm(cmd_vec) > 1e-3:
                    acc_guide = normalize(cmd_vec) * acc_magnitude
        
        # 4. 简化的空气阻力
        speed = np.linalg.norm(self.vel)
        k_drag = 0.0002 
        drag_acc_mag = k_drag * speed**2
        acc_drag = -normalize(self.vel) * drag_acc_mag

        # 5. 运动学积分
        acc_total = normalize(self.vel) * acc_thrust + acc_drag + acc_guide
        
        self.vel += acc_total * dt
        self.pos += self.vel * dt
        
        # 速度限制
        new_speed = np.linalg.norm(self.vel)
        if new_speed > self.max_speed:
            self.vel = normalize(self.vel) * self.max_speed
            
        # 6. 命中判定 (修复版：射线检测防止穿模)
        if target_valid and (not self.lost_lock or (dist_to_target is not None and dist_to_target <= self.terminal_range)):
            # A. 计算当前距离
            dist = np.linalg.norm(self.target.pos - self.pos)
            
            # B. 射线/线段检测
            # 计算这一步飞行的向量: P_prev -> P_curr
            flight_vec = self.pos - prev_pos
            flight_dist_sq = np.dot(flight_vec, flight_vec)
            
            is_hit = False
            
            if flight_dist_sq < 1e-6:
                # 几乎没动，直接用点距离判断
                if dist < 30.0: is_hit = True
            else:
                # 计算目标点到飞行线段的投影
                # 向量 P_prev -> Target
                to_target_vec = self.target.pos - prev_pos
                
                # 投影系数 t (0 ~ 1) 表示最近点在线段上的位置
                t = np.dot(to_target_vec, flight_vec) / flight_dist_sq
                t = np.clip(t, 0.0, 1.0)
                
                # 线段上离目标最近的点
                closest_point = prev_pos + flight_vec * t
                
                # 计算目标到该最近点的距离
                segment_dist = np.linalg.norm(self.target.pos - closest_point)
                
                if segment_dist < 30.0:
                    is_hit = True

            if is_hit:
                self.is_active = False
                self.target.hp -= 100 # 击毁
                self.target.is_active = False
                return True, self.target.uid
                
        # 7. 失效判定
        if self.pos[2] < 0 or new_speed < self.min_speed or self.time_alive > 60.0:
            self.is_active = False
            
        return False, None
