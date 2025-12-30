import numpy as np
from sim_core.entities.aircraft import Aircraft
from sim_core.entities.missile import Missile
from utils.geometry import get_distance, normalize

class Simulation:
    def __init__(self, engagement_cfg=None):
        self.aircrafts = [] # 所有的飞机
        self.missiles = []  # 所有的导弹
        self.entity_map = {} # uid -> entity
        self.dt = 0.05      # 物理步长
        self.time = 0.0
        engagement_cfg = engagement_cfg or {}
        self.wez_min = engagement_cfg.get("wez_min", 5000.0)
        self.wez_max = engagement_cfg.get("wez_max", 35000.0)
        if "shot_cos" in engagement_cfg:
            self.wez_angle = np.arccos(np.clip(engagement_cfg["shot_cos"], -1.0, 1.0))
        elif "aim_cos" in engagement_cfg:
            self.wez_angle = np.arccos(np.clip(engagement_cfg["aim_cos"], -1.0, 1.0))
        else:
            self.wez_angle = np.deg2rad(engagement_cfg.get("wez_angle_deg", 30.0))

    def reset_8v8(self, init_state=None):
        """初始化标准的 8v8 对抗场景"""
        self.aircrafts = []
        self.missiles = []
        self.entity_map = {}
        self.time = 0.0

        # === [修复] 默认高度范围 (无论是否有 init_state 都需要) ===
        alt_min, alt_max = 8000, 12000

        if init_state:
            red_center = init_state.get("red_center", [-20000, 0, 8000])
            blue_center = init_state.get("blue_center", [20000, 0, 8000])
            spread_range = init_state.get("spread_range", 28000)
            spacing = spread_range / max(1, (8 - 1))
            center_offset = init_state.get("center_offset", [0.0, 0.0, 0.0])
            pos_noise_range = init_state.get("pos_noise_range", 2000.0)
            heading_noise_deg = init_state.get("heading_noise_deg", 10.0)
        else:
            red_center = [-20000, 0, 8000]
            blue_center = [20000, 0, 8000]
            spacing = 4000
            
            # === [修改] 增大随机性 (仅在无 init_state 时覆盖默认值) ===
            center_offset = [
                np.random.uniform(-5000, 5000),
                np.random.uniform(-5000, 5000),
                np.random.uniform(-2000, 2000),
            ]
            pos_noise_range = 3000.0
            heading_noise_deg = 45.0

        red_center = [
            red_center[0] + center_offset[0],
            red_center[1] + center_offset[1],
            red_center[2] + center_offset[2],
        ]
        blue_center = [
            blue_center[0] - center_offset[0],
            blue_center[1] - center_offset[1],
            blue_center[2] + center_offset[2],
        ]
        heading_noise_rad = np.deg2rad(heading_noise_deg)
        
        # --- 红方 (Team 0) ---
        for i in range(8):
            uid = f"Red_{i}"
            pos_noise = np.random.uniform(-pos_noise_range, pos_noise_range, 3)
            
            # 使用 alt_min/alt_max 随机生成高度，覆盖原有的 Z 轴逻辑
            rand_alt = np.random.uniform(alt_min, alt_max)
            
            pos = [
                red_center[0] + pos_noise[0],
                red_center[1] + (i - 3.5) * spacing + pos_noise[1],
                rand_alt, # [修改] 强制使用随机高度
            ]
            heading = 0.0 + np.random.uniform(-heading_noise_rad, heading_noise_rad)
            speed = np.random.uniform(280, 320) # 随机速度
            vel = [speed * np.cos(heading), speed * np.sin(heading), 0]
            p = Aircraft(uid, 0, pos, vel, init_heading=heading)
            self.aircrafts.append(p)
            self.entity_map[uid] = p
            
        # --- 蓝方 (Team 1) ---
        for i in range(8):
            uid = f"Blue_{i}"
            pos_noise = np.random.uniform(-pos_noise_range, pos_noise_range, 3)
            
            # 蓝方也应用同样的高度随机逻辑
            rand_alt = np.random.uniform(alt_min, alt_max)
            
            pos = [
                blue_center[0] + pos_noise[0],
                blue_center[1] + (i - 3.5) * spacing + pos_noise[1],
                rand_alt, # [修改] 强制使用随机高度
            ]
            heading = np.pi + np.random.uniform(-heading_noise_rad, heading_noise_rad)
            speed = np.random.uniform(280, 320) # 随机速度
            vel = [speed * np.cos(heading), speed * np.sin(heading), 0]
            p = Aircraft(uid, 1, pos, vel, init_heading=heading) # 朝西
            self.aircrafts.append(p)
            self.entity_map[uid] = p
            
        print(f"Simulation Reset: 8 Red vs 8 Blue initialized.")

    def get_entity(self, uid):
        return self.entity_map.get(uid)

    def step(self, red_actions, blue_actions):
        """
        执行一步仿真
        red_actions: {uid: {'maneuver': int, 'fire_target': target_uid}}
        blue_actions: {uid: maneuver_int}
        """
        self.time += self.dt
        events = [] # 记录击杀事件
        
        # 1. 飞机更新 (移动 + 发射)
        for p in self.aircrafts:
            if not p.is_active: continue
            
            # --- 解析动作 ---
            maneuver_id = 0
            fire_target_uid = None
            
            if p.team == 0: # 红方
                cmd = red_actions.get(p.uid, {})
                if isinstance(cmd, dict):
                    maneuver_id = cmd.get('maneuver', 0)
                    fire_target_uid = cmd.get('fire_target')
                else:
                    maneuver_id = cmd # 兼容纯整数输入
            else: # 蓝方
                maneuver_id = blue_actions.get(p.uid, 0)
                
            # --- 执行物理机动 ---
            p.step(maneuver_id, self.dt)
            
            # --- 执行开火逻辑 ---
            # 限制：每步只能发一枚，且必须有弹，且目标存活
            if fire_target_uid and p.missile_count > 0:
                target = self.get_entity(fire_target_uid)
                if target and target.is_active:
                    vec_to_target = target.pos - p.pos
                    dist = np.linalg.norm(vec_to_target)
                    my_dir = normalize(p.vel)
                    los = normalize(vec_to_target)
                    angle = np.arccos(np.clip(np.dot(my_dir, los), -1, 1))
                    is_in_wez = (self.wez_min < dist < self.wez_max) and (angle <= self.wez_angle)
                    if is_in_wez:
                        p.missile_count -= 1
                        # 导弹UID命名: M_Red_0_1
                        m_uid = f"M_{p.uid}_{3-p.missile_count}"
                        new_missile = Missile(m_uid, p.team, p, target)
                        self.missiles.append(new_missile)
                        events.append({'type': 'FIRE', 'launcher': p.uid,'target': target.uid})

        # 2. 导弹更新
        # 收集所有活着的敌机作为潜在重规划目标
        active_reds = [p for p in self.aircrafts if p.team == 0 and p.is_active]
        active_blues = [p for p in self.aircrafts if p.team == 1 and p.is_active]
        
        for m in self.missiles:
            if not m.is_active: continue
            
            # 传入敌方列表供重规划使用
            enemies = active_blues if m.team == 0 else active_reds
            
            hit, hit_uid = m.update(self.dt, enemies)
            
            if hit:
                events.append({'type': 'KILL', 'killer': m.launcher_uid, 'victim': hit_uid})

        return events
