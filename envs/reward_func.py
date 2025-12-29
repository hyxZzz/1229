import numpy as np
from utils.geometry import get_distance, normalize

class RewardFunction:
    def __init__(self, team_id, reward_cfg=None, engage_cfg=None):
        self.team_id = team_id
        # 默认奖励参数 (会被传入的配置覆盖)
        self.reward_cfg = {
            "tick_survival": 0.01,
            "search_bonus": 0.0,
            "approach_scale": 0.5,       # 加大接近奖励系数
            "too_far_penalty": -0.05,
            "position_bonus": 0.2,       # 核心：高额的占位奖励
            "aim_bonus": 0.05,
            "good_shot_bonus": 3.0,      # 核心：高质量射击大奖
            "bad_shot_penalty": -2.0,    # 核心：乱射重罚
            "missile_support_scale": 0.0,
            "redundant_fire_penalty": -1.0,
            "be_shot_penalty": -5.0,
        }
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)
            
        # 默认交战参数
        self.engage_cfg = {
            "radar_range": 70000.0,
            "engage_range": 50000.0,
            "wez_min": 10000.0,
            "wez_max": 40000.0,
            "aim_cos": 0.85,
            "shot_cos": 0.95,            # 射击角度要求更严格
            "support_max_dist": 30000.0,
        }
        if engage_cfg:
            self.engage_cfg.update(engage_cfg)
        
    def compute_reward(self, agent, sim, action_dict):
        """
        修正后的奖励函数：强力引导“先占位后开火”策略
        """
        reward = 0.0
        
        # 1. 存活与死亡处理
        if not agent.is_active:
            return self.reward_cfg["be_shot_penalty"]
        reward += self.reward_cfg["tick_survival"]

        # 2. 获取所有敌机信息
        enemies = [e for e in sim.aircrafts if e.team != agent.team and e.is_active]
        if not enemies:
            return reward + 1.0 # 敌人全灭，给予额外存活奖

        # 3. 寻找最近的敌机 (作为主要关注对象)
        # 使用 lambda 简化距离计算
        nearest_enemy = min(enemies, key=lambda e: get_distance(agent.pos, e.pos))
        dist = get_distance(agent.pos, nearest_enemy.pos)
        
        # 计算相对几何关系
        vec_to_enemy = normalize(nearest_enemy.pos - agent.pos)
        my_heading = normalize(agent.vel)
        aim_dot = np.dot(vec_to_enemy, my_heading) # 1.0表示正对
        
        # --- 策略引导核心 ---

        # A. 接近奖励 (Approach): 引导飞向敌人
        # 只有在射程外才给接近奖励，防止发生碰撞或冲过头
        if dist > self.engage_cfg["engage_range"]:
            closing_speed = np.dot(agent.vel, vec_to_enemy)
            # 归一化：除以音速(约340)，使得输出大致在 [-1, 1] 之间
            reward += self.reward_cfg["approach_scale"] * (closing_speed / 340.0)

        # B. 角度/瞄准奖励 (Alignment): 只要把机头对准敌人，就给分
        # 这是为了让飞机学会由“飞向”转变为“咬尾/瞄准”
        if aim_dot > self.engage_cfg["aim_cos"]:
            reward += self.reward_cfg["aim_bonus"]

        # C. 完美攻击区 (WEZ) 持续奖励: 这是“占位”的核心
        # 条件：距离在 WEZ 内，且角度也满足严格的射击条件
        in_wez_dist = (self.engage_cfg["wez_min"] < dist < self.engage_cfg["wez_max"])
        in_wez_angle = (aim_dot > self.engage_cfg["shot_cos"])
        
        if in_wez_dist and in_wez_angle:
            # 这是一个非常有利的位置，给予高额持续奖励，鼓励智能体保持在这里
            # 这相当于告诉它：“这里是最佳位置，待在这里别动！”
            reward += self.reward_cfg["position_bonus"] 

        # --- 4. 开火判定 (Action Evaluation) ---
        my_action = action_dict.get(agent.uid, {})
        # 兼容处理：有些地方可能是直接传 int，有些是传 dict
        if isinstance(my_action, dict):
            fire_target_uid = my_action.get('fire_target')
            
            if fire_target_uid:
                # 获取被攻击的目标对象
                target = sim.get_entity(fire_target_uid)
                
                # 判定射击质量
                is_valid_shot = False
                if target and target.is_active:
                    t_dist = get_distance(agent.pos, target.pos)
                    t_vec = normalize(target.pos - agent.pos)
                    t_dot = np.dot(t_vec, my_heading)
                    
                    # 严格的判定标准：必须在 WEZ 距离内，且指向误差极小
                    dist_ok = (self.engage_cfg["wez_min"] <= t_dist <= self.engage_cfg["wez_max"])
                    angle_ok = (t_dot > self.engage_cfg["shot_cos"])
                    
                    if dist_ok and angle_ok:
                        is_valid_shot = True

                if is_valid_shot:
                    # 好射击：给予大奖
                    reward += self.reward_cfg["good_shot_bonus"]
                    
                    # 额外检查：避免重复射击（如果天上已经有我不久前发出的针对该目标的导弹）
                    # 避免一次性把弹打光，学会节省弹药
                    active_missiles_at_target = sum(
                        1 for m in sim.missiles 
                        if m.launcher_uid == agent.uid and m.target and m.target.uid == fire_target_uid and m.is_active
                    )
                    # 如果已经有超过1枚导弹（包含刚发射这枚）在飞向目标，视为冗余
                    # 注意：step函数里是先生成导弹再算奖励，所以这里至少有1枚
                    if active_missiles_at_target > 1:
                        reward += self.reward_cfg["redundant_fire_penalty"]
                else:
                    # 坏射击：给予重罚！这是学会“先占位”的关键
                    # 告诉它：没瞄准好就开火是极其错误的，比不作为还要糟糕
                    reward += self.reward_cfg["bad_shot_penalty"]

        # 5. 导弹支援奖励 (Missile Support)
        # 鼓励发射后继续在该方向保持有利态势 (模拟半主动雷达制导需求，或者单纯为了观察战果)
        for m in sim.missiles:
            if m.launcher_uid == agent.uid and m.is_active and m.target and m.target.is_active:
                # 只要有导弹活着并在追踪，每一步给一点点微小的奖励
                reward += 0.005 

        return reward