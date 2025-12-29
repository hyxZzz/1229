import numpy as np
from utils.geometry import get_distance, normalize

class RewardFunction:
    def __init__(self, team_id, reward_cfg=None, engage_cfg=None):
        self.team_id = team_id
        self.reward_cfg = {
            "tick_survival": 0.01,
            "search_bonus": 0.02,
            "approach_scale": 0.001,
            "too_far_penalty": -0.05,
            "position_bonus": 0.05,
            "aim_bonus": 0.03,
            "good_shot_bonus": 1.5,
            "bad_shot_penalty": -0.5,
            "missile_support_scale": 0.2,
            "redundant_fire_penalty": -0.2,
            "be_shot_penalty": -5.0,
        }
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)
        self.engage_cfg = {
            "radar_range": 70000.0,
            "engage_range": 50000.0,
            "wez_min": 15000.0,
            "wez_max": 35000.0,
            "aim_cos": 0.85,
            "shot_cos": 0.90,
            "support_max_dist": 30000.0,
        }
        if engage_cfg:
            self.engage_cfg.update(engage_cfg)
        
    def compute_reward(self, agent, sim, action_dict):
        """
        修正后的奖励函数：鼓励在 WEZ (有效攻击区) 内开火
        """
        reward = 0.0
        
        # 1. 存活奖励 (保持不变)
        if not agent.is_active:
            return self.reward_cfg["be_shot_penalty"] # 死亡惩罚
        reward += self.reward_cfg["tick_survival"]

        # 2. 寻找最近的敌机
        nearest_enemy = None
        min_dist = float('inf')
        for enemy in sim.aircrafts:
            if enemy.team != agent.team and enemy.is_active:
                d = get_distance(agent.pos, enemy.pos)
                if d < min_dist:
                    min_dist = d
                    nearest_enemy = enemy
        
        if nearest_enemy is None:
            return reward # 敌人全灭，躺赢

        # 3. 距离控制奖励 (逼迫接近，范围调整为 10km - 50km 为最佳)
        # 这种 shaping reward 有助于让飞机飞向敌人
        vec_to_enemy = nearest_enemy.pos - agent.pos
        dist = np.linalg.norm(vec_to_enemy)
        dir_to_enemy = vec_to_enemy / (dist + 1e-6)
        closing_speed = np.dot(agent.vel, dir_to_enemy)
        reward += self.reward_cfg["approach_scale"] * closing_speed
        if dist > self.engage_cfg["radar_range"] and closing_speed < 0.0:
            reward += self.reward_cfg["too_far_penalty"]
        if dist < self.engage_cfg["radar_range"]:
            reward += self.reward_cfg["search_bonus"]

        # 4. 角度优势奖励 (WEZ 持续奖励)
        vec_to_enemy = normalize(nearest_enemy.pos - agent.pos)
        my_heading = normalize(agent.vel)
        # 计算机头指向夹角
        dot = np.dot(vec_to_enemy, my_heading) # 1.0 = 正对
        
        # 如果机头大致指向敌人，且距离在雷达范围内，给持续奖励
        in_range = (min_dist < self.engage_cfg["engage_range"])
        aim_locked = (dot > self.engage_cfg["aim_cos"]) # 约 30度以内
        
        if in_range and aim_locked:
            reward += self.reward_cfg["aim_bonus"] # 持续瞄准奖励
        in_wez = (
            self.engage_cfg["wez_min"] <= min_dist <= self.engage_cfg["wez_max"]
            and dot > self.engage_cfg["aim_cos"]
        )
        if in_wez:
            reward += self.reward_cfg["position_bonus"]

        # --- 5. 开火判定 (核心修改) ---
        my_action = action_dict.get(agent.uid, {})
        if isinstance(my_action, dict):
            fire_target_uid = my_action.get('fire_target')
            
            if fire_target_uid:
                # 检查开火质量
                target = sim.get_entity(fire_target_uid)
                if target and target.is_active:
                    t_dist = get_distance(agent.pos, target.pos)
                    
                    # 计算针对该目标的指向角
                    t_vec = normalize(target.pos - agent.pos)
                    t_dot = np.dot(t_vec, my_heading)
                    
                    # 判定是否为“好的一次射击” (Good Shot)
                    # 条件：距离 < 40km 且 角度 < 30度 (dot > 0.86)
                    is_good_shot = (
                        self.engage_cfg["wez_min"] <= t_dist <= self.engage_cfg["wez_max"]
                        and t_dot > self.engage_cfg["shot_cos"]
                    )
                    
                    if is_good_shot:
                        reward += self.reward_cfg["good_shot_bonus"]
                        redundant_fire = any(
                            m.launcher_uid == agent.uid
                            and m.target
                            and m.target.uid == fire_target_uid
                            and m.is_active
                            for m in sim.missiles
                        )
                        if redundant_fire:
                            reward += self.reward_cfg["redundant_fire_penalty"]
                    else:
                        # 乱射惩罚 (比如背对敌人开火，或者距离过远)
                        reward += self.reward_cfg["bad_shot_penalty"]
                else:
                    # 攻击不存在的目标
                    reward += self.reward_cfg["bad_shot_penalty"]
        
        # 6. 导弹制导奖励 (保持不变，或稍微调大)
        for m in sim.missiles:
            if m.launcher_uid == agent.uid and m.is_active:
                if m.target and m.target.is_active:
                    m_dist = get_distance(m.pos, m.target.pos)
                    # 导弹越接近，奖励越大
                    support_max = self.engage_cfg["support_max_dist"]
                    support_gain = max(0.0, (support_max - m_dist) / support_max)
                    reward += self.reward_cfg["missile_support_scale"] * support_gain
                    
        return reward
