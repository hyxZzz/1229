import numpy as np
from utils.geometry import get_distance, get_vector, normalize

class RewardFunction:
    def __init__(self, team_id):
        self.team_id = team_id
        
    def compute_reward(self, agent, sim, action_dict):
        """
        修正后的奖励函数：鼓励在 WEZ (有效攻击区) 内开火
        """
        reward = 0.0
        
        # 1. 存活奖励 (保持不变)
        if not agent.is_active:
            return -5.0 # 死亡惩罚
        reward += 0.01

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
        reward += 0.001 * closing_speed
        if dist > 50000.0 and closing_speed < 100.0:
            reward -= 0.05

        # 4. 角度优势奖励 (WEZ 持续奖励)
        vec_to_enemy = normalize(nearest_enemy.pos - agent.pos)
        my_heading = normalize(agent.vel)
        # 计算机头指向夹角
        dot = np.dot(vec_to_enemy, my_heading) # 1.0 = 正对
        
        # 如果机头大致指向敌人，且距离在雷达范围内，给持续奖励
        in_range = (min_dist < 60000.0)
        aim_locked = (dot > 0.9) # 约 25度以内
        
        if in_range and aim_locked:
            reward += 0.05 # 持续瞄准奖励

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
                    is_good_shot = (t_dist < 40000.0) and (t_dot > 0.86)
                    
                    if is_good_shot:
                        reward += 1.0  # <--- 巨大的正向激励！鼓励开火！
                        # print(f"DEBUG: {agent.uid} Good Shot! Reward +2.0")
                    else:
                        # 乱射惩罚 (比如背对敌人开火，或者距离过远)
                        reward -= 0.5
                else:
                    # 攻击不存在的目标
                    reward -= 0.5
        
        # 6. 导弹制导奖励 (保持不变，或稍微调大)
        for m in sim.missiles:
            if m.launcher_uid == agent.uid and m.is_active:
                if m.target and m.target.is_active:
                    m_dist = get_distance(m.pos, m.target.pos)
                    # 导弹越接近，奖励越大
                    reward += 0.002 * (30000 - m_dist) / 10000.0
                    
        return reward