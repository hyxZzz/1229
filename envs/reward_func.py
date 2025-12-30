import numpy as np
from utils.geometry import get_distance, normalize

class RewardFunction:
    def __init__(self, team_id, reward_cfg=None):
        self.team_id = team_id
        # === [修改] 调整后的奖励参数，防止 Reward Hacking ===
        self.cfg = {
            # --- 1. 态势奖励 (大幅削弱，仅作微弱引导) ---
            # 原值 0.05 -> 现值 0.002。
            # 即使保持完美占位 2000 步，总分仅 4.0 分，远低于击杀分。
            "align_bonus": 0.002,        # 角度优势 (指向敌机)
            "dist_bonus": 0.002,         # 距离优势 (处于最佳射程)
            "height_bonus": 0.001,       # 高度优势 (能量储备)
            
            # --- 2. 动作奖励 (鼓励果断开火) ---
            # 完美发射奖励 2.0 -> 5.0，抵消乱射的负面影响，鼓励 Agent 在 WEZ 内尝试
            "valid_shot_bonus": 5.0,     # 完美发射奖励
            "invalid_shot_penalty": -2.0,# 乱射惩罚 (距离过远/角度过大)
            "redirect_bonus": 1.0,       # 成功重定向导弹奖励
            
            # --- 3. 结果奖励 (核心驱动力) ---
            # 击杀奖励 10.0 -> 100.0
            # 只要拿到一个人头，收益就远超整场比赛“伴飞”的收益。
            "kill_bonus": 100.0,         # 击杀奖励
            "be_shot_penalty": -10.0,    # 被击杀惩罚
            "crash_penalty": -20.0,      # [修改] 加大撞地/出界惩罚，防止为了追击而坠机
            
            # --- 4. [新增] 时间惩罚 ---
            # 每存活一步扣 0.01 分，迫使 Agent 追求快速胜利，而不是拖时间刷分
            "tick_penalty": -0.01,

            # --- 5. 阈值参数 (保持不变) ---
            "wez_max": 35000.0,          # 最大有效射程
            "wez_min": 5000.0,           # 最小安全距离
            "wez_angle": np.deg2rad(30), # 最大发射偏角
            "optimal_dist": 20000.0,     # 期望的最佳交战距离
        }
        if reward_cfg:
            self.cfg.update(reward_cfg)

    def get_reward(self, agent, sim, action_dict, events, redirection_events):
        """
        计算单步总奖励
        """
        # 0. 死亡/非活跃判定
        if not agent.is_active:
            # 检查是否刚刚死亡 (在 events 列表里)
            for e in events:
                # 撞地或出界
                if e['type'] == 'CRASH' and e['uid'] == agent.uid:
                    return self.cfg['crash_penalty']
                # 被击落 (被害者是我)
                if e['type'] == 'KILL' and e['victim'] == agent.uid:
                    return self.cfg['be_shot_penalty']
            return 0.0

        total_reward = 0.0
        
        # === [新增] 1. 时间/生存惩罚 (Tick Penalty) ===
        # 只要活着每一步都扣分，除非能拿到正向奖励抵消
        total_reward += self.cfg['tick_penalty']
        
        # --- 2. 计算态势奖励 (Positional Reward) ---
        # 这里的权重已经被大幅降低
        total_reward += self._calc_position_reward(agent, sim)
        
        # --- 3. 计算开火质量 (Shot Quality) ---
        # 检查是否在本步执行了发射指令
        my_action = action_dict.get(agent.uid)
        if isinstance(my_action, dict) and my_action.get('fire_target'):
            target_uid = my_action['fire_target']
            target = sim.get_entity(target_uid)
            
            # 只有当没有重定向发生时，才视为新发射 (Env 已经处理了逻辑，这里再校验一下)
            # 简单起见，如果 Env 允许了 fire_target 存在，就视为发射了
            shot_rew = self._calc_shot_quality(agent, target)
            total_reward += shot_rew
            
        # --- 4. 处理事件奖励 (Event Reward) ---
        # A. 重定向奖励
        for re in redirection_events:
            if re['launcher'] == agent.uid:
                total_reward += self.cfg['redirect_bonus']
        
        # B. 击杀奖励 (我是凶手)
        for e in events:
            if e['type'] == 'KILL':
                # 解析 killer uid (可能是 M_Red_0_1 -> Red_0)
                killer_uid = e['killer']
                # Env 传来的 killer 可能是导弹ID也可能是发射者ID，具体取决于 sim_core 实现
                # 假设 Simulation.step 中已经处理好归属，或者这里做一个简单判断
                if killer_uid == agent.uid or (f"M_{agent.uid}" in killer_uid):
                    total_reward += self.cfg['kill_bonus']

        return total_reward

    def _calc_position_reward(self, agent, sim):
        """
        计算占位优势：
        1. 角度：我的机头是否指向敌机？
        2. 距离：是否保持在 20km 左右的舒适区？
        """
        # 寻找最近的敌机
        enemies = [e for e in sim.aircrafts if e.team != agent.team and e.is_active]
        if not enemies:
            return 0.0
            
        # 找到最近的敌人
        nearest_enemy = min(enemies, key=lambda e: get_distance(agent.pos, e.pos))
        dist = get_distance(agent.pos, nearest_enemy.pos)
        
        # A. 指向性奖励 (Align Bonus)
        vec_to_enemy = normalize(nearest_enemy.pos - agent.pos)
        my_dir = normalize(agent.vel)
        align_dot = np.dot(my_dir, vec_to_enemy) # [-1, 1]
        
        r_align = align_dot * self.cfg['align_bonus']
        
        # B. 距离控制奖励 (Distance Bonus)
        # 使用高斯函数，在 optimal_dist 处达到峰值
        sigma = 10000.0
        r_dist = np.exp(-((dist - self.cfg['optimal_dist'])**2) / (2 * sigma**2)) * self.cfg['dist_bonus']
        
        # C. 高度/能量奖励 (Height Bonus)
        # 鼓励保持在 5000m - 10000m
        alt = agent.pos[2]
        r_height = 0.0
        if 5000 < alt < 12000:
            r_height = self.cfg['height_bonus']
        elif alt < 2000:
            r_height = -0.05 * (self.cfg['height_bonus'] * 100) # 低空惩罚 (稍微放大一点)
            
        return r_align + r_dist + r_height

    def _calc_shot_quality(self, agent, target):
        """
        评估发射质量：是否在 WEZ 内？
        """
        if not target or not target.is_active:
            return self.cfg['invalid_shot_penalty']
            
        dist = get_distance(agent.pos, target.pos)
        vec_to_target = normalize(target.pos - agent.pos)
        my_dir = normalize(agent.vel)
        
        # 计算偏角 (Off-Boresight Angle)
        angle = np.arccos(np.clip(np.dot(my_dir, vec_to_target), -1, 1))
        
        # 判定条件
        is_range_ok = self.cfg['wez_min'] < dist < self.cfg['wez_max']
        is_angle_ok = angle < self.cfg['wez_angle']
        
        if is_range_ok and is_angle_ok:
            return self.cfg['valid_shot_bonus']
        else:
            # 距离太远、太近或角度太大，都算乱射
            return self.cfg['invalid_shot_penalty']