import numpy as np
import yaml
from utils.geometry import get_distance, normalize
from sim_core.simulation import Simulation
from envs.obs_parser import ObservationParser
from envs.reward_func import RewardFunction
from agents.blue_rule.behavior_tree import BlueRuleAgent

class CombatEnv_8v8:
    def __init__(self, config_path="configs/env_config.yaml"):
        self.sim = None
        self.obs_parser = ObservationParser()
        self.reward_func = None
        self.blue_agents = {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        self.max_steps = self.cfg['max_steps']
        self.current_step = 0
        
        # 初始化新的奖励函数
        self.reward_func = RewardFunction(
            team_id=0,
            reward_cfg=self.cfg.get("rewards", {})
        )

    def reset(self):
        self.current_step = 0
        self.sim = Simulation()
        self.sim.reset_8v8(self.cfg.get("init_state", {}))
        
        self.blue_agents = {}
        for p in self.sim.aircrafts:
            if p.team == 1:
                self.blue_agents[p.uid] = BlueRuleAgent(p.uid, p.team)
        return self._get_all_observations()

    def step(self, red_actions):
        """
        red_actions: {uid: {'maneuver': int, 'fire_target': target_uid}}
        """
        self.current_step += 1
        
        # --- 1. 预处理动作与重定向逻辑 ---
        processed_red_actions = {}
        redirection_events = [] 
        
        for uid, cmd in red_actions.items():
            if not isinstance(cmd, dict):
                processed_red_actions[uid] = cmd
                continue
                
            maneuver_id = cmd.get('maneuver', 0)
            target_uid = cmd.get('fire_target')
            
            processed_red_actions[uid] = {
                'maneuver': maneuver_id,
                'fire_target': target_uid 
            }
            
            # 重定向判定 (Step 2 逻辑)
            if target_uid:
                my_missiles = [m for m in self.sim.missiles 
                               if m.is_active and m.launcher_uid == uid]
                lost_missiles = [m for m in my_missiles if m.lost_lock]
                
                if lost_missiles:
                    missile_to_redirect = lost_missiles[0]
                    target_entity = self.sim.get_entity(target_uid)
                    
                    if target_entity and target_entity.is_active:
                        missile_to_redirect.target = target_entity
                        missile_to_redirect.lost_lock = False
                        
                        redirection_events.append({
                            'launcher': uid,
                            'missile': missile_to_redirect.uid,
                            'new_target': target_uid,
                            'type': 'REDIRECT'
                        })
                        
                        # 重定向了就不发射新弹
                        processed_red_actions[uid]['fire_target'] = None
        
        # --- 2. 蓝方决策 ---
        blue_actions = {}
        for uid, agent in self.blue_agents.items():
            aircraft = self.sim.get_entity(uid)
            if aircraft and aircraft.is_active:
                act_id = agent.get_action(aircraft, self.sim.missiles, self.sim.aircrafts)
                blue_actions[uid] = act_id
        
        # --- 3. 仿真步进 ---
        sim_events = self.sim.step(processed_red_actions, blue_actions)
        
        # --- 4. 合并事件 ---
        all_events = sim_events + redirection_events
        
        # --- 5. 计算奖励 ---
        rewards = {}
        for p in self.sim.aircrafts:
            if p.team == 0:
                # 即使 agent 死了，也要结算事件奖励
                r = self.reward_func.get_reward(
                    p, self.sim, processed_red_actions, all_events, redirection_events
                )
                rewards[p.uid] = r

        # --- 6. 观测与结束判定 ---
        obs = self._get_all_observations()
        dones = {"__all__": False}
        infos = {"events": all_events} # 基础 info

        # === [核心修复] 补全 Trainer 所需的统计字段 ===
        # 1. 开火统计
        fire_count = sum(1 for e in all_events if e.get('type') == 'FIRE')
        
        # 2. 平均速度
        red_active_vels = [np.linalg.norm(p.vel) for p in self.sim.aircrafts if p.team == 0 and p.is_active]
        mean_speed = np.mean(red_active_vels) if red_active_vels else 0.0
        
        # 3. 平均交战距离
        red_pos = [p.pos for p in self.sim.aircrafts if p.team == 0 and p.is_active]
        blue_pos = [p.pos for p in self.sim.aircrafts if p.team == 1 and p.is_active]
        mean_dist = 0.0
        if red_pos and blue_pos:
            dists = []
            for rp in red_pos:
                for bp in blue_pos:
                    dists.append(np.linalg.norm(rp - bp))
            mean_dist = np.mean(dists) if dists else 0.0
            
        # 写入 infos
        infos['mean_speed'] = mean_speed
        infos['mean_dist'] = mean_dist
        infos['fire_count'] = fire_count
        # ==========================================

        # 统计存活
        red_alive = any(p.is_active for p in self.sim.aircrafts if p.team == 0)
        blue_alive = any(p.is_active for p in self.sim.aircrafts if p.team == 1)
        
        if not red_alive or not blue_alive or self.current_step >= self.max_steps:
            dones["__all__"] = True

        return obs, rewards, dones, infos

    def _get_all_observations(self):
        obs_dict = {}
        for p in self.sim.aircrafts:
            if p.team == 0 and p.is_active:
                obs_dict[p.uid] = self.obs_parser.get_obs(self.sim, p)
        return obs_dict