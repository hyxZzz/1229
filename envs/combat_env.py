import numpy as np
import yaml
from sim_core.simulation import Simulation
from envs.obs_parser import ObservationParser
from envs.reward_func import RewardFunction
from agents.blue_rule.behavior_tree import BlueRuleAgent

class CombatEnv_8v8:
    def __init__(self,config_path="configs/env_config.yaml"):
        # 初始化核心组件
        self.sim = None
        self.obs_parser = ObservationParser()
        self.reward_func = None
        
        # 蓝方规则智能体池
        self.blue_agents = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        self.max_steps = self.cfg['max_steps']
        self.current_step = 0
        self.reward_func = RewardFunction(
            team_id=0,
            reward_cfg=self.cfg.get("rewards", {}),
            engage_cfg=self.cfg.get("engagement", {}),
        ) # 红方奖励

    def reset(self):
        self.current_step = 0
        self.sim = Simulation() # 需要在 sim_core/simulation.py 中实现初始化逻辑
        self.sim.reset_8v8(self.cfg.get("init_state", {}))    # 初始化 8v8 布局
        
        # 初始化蓝方规则 Agent
        self.blue_agents = {}
        for p in self.sim.aircrafts:
            if p.team == 1: # Blue
                self.blue_agents[p.uid] = BlueRuleAgent(p.uid, p.team)
                
        return self._get_all_observations()

    def step(self, red_actions):
        """
        red_actions: Dict {uid: (action_id, fire_cmd_target_id)}
        """
        self.current_step += 1
        
        # 1. 获取蓝方动作 (规则生成)
        blue_actions = {}
        for uid, agent in self.blue_agents.items():
            # 从 sim 中找到对应的实体对象
            aircraft = self.sim.get_entity(uid)
            if aircraft and aircraft.is_active:
                act_id = agent.get_action(aircraft, self.sim.missiles, self.sim.aircrafts)
                blue_actions[uid] = act_id
        
        # 2. 执行仿真步进 (红方 + 蓝方)
        # 需要将 red_actions 和 blue_actions 合并传给 sim
        events = self.sim.step(red_actions, blue_actions)
        
        # 3. 计算奖励与观察
        obs = self._get_all_observations()
        rewards = {}
        dones = {"__all__": False}
        infos = {}

        red_speeds = []
        red_dists = []
        fires_this_step = 0

        # 统计开火数
        for e in events:
            if e['type'] == 'FIRE':
                launcher = self.sim.get_entity(e['launcher'])
                if launcher and launcher.team == 0:
                    fires_this_step += 1


        # 统计物理状态
        for p in self.sim.aircrafts:
            if p.team == 0 and p.is_active:
                # 1. 速度
                speed = np.linalg.norm(p.vel)
                red_speeds.append(speed)
                
                # 2. 距离最近敌人的距离
                min_d = 200000.0
                for enemy in self.sim.aircrafts:
                    if enemy.team == 1 and enemy.is_active:
                        d = np.linalg.norm(p.pos - enemy.pos)
                        if d < min_d: min_d = d
                if min_d < 190000: # 只有存在敌人才记录
                    red_dists.append(min_d)

        infos = {
            "mean_speed": np.mean(red_speeds) if red_speeds else 0.0,
            "mean_dist": np.mean(red_dists) if red_dists else 100000.0, # 默认100km
            "fire_count": fires_this_step
        }            
        
        # 处理事件奖励 (击杀/被击杀)
        # events 包含: [{'type': 'KILL', 'killer': uid, 'victim': uid}, ...]
        kill_reward_map = {}
        for e in events:
            if e['type'] == 'KILL':
                killer = self.sim.get_entity(e['killer'])
                if killer and killer.team == 0:
                    kill_reward_map[e['killer']] = self.cfg["rewards"].get("kill_bonus", 10.0)
        
        # 为每个红方智能体计算 Step 奖励
        active_reds = 0
        for p in self.sim.aircrafts:
            if p.team == 0: # Red
                if p.is_active:
                    active_reds += 1
                    base_rew = self.reward_func.compute_reward(p, self.sim, red_actions)
                    event_rew = kill_reward_map.get(p.uid, 0.0)
                    rewards[p.uid] = base_rew + event_rew
                else:
                    rewards[p.uid] = self.cfg["rewards"].get("be_shot_penalty", -5.0) # Dead
        
        # 判定结束
        if active_reds == 0 or self.current_step >= self.max_steps:
            dones["__all__"] = True
            
        # 检查蓝方是否全灭
        blue_alive = sum([1 for p in self.sim.aircrafts if p.team == 1 and p.is_active])
        if blue_alive == 0:
            dones["__all__"] = True
            # 给所有红方存活者加 50 分大奖
            team_win_bonus = self.cfg["rewards"].get("team_win_bonus", 50.0)
            for uid in rewards:
                rewards[uid] += team_win_bonus

        return obs, rewards, dones, infos

    def _get_all_observations(self):
        obs_dict = {}
        for p in self.sim.aircrafts:
            if p.team == 0 and p.is_active:
                obs_dict[p.uid] = self.obs_parser.get_obs(self.sim, p)
        return obs_dict
