import numpy as np
from sim_core.entities.base import Entity
from sim_core.dynamics import FlightDynamics
from sim_core.maneuver_lib import ManeuverLibrary
from utils.geometry import euler_to_quat

class Aircraft(Entity):
    def __init__(self, uid, team, init_pos, init_vel, init_heading=0):
        super().__init__(uid, team)
        # 初始化物理核心
        q = euler_to_quat(0, 0, init_heading)
        self.dynamics = FlightDynamics(init_pos, init_vel, q)
        self.maneuver_lib = ManeuverLibrary()
        
        # 战斗属性
        self.hp = 100
        self.missile_count = 3 if team == 0 else 0 # 只有红方带弹
        self.radar_angle = np.deg2rad(60) # 雷达视角
        
        self.pos = self.dynamics.pos.copy()
        self.vel = self.dynamics.vel.copy()
        
    def step(self, action_id, dt):
        """
        :param action_id: 机动动作ID (Int)
        """
        if not self.is_active:
            return
            
        # 1. 动作转指令
        curr_roll = self.dynamics.quat[0] # 简化获取roll
        cmd = self.maneuver_lib.get_action_cmd(action_id, curr_roll)
        
        # 2. 物理步进
        self.pos, self.vel, speed, n = self.dynamics.step(cmd, dt)
        
        # 3. 坠地判定
        if self.pos[2] <= 0:
            self.hp = 0
            self.is_active = False

    def get_state(self):
        # 返回给RL的状态向量 (简化版)
        return np.concatenate([self.pos, self.vel])