import numpy as np
from sim_core.entities.base import Entity
from sim_core.dynamics import FlightDynamics
from sim_core.maneuver_lib import ManeuverLibrary
from utils.geometry import euler_to_quat, quat_to_euler

class Aircraft(Entity):
    def __init__(self, uid, team, init_pos, init_vel, init_heading=0):
        super().__init__(uid, team)
        
        # 初始化姿态
        q = euler_to_quat(0, 0, init_heading)
        
        # [Step 1 兼容] 初始化新的运动学核心
        self.dynamics = FlightDynamics(init_pos, init_vel, q)
        self.maneuver_lib = ManeuverLibrary()
        
        # 战斗属性
        self.hp = 100
        # 红方载弹量 3 枚 (Step 2 逻辑依赖 missile_count 来生成 missile uid)
        self.missile_count = 3 if team == 0 else 0 
        self.radar_angle = np.deg2rad(60) # 雷达半宽视角
        
        # 同步状态
        self.pos = self.dynamics.pos.copy()
        self.vel = self.dynamics.vel.copy()
        
    def step(self, action_id, dt):
        """
        :param action_id: 机动动作ID (0-10)
        """
        if not self.is_active:
            return
            
        # 1. 动作转指令
        # [Step 1 修正] 新版 get_action_cmd 不需要 current_roll 参数
        cmd = self.maneuver_lib.get_action_cmd(action_id)
        
        # 2. 物理步进
        # FlightDynamics.step 返回: pos, vel, speed, n
        self.pos, self.vel, speed, n = self.dynamics.step(cmd, dt)
        
        # 3. 撞地判定 (简单的底层保护)
        if self.pos[2] <= 0:
            self.hp = 0
            self.is_active = False

    def get_state(self):
        # 返回给RL的状态向量 [pos(3), vel(3)]
        return np.concatenate([self.pos, self.vel])