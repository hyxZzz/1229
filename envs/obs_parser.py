import numpy as np
from utils.geometry import get_distance, get_vector, normalize, body_to_earth, quat_to_euler

class ObservationParser:
    def __init__(self, max_allies=8, max_enemies=8, max_missiles=24):
        self.max_allies = max_allies
        self.max_enemies = max_enemies
        self.max_missiles = max_missiles
        
        # 归一化常数定义 (基于 env_config.yaml)
        # 地图范围 +/- 150,000 -> 设为 150,000
        self.NORM_POS = 150000.0  
        # 相对距离关注 100km (雷达范围) -> 设为 100,000 比较合适
        self.NORM_REL_POS = 100000.0
        # 高度上限 25,000 -> 设为 30,000
        self.NORM_ALT = 30000.0
        # 最大速度约 600m/s -> 设为 1000
        self.NORM_VEL = 1000.0
        
    def get_obs(self, sim, agent):
        """
        为主视角 agent 生成观察向量
        """
        # 1. 自身特征 (使用地图级归一化)
        # Self: [x, y, z, vx, vy, vz, q0, q1, q2, q3, missile]
        self_state = np.array([
            agent.pos[0] / self.NORM_POS, 
            agent.pos[1] / self.NORM_POS, 
            agent.pos[2] / self.NORM_ALT,
            agent.vel[0] / self.NORM_VEL,  
            agent.vel[1] / self.NORM_VEL,  
            agent.vel[2] / self.NORM_VEL,
            agent.dynamics.quat[0], 
            agent.dynamics.quat[1], 
            agent.dynamics.quat[2], 
            agent.dynamics.quat[3],
            agent.missile_count / 3.0
        ], dtype=np.float32)

        # 2. 盟友特征 (使用相对距离归一化)
        allies_feat = np.zeros((self.max_allies, 7), dtype=np.float32)
        idx = 0
        for other in sim.aircrafts:
            if other.team == agent.team and other.uid != agent.uid:
                if idx >= self.max_allies: break
                if other.is_active:
                    # 相对位置
                    rel_pos = (other.pos - agent.pos) / self.NORM_REL_POS
                    # 相对速度
                    rel_vel = (other.vel - agent.vel) / self.NORM_VEL
                    
                    # 拼接: [rx, ry, rz, rvx, rvy, rvz, alive]
                    allies_feat[idx] = [*rel_pos, *rel_vel, 1.0] 
                idx += 1
        
        # 3. 敌机特征
        enemies_feat = np.zeros((self.max_enemies, 7), dtype=np.float32)
        idx = 0
        for other in sim.aircrafts:
            if other.team != agent.team:
                if idx >= self.max_enemies: break
                if other.is_active:
                    rel_pos = (other.pos - agent.pos) / self.NORM_REL_POS
                    rel_vel = (other.vel - agent.vel) / self.NORM_VEL
                    enemies_feat[idx] = [*rel_pos, *rel_vel, 1.0]
                idx += 1

        # 4. 导弹特征
        missiles_feat = np.zeros((self.max_missiles, 7), dtype=np.float32)
        idx = 0
        for m in sim.missiles:
            if not m.is_active: continue
            if idx >= self.max_missiles: break
            
            rel_pos = (m.pos - agent.pos) / self.NORM_REL_POS
            rel_vel = (m.vel - agent.vel) / self.NORM_VEL
            
            # 是否在攻击我？
            is_attacking_me = 1.0 if (m.target and m.target.uid == agent.uid) else 0.0
            
            missiles_feat[idx] = [*rel_pos, *rel_vel, is_attacking_me]
            idx += 1

        return {
            "self": self_state,
            "allies": allies_feat,
            "enemies": enemies_feat,
            "missiles": missiles_feat
        }