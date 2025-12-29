import numpy as np
from utils.geometry import get_distance, get_vector, normalize, body_to_earth, quat_to_euler

class ObservationParser:
    def __init__(self, max_allies=8, max_enemies=8, max_missiles=24):
        self.max_allies = max_allies
        self.max_enemies = max_enemies
        self.max_missiles = max_missiles
        
        # 特征维度
        # Self: [x, y, z, vx, vy, vz, roll, pitch, yaw, hp, missile_count] = 11
        # Other Aircraft: [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, is_alive] = 7
        # Missile: [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, is_locked_on_me] = 7
        
    def get_obs(self, sim, agent):
        """
        为主视角 agent 生成观察向量
        """
        # 1. 自身特征 (Global Norm 处理，防止数值过大)
        self_state = np.array([
            agent.pos[0] / 50000.0, agent.pos[1] / 50000.0, agent.pos[2] / 15000.0,
            agent.vel[0] / 1000.0,  agent.vel[1] / 1000.0,  agent.vel[2] / 1000.0,
            agent.dynamics.quat[0], agent.dynamics.quat[1], agent.dynamics.quat[2], agent.dynamics.quat[3],
            agent.missile_count / 3.0
        ], dtype=np.float32)

        # 2. 盟友特征 (Allies)
        allies_feat = np.zeros((self.max_allies, 7), dtype=np.float32)
        idx = 0
        for other in sim.aircrafts:
            if other.team == agent.team and other.uid != agent.uid:
                if idx >= self.max_allies: break
                if other.is_active:
                    rel_pos = (other.pos - agent.pos) / 50000.0
                    rel_vel = (other.vel - agent.vel) / 1000.0
                    allies_feat[idx] = [*rel_pos, *rel_vel, 1.0] # 1.0 is alive
                idx += 1
        
        # 3. 敌机特征 (Enemies)
        enemies_feat = np.zeros((self.max_enemies, 7), dtype=np.float32)
        idx = 0
        for other in sim.aircrafts:
            if other.team != agent.team:
                if idx >= self.max_enemies: break
                if other.is_active:
                    rel_pos = (other.pos - agent.pos) / 50000.0
                    rel_vel = (other.vel - agent.vel) / 1000.0
                    enemies_feat[idx] = [*rel_pos, *rel_vel, 1.0]
                idx += 1

        # 4. 导弹特征 (Missiles)
        missiles_feat = np.zeros((self.max_missiles, 7), dtype=np.float32)
        idx = 0
        for m in sim.missiles:
            if not m.is_active: continue
            if idx >= self.max_missiles: break
            
            # 只关注：我对它的威胁(如果是敌方导弹) 或 它对我的支援(如果是友方)
            # 这里简化：不仅看敌方导弹，也看友方导弹（避免重复攻击同一目标）
            
            rel_pos = (m.pos - agent.pos) / 50000.0
            rel_vel = (m.vel - agent.vel) / 1000.0
            
            # Feature: Is this missile attacking me?
            is_attacking_me = 1.0 if (m.target and m.target.uid == agent.uid) else 0.0
            
            missiles_feat[idx] = [*rel_pos, *rel_vel, is_attacking_me]
            idx += 1

        return {
            "self": self_state,
            "allies": allies_feat,
            "enemies": enemies_feat,
            "missiles": missiles_feat
        }