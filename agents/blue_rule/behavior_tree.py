import numpy as np
from sim_core.maneuver_lib import ManeuverLibrary
from agents.blue_rule.tactics import TacticsLibrary
from utils.geometry import get_distance, normalize

# --- 基础节点类 ---
class Node:
    def tick(self, agent, blackboard):
        raise NotImplementedError

class Selector(Node):
    def __init__(self, children):
        self.children = children
    def tick(self, agent, blackboard):
        for child in self.children:
            status = child.tick(agent, blackboard)
            if status != 'FAILURE':
                return status
        return 'FAILURE'

class Sequence(Node):
    def __init__(self, children):
        self.children = children
    def tick(self, agent, blackboard):
        for child in self.children:
            status = child.tick(agent, blackboard)
            if status != 'SUCCESS':
                return status
        return 'SUCCESS'

# --- 具体的战术感知节点 (逻辑保持不变) ---

class CheckIncomingMissile(Node):
    def __init__(self, danger_dist=15000.0):
        self.danger_dist = danger_dist

    def tick(self, agent, blackboard):
        missiles = blackboard.get('missiles', [])
        nearest_threat = None
        min_dist = float('inf')
        
        # 遍历所有敌方且锁定我的导弹
        for m in missiles:
            if m.is_active and m.team != agent.team and m.target and m.target.uid == agent.uid:
                d = get_distance(agent.pos, m.pos)
                if d < min_dist:
                    min_dist = d
                    nearest_threat = m
        
        if nearest_threat and min_dist < self.danger_dist:
            blackboard['threat_missile'] = nearest_threat
            blackboard['threat_dist'] = min_dist
            return 'SUCCESS'
        return 'FAILURE'

class CheckEnemyContact(Node):
    def __init__(self, max_dist=50000.0):
        self.max_dist = max_dist

    def tick(self, agent, blackboard):
        enemies = blackboard.get('enemies', [])
        if not enemies:
            return 'FAILURE'

        my_vel_dir = normalize(agent.vel)
        nearest_enemy = None
        min_dist = float('inf')

        for e in enemies:
            if not e.is_active or e.team == agent.team:
                continue
            vec_to_enemy = e.pos - agent.pos
            dist = np.linalg.norm(vec_to_enemy)
            
            # 距离过滤
            if dist > self.max_dist:
                continue
            
            # 视场角过滤 (假设雷达在机头)
            los = normalize(vec_to_enemy)
            cos_angle = np.dot(my_vel_dir, los)
            if cos_angle < np.cos(agent.radar_angle):
                continue
                
            if dist < min_dist:
                min_dist = dist
                nearest_enemy = e

        if nearest_enemy:
            blackboard['contact_enemy'] = nearest_enemy
            blackboard['contact_dist'] = min_dist
            return 'SUCCESS'
        return 'FAILURE'

# --- 具体的战术动作节点 (适配离散动作) ---

class ActionNotch(Node):
    def tick(self, agent, blackboard):
        threat = blackboard.get('threat_missile')
        if not threat:
            return 'FAILURE'
            
        # 1. 计算理想的 Notch 矢量 (垂直于导弹来袭方向)
        ideal_dir = TacticsLibrary.get_notch_vector(agent.pos, agent.vel, threat.pos)
        
        # 2. 如果速度过低，强制俯冲加速 (Energy Protection)
        speed = np.linalg.norm(agent.vel)
        if speed < 150.0:
            blackboard['output_action'] = ManeuverLibrary.ACTION_DIVE
            return 'SUCCESS'
            
        # 3. 在 11 个离散动作中选择最能贴合 ideal_dir 的
        best_action = select_best_discrete_action(agent, ideal_dir)
        
        blackboard['output_action'] = best_action
        return 'SUCCESS'

class ActionCrank(Node):
    def tick(self, agent, blackboard):
        contact = blackboard.get('contact_enemy')
        if not contact:
            return 'FAILURE'

        # 1. 计算理想的 Crank 矢量
        ideal_dir = TacticsLibrary.get_crank_vector(agent.pos, contact.pos, radar_fov_deg=55.0)

        # 2. 选择最佳离散动作
        best_action = select_best_discrete_action(agent, ideal_dir)

        blackboard['output_action'] = best_action
        return 'SUCCESS'

class ActionPatrol(Node):
    def __init__(self):
        self.tick_counter = 0

    def tick(self, agent, blackboard):
        self.tick_counter += 1
        cycle = 200 # 10秒一个周期
        phase = self.tick_counter % cycle
        
        # 简单的巡逻模式：左转一会 -> 平飞 -> 右转一会 -> 平飞
        if phase < 40:
            action = ManeuverLibrary.ACTION_LEFT_TURN 
        elif phase < 100:
            action = ManeuverLibrary.ACTION_MAINTAIN
        elif phase < 140:
            action = ManeuverLibrary.ACTION_RIGHT_TURN
        else:
            action = ManeuverLibrary.ACTION_MAINTAIN
            
        blackboard['output_action'] = action
        return 'SUCCESS'

# --- 核心辅助函数：离散动作选择器 ---

def select_best_discrete_action(agent, ideal_dir):
    """
    遍历 ManeuverLibrary 中的所有动作，预测下一帧的速度方向，
    选择与 ideal_dir 余弦相似度最高的动作。
    """
    best_action = ManeuverLibrary.ACTION_MAINTAIN
    max_score = -2.0 # Cosine similarity range [-1, 1]
    
    # 实例化库以获取参数
    lib = ManeuverLibrary()
    
    # 遍历动作 ID 0-10
    for action_id in range(11):
        # 获取该动作的动力学参数
        cmd = lib.get_action_cmd(action_id)
        
        # 预测该动作产生的方向
        pred_dir = _predict_velocity_direction(agent.vel, cmd)
        
        # 计算得分 (Dot Product)
        score = np.dot(pred_dir, ideal_dir)
        
        if score > max_score:
            max_score = score
            best_action = action_id
            
    return best_action

def _predict_velocity_direction(current_vel, cmd, dt=0.5):
    """
    轻量级的动力学预测，逻辑与 FlightDynamics.step 保持一致。
    dt 稍微取大一点(如0.5s)可以让预测更具前瞻性。
    """
    target_g = cmd['target_g']
    target_roll = cmd['target_roll']
    
    # 1. 构建坐标系
    speed = np.linalg.norm(current_vel)
    if speed < 1.0: return np.array([1,0,0])
    
    v_norm = current_vel / speed
    world_up = np.array([0, 0, 1])
    
    right = np.cross(v_norm, world_up)
    if np.linalg.norm(right) < 1e-3:
        right = np.array([1, 0, 0])
    right = normalize(right)
    
    body_up = np.cross(right, v_norm)
    
    # 2. 计算升力方向 (Lift Vector)
    sin_r = np.sin(target_roll)
    cos_r = np.cos(target_roll)
    lift_dir = normalize(body_up * cos_r + right * sin_r)
    
    # 3. 计算转弯加速度 (不考虑加减速，只看方向变化)
    g0 = 9.81
    acc_turn = lift_dir * (target_g * g0) + np.array([0, 0, -g0])
    
    # 去除切向分量 (只保留垂直分量)
    acc_turn_perp = acc_turn - v_norm * np.dot(acc_turn, v_norm)
    
    # 4. 预测新速度矢量
    v_new = current_vel + acc_turn_perp * dt
    return normalize(v_new)

# --- Agent 主类 ---

class BlueRuleAgent:
    def __init__(self, uid, team):
        self.uid = uid
        self.team = team
        self.blackboard = {} 
        
        # 行为树结构：优先躲避导弹 -> 其次维持接触(Crank) -> 最后巡逻
        self.behavior_tree = Selector([
            Sequence([
                CheckIncomingMissile(danger_dist=15000.0),
                ActionNotch()
            ]),
            Sequence([
                CheckEnemyContact(max_dist=60000.0),
                ActionCrank()
            ]),
            ActionPatrol()
        ])

    def get_action(self, my_aircraft, all_missiles, all_aircrafts=None):
        enemies = all_aircrafts or []
        self.blackboard = {
            'missiles': all_missiles,
            'enemies': enemies,
            'output_action': 0 # Default Maintain
        }
        self.behavior_tree.tick(my_aircraft, self.blackboard)
        return self.blackboard['output_action']