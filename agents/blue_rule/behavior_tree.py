import numpy as np
from sim_core.maneuver_lib import ManeuverLibrary
from agents.blue_rule.tactics import TacticsLibrary
from utils.geometry import get_distance, normalize, body_to_earth, quat_to_euler

# --- 基础节点类 ---
class Node:
    def tick(self, agent, blackboard):
        """
        :param agent: Aircraft 对象
        :param blackboard: 共享数据字典
        :return: 'SUCCESS', 'FAILURE', 'RUNNING'
        """
        raise NotImplementedError

class Selector(Node):
    """选择器 (OR): 只要有一个子节点成功，就停止并返回成功"""
    def __init__(self, children):
        self.children = children
    def tick(self, agent, blackboard):
        for child in self.children:
            status = child.tick(agent, blackboard)
            if status != 'FAILURE':
                return status
        return 'FAILURE'

class Sequence(Node):
    """序列器 (AND): 所有子节点必须成功"""
    def __init__(self, children):
        self.children = children
    def tick(self, agent, blackboard):
        for child in self.children:
            status = child.tick(agent, blackboard)
            if status != 'SUCCESS':
                return status
        return 'SUCCESS'

# --- 具体的战术感知节点 ---

class CheckIncomingMissile(Node):
    """感知：检查是否有导弹正在锁定我，且距离小于阈值"""
    def __init__(self, danger_dist=20000.0):
        self.danger_dist = danger_dist

    def tick(self, agent, blackboard):
        missiles = blackboard.get('missiles', [])
        
        nearest_threat = None
        min_dist = float('inf')
        
        for m in missiles:
            # 筛选条件：活着的 + 敌方的 + 锁定我的
            if m.is_active and m.team != agent.team and m.target.uid == agent.uid:
                d = get_distance(agent.pos, m.pos)
                if d < min_dist:
                    min_dist = d
                    nearest_threat = m
        
        if nearest_threat and min_dist < self.danger_dist:
            # 将威胁信息写入黑板，供后续动作节点使用
            blackboard['threat_missile'] = nearest_threat
            blackboard['threat_dist'] = min_dist
            # print(f"DEBUG: {agent.uid} detects missile {nearest_threat.uid} at {min_dist:.0f}m")
            return 'SUCCESS'
            
        return 'FAILURE'

# --- 目标态势感知节点 ---

class CheckEnemyContact(Node):
    """感知：检查雷达视角内是否有敌机"""
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
            if dist > self.max_dist:
                continue
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

# --- 具体的战术动作节点 ---

class ActionNotch(Node):
    """
    动作：切向机动 (Notching) / 急转规避
    原理：将导弹置于 3点钟或9点钟方向，并做最大过载机动
    """
    def tick(self, agent, blackboard):
        threat = blackboard.get('threat_missile')
        if not threat:
            return 'FAILURE'
            
        vec_to_missile = threat.pos - agent.pos
        my_vel_dir = normalize(agent.vel)
        target_dir = TacticsLibrary.get_notch_vector(agent.pos, agent.vel, threat.pos)

        cos_err = np.clip(np.dot(my_vel_dir, target_dir), -1.0, 1.0)
        if np.degrees(np.arccos(cos_err)) < 10.0:
            agent_action = ManeuverLibrary.ACTION_MAINTAIN
        else:
            cross_z = my_vel_dir[0] * target_dir[1] - my_vel_dir[1] * target_dir[0]
            if cross_z > 0:
                agent_action = ManeuverLibrary.ACTION_NOTCH_LEFT
            else:
                agent_action = ManeuverLibrary.ACTION_NOTCH_RIGHT
            
        blackboard['output_action'] = agent_action
        return 'SUCCESS'

class ActionCrank(Node):
    """动作：偏置机动 (Crank)"""
    def tick(self, agent, blackboard):
        contact = blackboard.get('contact_enemy')
        if not contact:
            return 'FAILURE'

        my_vel_dir = normalize(agent.vel)
        target_dir = TacticsLibrary.get_crank_vector(agent.pos, contact.pos, radar_fov_deg=55.0)
        cross_z = my_vel_dir[0] * target_dir[1] - my_vel_dir[1] * target_dir[0]

        if cross_z > 0:
            agent_action = ManeuverLibrary.ACTION_CRANK_LEFT
        else:
            agent_action = ManeuverLibrary.ACTION_CRANK_RIGHT

        blackboard['output_action'] = agent_action
        return 'SUCCESS'

class ActionPatrol(Node):
    def __init__(self):
        self.tick_counter = 0
    """动作：巡航/接敌"""
    def tick(self, agent, blackboard):
        # 如果没有威胁，保持平飞或向敌机接近
        # 这里简化为 Action 0 (Maintain)
        self.tick_counter += 1
        cycle = 100 # 每100帧一个周期
        phase = self.tick_counter % cycle
        
        if phase < 30:
            action = ManeuverLibrary.ACTION_LEFT_TURN # 这里的 Turn 是温和的平转
        elif phase < 60:
            action = ManeuverLibrary.ACTION_RIGHT_TURN
        else:
            action = ManeuverLibrary.ACTION_MAINTAIN
            
        blackboard['output_action'] = action
        return 'SUCCESS'

# --- 蓝方智能体封装 ---

class BlueRuleAgent:
    def __init__(self, uid, team):
        self.uid = uid
        self.team = team
        self.blackboard = {} 
        
        # 构建行为树
        # 逻辑优先级：
        # 1. 如果有致命威胁 (15km内) -> 执行急转规避 (Notch/Break)
        # 2. 如果发现敌机 -> 执行偏置机动 (Crank)
        # 3. 否则 -> 巡航 (Patrol)
        self.behavior_tree = Selector([
            Sequence([
                CheckIncomingMissile(danger_dist=15000.0),
                ActionNotch()
            ]),
            Sequence([
                CheckEnemyContact(max_dist=50000.0),
                ActionCrank()
            ]),
            ActionPatrol()
        ])

    def get_action(self, my_aircraft, all_missiles, all_aircrafts=None):
        """
        主入口：输入状态，输出动作ID
        """
        # 清空上一帧的黑板，填入新数据
        enemies = all_aircrafts or []
        self.blackboard = {
            'missiles': all_missiles,
            'enemies': enemies,
            'output_action': 0
        }
        
        # 运行行为树
        self.behavior_tree.tick(my_aircraft, self.blackboard)
        
        return self.blackboard['output_action']
