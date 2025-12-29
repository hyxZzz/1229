import numpy as np
from sim_core.maneuver_lib import ManeuverLibrary
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
            
        # 1. 计算导弹相对于我的方位
        # 将导弹位置转到机体坐标系，或者简单通过叉乘判断左右
        vec_to_missile = threat.pos - agent.pos
        
        # 转换到机体坐标系需要逆旋转，这里用简单的 Cross Product 判断左右
        # 获取机体的前向矢量 (Heading)
        q = agent.dynamics.quat
        my_vel_dir = normalize(agent.vel)
        
        # 计算 Relative Vector
        rel_vec = normalize(vec_to_missile)
        
        # 叉乘：(My_Vel) x (To_Missile)
        # 结果的 Z 分量 > 0 表示导弹在左边，< 0 表示在右边
        cross_z = my_vel_dir[0] * rel_vec[1] - my_vel_dir[1] * rel_vec[0]
        
        # 战术决策：
        # 如果导弹在左边，我应该向左急转（迎头闪避）还是向右急转（摆脱）？
        # 通常 Notch 是要让导弹矢量与速度矢量垂直。
        # 如果导弹在左前方，向左转会迅速减小距离（危险）；向右转会拉大角度。
        # 这里采用 High-G Break 逻辑：向着导弹来袭方向的“反侧”急转，或者向导弹方向急转迫使其过冲。
        # 简化策略：导弹在哪边，就往哪边急转 (Break Into)，这在近距格斗中能迫使导弹过载饱和。
        
        if cross_z > 0: 
            # 导弹在左 -> 向左急转 (Action 5: Break Left)
            agent_action = ManeuverLibrary.ACTION_BREAK_LEFT
        else:
            # 导弹在右 -> 向右急转 (Action 6: Break Right)
            agent_action = ManeuverLibrary.ACTION_BREAK_RIGHT
            
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
        # 2. 否则 -> 巡航 (Patrol)
        self.behavior_tree = Selector([
            Sequence([
                CheckIncomingMissile(danger_dist=15000.0),
                ActionNotch()
            ]),
            ActionPatrol()
        ])

    def get_action(self, my_aircraft, all_missiles):
        """
        主入口：输入状态，输出动作ID
        """
        # 清空上一帧的黑板，填入新数据
        self.blackboard = {'missiles': all_missiles, 'output_action': 0}
        
        # 运行行为树
        self.behavior_tree.tick(my_aircraft, self.blackboard)
        
        return self.blackboard['output_action']