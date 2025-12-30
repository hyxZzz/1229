import numpy as np
import math as m

class ManeuverLibrary:
    # 动作定义: [加速标识, 目标G值, 滚转角(rad), 垂直标识(备用)]
    # acc_flag: 0=保持, 2=加速, -2=减速
    # target_g: 目标过载 (G)
    # roll: 滚转角 (弧度)
    # flag: -1 通常表示纯垂直面机动 (在此简化模型中主要由 G 和 Roll 决定，可作为参考)
    
    MANEUVERS = {
        0: [0, 1, 0, 0],                # 匀速前飞 (Maintain)
        1: [2, 1, 0, 0],                # 加速前飞 (Accelerate)
        2: [-2, 1, 0, 0],               # 减速前飞 (Decelerate)
        3: [0, 2, 0, -1],               # 爬升 (Pull up)
        4: [0, 0.5, 0, -1],             # 俯冲 (Push down) - 修正: G=0可能导致完全失控，0.5G为适度推杆
        5: [0, 2, 0.25 * m.pi, -1],     # 左爬升 (Left Climb)
        6: [0, 2, -0.25 * m.pi, -1],    # 右爬升 (Right Climb)
        7: [0, 2, -0.75 * m.pi, -1],    # 左俯冲 (Left Dive) - 滚转135度拉杆
        8: [0, 2, 0.75 * m.pi, -1],     # 右俯冲 (Right Dive)
        9: [0, 2, m.acos(1 / 2), 0],    # 左转弯 (Left Turn) - 60度滚转 2G = 稳定盘旋
        10: [0, 2, -m.acos(1 / 2), 0],  # 右转弯 (Right Turn)
    }

    # 为了兼容 RL 动作空间定义 (0-10)
    ACTION_MAINTAIN = 0
    ACTION_ACCEL = 1
    ACTION_DECEL = 2
    ACTION_CLIMB = 3
    ACTION_DIVE = 4
    ACTION_LEFT_CLIMB = 5
    ACTION_RIGHT_CLIMB = 6
    ACTION_LEFT_DIVE = 7
    ACTION_RIGHT_DIVE = 8
    ACTION_LEFT_TURN = 9
    ACTION_RIGHT_TURN = 10

    def __init__(self):
        pass

    def get_action_cmd(self, action_id: int) -> dict:
        """
        根据 ID 返回具体的机动参数
        """
        # 默认返回匀速前飞
        params = self.MANEUVERS.get(action_id, [0, 1, 0, 0])
        
        return {
            'acc_flag': params[0],
            'target_g': params[1],
            'target_roll': params[2],
            'flag': params[3]
        }