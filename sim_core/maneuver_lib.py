import numpy as np

class ManeuverLibrary:
    """
    机动动作库：将离散动作ID转换为连续的控制指令
    Control Cmd Format: {
        'target_g': float,
        'target_roll': float,
        'target_pitch_rate': float,
        'throttle': float
    }
    """
    # 动作定义
    ACTION_MAINTAIN = 0  # 保持当前状态
    ACTION_LEFT_TURN = 1 # 左平转
    ACTION_RIGHT_TURN = 2 # 右平转
    ACTION_CLIMB = 3     # 爬升
    ACTION_DIVE = 4      # 俯冲
    ACTION_BREAK_LEFT = 5 # 左急转 (高过载)
    ACTION_BREAK_RIGHT = 6 # 右急转
    ACTION_PITCH_UP = 7  # 俯仰上仰
    ACTION_PITCH_DOWN = 8 # 俯仰下压
    ACTION_NOTCH_LEFT = 9 # 切向机动-左
    ACTION_NOTCH_RIGHT = 10 # 切向机动-右
    ACTION_CRANK_LEFT = 11 # 偏置机动-左
    ACTION_CRANK_RIGHT = 12 # 偏置机动-右
    ACTION_BARREL_ROLL = 13 # 桶滚
    ACTION_SLICING_TURN = 14 # 切片转弯
    ACTION_IMMELMANN = 15 # 半筋斗
    ACTION_SPLIT_S = 16 # 半滚俯冲
    ACTION_HIGH_YOYO = 17 # 高 Yo-Yo
    ACTION_LOW_YOYO = 18 # 低 Yo-Yo
    
    def __init__(self):
        pass

    def _coordinated_turn_g(self, roll_rad: float) -> float:
        cos_roll = np.cos(roll_rad)
        if abs(cos_roll) < 1e-3:
            return 9.0
        return 1.0 / cos_roll

    def get_action_cmd(self, action_id: int, current_roll: float) -> dict:
        cmd = {
            'throttle': 1.0, # 默认巡航油门
            'target_g': 1.0, # 默认1G平飞
            'target_roll': 0.0,
            'target_pitch_rate': 0.0
        }
        
        if action_id == self.ACTION_MAINTAIN:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 1.0
            
        elif action_id == self.ACTION_LEFT_TURN:
            cmd['target_roll'] = -np.deg2rad(45)
            cmd['target_g'] = self._coordinated_turn_g(cmd['target_roll'])
            
        elif action_id == self.ACTION_RIGHT_TURN:
            cmd['target_roll'] = np.deg2rad(45)
            cmd['target_g'] = self._coordinated_turn_g(cmd['target_roll'])
            
        elif action_id == self.ACTION_CLIMB:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 3.0 # 拉起
            cmd['throttle'] = 1.0 # 爬升加满油
            
        elif action_id == self.ACTION_DIVE:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 0.5 # 低过载俯冲
            cmd['throttle'] = 0.7 # 轻推油门防止超速
            cmd['target_pitch_rate'] = -np.deg2rad(5)
            
        elif action_id == self.ACTION_BREAK_LEFT:
            cmd['target_roll'] = -np.deg2rad(85) # 几乎垂直
            cmd['target_g'] = 9.0 # 最大过载
            cmd['throttle'] = 1.0 # 能量维持
            cmd['target_pitch_rate'] = np.deg2rad(2)
            
        elif action_id == self.ACTION_BREAK_RIGHT:
            cmd['target_roll'] = np.deg2rad(85)
            cmd['target_g'] = 9.0
            cmd['throttle'] = 1.0
            cmd['target_pitch_rate'] = np.deg2rad(2)

        elif action_id == self.ACTION_PITCH_UP:
            cmd['target_roll'] = current_roll
            cmd['target_g'] = 2.5
            cmd['target_pitch_rate'] = np.deg2rad(6)

        elif action_id == self.ACTION_PITCH_DOWN:
            cmd['target_roll'] = current_roll
            cmd['target_g'] = 0.5
            cmd['target_pitch_rate'] = -np.deg2rad(6)

        elif action_id == self.ACTION_NOTCH_LEFT:
            cmd['target_roll'] = -np.deg2rad(80)
            cmd['target_g'] = 6.0
            cmd['throttle'] = 1.0
            cmd['target_pitch_rate'] = 0.0

        elif action_id == self.ACTION_NOTCH_RIGHT:
            cmd['target_roll'] = np.deg2rad(80)
            cmd['target_g'] = 6.0
            cmd['throttle'] = 1.0
            cmd['target_pitch_rate'] = 0.0

        elif action_id == self.ACTION_CRANK_LEFT:
            cmd['target_roll'] = -np.deg2rad(30)
            cmd['target_g'] = 2.0
            cmd['throttle'] = 0.9

        elif action_id == self.ACTION_CRANK_RIGHT:
            cmd['target_roll'] = np.deg2rad(30)
            cmd['target_g'] = 2.0
            cmd['throttle'] = 0.9

        elif action_id == self.ACTION_BARREL_ROLL:
            cmd['target_roll'] = np.deg2rad(170)
            cmd['target_g'] = 1.5
            cmd['target_pitch_rate'] = np.deg2rad(3)

        elif action_id == self.ACTION_SLICING_TURN:
            cmd['target_roll'] = np.deg2rad(70)
            cmd['target_g'] = 1.2
            cmd['target_pitch_rate'] = -np.deg2rad(2)

        elif action_id == self.ACTION_IMMELMANN:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 4.0
            cmd['throttle'] = 1.0
            cmd['target_pitch_rate'] = np.deg2rad(10)

        elif action_id == self.ACTION_SPLIT_S:
            cmd['target_roll'] = np.deg2rad(180)
            cmd['target_g'] = 2.5
            cmd['throttle'] = 0.8
            cmd['target_pitch_rate'] = -np.deg2rad(8)

        elif action_id == self.ACTION_HIGH_YOYO:
            cmd['target_roll'] = np.deg2rad(50)
            cmd['target_g'] = 2.0
            cmd['target_pitch_rate'] = np.deg2rad(5)

        elif action_id == self.ACTION_LOW_YOYO:
            cmd['target_roll'] = np.deg2rad(50)
            cmd['target_g'] = 1.2
            cmd['target_pitch_rate'] = -np.deg2rad(5)
            
        return cmd
