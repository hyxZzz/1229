import numpy as np

class ManeuverLibrary:
    """
    机动动作库：将离散动作ID转换为连续的控制指令
    Control Cmd Format: {'target_g': float, 'target_roll': float, 'throttle': float}
    """
    # 动作定义
    ACTION_MAINTAIN = 0  # 保持当前状态
    ACTION_LEFT_TURN = 1 # 左平转
    ACTION_RIGHT_TURN = 2 # 右平转
    ACTION_CLIMB = 3     # 爬升
    ACTION_DIVE = 4      # 俯冲
    ACTION_BREAK_LEFT = 5 # 左急转 (高过载)
    ACTION_BREAK_RIGHT = 6 # 右急转
    
    def __init__(self):
        pass

    def get_action_cmd(self, action_id: int, current_roll: float) -> dict:
        cmd = {
            'throttle': 1.0, # 默认巡航油门
            'target_g': 1.0, # 默认1G平飞
            'target_roll': 0.0
        }
        
        if action_id == self.ACTION_MAINTAIN:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 1.0
            
        elif action_id == self.ACTION_LEFT_TURN:
            cmd['target_roll'] = -np.deg2rad(45)
            cmd['target_g'] = 3.0 # 温和转向
            
        elif action_id == self.ACTION_RIGHT_TURN:
            cmd['target_roll'] = np.deg2rad(45)
            cmd['target_g'] = 3.0
            
        elif action_id == self.ACTION_CLIMB:
            cmd['target_roll'] = 0.0
            cmd['target_g'] = 3.0 # 拉起
            cmd['throttle'] = 1.0 # 爬升加满油
            
        elif action_id == self.ACTION_DIVE:
            cmd['target_roll'] = np.deg2rad(180) # 倒扣
            cmd['target_g'] = 2.0 # 下拉
            
        elif action_id == self.ACTION_BREAK_LEFT:
            cmd['target_roll'] = -np.deg2rad(85) # 几乎垂直
            cmd['target_g'] = 9.0 # 最大过载
            cmd['throttle'] = 1.0 # 能量维持
            
        elif action_id == self.ACTION_BREAK_RIGHT:
            cmd['target_roll'] = np.deg2rad(85)
            cmd['target_g'] = 9.0
            cmd['throttle'] = 1.0
            
        return cmd