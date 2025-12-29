import numpy as np
from utils.geometry import normalize, get_distance

class TacticsLibrary:
    """
    战术动作计算库：计算具体的航向矢量
    """
    
    @staticmethod
    def get_notch_vector(agent_pos, agent_vel, threat_pos):
        """
        计算切向机动(Notch)的最佳矢量：与威胁连线垂直
        """
        # 视线矢量
        los = normalize(threat_pos - agent_pos)
        
        # 垂直矢量：LOS x (0,0,1) -> 水平面的垂直矢量
        # 如果 LOS 接近垂直，用 (1,0,0)
        up = np.array([0, 0, 1])
        if abs(los[2]) > 0.9:
            up = np.array([1, 0, 0])
            
        right_vec = normalize(np.cross(los, up))
        
        # 选择左右：选择与当前速度夹角更小的一侧（转弯代价小）
        if np.dot(agent_vel, right_vec) > 0:
            target_dir = right_vec
        else:
            target_dir = -right_vec
            
        return target_dir

    @staticmethod
    def get_crank_vector(agent_pos, target_pos, radar_fov_deg=55.0):
        """
        计算偏置机动(Crank)矢量：维持目标在雷达边缘
        """
        los = normalize(target_pos - agent_pos)
        
        # 将 LOS 旋转 FOV 角度
        fov_rad = np.deg2rad(radar_fov_deg)
        c, s = np.cos(fov_rad), np.sin(fov_rad)
        
        # 简单处理：在水平面内旋转
        x, y, z = los
        # 旋转矩阵 (2D rotation on xy)
        new_x = x * c - y * s
        new_y = x * s + y * c
        
        return normalize(np.array([new_x, new_y, z]))

    @staticmethod
    def get_break_vector(agent_pos, threat_pos, threat_vel):
        """
        计算急转(Break)矢量：通常向着导弹来袭方向急转，制造大角速度
        """
        # 简单的 Break：向导弹反方向并带一点垂直分量（俯冲或急升）
        los = normalize(threat_pos - agent_pos)
        
        # 这里的 Break 逻辑是：垂直于 LOS，并且尽量向下（利用地杂波或重力加速）
        notch_dir = TacticsLibrary.get_notch_vector(agent_pos, np.zeros(3), threat_pos)
        
        # 叠加向下分量
        break_dir = normalize(notch_dir + np.array([0, 0, -0.5]))
        
        return break_dir