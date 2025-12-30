import numpy as np
from utils.geometry import normalize

class TacticsLibrary:
    """
    战术几何库：只负责计算理想的航向矢量，不涉及具体机动指令
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
        
        # 将 LOS 旋转 FOV 角度 (简单起见，在水平面上旋转)
        fov_rad = np.deg2rad(radar_fov_deg)
        c, s = np.cos(fov_rad), np.sin(fov_rad)
        
        x, y, z = los
        # 旋转矩阵 (2D rotation on xy)
        new_x = x * c - y * s
        new_y = x * s + y * c
        
        # 保持 z 轴分量大致不变，归一化
        return normalize(np.array([new_x, new_y, z]))