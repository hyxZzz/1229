import numpy as np

def get_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """计算两点间欧几里得距离"""
    return np.linalg.norm(pos1 - pos2)

def get_vector(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """计算从pos1指向pos2的向量"""
    return pos2 - pos1

def normalize(v: np.ndarray) -> np.ndarray:
    """向量归一化，防止除零"""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    四元数转欧拉角 (Roll, Pitch, Yaw)
    输入 q: [w, x, y, z]
    输出: [phi, theta, psi] (单位：弧度)
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp) # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def euler_to_quat(roll, pitch, yaw) -> np.ndarray:
    """
    欧拉角转四元数
    输入: roll, pitch, yaw (弧度)
    输出: [w, x, y, z]
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

def get_aoa_and_beta(vel_body: np.ndarray):
    """
    计算攻角(Alpha)和侧滑角(Beta)
    vel_body: 机体坐标系下的速度向量 [u, v, w]
    """
    u, v, w = vel_body
    V_total = np.linalg.norm(vel_body)
    
    if V_total < 1e-3:
        return 0.0, 0.0
        
    # Alpha = arctan(w/u)
    alpha = np.arctan2(w, u)
    # Beta = arcsin(v/V)
    beta = np.arcsin(v / V_total)
    
    return alpha, beta

def body_to_earth(vec_body: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    利用四元数将向量从机体坐标系转换到惯性坐标系(Earth)
    v_earth = q * v_body * q_conjugate
    """
    # 这里为了性能直接展开计算公式，而非调用矩阵库
    w, x, y, z = q
    vx, vy, vz = vec_body
    
    r = np.zeros(3)
    r[0] = (1 - 2*(y**2 + z**2))*vx + (2*(x*y - z*w))*vy + (2*(x*z + y*w))*vz
    r[1] = (2*(x*y + z*w))*vx + (1 - 2*(x**2 + z**2))*vy + (2*(y*z - x*w))*vz
    r[2] = (2*(x*z - y*w))*vx + (2*(y*z + x*w))*vy + (1 - 2*(x**2 + y**2))*vz
    
    return r