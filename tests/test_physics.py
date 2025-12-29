import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_core.dynamics import FlightDynamics
from sim_core.maneuver_lib import ManeuverLibrary
from utils.geometry import euler_to_quat

def test_turn_energy():
    print("开始物理引擎测试: 9G急转能量损失验证...")
    
    # 初始条件: 高度5000米, 速度300m/s (约Mach 0.9), 水平飞行
    init_pos = [0, 0, 5000]
    init_vel = [300, 0, 0]
    init_quat = euler_to_quat(0, 0, 0)
    
    plane = FlightDynamics(init_pos, init_vel, init_quat)
    lib = ManeuverLibrary()
    
    # 记录数据
    positions = []
    velocities = []
    
    dt = 0.05
    duration = 10.0 # 模拟10秒
    steps = int(duration / dt)
    
    for i in range(steps):
        # 前2秒平飞，之后左急转
        action_id = lib.ACTION_MAINTAIN
        if i * dt > 2.0:
            action_id = lib.ACTION_BREAK_LEFT
            
        cmd = lib.get_action_cmd(action_id, 0) # 简化：不传当前roll，直接开环
        
        pos, vel, speed, n = plane.step(cmd, dt)
        
        positions.append(pos.copy())
        velocities.append(speed)
        
        if i % 20 == 0:
            print(f"T={i*dt:.2f}s | Speed={speed:.1f} m/s | Alt={pos[2]:.1f} m | G={n:.2f}")

    # 简单的断言
    final_speed = velocities[-1]
    # 9G转弯通常会掉很多速度 (诱导阻力巨大)
    # 如果速度反而增加了，说明阻力模型写错了
    assert final_speed < 300.0, "错误：急转弯时速度没有下降，能量机动违规！"
    print(f"测试通过：速度从 300.0 降至 {final_speed:.1f}，符合能量机动理论。")
    
    # 画图
    pos_arr = np.array(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], label='Flight Path')
    ax.set_title('9G Turn Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude')
    plt.legend()
    # plt.show() # 如果在服务器上运行请注释掉

if __name__ == "__main__":
    test_turn_energy()