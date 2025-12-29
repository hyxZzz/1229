import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 路径黑魔法：确保能导入 sim_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_core.dynamics import FlightDynamics
from sim_core.maneuver_lib import ManeuverLibrary
from utils.geometry import euler_to_quat

def test_energy_maneuverability():
    print(">>> [Test 1] 物理引擎能量机动性测试 <<<")
    
    # 1. 初始化：高度 5000m, 速度 300m/s (约 0.9 Mach)
    init_pos = [0, 0, 5000]
    init_vel = [300, 0, 0]
    init_quat = euler_to_quat(0, 0, 0)
    
    plane = FlightDynamics(init_pos, init_vel, init_quat)
    lib = ManeuverLibrary()
    
    dt = 0.05
    duration = 10.0 # 模拟 10 秒
    steps = int(duration / dt)
    
    velocities = []
    
    print(f"初始状态: Speed={np.linalg.norm(plane.vel):.1f} m/s, Alt={plane.pos[2]:.1f} m")
    
    # 2. 执行 9G 持续左急转 (ACTION_BREAK_LEFT)
    # 在现实物理中，持续 9G 会产生巨大的诱导阻力，导致动能急剧损失
    cmd = lib.get_action_cmd(lib.ACTION_BREAK_LEFT, 0)
    
    for i in range(steps):
        # 强制满油门，看阻力是否能克服推力
        cmd['throttle'] = 1.0 
        plane.step(cmd, dt)
        
        speed = np.linalg.norm(plane.vel)
        velocities.append(speed)

    final_speed = velocities[-1]
    speed_loss = velocities[0] - final_speed
    
    print(f"结束状态: Speed={final_speed:.1f} m/s")
    print(f"速度损失: {speed_loss:.1f} m/s")
    
    # 3. 验证标准
    # 如果速度没有显著下降（例如少于 50m/s），说明阻力模型过弱，RL 会利用这点无限机动
    if speed_loss > 50.0:
        print("✅ 测试通过：诱导阻力生效，能量限制正常。")
    else:
        print("❌ 测试失败：阻力过小，飞机像UFO一样运动！建议增大 dynamics.py 中的 K 值。")
        
    # 可视化
    plt.figure()
    plt.plot(np.arange(steps)*dt, velocities)
    plt.title("Speed vs Time during 9G Turn")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.savefig("debug_physics_energy.png")
    print("速度曲线已保存至 debug_physics_energy.png")

if __name__ == "__main__":
    test_energy_maneuverability()