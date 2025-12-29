import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 路径黑魔法：确保能导入 sim_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim_core.entities.aircraft import Aircraft
from sim_core.entities.missile import Missile

def test_replan_and_energy():
    print(">>> 开始导弹重规划与能量机动测试 <<<")
    
    # 1. 初始化战场实体
    # 红方载机
    red_jet = Aircraft(uid="Red_1", team=0, init_pos=[0, 0, 5000], init_vel=[300, 0, 0])
    
    # 蓝方靶机 A (正前方 10km)
    blue_target_a = Aircraft(uid="Blue_A", team=1, init_pos=[10000, 0, 5000], init_vel=[200, 0, 0])
    
    # 蓝方靶机 B (侧方大角度 5km, 模拟大机动后的目标)
    blue_target_b = Aircraft(uid="Blue_B", team=1, init_pos=[10000, 5000, 5000], init_vel=[200, 0, 0])
    
    enemies = [blue_target_a, blue_target_b]
    
    # 2. 发射导弹 (锁定 Target A)
    missile = Missile(uid="Missile_01", team=0, launcher=red_jet, target=blue_target_a)
    print(f"Initial Lock: {missile.target.uid} | Initial Speed: {np.linalg.norm(missile.vel):.1f} m/s")

    # 数据记录
    traj_missile = []
    traj_a = []
    traj_b = []
    speed_log = []
    
    dt = 0.05
    max_steps = 300 # 15秒
    
    target_killed = False
    
    for i in range(max_steps):
        t = i * dt
        m_speed = np.linalg.norm(missile.vel)
        
        # --- 记录 ---
        traj_missile.append(missile.pos.copy())
        traj_a.append(blue_target_a.pos.copy())
        traj_b.append(blue_target_b.pos.copy())
        speed_log.append(m_speed)
        
        # --- 触发事件：T=2.0秒时，Target A 突然死亡，迫使导弹做大机动转向 B ---
        if t > 2.0 and not target_killed:
            print(f"\n[EVENT] T={t:.2f}s: Target Blue_A KILLED! Triggering Replan...")
            blue_target_a.is_active = False 
            target_killed = True
            
        # --- 实体更新 ---
        blue_target_a.step(0, dt)
        blue_target_b.step(0, dt) # 蓝机B 保持平飞
        
        hit, hit_uid = missile.update(dt, enemies)
        
        if hit:
            print(f"\n[RESULT] T={t:.2f}s: Missile HIT {hit_uid}! Impact Speed: {m_speed:.1f} m/s")
            break
            
        if not missile.is_active:
            print(f"\n[RESULT] Missile died at T={t:.2f}s. Final Speed: {m_speed:.1f} m/s")
            break

    # --- 验证诱导阻力效果 ---
    # 检查转向期间（2.0s 后）是否有明显掉速
    # 提取 2.0s 时的速度 和 命中/结束时的速度
    idx_turn_start = int(2.0 / dt)
    if idx_turn_start < len(speed_log):
        speed_start = speed_log[idx_turn_start]
        speed_end = speed_log[-1]
        speed_drop = speed_start - speed_end
        
        print(f"\n>>> 能量分析 <<<")
        print(f"转向前速度: {speed_start:.1f} m/s")
        print(f"结束时速度: {speed_end:.1f} m/s")
        print(f"速度损失: {speed_drop:.1f} m/s")
        
        if speed_drop > 200.0:
            print("✅ 验证通过：大过载转向导致了显著的速度损失 (诱导阻力生效)。")
        else:
            print("⚠️ 警告：速度损失不明显，可能诱导阻力系数过小或转向不够剧烈。")

    # --- 3D 可视化 ---
    plot_trajectory(traj_missile, traj_a, traj_b)

def plot_trajectory(m_traj, a_traj, b_traj):
    m = np.array(m_traj)
    a = np.array(a_traj)
    b = np.array(b_traj)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(m[:,0], m[:,1], m[:,2], 'r-', label='Missile', linewidth=2)
    ax.plot(a[:,0], a[:,1], a[:,2], 'b--', label='Target A (Dead)')
    ax.plot(b[:,0], b[:,1], b[:,2], 'g-', label='Target B (Secondary)')
    
    ax.set_title('Missile Energy & Replan Test')
    ax.legend()
    
    output_path = 'missile_test_result.png'
    plt.savefig(output_path)
    print(f"轨迹图已保存至: {output_path}")

if __name__ == "__main__":
    test_replan_and_energy()