import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.combat_env import CombatEnv_8v8

def check_random_policy_safety():
    print("\n>>> [Test 4] 初始随机策略安全性检查 <<<")
    
    env = CombatEnv_8v8()
    env.reset()
    
    max_steps = 200 # 模拟 10 秒 (dt=0.05)
    red_uids = [f"Red_{i}" for i in range(8)]
    
    crashed_count = 0
    
    print(f"开始模拟 {max_steps} 步 (随机动作)...")
    
    for t in range(max_steps):
        # 生成随机动作
        actions = {}
        for uid in red_uids:
            # 随机机动 ID (0-18)
            man_id = np.random.randint(0, 19)
            actions[uid] = {'maneuver': man_id, 'fire_target': None}
            
        obs, rewards, dones, info = env.step(actions)
        
        if dones["__all__"]:
            print("所有飞机提前结束。")
            break
            
    # 统计存活率
    alive_count = 0
    for p in env.sim.aircrafts:
        if p.team == 0:
            if p.is_active:
                alive_count += 1
            elif p.pos[2] <= 0.1: # 撞地判断
                crashed_count += 1
                
    survival_rate = alive_count / 8.0
    print(f"模拟结束。")
    print(f"红方存活: {alive_count}/8")
    print(f"红方撞地: {crashed_count}")
    
    if survival_rate < 0.5:
        print("❌ 警告：超过 50% 的飞机在随机策略下撞地！")
        print("   建议：在 sim_core/entities/aircraft.py 或 dynamics.py 中加入底层保护逻辑。")
        print("   (例如：当高度 < 500m 且俯仰角 < 0 时，强制将 target_g 设为正值并拉起)")
    else:
        print("✅ 通过：初始策略下大部分飞机能保持飞行，环境适合 RL 探索。")

if __name__ == "__main__":
    check_random_policy_safety()