import matplotlib
# 强制使用无头模式，防止服务器报错
matplotlib.use('Agg') 

import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from envs.combat_env import CombatEnv_8v8
from agents.red_rl.policy import RedPolicy
from utils.render_tool import RenderTool

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_evaluation(model_path):
    # 1. 加载配置
    train_config = load_config("configs/train_config.yaml")
    
    # 2. 初始化环境
    env = CombatEnv_8v8()
    obs_dict = env.reset()
    
    # 3. 初始化并加载策略
    policy = RedPolicy(train_config['train'])
    
    if model_path:
        print(f"Loading model from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
    else:
        print("Warning: No model path provided, using random weights.")
    
    policy.eval() # 切换到网络评估模式 (LayerNorm等)
    
    # 4. 初始化渲染器
    renderer = RenderTool()
    
    # 5. 仿真循环
    done = False
    step = 0
    max_steps = 4000 # 对应 200秒
    
    red_uids = [f"Red_{i}" for i in range(8)]
    
    print(">>> Start Simulation Loop (Deterministic=False)...")
    
    total_fires = 0
    total_kills = 0

    try:
        while not done and step < max_steps:
            # 记录画面
            renderer.record_frame(env.sim)
            
            # --- 构造 Batch Obs ---
            batch_obs = {
                k: np.zeros((8, *v.shape), dtype=np.float32) 
                for k, v in list(obs_dict.values())[0].items()
            }
            
            for i, uid in enumerate(red_uids):
                if uid in obs_dict:
                    for k in batch_obs:
                        batch_obs[k][i] = obs_dict[uid][k]
            
            # --- 策略推理 ---
            # [关键修改] 使用 Stochastic 策略，还原训练时的行为
            acts, _, _ = policy.act(batch_obs, deterministic=False)
            
            # --- 转换动作 ---
            env_actions = {}
            all_enemies = [p for p in env.sim.aircrafts if p.team == 1]
            
            for i, uid in enumerate(red_uids):
                if uid not in obs_dict: continue
                
                man_id = acts[0][i]
                tar_idx = acts[1][i]
                
                fire_target = None
                if tar_idx < len(all_enemies):
                    target_obj = all_enemies[tar_idx]
                    if target_obj.is_active:
                        fire_target = target_obj.uid
                    
                env_actions[uid] = {'maneuver': man_id, 'fire_target': fire_target}
                
            # --- 环境步进 (关键：捕获 events) ---
            # 我们需要手动调用 sim.step 来获取 events，或者修改 combat_env 返回 events
            # 这里为了不改动 env 代码，我们利用 combat_env 内部的逻辑
            # 但是 combat_env.step 并没有返回 events，只返回了 rewards
            # 所以我们只能通过前后状态对比，或者信任 log
            
            # 为了调试，我们在 combat_env.py 外面直接看 sim 的导弹变化有点难
            # 最简单的方法：观察 env.step 后的 rewards
            # 如果 reward 有巨大的跳变 (+10)，说明有击杀
            # 如果 sim.missiles 数量增加了，说明有开火
            
            prev_missile_count = len(env.sim.missiles)
            obs_dict, rewards, dones, info = env.step(env_actions)
            curr_missile_count = len(env.sim.missiles)
            
            # 统计开火
            if curr_missile_count > prev_missile_count:
                new_fires = curr_missile_count - prev_missile_count
                total_fires += new_fires
                print(f"[Step {step}] 🔥 FIRE! Total fired: {new_fires}")

            # 统计击杀 (通过 Reward 推断)
            for uid, r in rewards.items():
                if r > 5.0: # 击杀奖励通常 > 5
                    print(f"[Step {step}] 💀 KILL CONFIRMED by {uid}! Reward: {r:.1f}")
                    total_kills += 1

            if dones.get("__all__", False):
                done = True
                
            step += 1
            
            # --- 日志打印 ---
            if step % 100 == 0 or done:
                red_alive = sum(1 for p in env.sim.aircrafts if p.team == 0 and p.is_active)
                blue_alive = sum(1 for p in env.sim.aircrafts if p.team == 1 and p.is_active)
                
                # 计算最近距离 (Min Dist) 而不是 Mean Dist，这更有意义
                min_dist = 200000.0
                for r in env.sim.aircrafts[:8]:
                    if r.is_active:
                        for b in env.sim.aircrafts[8:]:
                            if b.is_active:
                                d = np.linalg.norm(r.pos - b.pos)
                                if d < min_dist: min_dist = d
                
                print(f"Step {step}: Red={red_alive} | Blue={blue_alive} | Min Dist={min_dist/1000:.1f}km | Fires={total_fires}")
                
                if blue_alive == 0:
                    print(">>> VICTORY! Blue Team Annihilated.")
                    break

    except KeyboardInterrupt:
        print("Interrupted.")
    
    print(f">>> Finished. Total Fires: {total_fires}, Total Kills: {total_kills}")
    
    print("Generating GIF...")
    if len(renderer.history) > 2000:
        renderer.history = renderer.history[:2000] # 截取前2000帧
    renderer.animate(save_path="debug_result.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./checkpoints/model_epoch_100.pt")
    args = parser.parse_args()
    
    run_evaluation(args.model)