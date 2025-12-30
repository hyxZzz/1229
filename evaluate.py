import sys
import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 确保能导入项目根目录模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from envs.combat_env import CombatEnv_8v8
from agents.red_rl.policy import RedPolicy
from utils.geometry import get_distance  # [新增] 用于计算距离排序

def load_config(path):
    # 增加路径兼容性检查
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"⚠️ Config file not found: {path}")
            
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TrajectoryEvaluator:
    def __init__(self, model_path, config_path="configs/train_config.yaml"):
        # 1. 加载配置
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 初始化环境
        self.env = CombatEnv_8v8()
        
        # 3. 初始化并加载策略
        print(f"Loading model from: {model_path}")
        self.policy = RedPolicy(self.cfg['train'])
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            print("✅ Model loaded successfully.")
        else:
            print(f"⚠️ Warning: Model path '{model_path}' invalid! Using random weights for testing.")
        
        self.policy.eval()
        
        # 4. 轨迹数据存储容器
        self.trajectories = {
            'red': {},
            'blue': {},
            'missile': {}
        }

    def _get_sorted_enemies(self, agent, sim):
        """
        [关键修复] 复刻 ObservationParser 的排序逻辑。
        神经网络看到的敌人是按距离排序的（Index 0 = 最近），
        因此解析动作时也必须按距离排序，否则会攻击错误的目标。
        """
        enemies_list = []
        for other in sim.aircrafts:
            if other.team != agent.team and other.is_active:
                dist = get_distance(agent.pos, other.pos)
                enemies_list.append((dist, other))
        
        # 按距离从小到大排序
        enemies_list.sort(key=lambda x: x[0])
        
        # 返回排序后的实体对象列表
        return [e[1] for e in enemies_list]

    def run_episode(self):
        print("\n>>> Start Simulation (1 Episode)...")
        obs_dict = self.env.reset()
        done = False
        step = 0
        max_steps = 2000 # 约100秒
        
        red_uids = [f"Red_{i}" for i in range(8)]
        
        while not done and step < max_steps:
            # --- 1. 记录轨迹 ---
            self._record_positions()
            
            # --- 2. 构造 Batch Obs ---
            if not obs_dict:
                break

            # 使用第一个存活agent的obs shape来初始化batch
            first_obs = list(obs_dict.values())[0]
            batch_obs = {
                k: np.zeros((8, *v.shape), dtype=np.float32) 
                for k, v in first_obs.items()
            }
            
            # 填入数据 (保持红方顺序 0-7, 死亡的留为全0)
            active_red_indices = []
            for i, uid in enumerate(red_uids):
                if uid in obs_dict:
                    active_red_indices.append(i)
                    for k in batch_obs:
                        batch_obs[k][i] = obs_dict[uid][k]
            
            # --- 3. 策略推理 ---
            # 使用确定性策略进行评估
            with torch.no_grad():
                acts, _, _ = self.policy.act(batch_obs, deterministic=True)
            
            # --- 4. 转换动作为环境格式 ---
            env_actions = {}
            
            for i in active_red_indices:
                uid = red_uids[i]
                agent = self.env.sim.get_entity(uid)
                
                if not agent or not agent.is_active:
                    continue

                # 解包动作
                man_id = acts[0][i]
                tar_idx = acts[1][i] # 这里的 idx 对应的是“第k近的敌人”
                
                fire_target_uid = None
                
                # [关键步骤] 获取该 Agent 视角的排序后敌人列表
                sorted_enemies = self._get_sorted_enemies(agent, self.env.sim)
                
                # 映射目标: 只有当 idx 在有效范围内且不是“不开火”(idx=8)时
                if tar_idx < len(sorted_enemies):
                    target_obj = sorted_enemies[tar_idx]
                    if target_obj.is_active:
                        fire_target_uid = target_obj.uid
                
                env_actions[uid] = {
                    'maneuver': man_id, 
                    'fire_target': fire_target_uid
                }
            
            # --- 5. 环境步进 ---
            obs_dict, rewards, dones, info = self.env.step(env_actions)
            
            # 打印关键事件
            if 'events' in info:
                for e in info['events']:
                    if e['type'] == 'KILL':
                        print(f"[Step {step}] 💥 {e['killer']} KILLED {e['victim']}")
                    elif e['type'] == 'FIRE':
                        print(f"[Step {step}] 🚀 {e['launcher']} FIRED at {e['target']}")
            
            if dones.get("__all__", False):
                done = True
                
            step += 1
            if step % 200 == 0:
                print(f"Simulating step {step}...")
        
        print(f"Simulation Finished at step {step}.")
        self._record_positions() # 记录最后一帧

    def _record_positions(self):
        """记录当前帧所有实体的坐标"""
        # 1. 飞机
        for p in self.env.sim.aircrafts:
            if not p.is_active: continue
            
            category = 'red' if p.team == 0 else 'blue'
            if p.uid not in self.trajectories[category]:
                self.trajectories[category][p.uid] = []
            
            self.trajectories[category][p.uid].append(p.pos.copy())
            
        # 2. 导弹
        for m in self.env.sim.missiles:
            if not m.is_active: continue
            
            if m.uid not in self.trajectories['missile']:
                self.trajectories['missile'][m.uid] = []
            
            self.trajectories['missile'][m.uid].append(m.pos.copy())

    def plot_trajectories(self, output_file="combat_eval.png"):
        print("Generating 3D Trajectory Plot...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        has_data = False

        # --- 绘制红方 (Red) ---
        for uid, path in self.trajectories['red'].items():
            if len(path) < 1: continue
            has_data = True
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c='red', alpha=0.8, linewidth=1.5)
            # 起点和终点
            ax.scatter(path[0,0], path[0,1], path[0,2], c='red', marker='^', s=20) 
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='darkred', marker='x', s=30) 
            
        # --- 绘制蓝方 (Blue) ---
        for uid, path in self.trajectories['blue'].items():
            if len(path) < 1: continue
            has_data = True
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c='blue', alpha=0.6, linewidth=1.5, linestyle='--')
            ax.scatter(path[0,0], path[0,1], path[0,2], c='blue', marker='^', s=20)
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='darkblue', marker='x', s=30)

        # --- 绘制导弹 (Missile) ---
        for uid, path in self.trajectories['missile'].items():
            if len(path) < 2: continue
            path = np.array(path)
            # 导弹用黑色虚线
            ax.plot(path[:, 0], path[:, 1], path[:, 2], c='black', alpha=0.5, linewidth=0.8, linestyle=':')
            # 命中点/消失点
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='orange', marker='*', s=10)

        if not has_data:
            print("⚠️ No valid trajectory data recorded. Skipping plot.")
            return

        # 设置图形属性
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title("Air Combat 8v8 Evaluation Result")
        
        # 视场范围
        limit = 80000 
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(0, 25000)
        
        # 自定义图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Red Team (RL)'),
            Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Blue Team (Rule)'),
            Line2D([0], [0], color='black', lw=1, linestyle=':', label='Missile'),
            Line2D([0], [0], marker='x', color='black', label='End Pos', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='*', color='orange', label='Impact', markersize=8, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.savefig(output_file, dpi=150)
        print(f"✅ Trajectory plot saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 允许用户指定模型路径，默认为 None
    parser.add_argument("--model", type=str, default="/home/data/heyuxin/dqn_0715/1230/3/checkpoints/model_epoch_0.pt", help="Path to .pt model file")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 智能查找模型路径
    model_path = args.model
    if model_path is None:
        # 尝试一些常见的默认路径
        candidates = [
            "./checkpoints/model_epoch_50.pt",
            "./checkpoints/model_epoch_10.pt",
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                print(f"Auto-detected model: {model_path}")
                break
    
    evaluator = TrajectoryEvaluator(model_path, args.config)
    evaluator.run_episode()
    evaluator.plot_trajectories()