import numpy as np
import torch
import torch.optim as optim
from envs.combat_env import CombatEnv_8v8
from utils.geometry import get_distance, normalize
from agents.red_rl.policy import RedPolicy
from agents.red_rl.buffer import PPOBuffer

class RedTrainer:
    def __init__(self, config):
        self.config = config
        self.env = CombatEnv_8v8()
        self.policy = RedPolicy(config)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.device = self.policy.device
        
        # Buffer 配置
        # 需要手动定义 Obs Shape，这必须与 obs_parser 保持一致
        obs_shapes = {'self': 11, 'allies': (8, 7), 'enemies': (8, 7), 'missiles': (24, 7)}
        act_shapes = {} # 占位
        self.buffer = PPOBuffer(config['steps_per_epoch'], 8, obs_shapes, act_shapes, self.device)
        
    def _map_actions_to_env(self, man_indices, tar_indices, obs_uids):
        """
        将神经网络输出的 Index 转换为 Env 需要的 UID Dict
        必须针对每个 Agent 的视角，对敌机按距离排序，以匹配 ObservationParser 的逻辑。
        """
        env_actions = {}
        
        # 1. 获取所有活着的蓝方敌机 (作为基础列表)
        # 注意：必须过滤掉非活跃的，否则距离排序会包含死掉的飞机，导致索引偏移，与 ObsParser 不一致
        blue_enemies = [p for p in self.env.sim.aircrafts if p.team == 1 and p.is_active]
        
        for i, uid in enumerate(obs_uids):
            maneuver_id = int(man_indices[i])
            target_idx = int(tar_indices[i])
            fire_target_uid = None
            
            # 获取当前 Agent 对象
            agent = self.env.sim.get_entity(uid)
            
            if agent and agent.is_active:

                # 1. 计算该 Agent 到所有敌机的距离
                # 这一步是核心：必须模拟 Agent 看到的“世界”，即按距离远近排列的敌人
                sorted_enemies = []
                for enemy in blue_enemies:
                    dist = get_distance(agent.pos, enemy.pos) # 使用 utils.geometry 中的函数
                    sorted_enemies.append((dist, enemy))
                
                # 2. 按距离从小到大排序 (Obs Index 0 = Nearest Enemy)
                sorted_enemies.sort(key=lambda x: x[0])
                
                # 3. 根据 Network 输出的 Index 选择目标
                # target_idx 是网络输出的 0-8 (8代表不开火)
                # 只有当 target_idx 指向有效的敌机索引时才开火
                if target_idx < len(sorted_enemies):
                    # 取出排序后的第 target_idx 个敌机对象
                    target_obj = sorted_enemies[target_idx][1]
                    fire_target_uid = target_obj.uid
                # === [关键修复] 结束 ===
            
            env_actions[uid] = {
                'maneuver': maneuver_id,
                'fire_target': fire_target_uid
            }
            
        return env_actions

    def collect_rollouts(self):
        """
        采集数据循环
        """
        print(f"\n{'='*20} Start New Rollout Episode {'='*20}")

        obs_dict = self.env.reset()
        red_uids = [f"Red_{i}" for i in range(8)]
        ep_ret = np.zeros(8)
        
        # 统计变量
        stat_speed = []
        stat_dist = []
        stat_fire = 0
        
        for t in range(self.config['steps_per_epoch']):
            # 1. 整理 Batch Obs
            first_obs = list(obs_dict.values())[0]
            batch_obs = {
                k: np.zeros((8, *v.shape), dtype=np.float32) 
                for k, v in first_obs.items()
            }
            
            alive_mask = np.zeros(8, dtype=bool)
            for i, uid in enumerate(red_uids):
                if uid in obs_dict:
                    alive_mask[i] = True
                    for k in batch_obs:
                        batch_obs[k][i] = obs_dict[uid][k]
            
            # 2. 神经网络推理
            acts, logps, vals = self.policy.act(batch_obs)
            
            # 3. 构造 Env Action
            env_action_dict = self._map_actions_to_env(acts[0], acts[1], red_uids)
            
            # 4. Step
            next_obs_dict, rewards, dones, info = self.env.step(env_action_dict)

            # --- 日志打印 ---
            if 'events' in info:
                for event in info['events']:
                    if event['type'] == 'FIRE':
                        launcher = self.env.sim.get_entity(event['launcher'])
                        target = self.env.sim.get_entity(event['target'])
                        if launcher and target:
                            dist = get_distance(launcher.pos, target.pos)
                            vel_dir = normalize(launcher.vel)
                            los_dir = normalize(target.pos - launcher.pos)
                            angle = np.degrees(np.arccos(np.clip(np.dot(vel_dir, los_dir), -1, 1)))
                            print(f"[FIRE] 🚀 {launcher.uid} -> Locked {target.uid} | "
                                  f"Dist: {dist/1000:.1f}km | Angle: {angle:.1f}°")

                    elif event['type'] == 'KILL':
                        print(f"[KILL] 💥 {event['killer']} HIT {event['victim']}!")

            if info['mean_speed'] > 1.0: 
                stat_speed.append(info['mean_speed'])
            stat_dist.append(info['mean_dist'])
            stat_fire += info['fire_count']
            
            # 5. 整理 Reward
            rew_arr = np.zeros(8)
            for i, uid in enumerate(red_uids):
                rew_arr[i] = rewards.get(uid, 0.0)
            
            ep_ret += rew_arr
            
            # 6. 存入 Buffer
            self.buffer.store(batch_obs, acts, rew_arr, vals, (logps[0], logps[1]))
            
            obs_dict = next_obs_dict
            
            # 处理回合结束
            timeout = (t == self.config['steps_per_epoch'] - 1)
            all_done = dones.get("__all__", False)
            
            if all_done or timeout:
                if timeout and not all_done:
                    last_val_obs = {
                        k: np.zeros((8, *v.shape), dtype=np.float32) 
                        for k, v in first_obs.items()
                    }
                    for i, uid in enumerate(red_uids):
                        if uid in obs_dict:
                            for k in last_val_obs:
                                last_val_obs[k][i] = obs_dict[uid][k]
                    _, _, last_vals = self.policy.act(last_val_obs)
                else:
                    last_vals = np.zeros(8)
                    
                self.buffer.finish_path(last_vals)
                
                # 打印统计
                avg_speed = np.mean(stat_speed) if stat_speed else 0.0
                avg_dist_km = np.mean(stat_dist) / 1000.0 if stat_dist else 0.0
                red_left = sum(1 for p in self.env.sim.aircrafts if p.team==0 and p.is_active)
                blue_left = sum(1 for p in self.env.sim.aircrafts if p.team==1 and p.is_active)
                
                print(f"{'-'*10} Episode End {'-'*10}")
                print(f"[Result] Red Survivors: {red_left} | Blue Survivors: {blue_left}")
                print(f"  Ep Ret: {np.mean(ep_ret):.2f} | "
                      f"Spd: {avg_speed:.0f} m/s | "
                      f"Dist: {avg_dist_km:.1f} km | "
                      f"Fire: {stat_fire}")
                
                stat_speed = []
                stat_dist = []
                stat_fire = 0
                obs_dict = self.env.reset()
                ep_ret = np.zeros(8)

    def update(self):
        """
        PPO 更新逻辑
        """
        data = self.buffer.get()
        
        clip_ratio = self.config['clip_ratio']
        target_kl = self.config['target_kl']
        entropy_coef = self.config['entropy_coef']
        train_iters = self.config['train_iters']
        
        loss_pi_list = []
        loss_v_list = []
        loss_ent_list = [] # [新增] 记录 Entropy
        
        for i in range(train_iters):
            self.optimizer.zero_grad()
            
            # 1. 重新评估旧样本
            logps, ents, vals = self.policy.evaluate(
                data['obs'], 
                data['act_man'], 
                data['act_tar']
            )
            
            # 2. 计算 Ratio
            old_logp = data['logp_man'] + data['logp_tar']
            curr_logp = logps[0] + logps[1]
            ratio = torch.exp(curr_logp - old_logp)
            
            # 3. Policy Loss
            adv = data['adv']
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(surr1, surr2)).mean()
            
            # 4. Entropy Bonus
            total_entropy = ents[0] + ents[1]
            loss_ent = -total_entropy.mean() * entropy_coef
            
            # 5. Value Loss
            loss_v = ((vals - data['ret'])**2).mean()
            
            # Total Loss
            loss = loss_pi + loss_v + loss_ent
            
            # Kl Divergence check
            with torch.no_grad():
                approx_kl = (old_logp - curr_logp).mean().item()
            
            if approx_kl > 1.5 * target_kl:
                print(f"Early stopping at step {i} due to KL ({approx_kl:.4f})")
                break
                
            loss.backward()
            self.optimizer.step()
            
            loss_pi_list.append(loss_pi.item())
            loss_v_list.append(loss_v.item())
            loss_ent_list.append(total_entropy.mean().item()) # 记录原始 Entropy 值
            
        return np.mean(loss_pi_list), np.mean(loss_v_list), approx_kl, np.mean(loss_ent_list)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)