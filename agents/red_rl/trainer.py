import numpy as np
import torch
import torch.optim as optim
from envs.combat_env import CombatEnv_8v8
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
        man_indices: (8,) numpy array
        tar_indices: (8,) numpy array (0-7 for enemy, 8 for no-fire)
        obs_uids: List of uids corresponding to the batch order
        """
        env_actions = {}
        
        # 获取当前所有的敌机 UID 列表 (顺序必须与 obs_parser 生成特征的顺序一致)
        # 假设 sim.aircrafts 顺序不变，我们可以过滤出敌机
        all_enemies = [p for p in self.env.sim.aircrafts if p.team == 1] # Blue Team
        # 注意：obs_parser 是按照 sim.aircrafts 顺序遍历的，所以索引对应是一致的
        
        for i, uid in enumerate(obs_uids):
            # 1. 飞行机动
            maneuver_id = int(man_indices[i])
            
            # 2. 开火决策
            target_idx = int(tar_indices[i])
            fire_target_uid = None
            
            # 如果 target_idx < 8，说明想攻击第 target_idx 个敌机
            if target_idx < len(all_enemies):
                target_obj = all_enemies[target_idx]
                # 只有当目标活着时才通过指令
                if target_obj.is_active:
                    fire_target_uid = target_obj.uid
            
            env_actions[uid] = {
                'maneuver': maneuver_id,
                'fire_target': fire_target_uid
            }
            
        return env_actions

    def collect_rollouts(self):
        """
        采集数据循环
        """
        obs_dict = self.env.reset() # {'Red_0': {...}, 'Red_1': ...}
        
        # 必须保证 obs 按照固定顺序转成 batch
        # 我们的 Buffer 期望 (Num_Agents, ...) 的输入
        # 所以我们需要维护一个 active_uids 列表来对齐
        # 为了简化，我们假设总是处理 8 个红方 Agent (即使死了也输入，只是 mask 掉) -> 
        # 实际上 Obs Parser 只返回活着的 Agent 的 Obs。
        # 这里需要特别注意：Buffer 必须对齐。建议：Env 总是返回 8 个 Agent 的数据，死的全 0。
        # *修改 Env 逻辑太复杂，我们在 Trainer 做适配*
        
        # 策略：
        # Buffer 的第二维是 Num_Agents (8)。
        # 我们构建一个 Ordered List: ['Red_0', ..., 'Red_7']
        red_uids = [f"Red_{i}" for i in range(8)]
        
        ep_ret = np.zeros(8) # 记录本回合得分

        stat_speed = []
        stat_dist = []
        stat_fire = 0
        
        
        for t in range(self.config['steps_per_epoch']):
            # 1. 整理 Batch Obs
            # 创建全零的 Batch 容器
            batch_obs = {
                k: np.zeros((8, *v.shape), dtype=np.float32) 
                for k, v in list(obs_dict.values())[0].items()
            }
            
            # 填入数据
            alive_mask = np.zeros(8, dtype=bool)
            for i, uid in enumerate(red_uids):
                if uid in obs_dict:
                    alive_mask[i] = True
                    for k in batch_obs:
                        batch_obs[k][i] = obs_dict[uid][k]
            
            # 2. 神经网络推理
            # acts: (man_idx, tar_idx), vals: (8,)
            acts, logps, vals = self.policy.act(batch_obs)
            
            # 3. 构造 Env Action
            env_action_dict = self._map_actions_to_env(acts[0], acts[1], red_uids)
            
            # 4. Step
            next_obs_dict, rewards, dones, info = self.env.step(env_action_dict)

            # 只有当红方活着的时候，数据才有意义
            if info['mean_speed'] > 1.0: 
                stat_speed.append(info['mean_speed'])
            stat_dist.append(info['mean_dist'])
            stat_fire += info['fire_count']
            
            # 5. 整理 Reward (对齐到 8 agents)
            rew_arr = np.zeros(8)
            for i, uid in enumerate(red_uids):
                rew_arr[i] = rewards.get(uid, 0.0)
            
            ep_ret += rew_arr
            
            # 6. 存入 Buffer
            # 注意：即使是死掉的 Agent 也可以存进去，只要 Mask 掉 Advantage 即可，
            # 或者简单点，死掉的 Agent 得到 Reward=0, Next_Val=0，不会影响梯度太大
            self.buffer.store(batch_obs, acts, rew_arr, vals, (logps[0], logps[1]))
            
            # 更新 Obs
            obs_dict = next_obs_dict
            
            # 处理回合结束
            timeout = (t == self.config['steps_per_epoch'] - 1)
            all_done = dones.get("__all__", False)
            
            if all_done or timeout:
                # 推理最后一步 Value 用于 GAE
                if timeout and not all_done:
                    # 构造这一帧所有存活/死亡 Agent 的 Obs Batch
                    # 保持和循环开始处的 batch_obs 逻辑一致
                    last_val_obs = {
                        k: np.zeros((8, *v.shape), dtype=np.float32) 
                        for k, v in list(obs_dict.values())[0].items()
                    }
                    for i, uid in enumerate(red_uids):
                        if uid in obs_dict:
                            for k in last_val_obs:
                                last_val_obs[k][i] = obs_dict[uid][k]
                    
                    # 放入 device 计算
                    _, _, last_vals = self.policy.act(last_val_obs)
                else:
                    last_vals = np.zeros(8)
                    
                self.buffer.finish_path(last_vals)
                
                
                avg_speed = np.mean(stat_speed) if stat_speed else 0.0
                avg_dist_km = np.mean(stat_dist) / 1000.0 if stat_dist else 0.0
                
                print(f"  Ep Ret: {np.mean(ep_ret):.2f} | "
                      f"Spd: {avg_speed:.0f} m/s | "
                      f"Dist: {avg_dist_km:.1f} km | "
                      f"Fire: {stat_fire}")
                
                # 重置统计
                stat_speed = []
                stat_dist = []
                stat_fire = 0


                # 重置环境
                obs_dict = self.env.reset()
                ep_ret = np.zeros(8)

    def update(self):
        """
        PPO 更新逻辑
        """
        data = self.buffer.get()
        
        # 训练参数
        clip_ratio = self.config['clip_ratio']
        target_kl = self.config['target_kl']
        entropy_coef = self.config['entropy_coef']
        train_iters = self.config['train_iters']
        
        loss_pi_list = []
        loss_v_list = []
        
        for i in range(train_iters):
            self.optimizer.zero_grad()
            
            # 1. 重新评估旧样本
            # evaluate 返回的是 tuple (logp_man, logp_tar), (ent_man, ent_tar), val
            logps, ents, vals = self.policy.evaluate(
                data['obs'], 
                data['act_man'], 
                data['act_tar']
            )
            
            # 2. 计算 Ratio
            # Total LogP = LogP_Maneuver + LogP_Target
            # 假设两个动作独立
            old_logp = data['logp_man'] + data['logp_tar']
            curr_logp = logps[0] + logps[1]
            
            ratio = torch.exp(curr_logp - old_logp)
            
            # 3. Policy Loss (Clipped Surrogate)
            adv = data['adv']
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(surr1, surr2)).mean()
            
            # 4. Entropy Bonus (Max Entropy)
            # Ent = Ent_Man + Ent_Tar
            loss_ent = -(ents[0] + ents[1]).mean() * entropy_coef
            
            # 5. Value Loss (MSE)
            loss_v = ((vals - data['ret'])**2).mean()
            
            # Total Loss
            loss = loss_pi + loss_v + loss_ent
            
            # Kl Divergence check (Optional early stopping)
            with torch.no_grad():
                approx_kl = (old_logp - curr_logp).mean().item()
            if approx_kl > 1.5 * target_kl:
                print(f"Early stopping at step {i} due to KL")
                break
                
            loss.backward()
            self.optimizer.step()
            
            loss_pi_list.append(loss_pi.item())
            loss_v_list.append(loss_v.item())
            
        return np.mean(loss_pi_list), np.mean(loss_v_list), approx_kl

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)