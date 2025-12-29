import numpy as np
import torch

class PPOBuffer:
    def __init__(self, steps_per_epoch, num_agents, obs_shapes, act_shapes, device):
        self.max_size = steps_per_epoch
        self.num_agents = num_agents
        self.device = device
        self.ptr, self.path_start_idx = 0, 0
        
        # 初始化 Buffer 存储空间
        # 观测是字典，所以我们要分开存
        self.obs_buf = {
            'self': np.zeros((steps_per_epoch, num_agents, obs_shapes['self']), dtype=np.float32),
            'allies': np.zeros((steps_per_epoch, num_agents, *obs_shapes['allies']), dtype=np.float32),
            'enemies': np.zeros((steps_per_epoch, num_agents, *obs_shapes['enemies']), dtype=np.float32),
            'missiles': np.zeros((steps_per_epoch, num_agents, *obs_shapes['missiles']), dtype=np.float32)
        }
        
        # 动作有两个：机动(Index) + 目标(Index)
        self.act_man_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.int64)
        self.act_tar_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.int64)
        
        self.rew_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)
        self.logp_man_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)
        self.logp_tar_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)
        self.adv_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)
        self.ret_buf = np.zeros((steps_per_epoch, num_agents), dtype=np.float32)

    def store(self, obs, acts, rews, vals, logps):
        """
        obs: dict of (Num_Agents, Feat)
        acts: tuple (maneuver_idx, target_idx) each (Num_Agents,)
        """
        assert self.ptr < self.max_size
        
        # 存储 Obs Dict
        for k in self.obs_buf:
            self.obs_buf[k][self.ptr] = obs[k]
            
        self.act_man_buf[self.ptr] = acts[0]
        self.act_tar_buf[self.ptr] = acts[1]
        
        self.rew_buf[self.ptr] = rews
        self.val_buf[self.ptr] = vals
        self.logp_man_buf[self.ptr] = logps[0]
        self.logp_tar_buf[self.ptr] = logps[1]
        
        self.ptr += 1

    def finish_path(self, last_vals=0, gamma=0.99, lam=0.95):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # 构造 Rewards 和 Values 序列 (追加 last_val 用于计算最后一步 delta)
        rews = np.append(self.rew_buf[path_slice], last_vals[None, :], axis=0)
        vals = np.append(self.val_buf[path_slice], last_vals[None, :], axis=0)
        
        # GAE 计算
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        
        # 逆序计算 Advantage
        adv = np.zeros_like(self.rew_buf[path_slice])
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + gamma * lam * last_gae_lam
            adv[t] = last_gae_lam
            
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + vals[:-1] # Returns = Adv + Value
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        返回所有数据用于 Training (转换为 Tensor)
        输出维度: (Batch_Size * Num_Agents, ...)
        """
        assert self.ptr == self.max_size # 必须存满再取
        self.ptr, self.path_start_idx = 0, 0
        
        # 归一化 Advantage
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-5)
        
        # Flatten all buffers: (T, N, ...) -> (T*N, ...)
        def flatten(arr):
            return torch.as_tensor(arr.reshape(-1, *arr.shape[2:]), dtype=torch.float32, device=self.device)
        
        def flatten_long(arr):
            return torch.as_tensor(arr.reshape(-1, *arr.shape[2:]), dtype=torch.int64, device=self.device)

        data = {
            'obs': {k: flatten(self.obs_buf[k]) for k in self.obs_buf},
            'act_man': flatten_long(self.act_man_buf),
            'act_tar': flatten_long(self.act_tar_buf),
            'ret': flatten(self.ret_buf),
            'adv': flatten(self.adv_buf),
            'logp_man': flatten(self.logp_man_buf),
            'logp_tar': flatten(self.logp_tar_buf)
        }
        return data