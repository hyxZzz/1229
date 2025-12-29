import torch
import torch.nn as nn
from agents.red_rl.networks.transformer import AirCombatTransformer
from agents.red_rl.networks.heads import PolicyHead, ValueHead

class RedPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义输入维度 (参考 obs_parser.py)
        obs_shapes = {
            'self': 11,
            'allies': 7,
            'enemies': 7,
            'missiles': 7
        }
        
        # 定义动作空间
        act_dims = {
            'maneuver': 19, # 0-18
            'target': 8    # 8 enemies (index 0-7), output dim will be 9
        }
        
        # 初始化网络
        self.backbone = AirCombatTransformer(obs_shapes, hidden_dim=256)
        self.actor = PolicyHead(self.backbone.output_dim, act_dims)
        self.critic = ValueHead(self.backbone.output_dim)
        
        self.to(self.device)

    def _process_obs(self, obs_dict):
        """将 numpy dict 转换为 tensor dict 并移动到 device"""
        t_obs = {}
        for k, v in obs_dict.items():
            t_obs[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            # 增加 Batch 维度 (如果输入是单个 step)
            if t_obs[k].ndim == 1 or (k!='self' and t_obs[k].ndim == 2):
                t_obs[k] = t_obs[k].unsqueeze(0) 
        return t_obs

    def act(self, obs_dict, deterministic=False):
        """
        环境交互用：输入观测，输出动作
        """
        with torch.no_grad():
            t_obs = self._process_obs(obs_dict)
            features = self.backbone(t_obs)

            enemy_mask = t_obs['enemies'][:, :, 6] # (Batch, 8)
            
            man_logits, tar_logits = self.actor(features, masking_info=enemy_mask)
            
            # 1. 采样机动动作
            dist_man = torch.distributions.Categorical(logits=man_logits)
            if deterministic:
                action_man = torch.argmax(man_logits, dim=1)
            else:
                action_man = dist_man.sample()
                
            # 2. 采样攻击目标
            dist_tar = torch.distributions.Categorical(logits=tar_logits)
            if deterministic:
                action_tar = torch.argmax(tar_logits, dim=1)
            else:
                action_tar = dist_tar.sample()
            
            # 计算 log_prob (用于 Training Buffer)
            logp_man = dist_man.log_prob(action_man)
            logp_tar = dist_tar.log_prob(action_tar)
            
            # 获取 Value
            val = self.critic(features).squeeze(-1)
            
        return (action_man.cpu().numpy(), action_tar.cpu().numpy()), \
               (logp_man.cpu().numpy(), logp_tar.cpu().numpy()), \
               val.cpu().numpy()

    def evaluate(self, obs_dict, act_man, act_tar):
        """
        训练更新用：输入观测和动作，计算梯度所需数据
        """
        # obs_dict 已经是 tensor (由 Buffer 处理)
        features = self.backbone(obs_dict)
        
        man_logits, tar_logits = self.actor(features)
        v = self.critic(features).squeeze(-1)
        
        # 机动分布
        dist_man = torch.distributions.Categorical(logits=man_logits)
        logp_man = dist_man.log_prob(act_man)
        entropy_man = dist_man.entropy()
        
        # 目标分布
        dist_tar = torch.distributions.Categorical(logits=tar_logits)
        logp_tar = dist_tar.log_prob(act_tar)
        entropy_tar = dist_tar.entropy()
        
        return (logp_man, logp_tar), (entropy_man, entropy_tar), v
