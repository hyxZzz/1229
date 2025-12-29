import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyHead(nn.Module):
    def __init__(self, input_dim, action_dims):
        super().__init__()
        
        # 1. 飞行机动策略 (Discrete 7)
        # Cruise, Max_G, Zoom, Split_S, Barrel, Snake, YoYo
        self.maneuver_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims['maneuver']) # Output logits
        )
        
        # 2. 攻击目标选择 (Discrete 9: 8 enemies + 1 no_fire)
        # 对应：[打敌机0, 打敌机1, ... , 打敌机7, 不开火]
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims['target'] + 1) # +1 for "No Fire"
        )
        
    def forward(self, x, masking_info=None):  # <--- 修改这里：参数名必须是 masking_info
        """
        x: 全局特征向量
        masking_info: (Batch, 8) 1=Alive, 0=Dead. 用于屏蔽已死亡的敌机
        """
        maneuver_logits = self.maneuver_net(x)
        target_logits = self.target_net(x) # Shape: (Batch, 9)
        
        # --- Action Masking 逻辑 ---
        if masking_info is not None:
            # target_logits 前8位对应8个敌人，最后1位(索引8)是不开火
            # 我们只处理前8位，将死亡(0)对应的 logits 设为负无穷
            
            # 创建一个极小的数
            huge_neg = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
            
            # masking_info 是 (Batch, 8)，target_logits 是 (Batch, 9)
            # 使用 where: 如果 mask是1(活)，保持原值；如果是0(死)，替换为负无穷
            target_logits[:, :8] = torch.where(
                masking_info.bool(), 
                target_logits[:, :8], 
                huge_neg
            )

        return maneuver_logits, target_logits

class ValueHead(nn.Module):
    """Critic: 评估当前局面的胜率/价值"""
    def __init__(self, input_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.v_net(x)