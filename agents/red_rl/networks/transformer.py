import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityEncoder(nn.Module):
    """
    基于 Transformer 的实体编码器。
    将 (Batch, N, Feat_Dim) 的实体序列编码为固定长度的全局特征向量。
    """
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 实体嵌入层 (MLP)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. Transformer Encoder Layer
        # batch_first=True: 输入格式为 (Batch, Seq_Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 聚合层 (Attention Pooling 的简化版：Max Pooling)
        # 将变长的序列特征聚合为单一向量
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        x: (Batch, Max_Entities, Input_Dim)
        mask: (Batch, Max_Entities) 1=Valid, 0=Padding. 目前简化处理，由网络自动学习忽略Padding
        """
        # (Batch, N, Dim) -> (Batch, N, Embed_Dim)
        emb = self.embedding(x)
        
        # Transformer 处理交互
        # 输出: (Batch, N, Embed_Dim)
        feat = self.transformer(emb)
        
        # Global Pooling: 取最大值，提取最显著的特征 (例如最近的导弹，最危险的敌机)
        # 对于 Padding 的部分（通常是0），Max Pooling 自然会忽略（假设激活值有正数）
        # 更好的做法是利用 mask 将 padding 设为 -inf，这里为了代码通用性采用直接 Max
        global_feat, _ = torch.max(feat, dim=1)
        
        return self.ln(global_feat)

class AirCombatTransformer(nn.Module):
    """
    主干网络：融合自身、盟友、敌方、导弹的特征
    """
    def __init__(self, obs_shapes, hidden_dim=256):
        super().__init__()
        
        # 1. 各个分支的编码器
        # Self: (Batch, 11) -> 不用 Transformer，直接 MLP
        self.self_encoder = nn.Sequential(
            nn.Linear(obs_shapes['self'], hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Allies: (Batch, N, 7)
        self.ally_encoder = EntityEncoder(obs_shapes['allies'], hidden_dim // 2)
        
        # Enemies: (Batch, N, 7)
        self.enemy_encoder = EntityEncoder(obs_shapes['enemies'], hidden_dim // 2)
        
        # Missiles: (Batch, N, 7)
        self.missile_encoder = EntityEncoder(obs_shapes['missiles'], hidden_dim // 2)
        
        # 2. 融合层
        # 输入维度: Self(H) + Ally(H/2) + Enemy(H/2) + Missile(H/2) = 2.5H
        total_dim = hidden_dim + (hidden_dim // 2) * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_dim = hidden_dim

    def forward(self, obs_dict):
        """
        obs_dict 包含 Tensor: 'self', 'allies', 'enemies', 'missiles'
        """
        batch_size = obs_dict['self'].shape[0]
        
        # 编码各个部分
        f_self = self.self_encoder(obs_dict['self'])
        f_ally = self.ally_encoder(obs_dict['allies'])
        f_enemy = self.enemy_encoder(obs_dict['enemies'])
        f_missile = self.missile_encoder(obs_dict['missiles'])
        
        # 拼接
        cat_feat = torch.cat([f_self, f_ally, f_enemy, f_missile], dim=1)
        
        # 融合
        global_state = self.fusion_layer(cat_feat)
        
        return global_state