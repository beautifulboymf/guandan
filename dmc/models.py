import torch
import torch.nn as nn
import torch.nn.functional as F

class GuandanModel(nn.Module):
    """
    掼蛋 DMC 核心模型：结合 MLP 与 Cross-Attention 架构
    专门用于评估一个 (State, Action) 组合的最终胜率。
    """
    def __init__(self, hidden_dim=256):
        super(GuandanModel, self).__init__()
        
        # ==========================================
        # 1. 独立特征编码器 (Encoders)
        # ==========================================
        # Query 编码器 (我的手牌 108 + 候选动作 108 = 216)
        self.query_encoder = nn.Sequential(
            nn.Linear(216, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context 编码器 (未知牌 108 + 级牌 13 + 剩余牌数 84 = 205)
        self.context_encoder = nn.Sequential(
            nn.Linear(205, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # History 编码器 (谁出的 4 + 出的牌 108 = 112)
        self.history_encoder = nn.Sequential(
            nn.Linear(112, hidden_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. 交叉注意力层 (Cross-Attention)
        # ==========================================
        # 使用 4 个注意力头来捕获多维度的出牌习惯
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # ==========================================
        # 3. 终极得分融合网络 (Fusion MLP)
        # ==========================================
        # 拼接后的维度: Query(hidden) + Context(hidden) + AttentionOut(hidden) = 3 * hidden_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 输出唯一的一个打分 (Q-value)
        )

    def forward(self, query, context, history, history_mask):
        """
        前向传播计算打分。
        参数维度:
            query:        [Batch_Size, 216]
            context:      [Batch_Size, 205]
            history:      [Batch_Size, 15, 112]
            history_mask: [Batch_Size, 15]  (1为真实数据，0为Padding)
        """
        # 1. 编码基础特征 -> [Batch_Size, hidden_dim]
        q_feat = F.relu(self.query_encoder(query))
        c_feat = F.relu(self.context_encoder(context))
        
        # 2. 编码历史特征 -> [Batch_Size, 15, hidden_dim]
        h_feat = self.history_encoder(history)
        
        # ==========================================
        # 3. Cross-Attention 计算
        # ==========================================
        # 核心逻辑：用当前状态 (Query + Context) 去“检索”历史记录
        # 合并局部视角和大局视角作为 Transformer 的 Query
        state_rep = q_feat + c_feat 
        
        # MultiheadAttention 需要的 Q 维度是 [Batch_Size, Seq_Len, Embed_Dim]
        # 我们只有一个 state_rep，所以 Seq_Len = 1
        attn_query = state_rep.unsqueeze(1) # [Batch_Size, 1, hidden_dim]
        
        # K 和 V 都是历史序列
        attn_key = h_feat   # [Batch_Size, 15, hidden_dim]
        attn_value = h_feat # [Batch_Size, 15, hidden_dim]
        
        # 处理掩码陷阱：
        # PyTorch 的 key_padding_mask 要求需要忽略(Padding)的位置是 True！
        # 我们的 mask 是 1 代表有效，0 代表忽略。所以这里要反转一下。
        padding_mask = (history_mask == 0.0) # [Batch_Size, 15]
        
        # 安全保护机制：如果游戏刚开局，所有的历史都是 0 (全被 mask 了)
        # PyTorch 会报错或者输出 NaN。所以如果有一整行全是 True，我们强行把第一个位置设为 False 放行。
        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            padding_mask[all_masked, 0] = False 
            
        # 扔进注意力机制
        attn_out, _ = self.cross_attention(
            query=attn_query, 
            key=attn_key, 
            value=attn_value, 
            key_padding_mask=padding_mask
        )
        
        # 拿掉 Seq_Len 那个多余的维度 -> [Batch_Size, hidden_dim]
        attn_out = attn_out.squeeze(1) 
        
        # ==========================================
        # 4. 融合与输出
        # ==========================================
        # 把“当前的企图 (q)”、“场上的大局 (c)”、以及“历史的教训 (attn_out)” 拼在一起！
        final_rep = torch.cat([q_feat, c_feat, attn_out], dim=1) # [Batch_Size, 256 * 3 = 768]
        
        # 计算最终得分 -> [Batch_Size, 1]
        score = self.fusion_net(final_rep)
        
        return score