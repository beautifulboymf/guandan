import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """标准的残差块，带有 LayerNorm 和 LeakyReLU，稳定深度网络的梯度"""
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        out = self.ln2(self.fc2(out))
        return F.leaky_relu(out + residual, negative_slope=0.01)

class GuandanModel(nn.Module):
    """
    掼蛋 DMC 终极模型：多分支特征均衡(Embedding) + ResNet + Cross-Attention
    """
    def __init__(self, hidden_dim=256):
        super(GuandanModel, self).__init__()
        
        # ==========================================
        # 1. Query Encoder (216 -> 256)
        # ==========================================
        self.query_encoder = nn.Sequential(
            nn.Linear(216, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            ResBlock(hidden_dim)
        )
        
        # ==========================================
        # 2. Context 深度重构 (解决信息维度碾压)
        # ==========================================
        # 未知牌: 108维 -> 128维
        self.unseen_emb = nn.Sequential(nn.Linear(108, 128), nn.LeakyReLU(0.01))
        
        # 级牌整数(0~12): 通过 Embedding 映射为 64 维稠密向量
        self.level_emb = nn.Embedding(num_embeddings=13, embedding_dim=64)
        
        # 三人牌数整数(0~27): 通过 Embedding 为每个数字映射 32 维稠密向量
        self.cards_num_emb = nn.Embedding(num_embeddings=28, embedding_dim=32)
        
        # 融合拼接: 128(未知牌) + 64(级牌向量) + 3*32(三个人牌数向量) = 288 维
        self.context_fusion = nn.Sequential(
            nn.Linear(288, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim)
        )
        
        # ==========================================
        # 3. History 深度重构 (解耦位置与动作)
        # ==========================================
        # 位置: 4维 -> 升维到 32维
        self.hist_pos_emb = nn.Sequential(nn.Linear(4, 32), nn.LeakyReLU(0.01))
        # 牌型: 108维 -> 升维到 128维
        self.hist_act_emb = nn.Sequential(nn.Linear(108, 128), nn.LeakyReLU(0.01))
        
        # 拼接融合: 32 + 128 = 160 维 -> 映射到 hidden_dim (256)
        self.history_fusion = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim)
        )

        # ==========================================
        # 4. Cross-Attention
        # ==========================================
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1 
        )
        self.attn_ln = nn.LayerNorm(hidden_dim) 
        
        # ==========================================
        # 5. Fusion MLP (融合全维特征输出最终得分)
        # ==========================================
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            ResBlock(512),       
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)    
        )

    def forward(self, query, context, history, history_mask):
        # 1. 编码 Query
        q_feat = self.query_encoder(query)
        
        # 2. 截断、提取并分别查表/编码 Context
        # context 现长度 112 = 108(unseen) + 1(level整数) + 3(cards_num整数)
        unseen_cards = context[:, :108]
        level_info = context[:, 108].long()          # 第 108 位是级牌
        cards_num = context[:, 109:112].long()       # 第 109~111 位是三人剩余牌数
        
        u_feat = self.unseen_emb(unseen_cards)
        l_feat = self.level_emb(level_info)          # 查表得到 [Batch, 64]
        n_feat = self.cards_num_emb(cards_num).view(-1, 3 * 32) # 查表 [Batch, 3, 32] -> 展平 [Batch, 96]
        
        # 拼接融合
        c_feat = self.context_fusion(torch.cat([u_feat, l_feat, n_feat], dim=1))
        
        # 3. 截断、提取并分别编码 History
        # history 原长度 112 = 4(pos) + 108(act)
        pos_info = history[:, :, :4]
        act_info = history[:, :, 4:]
        
        p_feat = self.hist_pos_emb(pos_info)         # [Batch, Seq_Len, 32]
        a_feat = self.hist_act_emb(act_info)         # [Batch, Seq_Len, 128]
        
        h_feat = self.history_fusion(torch.cat([p_feat, a_feat], dim=-1))
        
        # 4. Cross-Attention
        state_rep = q_feat + c_feat 
        attn_query = state_rep.unsqueeze(1) 
        
        padding_mask = (history_mask == 0.0) 
        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            padding_mask[all_masked, 0] = False 
            
        attn_out, _ = self.cross_attention(
            query=attn_query, 
            key=h_feat, 
            value=h_feat, 
            key_padding_mask=padding_mask
        )
        attn_out = self.attn_ln(attn_out.squeeze(1) + state_rep) 
        
        # 5. 输出打分
        final_rep = torch.cat([q_feat, c_feat, attn_out], dim=1) 
        score = self.fusion_net(final_rep)
        
        return score