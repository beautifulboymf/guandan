import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 核心组件 1：Sparse MoE (稀疏混合专家网络)
# ==========================================
class SparseMoE(nn.Module):
    """
    基于 Top-2 路由的稀疏混合专家模块。
    极大提升模型参数量（算力上限），但不显著增加单步推理的 FLOPs。
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # 每个专家是一个加宽的 MLP 块
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1]) # 展平以适应任意维度的输入
        residual = x_flat
        
        # 路由选择
        gates = F.softmax(self.gate(x_flat), dim=-1)
        topk_weights, topk_indices = torch.topk(gates, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) # 归一化权重

        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i:i+1]
            
            for exp_id, expert in enumerate(self.experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    # 仅对路由到该专家的 Token 计算
                    out[mask] += weight[mask] * expert(x_flat[mask])
                    
        return self.ln(out + residual).view(original_shape)

# ==========================================
# 核心组件 2：带有 RoPE 的 Self-Attention 
# ==========================================
class RoPESelfAttention(nn.Module):
    """
    注入了 RoPE (Rotary Position Embedding) 旋转位置编码的自注意力模块。
    用于让 AI 极速精准理解 "刚才那一手" 和 "10手以前那一手" 的时间差距。
    """
    def __init__(self, dim, num_heads=8, max_seq_len=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        
        # 预计算 RoPE 的频率矩阵
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :]) # [1, 1, seq_len, head_dim]
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def apply_rotary_pos_emb(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
            
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x, padding_mask=None):
        residual = x
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 施加旋转位置编码！灵魂所在！
        q, k = self.apply_rotary_pos_emb(q, k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if padding_mask is not None:
            # padding_mask 中 True 代表是被 zero-padding 的冗余序列，需要屏蔽 (-inf)
            mask = padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.ln(self.proj(out) + residual)

# ==========================================
# 终极版 AI 架构 (含 MoE & RoPE)
# ==========================================
class GuandanModel(nn.Module):
    def __init__(self, hidden_dim=512): # 【升级】全网特征维度翻倍至 512
        super(GuandanModel, self).__init__()
        
        # 1. Query Encoder (深度加码)
        self.query_encoder = nn.Sequential(
            nn.Linear(216, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            SparseMoE(hidden_dim, num_experts=4)
        )
        
        # 2. Context Encoder
        self.unseen_emb = nn.Sequential(nn.Linear(108, 128), nn.LeakyReLU(0.01))
        self.level_emb = nn.Embedding(num_embeddings=13, embedding_dim=128) # 64 -> 128
        self.cards_num_emb = nn.Embedding(num_embeddings=28, embedding_dim=64) # 32 -> 64
        
        # Fusion: 128(unseen) + 128(level) + 3*64(cards_num) = 448
        self.context_fusion = nn.Sequential(
            nn.Linear(448, hidden_dim),
            nn.LayerNorm(hidden_dim),
            SparseMoE(hidden_dim, num_experts=4)
        )
        
        # 3. History Encoder 
        self.hist_pos_emb = nn.Sequential(nn.Linear(4, 64), nn.LeakyReLU(0.01))
        self.hist_act_emb = nn.Sequential(nn.Linear(108, 192), nn.LeakyReLU(0.01))
        
        self.history_proj = nn.Sequential(
            nn.Linear(256, hidden_dim), # 64 + 192 = 256
            nn.LayerNorm(hidden_dim)
        )
        
        # 【神级升级】：用带 RoPE 的 Self-Attention 替代原来的 MLP 处理内部时序
        self.history_rope_attn = RoPESelfAttention(hidden_dim, num_heads=8, max_seq_len=64)
        self.history_moe = SparseMoE(hidden_dim, num_experts=4)

        # 4. Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.1 
        )
        self.attn_ln = nn.LayerNorm(hidden_dim) 
        
        # 5. Fusion Net (终极融合也换成专家模块)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            SparseMoE(hidden_dim * 2, num_experts=4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1)    
        )

    def forward(self, query, context, history, history_mask):
        # 1. 编码 Query
        q_feat = self.query_encoder(query)
        
        # 2. 编码 Context
        unseen_cards = context[:, :108]
        level_info = context[:, 108].long() 
        cards_num = context[:, 109:112].long() 
        
        u_feat = self.unseen_emb(unseen_cards)
        l_feat = self.level_emb(level_info)
        n_feat = self.cards_num_emb(cards_num).view(-1, 3 * 64) 
        c_feat = self.context_fusion(torch.cat([u_feat, l_feat, n_feat], dim=1))
        
        # 3. 编码 History 序列
        pos_info = history[:, :, :4]
        act_info = history[:, :, 4:]
        p_feat = self.hist_pos_emb(pos_info) 
        a_feat = self.hist_act_emb(act_info) 
        
        h_feat = self.history_proj(torch.cat([p_feat, a_feat], dim=-1))
        
        # 处理 padding mask (True 表示这是无效 padding 数据，应该忽略)
        padding_mask = (history_mask == 0.0) 
        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            padding_mask[all_masked, 0] = False # 防止全 False 时 attention 报错
            
        # 运用 RoPE 自注意力彻底解构历史牌局时间线
        h_feat = self.history_rope_attn(h_feat, padding_mask=padding_mask)
        h_feat = self.history_moe(h_feat)
        
        # 4. 交叉注意力：用当前的(状态+候选动作)去审查历史记录
        state_rep = q_feat + c_feat 
        attn_query = state_rep.unsqueeze(1) 
        
        attn_out, _ = self.cross_attention(
            query=attn_query, key=h_feat, value=h_feat, key_padding_mask=padding_mask
        )
        attn_out = self.attn_ln(attn_out.squeeze(1) + state_rep) 
        
        # 5. 输出打分
        final_rep = torch.cat([q_feat, c_feat, attn_out], dim=1) 
        score = self.fusion_net(final_rep)
        
        return score