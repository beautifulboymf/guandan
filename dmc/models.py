import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    """加宽版残差块：增强非线性表达能力"""
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.ln1 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.ln1(self.fc1(x))) 
        out = self.ln2(self.fc2(out))
        return out + residual

class RoPESelfAttention(nn.Module):
    """旋转位置编码自注意力机制"""
    # 【修复处】：默认序列长度扩容到 256，完美向下兼容 128 的记忆长度！
    def __init__(self, dim, num_heads=8, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :]) 
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def apply_rotary_pos_emb(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def forward(self, x, padding_mask=None):
        residual = x
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = self.apply_rotary_pos_emb(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.ln(self.proj(out) + residual)

class GuandanModel(nn.Module):
    """
    【上帝视角重构版】双塔分离架构
    彻底解耦 State (局势) 与 Action (动作)，解决特征稀释，融入读心术头。
    """
    def __init__(self, hidden_dim=512): 
        super(GuandanModel, self).__init__()
        
        # ==========================================
        # 1. 局势感知模块 (State Encoders) - 防特征稀释设计
        # ==========================================
        self.hand_proj = nn.Sequential(nn.Linear(108, 256), nn.LayerNorm(256), nn.GELU())
        self.unseen_proj = nn.Sequential(nn.Linear(108, 256), nn.LayerNorm(256), nn.GELU())
        
        # 【核心新增】：宏观大局观特征升维器 (极其关键，防止 24 维特征被 512 维淹没)
        self.macro_proj = nn.Sequential(nn.Linear(24, 256), nn.LayerNorm(256), nn.GELU())
        
        self.level_emb = nn.Embedding(num_embeddings=13, embedding_dim=128) 
        self.cards_num_emb = nn.Embedding(num_embeddings=28, embedding_dim=64) 
        
        # 将上述所有状态特征完美融合
        # 256(hand) + 256(unseen) + 256(macro) + 128(level) + 192(3个玩家数量) = 1088
        self.state_mlp = nn.Sequential(
            nn.Linear(1088, hidden_dim), nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim)
        )
        
        # ==========================================
        # 2. 历史与注意力交互模块 (History & Attention)
        # ==========================================
        self.hist_pos_emb = nn.Sequential(nn.Linear(4, 64), nn.GELU())
        self.hist_act_emb = nn.Sequential(nn.Linear(108, 192), nn.GELU())
        self.history_proj = nn.Sequential(nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim))
        
        # 【核心修复】：显式将 max_seq_len 设为 128
        self.history_rope_attn = RoPESelfAttention(hidden_dim, num_heads=8, max_seq_len=128)
        self.history_res = ResBlock(hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.1 
        )
        self.attn_ln = nn.LayerNorm(hidden_dim) 
        
        # ==========================================
        # 3. 终极双塔输出 (Twin-Towers)
        # ==========================================
        # A 塔：【读心头】(Prediction Head)，用于输出另外 3 家 324 维的上帝视角手牌
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, 324) # 输出 324 维 Logits，不要加 Sigmoid，方便后续算 BCEWithLogitsLoss
        )
        
        # B 塔：【价值头】(Value Head)，用于评估具体的打牌动作
        self.action_proj = nn.Sequential(nn.Linear(108, hidden_dim), nn.LayerNorm(hidden_dim), ResBlock(hidden_dim))
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            ResBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)    
        )

    def forward(self, query, context, history, history_mask, return_preds=False, inference=False):
        """
        前向传播
        inference=True 时，开启极致加速：只计算 1 次 State 和 History，复用于所有 Action！
        """
        my_hand_feat = query[:, :108]
        action_feat = query[:, 108:]
        
        if inference:
            # 【史诗级加速】：局势特征对所有候选动作都是一模一样的，只取第 0 个算 1 次！
            _h_emb = self.hand_proj(my_hand_feat[0:1])
            _u_emb = self.unseen_proj(context[0:1, :108])
            _l_emb = self.level_emb(context[0:1, 108].long())
            _n_emb = self.cards_num_emb(context[0:1, 109:112].long()).view(1, -1) 
            _m_emb = self.macro_proj(context[0:1, 112:136]) 
            
            state_feat = self.state_mlp(torch.cat([_h_emb, _u_emb, _l_emb, _n_emb, _m_emb], dim=1))
            
            _hist_cat = torch.cat([
                self.hist_pos_emb(history[0:1, :, :4]), 
                self.hist_act_emb(history[0:1, :, 4:])
            ], dim=-1)
            hist_feat = self.history_proj(_hist_cat)
            
            padding_mask = (history_mask[0:1] == 0.0) 
            if padding_mask.all(dim=1).any(): padding_mask[padding_mask.all(dim=1), 0] = False 
                
            hist_feat = self.history_rope_attn(hist_feat, padding_mask=padding_mask)
            hist_feat = self.history_res(hist_feat)
            
            attn_out, _ = self.cross_attention(
                query=state_feat.unsqueeze(1), key=hist_feat, value=hist_feat, key_padding_mask=padding_mask
            )
            final_state_rep = self.attn_ln(attn_out.squeeze(1) + state_feat) 
            
            # 【极其关键】：算完唯一的一个终极表征后，复制扩展成和 action 一样的数量！
            final_state_rep = final_state_rep.expand(query.shape[0], -1)
            
        else:
            # 训练模式：正常处理 Batch 中不同的游戏局势
            h_emb = self.hand_proj(my_hand_feat)
            u_emb = self.unseen_proj(context[:, :108])
            l_emb = self.level_emb(context[:, 108].long())
            n_emb = self.cards_num_emb(context[:, 109:112].long()).view(-1, 3 * 64) 
            m_emb = self.macro_proj(context[:, 112:136]) 
            
            state_feat = self.state_mlp(torch.cat([h_emb, u_emb, l_emb, n_emb, m_emb], dim=1))
            
            hist_feat = self.history_proj(torch.cat([
                self.hist_pos_emb(history[:, :, :4]), 
                self.hist_act_emb(history[:, :, 4:])
            ], dim=-1))
            
            padding_mask = (history_mask == 0.0) 
            if padding_mask.all(dim=1).any(): padding_mask[padding_mask.all(dim=1), 0] = False 
                
            hist_feat = self.history_rope_attn(hist_feat, padding_mask=padding_mask)
            hist_feat = self.history_res(hist_feat)
            
            attn_out, _ = self.cross_attention(
                query=state_feat.unsqueeze(1), key=hist_feat, value=hist_feat, key_padding_mask=padding_mask
            )
            final_state_rep = self.attn_ln(attn_out.squeeze(1) + state_feat) 
            
        # --- 4. 读心术塔 (Prediction Head) ---
        hidden_preds = None
        if return_preds:
            hidden_preds = self.pred_head(final_state_rep)
            
        # --- 5. 动作打分塔 (Value Head) ---
        # 具体动作各不相同，必须全部独立计算
        a_emb = self.action_proj(action_feat)
        q_value = self.q_head(torch.cat([final_state_rep, a_emb], dim=1))
        
        if return_preds:
            return q_value, hidden_preds
        return q_value