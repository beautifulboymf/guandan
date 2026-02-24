import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    """最稳如老狗的加宽版残差块"""
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        # 中间隐藏层扩大两倍，增强非线性表达能力
        self.fc1 = nn.Linear(dim, dim * 2)
        self.ln1 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.ln1(self.fc1(x))) # 使用大模型标配的 GELU 激活函数
        out = self.ln2(self.fc2(out))
        return out + residual

class RoPESelfAttention(nn.Module):
    # 【神级修复】：将预分配的最大序列长度直接扩容到 256！
    # 这样不管你以后把 MAX_HISTORY 改成 64 还是 128，它都能动态自适应，永远不报错！
    def __init__(self, dim, num_heads=8, max_seq_len=256):
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
    def __init__(self, hidden_dim=512): # 【算力定海神针】：512维起步
        super(GuandanModel, self).__init__()
        
        self.query_encoder = nn.Sequential(
            nn.Linear(216, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), ResBlock(hidden_dim)
        )
        
        self.unseen_emb = nn.Sequential(nn.Linear(108, 128), nn.GELU())
        self.level_emb = nn.Embedding(num_embeddings=13, embedding_dim=128) 
        self.cards_num_emb = nn.Embedding(num_embeddings=28, embedding_dim=64) 
        
        self.context_fusion = nn.Sequential(
            nn.Linear(448, hidden_dim), nn.LayerNorm(hidden_dim),
            ResBlock(hidden_dim)
        )
        
        self.hist_pos_emb = nn.Sequential(nn.Linear(4, 64), nn.GELU())
        self.hist_act_emb = nn.Sequential(nn.Linear(108, 192), nn.GELU())
        self.history_proj = nn.Sequential(nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim))
        
        # 灵魂组件：RoPE 自注意力 + 强力 ResBlock 收尾
        self.history_rope_attn = RoPESelfAttention(hidden_dim, num_heads=8, max_seq_len=64)
        self.history_res = ResBlock(hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.1 
        )
        self.attn_ln = nn.LayerNorm(hidden_dim) 
        
        # 终极打分网络（深层强拟合）
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            ResBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)    
        )

    def forward(self, query, context, history, history_mask):
        q_feat = self.query_encoder(query)
        
        u_feat = self.unseen_emb(context[:, :108])
        l_feat = self.level_emb(context[:, 108].long())
        n_feat = self.cards_num_emb(context[:, 109:112].long()).view(-1, 3 * 64) 
        c_feat = self.context_fusion(torch.cat([u_feat, l_feat, n_feat], dim=1))
        
        h_feat = self.history_proj(torch.cat([
            self.hist_pos_emb(history[:, :, :4]), 
            self.hist_act_emb(history[:, :, 4:])
        ], dim=-1))
        
        padding_mask = (history_mask == 0.0) 
        if padding_mask.all(dim=1).any(): padding_mask[padding_mask.all(dim=1), 0] = False 
            
        h_feat = self.history_rope_attn(h_feat, padding_mask=padding_mask)
        h_feat = self.history_res(h_feat)
        
        state_rep = q_feat + c_feat 
        attn_out, _ = self.cross_attention(
            query=state_rep.unsqueeze(1), key=h_feat, value=h_feat, key_padding_mask=padding_mask
        )
        attn_out = self.attn_ln(attn_out.squeeze(1) + state_rep) 
        
        return self.fusion_net(torch.cat([q_feat, c_feat, attn_out], dim=1))