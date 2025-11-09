# src/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(seq_len, device):
    # returns (1, 1, seq_len, seq_len) mask with 0 for allowed, -inf for blocked
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
    # mask: 1 for allowed, 0 for masked
    return mask

def scaled_dot_product_attention(q, k, v, attn_mask=None):
    # q,k,v: (..., seq_len, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        # attn_mask expected to be broadcastable to scores; uses 1 for allowed, 0 for masked
        scores = scores.masked_fill(attn_mask == 0, float('-1e9'))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, L, d_model)
        B, L, _ = x.size()
        # linear projections
        q = self.q_lin(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        k = self.k_lin(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            # attn_mask expected (1,1,L,L) or (B,1,L,L), broadcastable
            attn_mask = attn_mask.to(x.device)

        out, attn = scaled_dot_product_attention(q, k, v, attn_mask)
        # out: (B, h, L, d_k) -> concat heads
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_lin(out)
        out = self.dropout(out)
        return out, attn
