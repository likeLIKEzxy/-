# src/decoder.py
import math
import torch
import torch.nn as nn
from src.attention import MultiHeadSelfAttention, scaled_dot_product_attention, causal_mask

# FeedForward and PositionalEncoding will be imported from model.py if needed,
# but for isolation we can define simple ones here if you plan to run independently.
# Otherwise, they will be referenced when imported in model.py.

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadCrossAttention(nn.Module):
    """Cross-Attention: Q from decoder, K/V from encoder."""
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

    def forward(self, x_q, kv, attn_mask=None):
        # x_q: (B, L_q, d_model)  - decoder queries
        # kv: (B, L_kv, d_model) - encoder outputs (keys & values)
        B, Lq, _ = x_q.size()
        Lk = kv.size(1)
        device = x_q.device

        q = self.q_lin(x_q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_lin(kv).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_lin(kv).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        out, attn = scaled_dot_product_attention(q, k, v, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_lin(out)
        out = self.dropout(out)
        return out, attn


class TransformerDecoderLayer(nn.Module):
    """Single layer of the Transformer decoder."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_kv, self_mask=None, cross_mask=None):
        if self.pre_norm:
            y = self.norm1(x)
            self_out, self_attn = self.self_attn(y, self_mask)
            x = x + self_out

            z = self.norm2(x)
            cross_out, cross_attn = self.cross_attn(z, enc_kv, cross_mask)
            x = x + cross_out

            w = self.norm3(x)
            ff_out = self.ffn(w)
            x = x + ff_out
        else:
            self_out, self_attn = self.self_attn(x, self_mask)
            x = x + self.dropout(self_out); x = self.norm1(x)
            cross_out, cross_attn = self.cross_attn(x, enc_kv, cross_mask)
            x = x + self.dropout(cross_out); x = self.norm2(x)
            ff_out = self.ffn(x)
            x = x + self.dropout(ff_out); x = self.norm3(x)

        return x, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    """Full Transformer decoder stack."""
    def __init__(self, vocab_size, d_model=128, n_layers=2, num_heads=4, d_ff=512,
                 max_seq_len=512, dropout=0.1, pos_enc_cls=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if pos_enc_cls is not None:
            self.pos_enc = pos_enc_cls(d_model, max_len=max_seq_len)
        else:
            # If no positional encoding class provided, use sinusoidal by default
            from src.model import PositionalEncoding
            self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_lin = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt_ids, enc_kv, return_hidden=False):
        # tgt_ids: (B, L_tgt)
        B, Lt = tgt_ids.size()
        device = tgt_ids.device

        tok = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        pos = self.pos_enc(tok).to(device)
        x = tok + pos

        self_mask = causal_mask(Lt, device)
        cross_mask = None

        self_attns, cross_attns = [], []
        for layer in self.layers:
            x, sa, ca = layer(x, enc_kv, self_mask=self_mask, cross_mask=cross_mask)
            self_attns.append(sa)
            cross_attns.append(ca)

        x = self.norm(x)
        logits = self.output_lin(x)
        if return_hidden:
            return logits, x, self_attns, cross_attns
        return logits, self_attns, cross_attns
