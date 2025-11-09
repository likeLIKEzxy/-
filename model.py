import torch
import torch.nn as nn
import math


# ===========================
# 🔹 位置编码模块
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


# ===========================
# 🔹 Encoder 模块（支持残差关闭）
# ===========================
class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, remove_residual=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.remove_residual = remove_residual

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        attn_output = self.dropout(attn_output)

        # 残差连接 + LayerNorm
        if not self.remove_residual:
            src = self.norm1(src + attn_output)
        else:
            src = self.norm1(attn_output)

        # Feed-forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        if not self.remove_residual:
            src = self.norm2(src + ff_output)
        else:
            src = self.norm2(ff_output)

        return src


# ===========================
# 🔹 Decoder 模块（支持残差关闭）
# ===========================
class TransformerDecoderLayerCustom(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, remove_residual=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.remove_residual = remove_residual

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt2 = self.dropout(tgt2)

        if not self.remove_residual:
            tgt = self.norm1(tgt + tgt2)
        else:
            tgt = self.norm1(tgt2)

        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)
        tgt2 = self.dropout(tgt2)

        if not self.remove_residual:
            tgt = self.norm2(tgt + tgt2)
        else:
            tgt = self.norm2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if not self.remove_residual:
            tgt = self.norm3(tgt + tgt2)
        else:
            tgt = self.norm3(tgt2)

        return tgt


# ===========================
# 🔹 整体 Transformer 模型
# ===========================
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1,
                 remove_positional_encoding=False,
                 remove_residual=False):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码（可选）
        self.pos_encoder = PositionalEncoding(d_model) if not remove_positional_encoding else None

        # 自定义 Encoder/Decoder 堆叠
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerCustom(d_model, nhead, dim_feedforward, dropout, remove_residual)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerCustom(d_model, nhead, dim_feedforward, dropout, remove_residual)
            for _ in range(num_decoder_layers)
        ])

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.remove_positional_encoding = remove_positional_encoding

    def encode(self, src):
        src = self.src_embed(src) * math.sqrt(self.d_model)
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        for layer in self.encoder_layers:
            src = layer(src)
        return src

    def decode(self, tgt, memory):
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        if self.pos_encoder is not None:
            tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        return tgt

    def forward(self, src, tgt):
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        output = output.transpose(0, 1)
        logits = self.output_layer(output)
        return logits
