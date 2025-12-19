from typing import Optional

import torch
from torch import nn


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ff_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        cross_out = self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(cross_out)

        x_norm = self.norm3(x)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ff_out)
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.src_token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.tgt_token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def _positional_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions)

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, src_len = src.size()
        pos = self._positional_embedding(src_len, src.device)
        x = self.src_token_emb(src) + pos.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, tgt_len = tgt.size()
        pos = self._positional_embedding(tgt_len, tgt.device)
        y = self.tgt_token_emb(tgt) + pos.unsqueeze(0).expand(batch_size, -1, -1)
        y = self.dropout(y)

        tgt_mask = _causal_mask(tgt_len, tgt.device)
        for layer in self.decoder_layers:
            y = layer(
                y,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
        y = self.decoder_norm(y)
        return self.output_proj(y)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(tgt_in, memory, src_key_padding_mask, tgt_key_padding_mask)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
        max_len: int,
        bos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask)
        batch_size = src.size(0)

        ys = torch.full(
            (batch_size, 1), bos_token_id, device=src.device, dtype=torch.long
        )
        for _ in range(max_len):
            tgt_key_padding_mask = ys.eq(pad_token_id)
            logits = self.decode(ys, memory, src_key_padding_mask, tgt_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

        return ys[:, 1:]
