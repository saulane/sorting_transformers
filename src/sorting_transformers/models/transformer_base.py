from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from sorting_transformers.models.positional import AlibiBias, PositionalEncoding, RelativePositionBias, RotaryEmbedding


@dataclass
class AttentionProjections:
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        comparison_normalize: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        position_bias: Optional[RelativePositionBias | AlibiBias] = None,
        shared_projections: Optional[AttentionProjections] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        self.position_bias = position_bias
        self.comparison_normalize = comparison_normalize

        if shared_projections is None:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        else:
            self.q_proj = shared_projections.q_proj
            self.k_proj = shared_projections.k_proj
            self.v_proj = shared_projections.v_proj
            self.out_proj = shared_projections.out_proj

        if comparison_normalize:
            self.q_norm = nn.LayerNorm(d_model)
            self.kv_norm = nn.LayerNorm(d_model)
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.q_norm = None
            self.kv_norm = None
            self.temperature = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        return_head_output: bool = False,
        head_override: Optional[torch.Tensor] = None,
        head_override_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        kv_input = x if kv is None else kv
        q_input = x
        if self.comparison_normalize:
            q_input = self.q_norm(q_input) * self.temperature
            kv_input = self.kv_norm(kv_input) * self.temperature

        q = self._shape(self.q_proj(q_input))
        k = self._shape(self.k_proj(kv_input))
        v = self._shape(self.v_proj(kv_input))

        if self.rope is not None and kv is None:
            q, k = self.rope(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.position_bias is not None and kv is None:
            bias = self.position_bias.get_bias(attn_scores.size(-1), attn_scores.device)
            attn_scores = attn_scores + bias.unsqueeze(0)

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        if head_override is not None:
            if head_override_mask is None:
                attn_output = head_override
            else:
                head_mask = head_override_mask.view(1, -1, 1, 1)
                attn_output = torch.where(head_mask, head_override, attn_output)

        merged = attn_output.transpose(1, 2).reshape(x.size(0), -1, self.d_model)
        out = self.out_proj(merged)
        out = self.dropout(out)

        head_out = attn_output if return_head_output else None
        weights = attn_weights if need_weights else None
        return out, weights, head_out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        comparison_normalize: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        position_bias: Optional[RelativePositionBias | AlibiBias] = None,
        shared_projections: Optional[AttentionProjections] = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            n_heads,
            dropout,
            comparison_normalize=comparison_normalize,
            rope=rope,
            position_bias=position_bias,
            shared_projections=shared_projections,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        return_head_output: bool = False,
        head_override: Optional[torch.Tensor] = None,
        head_override_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        x_norm = self.norm1(x)
        attn_out, attn_weights, head_out = self.self_attn(
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            return_head_output=return_head_output,
            head_override=head_override,
            head_override_mask=head_override_mask,
        )
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = x + self.dropout(ff_out)

        extras = {}
        if return_attn:
            extras["attn_weights"] = attn_weights
        if return_head_output:
            extras["head_output"] = head_out
        return x, extras


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        comparison_normalize: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        position_bias: Optional[RelativePositionBias | AlibiBias] = None,
        shared_projections: Optional[AttentionProjections] = None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            n_heads,
            dropout,
            comparison_normalize=comparison_normalize,
            rope=rope,
            position_bias=position_bias,
            shared_projections=shared_projections,
        )
        self.cross_attn = MultiheadAttention(
            d_model,
            n_heads,
            dropout,
            comparison_normalize=comparison_normalize,
            rope=None,
            position_bias=None,
            shared_projections=shared_projections,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        x_norm = self.norm1(x)
        attn_out, attn_weights, _ = self.self_attn(
            x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=return_attn,
        )
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        cross_out, cross_weights, _ = self.cross_attn(
            x_norm,
            kv=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attn,
        )
        x = x + self.dropout(cross_out)

        x_norm = self.norm3(x)
        ff_out = self.ffn(x_norm)
        x = x + self.dropout(ff_out)

        extras = {}
        if return_attn:
            extras["self_attn_weights"] = attn_weights
            extras["cross_attn_weights"] = cross_weights
        return x, extras


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
        positional: str = "absolute",
        tie_attention: bool = False,
        comparison_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.positional = positional

        self.src_token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)

        self.pos_emb = None
        rope = None
        position_bias = None
        if positional == "absolute":
            self.pos_emb = PositionalEncoding(max_seq_len, d_model)
        elif positional == "rel_bias":
            position_bias = RelativePositionBias(n_heads=n_heads, max_len=max_seq_len)
            self.register_parameter("rel_bias_weight", position_bias.weight)
        elif positional == "rope":
            rope = RotaryEmbedding(d_model // n_heads)
        elif positional == "alibi":
            position_bias = AlibiBias(n_heads=n_heads)
        else:
            raise ValueError(f"Unknown positional mode: {positional}")

        shared_proj = None
        if tie_attention:
            shared_proj = AttentionProjections(
                q_proj=nn.Linear(d_model, d_model),
                k_proj=nn.Linear(d_model, d_model),
                v_proj=nn.Linear(d_model, d_model),
                out_proj=nn.Linear(d_model, d_model),
            )

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    comparison_normalize=comparison_normalize,
                    rope=rope,
                    position_bias=position_bias,
                    shared_projections=shared_proj,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    comparison_normalize=comparison_normalize,
                    rope=rope,
                    position_bias=position_bias,
                    shared_projections=shared_proj,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def _positional_embedding(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.pos_emb is None:
            return None
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")
        return self.pos_emb(seq_len, device)

    def encode(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
        return_head_output: bool = False,
        layer_output_overrides: Optional[dict[int, torch.Tensor]] = None,
        attn_head_overrides: Optional[dict[int, dict]] = None,
    ) -> tuple[torch.Tensor, dict]:
        batch_size, src_len = src.size()
        pos = self._positional_embedding(src_len, src.device)
        x = self.src_token_emb(src)
        if pos is not None:
            x = x + pos.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.dropout(x)

        hidden_states = []
        attn_weights = []
        head_outputs = []

        for idx, layer in enumerate(self.encoder_layers):
            head_override = None
            head_mask = None
            if attn_head_overrides and idx in attn_head_overrides:
                override = attn_head_overrides[idx]
                head_override = override.get("head_output")
                head_mask = override.get("head_mask")

            x, extras = layer(
                x,
                key_padding_mask=src_key_padding_mask,
                return_attn=return_attn,
                return_head_output=return_head_output,
                head_override=head_override,
                head_override_mask=head_mask,
            )

            if layer_output_overrides and idx in layer_output_overrides:
                x = layer_output_overrides[idx]

            if return_hidden:
                hidden_states.append(x)
            if return_attn:
                attn_weights.append(extras["attn_weights"])
            if return_head_output:
                head_outputs.append(extras["head_output"])

        x = self.encoder_norm(x)
        extras = {}
        if return_hidden:
            extras["hidden_states"] = hidden_states
        if return_attn:
            extras["attn_weights"] = attn_weights
        if return_head_output:
            extras["head_outputs"] = head_outputs
        return x, extras

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        batch_size, tgt_len = tgt.size()
        pos = self._positional_embedding(tgt_len, tgt.device)
        y = self.tgt_token_emb(tgt)
        if pos is not None:
            y = y + pos.unsqueeze(0).expand(batch_size, -1, -1)
        y = self.dropout(y)

        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )

        self_attn = []
        cross_attn = []
        for layer in self.decoder_layers:
            y, extras = layer(
                y,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                return_attn=return_attn,
            )
            if return_attn:
                self_attn.append(extras["self_attn_weights"])
                cross_attn.append(extras["cross_attn_weights"])

        y = self.decoder_norm(y)
        logits = self.output_proj(y)

        extras = {}
        if return_attn:
            extras["self_attn_weights"] = self_attn
            extras["cross_attn_weights"] = cross_attn
        return logits, extras

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        return_hidden: bool = False,
        return_head_output: bool = False,
        layer_output_overrides: Optional[dict[int, torch.Tensor]] = None,
        attn_head_overrides: Optional[dict[int, dict]] = None,
    ) -> tuple[torch.Tensor, dict] | torch.Tensor:
        memory, enc_extras = self.encode(
            src,
            src_key_padding_mask=src_key_padding_mask,
            return_hidden=return_hidden,
            return_attn=return_attn,
            return_head_output=return_head_output,
            layer_output_overrides=layer_output_overrides,
            attn_head_overrides=attn_head_overrides,
        )
        logits, dec_extras = self.decode(
            tgt_in,
            memory,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            return_attn=return_attn,
        )

        if not (return_attn or return_hidden or return_head_output):
            return logits
        extras = {"encoder": enc_extras, "decoder": dec_extras}
        return logits, extras

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: Optional[torch.Tensor],
        max_len: int,
        bos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        memory, _ = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), bos_token_id, device=src.device, dtype=torch.long)
        for _ in range(max_len):
            tgt_key_padding_mask = ys.eq(pad_token_id)
            logits, _ = self.decode(
                ys,
                memory,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys[:, 1:]
