# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from args import ModelArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(num, den):
    return (num % den) == 0


def pad_at_dim(t, pad: tuple[int, int], dim=-1, value=0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t, groups=1):
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')


class always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = einx.subtract('j, i -> i j', k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads=None, slopes: list[int] | None = None):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = torch.Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, 'h -> h 1 1')

        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def forward_custom_pos(self, pos_i: torch.Tensor, pos_j: torch.Tensor | None = None):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract('... j, ... i -> ... i j', pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, 'b i j -> b 1 i j')

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = torch.arange(j - i, j, device=device)
        context_arange = torch.arange(j, device=device)
        bias = -einx.subtract('j, i -> 1 i j', context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=-3)

        self.register_buffer('bias', bias, persistent=False)
        return self.bias


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(nn.Linear(1, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU()))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim) if norm else None, nn.SiLU()))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = einx.subtract('i, j -> i j', seq_arange, context_arange)
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).bfloat16()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


# designed for causal
class CoPE(nn.Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """

    def __init__(
            self,
            dim,
            heads,
            max_pos,
            soft_onehot=False,
            talking_heads=False,
            soft_onehot_temp=5e-2
    ):
        super().__init__()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer('positions', torch.arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.max_pos - 1)

        logits_int = einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract('i, j -> i j', pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim=-1)
            cope_pos_emb = einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.alibi_heads = self.n_local_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        assert self.alibi_heads <= self.n_local_heads, 'number of ALiBi heads must be less than the total number of heads'
        self.rel_pos = AlibiPositionalBias(heads=self.alibi_heads, total_heads=self.n_local_heads)
        # self.rel_pos = DynamicPositionBias(dim=self.dim // 4, heads=self.n_local_heads, log_distance=False, depth=2, norm=True)
        # self.rel_pos = RelativePositionBias(scale=self.head_dim ** 0.5, causal=False, heads=self.n_local_heads, num_buckets=32, max_distance=128)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.rel_pos(xq.shape[-2], xk.shape[-2]).to(scores)

        if mask is not None:
            scores = scores + mask.unsqueeze(1)  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output projection
        proj = self.wo(output)
        return proj


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        # self.feed_forward = FeedForward(
        #     dim=args.dim,
        #     hidden_dim=args.dim,
        #     multiple_of=args.multiple_of,
        #     ffn_dim_multiplier=args.ffn_dim_multiplier,
        # )
        self.gate = nn.Sequential(
            nn.Linear(args.dim, args.dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=1e-05)
        self.ffn_norm = RMSNorm(args.dim, eps=1e-05)
        self.gate_norm = RMSNorm(args.dim, eps=1e-05)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attention_out = self.attention(self.attention_norm(x), freqs_cis, mask=mask)
        gate_out = self.gate(self.gate_norm(attention_out))
        h = x + attention_out * gate_out
        out = h  # + self.feed_forward(self.ffn_norm(h))
        return out, gate_out.squeeze(-1)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.meta_tokens = params.meta_embeddings
        if self.meta_tokens:
            self.meta_token_embeddings = torch.nn.Parameter(torch.randn(params.meta_embeddings, params.dim))

        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / params.dim ** 0.5)
        # self.positional_embeddings = ScaledSinusoidalEmbedding(params.dim)
        # self.positional_embeddings = AbsolutePositionalEmbedding(params.dim, params.max_seq_len, l2norm_embed=False)
        self.positional_embeddings = always(0)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=1e-05)
        self.output = nn.Linear(params.dim, params.output_size if params.output_size else params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len * 2, 500000.0, True)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, labels=None, depths=None, global_attention_mask: Optional[torch.Tensor] = None):
        return self.hierarchical_forward(tokens, initial_labels=labels, depths=depths, global_attention_mask=global_attention_mask, is_train=False)

    def forward_train(self, tokens: torch.Tensor, labels: torch.Tensor, depths, global_attention_mask: Optional[torch.Tensor] = None):
        return self.hierarchical_forward(tokens, initial_labels=labels, depths=depths, global_attention_mask=global_attention_mask, is_train=True)

    def hierarchical_forward(self, tokens: torch.Tensor, initial_labels: torch.Tensor = None, depths: torch.Tensor = None, global_attention_mask: Optional[torch.Tensor] = None, is_train: bool = None):
        _bsz, seqlen = tokens.shape
        seqlen += self.meta_tokens

        ignore_labels = torch.full((_bsz, self.meta_tokens), -100, dtype=torch.int, device=initial_labels.device)
        labels = torch.cat([ignore_labels, initial_labels], dim=1)

        e = self.tok_embeddings(tokens)
        h = torch.cat([self.meta_token_embeddings.unsqueeze(0).expand(_bsz, -1, -1), e], dim=1) if self.meta_tokens else e
        h = h + self.positional_embeddings(h)

        self.freqs_cis = self.freqs_cis.clone().to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        # mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
        # mask = torch.triu(mask, diagonal=1)
        # mask = mask.type_as(h)

        mask, w = torch.ones(seqlen, seqlen, device=tokens.device), 31 # needs to be odd
        mask = torch.triu(mask, diagonal=-w // 2) - torch.triu(mask, diagonal=w // 2 + 1)
        mask = torch.tril(mask)
        mask[mask == 0] = float("-inf")

        if global_attention_mask is not None:
            meta_mask = torch.ones(_bsz, self.meta_tokens, dtype=torch.bool, device=global_attention_mask.device)
            mask = merge_masks(mask, torch.cat([meta_mask, global_attention_mask], dim=1), labels)
        else:
            mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)

        current_layer_indexes = torch.zeros(_bsz, device=tokens.device).type(dtype=torch.long)
        invocations = torch.zeros(_bsz, device=tokens.device).type(dtype=torch.long)
        history = []
        all_next_layer_probs = [[] for _ in range(tokens.shape[0])]

        while True:
            history.append(current_layer_indexes.clone().detach())
            active_sequences_mask = (current_layer_indexes != self.params.n_layers)

            if not active_sequences_mask.any():
                break

            unique_layer_indices = torch.unique(current_layer_indexes[active_sequences_mask])

            for layer_index in unique_layer_indices:
                i = layer_index.item()
                this_batch = (current_layer_indexes == i) & active_sequences_mask
                batch_h = h[this_batch]
                batch_mask = mask[this_batch]
                layer = self.layers[i]
                h[this_batch], next_layer_probs = layer(batch_h, freqs_cis, mask=batch_mask)

                # last_layer_prob = 0.8
                # step = 0.2
                # layer_threshold = last_layer_prob - (self.params.n_layers * step) + step * (i + 1)
                layer_threshold = 0.7
                training_stability = -0.05 * invocations
                layer_threshold = training_stability + layer_threshold

                if labels is None or not is_train:
                    layer_probs = next_layer_probs[:, -1]
                    increment_decrement = torch.where(layer_probs > layer_threshold[this_batch], 1, -1)
                else:
                    layer_probs = torch.where(labels[this_batch] != -100, next_layer_probs, 2)
                    increment_decrement = torch.all(layer_probs > layer_threshold[this_batch].unsqueeze(1), dim=1).long() * 2 - 1

                for i, val in enumerate(this_batch):
                    if val:
                        probs = layer_probs[0].unsqueeze(0)
                        all_next_layer_probs[i].append(probs[probs <= 1.1])
                        layer_probs = layer_probs[1:]

                if depths is not None:
                    pos = invocations[this_batch]
                    depth = depths[this_batch]
                    increment_decrement = torch.tensor([is_forward_at_position_for_depth(p.item(), d.item()) for p, d in zip(pos, depth)], device=tokens.device)

                current_layer_indexes[this_batch] += increment_decrement
                current_layer_indexes = current_layer_indexes.clamp(min=0)

                invocations[this_batch] += 1

        h = self.norm(h)
        output = self.output(h).float()
        return output, labels, all_next_layer_probs, history


def merge_masks(local_mask: torch.Tensor, global_attention_mask: torch.Tensor, labels: torch.Tensor):
    batch_size, seq_len = global_attention_mask.size()
    expanded_local_mask = local_mask.unsqueeze(0).expand(batch_size, -1, -1)

    global_mask = global_attention_mask[:, :, None] + global_attention_mask[:, None, :]

    first_label_indices = (labels != -100).long().argmax(dim=1)
    pad_mask = torch.arange(seq_len, device=labels.device)[None, :] > first_label_indices[:, None]

    global_mask[pad_mask.unsqueeze(1).expand(-1, seq_len, -1)] = 0

    combined_mask = expanded_local_mask.clone()
    combined_mask[global_mask > 0.1] = 1
    return combined_mask


def is_forward_at_position_for_depth(pos, depth):
    seq = [1, 1, 1] + [-1, -1, 1, 1] * max(0, depth - 1) + [1]
    return seq[min(pos, len(seq) - 1)]
