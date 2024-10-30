import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import math

from model.llama import logger, LlamaRotaryEmbedding, \
    LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, \
    apply_rotary_pos_emb, repeat_kv
from model.module.linear_super import Linear_Super


class LlamaAttention_Super(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: LlamaConfig,
            attention_dropout,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = Linear_Super(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,
            num_head=self.num_heads, head_dim=self.head_dim
        )
        self.k_proj = Linear_Super(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias,
            num_head=self.num_key_value_heads, head_dim=self.head_dim
        )
        self.v_proj = Linear_Super(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias,
            num_head=self.num_key_value_heads, head_dim=self.head_dim
        )
        self.o_proj = Linear_Super(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        # self._init_rope()
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # sample_
        self.sample_hidden_size = None
        self.sample_num_heads = None
        self.sample_head_dim = None
        self.sample_rotary_emb = None
        self.sample_weight_start = None

    def set_sample_config(self, sample_hidden_size, sample_num_attention_heads, sample_weight_start):
        self.sample_hidden_size = sample_hidden_size
        self.sample_num_heads = sample_num_attention_heads
        self.sample_head_dim = self.sample_hidden_size // self.sample_num_heads
        self.sample_weight_start = sample_weight_start

        assert self.sample_hidden_size % self.sample_num_heads == 0

        self.q_proj.set_sample_config(
            self.hidden_size, self.sample_num_heads * self.sample_head_dim, sample_weight_start,
            qkv=True, sample_head_num=self.sample_num_heads, sample_head_dim=self.sample_head_dim
        )
        self.k_proj.set_sample_config(
            self.hidden_size, self.sample_num_heads * self.sample_head_dim, sample_weight_start,
            qkv=True, sample_head_num=self.sample_num_heads, sample_head_dim=self.sample_head_dim
        )
        self.v_proj.set_sample_config(
            self.hidden_size, self.sample_num_heads * self.sample_head_dim, sample_weight_start,
            qkv=True, sample_head_num=self.sample_num_heads, sample_head_dim=self.sample_head_dim
        )
        self.o_proj.set_sample_config(self.sample_hidden_size, self.hidden_size, sample_weight_start)

        self.num_key_value_heads = self.sample_num_heads
        self.num_key_value_groups = self.sample_num_heads // self.num_key_value_heads

        self.sample_rotary_emb = LlamaRotaryEmbedding(
            self.sample_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len,
            self.sample_num_heads if self.sample_num_heads else self.num_heads,
            self.sample_head_dim if self.sample_head_dim else self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads,
            self.sample_head_dim if self.sample_head_dim else self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads,
            self.sample_head_dim if self.sample_head_dim else self.head_dim
        ).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if self.sample_rotary_emb is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = self.sample_rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) \
                       / math.sqrt(self.sample_head_dim if self.sample_head_dim else self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        del attention_mask
        del query_states
        del key_states

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
                bsz,
                self.sample_num_heads if self.sample_num_heads else self.num_heads,
                q_len,
                self.sample_head_dim if self.sample_head_dim else self.head_dim
        ):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.sample_num_heads if self.sample_num_heads else self.num_heads, q_len, self.sample_head_dim if self.sample_head_dim else self.head_dim)}"
                f", but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(
            bsz, q_len,
            self.sample_hidden_size if self.sample_hidden_size else self.hidden_size
        )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value