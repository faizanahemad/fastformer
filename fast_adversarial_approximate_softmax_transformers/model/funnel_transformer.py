# coding=utf-8
# Copyright 2020-present Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Funnel Transformer model. """

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from performer_pytorch import SelfAttention, FastAttention

from transformers.activations import ACT2FN
# from transformers.configuration_funnel import FunnelConfig
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FunnelConfig"
_TOKENIZER_FOR_DOC = "FunnelTokenizer"

FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

INF = 1e6
EPS = 1e-6

FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/config.json",
    "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json",
    "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/config.json",
    "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/config.json",
    "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json",
    "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/config.json",
    "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/config.json",
    "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/config.json",
    "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json",
    "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/config.json",
}


from transformers import PretrainedConfig


class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.FunnelModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a Funnel Transformer model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Funnel Transformer `funnel-transformer/small
    <https://huggingface.co/funnel-transformer/small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the Funnel transformer. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~transformers.FunnelModel` or
            :class:`~transformers.TFFunnelModel`.
        block_sizes (:obj:`List[int]`, `optional`, defaults to :obj:`[4, 4, 4]`):
            The sizes of the blocks used in the model.
        block_repeats (:obj:`List[int]`, `optional`):
            If passed along, each layer of each block is repeated the number of times indicated.
        num_decoder_layers (:obj:`int`, `optional`, defaults to 2):
            The number of layers in the decoder (when not using the base model).
        d_model (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the model's hidden states.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (:obj:`int`, `optional`, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (:obj:`int`, `optional`, defaults to 3072):
            Inner dimension in the feed-forward blocks.
        hidden_act (:obj:`str` or :obj:`callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward blocks.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 3):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.FunnelModel` or
            :class:`~transformers.TFFunnelModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.1):
            The standard deviation of the `uniform initializer` for initializing all weight matrices in attention
            layers.
        initializer_std (:obj:`float`, `optional`):
            The standard deviation of the `normal initializer` for initializing the embedding matrix and the weight of
            linear layers. Will default to 1 for the embedding matrix and the value given by Xavier initialization for
            linear layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-9):
            The epsilon used by the layer normalization layers.
        pooling_type (:obj:`str`, `optional`, defaults to :obj:`"mean"`):
            Possible values are ``"mean"`` or ``"max"``. The way pooling is performed at the beginning of each block.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"relative_shift"`):
            Possible values are ``"relative_shift"`` or ``"factorized"``. The former is faster on CPU/GPU while the
            latter is faster on TPU.
    """
    model_type = "funnel"

    def __init__(
        self,
        vocab_size=30522,
        block_sizes=[6, 6, 6],
        block_repeats=None,
        num_decoder_layers=2,
        d_model=768,
        n_head=12, # 8
        d_head=64, # 32
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.1,
        initializer_std=None,
        layer_norm_eps=1e-9,
        pooling_type="mean",
        attention_type="relative_shift",
        ffn_groups=4,
        ffn_layers=1,
        ffn_width=4,
        qkv_transform_groups=4,
        embedding_size=768,
        num_highway_cls_tokens=7,
        position_biased_input=True,
        stride=2,
        untie_cls=False,
        separate_content_and_position_attention=False,
        approximate_attention=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.block_repeats = [1] * len(block_sizes) if block_repeats is None else block_repeats
        assert len(block_sizes) == len(
            self.block_repeats
        ), "`block_sizes` and `block_repeats` should have the same length."
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = ffn_width * d_model
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        assert pooling_type in [
            "mean",
            "max",
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
        self.pooling_type = pooling_type
        assert attention_type in [
            "relative_shift",
            "factorized",
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
        self.attention_type = attention_type
        self.ffn_groups = ffn_groups
        self.ffn_layers = ffn_layers
        self.qkv_transform_groups = qkv_transform_groups
        self.embedding_size = embedding_size
        self.position_biased_input = position_biased_input
        self.num_highway_cls_tokens = num_highway_cls_tokens
        self.approximate_attention = approximate_attention
        self.untie_cls = untie_cls
        self.separate_content_and_position_attention = separate_content_and_position_attention
        self.stride = stride
        assert position_biased_input or separate_content_and_position_attention


    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:

        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:

        drop_prob (float): the dropout probabilities

    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout


        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class FunnelEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        hidden_size = getattr(config, "d_model", config.hidden_size)
        self.hidden_size = hidden_size
        self.embedding_size = getattr(config, "embedding_size", )
        self.word_embeddings = nn.Embedding(config.vocab_size + config.num_highway_cls_tokens, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        self.embed_proj = None
        if self.embedding_size != hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, hidden_size, bias=False)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + config.num_highway_cls_tokens).expand((1, -1)))
        self.register_buffer("highway_position_ids", torch.arange(config.num_highway_cls_tokens).expand((1, -1)))
        if config.num_highway_cls_tokens > 0:
            self.register_buffer("highway_cls_tokens", torch.arange(config.vocab_size, config.vocab_size + config.num_highway_cls_tokens).expand((1, -1)))

    def forward(self, input_ids=None, input_embeds=None, token_type_ids=None, position_ids=None, mask=None):
        if input_embeds is None:
            input_shape = input_ids.size()
            input_shape = list(input_shape)
            input_shape[1] = input_shape[1] + self.config.num_highway_cls_tokens
            input_shape = tuple(input_shape)

            seq_length = input_shape[1]
            inputs_embeds = self.word_embeddings(input_ids)

            if self.config.num_highway_cls_tokens > 0:
                highway_embeddings = self.word_embeddings(self.highway_cls_tokens).expand((inputs_embeds.size(0), -1, -1))
                inputs_embeds = torch.cat((highway_embeddings, inputs_embeds), dim=1)
        else:
            input_shape = input_embeds.size()
            seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.cat((self.highway_position_ids.expand((position_ids.size(0), -1)), position_ids + self.config.num_highway_cls_tokens), dim=1)

        position_embeddings = self.position_embeddings(position_ids.long())

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            else:
                token_type_ids = torch.cat((torch.empty(input_shape[0], self.config.num_highway_cls_tokens, device=token_type_ids.device).fill_(token_type_ids[0][0]), token_type_ids), dim=1)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embed_proj:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = torch.cat((torch.ones(mask.size(0), self.config.num_highway_cls_tokens, dtype=mask.dtype, device=mask.device),mask), dim=1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings, position_embeddings.squeeze(0)


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    def __init__(self, config: FunnelConfig):
        super().__init__()
        self.config = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # dividide.
        self.cls_tokens_total = self.config.num_highway_cls_tokens + 1
        self.stride = self.config.stride

    def init_attention_inputs(self, inputs_embeds, position_embeds, attention_mask=None):
        """ Returns the attention inputs associated to the inputs of the model. """
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.seq_len = seq_len = inputs_embeds.size(1)
        position_embeds = self.get_position_embeds(seq_len, position_embeds, inputs_embeds.dtype, inputs_embeds.device)
        return (position_embeds, attention_mask)

    def get_position_embeds(self, seq_len, pos_embed, dtype, device):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shif attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config.d_model
        stride = self.stride
        # Notations from the paper, appending A.2.1, final formula.
        # We need to create and return all the possible vectors R for all blocks and shifts.

        pos = torch.arange(0, seq_len, dtype=dtype, device=device)
        pooled_pos = pos
        position_embeds_list = []
        for block_index in range(0, self.config.num_blocks):
            # For each block with block_index > 0, we need two types position embeddings:
            #   - Attention(pooled-q, unpooled-kv)
            #   - Attention(pooled-q, pooled-kv)
            # For block_index = 0 we only need the second one and leave the first one as None.

            # First type
            if block_index == 0:
                position_embeds_pooling = pos_embed
                position_embeds_no_pooling = pos_embed
            else:
                pooled_pos = self.stride_pool_pos(pos, block_index, stride)
                pooled_pos = pooled_pos[:2 * (pooled_pos.size(0)//2)]
                ppos = pooled_pos[:, None].expand(pooled_pos.size(0), pos_embed.size(1)).type(torch.long)
                position_embeds_pooling = torch.gather(pos_embed, 0, ppos)
                if block_index == 1:
                    position_embeds_no_pooling = pos_embed
                else:
                    apos = pos[:, None].expand(pos.size(0), pos_embed.size(1)).type(torch.long)
                    position_embeds_no_pooling = torch.gather(pos_embed, 0, apos)
                pos = pooled_pos

            position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
        return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index, stride):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """

        # Under separate <cls>, we treat the <cls> as the first token in
        # the previous block of the 1st real block. Since the 1st real
        # block always has position 1, the position of the previous block
        # will be at `1 - 2 ** block_index`.
        cls_pos = pos_id[:self.cls_tokens_total]
        pooled_pos_id = pos_id[self.cls_tokens_total:-1]
        return torch.cat([cls_pos, pooled_pos_id[::stride]], 0)

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)


        # TODO: check if even length in dim=1 (seq dim)
        cls_tokens = tensor[:, :self.cls_tokens_total]
        tensor = tensor[:, self.cls_tokens_total:]

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = F.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = F.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -F.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            tensor = tensor[:, 0, :, 0]
        elif ndim == 3:
            tensor = tensor[:, 0]

        tensor = torch.cat([cls_tokens, tensor.squeeze(1)], dim=1)
        tensor = tensor[:, :2 * (tensor.size(1) // 2)]
        return tensor

    def pre_attention_pooling(self, output, attention_inputs):
        """ Pool `output` and the proper parts of `attention_inputs` before the attention layer. """
        position_embeds, attention_mask = attention_inputs
        output = self.pool_tensor(output, mode=self.config.pooling_type, stride=self.stride)
        attention_inputs = (position_embeds, attention_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs, block_index):
        """ Pool the proper parts of `attention_inputs` after the attention layer. """
        position_embeds, attention_mask = attention_inputs
        attention_mask = self.pool_tensor(attention_mask, mode="min", stride=self.stride)
        position_embeds[block_index][0] = position_embeds[block_index][1]
        attention_inputs = (position_embeds, attention_mask)
        return attention_inputs


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, bias=bias)

    def forward(self, x):
        unsqueeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            unsqueeze = True
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        if unsqueeze:
            x = x.squeeze(0)
        return x


class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config: FunnelConfig, block_index):
        super().__init__()
        self.config = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head
        assert d_model % d_head == 0
        assert d_model % n_head == 0
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.untie_cls = self.config.untie_cls
        self.separate_content_and_position_attention = self.config.separate_content_and_position_attention
        self.approximate_attention = self.config.approximate_attention

        qkv_transform_groups = self.config.qkv_transform_groups
        if qkv_transform_groups > 1:
            assert n_head % qkv_transform_groups == 0 and n_head > qkv_transform_groups
            self.q_head = Conv1d(in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)
            self.k_head = Conv1d(in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
            self.v_head = Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, groups=qkv_transform_groups)
        else:
            self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
            self.k_head = nn.Linear(d_model, n_head * d_head)
            self.v_head = nn.Linear(d_model, d_model)

        if self.separate_content_and_position_attention:
            if qkv_transform_groups > 1:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
                self.pos_k_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
            else:
                self.pos_q_head = nn.Linear(config.embedding_size, n_head * d_head)
                self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head)
            self.c2p_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.p2c_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.pos_qln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)
            self.pos_kln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.scale = 1.0 / (d_head ** 0.5)
        self.cls_tokens_total = self.config.num_highway_cls_tokens + 1

    def forward(self, query, key, value, attention_inputs, output_attentions=False):
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, attention_mask = attention_inputs
        position_embeds = position_embeds[self.block_index]
        position_embed_of_key, position_embed_of_query = position_embeds
        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config.n_head, self.config.d_head
        d_model = self.config.d_model

        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, d_model // n_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        if self.separate_content_and_position_attention:
            v = self.c2p_bias * self.scale
            w = self.p2c_bias * self.scale
            pos_k_head = self.pos_kln(self.pos_k_head(position_embed_of_key)).view(context_len, n_head, d_head)
            pos_q_head = self.pos_qln(self.pos_q_head(position_embed_of_query)).view(seq_len, n_head, d_head)
            # print(query.size(), key.size(), position_embed_of_query.size(), position_embed_of_key.size())

            if self.untie_cls:
                pos_k_head = pos_k_head*F.pad(pos_k_head.new_ones([pos_k_head.size(0) - self.cls_tokens_total, pos_k_head.size(1), pos_k_head.size(2)]),
                                              (self.cls_tokens_total, 0, 0, 0, 0, 0))
                pos_q_head = pos_q_head * F.pad(pos_q_head.new_ones([pos_q_head.size(0) - self.cls_tokens_total, pos_q_head.size(1), pos_q_head.size(2)]),
                                                (self.cls_tokens_total, 0, 0, 0, 0, 0))

        if self.approximate_attention:
            if self.separate_content_and_position_attention:
                pass
            else:
                pass
        else:
            content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
            attn_score = content_score

            if self.separate_content_and_position_attention:
                c2p_score = torch.einsum("bind,jnd->bnij", q_head + v, pos_k_head)
                p2c_score = torch.einsum("ind,bjnd->bnij", pos_q_head, k_head + w)
                p2p_score = torch.einsum("ind,jnd->nij", pos_q_head, pos_k_head).expand(batch_size, n_head, seq_len, context_len)
                # merge attention scores
                attn_score = attn_score + c2p_score + p2c_score + p2p_score

            # precision safe in case of mixed precision training
            dtype = attn_score.dtype
            attn_score = attn_score.float()
            # perform masking
            if attention_mask is not None:
                attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
            # attention probability
            attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
            attn_prob = self.attention_dropout(attn_prob)

            # attention output, shape batch_size x seq_len x n_head x d_head
            attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)
            attn_out = attn_vec.reshape(batch_size, seq_len, d_model)
            # Shape shape batch_size x seq_len x d_model

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class ConvFFN(nn.Module):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, config):
        super().__init__()
        groups, layers = config.ffn_groups, config.ffn_layers
        d_model, d_inner = config.d_model, config.d_inner
        cin, cout = d_model, d_model
        act = config.hidden_act
        self.conv1d_in = nn.Conv1d(in_channels=cin, out_channels=d_inner, kernel_size=1, groups=groups)
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList() if layers > 0 else None
        for _ in range(layers):
            self.layers.append(nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=1, groups=groups))
        self.conv1d_out = nn.Conv1d(in_channels=d_inner, out_channels=cout, kernel_size=1, groups=groups)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.act = ACT2FN[act]

    def forward(self, x):
        h = x.permute(0, 2, 1)
        output = self.conv1d_in(h)
        output = self.act(output)
        output = self.activation_dropout(output)
        if self.layers:
            for ll in self.layers:
                output = ll(output)
                output = self.act(output)
                output = self.activation_dropout(output)
        output = self.conv1d_out(output)
        output = self.dropout(output)
        output = self.act(output)
        output = output.permute(0, 2, 1)
        return output


class BertFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.linear_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return h


class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        groups, layers = config.ffn_groups, config.ffn_layers
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        if groups > 1:
            assert config.d_model % groups == 0
            self.lin = nn.Linear(config.d_model, config.d_model)
            self.ffn = ConvFFN(config)
        else:
            self.lin = None
            self.ffn = BertFFN(config)

    def forward(self, hidden):
        if self.lin:
            h = self.activation_dropout(self.activation_function(self.lin(hidden)))
            h = self.ffn(h)
        else:
            h = self.ffn(hidden)
        return self.layer_norm(hidden + h)


class FunnelLayer(nn.Module):
    def __init__(self, config, block_index):
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(self, query, key, value, attention_inputs, output_attentions=False):
        attn = self.attention(query, key, value, attention_inputs, output_attentions=output_attentions)
        output = self.ffn(attn[0])
        return (output, attn[1]) if output_attentions else (output,)


class FunnelEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        position_embeds,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > 2
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)

                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs, block_index)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


def upsample(x, stride, target_len):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x

    cls = x[:, :1]
    x = x[:, 1:]
    output = torch.repeat_interleave(x, repeats=stride, dim=1)

    output = nn.functional.pad(output, (0, 0, 0, stride - 1, 0, 0))
    output = output[:, : target_len - 1]
    output = torch.cat([cls, output], dim=1)

    return output


class FunnelDecoder(nn.Module):
    def __init__(self, config: FunnelConfig):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.ModuleList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(
        self,
        final_hidden,
        first_block_hidden,
        position_embeds,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        upsampled_hidden = upsample(
            final_hidden,
            stride=self.config.stride ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
        )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            position_embeds,
            attention_mask=attention_mask,
        )

        for layer in self.layers:
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dense_prediction = nn.Linear(config.d_model, 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FunnelConfig
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                nn.init.normal_(module.weight, std=std)

            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)

        if classname.find("Conv1d") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape[:2]
                    fan_out, fan_in = fan_out, fan_in
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                nn.init.normal_(module.weight, std=std)

            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            if hasattr(module, "c2p_bias"):
                nn.init.uniform_(module.c2p_bias, b=self.config.initializer_range)
            if hasattr(module, "p2c_bias"):
                nn.init.uniform_(module.p2c_bias, b=self.config.initializer_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            nn.init.normal_(module.word_embeddings.weight, std=std)
            nn.init.normal_(module.position_embeddings.weight, std=std)
            if hasattr(module, "token_type_embeddings"):
                nn.init.normal_(module.token_type_embeddings.weight, std=std)



class FunnelClassificationHead(nn.Module):
    def __init__(self, config, n_labels):
        super().__init__()
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden):
        hidden = self.linear_hidden(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.linear_out(hidden)


@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FunnelForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA-style objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in `Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing <https://arxiv.org/abs/2006.03236>`__ by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.FunnelConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

FUNNEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    """
    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small-base",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelDecoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if self.config.num_highway_cls_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_shape[0], self.config.num_highway_cls_tokens, device=device), attention_mask], dim=1)

        inputs_embeds, position_embeds = self.embeddings(input_ids, inputs_embeds, token_type_ids)



        encoder_outputs = self.encoder(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.config.block_sizes[0]],
            position_embeds=position_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


add_start_docstrings(
    """
    Funnel Transformer model with a binary classification head on top as used during pretraining for identifying
    generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        self.init_weights()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples::

            >>> from transformers import FunnelTokenizer, FunnelForPreTraining
            >>> import torch

            >>> tokenizer = FunnelTokenizer.from_pretrained('funnel-transformer/small')
            >>> model = FunnelForPreTraining.from_pretrained('funnel-transformer/small', return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors= "pt")
            >>> logits = model(**inputs).logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


@add_start_docstrings("""Funnel Transformer Model with a `language modeling` head on top. """, FUNNEL_START_DOCSTRING)
class FunnelForMaskedLM(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="funnel-transformer/small",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":

    from transformers import AutoTokenizer
    model = FunnelForMaskedLM(FunnelConfig(separate_content_and_position_attention=True, stride=4))
    tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/intermediate-base")
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Params = %s" % (params/1_000_000))

    model = model.eval()

    def run():
        pt_batch = tokenizer(["We are running a BERT_large model in production on a Sagemaker instance, p2.xlarge, and running into P50 inference times of 9 seconds! While we’re switching to the g4xdn family of instances, there’s also an Inf1 instance to which we could switch. I’m also getting advice on switching to the compute optimized (c-family) or memory optimized (r6, r5, x1) instances.",
                              "Are you measuring time with model load time as well? If you measure model load time too then it will appear slow but once model is loaded subsequent.",
                              "Are you running the model on eval mode and with no gradient backprop? If you run model in training mode then it will be slower."
                              ],
                             padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            pt_outputs = model(**pt_batch)
        return [o.cpu() for o in pt_outputs]

    _ = run()
    _ = run()
    import time
    times = []
    for _ in range(10):
        st = time.time()
        _ = run()
        et = time.time() - st
        times.append(et)
    import numpy as np
    print(np.mean(times), np.std(times))

