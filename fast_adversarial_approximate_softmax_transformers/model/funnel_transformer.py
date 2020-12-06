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
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from performer_pytorch import SelfAttention, FastAttention
from collections import defaultdict

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

try:
    from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
except:
    pass


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

# TODO: check if all heads are aligned
# TODO: check if repeats are happening properly for layers
# TODO: check if upsampling (interleaving) is happening properly
# TODO: Fix parameter initialization

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
        block_channel_size=[576, 768, 960], # [512, 768, 1024]
        block_repeats=True,
        separate_compressiion_layer=False,
        num_decoder_layers=2,
        n_head=[(8,), (12,), (12,)], # 8
        use_cuda_conv=True,
        d_head=[72, 64, 80],  # 32
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.1,
        initializer_std=None,
        layer_norm_eps=1e-9,
        pooling_type="mean",  # learn, #learn_sdconv
        pooling_kernel_size=5,
        stride=2,
        attention_type="relative_shift",
        ffn_groups=4,
        ffn_layers=0,
        ffn_width=4,
        qkv_transform_groups=4,
        embedding_size=128,
        num_highway_cls_tokens=7,
        position_biased_input=True,
        untie_cls=False,
        separate_content_and_position_attention=False,
        approximate_attention=[False, False, False],
        sequence_dependent_position_transform=False,
        qkv_squeeze_fraction=1,
        light_first_layer=False,
        light_last_layer=False,
        compress_query_method="learn",
        compressed_query_attention_kernel_size=3,
        compressed_query_attention_stride=2,
        compressed_query_attention_layers=[],
        compressed_key_attention_layers=[],
        sdconv=[False, False, False],
        sdconv_kernel_size=[5, 7, 9],
        full_channel_separation=[False, False, False],
        short_rnn=[False, False, False],
        short_rnn_kernel=[128, 128, 128],
        short_rnn_overlap=[16, 16, 16],
        conv_layer_use_dynamic_conv=False,
        no_v_head=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        try:
            from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
        except:
            use_cuda_conv = False
        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.block_repeats = block_repeats
        self.separate_compressiion_layer = separate_compressiion_layer
        self.num_decoder_layers = num_decoder_layers
        self.n_head = [(n_head,)] * len(block_sizes) if isinstance(n_head, int) else n_head
        self.n_head = [h if isinstance(h, (list, tuple)) else (h,) for h in self.n_head]
        assert all([d % sum(h) == 0 for h, d in zip(self.n_head, block_channel_size)])
        assert len(self.n_head) == len(block_sizes)
        self.d_head = [d_head] * len(block_sizes) if isinstance(d_head, int) else d_head
        assert len(self.d_head) == len(block_sizes)
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        self.ffn_width = ffn_width
        self.pooling_kernel_size = pooling_kernel_size
        assert pooling_type in [

            "mean",
            "max",
            "learn",
            "learn_sdconv",
                        'learn_rnn',
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
        assert compress_query_method in [
            "mean",
            "learn",
            "learn_sdconv",
            'learn_rnn',
        ], f"Got {pooling_type} for `compress_query_method`"
        self.pooling_type = pooling_type
        self.compress_query_method = compress_query_method
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
        self.qkv_squeeze_fraction = qkv_squeeze_fraction
        self.approximate_attention = approximate_attention
        self.light_first_layer = light_first_layer
        self.light_last_layer = light_last_layer
        assert compressed_query_attention_kernel_size in [3, 5, 7, 9]
        self.compressed_query_attention_kernel_size = compressed_query_attention_kernel_size
        assert compressed_query_attention_stride in [2, 4]
        self.compressed_query_attention_stride = compressed_query_attention_stride
        self.compressed_query_attention_layers = compressed_query_attention_layers
        self.compressed_key_attention_layers = compressed_key_attention_layers
        self.untie_cls = untie_cls
        self.separate_content_and_position_attention = separate_content_and_position_attention
        self.sequence_dependent_position_transform = sequence_dependent_position_transform
        assert (sequence_dependent_position_transform and separate_content_and_position_attention) or (not sequence_dependent_position_transform and not separate_content_and_position_attention)
        assert separate_content_and_position_attention or position_biased_input
        self.stride = stride
        assert len(block_channel_size) == len(block_sizes)
        self.block_channel_size = block_channel_size
        self.short_rnn = [short_rnn] * len(block_sizes) if isinstance(short_rnn, bool) else short_rnn
        self.short_rnn_kernel = [short_rnn_kernel] * len(block_sizes) if isinstance(short_rnn_kernel, int) else short_rnn_kernel
        self.short_rnn_overlap = [short_rnn_overlap] * len(block_sizes) if isinstance(short_rnn_overlap, int) else short_rnn_overlap

        self.sdconv = [sdconv] * len(block_sizes) if isinstance(sdconv, bool) else sdconv
        self.full_channel_separation = [full_channel_separation] * len(block_sizes) if isinstance(full_channel_separation, bool) else full_channel_separation
        self.use_cuda_conv = use_cuda_conv
        self.conv_layer_use_dynamic_conv = conv_layer_use_dynamic_conv
        self.sdconv_kernel_size = [sdconv_kernel_size] * len(block_sizes) if isinstance(sdconv_kernel_size, int) else sdconv_kernel_size
        self.no_v_head = no_v_head
        assert position_biased_input or separate_content_and_position_attention
        assert not (separate_content_and_position_attention and any(approximate_attention))
        assert (sequence_dependent_position_transform and separate_content_and_position_attention) or not sequence_dependent_position_transform
        assert (any(approximate_attention) and position_biased_input) or not any(approximate_attention)
        assert len(approximate_attention) == len(block_sizes)  # + 1 for decoder
        if light_first_layer or any(self.short_rnn) or any(self.sdconv) or light_last_layer:
            assert position_biased_input

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)


vanilla_bert_base = FunnelConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                 ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                 untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False]*1, block_repeats=False)
vanilla_funnel_base = FunnelConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                   ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                   untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False]*3, )
repeated_funnel_base = FunnelConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                    ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                    untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False]*3,
                                    block_repeats=True, separate_compressiion_layer=True,)
repeated_funnel_channel_expanded_base = FunnelConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[480, 768, 960],
                                                     num_decoder_layers=2, n_head=[8, 12, 12], d_head=[48, 64, 80],
                                                     ffn_groups=4, qkv_transform_groups=4, embedding_size=128, num_highway_cls_tokens=0,
                                                     untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False]*3,
                                                     block_repeats=True, separate_compressiion_layer=False, )
vanilla_albert_base = FunnelConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                   ffn_groups=1, qkv_transform_groups=1, embedding_size=128, num_highway_cls_tokens=0,
                                   untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False]*1,
                                   block_repeats=True)



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

    def __init__(self, config: FunnelConfig):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        hidden_size = config.block_channel_size[0]
        self.hidden_size = hidden_size
        self.embedding_size = config.embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size + config.num_highway_cls_tokens, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + config.num_highway_cls_tokens, self.embedding_size)

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


def pool_tensor(tensor, cls_size, mode="mean", stride=2):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None

    # Do the pool recursively if tensor is a list or tuple of tensors.
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(pool_tensor(x, mode=mode, stride=stride) for x in tensor)


    # TODO: check if even length in dim=1 (seq dim)
    cls_tokens = tensor[:, :cls_size]
    tensor = tensor[:, cls_size:]

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


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    def __init__(self, config: FunnelConfig):
        super().__init__()
        self.config = config
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
        pooled_pos_id = pos_id[self.cls_tokens_total:]
        return torch.cat([cls_pos, pooled_pos_id[::stride]], 0)

    def pool_tensor(self, tensor, mode="mean", stride=2):
        return pool_tensor(tensor, self.cls_tokens_total, mode, stride)

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
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=True, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, bias=bias, stride=stride)

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


class CompressSeqWeighted(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.stride = config.stride if use_in_funnel else config.compressed_query_attention_stride
        d_head = config.d_head[block_index]
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.ffn = BertFFN(config, self.stride * d_model // n_head,
                           2 * self.stride * d_model // n_head, 0, self.stride)

    def forward(self, query):

        cls = query[:, :self.cls_tokens]
        query = query[:, self.cls_tokens:]
        target_len = int(np.ceil(query.size(1) / self.stride))
        batch_size, seq_len, dim = query.shape
        assert seq_len % self.stride == 0
        assert dim % self.n_head == 0
        seq_groups = seq_len // self.stride

        q = query.view(batch_size,
                       seq_groups, self.stride,
                       self.n_head, dim // self.n_head)  # B, seq_groups, stride, H, h_dim
        qw = self.ffn(q.permute(0, 1, 3, 2, 4).reshape(batch_size, seq_groups, self.n_head, self.stride * dim // self.n_head))

        qw = qw.permute(0, 1, 3, 2)  # (B, seq_groups, H, stridexh_dim) -> (B, Seq_groups, H, stride) -> (B, Seq_groups, stride, H)
        # qw -> (B, Seq_groups, stride, H) q -> (B, Seq_groups, stride, H, h_dim)
        q = (qw.unsqueeze(-1) * q).mean(2).view(batch_size, seq_groups, dim)  # B, S/r, H, Dh -> B, S/r, D
        q = nn.functional.pad(q, (0, 0, 0, target_len - q.shape[1], 0, 0))
        q = torch.cat([cls, q], dim=1)
        q = q[:, :2 * (q.size(1) // 2)]
        return q


class SequenceDependentPositionTransform(nn.Module):
    def __init__(self, config: FunnelConfig, d_pos_in, d_model_in, d_out, qkv_transform_groups, compress):
        super().__init__()
        act = config.hidden_act
        self.act = ACT2FN[act]
        self.cls_transform = Conv1d(in_channels=d_model_in, out_channels=d_pos_in, kernel_size=1, groups=qkv_transform_groups) if qkv_transform_groups > 1 else nn.Linear(d_model_in, d_pos_in)
        self.d_pos_in = d_pos_in
        self.ffn = BertFFN(config, 2 * d_pos_in, 4 * d_pos_in, 0, d_out)
        self.compress = nn.AvgPool1d(4) if compress else None

    def forward(self, seq, position_embeds, stride=1):
        seq_len, _ = position_embeds.shape
        batch_size, _, _ = seq.shape
        cls_token = seq[:, 0:1, :]
        cls_token = self.cls_transform(cls_token).expand(batch_size, seq_len, self.d_pos_in)
        position_embeds = position_embeds.expand(batch_size, seq_len, self.d_pos_in)
        embeds = torch.cat([cls_token, position_embeds], dim=2)
        embeds = self.ffn(embeds)
        if stride > 1:
            embeds = pool_tensor(embeds, 0, "mean", stride)

        return embeds


class ShortSeqRNN(nn.Module):
    def __init__(self, config: FunnelConfig, hidden_size, heads, head_size, kernel_size, overlap):
        super().__init__()
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.heads = heads
        self.kernel_size = kernel_size
        self.all_head_size = heads * head_size
        self.hidden_size = hidden_size
        act = config.hidden_act
        self.act = ACT2FN[act]
        assert hidden_size % (2 * heads) == 0
        self.head_size = head_size
        self.overlap = overlap

        self.gru = nn.RNN(hidden_size // self.heads, hidden_size // (2 * self.heads), 2,
                          nonlinearity="tanh",
                          bias=False, batch_first=True, dropout=0.0, bidirectional=True)
        # TODO: should we try to also put a linear layer after rnn and make rnn hidden size larger?

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key

        bs, seqlen, dim = query.shape
        context_len = key.shape[1]
        assert self.hidden_size == dim

        upsampled = False
        if seqlen < context_len:
            upsampled = True
            query = value + upsample(query, self.config.stride, context_len, self.cls_tokens)
            seqlen = context_len

        num_segments = int(np.ceil(seqlen / self.kernel_size))
        target_len = num_segments * self.kernel_size
        query = nn.functional.pad(query, (0, 0, self.overlap, target_len + self.overlap - seqlen, 0, 0))
        segs = []
        segments = []
        for i in range(num_segments):
            seg_start = i * self.kernel_size
            seg_end = (i+1)*self.kernel_size + 2*self.overlap
            seg = query[:, seg_start:seg_end]
            segs.append(seg)
            segments.append((seg_start, seg_end, seg_end - 2*self.overlap, seg_end-seg_start, ))

        query = torch.cat(segs, 0)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1)
        query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])

        query = self.gru(query)[0]
        query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        query = query.reshape(bs, -1, dim)[:, self.overlap:seqlen+self.overlap]

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        return query


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None, channels_last=True):
        super().__init__()
        self.channels_last = channels_last
        if padding is None:
            padding = (kernel_size - 1) // 2
        if groups is None:
            groups = in_channels
        if pointwise_groups is None:
            pointwise_groups = 1
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=groups, bias=False, stride=stride, padding=padding)
        self.pointwise_groups = pointwise_groups
        if pointwise_groups == 1:
            self.pointwise = nn.Linear(in_channels, out_channels)
        else:
            self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, groups=pointwise_groups, bias=bias, stride=1, padding=0)

        self.out_channels = out_channels

    def channel_move(self, x):
        if self.channels_last:
            return x.permute(0, 2, 1)
        return x

    def forward(self, inputs):
        # Support both B, H, C, S and B, C, S
        in_shape = inputs.shape
        bs = in_shape[0]
        dims = in_shape[-1] if self.channels_last else in_shape[-2]
        seqlen = in_shape[-2] if self.channels_last else in_shape[-1]
        inputs = inputs.view(-1, in_shape[-2], in_shape[-1])
        inputs = self.channel_move(inputs)
        inputs = self.depthwise(inputs)
        if self.pointwise_groups == 1:
            inputs = self.pointwise(inputs.permute(0, 2, 1))
            if not self.channels_last:
                inputs = inputs.permute(0, 2, 1)
            inputs = inputs.view(bs, -1 if self.channels_last else self.out_channels, self.out_channels if self.channels_last else -1)
        else:
            inputs = self.pointwise(inputs)
            inputs = self.channel_move(inputs).view(bs, -1 if self.channels_last else self.out_channels, self.out_channels if self.channels_last else -1)
        return inputs


class SDConv(nn.Module):
    def __init__(self, config: FunnelConfig, hidden_size, heads, head_size, kernel_size=9, stride=1):
        super().__init__()
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.heads = heads
        self.kernel_size = kernel_size
        self.all_head_size = heads * head_size
        self.hidden_size = hidden_size
        self.stride = stride
        act = config.hidden_act
        self.act = ACT2FN[act]
        assert hidden_size % heads == 0
        self.head_size = head_size
        self.separable_conv1d = SeparableConv1d(hidden_size, hidden_size, kernel_size, pointwise_groups=heads, stride=stride, channels_last=True)
        # self.conv_attn_kernel = nn.Linear(self.all_head_size, self.heads * self.kernel_size)  # Multi-head?
        self.conv_attn_kernel = nn.Conv1d(hidden_size, self.heads * self.kernel_size, 1, groups=heads)
        if config.no_v_head:
            self.conv_attn_point = nn.Identity()
        else:
            self.conv_attn_point = Conv1d(hidden_size, hidden_size, 1, heads)
        self.use_cuda_conv = config.use_cuda_conv
        if not self.use_cuda_conv:
            self.unfold1d = nn.Unfold(kernel_size=[kernel_size, 1], padding=[(kernel_size - 1) // 2, 0], stride=[stride, 1])
        else:
            self.padding_l = (self.kernel_size - 1) // 2

    def forward(self, query, key=None, value=None):
        # return query[:, :, :self.all_head_size]
        assert key is None or self.stride == 1
        assert value is None or self.stride == 1
        if key is None:
            key = query
        if value is None:
            value = key

        bs, seqlen, dim = query.shape
        context_len = key.shape[1]
        assert self.hidden_size == dim

        upsampled = False
        if seqlen < context_len:
            upsampled = True
            query = upsample(query, self.config.stride, context_len, self.cls_tokens)
            seqlen = context_len

        key_conv_attn_layer = self.separable_conv1d(key)
        if self.stride == 1:
            conv_attn_layer = key_conv_attn_layer * query
        else:
            conv_attn_layer = self.act(key_conv_attn_layer)
        conv_kernel_layer = self.conv_attn_kernel(conv_attn_layer.permute(0, 2, 1)).permute(0, 2, 1)  # Softmax only in kernel dim

        if not self.use_cuda_conv or self.stride != 1:
            conv_kernel_layer = conv_kernel_layer.reshape(-1, self.kernel_size, 1)  # BxSxH, k, 1
            conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

            # conv_out_layer
            conv_out_layer = self.conv_attn_point(value).transpose(1, 2).contiguous().unsqueeze(-1)  # B,D,Seq, 1
            unfold_conv_out_layer = self.unfold1d(conv_out_layer)  # B, D*kernel_size, seq
            # unfold_conv_out_layer.shape[2] below is sequence length after strided unfolding
            unfold_conv_out_layer = unfold_conv_out_layer.transpose(1, 2).reshape(bs, unfold_conv_out_layer.shape[2], -1, self.kernel_size)  # B, seq, D, kernel_size
            conv_out_layer = torch.reshape(
                unfold_conv_out_layer,
                [-1, self.hidden_size // self.heads, self.kernel_size])  # BxSxH, H_dim, kernel
            conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
            # seqlen = unfold_conv_out_layer.shape[1]
            conv_out = torch.reshape(conv_out_layer, [bs, unfold_conv_out_layer.shape[1], -1])  # B, S, H, H_dim
        else:
            # TODO: implement strides here
            conv_kernel_layer = conv_kernel_layer.reshape(
                bs, seqlen, -1, self.kernel_size)
            conv_kernel_layer = conv_kernel_layer.permute(0, 2, 3,
                                                          1).contiguous()
            # B H K T
            weights = torch.softmax(conv_kernel_layer, dim=-2)

            # B,C,T
            conv_out_layer = self.conv_attn_point(value).transpose(
                1, 2).contiguous()

            conv_out_layer = dynamicconvFunction.apply(
                conv_out_layer, weights,
                self.padding_l).transpose(1, 2).contiguous()
            conv_out = torch.reshape(conv_out_layer, [bs, seqlen, -1])

        if upsampled:
            conv_out = pool_tensor(conv_out, self.cls_tokens, "mean", self.config.stride)
        return conv_out


class CompressSeqSDConv(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.stride = config.stride if use_in_funnel else config.compressed_query_attention_stride
        kernel_size = config.pooling_kernel_size if use_in_funnel else config.compressed_query_attention_kernel_size
        d_head = config.d_head[block_index]
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.sd_conv = SDConv(config, d_model, n_head, d_head, kernel_size, self.stride)

    def forward(self, query):

        cls = query[:, :self.cls_tokens]
        query = query[:, self.cls_tokens:]
        target_len = int(np.ceil(query.size(1) / self.stride))
        query = self.sd_conv(query)
        query = nn.functional.pad(query, (0, 0, 0, target_len - query.shape[1], 0, 0))
        q = torch.cat([cls, query], dim=1)
        q = q[:, :2 * (q.size(1) // 2)]
        return q


class CompressSeqMeanPooling(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.compressed_query_attention = config.stride if use_in_funnel else config.compressed_query_attention_stride
        self.cls_tokens = self.config.num_highway_cls_tokens + 1

    def forward(self, query):
        return pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)


class CompressSeqShortSeqRNN(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.compressed_query_attention = config.stride if use_in_funnel else config.compressed_query_attention_stride
        d_head = config.d_head[block_index]
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.rnn = ShortSeqRNN(config, d_model, n_head, d_head, config.short_rnn_kernel[block_index], config.short_rnn_overlap[block_index])

    def forward(self, query):
        query = self.rnn(query)
        return pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)


class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index, last_layer_index=None):
        super().__init__()
        if last_layer_index is None:
            last_layer_index = layer_index
        self.config = config
        self.block_index = block_index
        d_model, all_head, d_head = config.block_channel_size[block_index], config.n_head[block_index], config.d_head[block_index]
        n_head = all_head[0]
        total_heads = sum(all_head)
        assert d_model % 16 == 0
        remaining_d_model = d_model
        self.sdconv = config.sdconv[block_index]
        self.full_channel_separation = config.full_channel_separation[block_index]
        if self.sdconv:
            self.n_conv_head = all_head[1]
            if self.full_channel_separation:
                self.conv_dims = (self.n_conv_head * d_model) // total_heads
                remaining_d_model -= self.conv_dims
            else:
                self.conv_dims = d_model
            self.sdconv = SDConv(config, self.conv_dims, self.n_conv_head, d_head, config.sdconv_kernel_size[block_index])

        self.short_rnn = config.short_rnn[block_index]
        if self.short_rnn:
            self.n_rnn_head = all_head[2]
            if self.full_channel_separation:
                self.rnn_dims = (self.n_rnn_head * d_model) // total_heads
                remaining_d_model -= self.rnn_dims
            else:
                self.rnn_dims = d_model
            assert self.rnn_dims % self.n_rnn_head == 0
            self.rnn = ShortSeqRNN(config, self.rnn_dims, self.n_rnn_head, d_head, config.short_rnn_kernel[block_index], config.short_rnn_overlap[block_index])

        self.n_head = n_head
        d_model = remaining_d_model
        self.block_index = block_index
        self.layer_index = layer_index
        self.is_encoder_layer = is_encoder_layer
        query_compression_layers = set([(block_index, ll) for ll in range(layer_index, last_layer_index+1)]).intersection(set(config.compressed_query_attention_layers))
        self.query_compression_layers = query_compression_layers
        compress_query = is_encoder_layer and len(query_compression_layers) > 0

        key_compression_layers = set([(block_index, ll) for ll in range(layer_index, last_layer_index + 1)]).intersection(
            set(config.compressed_key_attention_layers))
        compress_key = is_encoder_layer and len(key_compression_layers) > 0
        self.key_compression_layers = key_compression_layers

        if config.compress_query_method == 'learn':
            CompressionClass = CompressSeqWeighted
        elif config.compress_query_method == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.compress_query_method == 'learn_rnn':
            CompressionClass = CompressSeqShortSeqRNN
        elif config.compress_query_method == 'mean':
            CompressionClass = CompressSeqMeanPooling
        else:
            assert not compress_query and not compress_key

        assert not (is_first_layer_of_block and compress_query and block_index != 0)

        assert d_model % d_head == 0
        assert d_model % n_head == 0
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.d_model = d_model
        self.cls_tokens = self.config.num_highway_cls_tokens + 1

        self.untie_cls = self.config.untie_cls
        self.separate_content_and_position_attention = self.config.separate_content_and_position_attention
        self.sequence_dependent_position_transform = self.config.sequence_dependent_position_transform
        self.approximate_attention = self.config.approximate_attention[block_index]
        qkv_squeeze = self.config.qkv_squeeze_fraction > 1
        sq_frac = self.config.qkv_squeeze_fraction
        if qkv_squeeze:
            assert d_model % sq_frac == 0

        qkv_transform_groups = self.config.qkv_transform_groups
        if qkv_transform_groups > 1:
            # assert n_head % qkv_transform_groups == 0 and n_head >= qkv_transform_groups
            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)

            self.q_head = ConvFFN(config, d_model, d_model//sq_frac, d_out=n_head * d_head, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)

            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            self.k_head = ConvFFN(config, d_model, d_model//sq_frac, d_out=n_head * d_head, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = ConvFFN(config, d_model, d_model//sq_frac, d_out=d_model, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, groups=qkv_transform_groups)

        else:
            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)

            self.q_head = BertFFN(config, d_model, d_model//sq_frac, d_out=n_head * d_head) if qkv_squeeze else nn.Linear(d_model, n_head * d_head, bias=False)
            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            self.k_head = BertFFN(config, d_model, d_model//sq_frac, d_out=n_head * d_head) if qkv_squeeze else nn.Linear(d_model, n_head * d_head)
            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = BertFFN(config, d_model, d_model//sq_frac, d_out=d_model) if qkv_squeeze else nn.Linear(d_model, d_model)

        if self.approximate_attention:
            self.attn = FastAttention(dim_heads=d_head, nb_features=n_head * d_head, )
            assert not compress_key
        if self.separate_content_and_position_attention:
            if self.sequence_dependent_position_transform:
                self.pos_q_head = SequenceDependentPositionTransform(config, config.embedding_size, d_model, n_head * d_head, qkv_transform_groups, compress_query)
                self.pos_k_head = SequenceDependentPositionTransform(config, config.embedding_size, d_model, n_head * d_head, qkv_transform_groups, False)
            elif qkv_transform_groups > 1:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=4 if compress_query else 1, groups=qkv_transform_groups, stride=4 if compress_query else 1)
                self.pos_k_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
            else:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=4, groups=qkv_transform_groups, stride=4) if compress_query else nn.Linear(config.embedding_size, n_head * d_head)
                self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head)
            self.c2p_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.p2c_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.pos_qln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)
            self.pos_kln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(config.block_channel_size[block_index], eps=config.layer_norm_eps)
        self.scale = 1.0 / (d_head ** 0.5)

    def self_attention(self, query, key, value, attention_inputs, layer_index):
        batch_size, seq_len, _ = query.shape
        initial_seq_len = seq_len
        position_embeds, attention_mask = attention_inputs
        position_embeds = position_embeds[self.block_index]
        position_embed_of_key, position_embed_of_query = position_embeds
        context_len = key.shape[1]
        n_head, d_head = self.n_head, self.config.d_head[self.block_index]
        need_query_compress = (self.block_index, layer_index) in self.query_compression_layers
        need_key_compress = (self.block_index, layer_index) in self.key_compression_layers
        if need_query_compress:
            query = self.q_head_compress(query)
        if need_key_compress:
            key = self.k_head_compress(key)
            value = pool_tensor(value, self.cls_tokens, mode='mean', stride=self.config.compressed_query_attention_stride)

        seq_len = (2 * int(
            np.ceil((seq_len - self.cls_tokens) / self.config.compressed_query_attention_stride) // 2) + self.cls_tokens) if need_query_compress else seq_len
        context_len = (2 * int(
            np.ceil(
                (context_len - self.cls_tokens) / self.config.compressed_query_attention_stride) // 2) + self.cls_tokens) if need_key_compress else context_len
        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, self.d_model // n_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        if self.separate_content_and_position_attention:
            if need_query_compress:
                position_embed_of_query = pool_tensor(position_embed_of_query.unsqueeze(0), self.cls_tokens, mode='mean',
                                                      stride=self.config.compressed_query_attention_stride).squeeze()
            if need_key_compress:
                position_embed_of_key = pool_tensor(position_embed_of_key.unsqueeze(0), self.cls_tokens, mode='mean',
                                                    stride=self.config.compressed_query_attention_stride).squeeze()
            v = self.c2p_bias * self.scale
            w = self.p2c_bias * self.scale
            if self.sequence_dependent_position_transform:
                pos_k_head = self.pos_kln(self.pos_k_head(key, position_embed_of_key, 1)).view(batch_size, -1, n_head, d_head)
                pos_q_head = self.pos_qln(self.pos_q_head(query, position_embed_of_query, 1)).view(batch_size, -1, n_head, d_head)
                if self.untie_cls:
                    pos_k_head = pos_k_head * F.pad(
                        pos_k_head.new_ones([pos_k_head.size(0), pos_k_head.size(1) - self.cls_tokens, pos_k_head.size(2), pos_k_head.size(3)]),
                        (0, 0, self.cls_tokens, 0, 0, 0, 0, 0))
                    pos_q_head = pos_q_head * F.pad(
                        pos_q_head.new_ones([pos_q_head.size(0), pos_q_head.size(1) - self.cls_tokens, pos_q_head.size(2), pos_q_head.size(3)]),
                        (0, 0, self.cls_tokens, 0, 0, 0, 0, 0))
            else:

                pos_k_head = self.pos_kln(self.pos_k_head(position_embed_of_key)).view(context_len, n_head, d_head)
                pos_q_head = self.pos_qln(self.pos_q_head(position_embed_of_query)).view(seq_len, n_head, d_head)
                # print(query.size(), key.size(), position_embed_of_query.size(), position_embed_of_key.size())

                if self.untie_cls:
                    pos_k_head = pos_k_head * F.pad(pos_k_head.new_ones([pos_k_head.size(0) - self.cls_tokens, pos_k_head.size(1), pos_k_head.size(2)]),
                                                    (self.cls_tokens, 0, 0, 0, 0, 0))
                    pos_q_head = pos_q_head * F.pad(pos_q_head.new_ones([pos_q_head.size(0) - self.cls_tokens, pos_q_head.size(1), pos_q_head.size(2)]),
                                                    (self.cls_tokens, 0, 0, 0, 0, 0))

        if self.approximate_attention:
            # TODO: how to handle attention masks
            v_head = v_head.permute(0, 2, 1, 3)
            attn_vec = self.attn(q_head.permute(0, 2, 1, 3), k_head.permute(0, 2, 1, 3), v_head).permute(0, 2, 1, 3)
            if self.separate_content_and_position_attention:
                c2p_score = self.attn((q_head + v).permute(0, 2, 1, 3), pos_k_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                                                       3)
                p2c_score = self.attn(pos_q_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), (k_head + w).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                                                       3)
                p2p_score = self.attn(pos_q_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3),
                                      pos_k_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                     3)
                attn_vec = attn_vec + c2p_score + p2c_score + p2p_score
                # TODO: try adaptive weighting

        else:
            content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
            attn_score = content_score

            if self.separate_content_and_position_attention:
                if self.sequence_dependent_position_transform:
                    c2p_score = torch.einsum("bind,bjnd->bnij", q_head + v, pos_k_head)
                    p2c_score = torch.einsum("bind,bjnd->bnij", pos_q_head, k_head + w)
                    p2p_score = torch.einsum("bind,bjnd->bnij", pos_q_head, pos_k_head)
                else:
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
                # TODO: handle attention mask's pooling for qk pooling
                if need_key_compress:
                    attention_mask = pool_tensor(attention_mask, self.cls_tokens, mode='min', stride=self.config.compressed_query_attention_stride)
                attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
            # attention probability
            attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
            attn_prob = self.attention_dropout(attn_prob)

            # attention output, shape batch_size x seq_len x n_head x d_head
            attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)
        attn_out = attn_vec.reshape(batch_size, seq_len, self.d_model)
        # Shape shape batch_size x seq_len x d_model

        if need_query_compress:
            attn_out = upsample(attn_out, self.config.compressed_query_attention_kernel_size, initial_seq_len, self.cls_tokens)
        return attn_out, attn_prob

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        batch_size, seq_len, _ = query.shape
        query_temp = query
        if self.sdconv:
            if self.full_channel_separation:
                sdconv_out = self.sdconv(query[:, :, :self.conv_dims], key[:, :, :self.conv_dims], value[:, :, :self.conv_dims])
                query = query[:, :, self.conv_dims:]
                key = key[:, :, self.conv_dims:]
                value = value[:, :, self.conv_dims:]
            else:
                sdconv_out = self.sdconv(query, key, value)

        if self.short_rnn:
            if self.full_channel_separation:
                rnn_out = self.rnn(query[:, :, :self.rnn_dims], key[:, :, :self.rnn_dims], value[:, :, :self.rnn_dims])
                query = query[:, :, self.rnn_dims:]
                key = key[:, :, self.rnn_dims:]
                value = value[:, :, self.rnn_dims:]
            else:
                rnn_out = self.rnn(query, key, value)

        attn_out, attn_prob = self.self_attention(query, key, value, attention_inputs, layer_index)

        if self.sdconv:
            attn_out = torch.cat([sdconv_out, attn_out], dim=-1)
        if self.short_rnn:
            attn_out = torch.cat([rnn_out, attn_out], dim=-1)
        output = self.layer_norm(query_temp + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class ConvFFN(nn.Module):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, config: FunnelConfig, d_model, d_inner, groups, layers=0, d_out=None):
        super().__init__()
        d_out = d_model if d_out is None else d_out
        cin, cout = d_model, d_out
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
        output = output.permute(0, 2, 1)
        return output


class BertFFN(nn.Module):
    def __init__(self, config: FunnelConfig, d_model, d_inner, layers=0, d_out=None):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        d_out = d_model if d_out is None else d_out
        self.linear_2 = nn.Linear(d_inner, d_out)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList() if layers > 0 else None
        for _ in range(layers):
            self.layers.append(nn.Linear(d_inner, d_inner))

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h)
        if self.layers:
            for ll in self.layers:
                h = ll(h)
                h = self.act(h)
                h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return h


class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, is_last_layer_of_block, is_encoder_layer):
        super().__init__()
        groups, layers = config.ffn_groups, config.ffn_layers
        d_model, d_inner = config.block_channel_size[block_index], config.block_channel_size[block_index] * config.ffn_width
        d_next = config.block_channel_size[block_index + 1] if (block_index + 1) < len(config.block_channel_size) else d_model
        self.n_blocks = config.block_sizes[block_index] - 1
        self.need_dim_match = d_model != d_next and is_encoder_layer and is_last_layer_of_block
        self.diff = d_next - d_model
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.layer_norm = nn.LayerNorm(d_model, config.layer_norm_eps)
        if self.need_dim_match:
            self.dlayer_norm = nn.LayerNorm(d_next, config.layer_norm_eps)
            self.dlin = nn.Linear(d_model, self.diff)
        if groups > 1:
            assert d_model % groups == 0
            self.lin = nn.Linear(d_model, d_model)
            self.ffn = ConvFFN(config, d_model, d_inner, groups, layers)
        else:
            self.lin = None
            self.ffn = BertFFN(config, d_model, d_inner, layers)

    def forward(self, hidden, layer_index=None):
        dim_match = self.need_dim_match and layer_index == self.n_blocks
        if self.lin:
            h = self.activation_dropout(self.activation_function(self.lin(hidden)))
            h = self.ffn(h)
        else:
            h = self.ffn(hidden)
        pre_ffn = h
        if dim_match:
            hidden = nn.functional.pad(hidden, (0, self.diff, 0, 0, 0, 0))
            h = torch.cat((h, self.dlin(h)), 2)
            return pre_ffn, self.dlayer_norm(hidden + h)
        else:
            return pre_ffn, self.layer_norm(hidden + h)


class LightLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index, is_encoder_layer):
        cin = config.block_channel_size[block_index]
        cout = cin // 2
        self.is_encoder_layer = is_encoder_layer
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=cout, out_channels=cout, kernel_size=5, groups=sum(config.n_head[block_index]) // 2, padding=2, padding_mode='zeros')
        self.layer_norm = nn.LayerNorm(cout * 2, config.layer_norm_eps)
        self.activation_function = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.attention_dropout)
        self.cls_tokens = config.num_highway_cls_tokens + 1
        d_head = config.d_head[block_index]
        self.rnn = ShortSeqRNN(config, cout, sum(config.n_head[block_index]) // 2, d_head, config.short_rnn_kernel[block_index], config.short_rnn_overlap[block_index])
        self.lin = nn.Linear(cin, cin)
        self.cout = cout
        # padding

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        qcnn = self.c1(query[:, :, :self.cout].permute(0, 2, 1)).permute(0, 2, 1)
        qrnn = self.rnn(query[:, :, self.cout:])
        q = torch.cat((qcnn, qrnn), 2)
        q = self.activation_function(q)
        q = self.dropout(q)
        q = self.lin(q)
        res = self.layer_norm(query + q)
        return (res, res)


class FunnelLayer(nn.Module):
    def __init__(self, config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index, last_layer_index=None):
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index, last_layer_index)
        self.ffn = FunnelPositionwiseFFN(config, block_index, is_last_layer_of_block, is_encoder_layer)

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        attn = self.attention(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
        pre_ffn, output = self.ffn(attn[0], layer_index)
        return (output, pre_ffn, attn[1]) if output_attentions else (output, pre_ffn)


class FunnelEncoder(nn.Module):
    def __init__(self, config: FunnelConfig):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)

        block_channel_size = config.block_channel_size
        self.blocks = nn.ModuleList()
        self.channel_match = nn.ModuleDict()
        self.repeats = []
        for block_index, block_size in enumerate(config.block_sizes):
            cur_channels = block_channel_size[block_index]
            next_channels = block_channel_size[min(block_index+1, len(block_channel_size) - 1)]
            self.blocks.append(nn.ModuleList())
            self.repeats.append([])
            i = 0
            while i < block_size:
                if config.block_repeats:

                    if i == 0 and config.separate_compressiion_layer and block_index > 0:
                        inext = i + 1
                        self.blocks[block_index].append(FunnelLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
                        self.repeats[block_index].append(1)
                        i = inext
                    elif i < block_size:
                        if config.light_first_layer and block_index == 0 and i == 0:
                            self.blocks[block_index].append(LightLayer(config, block_index, True))
                            self.repeats[block_index].append(1)
                            i += 1
                        else:
                            reps = (block_size - (i)) if cur_channels != next_channels else (block_size - i)
                            inext = i + reps
                            self.blocks[block_index].append(FunnelLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i + reps))
                            self.repeats[block_index].append(reps)
                            i = inext
                    else:
                        ValueError()
                else:
                    inext = i + 1
                    if config.light_first_layer:
                        self.blocks[block_index].append(LightLayer(config, block_index, True))
                    else:
                        self.blocks[block_index].append(FunnelLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
                    self.repeats[block_index].append(1)
                    i = inext
        self.pool = None
        if config.pooling_type == 'learn':
            CompressionClass = CompressSeqWeighted
        elif config.pooling_type == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.pooling_type == 'learn_rnn':
            CompressionClass = CompressSeqShortSeqRNN
        elif config.pooling_type == 'mean':
            CompressionClass = CompressSeqMeanPooling
        if config.pooling_type in ["learn", 'mean', 'learn_sdconv', 'learn_rnn'] and config.stride > 1:
            pool = nn.ModuleDict()
            for block_index, _ in enumerate(config.block_sizes[1:]):
                bi = block_index + 1
                pool[str(block_index+1)] = CompressionClass(config, bi, config.block_channel_size[bi], sum(config.n_head[bi]), use_in_funnel=True)
            self.pool = pool

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
        pre_ffn_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, (block, repeat_block) in enumerate(zip(self.blocks, self.repeats)):
            pooling_flag = hidden.size(1) > 2
            pooling_flag = pooling_flag and block_index > 0 and self.config.stride > 1
            if pooling_flag:
                if self.pool:
                    pooled_hidden = self.pool[str(block_index)](hidden)
            layer_index = 0
            for (_, (layer, repeats)) in enumerate(zip(block, repeat_block)):
                for repeat_index in range(repeats):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
                    layer_index += 1
                    hidden = layer_output[0]
                    pre_ffn = layer_output[1]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs, block_index)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
                        pre_ffn_states = pre_ffn_states + (pre_ffn,)

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, pre_ffn_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


def upsample(x, stride, target_len, cls_tokens):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x

    cls = x[:, :cls_tokens]
    x = x[:, cls_tokens:]
    output = torch.repeat_interleave(x, repeats=stride, dim=1)

    output = nn.functional.pad(output, (0, 0, 0, (target_len - cls_tokens) - output.shape[1], 0, 0))
    output = output[:, : target_len - cls_tokens]
    output = torch.cat([cls, output], dim=1)

    return output


class FunnelDecoder(nn.Module):
    def __init__(self, config: FunnelConfig):
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.layers = nn.ModuleList()
        if config.num_decoder_layers > 0:
            if config.block_repeats:
                self.layers.extend([FunnelLayer(config, 0, True, True, False, 0)])
                if config.light_last_layer:
                    self.layers.extend([LightLayer(config, 0, False)])
                    self.repeats = [config.num_decoder_layers - 1] + [1]
                else:
                    self.repeats = [config.num_decoder_layers]
            else:
                if config.light_last_layer:
                    self.layers.extend([FunnelLayer(config, 0, i == config.num_decoder_layers - 1, i == 0, False, i) for i in range(config.num_decoder_layers - 1)])
                    self.repeats = [1] * config.num_decoder_layers
                    self.layers = nn.ModuleList([LightLayer(config, 0, False)])
                else:
                    self.layers.extend([FunnelLayer(config, 0, i == config.num_decoder_layers - 1, i == 0, False, i) for i in range(config.num_decoder_layers)])
                    self.repeats = [1] * config.num_decoder_layers

        else:
            self.layers = None

        block_channel_size = self.config.block_channel_size
        self.final_hidden_fc = None
        ffn_groups = self.config.ffn_groups
        if block_channel_size[0] != block_channel_size[-1]:
            if ffn_groups > 1:
                assert block_channel_size[0] % ffn_groups == 0 and block_channel_size[-1] % ffn_groups == 0
                self.final_hidden_fc = Conv1d(in_channels=block_channel_size[-1], out_channels=block_channel_size[0], kernel_size=1, groups=ffn_groups)
            else:
                self.final_hidden_fc = nn.Linear(block_channel_size[-1], block_channel_size[0])

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
        if self.final_hidden_fc:
            final_hidden = self.final_hidden_fc(final_hidden)
        if not self.layers:
            hidden = final_hidden
            all_hidden_states = (hidden,) if output_hidden_states else None
            all_attentions = () if output_attentions else None
            if not return_dict:
                return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
            return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)

        upsampled_hidden = upsample(
            final_hidden,
            stride=self.config.stride ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            cls_tokens=self.cls_tokens,
        )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            position_embeds,
            attention_mask=attention_mask,
        )

        layer_index = 0
        for layer, repeats in zip(self.layers, self.repeats):
            for _ in range(repeats):
                layer_output = layer(hidden, hidden, hidden, attention_inputs, layer_index, output_attentions=output_attentions)
                hidden = layer_output[0]
                layer_index += 1

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
        elif classname == "FastAttention":
            if not hasattr(module,  'projection_matrix'):
                projection_matrix = module.create_projection(device = next(self.parameters()).device)
                module.register_buffer('projection_matrix', projection_matrix)


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


class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelDecoder(config)
        self.cls_tokens = config.num_highway_cls_tokens + 1

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

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
            first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
            position_embeds=position_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        cls_tokens = decoder_outputs[0][:, :self.cls_tokens]
        decoder_outputs = (decoder_outputs[0][:, self.cls_tokens-1:], decoder_outputs[1:])

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            outputs += (cls_tokens, )
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        self.init_weights()

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


class FunnelForMaskedLM(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.lm_dim_match = None
        self.cls_tokens = config.num_highway_cls_tokens + 1
        if config.embedding_size != config.block_channel_size[0]:
            self.lm_dim_match = nn.Linear(config.block_channel_size[0], config.embedding_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

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
        if self.lm_dim_match:
            last_hidden_state = self.lm_dim_match(last_hidden_state)
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits[:, :, :self.config.vocab_size].view(-1, self.config.vocab_size), labels.view(-1))

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
    import time
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining

    # repeated_funnel_channel_expanded_base
    # FunnelConfig(separate_content_and_position_attention=True, stride=4)
    # FunnelConfig(separate_content_and_position_attention=False, stride=4, approximate_attention=False)
    config = FunnelConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=3,
                          sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=4, ffn_groups=4,
                          approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=[24, 64, 80], separate_compressiion_layer=True,
                          qkv_squeeze_fraction=1, light_last_layer=True, light_first_layer=True,
                          sdconv=True, full_channel_separation=True, short_rnn=True,
                          sdconv_kernel_size=[5, 7, 9],
                          compress_query_method="mean", compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                          compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                             (1, 1), (1, 2), (1, 3), (1, 4),
                                                             (2, 1), (2, 2), (2, 3), (2, 4)
                                                             ],
                          compressed_key_attention_layers=[(0, 3), (0, 4),
                                                           (1, 3), (1, 4),
                                                           (2, 3), (2, 4)
                                                           ],
                          #n_head=[(1, 0, 7), (1, 0, 11), (1, 0, 11)],
                          #n_head=[(1, 7, 0), (1, 11, 0), (1, 11, 0)],
                          #n_head=[(8,), (12,), (12,)],
                          n_head=[(2, 2, 4), (4, 4, 4), (4, 4, 4)],
                          block_channel_size=[384, 768, 960], no_v_head=True,
                          )
    model = FunnelForMaskedLM(config)
    tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/intermediate-base")

    # model = AutoModel.from_pretrained("funnel-transformer/intermediate")
    # tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/intermediate")
    #
    # model = AutoModel.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    #
    # model = AutoModel.from_pretrained("albert-base-v2")
    # tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    #

    # model = AutoModel.from_pretrained("microsoft/deberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    # model = AutoModel.from_pretrained("distilbert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # model = AutoModel.from_pretrained("squeezebert/squeezebert-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

    # TODO: Test Longformer for long sequences as well.
    # TODO: test mobilebert

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Params = %s" % (params/1_000_000))
    print(model)
    print(model.funnel.encoder.repeats)

    model = model.eval()

    t1 = """
With the success of language pretraining, it is highly desirable to develop more
efficient architectures of good scalability that can exploit the abundant unlabeled
data at a lower cost. To improve the efficiency, we examine the much-overlooked
redundancy in maintaining a full-length token-level presentation, especially for
tasks that only require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More
importantly, by re-investing the saved FLOPs from length reduction in constructing
a deeper or wider model, we further improve the model capacity. In addition, to
perform token-level predictions as required by common pretraining objectives,
Funnel-Transformer is able to recover a deep representation for each token from
the reduced hidden sequence via a decoder. Empirically, with comparable or fewer
FLOPs, Funnel-Transformer outperforms the standard Transformer on a wide
variety of sequence-level prediction tasks, including text classification, language
understanding, and reading comprehension. Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and
longer training times. To address these problems, we present two parameterreduction techniques to lower memory consumption and increase the training
speed of BERT (Devlin et al., 2019). Comprehensive empirical evidence shows
that our proposed methods lead to models that scale much better compared to
the original BERT. We also use a self-supervised loss that focuses on modeling
inter-sentence coherence, and show it consistently helps downstream tasks with
multi-sentence inputs. As a result, our best model establishes new state-of-the-art
results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.
Self-attention is a useful mechanism to build generative models for language and images. It determines the importance of context elements by comparing each element to the current time step. In this paper, we show that a very lightweight convolution can perform competitively to the best reported self-attention results. Next, we introduce dynamic convolutions which are simpler and more efficient than self-attention. We predict separate convolution kernels based solely on the current time-step in order to determine the importance of context elements. The number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic. Experiments on large-scale machine translation, language modeling and abstractive summarization show that dynamic convolutions improve over strong self-attention models. On the WMT'14 English-German test set dynamic convolutions achieve a new state of the art of 29.7 BLEU.
"""
    t2 = """
    Most popular optimizers for deep learning can be broadly categorized as adaptive methods (e.g. Adam) and accelerated schemes (e.g. stochastic gradient descent (SGD) with momentum). For many models such as convolutional neural networks (CNNs), adaptive methods typically converge faster but generalize worse compared to SGD; for complex settings such as generative adversarial networks (GANs), adaptive methods are typically the default because of their stability.We propose AdaBelief to simultaneously achieve three goals: fast convergence as in adaptive methods, good generalization as in SGD, and training stability. The intuition for AdaBelief is to adapt the stepsize according to the "belief" in the current gradient direction. Viewing the exponential moving average (EMA) of the noisy gradient as the prediction of the gradient at the next time step, if the observed gradient greatly deviates from the prediction, we distrust the current observation and take a small step; if the observed gradient is close to the prediction, we trust it and take a large step. We validate AdaBelief in extensive experiments, showing that it outperforms other methods with fast convergence and high accuracy on image classification and language modeling. Specifically, on ImageNet, AdaBelief achieves comparable accuracy to SGD. Furthermore, in the training of a GAN on Cifar10, AdaBelief demonstrates high stability and improves the quality of generated samples compared to a well-tuned Adam optimizer.
    """
    t3 = """
    We present the Open Graph Benchmark (OGB), a diverse set of challenging and realistic benchmark datasets to facilitate scalable, robust, and reproducible graph machine learning (ML) research. OGB datasets are large-scale, encompass multiple important graph ML tasks, and cover a diverse range of domains, ranging from social and information networks to biological networks, molecular graphs, source code ASTs, and knowledge graphs. For each dataset, we provide a unified evaluation protocol using meaningful application-specific data splits and evaluation metrics. In addition to building the datasets, we also perform extensive benchmark experiments for each dataset. Our experiments suggest that OGB datasets present significant challenges of scalability to large-scale graphs and out-of-distribution generalization under realistic data splits, indicating fruitful opportunities for future research. Finally, OGB provides an automated end-to-end graph ML pipeline that simplifies and standardizes the process of graph data loading, experimental setup, and model evaluation. OGB will be regularly updated and welcomes inputs from the community. OGB datasets as well as data loaders, evaluation scripts, baseline code, and leaderboards are publicly available.
    """
    large_texts = [
        t1,
        t2,
        t3,
        t2+t3
        ]

    very_large_texts = [
        t1+t2+t3,
        t2+t3 + t1,
        t3+t1+t2+t1+t2+t3,
        t1+t2+t3+t1,
        t1,
        t2,
        t3,
        t1 + t3
    ]
    small_max_length = 128 - config.num_highway_cls_tokens
    medium_max_length = 512 - config.num_highway_cls_tokens
    large_max_length = 1024 - config.num_highway_cls_tokens
    very_large_max_length = 1536 - config.num_highway_cls_tokens

    pt_batch = tokenizer(very_large_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    print("Input Sizes", pt_batch["input_ids"].size())

    profile = False
    forward_only = True
    fp16 = False
    device = torch.device("cpu")

    model = model.to(device)
    pt_batch = {k: v.to(device) for k, v in pt_batch.items()}

    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
    except:
        pass
    if forward_only:
        _ = model.eval()
    else:
        _ = model.train()

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(all_params)

    def run():
        if not forward_only:
            if fp16:
                with autocast():
                    output = model(**pt_batch, labels=pt_batch["input_ids"])
                    loss = output[0]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(**pt_batch, labels=pt_batch["input_ids"])
                loss = output[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            if fp16:
                with autocast():
                    with torch.no_grad():
                        pt_outputs = model(**pt_batch)
            else:
                with torch.no_grad():
                    pt_outputs = model(**pt_batch)
                return [o.cpu() for o in pt_outputs]

    if profile:
        import torch.autograd.profiler as profiler
        _ = [run() for _ in range(2)]
        with profiler.profile(record_shapes=True) as prof:
            _ = [run() for _ in range(5)]
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
    else:
        _ = [run() for _ in range(2)]
        times = []
        for _ in trange(10):
            st = time.time()
            _ = run()
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.4f, Lowest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.std(times)), times)

