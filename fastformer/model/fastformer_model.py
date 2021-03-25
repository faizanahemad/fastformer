import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import traceback


from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast

from fairscale.nn.misc import checkpoint_wrapper
from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap

import numpy as np
import math
import random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
try:
    from performer_pytorch import SelfAttention, FastAttention
except:
    pass
from collections import defaultdict
from torch.nn import TransformerDecoder
import time

from torch.utils.data import DataLoader

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

from fastformer.data import *
from fastformer.model.AdMSLoss import AdMSoftmaxLoss, BCELossFocal
from fastformer.utils import *

# from transformers.activations import ACT2FN
ACT2FN = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "gelu_new": nn.GELU,
    "gelu_fast": nn.GELU,
    "linear": nn.Linear,
    "sigmoid": nn.Sigmoid,
}

try:
    from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
except:
    pass

from fastformer.config import *

logger = logging.get_logger(__name__)

INF = 1e6
EPS = 1e-6


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


class Dropout(torch.nn.Module):
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


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: FastFormerConfig):
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

        if config.char_rnn:
            char_div = 1
            char_rnn_layers = config.char_rnn_layers
            char_rnn_vocab_size = config.char_rnn_vocab_size
            self.char_embeddings = nn.Embedding(char_rnn_vocab_size, self.embedding_size // (2 * char_div), padding_idx=pad_token_id)
            self.char_rnn = ShortSeqRNN(config, self.embedding_size // (2 * char_div), 1, self.embedding_size // (2 * char_div),
                                        config.char_rnn_window_size, config.char_rnn_window_overlap, char_rnn_layers, maintain_dim=True)
            self.char_proj = nn.Linear(self.embedding_size // (2 * char_div), self.hidden_size, bias=False)
            self.char_div = char_div

        self.embed_proj = nn.Identity()
        if self.embedding_size != hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.LayerNormPosEmb = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps) if config.separate_content_and_position_attention else nn.Identity()
        self.dropout = Dropout(config.hidden_dropout)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings + config.num_highway_cls_tokens).expand((1, -1)))
        self.register_buffer("highway_position_ids", torch.arange(config.num_highway_cls_tokens).expand((1, -1)))
        if config.num_highway_cls_tokens > 0:
            self.register_buffer("highway_cls_tokens", torch.arange(config.vocab_size, config.vocab_size + config.num_highway_cls_tokens).expand((1, -1)))

    def forward(self, input_ids=None, inputs_embeds=None, token_type_ids=None, position_ids=None, mask=None, char_ids=None, char_offsets=None,
                use_embed_proj=True, use_highway_embeds=True):
        if inputs_embeds is None:
            input_shape = input_ids.size()
            input_shape = list(input_shape)
            initial_seq_len = input_shape[1]
            if use_highway_embeds:
                input_shape[1] = input_shape[1] + self.config.num_highway_cls_tokens
            input_shape = tuple(input_shape)
            inputs_embeds = self.word_embeddings(input_ids)
        if self.config.num_highway_cls_tokens > 0 and use_highway_embeds:
            highway_embeddings = self.word_embeddings(self.highway_cls_tokens).expand((inputs_embeds.size(0), -1, -1))
            inputs_embeds = torch.cat((highway_embeddings, inputs_embeds), dim=1)
        else:
            input_shape = inputs_embeds.size()
        seq_length = input_shape[1]
        embeddings = inputs_embeds
        if use_embed_proj:
            embeddings = self.embed_proj(embeddings)

        if self.config.char_rnn and char_ids is not None:
            char_offsets = char_offsets.flatten(1, 2).unsqueeze(-1).expand(input_shape[0], -1, self.embedding_size // (2 * self.char_div))
            char_embeds = self.char_rnn(self.char_embeddings(char_ids))
            char_embeds = torch.gather(char_embeds, 1, char_offsets).view(input_shape[0], initial_seq_len, 2, self.embedding_size // (2 * self.char_div)).mean(2)
            if self.config.num_highway_cls_tokens > 0 and use_highway_embeds:
                char_embeds = torch.cat((highway_embeddings[:, :, :char_embeds.size(-1)], char_embeds), dim=1)
            char_embeds = self.char_proj(char_embeds) if use_embed_proj else char_embeds
            embeddings = embeddings + char_embeds

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        elif use_highway_embeds and self.config.num_highway_cls_tokens > 0:
            position_ids = torch.cat((self.highway_position_ids.expand((position_ids.size(0), -1)), position_ids + self.config.num_highway_cls_tokens),
                                     dim=1)

        position_embeddings = self.position_embeddings(position_ids.long())
        if self.position_biased_input:
            embeddings += (self.embed_proj(position_embeddings) if use_embed_proj else position_embeddings)

        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            elif use_highway_embeds and self.config.num_highway_cls_tokens > 0:
                token_type_ids = torch.cat(
                    (torch.empty(input_shape[0], self.config.num_highway_cls_tokens, device=token_type_ids.device).fill_(token_type_ids[0][0]), token_type_ids),
                    dim=1)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += (self.embed_proj(token_type_embeddings) if use_embed_proj else token_type_embeddings)

        embeddings = self.LayerNorm(embeddings) if use_embed_proj else embeddings

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                if use_highway_embeds and self.config.num_highway_cls_tokens > 0:
                    mask = torch.cat((torch.ones(mask.size(0), self.config.num_highway_cls_tokens, dtype=mask.dtype, device=mask.device), mask), dim=1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings) if use_embed_proj else embeddings
        return embeddings, self.LayerNormPosEmb(position_embeddings.squeeze(0)) if position_embeddings is not None else None


def pool_tensor_basic(tensor, mode="mean", stride=2):
    if tensor is None:
        return None

    if stride == 1:
        return tensor

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

    return tensor


def pool_tensor(tensor, cls_size, mode="mean", stride=2):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None

    if stride == 1:
        return tensor

    # Do the pool recursively if tensor is a list or tuple of tensors.
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(pool_tensor(x, mode=mode, stride=stride) for x in tensor)

    # TODO: check if even length in dim=1 (seq dim)
    cls_tokens, tensor = tensor.split([cls_size, tensor.size(1) - cls_size], 1)
    tensor = pool_tensor_basic(tensor, mode, stride)

    tensor = torch.cat([cls_tokens, tensor], dim=1)
    ts = tensor.size()
    padding_extra = 8 * math.ceil(ts[1] / 8) - ts[1]
    # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    if len(ts) == 3:
        tensor = nn.functional.pad(tensor, (0, 0, 0, padding_extra), mode="constant")
    elif len(ts) == 2:
        tensor = nn.functional.pad(tensor, (0, padding_extra), mode="constant")
    return tensor


class AttentionStructure(nn.Module):
    """
    Contains helpers for `MultiheadAttention `.
    """

    def __init__(self, config: FastFormerConfig):
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
                pooled_pos = pooled_pos[:2 * (pooled_pos.size(0) // 2)]
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

    def post_attention_pooling(self, attention_inputs, block_index):
        """ Pool the proper parts of `attention_inputs` after the attention layer. """
        position_embeds, attention_mask = attention_inputs
        attention_mask = self.pool_tensor(attention_mask, mode="min", stride=self.stride)
        position_embeds[block_index][0] = position_embeds[block_index][1]
        attention_inputs = (position_embeds, attention_mask)
        return attention_inputs


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=True, stride=1, dilation=1, padding=0, padding_mode='zeros'):
        super().__init__()
        self.pre_permute = True
        self.post_permute = True
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              groups=groups, bias=bias, stride=stride, dilation=dilation, padding=padding, padding_mode=padding_mode)

    def forward(self, x, pre_permute=True, post_permute=True):
        unsqueeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            unsqueeze = True
        if pre_permute and self.pre_permute:
            x = x.permute(0, 2, 1)
        x = self.conv(x)
        if post_permute and self.post_permute:
            x = x.permute(0, 2, 1)
        if unsqueeze:
            x = x.squeeze(0)
        return x


class SequenceDependentPositionTransform(nn.Module):
    def __init__(self, config: FastFormerConfig, d_pos_in, d_model_in, d_out, qkv_transform_groups, compress):
        super().__init__()
        act = config.hidden_act
        self.cls_transform = Conv1d(in_channels=d_model_in, out_channels=d_pos_in, kernel_size=1,
                                    groups=qkv_transform_groups) if qkv_transform_groups > 1 else nn.Linear(d_model_in, d_pos_in)
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
    def __init__(self, config: FastFormerConfig, hidden_size, heads, head_size, kernel_size, overlap, layers=1, maintain_dim=True):
        super().__init__()
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.heads = heads
        self.kernel_size = kernel_size
        self.all_head_size = heads * head_size
        self.hidden_size = hidden_size
        act = config.hidden_act
        assert hidden_size % (2 * heads) == 0
        self.head_size = head_size
        self.overlap = overlap
        self.gru = nn.ModuleList()
        self.gru_global = nn.ModuleList()
        for i in range(heads):
            rnn = nn.RNN(hidden_size // self.heads, hidden_size // (2 * self.heads), layers,
                         nonlinearity="tanh",
                         bias=True, batch_first=True, dropout=0.0, bidirectional=True)
            rnn2 = nn.RNN(hidden_size // self.heads, hidden_size // ((2 if maintain_dim else 1) * self.heads), layers,
                          nonlinearity="tanh",
                          bias=True, batch_first=True, dropout=0.0, bidirectional=True)
            # rnn = nn.GRU(hidden_size // self.heads, hidden_size // (2 * self.heads), layers,
            #              bias=True, batch_first=True, dropout=0.0, bidirectional=True)
            # rnn2 = nn.GRU(hidden_size // self.heads, hidden_size // ((2 if maintain_dim else 1) * self.heads), layers,
            #               bias=False, batch_first=True, dropout=0.0, bidirectional=True)
            self.gru.append(rnn)
            self.gru_global.append(rnn2)

    def forward(self, query, key=None, value=None):
        # st = time.time()
        if key is None:
            key = query
        if value is None:
            value = key

        bs, seqlen, _ = query.shape
        context_len = key.shape[1]
        # assert self.hidden_size == dim

        upsampled = False
        if seqlen < context_len:
            upsampled = True
            query = value + upsample(query, self.config.stride, context_len, self.cls_tokens)
            seqlen = context_len

        num_segments = int(np.ceil(seqlen / self.kernel_size))
        target_len = num_segments * self.kernel_size
        if target_len - seqlen > 0:
            query = nn.functional.pad(query, (0, 0, 0, target_len - seqlen, 0, 0))
        query = query.view(-1, self.kernel_size, query.shape[2])
        query = query.view(query.shape[0], self.kernel_size, self.heads, -1)

        query = query.permute(2, 0, 1, 3)
        processed_query = []
        # stg = time.time()
        for i in range(query.size(0)):
            # print("Short Seq RNN sizes = ", query[i].size(), query.size())
            self.gru[i].flatten_parameters()
            qp = self.gru[i](query[i])[0]
            processed_query.append(qp)
        query = torch.stack(processed_query, 0)
        query = query.view(query.size(0), bs, num_segments, self.kernel_size, query.size(-1))

        processed_query = []
        query_global = torch.cat((query[:, :, :, 0:1, :], query[:, :, :, -2:-1, :]), -2).mean(-2)
        for i in range(query_global.size(0)):
            self.gru_global[i].flatten_parameters()
            qp = self.gru_global[i](query_global[i])[0]
            processed_query.append(qp)
        query_global = torch.stack(processed_query, 0).unsqueeze(-2)
        query = query + query_global
        query = query.view(query.size(0), bs * num_segments, self.kernel_size, query.size(-1))
        query = query.permute(1, 2, 0, 3).reshape(-1, self.kernel_size, query.size(-1))

        # query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        # query = self.gru(query)[0]
        # query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        query = query.view(bs, -1, query.size(-1))[:, :seqlen]

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        # et = time.time()
        # print("ShortSeqRNN timing, Overall = %.5f" % (et - st), "Only Gru = %.5f" % (et - stg), query.size())
        return query


class ShortSeqRNNOld(nn.Module):
    def __init__(self, config: FastFormerConfig, hidden_size, heads, head_size, kernel_size, overlap, layers=1, maintain_dim=True):
        super().__init__()
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.heads = heads
        self.kernel_size = kernel_size
        self.all_head_size = heads * head_size
        self.hidden_size = hidden_size
        act = config.hidden_act
        assert hidden_size % (2 * heads) == 0
        self.head_size = head_size
        self.overlap = overlap
        self.gru = nn.ModuleList()
        for i in range(heads):
            rnn = nn.RNN(hidden_size // self.heads, hidden_size // ((2 if maintain_dim else 1) * self.heads), layers,
                         nonlinearity="tanh",
                         bias=True, batch_first=True, dropout=0.0, bidirectional=True)
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0_reverse', )
            self.gru.append(rnn)

    def forward(self, query, key=None, value=None):
        # st = time.time()
        if key is None:
            key = query
        if value is None:
            value = key

        bs, seqlen, _ = query.shape
        context_len = key.shape[1]
        # assert self.hidden_size == dim

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
            seg_end = (i + 1) * self.kernel_size + 2 * self.overlap
            seg = query[:, seg_start:seg_end]
            segs.append(seg)
            segments.append((seg_start, seg_end, seg_end - 2 * self.overlap, seg_end - seg_start,))

        query = torch.stack(segs, 0).transpose(0, 1)
        query = query.reshape(-1, query.shape[2], query.shape[3])
        query = query.view(query.shape[0], query.shape[1], self.heads, -1)

        query = query.permute(2, 0, 1, 3)
        processed_query = []
        # stg = time.time()
        for i in range(query.size(0)):
            # print("Short Seq RNN sizes = ", query[i].size(), query.size())
            qp = self.gru[i](query[i])[0]
            processed_query.append(qp)
        query = torch.stack(processed_query, 0)
        query = query.permute(1, 2, 0, 3).reshape(-1, query.shape[2], query.size(-1))

        # query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        # query = self.gru(query)[0]
        # query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        query = query[:, self.overlap:-self.overlap]
        query = query.reshape(bs, -1, query.size(-1))[:, :seqlen]

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        # et = time.time()
        # print("ShortSeqRNN timing, Overall = %.5f" % (et - st), "Only Gru = %.5f" % (et - stg), query.size())
        return query


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=False, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=groups, bias=False, stride=stride, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, bias=bias, kernel_size=1, groups=pointwise_groups)

        self.out_channels = out_channels

    def forward(self, inputs):
        """
        Expect inputs in channels first format
        :param inputs:
        :return:
        """
        inputs = inputs.permute(0, 2, 1)
        inputs = self.depthwise(inputs)
        inputs = self.pointwise(inputs)
        inputs = inputs.permute(0, 2, 1)
        return inputs


class SDConv(nn.Module):
    def __init__(self, config: FastFormerConfig, hidden_size, heads, head_size, kernel_size=9, stride=1):
        super().__init__()
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.heads = heads
        self.kernel_size = kernel_size
        self.all_head_size = heads * head_size
        self.hidden_size = hidden_size
        self.stride = stride
        act = config.hidden_act
        self.act = checkpoint_wrapper(ACT2FN[act](), offload_to_cpu=False)
        assert hidden_size % heads == 0
        self.head_size = head_size
        self.separable_conv1d = SeparableConv1d(hidden_size, hidden_size, kernel_size, pointwise_groups=heads, stride=stride)
        # self.conv_attn_kernel = nn.Linear(self.all_head_size, self.heads * self.kernel_size)  # Multi-head?
        self.conv_attn_kernel = Conv1d(hidden_size, self.heads * self.kernel_size, 1, groups=heads)
        if config.no_v_head:
            self.conv_attn_point = nn.Identity()
        else:
            self.conv_attn_point = Conv1d(hidden_size, hidden_size, 1, groups=heads)
        self.use_cuda_conv = config.use_cuda_conv
        if not self.use_cuda_conv or self.stride != 1:
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
        conv_kernel_layer = self.conv_attn_kernel(conv_attn_layer)  # Softmax only in kernel dim

        if not self.use_cuda_conv or self.stride != 1:
            conv_kernel_layer = conv_kernel_layer.reshape(-1, self.kernel_size, 1)  # BxSxH, k, 1
            conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

            # conv_out_layer
            conv_out_layer = self.conv_attn_point(value).permute(0, 2, 1).unsqueeze(-1)  # B,D,Seq, 1
            unfold_conv_out_layer = self.unfold1d(conv_out_layer)  # B, D*kernel_size, seq
            # unfold_conv_out_layer.shape[2] below is sequence length after strided unfolding
            unfold_conv_out_layer = unfold_conv_out_layer.transpose(1, 2)  # B, seq, D, kernel_size
            conv_out_layer = torch.reshape(unfold_conv_out_layer, [-1, self.hidden_size // self.heads, self.kernel_size])  # BxSxH, H_dim, kernel
            conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
            # seqlen = unfold_conv_out_layer.shape[1]
            conv_out = torch.reshape(conv_out_layer, [bs, unfold_conv_out_layer.shape[1], -1])  # B, S, D
        else:
            # TODO: implement strides here
            conv_kernel_layer = conv_kernel_layer.reshape(
                bs, seqlen, -1, self.kernel_size)
            conv_kernel_layer = conv_kernel_layer.permute(0, 2, 3,
                                                          1).contiguous()
            # B H K T
            weights = torch.softmax(conv_kernel_layer, dim=-2)

            # B,C,T
            conv_out_layer = self.conv_attn_point(value).permute(0, 2, 1).contiguous()

            conv_out_layer = dynamicconvFunction.apply(
                conv_out_layer, weights,
                self.padding_l).transpose(1, 2).contiguous()
            conv_out = torch.reshape(conv_out_layer, [bs, seqlen, -1])

        if upsampled:
            conv_out = pool_tensor(conv_out, self.cls_tokens, "mean", self.config.stride)
        return conv_out


class CompressSeqSDConv(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        expand_dims = config.expand_dim_before_pooling if use_in_funnel else False
        if expand_dims:
            self.expansion_factor = 2
            self.expand = Conv1d(d_model, d_model * (self.expansion_factor - 1), 1, n_head, False)
            self.contract = nn.Sequential(Conv1d(d_model * 2, d_model, 1, n_head, False), nn.LayerNorm(d_model, config.layer_norm_eps))
        else:
            self.expand = nn.Identity()
            self.contract = nn.LayerNorm(d_model, config.layer_norm_eps) if use_in_funnel else nn.Identity()
            self.expansion_factor = 1
        self.stride = config.stride if use_in_funnel else config.compressed_query_attention_stride
        kernel_size = config.pooling_kernel_size if use_in_funnel else config.compressed_query_attention_kernel_size
        d_head = config.d_head[block_index] * self.expansion_factor
        self.d_model, self.n_head, self.d_head = d_model * self.expansion_factor, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.sd_conv = SDConv(config, d_model * self.expansion_factor, n_head, d_head, kernel_size, self.stride)

    def forward(self, query):
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.stride)
        if self.expansion_factor > 1:
            query = torch.cat((self.expand(query), query), dim=-1)
        cls, query = query.split([self.cls_tokens, query.size(1) - self.cls_tokens], 1)
        target_len = qskip.shape[1] - self.cls_tokens
        query = self.sd_conv(query)
        if target_len - query.shape[1] > 0:
            query = nn.functional.pad(query, (0, 0, 0, target_len - query.shape[1], 0, 0))
        q = torch.cat([cls, query], dim=1)
        return qskip + self.contract(q)


class CompressSeqMeanPooling(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.compressed_query_attention = config.stride if use_in_funnel else config.compressed_query_attention_stride
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        expand_dims = config.expand_dim_before_pooling if use_in_funnel else False
        if expand_dims:
            self.expansion_factor = 2
            self.expand = Conv1d(d_model, d_model * (self.expansion_factor - 1), 1, n_head, False)
            self.contract = nn.Sequential(Conv1d(d_model * 2, d_model, 1, n_head, False), nn.LayerNorm(d_model, config.layer_norm_eps))
        else:
            self.expand = nn.Identity()
            self.contract = nn.LayerNorm(d_model, config.layer_norm_eps) if use_in_funnel else nn.Identity()
            self.expansion_factor = 1

    def forward(self, query):
        # st = time.time()
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)
        if self.expansion_factor > 1:
            query = torch.cat((self.expand(query), query), dim=-1)
        query = self.contract(pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention))
        # ext = time.time()
        # print("Mean pooling = %.5f" % (ext - st))
        return qskip + query


class MultiheadAttention(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                 last_layer_index=None, force_no_sdconv=False, force_no_rnn=False):
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
        self.sdconv = config.sdconv[block_index] and not force_no_sdconv
        self.full_channel_separation = config.full_channel_separation[block_index]
        if self.sdconv:
            self.n_conv_head = all_head[1]
            if self.full_channel_separation:
                self.conv_dims = (self.n_conv_head * d_model) // total_heads
                remaining_d_model -= self.conv_dims
            else:
                self.conv_dims = d_model
            self.sdconv = SDConv(config, self.conv_dims, self.n_conv_head, d_head, config.sdconv_kernel_size[block_index])

        self.short_rnn = config.short_rnn[block_index] and not force_no_rnn
        if self.short_rnn:
            self.n_rnn_head = all_head[2]
            if self.full_channel_separation:
                self.rnn_dims = (self.n_rnn_head * d_model) // total_heads
                remaining_d_model -= self.rnn_dims
            else:
                self.rnn_dims = d_model
            assert self.rnn_dims % self.n_rnn_head == 0
            self.rnn = ShortSeqRNN(config, self.rnn_dims, 1, self.rnn_dims, config.short_rnn_kernel[block_index], config.short_rnn_overlap[block_index])

        self.n_head = n_head
        d_model = remaining_d_model
        self.block_index = block_index
        self.layer_index = layer_index
        self.is_encoder_layer = is_encoder_layer
        query_compression_layers = set([(block_index, ll) for ll in range(layer_index, last_layer_index + 1)]).intersection(
            set(config.compressed_query_attention_layers))
        self.query_compression_layers = query_compression_layers
        compress_query = is_encoder_layer and len(query_compression_layers) > 0 and config.compressed_query_attention_stride != 1

        key_compression_layers = set([(block_index, ll) for ll in range(layer_index, last_layer_index + 1)]).intersection(
            set(config.compressed_key_attention_layers))
        compress_key = is_encoder_layer and len(key_compression_layers) > 0 and config.compressed_query_attention_stride != 1
        self.key_compression_layers = key_compression_layers
        self.learned_compress = config.compress_query_method in ['learn_sdconv', 'learn_rnn', 'learn']

        if config.compress_query_method == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.compress_query_method == 'mean':
            CompressionClass = CompressSeqMeanPooling
        elif config.compress_query_method is None:
            compress_query = False
            compress_key = False
        else:
            assert not compress_query and not compress_key

        assert not (is_first_layer_of_block and compress_query and block_index != 0)

        assert d_model % d_head == 0
        assert d_model % n_head == 0
        assert d_model % (n_head * d_head) == 0
        self.attention_dropout = Dropout(config.attention_dropout)
        self.d_model = d_model
        self.cls_tokens = self.config.num_highway_cls_tokens + 1

        self.untie_cls = self.config.untie_cls
        self.separate_content_and_position_attention = self.config.separate_content_and_position_attention
        self.sequence_dependent_position_transform = self.config.sequence_dependent_position_transform
        self.approximate_attention = self.config.approximate_attention[block_index]

        qkv_transform_groups = self.config.qkv_transform_groups
        if qkv_transform_groups > 1:
            # assert n_head % qkv_transform_groups == 0 and n_head >= qkv_transform_groups
            self.q_head = Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=True)
            self.k_head = Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)

            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)


            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = Conv1d(
                    in_channels=d_model, out_channels=d_model, kernel_size=1, groups=qkv_transform_groups)

        else:
            self.q_head = nn.Linear(d_model, n_head * d_head,
                                                                                                                            bias=True)
            self.k_head = nn.Linear(d_model, n_head * d_head, bias=False)

            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = nn.Linear(d_model, d_model)

        if self.approximate_attention:
            self.attn = FastAttention(dim_heads=d_head, nb_features=n_head * d_head, )
            assert not compress_key
        if self.separate_content_and_position_attention:
            if self.sequence_dependent_position_transform:
                self.pos_q_head = SequenceDependentPositionTransform(config, config.embedding_size, d_model, n_head * d_head, qkv_transform_groups,
                                                                     compress_query)
                self.pos_k_head = SequenceDependentPositionTransform(config, config.embedding_size, d_model, n_head * d_head, qkv_transform_groups, False)
            elif qkv_transform_groups > 1:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=4 if compress_query else 1,
                                         groups=qkv_transform_groups, stride=4 if compress_query else 1)
                self.pos_k_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)
            else:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=4, groups=qkv_transform_groups,
                                         stride=4) if compress_query else nn.Linear(config.embedding_size, n_head * d_head)
                self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head)
            self.c2p_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.p2c_bias = nn.Parameter(torch.zeros([n_head, d_head]))
            self.pos_qln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)
            self.pos_kln = nn.LayerNorm(n_head * d_head, eps=config.layer_norm_eps)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(config.block_channel_size[block_index], eps=config.layer_norm_eps)
        self.scale = 1.0 / (d_head ** 0.5)

    def self_attention(self, query, key, value, attention_inputs, layer_index):
        batch_size, seq_len, dim = query.shape
        initial_seq_len = seq_len
        position_embeds, attention_mask = attention_inputs
        context_len = key.shape[1]
        n_head, d_head = self.n_head, self.config.d_head[self.block_index]
        need_query_compress = (self.block_index, layer_index) in self.query_compression_layers and self.config.compressed_query_attention_stride != 1 and self.is_encoder_layer and self.config.compress_query_method is not None
        need_key_compress = (self.block_index, layer_index) in self.key_compression_layers and self.config.compressed_query_attention_stride != 1 and self.is_encoder_layer and self.config.compress_query_method is not None
        stride = dim // (n_head * d_head)
        if need_query_compress:
            query = self.q_head_compress(query)
            seq_len = (2 * int(
                np.ceil(
                    (seq_len - self.cls_tokens) / self.config.compressed_query_attention_stride) // 2) + self.cls_tokens) if need_query_compress else seq_len
        if need_key_compress:
            key = self.k_head_compress(key)
            value = pool_tensor(value, self.cls_tokens, mode='mean', stride=self.config.compressed_query_attention_stride)
            context_len = (2 * int(
                np.ceil(
                    (context_len - self.cls_tokens) / self.config.compressed_query_attention_stride) // 2) + self.cls_tokens) if need_key_compress else context_len
        # Shape batch_size x seq_len x n_head x d_head
        if self.learned_compress and need_query_compress:
            q_head = pool_tensor_basic(query.transpose(1, 2), "mean", stride).transpose(1, 2).view(batch_size, seq_len, n_head, d_head)
        else:
            q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        if self.learned_compress and need_key_compress:
            k_head = pool_tensor_basic(key.transpose(1, 2), "mean", stride).transpose(1, 2).view(batch_size, context_len, n_head, d_head)
        else:
            k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, self.d_model // n_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        if self.separate_content_and_position_attention:
            position_embeds = position_embeds[self.block_index]
            position_embed_of_key, position_embed_of_query = position_embeds
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
                if len(attention_mask.size()) == 2:
                    attention_mask = attention_mask[:, None, None].float()
                attn_score = attn_score - INF * (1 - attention_mask)
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
        assert query.size(1) > 0
        batch_size, seq_len, _ = query.shape
        query_temp = query
        if self.sdconv:

            if self.full_channel_separation:
                if key is None and value is None:
                    qp1, query = query.split([self.conv_dims, query.size(-1) - self.conv_dims], -1)
                    kp1, _ = qp1, query
                    vp1, _ = qp1, query
                else:
                    qp1, query = query.split([self.conv_dims, query.size(-1) - self.conv_dims], -1)
                    kp1, key = key.split([self.conv_dims, key.size(-1) - self.conv_dims], -1)
                    vp1, value = value.split([self.conv_dims, value.size(-1) - self.conv_dims], -1)
                sdconv_out = self.sdconv(qp1, kp1, vp1)
            else:
                if key is None and value is None:
                    key = query
                    value = query
                sdconv_out = self.sdconv(query, key, value)

        if self.short_rnn:
            if self.full_channel_separation:
                if key is None and value is None:
                    qp1, query = query.split([self.rnn_dims, query.size(-1) - self.rnn_dims], -1)
                    kp1, _ = qp1, query
                    vp1, _ = qp1, query
                else:
                    qp1, query = query.split([self.rnn_dims, query.size(-1) - self.rnn_dims], -1)
                    kp1, key = key.split([self.rnn_dims, key.size(-1) - self.rnn_dims], -1)
                    vp1, value = value.split([self.rnn_dims, value.size(-1) - self.rnn_dims], -1)
                rnn_out = self.rnn(qp1, kp1, vp1)
            else:
                if key is None and value is None:
                    key = query
                    value = query
                rnn_out = self.rnn(query, key, value)

        try:
            if key is None and value is None:
                key = query
                value = query
            attn_out, attn_prob = self.self_attention(query, key, value, attention_inputs, layer_index)
        except Exception as e:
            print("[Exception-in-train]: Query Shape = %s, Key shape = %s, Value shape = %s, layer_index = %s, block index = %s" % (query.size(), key.size(), value.size(), layer_index, self.block_index))
            # traceback.print_exc()
            # traceback.print_exception(*sys.exc_info())
            raise e

        if self.sdconv:
            attn_out = torch.cat([sdconv_out, attn_out], dim=-1)
        if self.short_rnn:
            attn_out = torch.cat([rnn_out, attn_out], dim=-1)
        if self.config.identity_preserving_norm:
            output = query_temp + self.layer_norm(attn_out)
        else:
            output = self.layer_norm(query_temp + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class ConvFFN(nn.Module):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, config: FastFormerConfig, d_model, d_inner, groups, layers=0, d_out=None):
        super().__init__()
        d_out = d_model if d_out is None else d_out
        cin, cout = d_model, d_out
        act = config.hidden_act
        self.conv1d_in = Conv1d(in_channels=cin, out_channels=d_inner, kernel_size=1, groups=groups, bias=True)
        self.conv1d_in.post_permute = False
        self.activation_dropout = Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList() if layers > 0 else None
        for _ in range(layers):
            cnn = Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=1, groups=groups)
            cnn.pre_permute=False
            cnn.post_permute=False
            self.layers.append(cnn)
        self.conv1d_out = Conv1d(in_channels=d_inner, out_channels=cout, kernel_size=1, groups=groups, bias=False)
        self.conv1d_out.pre_permute = False
        self.act = checkpoint_wrapper(ACT2FN[act](), offload_to_cpu=False)

    def forward(self, x):
        h = x
        output = self.conv1d_in(h)
        output = self.act(output)
        output = self.activation_dropout(output)
        if self.layers:
            for ll in self.layers:
                output = ll(output)
                output = self.act(output)
                output = self.activation_dropout(output)
        output = self.conv1d_out(output)

        return output


class BertFFN(nn.Module):
    def __init__(self, config: FastFormerConfig, d_model, d_inner, layers=0, d_out=None):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_inner, bias=True)
        self.activation_function = checkpoint_wrapper(ACT2FN[config.hidden_act](), offload_to_cpu=False)
        self.activation_dropout = Dropout(config.hidden_dropout)
        d_out = d_model if d_out is None else d_out
        self.linear_2 = nn.Linear(d_inner, d_out, bias=False)
        self.layers = nn.ModuleList() if layers > 0 else None
        for _ in range(layers):
            self.layers.append(nn.Linear(d_inner, d_inner, bias=False))

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
        return h


class PositionwiseFFN(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_encoder_layer):
        super().__init__()
        self.config = config
        groups, layers = config.ffn_groups, config.ffn_layers
        d_model, d_inner = config.block_channel_size[block_index], config.block_channel_size[block_index] * config.ffn_width
        d_next = config.block_channel_size[block_index + 1] if (block_index + 1) < len(config.block_channel_size) else d_model
        self.n_blocks = config.block_sizes[block_index] - 1
        self.need_dim_match = d_model != d_next and is_encoder_layer and is_last_layer_of_block
        self.diff = d_next - d_model
        self.d_model = d_model
        self.activation_function = checkpoint_wrapper(ACT2FN[config.hidden_act](), offload_to_cpu=False)
        self.layer_norm = nn.LayerNorm(d_model, config.layer_norm_eps)
        if self.need_dim_match:
            self.dlayer_norm = nn.LayerNorm(self.diff, config.layer_norm_eps)
            self.dlin = Conv1d(d_model, self.diff, 1, 8, bias=False) if d_model % 8 == 0 and self.diff % 8 == 0 and groups > 1 else nn.Linear(d_model, self.diff, bias=False)
        if groups > 1:
            assert d_model % groups == 0
            self.lin = nn.Linear(d_model, d_model)
            self.ffn = ConvFFN(config, d_model, d_inner, groups, layers)
        else:
            self.lin = nn.Identity()
            self.ffn = BertFFN(config, d_model, d_inner, layers)

    def forward(self, hidden, layer_index=None):
        dim_match = self.need_dim_match and layer_index == self.n_blocks
        h = self.lin(hidden)
        h = self.ffn(h)
        if dim_match:
            if self.config.identity_preserving_norm:
                h = self.layer_norm(h)
                pre_ffn = h
                dh = self.dlayer_norm(self.dlin(h))
                h = torch.cat((h, dh), 2)
                hidden = nn.functional.pad(hidden, (0, self.diff, 0, 0, 0, 0))
                h = hidden + h
            else:
                dh = self.dlin(h)
                pre_ffn = h
                h = torch.cat((self.layer_norm(h + hidden), self.dlayer_norm(dh)), 2)
        else:
            if self.config.identity_preserving_norm:
                h = self.layer_norm(h)
                pre_ffn = h
                h = hidden + h
            else:
                pre_ffn = h
                h = self.layer_norm(hidden + h)
        return pre_ffn, h


class LightLayer(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_encoder_layer):
        super().__init__()
        self.config = config
        cin = config.block_channel_size[block_index]
        cout = cin // 2
        self.is_encoder_layer = is_encoder_layer

        self.layer_norm = nn.LayerNorm(cout * 2, config.layer_norm_eps)
        self.activation_function = checkpoint_wrapper(ACT2FN[config.hidden_act](), offload_to_cpu=False)
        self.cls_tokens = config.num_highway_cls_tokens + 1
        # d_head = config.d_head[block_index]
        assert cout % (sum(config.n_head[block_index]) // 2) == 0
        self.c1 = SDConv(config, cout, sum(config.n_head[block_index]) // 2, cout // (sum(config.n_head[block_index]) // 2), config.sdconv_kernel_size[0])
        self.rnn = ShortSeqRNN(config, cout, 1, cout, config.short_rnn_kernel[block_index],
                               config.short_rnn_overlap[block_index])
        self.lin = nn.Linear(cin, cin, bias=False)
        self.cout = cout
        # padding

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        qcnn_in, qrnn_in = query.split([self.cout, query.size(-1) - self.cout], -1)
        qcnn = self.c1(qcnn_in)
        qrnn = self.rnn(qrnn_in)
        q = torch.cat((qcnn, qrnn), 2)
        q = self.activation_function(q)
        q = self.lin(q)
        if self.config.identity_preserving_norm:
            res = query + self.layer_norm(q)
        else:
            res = self.layer_norm(query + q)
        return (res, res)


class TransformerLayer(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_first_layer_of_block,
                 is_encoder_layer, layer_index, last_layer_index=None, alternate_ffn=True):
        super().__init__()
        self.attention = MultiheadAttention(config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                                            last_layer_index)
        self.ffn = PositionwiseFFN(config, block_index, is_last_layer_of_block, is_encoder_layer)
        self.alternate_ffn = alternate_ffn and (config.alternate_ffn if hasattr(config, "alternate_ffn") else True)
        self.block_index = block_index
        self.block_size = config.block_sizes[block_index]
        self.is_last_layer_of_block = is_last_layer_of_block

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        assert query.size(1) > 0
        attn = self.attention(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
        if self.alternate_ffn and layer_index % 2 == 0:
            h = self.ffn.layer_norm(self.ffn.lin(attn[0]))
            pre_ffn, output = None, h + attn[0]
        else:
            pre_ffn, output = self.ffn(attn[0], layer_index)
        pre_ffn = pre_ffn if self.block_size - 1 == layer_index else None
        return (output, pre_ffn, attn[1]) if output_attentions else (output, pre_ffn)


class TransformerEncoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
        self.config = config
        self.attention_structure = AttentionStructure(config)

        block_channel_size = config.block_channel_size
        self.blocks = nn.ModuleList()
        self.repeats = []
        for block_index, block_size in enumerate(config.block_sizes):
            cur_channels = block_channel_size[block_index]
            next_channels = block_channel_size[min(block_index + 1, len(block_channel_size) - 1)]
            self.blocks.append(nn.ModuleList())
            self.repeats.append([])
            i = 0
            while i < block_size:
                if config.block_repeats:

                    if i == 0 and config.separate_compressiion_layer and block_index > 0:
                        inext = i + 1
                        self.blocks[block_index].append(fsdp_wrapper(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i, alternate_ffn=False)))
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
                            self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i + reps))
                            self.repeats[block_index].append(reps)
                            i = inext
                    else:
                        ValueError()
                else:
                    inext = i + 1
                    if config.light_first_layer:
                        self.blocks[block_index].append(LightLayer(config, block_index, True))
                    else:
                        self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
                    self.repeats[block_index].append(1)
                    i = inext
        self.pool = None
        if config.pooling_type == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.pooling_type == 'mean':
            CompressionClass = CompressSeqMeanPooling
        if config.pooling_type in ['mean', 'learn_sdconv'] and config.stride > 1:
            pool = nn.ModuleDict()
            for block_index, _ in enumerate(config.block_sizes[1:]):
                bi = block_index + 1
                pool[str(block_index + 1)] = fsdp_wrapper(CompressionClass(config, bi, config.block_channel_size[bi], sum(config.n_head[bi]), use_in_funnel=True))
            self.pool = pool

    def forward_one_block(self, block_index, hidden, attention_inputs,
                          all_hidden_states, pre_ffn_states, all_attentions,
                          output_attentions=False, output_hidden_states=False):
        assert hidden.size(1) > 0
        (block, repeat_block) = self.blocks[block_index], self.repeats[block_index]
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
                # print("Block = ", block_index, ", Layer = ", layer_index, ", Sizes = ", query.size(), key.size(), value.size())
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

        return hidden, all_hidden_states, pre_ffn_states, all_attentions, attention_inputs

    def forward(
            self,
            inputs_embeds,
            position_embeds,
            attention_mask,
            output_attentions=False,
            output_hidden_states=False,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        assert inputs_embeds.size(1) > 0
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
        block_attention_masks = [attention_inputs[1]]
        for block_index, (_, _) in enumerate(zip(self.blocks, self.repeats)):
            # print("Block = ", block_index, ", Sizes = ", hidden.size(), attention_mask[0].size(), attention_mask[1].size(), all_hidden_states[-1].size())
            one_block_res = self.forward_one_block(block_index, hidden, attention_inputs, all_hidden_states, pre_ffn_states, all_attentions, output_attentions, output_hidden_states)
            hidden, all_hidden_states, pre_ffn_states, all_attentions, attention_inputs = one_block_res
            block_attention_masks.append(attention_inputs[1])

        return tuple(v for v in [hidden, all_hidden_states, pre_ffn_states, all_attentions, block_attention_masks] if v is not None)


def shift_right(input_ids, decoder_start_token_id, pad_token_id):

    assert (
        decoder_start_token_id is not None
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    # assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class TransformerCrossAttentionDecoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
        config = copy.deepcopy(config)
        config.short_rnn = [False]
        config.sdconv = [False]
        config.compressed_query_attention_layers = {}
        config.compressed_key_attention_layers = {}
        config.separate_content_and_position_attention = False
        config.sequence_dependent_position_transform = False
        config.attention_dropout = 0.0

        self.config = config
        block_index = 0
        all_head = config.n_head[block_index]

        total_heads = sum(all_head) // 2
        assert sum(all_head) % 2 == 0
        config.block_channel_size[block_index] = config.embedding_size
        config.block_channel_size[block_index] % total_heads == 0
        config.n_head[block_index] = (total_heads,) + tuple([0] * len(config.n_head[block_index][1:]))
        config.d_head[block_index] = config.block_channel_size[block_index] // total_heads
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.self_attn = MultiheadAttention(config, block_index, False, True, False, 0, force_no_sdconv=True, force_no_rnn=True)
        self.self_attn_lin = nn.Linear(config.block_channel_size[block_index], config.block_channel_size[block_index], bias=False)
        self.relu = nn.LeakyReLU()
        self.self_attn_ln = nn.LayerNorm(config.block_channel_size[block_index], config.layer_norm_eps)

    def forward(self, query, key, value, query_padding_mask, key_padding_mask, query_mask=None, key_mask=None):
        bs, seq_len, dim = query.shape
        query_temp = query
        if query_mask is None:
            query_mask = subsequent_mask(seq_len).to(query.device).unsqueeze(0)

        query_padding_mask = query_padding_mask[:, None, None].expand(bs, 1, seq_len, seq_len)

        if key_mask is None:
            key_mask = key_padding_mask

        (query, ) = self.self_attn(query, query, query, (None, torch.logical_and(query_mask, query_padding_mask).type(query_padding_mask.dtype)), 0, False)
        query = self.self_attn_lin(self.relu(query))
        (query,) = self.self_attn(query, key, value, (None, torch.logical_and(key_mask, key_padding_mask).type(key_padding_mask.dtype)), 0, False)
        query = self.self_attn_lin(query)
        return self.self_attn_ln(query)


def upsample(x, stride, target_len, cls_tokens):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x

    cls, x = x.split([cls_tokens, x.size(1) - cls_tokens], 1)
    output = torch.repeat_interleave(x, repeats=stride, dim=1)

    output = nn.functional.pad(output, (0, 0, 0, (target_len - cls_tokens) - output.shape[1], 0, 0))
    output = output[:, : target_len - cls_tokens]
    output = torch.cat([cls, output], dim=1)

    return output

# TODO: Decoder layer keep 1 layer with Q=1st block hidden, K,V= 3rd block hidden. This does better upsampling than current method


class TransformerDecoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
        config = copy.deepcopy(config)
        config.alternate_ffn = False
        self.config = config
        self.attention_structure = AttentionStructure(config)
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.layers = nn.ModuleList()
        if config.num_decoder_layers > 0:
            if config.block_repeats:
                self.layers.extend([TransformerLayer(config, 0, True, True, False, 0)])
                if config.light_last_layer:
                    self.layers.extend([LightLayer(config, 0, False)])
                    self.repeats = [config.num_decoder_layers - 1] + [1]
                else:
                    self.repeats = [config.num_decoder_layers]
            else:
                if config.light_last_layer:
                    self.layers.extend(
                        [TransformerLayer(config, 0, i == config.num_decoder_layers - 1, i == 0, False, i) for i in range(config.num_decoder_layers - 1)])
                    self.repeats = [1] * config.num_decoder_layers
                    self.layers = nn.ModuleList([LightLayer(config, 0, False)])
                else:
                    self.layers.extend(
                        [TransformerLayer(config, 0, i == config.num_decoder_layers - 1, i == 0, False, i) for i in range(config.num_decoder_layers)])
                    self.repeats = [1] * config.num_decoder_layers

        else:
            self.layers = None

        block_channel_size = self.config.block_channel_size

    def forward(
            self,
            final_hidden,
            first_block_hidden,
            position_embeds,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ):

        if not self.layers:
            hidden = final_hidden
            all_hidden_states = (hidden,) if output_hidden_states else None
            all_attentions = () if output_attentions else None
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)

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

        return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)


class DiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config: FastFormerConfig):
        super().__init__()
        self.config = config
        self.dense = Conv1d(config.block_channel_size[0], config.block_channel_size[0] // 2, 1, config.ffn_groups, bias=False) if config.ffn_groups > 1 else nn.Linear(config.block_channel_size[0], config.block_channel_size[0] // 2, bias=False)
        self.dense_prediction = (nn.Linear(config.block_channel_size[0] // 2, 1))
        self.act = checkpoint_wrapper(ACT2FN[self.config.hidden_act](), offload_to_cpu=False)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class FastFormerPreTrainedModel(PreTrainedModel):
    config_class = FastFormerConfig
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
        elif classname == "MultiheadAttention":
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            if hasattr(module, "c2p_bias"):
                nn.init.uniform_(module.c2p_bias, b=self.config.initializer_range)
            if hasattr(module, "p2c_bias"):
                nn.init.uniform_(module.p2c_bias, b=self.config.initializer_range)
        elif classname == "Embeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            nn.init.normal_(module.word_embeddings.weight, std=std)
            nn.init.normal_(module.position_embeddings.weight, std=std)
            if hasattr(module, "token_type_embeddings"):
                nn.init.normal_(module.token_type_embeddings.weight, std=std)
        elif classname == "FastAttention":
            if not hasattr(module, 'projection_matrix'):
                projection_matrix = module.create_projection(device=next(self.parameters()).device)
                module.register_buffer('projection_matrix', projection_matrix)


class FastFormerModel(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, tokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = fsdp_wrapper(TransformerDecoder(config))
        self.cls_tokens = config.num_highway_cls_tokens + 1

        block_channel_size = self.config.block_channel_size
        ffn_groups = self.config.ffn_groups
        self.final_hidden_fc = nn.Identity()
        if block_channel_size[0] != block_channel_size[-1]:
            if ffn_groups > 1:
                assert block_channel_size[0] % ffn_groups == 0 and block_channel_size[-1] % ffn_groups == 0
                self.final_hidden_fc = Conv1d(in_channels=block_channel_size[-1], out_channels=block_channel_size[0], kernel_size=1, groups=ffn_groups, bias=False)
            else:
                self.final_hidden_fc = nn.Linear(block_channel_size[-1], block_channel_size[0], bias=False)
            self.final_hidden_fc = fsdp_wrapper(nn.Sequential(self.final_hidden_fc, nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)))

        self.embed_proj_transpose = nn.Identity() if self.config.identity_preserving_norm else nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)
        if config.embedding_size != config.block_channel_size[0]:
            ep = nn.Linear(config.block_channel_size[0], config.embedding_size, bias=True)
            # ep.weight = nn.Parameter(self.embeddings.embed_proj.weight.transpose(0, 1))
            if self.config.identity_preserving_norm:
                self.embed_proj_transpose = ep
            else:
                self.embed_proj_transpose = nn.Sequential(nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps),
                                                          ep)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size + config.num_highway_cls_tokens)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            char_ids=None, char_offsets=None,
            run_decoder=True,
            run_answering=True,
            bypass_embeddings=False,
            **kwargs,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        assert input_shape[1] > 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if self.config.num_highway_cls_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_shape[0], self.config.num_highway_cls_tokens, device=device), attention_mask], dim=1)

        if bypass_embeddings:
            assert inputs_embeds is not None
            position_embeds = kwargs.pop("position_embeds", None)
            assert position_embeds is not None
        else:
            inputs_embeds, position_embeds = self.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets,)
        assert inputs_embeds.size(1) > 0

        encoder_outputs = self.encoder(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        final_hidden = self.final_hidden_fc(encoder_outputs[0])
        outputs = dict(final_hidden=final_hidden, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, position_embeds=position_embeds, input_shape=input_shape)

        if hasattr(self, "decoder") and (run_decoder or run_answering):

            decoder_outputs = self.decoder(
                final_hidden=final_hidden,
                first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
                position_embeds=position_embeds,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )
            outputs["decoder_outputs"] = decoder_outputs

        if run_answering:
            assert hasattr(self, "embed_proj_transpose")
            assert hasattr(self, "decoder")
            answering_hidden = self.embed_proj_transpose(decoder_outputs[0][:, self.cls_tokens: self.cls_tokens + 128])
            answering_logits = self.lm_head(answering_hidden)[:, :, :self.config.vocab_size]
            outputs["answering_logits"] = answering_logits
            outputs["answering_hidden"] = answering_hidden

        return outputs


class FastFormerForMaskedLM(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig):
        super().__init__(config)

        self.funnel = FastFormerModel(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.lm_dim_match = None
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.accuracy_hist = defaultdict(list)
        self.loss_ce = CrossEntropyLoss(ignore_index=config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            char_ids=None, char_offsets=None,
    ):

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            char_ids=char_ids, char_offsets=char_offsets,
        )

        last_hidden_state = outputs["decoder_outputs"][0]
        prediction_logits = None
        active_loss = attention_mask == 1

        masked_lm_loss = None
        if labels is not None:
            if self.lm_dim_match:
                last_hidden_state = self.funnel.embed_proj_transpose(last_hidden_state)
            prediction_logits = self.lm_head(last_hidden_state)
            loss_fct = self.loss_ce  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits[:, :, :self.config.vocab_size].view(-1, self.config.vocab_size), labels.view(-1))
            predictions = prediction_logits.detach().argmax(dim=-1)
            labels = (labels == predictions).float()
            self.accuracy_hist["lm"].append(float(labels[active_loss].float().mean()))
            self.accuracy_hist['lm_loss'].append(float(masked_lm_loss))

        output = (prediction_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output



from abc import ABC, abstractmethod


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss


def hook(grad):
    is_nan_inf = torch.logical_not(torch.isfinite(grad))
    if is_nan_inf.any():
        # print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Inf" % (get_time_string(), name_of_param))
        grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-2), grad)
        grad = torch.clamp_(grad, -1e1, 1e1)
        # grad = F.normalize(grad, 2, -1, eps=config.layer_norm_eps)
        # grad = grad / grad.norm(2, -1, True)
        return grad
    else:
        return None


class FastFormerForFusedELECTRAPretraining(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, model: FastFormerModel = None, tokenizer = None, aitm=False, alum=False,
                 adv_lm_w=1.0, adv_ascent_steps=1, aitm_clip_min=0.1, aitm_clip_max=0.9, adv_step_size=1e-3,
                 adv_epsilon=1e-2, aitm_noise_var=0.1, adv_w=1.0, alum_aitm_alternate=False,
                 input_cls_orthogonal_w=0.5, first_block_cls_orthogonal_w=0.1,
                 electra_loss_w=1.0, lm_loss_w=1.0, sentence_order_prediction_w=1.0, contrastive_w=1.0, contrastive_temperature=5e-2,
                answering_lm_w=1.0, highway_cls_ar_w=1.0, additive_margin_softmax_w=0.3):
        super().__init__(config)
        self.data_parallel = False
        self.config = config
        self.tokenizer = copy.deepcopy(tokenizer)
        self.funnel: FastFormerModel = FastFormerModel(config, tokenizer) if model is None else model
        self.cls_tokens = config.num_highway_cls_tokens
        self.discriminator_predictions = fsdp_wrapper(DiscriminatorPredictions(config))
        self.pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0
        if additive_margin_softmax_w == 0:
            self.ce = CrossEntropyLoss(ignore_index=-100)
            self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
            self.loss_bce = nn.BCEWithLogitsLoss()
        else:
            self.ce = AdMSoftmaxLoss(ignore_index=-100, m=additive_margin_softmax_w)
            self.loss_ce = AdMSoftmaxLoss(ignore_index=self.pad_token_id, m=additive_margin_softmax_w)
            self.loss_bce = BCELossFocal()
        if sentence_order_prediction_w > 0:
            self.sentence_order_prediction_w = sentence_order_prediction_w
            self.sent_predict_fc = nn.Linear(config.block_channel_size[0], (self.cls_tokens + 1))

        if highway_cls_ar_w > 0:
            assert config.position_biased_input
            self.sentence_task_attn = fsdp_wrapper(TransformerCrossAttentionDecoder(config))

        self.alum_aitm_alternate = alum_aitm_alternate
        self.lm_loss_w = lm_loss_w
        self.input_cls_orthogonal_w = input_cls_orthogonal_w
        self.first_block_cls_orthogonal_w = first_block_cls_orthogonal_w
        self.electra_loss_w = electra_loss_w
        self.reccord_loss = False
        self.record_accuracy = False
        self.timing_hist = list()
        self.aitm = aitm
        self.alum = alum
        self.aitm_noise_var = aitm_noise_var
        self.aitm_clip_min = aitm_clip_min
        self.aitm_clip_max = aitm_clip_max
        self.adv_lm_w = adv_lm_w
        self.adv_step_size = adv_step_size
        self.adv_epsilon = adv_epsilon
        self.adv_ascent_steps = adv_ascent_steps
        self.adv_w = adv_w
        self.answering_lm_w = answering_lm_w
        self.highway_cls_ar_w = highway_cls_ar_w
        self.contrastive_w = contrastive_w
        self.contrastive_temperature = contrastive_temperature
        self.additive_margin_softmax_w = additive_margin_softmax_w
        if alum_aitm_alternate:
            pass
        elif aitm:
            self.adv_lm_w = -1 * abs(self.adv_lm_w)
            assert not alum
        elif alum:
            assert not aitm
            self.adv_lm_w = abs(self.adv_lm_w)
            self.aitm_clip_min = self.aitm_clip_max = 1.0
        self.init_weights()

    def get_output_embeddings(self):
        return None

    def adv_project(self, grad, norm_type='inf', eps=1e-4):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def aitm_loss(self, funnel_inputs, funnel_outputs, mlm_predictions, mlm_correct,
                  sent_order_predictions, electra_predictions,
                  labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
                  contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None,
                  reverse_loss=False):
        funnel_inputs = dict(**funnel_inputs)
        funnel_inputs["input_embeds"] = funnel_outputs["input_embeds"]
        funnel_inputs["position_embeds"] = funnel_outputs["position_embeds"]
        funnel_inputs["bypass_embeddings"] = True
        active_loss = funnel_inputs["attention_mask"].bool()
        new_funnel_outputs = self.funnel(funnel_inputs)
        encoder_outputs = new_funnel_outputs["encoder_outputs"]
        first_block_hidden = encoder_outputs[2][self.config.block_sizes[0]]
        first_block_hidden = self.funnel.embed_proj_transpose(first_block_hidden[:, self.cls_tokens:][active_loss].contiguous())

        if self.adv_lm_w > 0:
            clip_min = clip_max = 1.0
        else:
            clip_min, clip_max = self.aitm_clip_min, self.aitm_clip_max
        lm_pre_kl = self.adv_lm_w * (
                    mlm_correct.float().clamp(clip_min, clip_max) * KL(first_block_hidden, mlm_predictions.detach(), reduction="none").sum(
                -1)).mean(0).sum()
        if reverse_loss:
            lm_kl_rev = self.adv_lm_w * (
                    mlm_correct.float().clamp(clip_min, clip_max) * KL(first_block_hidden.detach(), mlm_predictions, reduction="none").sum(
                -1)).mean(0).sum()
            lm_pre_kl = (lm_pre_kl + lm_kl_rev) / 2.0


        sent_order_pre_kl = 0
        if self.sentence_order_prediction_w > 0 and sent_order_predictions is not None:
            sent_order_block_hidden_cls = new_funnel_outputs["final_hidden"][:, 1:self.cls_tokens + 1] + new_funnel_outputs["final_hidden"][:, 0].unsqueeze(1)
            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls)
            sent_order_pre_kl = KL(sent_order_logits, sent_order_predictions.detach(), reduction="batchmean")
            if reverse_loss:
                sent_order_pre_kl = (sent_order_pre_kl + KL(sent_order_logits.detach(), sent_order_predictions, reduction="batchmean")) / 2.0

        decoder_outputs = new_funnel_outputs["decoder_outputs"]
        discriminator_sequence_output = decoder_outputs[0][:, self.cls_tokens:][active_loss].contiguous()
        electra_logits = self.discriminator_predictions(discriminator_sequence_output)
        electra_pre_kl = KL(electra_logits, electra_predictions.detach(), reduction="batchmean")
        if reverse_loss:
            electra_pre_kl = (electra_pre_kl + KL(electra_logits.detach(), electra_predictions, reduction="batchmean")) / 2.0

        contrastive_kl = 0.0
        if contrastive_anchors is not None:
            contrastive_block_hidden = new_funnel_outputs["final_hidden"][:, self.cls_tokens + 1:]

            dpow = self.config.stride ** 2
            contrastive_positives = recursive_op(contrastive_positives, lambda x: int(x / dpow))
            contrastive_anchors = recursive_op(contrastive_anchors, lambda x: int(x / dpow))

            anchors = [contrastive_block_hidden[anchor_batch_pos, [anchor[0], anchor[1]]].mean(0) for anchor_batch_pos, anchors in enumerate(contrastive_anchors)
                       for anchor in anchors]
            contrastive_positives = [[[*cp, batch_pos] for cp in anchor_cp] for batch_pos, anchors_cp in enumerate(contrastive_positives) for anchor_cp in
                                     anchors_cp]
            n_positives_per_anchor = max([len(a) for a in contrastive_positives])
            contrastive_positives = [[a[i]] for i in range(n_positives_per_anchor) for a in contrastive_positives if len(a) > 0]
            # contrastive_positives = torch.tensor(contrastive_positives).transpose(0, 1).tolist()

            positives = [contrastive_block_hidden[anchor_pos[-1], [anchor_pos[0], anchor_pos[1]]].mean(0) for pos in contrastive_positives for anchor_pos in pos]
            n_anchors = len(anchors)
            n_positives = len(positives)
            assert n_positives == 0 or n_anchors == 0 or n_positives % n_anchors == 0
            assert n_positives == 0 or n_anchors == 0 or (n_positives / n_anchors) == n_positives_per_anchor
            if n_positives == 0 or n_anchors == 0:
                pass
            else:
                contrastive_block_hidden = torch.stack(anchors + positives)
                if len(contrastive_block_hidden.size()) == 2:
                    contrastive_block_hidden = contrastive_block_hidden[:, :128]
                elif len(contrastive_block_hidden.size()) == 3:
                    contrastive_block_hidden = contrastive_block_hidden[:, :, :128]


                contrastive_block_hidden = contrastive_block_hidden / (contrastive_block_hidden.norm(2, -1, True) + self.config.layer_norm_eps)
                contrastive_block_matrix = contrastive_block_hidden.mm(contrastive_block_hidden.t()) / self.contrastive_temperature
                contrastive_block_matrix = contrastive_block_matrix * (1 - torch.eye(contrastive_block_matrix.size(0), device=contrastive_block_matrix.device))

                contrastive_kl = KL(contrastive_block_matrix, contrastive_logits.detach(), reduction="batchmean")
                if reverse_loss:
                    contrastive_kl = (contrastive_kl + KL(contrastive_block_matrix.detach(), contrastive_logits, reduction="batchmean")) / 2.0

        answering_lm_loss_kl = 0.0
        run_answering = labels_pet_input_ids is not None
        if run_answering:
            answering_hidden = new_funnel_outputs["answering_hidden"]
            answering_lm_loss_kl = KL(answering_hidden, funnel_outputs["answering_hidden"].detach(), reduction="batchmean")
            if reverse_loss:
                answering_lm_loss_kl = (answering_lm_loss_kl + KL(answering_hidden.detach(), funnel_outputs["answering_hidden"], reduction="batchmean")) / 2.0

            answering_lm_loss_kl = self.answering_lm_w * answering_lm_loss_kl

        return self.lm_loss_w * lm_pre_kl, self.sentence_order_prediction_w * sent_order_pre_kl, self.electra_loss_w * electra_pre_kl, answering_lm_loss_kl, self.contrastive_w * contrastive_kl

    def forward_for_aitm(self, funnel_inputs, funnel_outputs,
                         mlm_predictions, mlm_correct, sent_order_predictions, electra_predictions,
                         labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
                         contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None):
        if self.alum_aitm_alternate:
            self.adv_lm_w = -1 * self.adv_lm_w
        embed = funnel_outputs["input_embeds"]
        funnel_outputs = dict(**funnel_outputs)
        noise = embed.new(embed.size()).normal_(0, 1) * self.aitm_noise_var
        noise.requires_grad_()
        for _ in range(self.adv_ascent_steps):
            newembed = embed.detach() + noise
            funnel_outputs["input_embeds"] = newembed
            lm_pre_kl, sent_order_pre_kl, electra_pre_kl, answering_lm_pre_kl, contrastive_pre_kl = self.aitm_loss(funnel_inputs, funnel_outputs,
                                                                                                                   mlm_predictions, mlm_correct,
                                                                                                                   sent_order_predictions, electra_predictions,
                                                                                                                   labels_pet_input_ids,
                                                                                                                   labels_pet_attention_mask,
                                                                                                                   labels_pet_max_length,
                                                                                                                   contrastive_anchors, contrastive_positives, contrastive_logits)

            adv_loss = electra_pre_kl + sent_order_pre_kl + lm_pre_kl + answering_lm_pre_kl + contrastive_pre_kl
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0.0

            noise = noise + delta_grad * self.adv_step_size

        noise = self.adv_project(noise, eps=self.adv_epsilon)

        #
        newembed = embed + noise.detach()
        funnel_outputs["input_embeds"] = newembed
        lm_post_kl, sent_order_post_kl, electra_post_kl, answering_lm_post_kl, contrastive_post_kl = self.aitm_loss(funnel_inputs, funnel_outputs,
                                                                                                                    mlm_predictions, mlm_correct,
                                                                                                                    sent_order_predictions, electra_predictions,
                                                                                                                    labels_pet_input_ids,
                                                                                                                    labels_pet_attention_mask,
                                                                                                                    labels_pet_max_length,
                                                                                                                    contrastive_anchors, contrastive_positives,
                                                                                                                    contrastive_logits,
                                                                                                                    reverse_loss=True)
        adv_loss = sent_order_post_kl + electra_post_kl + (lm_post_kl ** 2) + answering_lm_post_kl + contrastive_post_kl
        return self.adv_w * adv_loss

    def get_emb(self, embedding_generation_params):
        input_ids, token_type_ids, inputs_embeds, char_ids, char_offsets = embedding_generation_params
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        inputs_embeds, position_embeds = self.funnel.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets, )
        return inputs_embeds, position_embeds, input_shape

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            labels_segment_index=None,
            char_ids=None, char_offsets=None,
            highway_cls_ar_input_ids=None, highway_cls_ar__attention_mask=None,
            labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
            contrastive_anchors=None, contrastive_positives=None,
            **kwargs

    ):
        all_inputs = dict(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          labels_segment_index=labels_segment_index,
                          char_ids=char_ids, char_offsets=char_offsets,
                          highway_cls_ar_input_ids=highway_cls_ar_input_ids, highway_cls_ar__attention_mask=highway_cls_ar__attention_mask,
                          labels_pet_input_ids=labels_pet_input_ids, labels_pet_attention_mask=labels_pet_attention_mask, labels_pet_max_length=labels_pet_max_length,
                          contrastive_anchors=contrastive_anchors, contrastive_positives=contrastive_positives,
                          **kwargs)

        timing_dict = list()
        accuracy_hist = defaultdict()
        record_accuracy = self.record_accuracy or kwargs.pop("record_accuracy", False)
        st = time.time()
        run_answering = labels_pet_input_ids is not None
        assert run_answering
        funnel_inputs = dict(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             inputs_embeds=inputs_embeds,
                             char_ids=char_ids, char_offsets=char_offsets,
                             run_decoder=True,
                             run_answering=run_answering, )

        assert attention_mask is not None
        tokenizer_attn_mask = attention_mask
        # with autocast(enabled=kwargs.pop("autocast", False)):
        funnel_outputs = self.funnel(**funnel_inputs)
        inputs_embeds = funnel_outputs["inputs_embeds"]
        inputs_embeds_cls = inputs_embeds[:, :self.funnel.cls_tokens]
        final_hidden = funnel_outputs["final_hidden"]
        at_cast = kwargs.pop("autocast", False)
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, input_ids = %s, attention_mask = %s" % (get_time_string(), random.sample(input_ids.reshape(-1).tolist(), 8), random.sample(attention_mask.reshape(-1).tolist(), 8)))
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, char_ids = %s, char_offsets = %s" % (get_time_string(), random.sample(char_ids.reshape(-1).tolist(), 8), random.sample(char_offsets.reshape(-1).tolist(), 8)))
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Input embeds = %s, Input embeds CLS = %s" % (get_time_string(), random.sample(inputs_embeds.reshape(-1).tolist(), 8), random.sample(inputs_embeds_cls.reshape(-1).tolist(), 8)))
        et = time.time() - st
        timing_dict.append(("prepare_encoder_input", et))
        encoder_outputs = funnel_outputs["encoder_outputs"]
        et = time.time() - st
        timing_dict.append(("encoder_outputs", et))
        answering_lm_loss = 0.0
        if run_answering:
            alen = labels_pet_input_ids.size(1)
            assert alen <= funnel_outputs["answering_logits"].size(1)
            answering_logits = funnel_outputs["answering_logits"][:, :alen]
            answering_lm_loss = self.answering_lm_w * self.loss_ce(answering_logits.reshape(-1, self.config.vocab_size), labels_pet_input_ids.reshape(-1))
            if record_accuracy and "answer" in kwargs:
                answering_predictions = answering_logits.detach().argmax(dim=-1)
                answering_predictions = answer_decoder(answering_predictions, self.tokenizer)
                final_labels, final_predictions = [], []
                for lbl, prd in zip(kwargs.pop("answer", None), answering_predictions):
                    if len(prd) > len(lbl):
                        prd = prd[:len(lbl)]
                    if len(prd) < len(lbl):
                        prd = prd + ([''] * (len(lbl) - len(prd)))
                    final_labels.extend(lbl)
                    final_predictions.extend(prd)
                score = accuracy_score(final_labels, final_predictions)
                accuracy_hist["answering_lm_accuracy"] = score

        first_block_hidden = encoder_outputs[2][self.config.block_sizes[0]]
        first_block_cls = first_block_hidden[:, :self.funnel.cls_tokens]
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, first_block_hidden = %s, first_block_cls CLS = %s" % (get_time_string(), random.sample(first_block_hidden.reshape(-1).tolist(), 8), random.sample(first_block_cls.reshape(-1).tolist(), 8)))
        et = time.time() - st
        timing_dict.append(("lm_logits", et))

        third_block_hidden = encoder_outputs[1][sum(self.config.block_sizes)]  # for last block both input and output shapes are same
        loss_contrastive = 0.0
        contrastive_block_matrix = None
        contrastive_anchors_copy = contrastive_positives_copy = None
        if contrastive_anchors is not None:
            bs = input_ids.size(0)
            if len(contrastive_positives) > bs and len(contrastive_positives) % bs == 0 and len(contrastive_positives)/bs == torch.cuda.device_count():
                did = torch.cuda.current_device()
                contrastive_positives = contrastive_positives[did*bs: (did+1)*bs]
                contrastive_anchors = contrastive_anchors[did * bs: (did + 1) * bs]

            contrastive_anchors_copy, contrastive_positives_copy = copy.deepcopy(contrastive_anchors), copy.deepcopy(contrastive_positives)
            contrastive_block_hidden = third_block_hidden[:, self.cls_tokens + 1:]
            assert len(contrastive_block_hidden.size()) == 3
            contrastive_block_hidden = contrastive_block_hidden[:, :, :128]
            dpow = self.config.stride ** 2
            contrastive_positives = recursive_op(contrastive_positives, lambda x: int(x / dpow))
            contrastive_anchors = recursive_op(contrastive_anchors, lambda x: int(x / dpow))
            # contrastive_len = contrastive_block_hidden.size(1)
            # for batch_positive in contrastive_positives:
            #     for positives_for_anchor in batch_positive:
            #         for positive in positives_for_anchor:
            #             if positive[0]==positive[1]:
            #                 if positive[1] < contrastive_len:
            #                     positive[1] = positive[1] + 1
            #                 elif positive[0] > 0:
            #                     positive[0] = positive[0] - 1
            #
            #             positive[0] = min(positive[0], positive[1] - 1)
            # for batch_anchor in contrastive_anchors:
            #     for anch in batch_anchor:
            #         if anch[0] == anch[1]:
            #             if anch[1] < contrastive_len:
            #                 anch[1] = anch[1] + 1
            #             elif anch[0] > 0:
            #                 anch[0] = anch[0] - 1
            # print("Anchors Batch size = %s, Input Batch Size = %s" % (len(contrastive_anchors), input_ids.size()))
            anchors = [contrastive_block_hidden[anchor_batch_pos, [anchor[0], anchor[1]]].mean(0) for anchor_batch_pos, anchors in enumerate(contrastive_anchors) for anchor in anchors]
            contrastive_positives = [[[*cp, batch_pos] for cp in anchor_cp] for batch_pos, anchors_cp in enumerate(contrastive_positives) for anchor_cp in anchors_cp]
            n_positives_per_anchor = max([len(a) for a in contrastive_positives])
            contrastive_positives = [[a[i]] for i in range(n_positives_per_anchor) for a in contrastive_positives if len(a) > 0]
            # contrastive_positives = torch.tensor(contrastive_positives).transpose(0, 1).tolist()

            positives = [contrastive_block_hidden[anchor_pos[-1], [anchor_pos[0], anchor_pos[1]]].mean(0) for pos in contrastive_positives for anchor_pos in pos]
            n_anchors = len(anchors)
            n_positives = len(positives)
            assert n_positives == 0 or n_anchors == 0 or n_positives % n_anchors == 0
            assert n_positives == 0 or n_anchors == 0 or (n_positives / n_anchors) == n_positives_per_anchor
            if n_positives == 0 or n_anchors == 0:
                pass
            else:
                contrastive_block_hidden = torch.stack(anchors + positives)
                contrastive_block_hidden = contrastive_block_hidden / (contrastive_block_hidden.norm(2, -1, True).detach() + self.config.layer_norm_eps)
                contrastive_block_matrix = contrastive_block_hidden.mm(contrastive_block_hidden.t()) / self.contrastive_temperature
                contrastive_block_matrix = contrastive_block_matrix * (1 - torch.eye(contrastive_block_matrix.size(0), device=contrastive_block_matrix.device))
                labels_contrastive = torch.tensor(list(range(n_anchors)) * n_positives_per_anchor, device=contrastive_block_matrix.device)
                loss_contrastive = self.ce(contrastive_block_matrix[n_anchors:], labels_contrastive)
                if record_accuracy:
                    accuracy_hist["contrastive_accuracy"] = ((contrastive_block_matrix[n_anchors:].detach().argmax(dim=-1) == labels_contrastive).sum().item() / n_positives)
                mask1 = torch.ones(n_anchors, contrastive_block_matrix.size(1), device=contrastive_block_hidden.device)
                mask2 = torch.zeros(n_anchors, contrastive_block_matrix.size(1), device=contrastive_block_hidden.device)
                const = 1e3
                for i in range(n_positives_per_anchor):
                    mask1[list(range(n_anchors)), torch.tensor(list(range(n_anchors))) + (n_anchors * (i + 1))] = -const
                vertical_lc = 0.0
                for i in range(n_positives_per_anchor):

                    labels_contrastive = torch.tensor(list(range(n_anchors)), device=contrastive_block_hidden.device) + (n_anchors * (i + 1))
                    mask_c = mask2.clone()
                    mask_c[list(range(n_anchors)), torch.tensor(list(range(n_anchors)), device=contrastive_block_hidden.device) + (n_anchors * (i + 1))] = const + 1
                    mask_c = mask1 + mask_c
                    l2 = self.ce(contrastive_block_matrix[:n_anchors] * mask_c, labels_contrastive)
                    vertical_lc += l2
                vertical_lc /= n_positives_per_anchor
                loss_contrastive += vertical_lc
            loss_contrastive = self.contrastive_w * loss_contrastive
        et = time.time() - st
        timing_dict.append(("contrastive_loss", et))
        cls_orthogonal_loss = 0.0
        if self.input_cls_orthogonal_w > 0 and self.training:
            inputs_embeds_cls = inputs_embeds_cls/(inputs_embeds_cls.norm(2, -1, True).detach() + self.config.layer_norm_eps)
            inputs_embeds_cls = inputs_embeds_cls.bmm(inputs_embeds_cls.transpose(1, 2))
            input_cls_orthogonal_loss = self.input_cls_orthogonal_w * (inputs_embeds_cls ** 2).mean()
            cls_orthogonal_loss += input_cls_orthogonal_loss

        if self.first_block_cls_orthogonal_w > 0 and self.training:
            first_block_cls = first_block_cls/(first_block_cls.norm(2, -1, True).detach() + self.config.layer_norm_eps)
            first_block_cls = first_block_cls.bmm(first_block_cls.transpose(1, 2))
            first_block_cls_orthogonal_loss = self.first_block_cls_orthogonal_w * (first_block_cls ** 2).mean()
            cls_orthogonal_loss += first_block_cls_orthogonal_loss

        et = time.time() - st
        timing_dict.append(("cls_orthogonal_loss", et))
        sentence_order_loss = 0.0
        highway_cls_ar_loss = 0.0
        sent_order_logits = None
        if self.sentence_order_prediction_w > 0 and labels_segment_index is not None:
            mx_labels = labels_segment_index.max(-1)[0].view(-1)
            first_cls = final_hidden[:, 0]
            labels_segment_index = labels_segment_index.view(-1)
            sent_order_block_hidden_cls = final_hidden[:, 1:self.cls_tokens + 1] + first_cls.unsqueeze(1)
            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls).view(-1, (self.cls_tokens + 1))
            sent_order_loss = self.loss_ce(sent_order_logits, labels_segment_index) + self.loss_ce(self.sent_predict_fc(first_cls), mx_labels)
            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, sent_order_block_hidden_cls = %s" % (get_time_string(), random.sample(sent_order_block_hidden_cls.reshape(-1).tolist(), 32)))
            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Logits and Labels SOP = %s" % (get_time_string(), list(zip(sent_order_logits.detach().reshape(-1, (self.cls_tokens + 1)).tolist(), labels_segment_index.reshape(-1).tolist()))[:4]))
            if record_accuracy:
                sent_order_out = sent_order_logits.detach().argmax(dim=-1) == labels_segment_index
                # self.accuracy_hist["sent_order"].append({"all": sent_order_out.detach().cpu(), "mean": float(sent_order_out.sum() / len(sent_order_out[labels_segment_index != 0].reshape(-1))), "alt_mean": float(sent_order_out[labels_segment_index != 0].float().mean().detach().cpu())})
                accuracy_hist["sent_order_accuracy"] = (float(sent_order_out[labels_segment_index != 0].detach().float().mean().cpu()))

            sentence_order_loss = self.sentence_order_prediction_w * sent_order_loss
        et = time.time() - st
        timing_dict.append(("sentence_order_loss", et))

        if self.highway_cls_ar_w > 0 and highway_cls_ar_input_ids is not None and self.config.num_highway_cls_tokens > 0:
            highway_block_hidden = self.funnel.embed_proj_transpose(final_hidden[:, :self.cls_tokens + 1])
            highway_cls_ar_inputs_embeds, _ = self.funnel.embeddings(shift_right(highway_cls_ar_input_ids, self.pad_token_id, self.pad_token_id), None, None, char_ids=None, char_offsets=None, use_embed_proj=False, use_highway_embeds=False)
            hshape = highway_cls_ar_inputs_embeds.size()
            assert hshape[1] > 0
            if hshape[1] <= 32:
                hshape2 = hshape[1]
            else:
                hshape2 = 32 * (hshape[1] // 32)
            assert hshape2 > 0
            key_attention = encoder_outputs[-1][2][:, :highway_block_hidden.size(1)]
            highway_cls_ar_inputs_embeds = highway_cls_ar_inputs_embeds[:, :hshape2]
            highway_cls_ar__attention_mask = highway_cls_ar__attention_mask[:, :hshape2]
            if hshape2 > 128:
                highway_cls_ar_inputs_embeds = highway_cls_ar_inputs_embeds.reshape(-1, hshape2 // 4, hshape[2])
                highway_cls_ar__attention_mask = highway_cls_ar__attention_mask.reshape(-1, hshape2 // 4)
                highway_block_hidden = torch.repeat_interleave(highway_block_hidden, repeats=4, dim=0)
                key_attention = torch.repeat_interleave(key_attention, repeats=4, dim=0)

            highway_cls_ar_out = self.sentence_task_attn(highway_cls_ar_inputs_embeds, highway_block_hidden, highway_block_hidden, highway_cls_ar__attention_mask, key_attention)
            # highway_cls_ar_out = self.sentence_task_attn(highway_cls_ar_out, highway_block_hidden, highway_block_hidden, highway_cls_ar__attention_mask, key_attention)

            highway_cls_ar_out = self.funnel.lm_head(highway_cls_ar_out)[:, :, :self.config.vocab_size]
            highway_cls_ar_input_ids = highway_cls_ar_input_ids[:, :hshape2]
            highway_cls_ar_out = highway_cls_ar_out.reshape(-1, self.config.vocab_size)
            highway_cls_ar_input_ids = highway_cls_ar_input_ids.reshape(-1)
            highway_cls_ar_loss = self.highway_cls_ar_w * self.loss_ce(highway_cls_ar_out, highway_cls_ar_input_ids)

            if record_accuracy:
                highway_cls_ar_out = highway_cls_ar_out.detach().argmax(dim=-1)
                # self.accuracy_hist["highway_cls_ar_sentence_outputs"].append({"actual": tokenizer.decode(highway_cls_ar_input_ids[0, 1:21].tolist()), "predictions": tokenizer.decode(highway_cls_ar_out[0, 1:21].tolist())})
                highway_cls_ar_out = highway_cls_ar_out[highway_cls_ar_input_ids != self.pad_token_id].reshape(-1) == highway_cls_ar_input_ids[highway_cls_ar_input_ids != self.pad_token_id].reshape(-1)
                accuracy_hist["highway_cls_ar_sentence_accuracy"] = (float(highway_cls_ar_out.detach().float().cpu().numpy().mean()))

        et = time.time() - st
        timing_dict.append(("highway_cls_ar_sentence_loss", et))
        active_loss = tokenizer_attn_mask.bool()
        first_block_hidden = self.funnel.embed_proj_transpose(first_block_hidden[:, self.cls_tokens:][active_loss].contiguous())
        prediction_logits = self.funnel.lm_head(first_block_hidden)[:, :self.config.vocab_size]
        active_labels = labels[active_loss].reshape(-1)
        active_prediction_logits = prediction_logits.reshape(-1, self.config.vocab_size)
        masked_lm_loss = self.lm_loss_w * self.loss_ce(active_prediction_logits, active_labels)
        labels = (active_labels == active_prediction_logits.detach().argmax(dim=-1)).detach().float()
        if record_accuracy:
            # predictions = prediction_logits.argmax(dim=-1)
            # self.accuracy_hist["lm_preds"].append({"predictions": "".join(self.tokenizer.decode(predictions[0, 1:21].tolist())), "actuals": "".join(self.tokenizer.decode(labels[0, 1:21].tolist()))})
            accuracy_hist["lm_accuracy"] = (float(labels.float().cpu().numpy().mean()))

        et = time.time() - st
        timing_dict.append(("lm_accuracy_loss", et))

        decoder_outputs = funnel_outputs["decoder_outputs"]

        et = time.time() - st
        timing_dict.append(("decoder_outputs", et))

        # cls_tokens = decoder_outputs[0][:, :self.cls_tokens + 1]
        # decoder_outputs = (decoder_outputs[0][:, self.cls_tokens + 1:], decoder_outputs[1:])
        discriminator_sequence_output = decoder_outputs[0][:, self.cls_tokens:][active_loss].contiguous()
        logits = self.discriminator_predictions(discriminator_sequence_output)
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, discriminator_sequence_output = %s, logits = %s" % (get_time_string(), random.sample(discriminator_sequence_output.reshape(-1).tolist(), 8), random.sample(logits.reshape(-1).tolist(), 8)))

        et = time.time() - st
        timing_dict.append(("electra_discriminator_logits", et))

        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Logits and Labels for electra = %s" % (get_time_string(), list(zip(active_logits.detach().tolist(), labels.tolist()))[:4]))

        loss = self.electra_loss_w * self.loss_bce(logits, labels)
        if record_accuracy:
            accuracy_hist["electra_accuracy"] = (torch.mean(((torch.sigmoid(logits.detach()) > 0.5).type(torch.int64) == labels).type(torch.float)).item())
            # if self.record_accuracy:
            #     self.accuracy_hist.append(accuracy_hist)

        et = time.time() - st
        timing_dict.append(("electra_discriminator_accuracy", et))

        electra_loss = loss

        et = time.time() - st
        adv_loss = torch.tensor(0.0)
        timing_dict.append(("aitm_alum_start", et))
        if (self.aitm or self.alum) and self.training:
            adv_loss = self.forward_for_aitm(funnel_inputs, funnel_outputs, first_block_hidden, labels, sent_order_logits, logits,
                                             labels_pet_input_ids, labels_pet_attention_mask, labels_pet_max_length,
                                             contrastive_anchors_copy, contrastive_positives_copy, contrastive_block_matrix)
            loss = loss + adv_loss

        et = time.time() - st
        timing_dict.append(("aitm_alum_end", et))

        loss = loss + masked_lm_loss + sentence_order_loss + answering_lm_loss + highway_cls_ar_loss + cls_orthogonal_loss + loss_contrastive
        et = time.time() - st
        timing_dict = [(k, 100 * (v/et)) for k, v in timing_dict]
        self.timing_hist.append(timing_dict)
        self.timing_hist = self.timing_hist[-100:]

        # TODO: return one loss
        # TODO: check various loss history and accuracies over time
        # TODO: Make a separate wrapper for AITM and ALUM vs making it here?

        # TODO: CLS correction needed

        loss_dict = dict()
        if record_accuracy:
            accuracy_hist = {k: v.detach() if hasattr(v, "detach") else v for k, v in accuracy_hist.items()}
            loss_dict = dict(masked_lm_loss=float(masked_lm_loss), sentence_order_loss=float(sentence_order_loss), answering_lm_loss=float(answering_lm_loss),
                             highway_cls_ar_loss=float(highway_cls_ar_loss), cls_orthogonal_loss=float(cls_orthogonal_loss),
                             loss_contrastive=float(loss_contrastive), adv_loss=float(adv_loss), electra_loss=float(electra_loss), loss=float(loss))
            loss_dict = {k: v.detach() if hasattr(v, "detach") else v for k, v in loss_dict.items()}
        results = dict(loss=loss, loss_dict=loss_dict, timing_dict=timing_dict, accuracy_hist=accuracy_hist)
        if self.data_parallel:
            results = [results]
        # TODO: return pet answer
        # decoder_output=decoder_outputs[0], decoder_cls=cls_tokens, encoder_output=encoder_outputs[0][:, self.cls_tokens + 1:], encoder_cls=encoder_outputs[0][:, :self.cls_tokens + 1], encoder_hidden_states=encoder_outputs[1])
        return results


if __name__ == "__main__":
    import time
    import argparse
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW

    torch.backends.cudnn.benchmark = True
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining, CTRLConfig, CTRLPreTrainedModel
    from transformers.models.deberta import DebertaModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default='cpu',
                    help="Device")
    ap.add_argument("--config", type=str, default='md_config',
                    help="Config")
    ap.add_argument("--texts", type=str, default='large_texts',
                    help="Text Set")
    ap.add_argument("--profile", type=str2bool, default=False)
    ap.add_argument("--sdconv", type=str2bool, default=False)
    ap.add_argument("--forward_only", type=str2bool, default=False)
    ap.add_argument("--fp16", type=str2bool, default=False)
    ap.add_argument("--aitm", type=str2bool, default=False)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", type=str, default='fastformer_fused_electra')  # fastformer_mlm, fastformer_electra, fastformer_fused_electra, fastformer, microsoft/deberta-base, roberta-base, distilroberta-base, funnel-transformer/intermediate

    args = vars(ap.parse_args())
    forward_only = args["forward_only"]
    device = args["device"]
    profile = args["profile"]
    fp16 = args["fp16"]
    model_name = args["model"]
    aitm = args["aitm"]
    sdconv = args["sdconv"]
    batch_size = args["batch_size"]
    length = args["length"]
    lr = args["lr"]
    texts = args["texts"]
    config = dict(md_config=md_config, sm_config=sm_config)[args["config"]]
    epochs = args["epochs"]
    if aitm:
        assert not forward_only and model_name == "fastformer_fused_electra"
    HuggingFaceModelClass = AutoModel if forward_only else AutoModelForMaskedLM

    small_max_length = 128
    medium_max_length = 512
    large_max_length = 1024
    very_large_max_length = 1536
    texts = dict(large_texts=large_texts, very_small_texts=very_small_texts, small_texts=small_texts, hetero_texts=hetero_texts)[texts]

    tokenizer = get_tokenizer("bert")
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = length
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens
    if model_name not in ["fastformer_mlm", "fastformer_electra", "fastformer_fused_electra", "fastformer"]:
        config.tokenizer_length = min(config.tokenizer_length, 512)
        config.max_position_embeddings = min(config.tokenizer_length, 512)
        config.num_highway_cls_tokens = 0
    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)
    if batch_size > len(texts):
        for _ in range(8):
            texts += texts
    dataset = SmallTextDataset(texts)
    config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
    dataset = TokenizerDataset(config, tokenizer, char_to_id,
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length),
                               # sentence_jumble_proba=((1024, 0.1),), word_noise_proba=((1024, 0.1),),
                               max_jumbling_span_length=2,
                               dataset=dataset)
    dataset.training = True

    if "fastformer" in model_name:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, prefetch_factor=8, num_workers=2)
        size_dicts_t = {128: batch_size, 256: batch_size, 512: batch_size, 768: batch_size, 1024: batch_size}
        pt_batch = next(custom_batching_fn(dataloader, size_dicts_t, True))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, prefetch_factor=2, num_workers=0)
        iter_dataloader = iter(dataloader)
        pt_batch = next(iter_dataloader)

    if "fastformer" in model_name:
        sm_pt_batch = dict(input_ids=pt_batch["input_ids"], attention_mask=pt_batch["attention_mask"],
                        char_offsets=pt_batch["char_offsets"], char_ids=pt_batch["char_ids"])
        if model_name == "fastformer_fused_electra":
            model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, adv_step_size=1e-3, lm_loss_w=5.0, electra_loss_w=1.0, highway_cls_ar_w=2.0,
                                                         aitm=aitm, alum=False, alum_aitm_alternate=False)
            sm_pt_batch = pt_batch
            assert not forward_only
        if model_name == "fastformer_mlm":
            model = FastFormerForMaskedLM(config)
        if model_name == "fastformer":
            model = FastFormerModel(config)

        pt_batch = sm_pt_batch

    else:
        pt_batch = dict(input_ids=pt_batch["input_ids"], attention_mask=pt_batch["attention_mask"])
        if "electra" in model_name:
            HuggingFaceModelClass = ElectraForPreTraining
        elif "deberta" in model_name:
            HuggingFaceModelClass = AutoModel
        model = HuggingFaceModelClass.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    # tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # model = AutoModelForMaskedLM.from_pretrained("funnel-transformer/small-base")
    # tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small-base")

    # model = AutoModelForMaskedLM.from_pretrained("google/electra-small-discriminator")
    # tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    # model = AutoModelForMaskedLM.from_pretrained("nlpaueb/legal-bert-small-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")

    # model = AutoModelForMaskedLM.from_pretrained("nreimers/BERT-Small-L-4_H-512_A-8")
    # tokenizer = AutoTokenizer.from_pretrained("nreimers/BERT-Small-L-4_H-512_A-8")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-8_H-256_A-4")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-8_H-256_A-4")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-768_A-12")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-768_A-12")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-512_A-8")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-512_A-8")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-256_A-4")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-256_A-4")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-4_H-768_A-12")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-4_H-768_A-12")

    # TODO: Test Longformer for long sequences as well.
    # TODO: test mobilebert

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Params = %s" % (numel(model) / 1_000_000))

    if hasattr(model, "funnel"):
        funnel = model.funnel
        model_parameters = list(filter(lambda p: p.requires_grad, funnel.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Funnel Params = %s" % (numel(funnel) / 1_000_000))
    print(model)
    # print(model.funnel.encoder.repeats if hasattr(model, "funnel") else "")

    model = model.eval()
    config = md_config

    if "google/electra-base-discriminator" in model_name:
        labels = torch.randint_like(pt_batch["input_ids"], 0, 2)
        labels = labels.to(device)
    else:
        labels = pt_batch["label_mlm_input_ids"] if "label_mlm_input_ids" in pt_batch else pt_batch["input_ids"]
        labels = labels.to(device)
    if "labels" in pt_batch:
        del pt_batch["labels"]
    print("Input Sizes = ", {k: v.size() if hasattr(v, "size") else len(v) for k, v in pt_batch.items()})

    device = torch.device(device)
    # torch.autograd.set_detect_anomaly(True)

    # checkpoint = torch.load("model/fastformer_checkpoint-step-3999.pth", map_location=str(device))
    # import torch.distributed as dist
    # from torch.nn.parallel import DistributedDataParallel as DDP
    #
    # dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://%s:%s" % ("127.0.0.1", "9999"))
    # ddp_model = DDP(model, device_ids=None, find_unused_parameters=True, bucket_cap_mb=5)
    # ddp_model.load_state_dict(checkpoint['model'])
    # model = ddp_model

    # checkpoint = torch.load("model/error-model.pth", map_location=str(device))
    # model.load_state_dict(checkpoint)
    pt_batch = torch.load("model/error-input.pth", map_location=str(device))
    labels = pt_batch.pop("labels", None)

    model = model.to(device)
    pt_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in pt_batch.items()}


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
    optimizer = AdamW(all_params, lr=lr, eps=1e-6, weight_decay=1e-2)
    torch.autograd.set_detect_anomaly(True)

    def get_unused_params(model):
        for name, params in model.named_parameters():
            if params.grad is None:
                print(name)

    def run():
        if not forward_only:
            if fp16:
                with autocast():
                    if isinstance(model, (AutoModel, DebertaModel)):
                        output = model(**pt_batch)
                        output = ((output['last_hidden_state'][:, 0] - random.random()).mean(),)
                    else:
                        output = model(**pt_batch, labels=labels)
                    loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                    scaler.scale(loss).backward()
                    get_unused_params(model)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if isinstance(model, (AutoModel, DebertaModel)):
                    output = model(**pt_batch)
                    output = ((output['last_hidden_state'][:, 0] - random.random()).mean(),)
                else:
                    output = model(**pt_batch, labels=labels)
                loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                loss.backward()
                get_unused_params(model)
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
            return pt_outputs


    if profile:
        import torch.autograd.profiler as profiler

        _ = [run() for _ in range(2)]
        with profiler.profile(record_shapes=True) as prof:
            _ = [run() for _ in range(epochs)]
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
    else:
        _ = [run() for _ in range(1)]
        times = []
        for _ in trange(epochs):
            st = time.time()
            _ = run()
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.4f, Lowest = %.4f, Highest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.max(times), np.std(times)))

    if not forward_only and hasattr(model, "accuracy_hist"):
        from pprint import pprint
        pprint({k: v[-5:] for k, v in model.accuracy_hist.items()})
        pprint({k: v[:5] for k, v in model.accuracy_hist.items()})
        if hasattr(model, "timing_hist"):
            pprint(model.timing_hist[-1])
            import pandas as pd
            th = pd.DataFrame([td for tm in model.timing_hist for td in tm], columns = ["step", "cumulative"])
            th = th.groupby("step", sort=False)[["cumulative"]].mean()
            timings = [0] + list(th["cumulative"].values)[:-1]
            th["differ"] = timings
            th["absolute"] = th["cumulative"] - th["differ"]
            th.drop(columns=["differ"], inplace=True)
            pprint(th)
        # pprint(model.loss_hist)
