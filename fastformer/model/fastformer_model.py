import copy
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
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
from transformers.activations import ACT2FN
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

from fastformer.data import very_large_texts, TokenizerDataset, collate_fn
from fastformer.data.sample_data import SmallTextDataset, small_texts, large_texts, very_large_texts, very_small_texts
from fastformer.model.AdMSLoss import AdMSoftmaxLoss, BCELossFocal

try:
    from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
except:
    pass

from fastformer.config import *

logger = logging.get_logger(__name__)

INF = 1e6
EPS = 1e-6

def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


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
            char_rnn_layers = config.char_rnn_layers
            char_rnn_vocab_size = config.char_rnn_vocab_size
            self.char_embeddings = nn.Embedding(char_rnn_vocab_size, self.embedding_size // 4, padding_idx=pad_token_id)
            self.char_rnn = ShortSeqRNN(config, self.embedding_size // 4, 1, self.embedding_size // 4,
                                        config.char_rnn_window_size, config.char_rnn_window_overlap, char_rnn_layers)

        self.embed_proj = nn.Identity()
        self.char_embed_proj = nn.Linear(self.embedding_size // 4, hidden_size, bias = False)
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

    def forward(self, input_ids=None, input_embeds=None, token_type_ids=None, position_ids=None, mask=None, char_ids=None, char_offsets=None, use_position_embeddings=True):
        if input_embeds is None:
            input_shape = input_ids.size()
            input_shape = list(input_shape)
            initial_seq_len = input_shape[1]
            input_shape[1] = input_shape[1] + self.config.num_highway_cls_tokens
            input_shape = tuple(input_shape)

            inputs_embeds = self.word_embeddings(input_ids)
            if self.config.num_highway_cls_tokens > 0:
                highway_embeddings = self.word_embeddings(self.highway_cls_tokens).expand((inputs_embeds.size(0), -1, -1))
                inputs_embeds = torch.cat((highway_embeddings, inputs_embeds), dim=1)
            char_embeds = None
            if self.config.char_rnn and char_ids is not None:
                char_offsets = char_offsets.flatten(1, 2).unsqueeze(-1).expand(input_shape[0], -1, self.embedding_size // 4)
                char_embeds = self.char_rnn(self.char_embeddings(char_ids))
                char_embeds = torch.gather(char_embeds, 1, char_offsets).view(input_shape[0], initial_seq_len, 2, self.embedding_size // 4).mean(2)
                if self.config.num_highway_cls_tokens > 0:
                    char_embeds = torch.cat((highway_embeddings[:, :, :char_embeds.size(-1)], char_embeds), dim=1)
                char_embeds = self.char_embed_proj(char_embeds)


        else:
            input_shape = input_embeds.size()
        seq_length = input_shape[1]

        embeddings = inputs_embeds
        position_embeddings = None
        if use_position_embeddings:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            else:
                position_ids = torch.cat((self.highway_position_ids.expand((position_ids.size(0), -1)), position_ids + self.config.num_highway_cls_tokens), dim=1)

            position_embeddings = self.position_embeddings(position_ids.long())

            if self.position_biased_input:
                # print("Seq len = ", seq_length, "embeddings dim = ", embeddings.size(), position_embeddings.size(), self.position_ids.size())
                embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            else:
                token_type_ids = torch.cat(
                    (torch.empty(input_shape[0], self.config.num_highway_cls_tokens, device=token_type_ids.device).fill_(token_type_ids[0][0]), token_type_ids),
                    dim=1)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embed_proj:
            embeddings = self.embed_proj(embeddings)
        if char_embeds is not None:
            embeddings = embeddings + char_embeds

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = torch.cat((torch.ones(mask.size(0), self.config.num_highway_cls_tokens, dtype=mask.dtype, device=mask.device), mask), dim=1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
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
    cls_tokens = tensor[:, :cls_size]
    tensor = tensor[:, cls_size:]
    tensor = pool_tensor_basic(tensor, mode, stride)

    tensor = torch.cat([cls_tokens, tensor.squeeze(1)], dim=1)
    tensor = tensor[:, :2 * (tensor.size(1) // 2)]
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
        if kernel_size == 1 and stride == 1 and dilation == 1 and False:
            self.conv = nn.Linear(in_channels, out_channels, bias=bias)
            self.pre_permute = False
            self.post_permute = False
        else:
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
        self.act = ACT2FN[act]
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
    def __init__(self, config: FastFormerConfig, hidden_size, heads, head_size, kernel_size, overlap, layers=1):
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
        self.gru = nn.ModuleList()
        for i in range(heads):
            self.gru.append(nn.RNN(hidden_size // self.heads, hidden_size // (2 * self.heads), layers,
                                   nonlinearity="tanh",
                                   bias=False, batch_first=True, dropout=0.0, bidirectional=True))
        # TODO: should we try to also put a linear layer after rnn and make rnn hidden size larger?

    def forward(self, query, key=None, value=None):
        st = time.time()
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
            seg_end = (i + 1) * self.kernel_size + 2 * self.overlap
            seg = query[:, seg_start:seg_end]
            segs.append(seg)
            segments.append((seg_start, seg_end, seg_end - 2 * self.overlap, seg_end - seg_start,))

        query = torch.stack(segs, 0).transpose(0, 1)
        query = query.reshape(-1, query.shape[2], query.shape[3])
        query = query.view(query.shape[0], query.shape[1], self.heads, -1)

        query = query.permute(2, 0, 1, 3)
        processed_query = []
        stg = time.time()
        for i in range(query.size(0)):
            # print("Short Seq RNN sizes = ", query[i].size(), query.size())
            qp = self.gru[i](query[i])[0]
            processed_query.append(qp)
        query = torch.stack(processed_query, 0)
        query = query.permute(1, 2, 0, 3).reshape(-1, query.shape[2], dim)

        # query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        # query = self.gru(query)[0]
        # query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        query = query[:, self.overlap:-self.overlap]
        query = query.reshape(bs, -1, dim)[:, :seqlen]

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        et = time.time()
        # print("ShortSeqRNN timing, Overall = %.5f" % (et - st), "Only Gru = %.5f" % (et - stg), query.size())
        return query


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None):
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
        self.act = ACT2FN[act]
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
            self.expand = Conv1d(d_model, d_model * 2, 1, n_head, False)
            self.contract = nn.Sequential(Conv1d(d_model * 2, d_model, 1, n_head, False), nn.LayerNorm(d_model, config.layer_norm_eps))
            self.expansion_factor = 2
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
        query = self.expand(query)
        cls = query[:, :self.cls_tokens]
        query = query[:, self.cls_tokens:]
        target_len = qskip.shape[1] - self.cls_tokens
        query = self.sd_conv(query)
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
            self.expand = Conv1d(d_model, d_model * 2, 1, n_head, False)
            self.contract = nn.Sequential(Conv1d(d_model * 2, d_model, 1, n_head, False), nn.LayerNorm(d_model, config.layer_norm_eps))
            self.expansion_factor = 2
        else:
            self.expand = nn.Identity()
            self.contract = nn.LayerNorm(d_model, config.layer_norm_eps) if use_in_funnel else nn.Identity()
            self.expansion_factor = 1

    def forward(self, query):
        st = time.time()
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)
        query = self.expand(query)
        query = self.contract(pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention))
        ext = time.time()
        # print("Mean pooling = %.5f" % (ext - st))
        return qskip + query


class CompressSeqShortSeqRNN(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.compressed_query_attention = config.stride if use_in_funnel else config.compressed_query_attention_stride
        expand_dims = config.expand_dim_before_pooling if use_in_funnel else False
        if expand_dims:
            self.expand = Conv1d(d_model, d_model * 2, 1, n_head, False)
            self.contract = nn.Sequential(Conv1d(d_model * 2, d_model, 1, n_head, False), nn.LayerNorm(d_model, config.layer_norm_eps))
            self.expansion_factor = 2
        else:
            self.expand = nn.Identity()
            self.contract = nn.LayerNorm(d_model, config.layer_norm_eps) if use_in_funnel else nn.Identity()
            self.expansion_factor = 1
        d_head = config.d_head[block_index] * self.expansion_factor
        self.d_model, self.n_head, self.d_head = d_model * self.expansion_factor, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.rnn = ShortSeqRNN(config, d_model * self.expansion_factor, 1, d_model * self.expansion_factor, config.short_rnn_kernel[block_index], config.short_rnn_overlap[block_index])

    def forward(self, query):
        st = time.time()
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)
        query = self.expand(query)
        rnnst = time.time()
        query = self.rnn(query)
        rnnet = time.time()
        query = self.contract(pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention))
        ext = time.time()
        # print("RNN pooling Total = %.5f" % (ext - st), "Only RNN in pooling = %.5f" % (rnnet - rnnst), "RNN Extra = %.5f" % ((ext - st) - (rnnet - rnnst)))
        return query + qskip


class CompressSeqWeighted(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, d_model, n_head, use_in_funnel=False):
        super().__init__()
        self.config = config
        self.stride = config.stride if use_in_funnel else config.compressed_query_attention_stride
        d_head = config.d_head[block_index]
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_head
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.contract = nn.LayerNorm(d_model, config.layer_norm_eps) if use_in_funnel else nn.Identity()
        self.ffn = BertFFN(config, self.stride * d_model // n_head,
                           2 * self.stride * d_model // n_head, 0, self.stride)

    def forward(self, query):
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.stride)
        cls = query[:, :self.cls_tokens]
        query = query[:, self.cls_tokens:]
        target_len = qskip.shape[1] - self.cls_tokens
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
        return qskip + self.contract(q)


class ChannelCompress(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.stride = in_dim // out_dim
        assert in_dim % out_dim == 0

    def forward(self, x):
        return pool_tensor_basic(x.transpose(1, 2), "mean", self.stride).transpose(1, 2)


class MultiheadAttention(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                 last_layer_index=None):
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

        if config.compress_query_method == 'learn':
            CompressionClass = CompressSeqWeighted
        elif config.compress_query_method == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.compress_query_method == 'learn_rnn':
            CompressionClass = CompressSeqShortSeqRNN
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
        qkv_squeeze = self.config.qkv_squeeze_fraction > 1
        sq_frac = self.config.qkv_squeeze_fraction
        if qkv_squeeze:
            assert d_model % sq_frac == 0

        qkv_transform_groups = self.config.qkv_transform_groups
        if qkv_transform_groups > 1:
            # assert n_head % qkv_transform_groups == 0 and n_head >= qkv_transform_groups
            self.q_head = ConvFFN(config, d_model, d_model // sq_frac, d_out=n_head * d_head, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)
            self.k_head = ConvFFN(config, d_model, d_model // sq_frac, d_out=n_head * d_head, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups)

            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)


            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = ConvFFN(config, d_model, d_model // sq_frac, d_out=d_model, groups=qkv_transform_groups) if qkv_squeeze else Conv1d(
                    in_channels=d_model, out_channels=d_model, kernel_size=1, groups=qkv_transform_groups)

        else:
            self.q_head = BertFFN(config, d_model, d_model // sq_frac, d_out=n_head * d_head) if qkv_squeeze else nn.Linear(d_model, n_head * d_head,
                                                                                                                            bias=False)
            self.k_head = BertFFN(config, d_model, d_model // sq_frac, d_out=n_head * d_head) if qkv_squeeze else nn.Linear(d_model, n_head * d_head)

            if compress_query:
                self.q_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if compress_key:
                self.k_head_compress = CompressionClass(config, block_index, d_model, n_head)

            if config.no_v_head:
                self.v_head = nn.Identity()
            else:
                self.v_head = BertFFN(config, d_model, d_model // sq_frac, d_out=d_model) if qkv_squeeze else nn.Linear(d_model, d_model)

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
        self.conv1d_in = Conv1d(in_channels=cin, out_channels=d_inner, kernel_size=1, groups=groups)
        self.activation_dropout = Dropout(config.activation_dropout)
        self.layers = nn.ModuleList() if layers > 0 else None
        for _ in range(layers):
            self.layers.append(Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=1, groups=groups))
        self.conv1d_out = Conv1d(in_channels=d_inner, out_channels=cout, kernel_size=1, groups=groups)

        self.dropout = Dropout(config.hidden_dropout)
        self.act = ACT2FN[act]

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
        self.linear_1 = nn.Linear(d_model, d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = Dropout(config.activation_dropout)
        d_out = d_model if d_out is None else d_out
        self.linear_2 = nn.Linear(d_inner, d_out)
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
        self.activation_function = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(d_model, config.layer_norm_eps)
        if self.need_dim_match:
            self.dlayer_norm = nn.LayerNorm(d_next, config.layer_norm_eps)
            self.dlin = Conv1d(d_model, self.diff, 1, 4) if d_model % 4 == 0 and self.diff % 4 == 0 else nn.Linear(d_model, self.diff)
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
        pre_ffn = h
        if dim_match:
            hidden = nn.functional.pad(hidden, (0, self.diff, 0, 0, 0, 0))
            h = torch.cat((h, self.dlin(h)), 2)
            if self.config.identity_preserving_norm:
                h = hidden + self.dlayer_norm(h)
            else:
                h = self.dlayer_norm(hidden + h)
        else:
            if self.config.identity_preserving_norm:
                h = hidden + self.layer_norm(h)
            else:
                h = self.layer_norm(hidden + h)
        return pre_ffn, h


class LightLayer(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_encoder_layer):
        self.config = config
        cin = config.block_channel_size[block_index]
        cout = cin // 2
        self.is_encoder_layer = is_encoder_layer
        super().__init__()
        self.c1 = Conv1d(in_channels=cout, out_channels=cout, kernel_size=5, groups=sum(config.n_head[block_index]) // 2, padding=2, padding_mode='zeros')
        self.layer_norm = nn.LayerNorm(cout * 2, config.layer_norm_eps)
        self.activation_function = ACT2FN[config.hidden_act]
        self.cls_tokens = config.num_highway_cls_tokens + 1
        d_head = config.d_head[block_index]
        assert cout % (sum(config.n_head[block_index]) // 2) == 0
        self.c1 = SDConv(config, cout, sum(config.n_head[block_index]) // 2, cout // (sum(config.n_head[block_index]) // 2), config.sdconv_kernel_size[0])
        self.rnn = nn.RNN(cout, cout // 2, 1, nonlinearity="tanh",
                          bias=False, batch_first=True, dropout=0.0, bidirectional=True)
        # self.rnn = ShortSeqRNN(config, cout, 1, cout, config.short_rnn_kernel[block_index],
        #                        config.short_rnn_overlap[block_index])
        self.lin = nn.Linear(cin, cin)
        self.cout = cout
        # padding

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        qcnn = self.c1(query[:, :, :self.cout])
        qrnn = self.rnn(query[:, :, self.cout:])[0]
        q = torch.cat((qcnn, qrnn), 2)
        q = self.activation_function(q)
        q = self.lin(q)
        if self.config.identity_preserving_norm:
            res = query + self.layer_norm(q)
        else:
            res = self.layer_norm(query + q)
        return (res, res)


class TransformerLayer(nn.Module):
    def __init__(self, config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index, last_layer_index=None):
        super().__init__()
        self.attention = MultiheadAttention(config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                                            last_layer_index)
        self.ffn = PositionwiseFFN(config, block_index, is_last_layer_of_block, is_encoder_layer)

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        attn = self.attention(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
        pre_ffn, output = self.ffn(attn[0], layer_index)
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
                        self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
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
                pool[str(block_index + 1)] = CompressionClass(config, bi, config.block_channel_size[bi], sum(config.n_head[bi]), use_in_funnel=True)
            self.pool = pool

    def forward_one_block(self, block_index, hidden, attention_inputs,
                          all_hidden_states, pre_ffn_states, all_attentions,
                          output_attentions=False, output_hidden_states=False):
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

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

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

        self.config = config
        block_index = 0
        all_head = config.n_head[block_index]
        total_heads = sum(all_head)
        config.n_head[block_index] = (total_heads,) + config.n_head[block_index][1:]
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.self_attn = MultiheadAttention(config, block_index, False, True, False, 0)
        self.self_attn_lin = nn.Linear(config.block_channel_size[block_index], config.block_channel_size[block_index])
        self.self_attn_ln = nn.LayerNorm(config.block_channel_size[block_index], config.layer_norm_eps)
        self.cross_attn = MultiheadAttention(config, block_index, False, True, False, 0)
        self.ffn = PositionwiseFFN(config, block_index, False, False)

    def forward(self, query, key, value, query_padding_mask, key_padding_mask, query_mask=None, key_mask=None):
        bs, seq_len, dim = query.shape
        query_temp = query
        if query_mask is None:
            query_mask = subsequent_mask(seq_len).to(query.device).unsqueeze(0)

        query_padding_mask = query_padding_mask[:, None, None].expand(bs, 1, seq_len, seq_len)

        if key_mask is None:
            key_mask = key_padding_mask

        (query, ) = self.self_attn(query, query, query, (None, torch.logical_and(query_mask, query_padding_mask).type(query_padding_mask.dtype)), 0, False)
        query = self.self_attn_ln(self.self_attn_lin(query))
        query = query_temp + query
        (query,) = self.cross_attn(query, key, value, (None, torch.logical_and(key_mask, key_padding_mask).type(key_padding_mask.dtype)), 0, False)
        _, query = self.ffn(query)
        return query


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

# TODO: Decoder layer keep 1 layer with Q=1st block hidden, K,V= 3rd block hidden. This does better upsampling than current method


class TransformerDecoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
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
        self.decoder_ln = nn.LayerNorm(block_channel_size[0], config.layer_norm_eps)
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
    ):
        if self.final_hidden_fc:
            final_hidden = self.final_hidden_fc(final_hidden)
        final_hidden = final_hidden[:, :, :first_block_hidden.shape[-1]]
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
        hidden = self.decoder_ln(hidden)

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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.block_channel_size[0], config.block_channel_size[0])
        self.dense_prediction = nn.Linear(config.block_channel_size[0], 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
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


@dataclass
class PreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FastFormerModel(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.answering_ffn = nn.Sequential(Conv1d(config.block_channel_size[-1], config.block_channel_size[0], 1, config.ffn_groups, bias=False), nn.GELU(), nn.Linear(config.block_channel_size[0], config.embedding_size, bias=False))
        self.answering_ffn[2].weight = nn.Parameter(self.embeddings.embed_proj.weight.transpose(0, 1))
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
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
            output_attentions=None,
            output_hidden_states=None,
            char_ids=None, char_offsets=None,
            run_decoder=True,
            run_answering=True,
            run_auto_regressive=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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

        inputs_embeds, position_embeds = self.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets,)

        encoder_outputs = self.encoder(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        answering = ()
        if run_answering and hasattr(self, "answering_ffn"):
            answering_hidden = self.answering_ffn(encoder_outputs[0][:, self.cls_tokens - 1:])
            answering_logits = self.lm_head(answering_hidden)[:, :, :self.config.vocab_size]
            answering_predictions = answering_logits.argmax(dim=-1)
            answering += (answering_logits, answering_predictions,)

        if hasattr(self, "decoder") and run_decoder:

            decoder_outputs = self.decoder(
                final_hidden=encoder_outputs[0],
                first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
                position_embeds=position_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            encoder_cls_tokens = encoder_outputs[0][:, :self.cls_tokens]
            encoder_outputs = (encoder_outputs[0][:, self.cls_tokens - 1:], encoder_outputs[1:])
            cls_tokens = decoder_outputs[0][:, :self.cls_tokens]
            decoder_outputs = (decoder_outputs[0][:, self.cls_tokens - 1:], decoder_outputs[1:])
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[0], encoder_outputs[1] + decoder_outputs[1],)
            if output_attentions:
                outputs = outputs + (encoder_outputs[3] + decoder_outputs[2],)
        else:
            encoder_cls_tokens = encoder_outputs[0][:, :self.cls_tokens]
            encoder_outputs = (encoder_outputs[0][:, self.cls_tokens - 1:], encoder_outputs[1:])
            outputs = (encoder_outputs[0],)
            cls_tokens = encoder_cls_tokens
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[0], encoder_outputs[1],)
            if output_attentions:
                outputs = outputs + (encoder_outputs[3],)

        outputs += (encoder_cls_tokens, cls_tokens,)
        outputs += answering
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
        if config.embedding_size != config.block_channel_size[0]:
            self.lm_dim_match = nn.Linear(config.block_channel_size[0], config.embedding_size)

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

        last_hidden_state = outputs[0]
        prediction_logits = None
        input_shape = input_ids.size()
        active_loss = attention_mask == 1

        masked_lm_loss = None
        if labels is not None:
            if self.lm_dim_match:
                last_hidden_state = self.lm_dim_match(last_hidden_state)
            prediction_logits = self.lm_head(last_hidden_state)
            loss_fct = self.loss_ce  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits[:, :, :self.config.vocab_size].view(-1, self.config.vocab_size), labels.view(-1))
            predictions = prediction_logits.argmax(dim=-1)
            labels = (labels == predictions).float()
            self.accuracy_hist["lm"].append(float(labels[active_loss].float().mean()))

        output = (prediction_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output



from abc import ABC, abstractmethod


class FastFormerForELECTRAPretraining(FastFormerPreTrainedModel):
    def __init__(self, config_generator: FastFormerConfig, config_discriminator: FastFormerConfig,
                 generator: FastFormerModel = None, discriminator: FastFormerModel=None, tokenizer = None,
                 electra_loss_w=1.0, lm_loss_w=1.0,
                 alum=False):
        super().__init__(config)
        assert config_discriminator.embedding_size == config_generator.embedding_size
        assert config_discriminator.vocab_size == config_generator.vocab_size
        assert config_discriminator.block_channel_size[0] == config_generator.block_channel_size[0]
        self.tokenizer = tokenizer
        self.config = config_discriminator
        self.funnel: FastFormerModel = FastFormerModel(config_discriminator) if discriminator is None else discriminator
        self.lm_head = nn.Linear(config_generator.embedding_size, config.vocab_size)
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.discriminator_predictions = DiscriminatorPredictions(config)
        self.loss_ce = CrossEntropyLoss(ignore_index=config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0)
        self.generator = FastFormerModel(config_generator) if generator is None else generator
        assert self.generator.config.embedding_size == self.funnel.config.embedding_size
        if self.generator.config.embedding_size != self.generator.config.block_channel_size[0]:
            self.lm_dim_match = nn.Linear(self.generator.config.block_channel_size[0], self.generator.config.embedding_size)
        self.funnel.embeddings = self.generator.embeddings
        self.lm_loss_w = lm_loss_w
        self.electra_loss_w = electra_loss_w
        self.loss_hist = defaultdict(list)
        self.accuracy_hist = defaultdict(list)
        self.init_weights()

    def get_input_embeddings(self):
        return self.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None, char_ids=None, char_offsets=None,):

        # TODO: can do forward pass of embedding layer once instead of twice?
        # TODO: can share the full 1st block instead of just embedding? Is this similar to fused ELECTRA

        outputs = self.generator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            char_ids=char_ids, char_offsets=char_offsets,
            output_attentions=False,
            output_hidden_states=False,
        )
        input_shape = input_ids.size()

        last_hidden_state = outputs[0]
        if self.lm_dim_match:
            last_hidden_state = self.lm_dim_match(last_hidden_state)
        prediction_logits = self.lm_head(last_hidden_state)
        loss_fct = self.loss_ce
        masked_lm_loss = self.lm_loss_w * loss_fct(prediction_logits[:, :, :self.config.vocab_size].view(-1, self.config.vocab_size), labels.view(-1))
        predictions = prediction_logits.argmax(dim=-1)
        labels = (labels == predictions).float()
        mlm_positions = input_ids == self.tokenizer.mask_token_id
        self.accuracy_hist["lm"].append(float(labels[active_loss].sum() / len(labels[active_loss].view(-1))))
        self.accuracy_hist["masked_lm"].append(float(labels[mlm_positions].sum() / len(labels[mlm_positions].view(-1))))

        outputs = self.funnel(input_ids=predictions, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False, output_hidden_states=True,)
        discriminator_sequence_output = outputs[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)

        active_loss = attention_mask.view(-1, input_shape[1]) == 1
        loss_fct = nn.BCEWithLogitsLoss()
        active_logits = logits.view(-1, input_shape[1])[active_loss]
        active_labels = labels[active_loss]
        loss = self.electra_loss_w * loss_fct(active_logits, active_labels)

        self.accuracy_hist["electra"].append(torch.mean((torch.sigmoid(active_logits) > 0.5).type(torch.int64) == active_labels).item())
        self.loss_hist["electra_loss"].append(float(loss))
        self.loss_hist["masked_lm_loss"].append(float(masked_lm_loss))

        results = dict(loss=loss + masked_lm_loss,
                       decoder_output=outputs[0], decoder_cls=outputs[-1],
                       encoder_output=outputs[1], encoder_cls=outputs[-2],
                       encoder_hidden_states=outputs[2])
        return results


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss


class FastFormerForFusedELECTRAPretraining(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, model: FastFormerModel = None, tokenizer = None, aitm=False, alum=False,
                 adv_lm_w=1.0, adv_ascent_steps=1, aitm_clip_min=0.1, aitm_clip_max=0.9, adv_step_size=1e-3,
                 adv_epsilon=1e-2, aitm_noise_var=0.1, adv_w=1.0, alum_aitm_alternate=False,
                 input_cls_orthogonal_w=0.5, first_block_cls_orthogonal_w=0.1, second_block_cls_orthogonal_w=0.05, third_block_cls_orthogonal_w=0.0,
                 electra_loss_w=1.0, lm_loss_w=1.0, sentence_order_prediction_w=1.0, word_order_prediction_w=1.0, contrastive_w=1.0, contrastive_temperature=5e-2,
                 gap_sentence_prediction_w=1.0, answering_lm_w=1.0, highway_cls_ar_w=1.0, additive_margin_softmax_w=0.3):
        super().__init__(config)

        self.config = config
        self.tokenizer = tokenizer
        self.funnel: FastFormerModel = FastFormerModel(config) if model is None else model
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.cls_tokens = config.num_highway_cls_tokens
        self.discriminator_predictions = DiscriminatorPredictions(config)
        self.contrastive_ffn = ConvFFN(config, config.block_channel_size[-1], config.block_channel_size[-1], config.ffn_groups, 0, config.block_channel_size[0])
        self.pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0
        self.ce = CrossEntropyLoss(ignore_index=-100)
        if additive_margin_softmax_w == 0:
            self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
            self.loss_bce = nn.BCEWithLogitsLoss()
        else:
            self.loss_ce = AdMSoftmaxLoss(ignore_index=self.pad_token_id)
            self.loss_bce = BCELossFocal()
        self.lm_dim_match = nn.Linear(config.block_channel_size[0], config.embedding_size, bias=False)
        self.lm_dim_match.weight = nn.Parameter(self.funnel.embeddings.embed_proj.weight.transpose(0, 1))
        if sentence_order_prediction_w > 0:
            self.sentence_order_prediction_w = sentence_order_prediction_w
            self.order_prediction_fc = nn.Identity()
            self.sent_predict_fc = nn.Linear(config.block_channel_size[-1], self.cls_tokens)
            self.two_sent_order_fc = Conv1d(config.block_channel_size[-1], 1, 2, 1)
            self.three_sent_order_fc = Conv1d(config.block_channel_size[-1], 1, 3, 1, dilation=2)

        if word_order_prediction_w > 0 or gap_sentence_prediction_w > 0 or highway_cls_ar_w > 0:
            assert config.position_biased_input
            self.word_gap_prediction_fc = nn.Sequential(Conv1d(self.config.block_channel_size[1], self.config.block_channel_size[0], kernel_size=1, groups=4),
                                                        nn.LayerNorm(self.config.block_channel_size[0], config.layer_norm_eps))
            self.initiator_emb = nn.Embedding(4, self.config.block_channel_size[0])
            self.sentence_task_attn = TransformerCrossAttentionDecoder(config)
            self.word_order_prediction_w = word_order_prediction_w
            self.gap_sentence_prediction_w = gap_sentence_prediction_w

        self.alum_aitm_alternate = alum_aitm_alternate
        self.lm_loss_w = lm_loss_w
        self.input_cls_orthogonal_w = input_cls_orthogonal_w
        self.first_block_cls_orthogonal_w = first_block_cls_orthogonal_w
        self.second_block_cls_orthogonal_w = second_block_cls_orthogonal_w
        self.third_block_cls_orthogonal_w = third_block_cls_orthogonal_w
        self.register_buffer("diag_mat", 1 - torch.eye(self.cls_tokens + 1, self.cls_tokens + 1, device=next(self.parameters()).device))
        self.electra_loss_w = electra_loss_w
        self.loss_hist = defaultdict(list)
        self.accuracy_hist = defaultdict(list)
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
        self.funnel.decoder.final_hidden_fc = None
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def aitm_loss(self, embed, position_embeds, attention_mask, mlm_predictions, mlm_correct,
                  sent_order_predictions, electra_predictions,
                  labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None, answering_predictions=None,
                  contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None,
                  reverse_loss=False):
        encoder_outputs = self.funnel.encoder(
            embed,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        first_block_hidden = encoder_outputs[2][self.config.block_sizes[0]]
        first_block_hidden = self.lm_dim_match(first_block_hidden[:, (self.cls_tokens + 1):])
        second_block_hidden = encoder_outputs[2][sum(self.config.block_sizes[0:2])]
        third_block_hidden = encoder_outputs[1][sum(self.config.block_sizes)]

        clip_min, clip_max = self.aitm_clip_min, self.aitm_clip_max
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
            sent_order_block_hidden_cls = self.order_prediction_fc(third_block_hidden[:, :self.cls_tokens + 1])
            sent_order_block_hidden_cls = sent_order_block_hidden_cls[:, 1:self.cls_tokens + 1] + sent_order_block_hidden_cls[:, 0].unsqueeze(1)
            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls)
            sent_order_pre_kl = KL(sent_order_logits, sent_order_predictions.detach(), reduction="batchmean")
            if reverse_loss:
                sent_order_pre_kl = (sent_order_pre_kl + KL(sent_order_logits.detach(), sent_order_predictions, reduction="batchmean")) / 2.0

        decoder_outputs = self.funnel.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
            position_embeds=position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        discriminator_sequence_output = decoder_outputs[0][:, (self.cls_tokens + 1):]
        electra_logits = self.discriminator_predictions(discriminator_sequence_output)
        electra_pre_kl = KL(electra_logits, electra_predictions.detach(), reduction="batchmean")
        if reverse_loss:
            electra_pre_kl = (electra_pre_kl + KL(electra_logits.detach(), electra_predictions, reduction="batchmean")) / 2.0

        contrastive_kl = 0.0
        if contrastive_anchors is not None:
            contrastive_block_hidden = third_block_hidden[:, (self.cls_tokens + 1):]

            contrastive_positives = (torch.tensor(contrastive_positives) / self.config.stride ** 2).type(torch.int64).tolist()
            contrastive_anchors = (torch.tensor(contrastive_anchors) / self.config.stride ** 2).type(torch.int64).tolist()

            n_positives_per_anchor = len(contrastive_positives[0][0])
            anchors = [contrastive_block_hidden[anchor_batch_pos, anchor[0]:anchor[1]].mean(0) for anchor_batch_pos, anchors in enumerate(contrastive_anchors)
                       for anchor in anchors]
            contrastive_positives = [[[*cp, batch_pos] for cp in anchor_cp] for batch_pos, anchors_cp in enumerate(contrastive_positives) for anchor_cp in
                                     anchors_cp]

            contrastive_positives = torch.tensor(contrastive_positives).transpose(0, 1).tolist()

            positives = [contrastive_block_hidden[anchor_pos[-1], anchor_pos[0]: anchor_pos[1]].mean(0) for pos in contrastive_positives for anchor_pos in pos]
            n_anchors = len(anchors)
            n_positives = len(positives)
            assert n_positives % n_anchors == 0
            assert (n_positives / n_anchors) == n_positives_per_anchor
            contrastive_block_hidden = torch.stack(anchors + positives)
            contrastive_block_hidden = self.contrastive_ffn(contrastive_block_hidden.unsqueeze(1)).squeeze()
            contrastive_block_hidden = contrastive_block_hidden / contrastive_block_hidden.norm(2, -1, True)
            contrastive_block_matrix = contrastive_block_hidden.mm(contrastive_block_hidden.t()) / self.contrastive_temperature
            contrastive_block_matrix = contrastive_block_matrix * (1 - torch.eye(contrastive_block_matrix.size(0), device=contrastive_block_matrix.device))
            contrastive_kl = KL(contrastive_block_matrix, contrastive_logits.detach(), reduction="batchmean")
            if reverse_loss:
                contrastive_kl = (contrastive_kl + KL(contrastive_block_matrix.detach(), contrastive_logits, reduction="batchmean")) / 2.0

        answering_lm_loss_kl = 0.0
        if labels_pet_input_ids is not None:
            encoder_last_layer_out = encoder_outputs[0][:, self.cls_tokens + 1:]
            alen = min(encoder_last_layer_out.size(1), labels_pet_input_ids.size(1))
            assert labels_pet_input_ids.size(1) <= encoder_last_layer_out.size(1)

            answering_hidden = self.funnel.answering_ffn(encoder_last_layer_out[:, :alen])
            answering_lm_loss_kl = KL(answering_hidden, answering_predictions.detach(), reduction="batchmean")
            if reverse_loss:
                answering_lm_loss_kl = (answering_lm_loss_kl + KL(answering_hidden.detach(), answering_predictions, reduction="batchmean")) / 2.0

            answering_lm_loss_kl = self.answering_lm_w * answering_lm_loss_kl

        return self.lm_loss_w * lm_pre_kl, self.sentence_order_prediction_w * sent_order_pre_kl, self.electra_loss_w * electra_pre_kl, answering_lm_loss_kl, self.contrastive_w * contrastive_kl

    def forward_for_aitm(self, embed, position_embeds, attention_mask,
                         mlm_predictions, mlm_correct, sent_order_predictions, electra_predictions,
                         labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None, answering_logits=None,
                         contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None):
        if self.alum_aitm_alternate:
            self.adv_lm_w = -1 * self.adv_lm_w
        noise = embed.new(embed.size()).normal_(0, 1) * self.aitm_noise_var
        noise.requires_grad_()
        for _ in range(self.adv_ascent_steps):
            newembed = embed.detach() + noise
            lm_pre_kl, sent_order_pre_kl, electra_pre_kl, answering_lm_pre_kl, contrastive_pre_kl = self.aitm_loss(newembed, position_embeds, attention_mask,
                                                                                                                   mlm_predictions, mlm_correct,
                                                                                                                   sent_order_predictions, electra_predictions,
                                                                                                                   labels_pet_input_ids,
                                                                                                                   labels_pet_attention_mask,
                                                                                                                   labels_pet_max_length, answering_logits,
                                                                                                                   contrastive_anchors, contrastive_positives, contrastive_logits)

            adv_loss = electra_pre_kl + sent_order_pre_kl + lm_pre_kl + answering_lm_pre_kl + contrastive_pre_kl
            self.loss_hist["electra_pre_kl"].append(float(electra_pre_kl))
            self.loss_hist["lm_pre_kl"].append(float(lm_pre_kl))
            self.loss_hist["sent_order_pre_kl"].append(float(sent_order_pre_kl))
            self.loss_hist["adv_loss_pre_kl"].append(float(adv_loss))
            self.loss_hist["answering_lm_pre_kl"].append(float(answering_lm_pre_kl))
            self.loss_hist["contrastive_pre_kl"].append(float(contrastive_pre_kl))
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0.0

            noise = noise + delta_grad * self.adv_step_size

        noise = self.adv_project(noise, eps=self.adv_epsilon)

        #
        newembed = embed + noise.detach()
        lm_post_kl, sent_order_post_kl, electra_post_kl, answering_lm_post_kl, contrastive_post_kl = self.aitm_loss(newembed, position_embeds, attention_mask,
                                                                                                                    mlm_predictions, mlm_correct,
                                                                                                                    sent_order_predictions, electra_predictions,
                                                                                                                    labels_pet_input_ids,
                                                                                                                    labels_pet_attention_mask,
                                                                                                                    labels_pet_max_length, answering_logits,
                                                                                                                    contrastive_anchors, contrastive_positives,
                                                                                                                    contrastive_logits,
                                                                                                                    reverse_loss=True)
        self.loss_hist["electra_post_kl"].append(float(electra_post_kl))
        self.loss_hist["lm_post_kl"].append(float(lm_post_kl))
        self.loss_hist["sent_order_post_kl"].append(float(sent_order_post_kl))
        self.loss_hist["answering_lm_post_kl"].append(float(answering_lm_post_kl))
        self.loss_hist["contrastive_post_kl"].append(float(contrastive_post_kl))
        adv_loss = sent_order_post_kl + electra_post_kl + (lm_post_kl ** 2) + answering_lm_post_kl + contrastive_post_kl
        self.loss_hist["adv_loss_post_kl"].append(float(adv_loss))
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask

        inputs_embeds, position_embeds = self.funnel.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets, )
        return inputs_embeds, position_embeds, input_shape

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            labels_two_sentence_order=None,
            labels_three_sentence_dilated_order=None,
            labels_segment_index=None,
            output_attentions=None,
            output_hidden_states=None,
            char_ids=None, char_offsets=None,
            gap_sentence_input_ids=None, gap_sentence_attention_mask=None,
            highway_cls_ar_input_ids=None, highway_cls_ar__attention_mask=None,
            jumble_sentence_input_ids=None, jumble_sentence_attention_mask=None,
            labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
            contrastive_anchors=None, contrastive_positives=None,
            **kwargs

    ):

        timing_dict = list()
        st = time.time()
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        embedding_generation_params = input_ids, token_type_ids, inputs_embeds, char_ids, char_offsets
        inputs_embeds, position_embeds, input_shape = self.get_emb(embedding_generation_params)
        et = time.time() - st
        timing_dict.append(("get_emb", et))
        # TODO: deal with head_mask
        assert attention_mask is not None
        tokenizer_attn_mask = attention_mask
        tokenizer = self.tokenizer
        if self.config.num_highway_cls_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_shape[0], self.config.num_highway_cls_tokens, device=device), attention_mask], dim=1)
        inputs_embeds_cls = inputs_embeds[:, :self.funnel.cls_tokens]
        et = time.time() - st
        timing_dict.append(("prepare_encoder_input", et))
        encoder_outputs = self.funnel.encoder(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        et = time.time() - st
        timing_dict.append(("encoder_outputs", et))
        answering_lm_loss = 0.0
        answering_hidden = None
        if labels_pet_input_ids is not None:
            encoder_last_layer_out = encoder_outputs[0][:, self.cls_tokens + 1:]
            alen = min(encoder_last_layer_out.size(1), labels_pet_input_ids.size(1))
            assert labels_pet_input_ids.size(1) <= encoder_last_layer_out.size(1)

            answering_hidden = self.funnel.answering_ffn(encoder_last_layer_out[:, :alen])
            answering_logits = self.funnel.lm_head(answering_hidden)[:, :, :self.config.vocab_size]
            answering_predictions = answering_logits.argmax(dim=-1)
            loss_fct = self.loss_ce
            answering_lm_loss = self.answering_lm_w * loss_fct(answering_logits.view(-1, self.config.vocab_size), labels_pet_input_ids[:, :alen].reshape(-1))
            self.loss_hist["answering_lm_loss"].append(float(answering_lm_loss))
            answering_lm_correct = answering_predictions == labels_pet_input_ids[:, :alen]
            self.accuracy_hist["answering_lm"].append(float(answering_lm_correct.sum() / len(answering_lm_correct.view(-1))))

        first_block_hidden = encoder_outputs[2][self.config.block_sizes[0]]
        first_block_cls = first_block_hidden[:, :self.funnel.cls_tokens]
        first_block_hidden = self.lm_dim_match(first_block_hidden[:, (self.cls_tokens + 1):])
        prediction_logits = self.lm_head(first_block_hidden)[:, :, :self.config.vocab_size]
        et = time.time() - st
        timing_dict.append(("lm_logits", et))

        second_block_hidden = encoder_outputs[2][sum(self.config.block_sizes[0:2])]
        second_block_cls = second_block_hidden[:, :self.funnel.cls_tokens]
        third_block_hidden = encoder_outputs[1][sum(self.config.block_sizes)]  # for last block both input and output shapes are same
        third_block_cls = third_block_hidden[:, :self.funnel.cls_tokens]
        loss_contrastive = 0.0
        contrastive_block_matrix = None
        contrastive_anchors_copy, contrastive_positives_copy = copy.deepcopy(contrastive_anchors), copy.deepcopy(contrastive_positives)
        if contrastive_anchors is not None:
            contrastive_block_hidden = third_block_hidden[:, (self.cls_tokens + 1):]

            contrastive_positives = (torch.tensor(contrastive_positives) / self.config.stride ** 2).type(torch.int64).tolist()
            contrastive_anchors = (torch.tensor(contrastive_anchors) / self.config.stride ** 2).type(torch.int64).tolist()

            n_positives_per_anchor = len(contrastive_positives[0][0])
            anchors = [contrastive_block_hidden[anchor_batch_pos, anchor[0]:anchor[1]].mean(0) for anchor_batch_pos, anchors in enumerate(contrastive_anchors) for anchor in anchors]
            contrastive_positives = [[[*cp, batch_pos] for cp in anchor_cp] for batch_pos, anchors_cp in enumerate(contrastive_positives) for anchor_cp in anchors_cp]

            contrastive_positives = torch.tensor(contrastive_positives).transpose(0, 1).tolist()

            positives = [contrastive_block_hidden[anchor_pos[-1], anchor_pos[0]: anchor_pos[1]].mean(0) for pos in contrastive_positives for anchor_pos in pos]
            n_anchors = len(anchors)
            n_positives = len(positives)
            assert n_positives % n_anchors == 0
            assert (n_positives / n_anchors) == n_positives_per_anchor
            contrastive_block_hidden = torch.stack(anchors + positives)
            contrastive_block_hidden = self.contrastive_ffn(contrastive_block_hidden.unsqueeze(1)).squeeze()
            contrastive_block_hidden = contrastive_block_hidden / contrastive_block_hidden.norm(2, -1, True)
            contrastive_block_matrix = contrastive_block_hidden.mm(contrastive_block_hidden.t()) / self.contrastive_temperature
            contrastive_block_matrix = contrastive_block_matrix * (1 - torch.eye(contrastive_block_matrix.size(0), device=contrastive_block_matrix.device))
            labels_contrastive = torch.tensor(list(range(n_anchors)) * n_positives_per_anchor, device=contrastive_block_matrix.device)

            loss_contrastive = self.ce(contrastive_block_matrix[n_anchors:], labels_contrastive)
            self.accuracy_hist["contrastive"].append((contrastive_block_matrix[n_anchors:].argmax(dim=-1) == labels_contrastive).sum().item() / n_positives)
            mask1 = torch.ones(n_anchors, contrastive_block_matrix.size(1), device=contrastive_block_hidden.device)
            mask2 = torch.zeros(n_anchors, contrastive_block_matrix.size(1), device=contrastive_block_hidden.device)
            for i in range(n_positives_per_anchor):
                mask1[list(range(n_anchors)), torch.tensor(list(range(n_anchors))) + (n_anchors * (i + 1))] = 0
            vertical_lc = 0.0
            for i in range(n_positives_per_anchor):

                labels_contrastive = torch.tensor(list(range(n_anchors)), device=contrastive_block_hidden.device) + (n_anchors * (i + 1))
                mask_c = mask2.clone()
                mask_c[list(range(n_anchors)), torch.tensor(list(range(n_anchors)), device=contrastive_block_hidden.device) + (n_anchors * (i + 1))] = 1
                mask_c = mask1 + mask_c
                l2 = self.ce(contrastive_block_matrix[:n_anchors] * mask_c, labels_contrastive)
                vertical_lc += l2
            vertical_lc /= n_positives_per_anchor
            loss_contrastive += vertical_lc
        loss_contrastive = self.contrastive_w * loss_contrastive
        self.loss_hist["contrastive_loss"].append(float(loss_contrastive))
        et = time.time() - st
        timing_dict.append(("contrastive_loss", et))
        cls_orthogonal_loss = 0.0
        if self.input_cls_orthogonal_w > 0 and self.training:
            inputs_embeds_cls = inputs_embeds_cls/inputs_embeds_cls.norm(2, -1, True)
            inputs_embeds_cls = inputs_embeds_cls.bmm(inputs_embeds_cls.transpose(1, 2))
            input_cls_orthogonal_loss = self.input_cls_orthogonal_w * ((inputs_embeds_cls * self.diag_mat) ** 2).mean()
            self.loss_hist["input_cls_orthogonal_loss"].append(float(input_cls_orthogonal_loss))
            cls_orthogonal_loss += input_cls_orthogonal_loss

        if self.first_block_cls_orthogonal_w > 0 and self.training:
            first_block_cls = first_block_cls/first_block_cls.norm(2, -1, True)
            first_block_cls = first_block_cls.bmm(first_block_cls.transpose(1, 2))
            first_block_cls_orthogonal_loss = self.first_block_cls_orthogonal_w * ((first_block_cls * self.diag_mat) ** 2).mean()
            self.loss_hist["first_block_cls_orthogonal_loss"].append(float(first_block_cls_orthogonal_loss))
            cls_orthogonal_loss += first_block_cls_orthogonal_loss

        if self.second_block_cls_orthogonal_w > 0 and self.training:
            second_block_cls = second_block_cls/second_block_cls.norm(2, -1, True)
            second_block_cls = second_block_cls.bmm(second_block_cls.transpose(1, 2))
            second_block_cls_orthogonal_loss = self.second_block_cls_orthogonal_w * ((second_block_cls * self.diag_mat) ** 2).mean()
            self.loss_hist["second_block_cls_orthogonal_loss"].append(float(second_block_cls_orthogonal_loss))
            cls_orthogonal_loss += second_block_cls_orthogonal_loss

        if self.third_block_cls_orthogonal_w > 0 and self.training:
            third_block_cls = third_block_cls/third_block_cls.norm(2, -1, True)
            third_block_cls = third_block_cls.bmm(third_block_cls.transpose(1, 2))
            third_block_cls_orthogonal_loss = self.third_block_cls_orthogonal_w * ((third_block_cls * self.diag_mat) ** 2).mean()
            self.loss_hist["third_block_cls_orthogonal_loss"].append(float(third_block_cls_orthogonal_loss))
            cls_orthogonal_loss += third_block_cls_orthogonal_loss

        et = time.time() - st
        timing_dict.append(("cls_orthogonal_loss", et))
        sentence_order_loss = 0.0
        word_order_loss = 0.0
        gap_sentence_loss = 0.0
        highway_cls_ar_loss = 0.0
        sent_order_logits = None
        if self.sentence_order_prediction_w > 0 and labels_segment_index is not None and self.training:
            sent_order_block_hidden_cls = self.order_prediction_fc(third_block_hidden[:, :self.cls_tokens + 1])
            sent_order_block_hidden_cls = sent_order_block_hidden_cls[:, 1:self.cls_tokens + 1] + sent_order_block_hidden_cls[:, 0].unsqueeze(1)
            two_sent_order_hidden = torch.cat((sent_order_block_hidden_cls, sent_order_block_hidden_cls[:, 0:1]), 1)
            three_sent_order_hidden = torch.cat((sent_order_block_hidden_cls, sent_order_block_hidden_cls[:, 0:4]), 1)

            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls)
            sent_order_loss = self.loss_ce(sent_order_logits.view(-1, self.cls_tokens), labels_segment_index.view(-1))
            self.loss_hist["sent_order_loss"].append(float(sent_order_loss))
            sent_order_out = sent_order_logits.argmax(dim=-1) == labels_segment_index
            self.accuracy_hist["sent_order"].append(float(sent_order_out.sum() / len(sent_order_out.view(-1))))

            two_sent_order_preds = self.two_sent_order_fc(two_sent_order_hidden).view(-1, self.cls_tokens)
            three_sent_order_preds = self.three_sent_order_fc(three_sent_order_hidden).view(-1, self.cls_tokens)
            two_sent_loss = self.loss_bce(two_sent_order_preds, labels_two_sentence_order.float())
            three_sent_loss = self.loss_bce(three_sent_order_preds, labels_three_sentence_dilated_order.float())
            self.loss_hist["two_sent_loss"].append(float(two_sent_loss))
            self.loss_hist["three_sent_loss"].append(float(three_sent_loss))
            self.accuracy_hist["two_sent_order"].append(float(((two_sent_order_preds > 0.) == labels_two_sentence_order).sum() / len(labels_two_sentence_order.view(-1))))
            self.accuracy_hist["three_sent_order"].append(
                float(((three_sent_order_preds > 0.) == labels_three_sentence_dilated_order).sum() / len(labels_three_sentence_dilated_order.view(-1))))

            sentence_order_loss = self.sentence_order_prediction_w * (sent_order_loss + two_sent_loss + three_sent_loss)
        et = time.time() - st
        timing_dict.append(("sentence_order_loss", et))

        if (self.word_order_prediction_w > 0 and jumble_sentence_input_ids is not None) or \
                (self.gap_sentence_prediction_w > 0 and gap_sentence_input_ids is not None) or \
                (self.highway_cls_ar_w > 0 and highway_cls_ar_input_ids is not None):
            highway_block_hidden = self.word_gap_prediction_fc(second_block_hidden[:, :self.cls_tokens + 1])

        if self.word_order_prediction_w > 0 and jumble_sentence_input_ids is not None and self.training and not (self.highway_cls_ar_w > 0 and highway_cls_ar_input_ids is not None):
            word_order_inputs_embeds, _ = self.funnel.embeddings(shift_right(jumble_sentence_input_ids, self.pad_token_id, self.pad_token_id), None, None, char_ids=None, char_offsets=None, )
            initiator_emb = self.initiator_emb(torch.tensor(0, device=highway_block_hidden.device))[None, None, :]
            word_order_inputs_embeds = word_order_inputs_embeds + initiator_emb
            if self.config.num_highway_cls_tokens > 0:
                jumble_sentence_attention_mask = torch.cat(
                    [torch.ones(jumble_sentence_attention_mask.shape[0], self.config.num_highway_cls_tokens, device=jumble_sentence_attention_mask.device),
                     jumble_sentence_attention_mask], dim=1)

            word_order_out = self.sentence_task_attn(word_order_inputs_embeds, highway_block_hidden, highway_block_hidden, jumble_sentence_attention_mask, encoder_outputs[-1][2]) # [:, :self.cls_tokens + 1]
            word_order_out = self.lm_dim_match(word_order_out[:, (self.funnel.cls_tokens - 1):])
            word_order_out = self.lm_head(word_order_out)[:, :, :self.config.vocab_size]
            word_order_loss = self.word_order_prediction_w * self.loss_ce(word_order_out.view(-1, self.config.vocab_size), jumble_sentence_input_ids.view(-1))
            word_order_out = word_order_out.argmax(dim=-1) == jumble_sentence_input_ids
            self.accuracy_hist["word_order"].append(float(word_order_out.sum() / len(word_order_out.view(-1))))
            self.loss_hist["word_order_loss"].append(float(word_order_loss))

        if self.gap_sentence_prediction_w > 0 and gap_sentence_input_ids is not None and self.training and not (self.highway_cls_ar_w > 0 and highway_cls_ar_input_ids is not None):
            gap_inputs_embeds, _ = self.funnel.embeddings(shift_right(gap_sentence_input_ids, self.pad_token_id, self.pad_token_id), None, None, char_ids=None, char_offsets=None, )
            initiator_emb = self.initiator_emb(torch.tensor(1, device=highway_block_hidden.device))[None, None, :]
            gap_inputs_embeds = gap_inputs_embeds + initiator_emb
            if self.config.num_highway_cls_tokens > 0:
                gap_sentence_attention_mask = torch.cat(
                    [torch.ones(gap_sentence_attention_mask.shape[0], self.config.num_highway_cls_tokens, device=gap_sentence_attention_mask.device),
                     gap_sentence_attention_mask], dim=1)

            gap_sentence_out = self.sentence_task_attn(gap_inputs_embeds, highway_block_hidden, highway_block_hidden, gap_sentence_attention_mask, encoder_outputs[-1][2]) # [:, :self.cls_tokens + 1]
            gap_sentence_out = self.lm_dim_match(gap_sentence_out[:, (self.funnel.cls_tokens - 1):])
            gap_sentence_out = self.lm_head(gap_sentence_out)[:, :, :self.config.vocab_size]
            gap_sentence_loss = self.gap_sentence_prediction_w * self.loss_ce(gap_sentence_out.view(-1, self.config.vocab_size), gap_sentence_input_ids.view(-1))
            gap_sentence_out = gap_sentence_out.argmax(dim=-1) == gap_sentence_input_ids
            self.accuracy_hist["gap_sentence"].append(float(gap_sentence_out.sum() / len(gap_sentence_out.view(-1))))
            self.loss_hist["gap_sentence_loss"].append(float(gap_sentence_loss))

        if self.highway_cls_ar_w > 0 and highway_cls_ar_input_ids is not None and self.training:
            highway_cls_ar_inputs_embeds, _ = self.funnel.embeddings(shift_right(highway_cls_ar_input_ids, self.pad_token_id, self.pad_token_id), None, None, char_ids=None, char_offsets=None, )
            highway_cls_ar_inputs_embeds_non_positional, _ = self.funnel.embeddings(highway_cls_ar_input_ids, None, None,
                                                                                    char_ids=None, char_offsets=None, use_position_embeddings=False)
            initiator_emb = self.initiator_emb(torch.tensor(2, device=highway_block_hidden.device))[None, None, :]
            highway_cls_ar_inputs_embeds = highway_cls_ar_inputs_embeds + initiator_emb

            if self.config.num_highway_cls_tokens > 0:
                highway_cls_ar__attention_mask = torch.cat(
                    [torch.ones(highway_cls_ar__attention_mask.shape[0], self.config.num_highway_cls_tokens, device=highway_cls_ar__attention_mask.device),
                     highway_cls_ar__attention_mask], dim=1)

            highway_cls_ar_out = self.sentence_task_attn(highway_cls_ar_inputs_embeds, highway_block_hidden, highway_block_hidden, highway_cls_ar__attention_mask, encoder_outputs[-1][2][:, :highway_block_hidden.size(1)])
            highway_cls_ar_out = self.sentence_task_attn(highway_cls_ar_out, highway_cls_ar_inputs_embeds_non_positional, highway_cls_ar_inputs_embeds_non_positional, highway_cls_ar__attention_mask,
                                                         highway_cls_ar__attention_mask)

            highway_cls_ar_out = self.lm_dim_match(highway_cls_ar_out[:, (self.funnel.cls_tokens - 1):])
            highway_cls_ar_out = self.lm_head(highway_cls_ar_out)[:, :, :self.config.vocab_size]
            highway_cls_ar_loss = self.highway_cls_ar_w * self.loss_ce(highway_cls_ar_out.reshape(-1, self.config.vocab_size), highway_cls_ar_input_ids.reshape(-1))
            highway_cls_ar_out = highway_cls_ar_out.argmax(dim=-1)
            self.accuracy_hist["highway_cls_ar_sentence_outputs"].append((tokenizer.decode(highway_cls_ar_input_ids[0, 1:21].tolist()), tokenizer.decode(highway_cls_ar_out[0, 1:21].tolist())))
            highway_cls_ar_out = highway_cls_ar_out == highway_cls_ar_input_ids
            self.accuracy_hist["highway_cls_ar_sentence"].append(float(highway_cls_ar_out.float().mean()))
            self.loss_hist["highway_cls_ar_sentence_loss"].append(float(highway_cls_ar_loss))

        et = time.time() - st
        timing_dict.append(("highway_cls_ar_sentence_loss", et))
        tokenizer_attn_mask = tokenizer_attn_mask[:, 1:]
        active_loss = tokenizer_attn_mask.view(-1, input_shape[1] - 1) == 1
        assert labels is not None

        loss_fct = self.loss_ce  # -100 index = padding token
        labels = labels[:, 1:]
        active_labels = labels[active_loss]
        active_prediction_logits = prediction_logits[active_loss]
        masked_lm_loss = self.lm_loss_w * loss_fct(active_prediction_logits.reshape(-1, self.config.vocab_size), active_labels.reshape(-1))
        predictions = prediction_logits.argmax(dim=-1)
        self.accuracy_hist["lm_preds"].append(("".join(self.tokenizer.decode(predictions[0, 1:21].tolist())), "".join(self.tokenizer.decode(labels[0, 1:21].tolist()))))
        labels = (labels == predictions).float()
        mlm_positions = input_ids == self.tokenizer.mask_token_id
        self.accuracy_hist["lm"].append(float(labels[active_loss].float().mean()))
        mlm_positions = mlm_positions[:, 1:]
        self.accuracy_hist["masked_lm"].append(float(labels[mlm_positions].float().mean()))

        et = time.time() - st
        timing_dict.append(("lm_accuracy_loss", et))

        decoder_outputs = self.funnel.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
            position_embeds=position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
        )

        et = time.time() - st
        timing_dict.append(("decoder_outputs", et))

        cls_tokens = decoder_outputs[0][:, :self.cls_tokens + 1]
        decoder_outputs = (decoder_outputs[0][:, self.cls_tokens + 1:], decoder_outputs[1:])
        discriminator_sequence_output = decoder_outputs[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)

        et = time.time() - st
        timing_dict.append(("electra_discriminator_logits", et))

        active_logits = logits.view(-1, input_shape[1] - 1)[active_loss]
        active_labels = labels[active_loss]
        loss = self.electra_loss_w * self.loss_bce(active_logits, active_labels)
        self.accuracy_hist["electra"].append(torch.mean(((torch.sigmoid(active_logits) > 0.5).type(torch.int64) == active_labels).type(torch.float)).item())

        et = time.time() - st
        timing_dict.append(("electra_discriminator_accuracy", et))
        # TODO: Store losses here

        self.loss_hist["electra_loss"].append(float(loss))
        self.loss_hist["lm_loss"].append(float(masked_lm_loss))
        self.loss_hist["sentence_order_loss"].append(float(sentence_order_loss))
        self.loss_hist["word_order_loss"].append(float(word_order_loss))
        self.loss_hist["gap_sentence_loss"].append(float(gap_sentence_loss))

        et = time.time() - st
        timing_dict.append(("aitm_alum_start", et))
        if (self.aitm or self.alum) and self.training:
            adv_loss = self.forward_for_aitm(inputs_embeds, position_embeds, attention_mask, first_block_hidden, labels, sent_order_logits, logits,
                                             labels_pet_input_ids, labels_pet_attention_mask, labels_pet_max_length, answering_hidden,
                                             contrastive_anchors_copy, contrastive_positives_copy, contrastive_block_matrix)
            loss = loss + adv_loss

        et = time.time() - st
        timing_dict.append(("aitm_alum_end", et))

        loss = loss + masked_lm_loss + sentence_order_loss + word_order_loss + gap_sentence_loss + answering_lm_loss + highway_cls_ar_loss + cls_orthogonal_loss + loss_contrastive

        et = time.time() - st
        timing_dict = [(k, 100 * (v/et)) for k, v in timing_dict]
        self.timing_hist.append(timing_dict)

        # TODO: return one loss
        # TODO: check various loss history and accuracies over time
        # TODO: Make a separate wrapper for AITM and ALUM vs making it here?

        # TODO: CLS correction needed
        results = dict(loss=loss,
                       decoder_output=decoder_outputs[0], decoder_cls=cls_tokens,
                       encoder_output=encoder_outputs[0][:, self.cls_tokens + 1:], encoder_cls=encoder_outputs[0][:, :self.cls_tokens + 1], encoder_hidden_states=encoder_outputs[1])
        return results


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_tokenizer(name):
    from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
    if "roberta" in name:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif "bert" in name:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    setattr(tokenizer, "_sentence_mask_token", "[MASK1]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["sentence_mask_token"]
    tokenizer.add_special_tokens({"sentence_mask_token": "[MASK1]"})

    setattr(tokenizer, "_answer_option_separator_token", "[ANSWER_OPTION_SEP]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_separator_token"]
    tokenizer.add_special_tokens({"answer_option_separator_token": "[ANSWER_OPTION_SEP]"})

    setattr(tokenizer, "_answer_option_begin_token", "[ANSWER_OPTION_BEGIN]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_begin_token"]
    tokenizer.add_special_tokens({"answer_option_begin_token": "[ANSWER_OPTION_BEGIN]"})

    setattr(tokenizer, "_answer_option_end_token", "[ANSWER_OPTION_END]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_end_token"]
    tokenizer.add_special_tokens({"answer_option_end_token": "[ANSWER_OPTION_END]"})

    setattr(tokenizer, "_answer_end_token", "[ANSWER_END]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_end_token"]
    tokenizer.add_special_tokens({"answer_end_token": "[ANSWER_END]"})
    n_question_tokens = 8
    for i in range(n_question_tokens):
        setattr(tokenizer, "_question_token_%s" % i, "[QUESTION_%s]" % i)
        setattr(tokenizer, "_answer_token_%s" % i, "[ANSWER_%s]" % i)
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["question_token_%s" % i]
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_token_%s" % i]
        tokenizer.add_special_tokens({"question_token_%s" % i: "[QUESTION_%s]" % i, "answer_token_%s" % i: "[ANSWER_%s]" % i})

    return tokenizer


if __name__ == "__main__":
    import time
    import argparse
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW

    torch.backends.cudnn.benchmark = True

    from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining, CTRLConfig, CTRLPreTrainedModel
    from transformers.models.deberta import DebertaModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default='cpu',
                    help="Device")
    ap.add_argument("--config", type=str, default='md_config_funnel',
                    help="Config")
    ap.add_argument("--profile", type=str2bool, default=False)
    ap.add_argument("--sdconv", type=str2bool, default=False)
    ap.add_argument("--forward_only", type=str2bool, default=False)
    ap.add_argument("--fp16", type=str2bool, default=False)
    ap.add_argument("--aitm", type=str2bool, default=False)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
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
    config = dict(md_config=md_config, md_config_rnn=md_config_rnn, md_config_funnel=md_config_funnel,
                  sm_config=sm_config, md_config_sdconv=md_config_sdconv,
                  md_config_funnel_mp=md_config_funnel_mp, md_config_funnel_sp=md_config_funnel_sp,
                  md_config_funnel_lp=md_config_funnel_lp, md_config_funnel_rp=md_config_funnel_rp)[args["config"]]
    epochs = args["epochs"]
    if aitm:
        assert not forward_only and model_name == "fastformer_fused_electra"
    HuggingFaceModelClass = AutoModel if forward_only else AutoModelForMaskedLM

    small_max_length = 128
    medium_max_length = 512
    large_max_length = 1024
    very_large_max_length = 1536
    texts = very_large_texts

    tokenizer = get_tokenizer("bert")
    config.tokenizer_length = large_max_length
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens
    if model_name not in ["fastformer_mlm", "fastformer_electra", "fastformer_fused_electra", "fastformer"]:
        config.tokenizer_length = min(config.tokenizer_length, 512)
        config.max_position_embeddings = min(config.tokenizer_length, 512)
        config.num_highway_cls_tokens = 0
    char_to_id = sorted([k for k, v in AutoTokenizer.from_pretrained("bert-base-uncased").get_vocab().items() if len(k) == 1]) + [" ", "\n"]
    char_to_id = dict(zip(char_to_id, range(2, len(char_to_id) + 2)))
    if batch_size > len(texts):
        for _ in range(batch_size // len(texts)):
            texts += texts
    dataset = SmallTextDataset(texts)
    assert config.tokenizer_length % 16 == 0  # Due to our collate fn
    dataset = TokenizerDataset(config, tokenizer, char_to_id,
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length),
                               sentence_jumble_proba=((1024, 0.0),), word_noise_proba=((1024, 0.0),), max_jumbling_span_length=2,
                               dataset=dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, prefetch_factor=2, num_workers=0)
    pt_batch = next(iter(dataloader))

    if "fastformer" in model_name:
        sm_pt_batch = dict(input_ids=pt_batch["input_ids"], attention_mask=pt_batch["attention_mask"],
                        char_offsets=pt_batch["char_offsets"], char_ids=pt_batch["char_ids"])
        if model_name == "fastformer_electra":
            model = FastFormerForELECTRAPretraining(sm_config, config, tokenizer=tokenizer)
            assert not forward_only
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
    optimizer = AdamW(all_params, lr=5e-4, eps=1e-6, weight_decay=1e-2)


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
        print("Time Taken = %.4f, Lowest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.std(times)), times)

    if not forward_only and hasattr(model, "accuracy_hist"):
        from pprint import pprint
        if hasattr(model, "highway_cls_ar_sentence_outputs"):
            del model.accuracy_hist["highway_cls_ar_sentence_outputs"]
        pprint({k: v[-10:] for k, v in model.accuracy_hist.items()})
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
