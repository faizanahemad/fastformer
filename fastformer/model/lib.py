import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import traceback

try:
    from fairscale.nn.misc import checkpoint_wrapper
    from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
except:
    pass


import numpy as np
import math
import random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from fastformer.utils import *


INF = 1e6
EPS = 1e-6

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

            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0_reverse', )
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
            self.gru[i].flatten_parameters()
            qp = self.gru[i](query[i])[0]
            processed_query.append(qp)
        query = torch.stack(processed_query, 0)

        query = query[:, :, self.overlap:-self.overlap]
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

        query = query.permute(1, 2, 0, 3).reshape(-1, query.shape[2], query.size(-1))

        # query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        # query = self.gru(query)[0]
        # query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        # query = query[:, self.overlap:-self.overlap]
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

        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=in_channels, bias=False, stride=stride, padding=padding)
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
        self.conv_attn_kernel = nn.Conv1d(in_channels=hidden_size, out_channels=self.heads * self.kernel_size,
                                          kernel_size=kernel_size, groups=heads, bias=False, stride=stride, padding=(kernel_size - 1) // 2)
        # self.conv_attn_kernel = nn.Linear(self.all_head_size, self.heads * self.kernel_size)  # Multi-head?
        # if config.no_v_head:
        #     self.conv_attn_point = nn.Identity()
        # else:
        #     self.conv_attn_point = nn.Linear(hidden_size, hidden_size, bias=False)
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

        conv_kernel_layer = self.conv_attn_kernel(query.transpose(1, 2)).transpose(1, 2)  # Softmax only in kernel dim

        if not self.use_cuda_conv or self.stride != 1:
            conv_kernel_layer = conv_kernel_layer.reshape(-1, self.kernel_size, 1)  # BxSxH, k, 1
            conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

            # conv_out_layer
            conv_out_layer = value.permute(0, 2, 1).unsqueeze(-1)  # B,D,Seq, 1
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
            conv_out_layer = value.permute(0, 2, 1).contiguous()

            conv_out_layer = dynamicconvFunction.apply(
                conv_out_layer, weights,
                self.padding_l).transpose(1, 2).contiguous()
            conv_out = torch.reshape(conv_out_layer, [bs, seqlen, -1])

        if upsampled:
            conv_out = pool_tensor(conv_out, self.cls_tokens, "mean", self.config.stride)
        return conv_out


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

            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0',)
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'weight_ih_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_hh_l0_reverse', )
            # rnn = torch.nn.utils.weight_norm(rnn, 'bias_ih_l0_reverse', )
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
            self.gru[i].flatten_parameters()
            qp = self.gru[i](query[i])[0]
            processed_query.append(qp)
        query = torch.stack(processed_query, 0)

        query = query[:, :, self.overlap:-self.overlap]
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

        query = query.permute(1, 2, 0, 3).reshape(-1, query.shape[2], query.size(-1))

        # query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        # query = self.gru(query)[0]
        # query = query.reshape(-1, self.heads, query.shape[1], query.shape[2]).transpose(1, 2).view(-1, query.shape[1], self.heads * query.shape[2])
        # query = query[:, self.overlap:-self.overlap]
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

        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=in_channels, bias=False, stride=stride, padding=padding)
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
        self.conv_attn_kernel = nn.Conv1d(in_channels=hidden_size, out_channels=self.heads * self.kernel_size,
                                          kernel_size=kernel_size, groups=heads, bias=False, stride=stride, padding=(kernel_size - 1) // 2)
        # self.conv_attn_kernel = nn.Linear(self.all_head_size, self.heads * self.kernel_size)  # Multi-head?
        # if config.no_v_head:
        #     self.conv_attn_point = nn.Identity()
        # else:
        #     self.conv_attn_point = nn.Linear(hidden_size, hidden_size, bias=False)
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

        conv_kernel_layer = self.conv_attn_kernel(query.transpose(1, 2)).transpose(1, 2)  # Softmax only in kernel dim

        if not self.use_cuda_conv or self.stride != 1:
            conv_kernel_layer = conv_kernel_layer.reshape(-1, self.kernel_size, 1)  # BxSxH, k, 1
            conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

            # conv_out_layer
            conv_out_layer = value.permute(0, 2, 1).unsqueeze(-1)  # B,D,Seq, 1
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
            conv_out_layer = value.permute(0, 2, 1).contiguous()

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
            self.contract = nn.Identity()
            self.expansion_factor = 1

    def forward(self, query):
        # st = time.time()
        qskip = pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention)
        if self.expansion_factor == 1:
            return qskip
        else:
            query = torch.cat((self.expand(query), query), dim=-1)
        query = self.contract(pool_tensor(query, self.cls_tokens, mode='mean', stride=self.compressed_query_attention))
        # ext = time.time()
        # print("Mean pooling = %.5f" % (ext - st))
        return qskip + query

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
            self.layer_norm = nn.LayerNorm(d_next, config.layer_norm_eps)
            self.dim_match_stride = int(np.ceil(d_model / self.diff))
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
                pre_ffn = h
                dh = pool_tensor_basic(h.transpose(1, 2), stride=self.dim_match_stride).transpose(1, 2)[:, :, :self.diff]
                h = torch.cat((h, dh), 2)
                h = self.layer_norm(h)
                hidden = nn.functional.pad(hidden, (0, self.diff, 0, 0, 0, 0))
                h = hidden + h
            else:
                hplus = h + hidden
                dh = pool_tensor_basic(hplus.transpose(1, 2), stride=self.dim_match_stride).transpose(1, 2)[:, :, :self.diff]
                pre_ffn = h
                h = self.layer_norm(torch.cat((hplus, dh), 2))
        else:
            if self.config.identity_preserving_norm:
                h = self.layer_norm(h)
                pre_ffn = h
                h = hidden + h
            else:
                pre_ffn = h
                h = self.layer_norm(hidden + h)
        return pre_ffn, h



