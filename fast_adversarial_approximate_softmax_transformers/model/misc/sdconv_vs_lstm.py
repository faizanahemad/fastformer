import numpy as np
import torch
from torch import nn
from torch.autograd import profiler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from performer_pytorch import SelfAttention, FastAttention
from collections import defaultdict
from transformers.activations import ACT2FN

from fast_adversarial_approximate_softmax_transformers.model.funnel_transformer import FunnelConfig

try:
    from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
except:
    pass


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


class ShortSeqLSTM(nn.Module):
    def __init__(self, config: FunnelConfig, hidden_size, heads, head_size, kernel_size, overlap, stride=1):
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
        self.overlap = overlap

        self.gru = nn.RNN(hidden_size, self.all_head_size // 2, 1, nonlinearity="relu", bias=False, batch_first=True, dropout=0.0, bidirectional=True)
        self.unfold1d = nn.Unfold(kernel_size=[kernel_size + 2 * overlap, 1], padding=[overlap, 0], stride=[kernel_size, 1])

    def forward(self, query, key=None, value=None):
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
            query = value + upsample(query, self.config.stride, context_len, self.cls_tokens)
            seqlen = context_len

        query = nn.functional.pad(query, (0, 0, 0, self.kernel_size, 0, 0))
        query = self.unfold1d(query.transpose(1, 2).unsqueeze(-1))
        query = query.transpose(1, 2).reshape(-1, dim, self.kernel_size + 2 * self.overlap).transpose(1, 2)  # B, S, D
        query = self.gru(query)[0]
        query = query.view(bs, -1, self.kernel_size + 2 * self.overlap, dim)
        query = query[:, :, self.overlap:-self.overlap, :]
        query = query.reshape(bs, -1, dim)[:, :seqlen]  # B, num_splits, S, D

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        if self.stride > 1:
            query = pool_tensor(query, 0, "mean", self.stride)
        return query


class ShortSeqLSTMv2(nn.Module):
    def __init__(self, config: FunnelConfig, hidden_size, heads, head_size, kernel_size, overlap, stride=1):
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
        self.overlap = overlap

        self.gru = nn.RNN(hidden_size, self.all_head_size // 2, 1,
                          nonlinearity="relu",
                          bias=False, batch_first=True, dropout=0.0, bidirectional=True)

    def forward(self, query, key=None, value=None):
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
        query = self.gru(query)[0]
        query = query.reshape(bs, -1, dim)[:, self.overlap:seqlen+self.overlap]

        if upsampled:
            query = pool_tensor(query, self.cls_tokens, "mean", self.config.stride)

        if self.stride > 1:
            query = pool_tensor(query, 0, "mean", self.stride)
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
            inputs = inputs.view(bs, -1 if self.channels_last else dims, dims if self.channels_last else -1)
        else:
            inputs = self.pointwise(inputs)
            inputs = self.channel_move(inputs).view(bs, -1 if self.channels_last else dims, dims if self.channels_last else -1)
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
        self.separable_conv1d = SeparableConv1d(hidden_size, self.all_head_size, kernel_size, pointwise_groups=heads, stride=stride, channels_last=True)
        # self.conv_attn_kernel = nn.Linear(self.all_head_size, self.heads * self.kernel_size)  # Multi-head?
        self.conv_attn_kernel = nn.Conv1d(self.all_head_size, self.heads * self.kernel_size, 1, groups=heads)
        self.conv_attn_point = nn.Linear(hidden_size, self.all_head_size)
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
            unfold_conv_out_layer = unfold_conv_out_layer.transpose(1, 2).reshape(bs, unfold_conv_out_layer.shape[2], -1,
                                                                                  self.kernel_size)  # B, seq, D, kernel_size
            conv_out_layer = torch.reshape(
                unfold_conv_out_layer,
                [-1, self.head_size, self.kernel_size])  # BxSxH, H_dim, kernel
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


class ShortSeqRNN(nn.Module):
    def __init__(self, config: FunnelConfig, hidden_size, heads, head_size, kernel_size, overlap, stride=1):
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
        self.overlap = overlap

        self.gru = nn.RNN(hidden_size // self.heads, hidden_size // (2 * self.heads), 1,
                          nonlinearity="tanh",
                          bias=False, batch_first=True, dropout=0.0, bidirectional=True)

    def forward(self, query, key=None, value=None):
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

        if self.stride > 1:
            query = pool_tensor(query, 0, "mean", self.stride)
        return query


import time
import numpy as np
from tqdm.auto import tqdm, trange
t = torch.randn(8, 512, 768)
sdconv = SDConv(FunnelConfig(), 768, 12, 64, 9, 1)
lstm = nn.GRU(768, 768 // 2, 1, False, True, bidirectional=True)
rnn = nn.RNN(768, 768 // 2, 1, nonlinearity="relu", bias=False, batch_first=True, dropout=0.0, bidirectional=True)
sh_rnn = ShortSeqRNN(FunnelConfig(), 768, 12, 64, 128, 8, 1)
short_lstm = ShortSeqLSTMv2(FunnelConfig(), 768, 12, 64, 128, 8, 1)

model = sh_rnn
model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
params = sum([np.prod(p.size()) for p in model_parameters])
print("Trainable Params = %s" % (params/1_000))
model = model.eval()

_ = [model(t) for _ in trange(1)]
print(t.size(), model(t).size()) # model(t)[0].size()
with profiler.profile(record_shapes=True) as prof:
    _ = [model(t) for _ in trange(5)]


print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))


def test_conv_v1():
    # cnn = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=9, groups=768, bias=True, stride=1, padding=9 // 2)
    with torch.no_grad():
        _ = [model(t) for _ in range(10)]
        times = []

        for _ in trange(100):
            st = time.time()
            _ = model(t)
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.5f, variance = %.5f" % (np.mean(times), np.std(times)))


test_conv_v1()
