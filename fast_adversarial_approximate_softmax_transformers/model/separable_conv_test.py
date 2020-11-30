from typing import Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
import torch.autograd.profiler as profiler
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvNd, _size_1_t, _single, Tensor


class Conv1d(_ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, False,
                                     _single(0), groups, bias, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups,
                                   self.padding_mode)


def conv1d_same_padding(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        padding_mode="zeros"):
    if padding == "same":
        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)

        if padding_rows > 0:
            if padding_mode == "zeros":
                input = F.pad(
                    input,
                    [padding_rows // 2, padding_rows - padding_rows // 2],
                    mode="constant",
                    value=0)
            else:
                input = F.pad(
                    input,
                    [padding_rows // 2, padding_rows - padding_rows // 2],
                    mode=padding_mode)
        padding = (0, )

    return F.conv1d(input,
                    weight,
                    bias,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups)


class SeparableConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[str, int] = 0,
                 dilation: int = 1,
                 use_bias: bool = True):
        super().__init__()
        self.use_bias: bool = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1))
        else:
            self.register_parameter('bias', None)

        self.depthwise = Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=in_channels,
                                bias=False)

        self.pointwise = Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                bias=False)

    def forward(self, inputs):
        out = self.pointwise(self.depthwise(inputs))
        if self.bias is not None:
            out += self.bias
        return out


class SeparableConv1dV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None, channels_last=False):
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
        in_shape = inputs.shape
        inputs = inputs.view(-1, in_shape[-2], in_shape[-1])
        inputs = self.channel_move(inputs)
        inputs = self.depthwise(inputs)
        if self.pointwise_groups == 1:
            inputs = self.pointwise(inputs.permute(0, 2, 1))
            if not self.channels_last:
                inputs = inputs.permute(0, 2, 1)
            inputs = inputs.view(*in_shape)
        else:
            inputs = self.pointwise(inputs)
            inputs = self.channel_move(inputs).view(*in_shape)
        return inputs



import time
import numpy as np

t = torch.randn(8, 768, 256)

# cnn = SeparableConv1d(768, 768, 9, 1)
cnn = SeparableConv1dV2(768, 768, 9, groups=768, pointwise_groups=12)

model_parameters = list(filter(lambda p: p.requires_grad, cnn.parameters()))
params = sum([np.prod(p.size()) for p in model_parameters])
print("Trainable Params = %s" % (params/1_000))

cnn = cnn.eval()


_ = [cnn(t) for _ in range(2)]
print(t.size(), cnn(t).size())
with profiler.profile(record_shapes=True) as prof:
    _ = [cnn(t) for _ in range(10)]


print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))


def test_conv_v1():
    t = torch.randn(8, 768, 256)
    # cnn = SeparableConv1d(768, 768, 9, 1)
    cnn = SeparableConv1dV2(768, 768, 9, groups=768, pointwise_groups=1)

    # cnn = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=9, groups=768, bias=True, stride=1, padding=9 // 2)
    cnn = cnn.eval()
    with torch.no_grad():
        _ = [cnn(t) for _ in range(10)]
        times = []

        for _ in range(1000):
            st = time.time()
            _ = cnn(t)
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.5f, variance = %.5f" % (np.mean(times), np.std(times)))


test_conv_v1()


