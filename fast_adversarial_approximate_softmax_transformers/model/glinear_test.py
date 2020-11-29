import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
import torch.autograd.profiler as profiler


class GLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups=1,
                 bias: bool = True) -> None:
        super(GLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # groups
        self.groups = groups
        self.group_in_dim = self.in_features // self.groups
        self.group_out_dim = self.out_features // self.groups
        self.weight = nn.Parameter(
            torch.Tensor(self.groups, self.group_in_dim, self.group_out_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.in_features
        bs = input.shape[0]
        input = input.view(-1, self.groups, self.group_in_dim).transpose(0, 1)
        outputs = torch.matmul(input,
                               self.weight).transpose(0, 1).contiguous().view(
                                   bs, -1, self.out_features)
        if self.bias is not None:
            outputs += self.bias
        return outputs


class GLinearV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups=1,
                 bias: bool = True) -> None:
        super(GLinearV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # groups
        self.groups = groups
        self.group_in_dim = self.in_features // self.groups
        self.group_out_dim = self.out_features // self.groups
        self.weight = nn.Parameter(
            torch.Tensor(self.groups, self.group_in_dim, self.group_out_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bs = input.shape[0]
        input = input.view(-1, self.groups, self.group_in_dim).transpose(0, 1)
        outputs = torch.matmul(input,
                               self.weight).transpose(0, 1).reshape(
                                   bs, -1, self.out_features)
        if self.bias is not None:
            outputs += self.bias
        return outputs


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


import time
import numpy as np


def test_conv_v1():
    t = torch.randn(8, 512, 768)
    cnn = Conv1d(768, 768, 1, 4)
    # cnn = GLinear(768, 768, 1)
    # cnn = nn.Linear(768, 768)
    cnn = cnn.eval()
    with torch.no_grad():
        _ = [cnn(t) for _ in range(1)]
        times = []

        for _ in range(10):
            st = time.time()
            _ = cnn(t)
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.5f, variance = %.5f" % (np.mean(times), np.std(times)))


# test_conv_v1()


t = torch.randn(8, 512, 768)
# cnn = Conv1d(768, 768, 1, 4)
cnn = GLinear(768, 768, 4)
# cnn = nn.Linear(768, 768)

model_parameters = list(filter(lambda p: p.requires_grad, cnn.parameters()))
params = sum([np.prod(p.size()) for p in model_parameters])
print("Trainable Params = %s" % (params/1_000))

cnn = cnn.eval()


_ = [cnn(t) for _ in range(1)]
with profiler.profile(record_shapes=True) as prof:
    _ = [cnn(t) for _ in range(5)]


print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
