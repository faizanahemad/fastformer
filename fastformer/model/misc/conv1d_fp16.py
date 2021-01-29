import copy
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import random
import torch
from torch import nn

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              groups=groups, bias=bias, stride=stride, dilation=dilation)

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


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        if groups is None:
            groups = in_channels
        if pointwise_groups is None:
            pointwise_groups = 1
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=groups, bias=False, stride=stride, padding=padding)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, groups=pointwise_groups, bias=bias, stride=1, padding=0)

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


class SeparableConv1dV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=groups, bias=False, stride=stride, padding=padding)
        self.pointwise = nn.Linear(in_channels, out_channels, bias=bias)

        self.out_channels = out_channels

    def forward(self, inputs):
        """
        Expect inputs in channels first format
        :param inputs:
        :return:
        """
        inputs = inputs.permute(0, 2, 1)
        inputs = self.depthwise(inputs)
        inputs = inputs.permute(0, 2, 1)
        inputs = self.pointwise(inputs)
        return inputs


class SeparableConv1dV3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None, pointwise_groups=None,
                 bias=True, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=1, bias=False, stride=stride, padding=padding)

        self.out_channels = out_channels

    def forward(self, inputs):
        """
        Expect inputs in channels first format
        :param inputs:
        :return:
        """
        inputs = inputs.permute(0, 2, 1)
        inputs = self.depthwise(inputs)
        inputs = inputs.permute(0, 2, 1)
        return inputs


if __name__ == "__main__":
    import time
    import argparse
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW



    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default='cpu',
                    help="Device")
    ap.add_argument("--forward_only", type=str2bool, default=False)
    ap.add_argument("--fp16", type=str2bool, default=False)
    ap.add_argument("--profile", type=str2bool, default=False)
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--kernel", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--model", type=str, default="conv")
    ap.add_argument("--cudnn_benchmark", type=str2bool, default=False)

    args = vars(ap.parse_args())
    forward_only = args["forward_only"]
    device = args["device"]
    profile = args["profile"]
    fp16 = args["fp16"]
    channels = args["channels"]
    groups = args["groups"]
    kernel = args["kernel"]
    batch_size = args["batch_size"]
    stride = args["stride"]
    model = args["model"]
    cudnn_benchmark = args["cudnn_benchmark"]
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if model == "conv":
        model = Conv1d(channels, channels, kernel, groups).to(device)
    elif model == "sep":
        model = SeparableConv1d(channels, channels, kernel, pointwise_groups=groups, stride=stride).to(device)
    elif model == "sep_v2":
        model = SeparableConv1dV2(channels, channels, kernel, pointwise_groups=groups, stride=stride).to(device)
    elif model == "sep_v3":
        model = SeparableConv1dV3(channels, channels, kernel, pointwise_groups=groups, stride=stride).to(device)
    elif model == "linear":
        model = nn.Linear(channels, channels).to(device)
    else:
        raise ValueError

    try:
        from torch.cuda.amp import GradScaler, autocast
    
        scaler = GradScaler()
    except:
        pass

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(model)
    print("Trainable Params = %s" % (numel(model) / 1_000))

    if forward_only:
        _ = model.eval()
    else:
        _ = model.train()


    tensor = torch.randn(batch_size, 512, channels).to(device)

    epochs = 100

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(all_params, lr=5e-4, eps=1e-6, weight_decay=1e-2)

    def run():
        if not forward_only:
            if fp16:
                lv = torch.randn(1, device=device)
                with autocast():
                    output = model(tensor)
                    loss = ((output - lv) ** 2).mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(tensor)
                loss = ((output - torch.randn(1, device=device)) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            if fp16:
                with autocast():
                    with torch.no_grad():
                        pt_outputs = model(tensor)

            else:
                with torch.no_grad():
                    pt_outputs = model(tensor)
            return pt_outputs


    if profile:
        import torch.autograd.profiler as profiler

        _ = [run() for _ in range(10)]
        with profiler.profile(record_shapes=True) as prof:
            _ = [run() for _ in range(epochs)]
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
    else:
        _ = [run() for _ in range(10)]
        times = []
        for _ in trange(epochs):
            st = time.time()
            _ = run()
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.4f, Lowest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.std(times)))

