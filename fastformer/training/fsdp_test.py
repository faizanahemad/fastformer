import argparse
import os
from datetime import datetime, timedelta
from pytz import timezone
import time

from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
from fairscale.nn.misc import checkpoint_wrapper
from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from tqdm import trange


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


def get_time_string():
    # + timedelta(hours=5, minutes=30)
    return (datetime.fromtimestamp(time.mktime(time.gmtime(time.time())))).astimezone(timezone('Asia/Kolkata')).strftime("[%a, %d %b %Y, %H:%M:%S %Z]")


def main(local_rank, *args):
    torch.backends.cudnn.benchmark = True
    init_method = "tcp://%s:%s" % ("0.0.0.0", "9999")
    torch.distributed.init_process_group(backend="nccl", rank=local_rank, world_size=8, init_method=init_method)
    print("[Train]: Time = %s, Initialized Dist Process for Rank = %s" % (get_time_string(), local_rank))
    device = torch.device(f'cuda:{local_rank}')  # Unique only on individual node.
    torch.cuda.set_device(device)
    torch.cuda.set_device(device)
    fsdp_params = dict(mixed_precision=True, flatten_parameters=True,
                       bucket_cap_mb=25, reshard_after_forward=False, fp32_reduce_scatter=False,
                       cpu_offload=False, move_grads_to_cpu=False, process_group=torch.distributed.group.WORLD)
    nn_model = nn.Sequential(nn.Linear(200, 200),
                             FullyShardedDDP(checkpoint_wrapper(nn.Linear(200, 200), offload_to_cpu=True), **fsdp_params),
                             checkpoint_wrapper(nn.GELU(), offload_to_cpu=True),
                             nn.LayerNorm(200, eps=1e-7),
                             nn.Linear(200, 64)
                             ).cuda()

    model = FullyShardedDDP(nn_model, **fsdp_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-7, weight_decay=1e-2,
                                  betas=(0.9, 0.99))
    optimizer.zero_grad(set_to_none=True)

    for i in range(1000):
        optimizer.zero_grad(set_to_none=True)
        fake_inputs = torch.randn(32, 200, device=device)
        fake_labels = torch.randn(32, 64, device=device)
        outputs = model(fake_inputs)
        loss = ((outputs - fake_labels) ** 2).mean()
        loss.backward()
        model.clip_grad_norm_(1.0)
        optimizer.step()
        if i % 100 == 0:
            print("Loss = %s, rank = %s" % (loss.item(), local_rank))

    state_dict = model.state_dict()
    nn_model = nn.Sequential(nn.Linear(200, 200),
                             nn.Linear(200, 200),
                             checkpoint_wrapper(nn.GELU(), offload_to_cpu=True),
                             nn.LayerNorm(200, eps=1e-7),
                             nn.Linear(200, 64)
                             ).cuda()
    nn_model.load_state_dict(state_dict)
    print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(nn_model) / 1_000_000))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main, nprocs=8, args=(), join=True)

