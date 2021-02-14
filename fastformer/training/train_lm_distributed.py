# TODO: support multiple datasets via config
# TODO: support all model types including normal BERT / Roberta
# TODO: Wandb monitoring for master node rank - 0 process
# TODO: model checkpointing for root process
# TODO: base on huggingface run_lm.py
# TODO: support passing hyperpraram config file
# TODO: CPU and GPU training support
# TODO: save model
# TODO: Resume training from saved checkpoint
# TODO: Use TQDM and progress bar as well as metrics for speed

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import os
import argparse
import time
from tqdm.auto import tqdm, trange
from torch.optim import AdamW
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--model_config', required=True, type=str,
                        help='model config')
    parser.add_argument('--train_config', required=True, type=str,
                        help='Train config')
    parser.add_argument('--master_addr', type=str, required='MASTER_ADDR' not in os.environ,
                        default=None if 'MASTER_ADDR' not in os.environ else os.environ['MASTER_ADDR'],
                        help='Master ADDR')
    parser.add_argument('--master_port', type=str, required='MASTER_PORT' not in os.environ,
                        default=None if 'MASTER_PORT' not in os.environ else os.environ['MASTER_PORT'],
                        help='Master PORT')
    parser.add_argument('--dist_backend', type=str, required=False,
                        default='nccl',
                        help='Distributed Backend')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_PORT

    seed = 0
    args.seed = seed

    return args


def validate_mlm_dataset(model, dataset):
    pass


def validate_qna_dataset(model, dataset):
    pass


def validate_superglue(model, datasets):
    pass


def cleanup():
    # save model
    dist.destroy_process_group()


def train(local_rank, args):
    # Build dataset and dataloader with distributed sampler
    # Build model with DDP
    rank = args.nr * args.gpus + local_rank
    dist.init_process_group(args.dist_backend, rank=rank, world_size=args.world_size)
    device = torch.device(f'cuda:{local_rank}')  # Unique only on individual node.
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Check if all initialised model weights are same??

    pass


if __name__ == "__main__":
    args = training_args()

    try:
        mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
    finally:
        cleanup()

