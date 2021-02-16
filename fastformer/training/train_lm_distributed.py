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
from torch.cuda.amp import GradScaler, autocast
from fastformer.data import *
from fastformer.config import *
from fastformer.utils import *
from fastformer.model import *


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--model_config', required=True, type=str,
                        help='model config')
    parser.add_argument('--optimizer_config', required=True, type=str,
                        help='Optimizer config')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')
    parser.add_argument('--model_save_dir', required=True, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=True, type=str,
                        help='Save Name')

    parser.add_argument('--validate_only', required=False, type=str2bool, default=False,
                        help='Validate Only')

    parser.add_argument('--test_only', required=False, type=str2bool, default=False,
                        help='Test Only')

    parser.add_argument('--train_dataset', required=False, type=str,
                        help='Train Dataset')

    parser.add_argument('--validation_dataset', required=False, type=str,
                        help='Validation Dataset')

    parser.add_argument('--test_fastformer', required=False, type=str,
                        help='Test Dataset')

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
    args.world_size = args.gpus_per_node * args.nodes
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

    dist.destroy_process_group()


def train(local_rank, args):
    # Build dataset and dataloader with distributed sampler
    # Build model with DDP
    rank = args.nr * args.gpus_per_node + local_rank
    dist.init_process_group(args.dist_backend, rank=rank, world_size=args.world_size)
    device = torch.device(f'cuda:{local_rank}')  # Unique only on individual node.
    torch.cuda.set_device(device)

    set_seeds(args.seed)

    model_config = dict(**args["model_config"])
    config = dict(md_config=md_config, sm_config=sm_config, lg_config=lg_config)[model_config.pop("model_size")]
    optimizer_config = dict(md_config=md_config, sm_config=sm_config, lg_config=lg_config)[args["optimizer_config"]]
    tokenizer = get_tokenizer(args["model_config"]["tokenizer_name"])
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = 1024
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens

    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)
    char_to_id = sorted([k for k, v in AutoTokenizer.from_pretrained("bert-base-uncased").get_vocab().items() if len(k) == 1]) + [" ", "\n"]
    char_to_id = dict(zip(char_to_id, range(2, len(char_to_id) + 2)))

    model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **model_config).to(device)
    if args["pretrained_model"] is not None:
        model.load_state_dict(torch.load(args["pretrained_model"], map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank}))
    # Take model to local rank
    ddp_model = DDP(model, device_ids=[rank])
    all_params = list(filter(lambda p: p.requires_grad, ddp_model.parameters()))
    optc = optimizer_config
    optimizer = AdamW(all_params, lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"]))
    optimizer.zero_grad()
    scaler = GradScaler()

    model_save_dir = args["model_save_dir"]
    assert os.path.exists(model_save_dir)
    model_save_name = args["model_save_name"]
    # Take inputs to local_rank

    # TODO: validate on multigpu, sort the val datasets alphabetically and let the gpu with rank == dataset rank in sort pick up the dataset. GPUs with rank > len(datasetDict) stay idle.
    # TODO: select one dataset and make full batch from it, this way rebalancing can be easy.
    # TODO: dataset rebalancing.
    # TODO: save model only in local_rank == 0 process
    # TODO: Check if all initialised model weights are same??
    # I've been tracking an ema of sample training loss during training and using that to guide weighted data sampling (rather than the typical uniform sampling). Seems to help with a variety of real world datasets where the bulk of the data is often very similar and easy to learn but certain subpopulations are much more challenging.

    pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = training_args()

    try:
        mp.spawn(train, nprocs=args.gpus_per_node, args=(args,), join=True)
    finally:
        cleanup()

