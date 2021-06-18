import copy
import sys
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import os
import argparse
import time
from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm, trange
from torch.optim import AdamW
import torch.distributed as dist
import traceback
from torch.multiprocessing import Process
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from torch.cuda.amp import GradScaler, autocast
from fastformer.data import *
from fastformer.config import *
from fastformer.optimizers import Novograd, RangerLars
from fastformer.data.dataset import datadict_iterator, MTTDataset
from fastformer.utils import *
from fastformer.model import *
from transformers import optimization
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import wandb
from pytz import timezone
from datetime import datetime, timedelta
from torch.utils.data.dataloader import DataLoader
from collections import Counter
from fastformer.model.mtt_model import MTTModel, MultiTaskHighwayCLSPretraining, get_mtt_backbone
import torchvision.transforms as transforms
import multiprocessing
import signal
from torch.multiprocessing.spawn import _prctl_pr_set_pdeathsig
from torch.multiprocessing import Process, ProcessContext

try:
    from gpustat import print_gpustat
except:
    def print_gpustat(*args, **kwargs):
        print("NO GPUSTAT")

try:
    from torch.cuda.amp import GradScaler, autocast
except:
    pass


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='Num Classes')
    parser.add_argument('--model_config', required=True, type=str,
                        help='model config')

    parser.add_argument('--wandb_name', required=False, type=str, default="",
                        help='wandb_name')

    parser.add_argument('--epochs', default=10, type=int,
                        help='Epochs')
    parser.add_argument('--cls_tokens', default=1, type=int,
                        help='cls_tokens')

    parser.add_argument('--lm_layers', type=int, required=False,
                        help='lm_layers')
    parser.add_argument('--electra_layers', type=int, required=False,
                        help='electra_layers')
    parser.add_argument('--lm_layers_total', type=int, required=False,
                        help='lm_layers_total')
    parser.add_argument('--electra_layers_total', type=int, required=False,
                        help='electra_layers_total')
    parser.add_argument('--drop_unused_layers', action="store_true", default=False,
                        help='drop_unused_layers')
    parser.add_argument('--consecutive_layers', action="store_true", default=False,
                        help='consecutive_layers')
    parser.add_argument('--exclude_layers', action="store_true", default=False,
                        help='exclude_layers')
    parser.add_argument('--sampling_alpha', type=float, required=False,
                        help='sampling_alpha weight')
    parser.add_argument('--schedule_sampling_alpha', action="store_true", default=False,
                        help='schedule_sampling_alpha')
    parser.add_argument('--keep_last_layer', action="store_true", default=False,
                        help='keep_last_layer')
    parser.add_argument('--move_unused_to_cpu', action="store_true", default=False,
                        help='move_unused_to_cpu')

    parser.add_argument('--start_from_proba', default=0.0, type=float,
                        help='start_from_proba')

    parser.add_argument('--lm_temperature', default=2.0, type=float,
                        help='lm_temperature')

    parser.add_argument('--total_steps', type=int, required=False,
                        help='total_steps')
    parser.add_argument('--freeze_last_layer', default=2, type=int,
                        help='freeze_last_layer')

    parser.add_argument('--batch_size', required=True, type=int,
                        help='Batch Size')

    parser.add_argument('--warmup_steps', default=1_000, type=int,
                        help='warmup_steps')
    parser.add_argument('--teacher_warmup_steps', default=1_000, type=int,
                        help='teacher_warmup_steps')
    parser.add_argument('--lr', default=optimizer_config.lr, type=float,
                        help='lr')
    parser.add_argument('--lr_steps', default=5, type=int,
                        help='lr_steps')
    parser.add_argument('--weight_decay', default=optimizer_config.weight_decay, type=float,
                        help='weight_decay')
    parser.add_argument('--gradient_clipping', default=optimizer_config.gradient_clipping, type=float,
                        help='gradient_clipping')
    parser.add_argument('--beta_1', default=optimizer_config.beta_1, type=float,
                        help='beta_1')
    parser.add_argument('--beta_2', default=optimizer_config.beta_2, type=float,
                        help='beta_2')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--model_save_dir', required=True, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=True, type=str,
                        help='Save Name')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--sentence_order_prediction_w', type=float, required=False, default=0.0,
                        help='sentence_order_prediction weight')

    parser.add_argument('--dino_w', type=float, required=False, default=0.0,
                        help='dino_w weight')

    parser.add_argument('--generator_w', type=float, required=False, default=0.0,
                        help='generator_w weight')

    parser.add_argument('--discriminator_w', type=float, required=False, default=0.0,
                        help='discriminator_w weight')

    parser.add_argument('--attention_penalty_w', type=float, required=False, default=0.0,
                        help='attention_penalty_w weight')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--no_autocast', action="store_true", default=False,
                        help='Avoid Autocast')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--backward_hook', action="store_true", default=False,
                        help='Backward Hook for gradients')

    parser.add_argument('--init_method', required=False, type=str, default="tcp",
                        help='init_method')

    parser.add_argument('--optimizer', required=False, type=str, default="adamw",
                        help='optimizer')

    parser.add_argument('--num_workers', required=False, type=int, default=0,
                        help='Dataloader workers')

    parser.add_argument('--master_addr', type=str, required='MASTER_ADDR' not in os.environ,
                        default=None if 'MASTER_ADDR' not in os.environ else os.environ['MASTER_ADDR'],
                        help='Master ADDR')
    parser.add_argument('--master_port', type=str, required='MASTER_PORT' not in os.environ,
                        default=None if 'MASTER_PORT' not in os.environ else os.environ['MASTER_PORT'],
                        help='Master PORT')
    parser.add_argument('--dist_backend', type=str, required=False,
                        default='nccl',
                        help='Distributed Backend')
    parser.add_argument('--log_every_steps', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--dataset', required=False, type=str,
                        help='Dataset')

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 61687
    args.seed = seed
    return vars(args)


def dataset_builder(location, params):
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    dataset = Dataset.load_from_disk(location)
    dataset = MTTDataset(dataset=dataset, **params)
    return dataset


def build_dataloader(location, shuffle_dataset, batch_size, tokenizer, cls_tokens, world_size=1, num_workers=None):
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    import os
    num_workers = min(max(os.cpu_count() // 2, 1), 4) if num_workers is None else num_workers

    student_args = dict(cls_tokens=cls_tokens, vocab_size=len(tokenizer), tokenizer=tokenizer,
                        tokenizer_args=dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512 - (cls_tokens - 1)),
                        word_mask_proba=((0, 0.15), (32, 0.15), (128, 0.2), (512, 0.2)),
                        max_span_length=1, max_jumbling_span_length=2, jumble_sentence=True)

    kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else dict()
    dataset = dataset_builder(location, student_args)
    train_loader = DataLoader(dataset, sampler=None if single_node else DistributedSampler(dataset, shuffle=shuffle_dataset),
                              batch_size=batch_size, shuffle=shuffle_dataset and single_node,
                              num_workers=num_workers, pin_memory=True, **kwargs)

    return train_loader


def train(local_rank, args):
    torch.backends.cudnn.benchmark = True
    import os
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # too many barriers / one node data parallel and multiple node DDP
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ["NCCL_DEBUG"] = "WARN"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # gpu_device = 0
    gpu_device = local_rank
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args["wandb_dryrun"]:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_SILENT"] = "true"
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    torch.backends.cudnn.benchmark = True
    rank = args["nr"] if args["cpu"] else (args["nr"] * args["gpus_per_node"] + local_rank)
    nr = args["nr"]
    if args["cpu"]:
        assert local_rank == 0
        device = torch.device("cpu")
        args["dist_backend"] = "gloo"
        # init_method = "tcp://%s:%s" % ("127.0.0.1", "9999")
    else:
        device = torch.device(f'cuda:{gpu_device}')  # Unique only on individual node.
        torch.cuda.set_device(device)
    if args["init_method"] == "tcp":
        if args["nr"] == 0:
            args["master_addr"] = "0.0.0.0"
        init_method="tcp://%s:%s" % (args["master_addr"], args["master_port"])
    elif args["init_method"] == "file":
        init_method = 'file://%s/%s' % (args["master_addr"], args["master_port"])
    else:
        raise ValueError

    rnd = torch.tensor(0.0, device="cpu")
    if args["world_size"] > 1:
        dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
        rnd = torch.tensor(int(time.time())).to(device)
        dist.broadcast(rnd, 0)
    barrier = get_barrier(args["world_size"] > 1)
    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(timezone('Asia/Kolkata')).strftime(format)
    ds_name = list(filter(lambda x: len(x.strip()) > 0, args["dataset"].split("/")))[-1].replace("train_fastformer_resampled_", "")
    set_seeds(args["seed"])
    batch_size = 8

    optimizer_config.lr = args["lr"]
    optimizer_config.weight_decay = args["weight_decay"]
    optimizer_config.gradient_clipping = args["gradient_clipping"]
    optimizer_config.beta_1 = args["beta_1"]
    optimizer_config.beta_2 = args["beta_2"]

    eps = 1e-4
    if args["no_autocast"]:
        optimizer_config.eps = 1e-7
        eps = 1e-7

    reinit = args["pretrained_model"] is None or "pretrained_model" not in args or args["pretrained_model"] == ""
    backbone, tokenizer = get_mtt_backbone(args["model_config"], args["cls_tokens"], args["consecutive_layers"], args["sampling_alpha"], reinit)
    teacher_backbone, _ = get_mtt_backbone(args["model_config"], args["cls_tokens"], False, None, reinit)

    batch_size = args["batch_size"] if "batch_size" in args and isinstance(args["batch_size"], int) else batch_size
    generator_w = args["generator_w"] if "generator_w" in args else 0.0
    discriminator_w = args["discriminator_w"] if "discriminator_w" in args else 0.0
    dino_w = args["dino_w"] if "dino_w" in args else 0.0
    sentence_order_prediction_w = args["sentence_order_prediction_w"] if "sentence_order_prediction_w" in args else 0.0
    attention_penalty_w = args["attention_penalty_w"] if "attention_penalty_w" in args else 0.0

    student = MTTModel(backbone, tokenizer, args["cls_tokens"],
                       generator_w=generator_w, discriminator_w=discriminator_w,
                       dino_w=dino_w, sentence_order_prediction_w=sentence_order_prediction_w, attention_penalty_w=attention_penalty_w,
                       lm_layers=args["lm_layers"], electra_layers=args["electra_layers"],
                       lm_layers_total=args["lm_layers_total"], electra_layers_total=args["electra_layers_total"],
                       drop_unused_layers=args["drop_unused_layers"], approximate_unused_layers=args["consecutive_layers"],
                       exclude_layers=args["exclude_layers"], keep_last_layer=args["keep_last_layer"],
                       lm_temperature=args["lm_temperature"])
    teacher = MTTModel(teacher_backbone, tokenizer, args["cls_tokens"],
                       generator_w=0.0, discriminator_w=0.0,
                       dino_w=1.0, sentence_order_prediction_w=0.0, attention_penalty_w=0.0,
                       lm_layers=None, electra_layers=None,
                       lm_layers_total=args["lm_layers_total"], electra_layers_total=args["electra_layers_total"],
                       lm_temperature=args["lm_temperature"])
    teacher = teacher.eval()
    model = MultiTaskHighwayCLSPretraining(student, teacher, eps, device if args["move_unused_to_cpu"] else None).to(device)
    trainable_model = student

    if dino_w == 0:
        model.teacher = None
        teacher = None
        clean_memory()
    del teacher
    if local_rank == 0 and rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(trainable_model) / 1_000_000))

    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]):
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)

        try:
            trainable_model.load_state_dict(state_dict, strict=True)
            load_type = "strict"
        except:
            try:
                try:
                    trainable_model.load_state_dict(state_dict, strict=False)
                    load_type = "not_strict"
                except:
                    state_dict = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
                    trainable_model.load_state_dict(state_dict, strict=False)
                    load_type = "not_strict_no_ffn"
            except:
                try:
                    try:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        trainable_model.load_state_dict(state_dict, strict=True)
                        load_type = "strict-from-ddp"
                    except:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
                        trainable_model.load_state_dict(state_dict, strict=True)
                        load_type = "strict-from-ddp-no-ffn"
                except:
                    try:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        trainable_model.load_state_dict(state_dict, strict=False)
                        load_type = "not_strict-from-ddp"
                    except:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
                        trainable_model.load_state_dict(state_dict, strict=False)
                        load_type = "not_strict-from-ddp-no-ffn"
        if dino_w > 0:
            student_teacher_param_update(model.student, model.teacher, 0.001, device if args["move_unused_to_cpu"] else None)

        print("[Train]: Time = %s, Loaded Pretrained model with Load type = %s, Torch Version = %s" % (get_time_string(), load_type, torch.__version__))
        del state_dict
    model = model.train()

    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in model.named_parameters() if v.requires_grad}))
    if args["world_size"] > 1:
        # model = FSDP(model, **fsdp_params)  # find_unused_parameters=True

        trainable_model = DDP(trainable_model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=True, bucket_cap_mb=50)  # find_unused_parameters=True
        model.student = trainable_model

    if dino_w > 0:
        model.teacher = model.teacher.eval()
        student_teacher_param_update(model.student, model.teacher, 0.01, device if args["move_unused_to_cpu"] else None)
    try:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        trainable_model.register_comm_hook(state=None, hook=fp16_compress_hook)

    except:
        print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))

    del backbone
    del teacher_backbone
    del student
    clean_memory()
    barrier()
    optc = optimizer_config.to_dict()
    trainable_params = list(filter(lambda p: p.requires_grad, trainable_model.parameters()))
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"],
                                      betas=(optc["beta_1"], optc["beta_2"]))
    elif args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=optc["lr"], momentum=0.9, weight_decay=optc["weight_decay"], nesterov=True)
    elif args["optimizer"] == "novograd":
        optimizer = Novograd(trainable_params, lr=optc["lr"], eps=optc["eps"], betas=(optc["beta_1"], optc["beta_2"]), weight_decay=optc["weight_decay"],)
    elif args["optimizer"] == "rangerlars":
        optimizer = RangerLars(trainable_params, lr=optc["lr"], eps=optc["eps"], betas=(optc["beta_1"], optc["beta_2"]), weight_decay=optc["weight_decay"],)
    else:
        raise ValueError
    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in trainable_model.named_parameters() if v.requires_grad}))
    del trainable_params
    optimizer.zero_grad(set_to_none=True)
    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]

    set_seeds(args["seed"] + rank)
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
    dataloader = build_dataloader(args["dataset"], args["shuffle_dataset"], batch_size,
                                  tokenizer, args["cls_tokens"],
                                  world_size=args["world_size"], num_workers=args["num_workers"])
    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    iter_size = max(args["accumulation_steps"], 1)
    no_sync = iter_size > 1
    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    # scheduler = optimization.get_linear_schedule_with_warmup(optimizer, optc["warmup_steps"], args["epochs"] * len(dataloader))
    steps_per_epoch = int(np.ceil(len(dataloader.sampler) / (batch_size * iter_size)) if dataloader.sampler is not None else (len(dataloader) / iter_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], epochs=args["epochs"], steps_per_epoch=steps_per_epoch,
                                                    div_factor=1e4, three_phase=False, pct_start=0.06, anneal_strategy="linear", cycle_momentum=False)

    # scheduler1 = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(steps_per_epoch * args["epochs"]) // args["lr_steps"], gamma=0.5)
    # scheduler = [scheduler1, scheduler2]
    if local_rank == 0:
        print("[Train]: Time = %s, Optimizer and Scheduler Initialised, max lr = %.5f, epochs = %s, steps_per_epoch = %s, batch size = %s, dataloader length = %s, Sampler Present = %s, Sampler Length = %s" %
              (get_time_string(), optc["lr"], args["epochs"], steps_per_epoch, batch_size, len(dataloader), dataloader.sampler is not None, len(dataloader.sampler) if dataloader.sampler is not None else -1))

    barrier()

    gradient_clipping = optc["gradient_clipping"]

    if local_rank == 0:
        group = "%s-%s-%s-%sN-%s" % (args["wandb_name"], ds_name, args["model_config"], args["nodes"], time_string)
        wandb_init_args = dict(project="de_lm", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                               config={"args":args, "optimizer_config": optc},
                               settings=wandb.Settings(start_method="fork"))

        time.sleep(random.random() * 5)
        wandb.init(**wandb_init_args)
        print("[Train]: WandB-watch added over model for Rank = %s" % rank)

    full_times = []
    batch_times = []
    model_times = []
    model.zero_grad(set_to_none=True)
    samples_processed = 0
    samples_processed_this_log_iter = 0
    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)

    def hook(grad):
        is_nan_inf = torch.logical_not(torch.isfinite(grad))
        if is_nan_inf.any():
            # print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
            grad[is_nan_inf] = 0.0
            return grad
        return None

    if not args["no_autocast"] and args["backward_hook"]:
        for name, param in model.named_parameters():
            param.register_hook(hook)


    dino_center = torch.zeros(model.dino_dims, device=device) if dino_w > 0 else None
    discriminator_dino_center = torch.zeros(model.dino_dims, device=device) if dino_w > 0 else None
    total_steps = args["epochs"] * len(dataloader)
    for epoch in range(args["epochs"]):

        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        else:
            print("Time = %s: Unable to set Epoch = %s" % (get_time_string(), epoch))

        start_time = time.time()
        for step, batch in enumerate(dataloader):

            steps_done = epoch * len(dataloader) + step
            teacher_update_w = np.interp(steps_done, [0, args["teacher_warmup_steps"]], [0.95, 0.999])
            inner_model = getattr(trainable_model, "module", trainable_model)
            if hasattr(inner_model, "start_from_proba"):
                start_from_proba = np.interp(steps_done, [0, args["warmup_steps"], args["warmup_steps"] * 2], [0.0, 0.0, args["start_from_proba"]])
                inner_model.start_from_proba = start_from_proba
            if hasattr(inner_model.backbone.encoder, "sampling_alpha") and args["sampling_alpha"] is not None and args["sampling_alpha"] != 1.0:
                sampling_alpha = np.interp(steps_done, [0, args["warmup_steps"], args["warmup_steps"] * 2], [1.0, 1.0, args["sampling_alpha"]])
                inner_model.backbone.encoder.sampling_alpha = sampling_alpha
                inner_model.sampling_alpha = sampling_alpha

            # Beta updates for AdamW
            # beta_1 = np.interp(steps_done, [0, args["warmup_steps"]], [optc["beta_1"], 0.9])
            # beta_2 = np.interp(steps_done, [0, args["warmup_steps"]], [optc["beta_2"], 0.98])
            # optimizer.param_groups[0]["betas"] = (beta_1, beta_2)
            #
            #
            # generator_w_progressive = generator_w
            # discriminator_w_progressive = discriminator_w
            #
            # model.generator_w = generator_w_progressive
            # model.discriminator_w = discriminator_w_progressive

            if isinstance(batch, dict):
                key = list(batch.keys())[0]
                bs_size = list(batch[key].size())
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
            else:
                batch[0] = batch[0].to(device, non_blocking=True)
                batch[1] = batch[1].to(device, non_blocking=True)
                bs_size = list(batch[0].size())
                key = 0

            gen_batch_time = time.time() - start_time
            batch_times.append(gen_batch_time)
            if (steps_done + 1) % save_every_steps == 0 or (args["total_steps"] is not None and (steps_done + 1) >= args["total_steps"]):
                state_dict = trainable_model.state_dict() if not isinstance(trainable_model, DDP) else trainable_model.module.state_dict()
                if local_rank == 0:
                    torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
                del state_dict
                clean_memory()
                barrier()
                if args["total_steps"] is not None and (steps_done + 1) >= args["total_steps"]:
                    return

            samples_processed += int(batch[key].size(0))
            samples_processed_this_log_iter += int(batch[key].size(0))
            inner_args = dict(no_autocast=args["no_autocast"], cpu=args["cpu"])
            validation_iter = (steps_done + 1) % log_every_steps == 0 or step == 0
            model_start = time.time()
            if no_sync and (step + 1) % iter_size != 0 and hasattr(trainable_model, "no_sync"):
                with trainable_model.no_sync():
                    output = train_inner_loop(inner_args, model, batch, optimizer,
                                              scheduler, gradient_clipping, iter_size=iter_size,
                                              no_sync=True, validation_iter=validation_iter,
                                              dino_center=dino_center,
                                              discriminator_dino_center=discriminator_dino_center,
                                              freeze_last_layer=epoch < args["freeze_last_layer"], step=steps_done + 1)
                model_times.append(time.time() - model_start)
            else:
                output = train_inner_loop(inner_args, model, batch, optimizer,
                                          scheduler, gradient_clipping, iter_size=iter_size,
                                          no_sync=False, validation_iter=validation_iter,
                                          dino_center=dino_center,
                                          discriminator_dino_center=discriminator_dino_center,
                                          freeze_last_layer=epoch < args["freeze_last_layer"], step=steps_done + 1)
                optimizer.zero_grad(set_to_none=True)
                model_times.append(time.time() - model_start)

            del batch
            if dino_w > 0 and (step + 1) % iter_size:
                student_teacher_param_update(model.student, model.teacher, teacher_update_w, device if args["move_unused_to_cpu"] else None)
            dino_center = output.pop("dino_center", None)
            discriminator_dino_center = output.pop("discriminator_dino_center", None)
            if dino_w > 0 and (step + 1) % (4 * iter_size) == 0 and args["world_size"] > 1:
                if dino_center is not None:
                    dtype = dino_center.dtype
                    dino_center = dino_center.type(torch.float64) / args["world_size"]
                    torch.distributed.all_reduce(dino_center, torch.distributed.ReduceOp.SUM)
                    dino_center = dino_center.type(dtype)
                if discriminator_dino_center is not None:
                    dtype = discriminator_dino_center.dtype
                    discriminator_dino_center = discriminator_dino_center.type(torch.float64) / args["world_size"]
                    torch.distributed.all_reduce(discriminator_dino_center, torch.distributed.ReduceOp.SUM)
                    discriminator_dino_center = discriminator_dino_center.type(dtype)
            if (step + 1) % (4 * iter_size) == 0 and hasattr(getattr(trainable_model, "module", trainable_model).backbone, "layer_normalizers") and args["world_size"] > 1:
                layer_normalizers = getattr(trainable_model, "module", trainable_model).backbone.layer_normalizers
                dtype = layer_normalizers.dtype
                layer_normalizers = layer_normalizers.type(torch.float64)
                torch.distributed.all_reduce(layer_normalizers, torch.distributed.ReduceOp.SUM)
                layer_normalizers = layer_normalizers / args["world_size"]
                getattr(trainable_model, "module", trainable_model).backbone.layer_normalizers = layer_normalizers.type(dtype)

            full_time = time.time() - start_time
            full_times.append(full_time)
            if step == 0 and epoch == 0 and local_rank == 0:
                print("[Train]: Time = %s, First Batch Training for Rank = %s" % (get_time_string(), rank))
            if (steps_done + 1) % log_every_steps == 0 or step == 0:
                if local_rank == 0:

                    steps_remaining = total_steps - steps_done
                    # print({k for k, v in output.items() if isinstance(v, torch.Tensor)})
                    output = {k: float(v) if v else v for k, v in output.items()}
                    samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
                    wandb_log = dict(lr=optimizer.param_groups[0]['lr'], epoch=epoch+1, step=step, samples_processed=samples_processed, samples_per_second=samples_per_second,
                                     batch_times=np.mean(batch_times), full_times=np.mean(full_times), model_times=np.mean(model_times), steps_remaining=steps_remaining, pct_complete=(100 * steps_done / total_steps),
                                     **{k: v for k, v in output.items() if v is not None})

                    wandb.log(wandb_log)
                    print("[Train]: Time = %s, Epoch = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Details = %s, LR = %s" %
                          (get_time_string(), epoch+1, rank, step, samples_processed, bs_size, output, optimizer.param_groups[0]['lr']))
                    print("[Train-Timings]: Time = %s, Batch time = %.4f, Full Time = %.4f, Model Time = %.4f, samples_per_second = %s, steps_remaining = %s, pct_complete = %.4f" % (
                        get_time_string(), np.mean(batch_times), np.mean(full_times), np.mean(model_times), samples_per_second, steps_remaining, (100 * steps_done / total_steps),))
                batch_times = []
                full_times = []
                model_times = []
                samples_processed_this_log_iter = 0

                # clean_memory()
                # barrier()
            del output
            del bs_size
            start_time = time.time()
    print("Time = %s, Finished Training for Rank = %s" % (get_time_string(), rank))
    state_dict = trainable_model.state_dict() if not isinstance(trainable_model, DDP) else trainable_model.module.state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
    del model


def forward_backward(is_fp16, batch, validation_iter, dino_center, discriminator_dino_center, step, ddp_model, freeze_last_layer, iter_size, scaler):
    with autocast(is_fp16):
        output = ddp_model(**batch, validation_iter=validation_iter, dino_center=dino_center, discriminator_dino_center=discriminator_dino_center, rng_seed=step)
        last_layer = ddp_model.get_last_dino_layer()
        if freeze_last_layer and last_layer is not None:
            for p in last_layer.parameters():
                p.grad = None

    loss = output.pop("loss") / iter_size
    if is_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    _ = output.pop("predictions", None)
    loss_float = loss.item()
    del loss
    return output, loss_float


def train_inner_loop(args, ddp_model, batch, optimizer, scheduler, gradient_clipping, iter_size=1,
                     no_sync=False, validation_iter=False, dino_center=None, discriminator_dino_center=None,
                     scaler=None, freeze_last_layer=False, step=None):
    trainable_model = ddp_model.student if isinstance(ddp_model, (PatchCLR, MultiTaskHighwayCLSPretraining)) else ddp_model
    is_fp16 = isinstance(trainable_model, DDP) and scaler is not None
    if no_sync:
        with trainable_model.no_sync():
            output, loss_float = forward_backward(is_fp16, batch, validation_iter, dino_center, discriminator_dino_center, step, ddp_model, freeze_last_layer,
                                                  iter_size, scaler)
    else:
        output, loss_float = forward_backward(is_fp16, batch, validation_iter, dino_center, discriminator_dino_center, step, ddp_model, freeze_last_layer,
                                              iter_size, scaler)

    # print([name for name, params in (ddp_model.student if isinstance(ddp_model, (PatchCLR, MultiTaskHighwayCLSPretraining)) else ddp_model).named_parameters() if params.grad is None])
    if not no_sync:
        if is_fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            if isinstance(scheduler, list):
                for sch in scheduler:
                    sch.step()
            else:
                scheduler.step()
        else:
            trainable_model.clip_grad_norm_(gradient_clipping) if isinstance(trainable_model, FSDP) else torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), gradient_clipping)
            optimizer.step()
            if isinstance(scheduler, list):
                for sch in scheduler:
                    sch.step()
            else:
                scheduler.step()

    if np.isnan(loss_float):
        es = "[Train-Exception]: Time = %s, NAN Loss, Scale = %s, loss_dict = %s, lr = %s" % (
            get_time_string(), None, loss_float, optimizer.param_groups[0]['lr'])
        raise ValueError(es)
    _ = output.pop("logits", None)
    _ = output.pop("predictions", None)
    return dict(loss=loss_float, **output)


def train_catch_exception(local_rank, args):
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    nr = args["nr"]
    try:
        train(local_rank, args)
    except Exception as e:
        import traceback
        print("[Exception-in-train]: Node Rank = %s, Local Rank = %s, Rank = %s, Exception = %s, \n Trace = %s" % (nr, local_rank, rank, e, traceback.format_exc()))
        # traceback.print_tb(e.__traceback__)
        # traceback.print_exception(*sys.exc_info())
        traceback.print_exc()
        raise e


def _wrap(fn, i, args, error_queue):
    fn = dill.loads(fn)
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(i, *args)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(dill.dumps(fn), i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)
        # start_processes(train, (args,), args["gpus_per_node"], True, False, start_method='spawn')



