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
from fastformer.data.dataset import datadict_iterator
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
from fastformer.model.fastformer_vision_model import FastFormerVisionModel, PatchCLR, ClassificationModel
import torchvision.transforms as transforms

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

    parser.add_argument('--epochs', default=10, type=int,
                        help='Epochs')
    parser.add_argument('--pct_simclr_simple', default=20, type=int,
                        help='pct_simclr_simple')

    parser.add_argument('--batch_size', required=True, type=int,
                        help='Batch Size')
    parser.add_argument('--warm_restart_key_encoder', required=False, type=int,
                        help='Warm Restarts Epochs for Key Encoder')
    parser.add_argument('--total_samples_patchclr', default=131_072*2, type=int,
                        help='Patch Negatives')
    parser.add_argument('--total_samples_simclr', default=131_072*2, type=int,
                        help='SimCLR Negatives')

    parser.add_argument('--warmup_steps', default=optimizer_config.warmup_steps, type=int,
                        help='warmup_steps')
    parser.add_argument('--lr', default=optimizer_config.lr, type=float,
                        help='lr')
    parser.add_argument('--weight_decay', default=optimizer_config.weight_decay, type=float,
                        help='weight_decay')
    parser.add_argument('--gradient_clipping', default=optimizer_config.gradient_clipping, type=float,
                        help='gradient_clipping')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')

    parser.add_argument('--patchclr_heads', default=1, type=int,
                        help='Patchclr heads')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--model_save_dir', required=True, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=True, type=str,
                        help='Save Name')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--moco', action="store_true", default=False,
                        help='Momentum Contrast')

    parser.add_argument('--simclr_moco', action="store_true", default=False,
                        help='Small Batch SIMCLR then Momentum Contrast')

    parser.add_argument('--simclr_w', type=float, required=False,
                        help='simclr weight')

    parser.add_argument('--patchclr_w', type=float, required=False,
                        help='patchclr weight')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--deit', action="store_true", default=False,
                        help='DEIT')

    parser.add_argument('--deit_classifier', action="store_true", default=False,
                        help='DEIT Classifier')

    parser.add_argument('--no_autocast', action="store_true", default=False,
                        help='Avoid Autocast')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--backward_hook', action="store_true", default=False,
                        help='Backward Hook for gradients')

    parser.add_argument('--init_method', required=False, type=str, default="tcp",
                        help='init_method')

    parser.add_argument('--optimizer', required=False, type=str, default="rangerlars",
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

    parser.add_argument('--mode', required=True, type=str,
                        help='Run Model', choices=['clr', 'linear_probe', 'full_train', 'validation'])
    parser.add_argument('--dataset', required=False, type=str,
                        help='Dataset')

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)
    args.moco = args.moco if args.mode == 'clr' else False
    args.simclr_moco = args.simclr_moco if args.mode == 'clr' else False
    args.pct_simclr_simple = args.pct_simclr_simple if args.mode == 'clr' else 0

    args.moco = False if args.simclr_moco else args.moco
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 139
    args.seed = seed
    return vars(args)


def model_train_validation_switch(model, args, train=True):
    if train:
        model.train()
        if args["mode"] in ['clr', 'full_train']:
            model.train()
        elif args["mode"] == "linear_probe":
            model.backbone = model.backbone.eval()
        else:
            model = model.eval()
    else:
        model = model.eval()
    return model


class CLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, x1_transform, x2_transform, to_tensor, simclr_w, patchclr_w):
        self.dataset = dataset
        self.x1_transform = x1_transform
        self.x2_transform = x2_transform
        self.to_tensor = to_tensor
        self.patchclr_proba = 0.9 if simclr_w is None or simclr_w == 0 else 0.2
        self.patchclr_proba = 0.0 if patchclr_w is None or patchclr_w == 0 else self.patchclr_proba
        self.small_shape_transforms = transforms.Compose([
            transforms.RandomAffine(15, (0.05, 0.05), (0.9, 1.1), 5),
            transforms.RandomChoice([
                # transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.05),
                transforms.RandomResizedCrop(360, scale=(0.8, 1.0)),
            ])

        ])

    def __getitem__(self, item):

        x, label = self.dataset[item]
        # In 25% case X1 is not transformed so the patch_clr task becomes patch reconstruction by aggregation from neighbouring patches.
        if self.x1_transform and random.random() < 0.75:
            x1 = self.to_tensor(self.x1_transform(x.copy()))
        else:
            x1 = self.to_tensor(x)

        if random.random() < self.patchclr_proba:
            patch_clr_or_not = True
            x = self.small_shape_transforms(x) if random.random() < 0.5 else x
            if self.x2_transform:
                x2 = self.to_tensor(self.x2_transform(x.copy()))
            else:
                x2 = self.to_tensor(x)
        else:
            patch_clr_or_not = False
            x, label = self.dataset[item]
            if self.x2_transform:
                x2 = self.to_tensor(self.x2_transform(x.copy()))
            else:
                x2 = self.to_tensor(x)

        return dict(x1=x1, x2=x2, patch_clr_or_not=patch_clr_or_not)

    def __len__(self):
        return len(self.dataset)


class ImageNetValidation:
    def __init__(self, model, location, batch_size, device, num_workers=8):
        val_loader = build_dataloader(location, "validation", False, batch_size, 1, num_workers)
        train_loader = build_dataloader(location, "validation", False, batch_size, 1, num_workers, split="train")
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.device = device
        print("[Validation]: Time = %s, train length = %s, Val length = %s, batch size = %s" % (get_time_string(), len(train_loader), len(val_loader), batch_size))

    def __call__(self, ):
        model = self.model
        device = self.device
        train_loader = self.train_loader
        val_loader = self.val_loader
        predictions, labels = [], []
        for x, label in tqdm(val_loader, desc="Validation", miniters=10, mininterval=10.0):
            labels.extend(label.tolist())
            x = x.to(device)
            pred = model(x)["predictions"]
            predictions.extend(pred.cpu().tolist())
        val_acc = accuracy_score(labels, predictions)
        print("[Validation]: Time = %s, Val Acc = %.5f" % (get_time_string(), val_acc))

        predictions, labels = [], []
        for x, label in tqdm(train_loader, desc="Train", miniters=100, mininterval=100.0):
            labels.extend(label.tolist())
            x = x.to(device)
            pred = model(x)["predictions"]
            predictions.extend(pred.cpu().tolist())
        train_acc = accuracy_score(labels, predictions)

        print("[Validation]: Time = %s, Train Acc = %.5f, Val Acc = %.5f" % (get_time_string(), train_acc, val_acc))


def build_dataloader(location, mode, shuffle_dataset, batch_size, world_size=1, num_workers=1, split=None, simclr_w=None, patchclr_w=None):
    from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, MNIST, STL10, SVHN, Places365, ImageNet
    single_node = world_size == 1

    split = split if split is not None else ("val" if mode == "validation" else "train")
    image_transforms = get_image_augmetations(mode)

    if "imagenet" in location.lower():
        dataset = ImageNet(location, split, transform=image_transforms["shape_transforms"])
    elif "cifar10" in location.lower():
        dataset = CIFAR10(root=location, train=split != "val", download=True, transform=image_transforms["shape_transforms"])

    if mode == "clr":
        # print("[Train]: Time = %s, %s, %s, %s" % (get_time_string(), image_transforms["shape_transforms"], image_transforms["non_shape_transforms"], image_transforms["to_tensor"]))
        dataset = CLRDataset(dataset, image_transforms["non_shape_transforms"], image_transforms["non_shape_transforms"], image_transforms["to_tensor"], simclr_w=simclr_w, patchclr_w=patchclr_w)

    loader = DataLoader(dataset, sampler=None if single_node else DistributedSampler(dataset, shuffle=shuffle_dataset),
                        batch_size=batch_size, shuffle=shuffle_dataset and single_node,
                        prefetch_factor=8, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    return loader


def train(local_rank, args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # too many barriers / one node data parallel and multiple node DDP
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ["NCCL_DEBUG"] = "WARN"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # gpu_device = 0
    gpu_device = local_rank
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    print("[Train]: Time = %s, Prepare to init Dist Process for Rank = %s" % (get_time_string(), rank))
    if args["init_method"] == "tcp":
        if args["nr"] == 0:
            args["master_addr"] = "0.0.0.0"
        init_method="tcp://%s:%s" % (args["master_addr"], args["master_port"])
    elif args["init_method"] == "file":
        init_method = 'file://%s/%s' % (args["master_addr"], args["master_port"])
    else:
        raise ValueError

    dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
    print("[Train]: Time = %s, Initialized Dist Process for Rank = %s" % (get_time_string(), rank))
    barrier = get_barrier(True)
    rnd = torch.tensor(int(time.time())).to(device)
    dist.broadcast(rnd, 0)
    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(timezone('Asia/Kolkata')).strftime(format)
    ds_name = list(filter(lambda x: len(x.strip()) > 0, args["dataset"].split("/")))[-1].replace("train_fastformer_resampled_", "")
    group = "%s-%s-%sN-%s" % (ds_name, args["model_config"], args["nodes"], time_string)
    set_seeds(args["seed"])
    model_size = args["model_config"]
    batch_size = get_vision_batch_size(args["model_config"], not args["no_autocast"], args["mode"])
    config = vision_config_dict[model_size]

    optimizer_config.lr = args["lr"]
    optimizer_config.weight_decay = args["weight_decay"]
    optimizer_config.warmup_steps = args["warmup_steps"]
    optimizer_config.gradient_clipping = args["gradient_clipping"]

    if args["world_size"] != 128:
        optimizer_config.lr = optimizer_config.lr * (args["world_size"]/128)
    config.eps = 1e-4
    if args["no_autocast"]:
        optimizer_config.eps = 1e-5
        config.layer_norm_eps = 1e-5
        config.eps = 1e-5
        optimizer_config.gradient_clipping = 4 * optimizer_config.gradient_clipping

    fsdp_params = configure_fsdp(not args["no_autocast"], True if not args["no_autocast"] else False, True)
    reinit = not args["deit"] and not (args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]))
    print("[Train]: Time = %s, Reinit = %s" % (get_time_string(), reinit))
    fsdp_wrapper(wrap_type=0, init=True)
    if args["deit"]:
        batch_size = get_vision_batch_size("vision_md_config", not args["no_autocast"], args["mode"])
        backbone = get_pretrained_deit(not args["deit_classifier"])
    else:
        backbone = FastFormerVisionModel(config, reinit=True)

    batch_size = args["batch_size"] if "batch_size" in args and isinstance(args["batch_size"], int) else batch_size
    simclr_w = args["simclr_w"] if "simclr_w" in args else 2.0
    patchclr_w = args["patchclr_w"] if "patchclr_w" in args else 0.5

    if args["mode"] == "clr":
        if args["deit"]:
            model = PatchCLR(backbone, 768, config.layer_norm_eps, patchclr_w=patchclr_w, simclr_w=simclr_w, clustering_w=0.0, gap_bias_w=0.0,
                             reinit=reinit, patchclr_use_extra_negatives=False, simclr_use_extra_negatives=False, moco=args["moco"], heads=args["patchclr_heads"]).to(device)
        else:
            model = PatchCLR(backbone, config.block_channel_size[-1], config.layer_norm_eps, patchclr_w=patchclr_w, simclr_w=simclr_w, clustering_w=0.0, gap_bias_w=0.0,
                             reinit=reinit, patchclr_use_extra_negatives=True, simclr_use_extra_negatives=True, moco=args["moco"], heads=args["patchclr_heads"]).to(device)
    elif args["mode"] in ['linear_probe', 'full_train', 'validation']:
        model = ClassificationModel(backbone, args["num_classes"], 768 if args["deit"] else config.block_channel_size[1], train_backbone=True if "full_train" else False,
                                    reinit_backbone=not args["deit"] and not (args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]))).to(device)
        if args["deit"] and args["deit_classifier"]:
            model.head = nn.Identity().to(device)
        if args["mode"] == "linear_probe":
            model.backbone = model.backbone.eval()
            for v in model.backbone.parameters():
                v.requires_grad = False
    else:
        raise ValueError

    if local_rank == 0 and not args["moco"]:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(model) / 1_000_000))
        print(type(model))
        print(model)
        # if "pretrained_model" in args and args["pretrained_model"] is not None and args["mode"] == "clr":
        #     check_patch_clr_acc(model, args["mode"], device, args["pretrained_model"], config)

    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]):
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)

        try:
            model.load_state_dict(state_dict, strict=True)
            load_type = "strict"
        except:
            try:
                try:
                    model.load_state_dict(state_dict, strict=False)
                    load_type = "not_strict"
                except:
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ffn.")}
                    model.load_state_dict(state_dict, strict=False)
                    load_type = "not_strict_no_ffn"
            except:
                try:
                    try:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        model.load_state_dict(state_dict, strict=True)
                        load_type = "strict-from-ddp"
                    except:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ffn.")}
                        model.load_state_dict(state_dict, strict=True)
                        load_type = "strict-from-ddp-no-ffn"
                except:
                    try:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        model.load_state_dict(state_dict, strict=False)
                        load_type = "not_strict-from-ddp"
                    except:
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ffn.")}
                        model.load_state_dict(state_dict, strict=False)
                        load_type = "not_strict-from-ddp-no-ffn"

        print("[Train]: Time = %s, Loaded Pretrained model with Load type = %s, Torch Version = %s" % (get_time_string(), load_type, torch.__version__))
        del state_dict
    # if local_rank == 0 and not (args["moco"] or args["simclr_moco"]):
    #     check_patch_clr_acc(model, args["mode"], device, args["pretrained_model"], config)
    model = model_train_validation_switch(model, args, train=True)
    if args["mode"] == "validation" and local_rank == 0:
        assert args["world_size"] == 1
        assert isinstance(model, ClassificationModel)
        ImageNetValidation(model, args["dataset"], batch_size, device, args["num_workers"])()
        return

    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in model.named_parameters() if v.requires_grad}))
    if args["world_size"] > 1:
        # ddp_model = FSDP(model, **fsdp_params)  # find_unused_parameters=True
        ddp_model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False, bucket_cap_mb=10)  # find_unused_parameters=True

    else:
        ddp_model = model

    if (args["moco"] or args["simclr_moco"]) and args["mode"] == "clr":
        print("[Train]: Time = %s, Init MOCO backbone" % (get_time_string()))
        if isinstance(ddp_model, DDP):
            key_backbone = copy.deepcopy(ddp_model.module.backbone).to(device)
            key_backbone.load_state_dict(copy.deepcopy(ddp_model.module.backbone.state_dict()))
            key_ffn = copy.deepcopy(ddp_model.module.ffn).to(device)
            key_ffn.load_state_dict(copy.deepcopy(ddp_model.module.ffn.state_dict()))
        else:
            key_backbone = copy.deepcopy(backbone).to(device)
            key_backbone.load_state_dict(copy.deepcopy(backbone.state_dict()))
            key_ffn = copy.deepcopy(model.ffn).to(device)
            key_ffn.load_state_dict(copy.deepcopy(model.ffn.state_dict()))
    try:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
    except:
        print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))

    del backbone
    clean_memory()
    _ = model_train_validation_switch(ddp_model.module if hasattr(ddp_model, "module") else ddp_model, args, train=True)
    optc = optimizer_config.to_dict()
    trainable_params = list(filter(lambda p: p.requires_grad, ddp_model.parameters()))
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"],
                                      betas=(optc["beta_1"], optc["beta_2"]))
    elif args["optimizer"] == "novograd":
        optimizer = Novograd(ddp_model.parameters(), lr=optc["lr"], eps=optc["eps"], betas=(optc["beta_1"], optc["beta_2"]), weight_decay=optc["weight_decay"],)
    elif args["optimizer"] == "rangerlars":
        optimizer = RangerLars(ddp_model.parameters(), lr=optc["lr"], eps=optc["eps"], betas=(optc["beta_1"], optc["beta_2"]), weight_decay=optc["weight_decay"],)
    else:
        raise ValueError
    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in ddp_model.named_parameters() if v.requires_grad}))
    optimizer.zero_grad(set_to_none=True)
    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]

    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
    dataloader = build_dataloader(args["dataset"], args["mode"], args["shuffle_dataset"], batch_size,
                                  world_size=args["world_size"], num_workers=args["num_workers"],
                                  simclr_w=simclr_w, patchclr_w=patchclr_w)
    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    iter_size = max(args["accumulation_steps"], 1)
    no_sync = iter_size > 1
    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    # scheduler = optimization.get_linear_schedule_with_warmup(optimizer, optc["warmup_steps"], args["epochs"] * len(dataloader))
    steps_per_epoch = int(np.ceil(len(dataloader.sampler) / (batch_size * iter_size)) if dataloader.sampler is not None else len(dataloader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], epochs=args["epochs"], steps_per_epoch=steps_per_epoch,
                                                    div_factor=1e2, three_phase=False, pct_start=0.3, anneal_strategy="linear")
    gradient_clipping = optc["gradient_clipping"]
    print("[Train]: Time = %s, max lr = %.5f, epochs = %s, steps_per_epoch = %s, batch size = %s, dataloader length = %s, Sampler Present = %s, Sampler Length = %s" %
          (get_time_string(), optc["lr"], args["epochs"], steps_per_epoch, batch_size, len(dataloader), dataloader.sampler is not None, len(dataloader.sampler) if dataloader.sampler is not None else -1))

    if local_rank == 0:
        wandb_init_args = dict(project="patchclr", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                               config={"args":args, "config": config, "optimizer_config": optc},
                               settings=wandb.Settings(start_method="fork"))

        time.sleep(random.random() * 5)
        wandb.init(**wandb_init_args)
        print("[Train]: WandB-watch added over model for Rank = %s" % rank)

    full_times = []
    batch_times = []
    ddp_model.zero_grad(set_to_none=True)
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
        for name, param in ddp_model.named_parameters():
            param.register_hook(hook)

    extra_negative_repr_simclr = None
    extra_negative_repr_patchclr = None
    total_steps = args["epochs"] * len(dataloader)
    pct_simclr_simple = args["pct_simclr_simple"]
    assert pct_simclr_simple * 4 <= 50
    assert not args["moco"] or pct_simclr_simple == 0
    assert not args["simclr_moco"] or pct_simclr_simple > 0
    total_samples_simclr = args["total_samples_simclr"] if "total_samples_simclr" in args and args["total_samples_simclr"] is not None else (
                16 * batch_size * args["world_size"])  # args["world_size"] to max
    total_samples_simclr = 8 * (total_samples_simclr // 8)
    total_samples_patchclr = args["total_samples_patchclr"]
    for epoch in range(args["epochs"]):

        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
            print("Time = %s [custom_batching_fn]: Distributed Sampler Epoch = %s" % (get_time_string(), epoch))
        else:
            print("Time = %s [custom_batching_fn]: Unable to set Epoch = %s" % (get_time_string(), epoch))

        start_time = time.time()
        for step, batch in enumerate(dataloader):
            steps_done = epoch * len(dataloader) + step
            pct_done = (100 * steps_done / total_steps)
            gen_batch_time = time.time() - start_time
            batch_times.append(gen_batch_time)
            if isinstance(batch, dict):
                key = list(batch.keys())[0]
                bs_size = list(batch[key].size())
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
            else:
                batch[0] = batch[0].to(device, non_blocking=True)
                batch[1] = batch[1].to(device, non_blocking=True)
                bs_size = list(batch[0].size())
                key = 0

            samples_per_machine_simclr = total_samples_simclr // args["world_size"]
            model.simclr_use_extra_negatives = False
            model.patchclr_use_extra_negatives = False
            if hasattr(ddp_model, "module"):
                ddp_model.module.simclr_use_extra_negatives = False
                ddp_model.module.patchclr_use_extra_negatives = False

            if pct_done < pct_simclr_simple:
                samples_per_machine_simclr = 0
                model.simclr_use_extra_negatives = False
                model.patchclr_use_extra_negatives = False
                if hasattr(ddp_model, "module"):
                    ddp_model.module.simclr_use_extra_negatives = False
                    ddp_model.module.patchclr_use_extra_negatives = False
            else:
                if args["simclr_moco"]:
                    samples_per_machine_simclr = int(samples_per_machine_simclr * min(1, max(0, (pct_done - pct_simclr_simple) / ((pct_simclr_simple * 4) - pct_simclr_simple))))
                    model.moco = True
                    ddp_model.module.moco = True
                model.simclr_use_extra_negatives = True
                model.patchclr_use_extra_negatives = True
                if hasattr(ddp_model, "module"):
                    ddp_model.module.simclr_use_extra_negatives = True
                    ddp_model.module.patchclr_use_extra_negatives = True

            if (steps_done + 1) % save_every_steps == 0:
                state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
                barrier()
                if local_rank == 0:
                    torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
                del state_dict
                clean_memory()
                barrier()

            samples_processed += int(batch[key].size(0))
            samples_processed_this_log_iter += int(batch[key].size(0))
            inner_args = dict(no_autocast=args["no_autocast"], cpu=args["cpu"], mode=args["mode"])
            if args["moco"] or (args["simclr_moco"] and pct_done >= pct_simclr_simple):
                if step == 0 and "warm_restart_key_encoder" in args and args["warm_restart_key_encoder"] is not None and ((epoch + 1) % args["warm_restart_key_encoder"] == 0):
                    key_backbone.load_state_dict(ddp_model.module.backbone.state_dict() if hasattr(ddp_model, "module") else ddp_model.backbone.state_dict())
                    key_ffn.load_state_dict(ddp_model.module.ffn.state_dict() if hasattr(ddp_model, "module") else ddp_model.ffn.state_dict())
                    extra_negative_repr_patchclr = None
                    extra_negative_repr_simclr = None
                key_backbone = key_backbone.eval()
                key_ffn = key_ffn.eval()
                if hasattr(ddp_model, "module"):
                    ddp_model.module.key_backbone = key_backbone
                    ddp_model.module.key_ffn = key_ffn
                else:
                    ddp_model.key_backbone = key_backbone
                    ddp_model.key_ffn = key_ffn
                model.key_backbone = key_backbone
                model.key_ffn = key_ffn
            try:
                if no_sync and (step + 1) % iter_size != 0 and hasattr(ddp_model, "no_sync"):
                    with ddp_model.no_sync():
                        output = train_inner_loop(inner_args, ddp_model, batch, optimizer,
                                                  scheduler, gradient_clipping, iter_size=iter_size,
                                                  no_sync=True, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0 and not args["no_autocast"], extra_negative_repr_simclr=extra_negative_repr_simclr, extra_negative_repr_patchclr=extra_negative_repr_patchclr)
                else:
                    output = train_inner_loop(inner_args, ddp_model, batch, optimizer,
                                              scheduler, gradient_clipping, iter_size=iter_size,
                                              no_sync=False, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0 and not args["no_autocast"], extra_negative_repr_simclr=extra_negative_repr_simclr, extra_negative_repr_patchclr=extra_negative_repr_patchclr)
                    optimizer.zero_grad(set_to_none=True)
                    if args["mode"] == "clr" and (args["moco"] or args["simclr_moco"]):
                        if pct_done < pct_simclr_simple:
                            key_backbone.load_state_dict(ddp_model.module.backbone.state_dict() if hasattr(ddp_model, "module") else ddp_model.backbone.state_dict())
                            key_ffn.load_state_dict(ddp_model.module.ffn.state_dict() if hasattr(ddp_model, "module") else ddp_model.ffn.state_dict())
                        else:
                            key_backbone.load_state_dict({k_key: 0.99 * v_key + 0.01 * v_query for (k_key, v_key), (k_query, v_query) in zip(key_backbone.state_dict().items(), ddp_model.module.backbone.state_dict().items() if hasattr(ddp_model, "module") else ddp_model.backbone.state_dict().items())})
                            key_ffn.load_state_dict({k_key: 0.99 * v_key + 0.01 * v_query for (k_key, v_key), (k_query, v_query) in
                                                          zip(key_ffn.state_dict().items(), ddp_model.module.ffn.state_dict().items() if hasattr(ddp_model, "module") else ddp_model.ffn.state_dict().items())})

                if args["mode"] == "clr" and ((step + 1) % iter_size != 0 or iter_size == 1) and (hasattr(ddp_model, "module") and ddp_model.module.simclr_use_extra_negatives) and samples_per_machine_simclr >= 1 and simclr_w is not None and simclr_w > 0:
                    extra_negative_repr_simclr = output.pop("extra_negative_repr_simclr", None)

                    if extra_negative_repr_simclr is not None:
                        if epoch == 0 and step <= iter_size:
                            start = max(extra_negative_repr_simclr.size(0) - samples_per_machine_simclr, 0)
                            most_recent_simclr = extra_negative_repr_simclr[start:]
                        else:
                            start = max(extra_negative_repr_simclr.size(0) - samples_per_machine_simclr, 0)
                            most_recent_simclr = extra_negative_repr_simclr[start:]
                        empty_size = most_recent_simclr.size()
                        tensor_list = [most_recent_simclr.new_empty(empty_size) for _ in range(args["world_size"])]
                        torch.distributed.all_gather(tensor_list, most_recent_simclr.contiguous())

                        extra_negative_repr_simclr = torch.stack(tensor_list, 1).view(most_recent_simclr.size(0) * args["world_size"], extra_negative_repr_simclr.size(-1))
                        output["extra_negative_repr_simclr"] = extra_negative_repr_simclr


                extra_negative_repr_simclr = output.pop("extra_negative_repr_simclr", None)
                extra_negative_repr_patchclr = output.pop("extra_negative_repr_patchclr", None)
                if extra_negative_repr_patchclr is not None and extra_negative_repr_patchclr.size(0) > total_samples_patchclr:
                    extra_negative_repr_patchclr = extra_negative_repr_patchclr[extra_negative_repr_patchclr.size(0) - total_samples_patchclr:]
            except Exception as e:
                es = "[Train-Exception]: Time = %s, Step = %s for Rank = %s, Scale = %s, input_size = %s, lr = %s" % (
                    get_time_string(), step, rank, None, bs_size, optimizer.param_groups[0]['lr'])
                print(es)
                torch.save(batch, os.path.join(os.getcwd(), "error-input.pth"))

                reraise(e, es)
            full_time = time.time() - start_time
            full_times.append(full_time)
            if step == 0 and epoch == 0:
                print("[Train]: Time = %s, First Batch Training for Rank = %s" % (get_time_string(), rank))
            if (steps_done + 1) % log_every_steps == 0:
                if local_rank == 0:
                    simclr_negative = extra_negative_repr_simclr.size(0) if extra_negative_repr_simclr is not None else 0
                    patchclr_negative = extra_negative_repr_patchclr.size(0) if extra_negative_repr_patchclr is not None else 0
                    steps_remaining = total_steps - steps_done
                    output = {k: float(v) if v else v for k, v in output.items()}
                    samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
                    time.sleep(random.random() + 0.1)
                    wandb_log = dict(lr=optimizer.param_groups[0]['lr'], epoch=epoch+1, step=step, samples_processed=samples_processed, samples_per_second=samples_per_second,
                                   batch_times=np.mean(batch_times), full_times=np.mean(full_times), steps_remaining=steps_remaining, pct_complete=(100 * steps_done / total_steps),
                                     simclr_negative=simclr_negative, patchclr_negative=patchclr_negative,
                                   **{k: v for k, v in output.items() if v is not None})

                    wandb.log(wandb_log)
                    print("[Train]: Time = %s, Epoch = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Details = %s, LR = %s" %
                          (get_time_string(), epoch+1, rank, step, samples_processed, bs_size, output, optimizer.param_groups[0]['lr']))
                    print("[Train-Timings]: Time = %s, Batch time = %.4f, Full Time = %.4f, samples_per_second = %s, steps_remaining = %s, pct_complete = %.4f, simclr_use_extra_negatives = %s,\nsamples_per_machine_simclr = %s, total_samples_simclr = %s, simclr_negative = %s, patchclr_negative = %s" % (
                    get_time_string(), np.mean(batch_times), np.mean(full_times), samples_per_second, steps_remaining, (100 * steps_done / total_steps), ddp_model.module.simclr_use_extra_negatives if hasattr(ddp_model, "module") else ddp_model.simclr_use_extra_negatives,
                    samples_per_machine_simclr, total_samples_simclr, simclr_negative, patchclr_negative))
                batch_times = []
                full_times = []
                samples_processed_this_log_iter = 0

                clean_memory()
                barrier()
            del batch
            del output
            del bs_size
            start_time = time.time()
    print("Time = %s, Finished Training for Rank = %s" % (get_time_string(), rank))
    state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
    del ddp_model
    if rank == 0 and args["mode"] in ['linear_probe', 'full_train']:
        model = ClassificationModel(FastFormerVisionModel(config), args["num_classes"], 768 if args["deit"] else config.block_channel_size[1],
                                    train_backbone=True if "full_train" else False).to(device)
        model.load_state_dict(state_dict, strict=True)
        assert isinstance(model, ClassificationModel)
        ImageNetValidation(model, args["dataset"], batch_size, device, args["num_workers"])()


def train_inner_loop(args, ddp_model, batch, optimizer, scheduler, gradient_clipping, iter_size=1,
                     no_sync=False, zero_grad_check=False, extra_negative_repr_patchclr=None, extra_negative_repr_simclr=None,
                     scaler=None):
    is_fp16 = isinstance(ddp_model, DDP) and scaler is not None
    with autocast(is_fp16):
        if args["mode"] == "clr":
            x1 = batch["x1"]
            x2 = batch["x2"]
            patch_clr_or_not = batch["patch_clr_or_not"]
            output = ddp_model(x1, x2, patch_clr_or_not, extra_negative_repr_patchclr=extra_negative_repr_patchclr, extra_negative_repr_simclr=extra_negative_repr_simclr)
        elif args["mode"] == "linear_probe" or args["mode"] == "full_train":
            x = batch[0]
            labels = batch[1]
            output = ddp_model(x, labels)
        else:
            raise ValueError

    loss = output.pop("loss") / iter_size
    if is_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    _ = output.pop("predictions", None)
    zgradders = []
    inf_gradders = []
    if zero_grad_check:
        zgradders = [name for name, params in ddp_model.named_parameters() if torch.all(params.grad == 0).item()]
        if len(zgradders):
            print("[Train]: Time = %s, Zero Grads: " % get_time_string(), zgradders)
        inf_gradders = [name for name, params in ddp_model.named_parameters() if torch.any(torch.logical_not(torch.isfinite(params.grad))).item()]
        if len(inf_gradders):
            print("[Train]: Time = %s, INF/NAN Grads: " % get_time_string(), inf_gradders)
    # print([name for name, params in ddp_model.named_parameters() if params.grad is None])
    if not no_sync:
        if is_fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            ddp_model.clip_grad_norm_(gradient_clipping) if isinstance(ddp_model, FSDP) else torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step()

    if np.isnan(loss.detach().cpu().item()):
        es = "[Train-Exception]: Time = %s, NAN Loss, Scale = %s, loss_dict = %s, lr = %s" % (
            get_time_string(), None, loss, optimizer.param_groups[0]['lr'])
        raise ValueError(es)
    _ = output.pop("logits", None)
    _ = output.pop("predictions", None)
    return dict(loss=loss, **output, zero_grad=len(zgradders), inf_grad=len(inf_gradders))


def train_catch_exception(local_rank, args):
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    nr = args["nr"]
    try:
        train(local_rank, args)
    except Exception as e:
        print("[Exception-in-train]: Node Rank = %s, Local Rank = %s, Rank = %s, Exception = %s, \n Trace = %s" % (nr, local_rank, rank, e, traceback.format_exc()))
        # traceback.print_tb(e.__traceback__)
        # traceback.print_exception(*sys.exc_info())
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)



