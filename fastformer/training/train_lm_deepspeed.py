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

# import pickle4reducer
# import multiprocessing as mp
# ctx = mp.get_context()
# ctx.reducer = pickle4reducer.Pickle4Reducer()
import shutil
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
import deepspeed


def training_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--resume', required=False, type=str,
                        help='Resume From')
    parser.add_argument('--checkpoint', required=False, type=str,
                        help='Checkpoint Location')

    parser.add_argument('--model_save_dir', required=True, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=True, type=str,
                        help='Save Name')

    parser.add_argument('--validate_on_start', action="store_true", default=False,
                        help='Validate before training')

    parser.add_argument('--skip_steps', action="store_true", default=False,
                        help='Skip already trained steps while continuing training')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--validate_only', action="store_true", default=False,
                        help='Validate Only')

    parser.add_argument('--test_only', action="store_true", default=False,
                        help='Test Only')

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

    parser.add_argument('--train_dataset', required=False, type=str,
                        help='Train Dataset')

    parser.add_argument('--validation_dataset', required=False, type=str,
                        help='Validation Dataset')

    parser.add_argument('--test_dataset', required=False, type=str,
                        help='Test Dataset')

    parser.add_argument('--test_fastformer', required=False, type=str,
                        help='Test Dataset')

    parser.add_argument('--init_method', required=False, type=str, default="tcp",
                        help='init_method')

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
    parser.add_argument('--validate_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 7919
    args.seed = seed
    assert hasattr(args, "test_dataset") or not args["test_only"]
    assert hasattr(args, "validation_dataset") or not args["validate_only"]
    return vars(args)


def cleanup():

    dist.destroy_process_group()


def build_dataloader(location, shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, size_dicts, continuous_iter=True, world_size=1, num_workers=1):
    assert max(size_dicts.values()) % min(size_dicts.values()) == 0
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    min_size = gcd_array(size_dicts.values())
    prefetch_factor = 2 * (max(size_dicts.values()) // min_size)
    try:
        train_dataset = Dataset.load_from_disk(location)
        train_dataset = TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), train_dataset)
        if num_workers > 0:
            train_loader = DataLoader(train_dataset, sampler=None if single_node else DistributedSampler(train_dataset, shuffle=shuffle_dataset),
                                      batch_size=min_size, collate_fn=collate_fn, shuffle=shuffle_dataset and single_node,
                                      prefetch_factor=prefetch_factor,
                                      num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, sampler=None if single_node else DistributedSampler(train_dataset, shuffle=shuffle_dataset), batch_size=min_size,
                                      collate_fn=collate_fn, shuffle=shuffle_dataset and single_node,
                                      num_workers=0, pin_memory=True)
        train_loader = custom_batching_fn(train_loader, size_dicts, continuous_iter)
    except:
        train_dataset = DatasetDict.load_from_disk(location)
        train_dataset = {k: v for k, v in train_dataset.items() if len(v) >= world_size}
        train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_dataset.items()}
        lsum = sum(train_dataset_sampling_proba.values())
        train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
        train_dataset = {k: TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), v) for k, v in train_dataset.items()}
        # for v in train_dataset.values():
        #     v.training = False
        if num_workers > 0:
            train_loader = {k: DataLoader(v, sampler=None if single_node else DistributedSampler(v, shuffle=shuffle_dataset, ), shuffle=shuffle_dataset and single_node,
                                          batch_size=min_size, collate_fn=collate_fn, prefetch_factor=prefetch_factor, num_workers=num_workers) for k, v in train_dataset.items()}
        else:
            train_loader = {
                k: DataLoader(v, sampler=None if single_node else DistributedSampler(v, shuffle=shuffle_dataset, ), shuffle=shuffle_dataset and single_node,
                              batch_size=min_size, collate_fn=collate_fn,
                              num_workers=0) for k, v in train_dataset.items()}
        train_loader = {k: custom_batching_fn(dataloader, size_dicts, continuous_iter) for k, dataloader in train_loader.items()}
        train_loader = datadict_iterator(train_loader, train_dataset_sampling_proba)
    return train_loader


def train_catch_exception(local_rank, args):
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    nr = args["nr"]
    try:
        train(local_rank, args)
    except Exception as e:
        print("[Exception-in-train]: Node Rank = %s, Local Rank = %s, Rank = %s, Exception = %s, Trace = %s" % (nr, local_rank, rank, e, traceback.format_exc()))
        # traceback.print_tb(e.__traceback__)
        # traceback.print_exception(*sys.exc_info())
        traceback.print_exc()
        raise e


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
    device = torch.device(f'cuda:{gpu_device}')  # Unique only on individual node.
    torch.cuda.set_device(device)
    print("[Train]: Time = %s, Prepare to init Dist Process for Rank = %s" % (get_time_string(), rank))

    if args["nr"] == 0:
        args["master_addr"] = "0.0.0.0"
    init_method="tcp://%s:%s" % (args["master_addr"], args["master_port"])

    deepspeed.init_distributed(distributed_port=int(args["master_port"]), init_method=init_method)


    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(timezone('Asia/Kolkata')).strftime(format)
    ds_name = list(filter(lambda x: len(x.strip()) > 0, args["train_dataset"].split("/")))[-1].replace("train_fastformer_resampled_", "")
    group = "%s-%s-nodes-%s" % (ds_name, args["nodes"], time_string)
    set_seeds(args["seed"])
    mconf = model_config.to_dict()
    config = dict(md_config=md_config, sm_config=sm_config, lg_config=lg_config)[mconf.pop("model_size")]
    tokenizer = get_tokenizer(mconf.pop("tokenizer_name"))
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = 1024
    config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens

    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)

    model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
    print("[Train]: Trainable Params = %s" % (numel(model) / 1_000_000))
    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]) and rank == 0:
        model.load_state_dict(torch.load(args["pretrained_model"], map_location='cuda:%d' % gpu_device))

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters())


    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
    shuffle_dataset = args["shuffle_dataset"]

    train_loader = build_dataloader(args["train_dataset"], shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, world_size=args["world_size"], num_workers=args["num_workers"], no_autocast=args["no_autocast"])

    print("[Train]: Data Loaded for Rank = %s" % rank)

    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    print("[Train]: Scheduler Created for Rank = %s" % rank)
    other_load_details = None
    if "resume" in args and isinstance(args["resume"], str) and len(args["resume"].strip()) > 0:
        _, other_load_details = model_engine.load_checkpoint(model_save_dir, args["resume"])
        step = other_load_details['step']

    else:
        print("[Train]: No Resume for Rank = %s" % rank)
    _ = model.train()

    # print("[Train]: Init Wandb-watch added over model for Rank = %s" % rank)
    # wandb.watch(model, log="all", log_freq=log_every_steps)
    print("[Train]: WandB-watch added over model for Rank = %s" % rank)
    batch_times = []
    model_times = []
    full_times = []
    samples_processed = 0
    samples_processed_this_log_iter = 0
    print("[Train]: Time = %s, Start Training for Rank = %s" % (get_time_string(), rank))
    if local_rank == 0:
        wandb_init_args = dict(project="fastformer", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                               config={"args":args, "model_config": mconf, "config": config, "optimizer_config": optc},
                               settings=wandb.Settings(start_method="fork"))

        time.sleep(random.random() * 5)
        wandb.init(**wandb_init_args)

    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)

    def get_hook(name_of_param=None):
        if name_of_param is None:
            def hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-3), grad)
                    # grad = torch.clamp_(grad, -1e1, 1e1)
                    return grad
                else:
                    return None
            return hook
        else:
            def named_hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
                    grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-3), grad)
                    # grad = torch.clamp_(grad, -1e1, 1e1)
                    return grad
                else:
                    return None
            return named_hook
    if args["backward_hook"]:
        for name, param in model.named_parameters():
            if "embeddings" in name or "sent_predict_fc" in name or "embed_proj_transpose" in name or "embed_proj" in name or "lm_head" in name or "contrastive_ffn" in name or "encoder.blocks.0" in name: #
                param.register_hook(get_hook())
            else:
                param.register_hook(get_hook())

    start_time = time.time()
    for step, batch in enumerate(train_loader):
        gen_batch_time = time.time() - start_time
        batch_times.append(gen_batch_time)
        bs_size = list(batch["input_ids"].size())
        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
        electra_loss_w = float(((step + 1) / (2 * 10000)) * mconf["electra_loss_w"])
        model_engine.model.electra_loss_w = electra_loss_w

        if (step + 1) % save_every_steps == 0:
            client_sd = dict()
            client_sd['step'] = step
            ckpt_id = step
            model_engine.save_checkpoint(model_save_dir, ckpt_id, client_sd=client_sd)

        record_accuracy = False
        if (step + 1) % log_every_steps == 0:
            if local_rank == 0:
                record_accuracy = True

        batch["record_accuracy"] = record_accuracy
        labels = batch["label_mlm_input_ids"] if "label_mlm_input_ids" in batch else batch["input_ids"]
        labels = labels.to(device, non_blocking=True)
        model_start_time = time.time()
        samples_processed += int(batch["input_ids"].size(0))
        samples_processed_this_log_iter += int(batch["input_ids"].size(0))
        # clean_memory()
        # print("Step = %s, Before:, for Rank = %s, input_size = %s, Allocated = %.3f, Max Allocated = %.3f, Percent = %s" %
        #       (step, rank, batch["input_ids"].size(), torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() /1e6, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()))  # torch.cuda.memory_summary()

        # forward() method
        output = model_engine(**batch, labels=labels)
        loss = output["loss"]
        # runs backpropagation
        model_engine.backward(loss)

        # weight update
        model_engine.step()


        # clean_memory()
        # print("Step = %s, After: , for Rank = %s, input_size = %s, Allocated = %.3f, Max Allocated = %.3f, Percent = %s" %
        #       (step, rank, batch["input_ids"].size(), torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() / 1e6,
        #        torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()))  # torch.cuda.memory_summary()

        model_end_time = time.time() - model_start_time
        model_times.append(model_end_time)
        full_time = time.time() - start_time
        full_times.append(full_time)
        if step == 0:
            print("[Train]: Time = %s, First Batch Training for Rank = %s" % (get_time_string(), rank))
        if (step + 1) % log_every_steps == 0:
            if local_rank == 0:
                samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
                acc_dict = output["accuracy_hist"]
                loss_dict = output["loss_dict"]
                time.sleep(random.random() + 0.1)
                wandb.log(dict(lr=optimizer.param_groups[0]['lr'], step=step, samples_processed=samples_processed, samples_per_second=samples_per_second, batch_x_sequence=np.prod(bs_size[:2]),
                               batch_times=np.mean(batch_times), model_times=np.mean(model_times), full_times=np.mean(full_times), scale=scaler.get_scale(),
                               **loss_dict, **acc_dict))
                print("[Train]: Time = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Loss = %s, Accuracy = %s, LR = %s" %
                      (get_time_string(), rank, step, samples_processed,
                       bs_size, loss_dict, output["accuracy_hist"], optimizer.param_groups[0]['lr']))
                print("[Train-Timings]: Time = %s, Batch time = %.4f, Model Time = %.4f, Full time = %.4f, samples_per_second = %s" % (get_time_string(), np.mean(batch_times), np.mean(model_times), np.mean(full_times), samples_per_second))
                del acc_dict
                del loss_dict

            batch_times = []
            model_times = []
            full_times = []
            samples_processed_this_log_iter = 0


        del batch
        del labels
        del output
        del bs_size
        start_time = time.time()


# I've been tracking an ema of sample training loss during training and using that to guide weighted data sampling (rather than the typical uniform sampling). Seems to help with a variety of real world datasets where the bulk of the data is often very similar and easy to learn but certain subpopulations are much more challenging.


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        try:
            mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)
        finally:
            cleanup()

