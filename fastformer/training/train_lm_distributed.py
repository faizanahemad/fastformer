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

    parser.add_argument('--shuffle_dataset', required=False, type=str2bool, default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', required=False, type=str2bool, default=False,
                        help='Shuffle Train')

    parser.add_argument('--train_dataset', required=False, type=str,
                        help='Train Dataset')

    parser.add_argument('--validation_dataset', required=False, type=str,
                        help='Validation Dataset')

    parser.add_argument('--test_dataset', required=False, type=str,
                        help='Test Dataset')

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
    parser.add_argument('--log_every_steps', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validate_every_steps', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_every_steps', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.world_size = args.gpus_per_node * args.nodes
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 0
    args.seed = seed
    assert hasattr(args, "test_dataset") or not args["test_only"]
    assert hasattr(args, "validation_dataset") or not args["validate_only"]
    return vars(args)


class SuperGLUEValidator:
    def __init__(self, location, model, config):
        self.location = location
        self.model = model
        self.config = config
        self.superglue_validation_set = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc_fixed']

    def read_data(self):
        import glob
        datasets = glob.glob(os.path.join(self.location, "superglue_*"))
        datadict = dict()
        for d in datasets:
            load_point = os.path.join(self.location, d)
            try:
                ds = Dataset.load_from_disk(load_point)
            except:
                ds = DatasetDict.load_from_disk(load_point)
            datadict[d.replace("superglue_", '')] = ds
        return datadict

    def __call__(self, generate_test_predictions=True):
        datadict = self.read_data()
        tokenizer = self.model.tokenizer
        collate_fn = get_collate_fn(self.config.num_highway_cls_tokens, tokenizer.pad_token_id)
        for d in self.superglue_validation_set:
            # record answers
            # Start from Dataset of superglue from huggingface datasets and build labels
            # For prediction use majority voting from superglue datasets we made.
            # Validation fastformer dataset can be used?
            dataset = datadict[d]["validation"]
            labels = [dataset[i] for i in range(len(dataset))]

            dataset = TokenizerDataset(self.config, tokenizer, char_to_id,
                                       dict(padding="max_length", truncation=True, return_tensors="pt", max_length=self.config.tokenizer_length),
                                       dataset)
            dataset.training = False
            data_loader = DataLoader(dataset, sampler=None, batch_size=1, collate_fn=None,
                                     prefetch_factor=8, num_workers=4)
            data_loader = custom_batching_fn(data_loader, size_dicts, collate_fn, False)


class LargeValidator:
    def __init__(self, location, model, config, device, tokenizer):
        self.location = location
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

    def __call__(self):
        datadict = DatasetDict.load_from_disk(self.location)
        tokenizer = self.tokenizer
        model = self.model.to(self.device)
        model = model.eval()
        collate_fn = get_collate_fn(self.config.num_highway_cls_tokens, tokenizer.pad_token_id)
        results = dict()
        for k, dataset in datadict.items():
            cns = dataset.column_names
            predictions = []
            if 'answer' in cns:
                labels = [dataset[i] for i in range(len(dataset))]
            dataset = TokenizerDataset(self.config, tokenizer, char_to_id,
                                       dict(padding="max_length", truncation=True, return_tensors="pt", max_length=self.config.tokenizer_length),
                                       dataset)
            dataset.training = False
            record_accuracy = False
            if 'answer' not in cns:
                dataset.training = True
                record_accuracy = True
            loader = DataLoader(dataset, sampler=None, batch_size=1, collate_fn=None, prefetch_factor=8, num_workers=4)
            loader = custom_batching_fn(loader, size_dicts, collate_fn, False)
            for pt_batch in loader:
                pt_batch["record_accuracy"] = record_accuracy
                pt_batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in pt_batch.items()}
                if 'answer' in cns:
                    with autocast():
                        with torch.no_grad():
                            funnel_inputs = dict(input_ids=pt_batch["input_ids"],
                                                 attention_mask=pt_batch["attention_mask"],
                                                 token_type_ids=pt_batch["token_type_ids"],
                                                 inputs_embeds=None,
                                                 char_ids=pt_batch["char_ids"], char_offsets=pt_batch["char_offsets"],
                                                 run_decoder=False,
                                                 run_answering=True)
                            output = model.module.funnel(**funnel_inputs)
                            answering_predictions = output["answering_logits"].argmax(dim=-1)
                            answering_predictions = answer_decoder(answering_predictions, tokenizer)
                            predictions.extend(answering_predictions)

                else:
                    labels = pt_batch["label_mlm_input_ids"] if "label_mlm_input_ids" in pt_batch else pt_batch["input_ids"]
                    labels = labels.to(self.device)
                    with autocast():
                        with torch.no_grad():
                            output = model(**pt_batch, labels=labels)["accuracy_hist"]
                            predictions.append(output)
            if 'answer' in cns:
                final_labels, final_predictions = [], []
                for lbl, prd in zip(labels, predictions):
                    if len(prd) > len(lbl):
                        prd = prd[:len(lbl)]
                    if len(prd) < len(lbl):
                        prd = prd + ([''] * (len(lbl) - len(prd)))
                    final_labels.extend(lbl)
                    final_predictions.extend(prd)
                score = accuracy_score(final_labels, final_predictions)
                results[k] = dict(accuracy=score)
            else:
                results[k] = pd.DataFrame.from_records(predictions).mean().to_dict()
        model = model.train()
        return results


def cleanup():

    dist.destroy_process_group()


def build_dataloader(location, shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, single_node=False, continuous_iter=True):
    try:
        train_dataset = Dataset.load_from_disk(location)
        train_dataset = TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), train_dataset)
        train_loader = DataLoader(train_dataset, sampler=None if single_node else DistributedSampler(train_dataset, shuffle=shuffle_dataset), batch_size=1, collate_fn=None, prefetch_factor=8, num_workers=4)
        train_loader = custom_batching_fn(train_loader, size_dicts, collate_fn, continuous_iter)
    except:
        train_dataset = DatasetDict.load_from_disk(location)
        train_dataset = {k: v for k, v in train_dataset.items() if len(v) > 0}
        train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_dataset.items()}
        lsum = sum(train_dataset_sampling_proba.values())
        train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
        train_dataset = {k: TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), v) for k, v in train_dataset.items()}
        # for v in train_dataset.values():
        #     v.training = False
        train_loader = {k: DataLoader(v, sampler=None if single_node else DistributedSampler(v, shuffle=shuffle_dataset, ), batch_size=12, collate_fn=collate_fn, prefetch_factor=4, num_workers=2) for k, v in train_dataset.items()}
        # train_loader = {k: custom_batching_fn(dataloader, size_dicts, collate_fn, continuous_iter) for k, dataloader in train_loader.items()}
        train_loader = datadict_iterator(train_loader, train_dataset_sampling_proba)
    return train_loader


def get_barrier(activate):
    def barrier():
        if activate:
            torch.distributed.barrier()
    return barrier


def train(local_rank, args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    torch.backends.cudnn.benchmark = True
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    if args["cpu"]:
        assert args["world_size"] == 1
        device = torch.device("cpu")
        barrier = get_barrier(False)
    else:
        dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"])
        device = torch.device(f'cuda:{local_rank}')  # Unique only on individual node.
        torch.cuda.set_device(device)
        barrier = get_barrier(True)

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
    print("Trainable Params = %s" % (numel(model) / 1_000_000))
    if args["pretrained_model"] is not None:
        model.load_state_dict(torch.load(args["pretrained_model"], map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank}))
    # Take model to local rank
    if args["cpu"]:
        ddp_model = model
    else:
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    all_params = list(filter(lambda p: p.requires_grad, ddp_model.parameters()))
    optc = optimizer_config.to_dict()
    optimizer = AdamW(all_params, lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"]))
    optimizer.zero_grad()
    scaler = GradScaler()

    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]
    if rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
    assert os.path.exists(model_save_dir)
    barrier()
    shuffle_dataset = args["shuffle_dataset"]
    sampling_fraction = optc["sampling_fraction"]
    if not args["validate_only"] and not args["test_only"]:
        train_loader = build_dataloader(args["train_dataset"], shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, args["world_size"] == 1)

    validate_every_steps = args["validate_every_steps"]
    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    gradient_clipping = optc["gradient_clipping"]
    _ = model.train()
    barrier()

    start_time = time.time()
    batch_times = []
    model_times = []
    full_times = []
    print("Start Training for Rank = %s" % rank)
    for step, batch in enumerate(train_loader):
        # if step <= 39:
        #     continue
        gen_batch_time = time.time() - start_time
        batch_times.append(gen_batch_time)
        if (step + 1) % save_every_steps == 0:
            if rank == 0:
                torch.save(ddp_model.state_dict(), os.path.join(model_save_dir, model_save_name))
            barrier()
        if (step + 1) % validate_every_steps == 0:
            if rank == 0:
                val_results = LargeValidator(args["validation_dataset"], ddp_model, config, device, tokenizer)()
                print("Rank = %s, steps = %s, Val = %s" % (rank, step, val_results))
            barrier()
        record_accuracy = False
        if (step + 1) % log_every_steps == 0:
            record_accuracy = True

        batch["record_accuracy"] = record_accuracy
        labels = batch["label_mlm_input_ids"] if "label_mlm_input_ids" in batch else batch["input_ids"]
        labels = labels.to(device)
        model_start_time = time.time()
        if args["cpu"]:
            output = ddp_model(**batch, labels=labels)
            loss = output["loss"]
            loss_dict = output["loss_dict"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, gradient_clipping)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            with autocast():

                output = ddp_model(**batch, labels=labels)
                loss = output["loss"]
                loss_dict = output["loss_dict"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, gradient_clipping)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        model_end_time = time.time() - model_start_time
        model_times.append(model_end_time)

        if (step + 1) % log_every_steps == 0:
            print("Rank = %s, steps = %s, batch_size = %s, Loss = %s, Accuracy = %s" % (rank, step, batch["input_ids"].size(), loss_dict, output["accuracy_hist"]))
            print("Batch time = %s, Model Time = %s, Full time = %s" % (np.mean(batch_times), np.mean(model_times), np.mean(full_times)))
            batch_times = []
            model_times = []
            full_times = []
            clean_memory()
            barrier()
        full_time = time.time() - start_time
        full_times.append(full_time)
        start_time = time.time()


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
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1:
        train(0, args)
    else:
        try:
            mp.spawn(train, nprocs=args["gpus_per_node"], args=(args,), join=True)
        finally:
            cleanup()

