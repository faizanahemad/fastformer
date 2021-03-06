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

# from fastformer.training.pickle4reducer import *
# import pickle4reducer
# import multiprocessing as mp
# ctx = mp.get_context("spawn")
# ctx.reducer = Pickle4Reducer()

import multiprocessing
import traceback
import dill
import signal
import copy
import warnings
warnings.simplefilter("ignore")
import shutil
import sys
import traceback
import jsonlines
import jsonlines as jsonlines
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate
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
from torch.multiprocessing import Process, ProcessContext
import torch.multiprocessing as mp
import torch.optim as optim
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from torch.cuda.amp import GradScaler, autocast
from fastformer.data import *
from fastformer.config import *
from fastformer.data.dataset import datadict_iterator, superglue_test, MTTDataset
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
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import logging

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


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
    parser.add_argument('--lr', default=optimizer_config.lr, type=float,
                        help='lr')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Epochs')
    parser.add_argument('--weight_decay', default=0.1, type=float,
                        help='weight_decay')

    parser.add_argument('--hpo', required=False, type=str,
                        help='hpo dict with lr, epochs, warmup steps')

    parser.add_argument('--dataset_key', required=False, type=str,
                        help='dataset_key')
    parser.add_argument('--cls_tokens', default=1, type=int,
                        help='cls_tokens')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')
    parser.add_argument('--batch_size', required=False, type=int,
                        help='Batch Size')
    parser.add_argument('--seed', required=False, type=int, default=3123,
                        help='seed')
    parser.add_argument('--enable_layer_normalizers', action="store_true", default=False,
                        help='enable_layer_normalizers')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--resume', required=False, type=str,
                        help='Resume From')
    parser.add_argument('--checkpoint', required=False, type=str,
                        help='Checkpoint Location')

    parser.add_argument('--model_save_dir', required=False, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=False, type=str,
                        help='Save Name')

    parser.add_argument('--validate_on_start', action="store_true", default=False,
                        help='Validate before training')

    parser.add_argument('--skip_steps', action="store_true", default=False,
                        help='Skip already trained steps while continuing training')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--validate_only', action="store_true", default=False,
                        help='Validate Only')

    parser.add_argument('--test_only', action="store_true", default=True,
                        help='Test Only')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--finetune', action="store_true", default=False,
                        help='finetune')

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

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    assert hasattr(args, "test_dataset") or not args["test_only"]
    assert hasattr(args, "validation_dataset") or not args["validate_only"]
    return vars(args)


class rproc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, x):
        tokenizer = self.tokenizer
        answers = x["answers"][0]
        entities = x["entities"][0]
        idx = x["idx"][0]["query"]
        passage = x["passage"][0]
        query = x["query"][0]

        text = passage + f" {tokenizer.sep_token} " + query
        rd = defaultdict(list)
        for i, e in enumerate(entities):
            rd['idx'].append(idx)
            rd['text'].append(text + f" {tokenizer.sep_token} " + e)
            rd['choice'].append(i)
            rd['label'].append(e in answers)
            rd['entities'].append(entities)
        return rd


class wsc_proc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, x):
        tokenizer = self.tokenizer
        words = x["text"].split()
        words = words[:x["span1_index"]] + ["[%s]" % (words[x["span1_index"]])] + words[(x["span1_index"] + 1):x["span2_index"]] + [
            "[%s]" % (words[x["span2_index"]])] + words[x["span2_index"]:]
        modified_text = " ".join(words)
        text = x["text"] + f" {tokenizer.sep_token} " + modified_text + f" {tokenizer.sep_token} " + words[x["span1_index"]] + f" {tokenizer.sep_token} " + words[x["span2_index"]]
        return dict(text=text)


class SuperGlueTest:
    def __init__(self, location, model, config, device, tokenizer, rank, world_size, size_dicts, epochs, lr, 
                 seed, batch_size, accumulation_steps,
                 weight_decay, cls_tokens=1, enable_layer_normalizers=False, hpo=None, dataset_key=None, finetune=True):
        self.location = location
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.size_dicts = size_dicts
        self.finetune = finetune
        self.cls_tokens = cls_tokens
        self.hpo = eval(hpo) if hpo is not None else None
        self.seed = seed
        self.enable_layer_normalizers = enable_layer_normalizers

        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.iter_size = accumulation_steps
        self.dataset_key = dataset_key
        self.task_word_map = dict(boolq=dict(true="true", false="false", yes="true", no="false"),
                                  cb=dict(agree="entailment", entailment="entailment", entail="entailment", contradiction="contradiction",
                                          contradict="contradiction", disagree="contradiction", neutral="neutral"),
                                  copa={"0": 0, "1": 1}, multirc=dict(true=1, false=0, yes=1, no=0), record=dict(),
                                  rte=dict(agree= "entailment", entailment="entailment", entail="entailment", contradiction="not_entailment", contradict= "not_entailment", disagree= "not_entailment", neutral= "not_entailment"))
        self.task_word_map["wic"] = self.task_word_map["boolq"]
        self.task_word_map["axg"] = self.task_word_map["rte"]
        self.task_word_map["axb"] = self.task_word_map["rte"]
        self.task_word_map["wsc.fixed"] = self.task_word_map["boolq"]
        # self.epoch_per_dataset = {"boolq": 35, 'cb': 150, 'copa': 200, 'multirc': 19, 'record': 2, 'rte': 200, 'wic': 100, 'wsc.fixed': 200}
        self.epochs = epochs
        # self.lr_per_dataset = {"boolq": lr, 'cb': lr, 'copa': lr, 'multirc': lr, 'record': lr, 'rte': lr, 'wic': lr, 'wsc.fixed': lr}

        self.num_to_word = dict(boolq={0: "false", 1: "true"}, cb={0: "entailment", 1: "contradiction", 2: "neutral"}, rte={0: "entailment", 1: "not_entailment"})

        self.superglue_file_names = dict(zip(['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed', 'axb', 'axg'],
                                             ["BoolQ.jsonl", "CB.jsonl", "COPA.jsonl", "MultiRC.jsonl", "ReCoRD.jsonl", "RTE.jsonl",
                                              "WiC.jsonl", "WSC.jsonl", "AX-b.jsonl", "AX-g.jsonl"]))

    def build_model(self, model):
        set_seeds(self.seed)
        batch_size = self.batch_size
        dataloader_params = dict(persistent_workers=True, prefetch_factor=2)
        if isinstance(model, (FastFormerModel, FastFormerPreTrainedModel, FastFormerForClassification, FastFormerForFusedELECTRAPretraining)):
            model = model.train()
            optimizer_config.eps = 1e-7
            model.config.eps = 1e-7
            tokenizer = model.tokenizer
        elif isinstance(model, str):
            if "deberta" in model.lower() or "large" in model.lower():
                batch_size = batch_size // 2
                self.iter_size *= 2
            if "conv" in model.lower():

                dataloader_params = dict()
            if "fast-conv" in model.lower():

                dataloader_params = dict(persistent_workers=True, prefetch_factor=2)
            if self.finetune:
                batch_size = batch_size // 2
                self.iter_size *= 2
            from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining, CTRLConfig, CTRLPreTrainedModel
            from transformers.models.deberta import DebertaModel
            if os.path.exists(model):
                model_name = model.split("/")[-1].split(".")[0]
                try:
                    main_model, tokenizer = get_mtt_backbone(model_name, self.cls_tokens, self.enable_layer_normalizers, None, reinit=False, train_layer_normalizers=False, enable_layer_normalizers_statistics=self.enable_layer_normalizers, dropout_prob=0.1)
                except:
                    main_model, tokenizer = get_mtt_backbone(model, self.cls_tokens, self.enable_layer_normalizers, None, reinit=False, train_layer_normalizers=False, enable_layer_normalizers_statistics=self.enable_layer_normalizers, dropout_prob=0.1)
                main_model = main_model.to(self.device)
                state_dict = torch.load(model, map_location=self.device)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                try:
                    main_model.load_state_dict(state_dict, strict=True)
                except:
                    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
                    main_model.load_state_dict(state_dict, strict=True)
                model = main_model
            else:

                tokenizer = AutoTokenizer.from_pretrained(model)
                model = AutoModel.from_pretrained(model)
            model = model.train()
            optimizer_config.eps = 1e-7
            for p in model.parameters():
                p.requires_grad = self.finetune
            if not self.finetune:
                model = model.eval()
        elif isinstance(model, DDP):
            tokenizer = model.module.tokenizer
            model = model.module
            assert isinstance(model, FastFormerForClassification)
        else:
            print(type(model))
            raise ValueError
        return dict(model=model, tokenizer=tokenizer, dataloader_params=dataloader_params, batch_size=batch_size)

    def prepare_classifier(self, model_dict, dataset, device, num_classes, dataset_key, rank, reinit=False):
        set_seeds(self.seed)
        train_backbone = self.finetune
        num_workers = 4
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        dataloader_params = model_dict["dataloader_params"]
        batch_size = model_dict["batch_size"]

        # rnd = torch.tensor(random.randint(0, 2**32 - 1)).to(device)
        # dist.broadcast(rnd, 0)

        optc = optimizer_config.to_dict()
        optimizer = None
        scheduler = None
        ddp_model = model
        if reinit or not isinstance(model, (FastFormerForClassification, DDP)):
            classifier = FastFormerForClassification(model.config if hasattr(model, "config") else None, num_classes, model, tokenizer, train_backbone=train_backbone)
            classifier.backbone = copy.deepcopy(model.backbone if hasattr(model, "backbone") else model)
            classifier = classifier.to(device)
            del model
            model = classifier
            ddp_model = DDP(model, device_ids=None if self.device == torch.device("cpu") else [self.device], find_unused_parameters=True, bucket_cap_mb=10)  # find_unused_parameters=True
            try:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
            except:
                print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))
            clean_memory()
            optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=self.lr, eps=optc["eps"], weight_decay=self.weight_decay,
                                          betas=(optc["beta_1"], optc["beta_2"]))
            optimizer.zero_grad(set_to_none=True)

        collate_fn = get_collate_fn(model.config.num_highway_cls_tokens if hasattr(model, "config") and isinstance(model.config, FastFormerConfig) else 0, tokenizer.pad_token_id)

        train = None
        if "train" in dataset:
            train = MTTDataset(self.cls_tokens, len(tokenizer), tokenizer,
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512 - (self.cls_tokens - 1)),
                               dataset["train"])
            train.training = False
            train = DataLoader(train, sampler=None if self.world_size == 1 else DistributedSampler(train, shuffle=True), batch_size=batch_size,
                               collate_fn=collate_fn, num_workers=num_workers, shuffle=self.world_size==1, **dataloader_params)

            iter_size = self.iter_size
            steps_per_epoch = int(np.ceil(len(train.sampler) / (batch_size * iter_size)) if train.sampler is not None else (len(train) / iter_size))
            print("epochs = ", int(self.epochs), " steps_per_epoch=", steps_per_epoch, " lr=", self.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, epochs=int(self.epochs), steps_per_epoch=steps_per_epoch, div_factor=1e2,
                                                            three_phase=False, pct_start=0.1, anneal_strategy="linear")

        validation = None
        if "validation" in dataset:
            validation = MTTDataset(self.cls_tokens, len(tokenizer), tokenizer,
                                    dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512 - (self.cls_tokens - 1)),
                                    dataset["validation"])
            validation.training = False
            validation = DataLoader(validation, sampler=None, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                                    shuffle=False, **dataloader_params)

        test = None
        test_idx = None
        if rank == 0:
            test = MTTDataset(self.cls_tokens, len(tokenizer), tokenizer,
                              dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512 - (self.cls_tokens - 1)),
                              dataset["test"])
            test.training = False
            if "idx" in dataset["test"][0]:
                test_idx = [dataset["test"][i]["idx"] for i in range(len(dataset["test"]))]
            else:
                test_idx = list(range(len(dataset["test"])))
            test = DataLoader(test, sampler=None, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                              shuffle=False, **dataloader_params)

        return dict(model=ddp_model, optimizer=optimizer, scheduler=scheduler, train=train, tokenizer=tokenizer,
                    validation=validation, test=test, optc=optc, test_idx=test_idx, num_classes=num_classes,
                    dataset_key=dataset_key, rank=rank, train_backbone=train_backbone)

    def train_classifier(self, model, device, classifier_data, predict_only=False):
        all_val_loss = []
        all_val_acc = []
        all_train_acc = []
        val_acc = -1
        val_loss = -1
        stored_state_val_loss = -1
        stored_state_val_acc = -1
        train_acc = -1
        epochs = -1
        rank = classifier_data["rank"]
        dataset_key = classifier_data["dataset_key"]
        train_backbone = classifier_data["train_backbone"]
        max_allowed_epochs = int(self.epochs)

        broken = False
        stored_state = None
        if not predict_only:
            gradient_clipping = classifier_data["optc"]["gradient_clipping"]
            scheduler = classifier_data["scheduler"]
            optimizer = classifier_data["optimizer"]

            optimizer.zero_grad(set_to_none=True)
            iter_size = self.iter_size
            epochs = 0
            # (len(all_val_loss) >= 3 and (all_val_loss[-1] <= all_val_loss[-2] or all_val_loss[-2] <= all_val_loss[-3]))
            pbar = None
            if rank == 0:
                pbar = tqdm(total=max_allowed_epochs * len(classifier_data["train"]), desc="%s train" % dataset_key)
            while epochs < max_allowed_epochs:
                train_labels, train_predictions = [], []
                model = model.train()
                trainer = classifier_data["train"]
                if hasattr(trainer, "sampler") and hasattr(trainer.sampler, "set_epoch"):
                    trainer.sampler.set_epoch(epochs)
                for step, batch in enumerate(trainer):
                    batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}

                    label = batch.pop("label")
                    train_labels.extend(label.cpu().tolist())
                    # print(label.size(), batch['input_ids'].size())
                    if (step + 1) % iter_size != 0:
                        with model.no_sync():
                            output = model(**batch, label=label)
                            loss = output["loss"] / iter_size
                            loss.backward()
                    else:
                        output = model(**batch, label=label)
                        loss = output["loss"] / iter_size
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    preds = output["predictions"].cpu().tolist()
                    train_predictions.extend(preds)
                    if rank == 0:
                        pbar.update()

                train_predictions = (np.array(train_predictions) > 0.5) if classifier_data["num_classes"] == 1 else train_predictions
                train_acc = accuracy_score(train_labels, train_predictions)
                all_train_acc.append(train_acc)
                continue_training = torch.tensor(2).to(device)
                per_epoch = 3 if max_allowed_epochs < 50 and not train_backbone else 5
                model = model.eval()
                if epochs % per_epoch == 0 and rank == 0 and self.hpo is None and False:
                    inner_model = model.module
                    labels, predictions, val_losses = [], [], []
                    with model.no_sync():
                        for step, batch in enumerate(tqdm(classifier_data["validation"], desc="%s validation" % dataset_key)):
                            batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
                            label = batch.pop("label")
                            labels.extend(label.cpu().tolist())
                            with torch.no_grad():
                                output = inner_model(**batch, label=label)
                            val_loss = output["loss"].detach().cpu().item()
                            val_preds = output["predictions"].cpu().tolist()
                            val_preds = val_preds if isinstance(val_preds, (list, tuple)) else [val_preds]
                            predictions.extend(val_preds)
                            val_losses.append(val_loss)
                    cur_val_loss = np.mean(val_losses)
                    # cur_val_loss = torch.tensor(cur_val_loss).to(device)
                    # tensor_list = [cur_val_loss.new_empty(cur_val_loss.size()) for _ in range(self.world_size)]
                    # torch.distributed.all_gather(tensor_list, cur_val_loss)
                    # cur_val_loss = torch.stack(tensor_list).mean().item()
                    all_val_loss.append(cur_val_loss)
                    val_acc = accuracy_score(labels, (np.array(predictions) > 0.5) if classifier_data["num_classes"] == 1 else predictions)
                    # val_acc = torch.tensor(val_acc).to(device)
                    # tensor_list = [val_acc.new_empty(val_acc.size()) for _ in range(self.world_size)]
                    # torch.distributed.all_gather(tensor_list, val_acc)
                    # val_acc = torch.stack(tensor_list).mean().item()
                    all_val_acc.append(val_acc)

                    if len(all_val_loss) >= 3 and all_val_loss[-1] > all_val_loss[-2] and all_val_loss[-2] > all_val_loss[-3] and epochs > max(max_allowed_epochs / 2, 3):
                        continue_training = torch.tensor(0).to(device)
                    elif (len(all_val_loss) >= 2 and all_val_loss[-1] <= all_val_loss[-2]) or stored_state is None:
                        continue_training = torch.tensor(1).to(device)
                        stored_state_val_acc = val_acc
                        stored_state_val_loss = all_val_loss[-1]

                epochs += 1
                torch.distributed.barrier()
                dist.broadcast(continue_training, 0)
                if continue_training.item() == 0 and self.hpo is None and False:
                    model.load_state_dict(stored_state)
                    optimizer.zero_grad(set_to_none=True)
                    broken = True
                    break
                elif continue_training.item() == 1 and self.hpo is None:
                    stored_state = copy.deepcopy(model.state_dict().copy())

            torch.distributed.barrier()
            if rank == 0:
                pbar.close()

            if stored_state is not None:
                model.load_state_dict(stored_state)
                stored_state = {k.replace("module.", ""): v for k, v in stored_state.items()}
                model.module.load_state_dict(stored_state, strict=True)

            if rank == 0:
                inner_model = model.module
                if stored_state is not None:
                    stored_state = {k.replace("module.", ""): v for k, v in stored_state.items()}
                    inner_model.load_state_dict(stored_state, strict=True)
                labels, predictions, val_losses = [], [], []
                with model.no_sync():
                    for step, batch in enumerate(tqdm(classifier_data["validation"], desc="%s validation" % dataset_key)):
                        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
                        label = batch.pop("label")
                        labels.extend(label.cpu().tolist())
                        with torch.no_grad():
                            output = inner_model(**batch, label=label)
                        val_loss = output["loss"].detach().cpu().item()
                        val_preds = output["predictions"].cpu().tolist()
                        val_preds = val_preds if isinstance(val_preds, (list, tuple)) else [val_preds]
                        predictions.extend(val_preds)
                        val_losses.append(val_loss)
                cur_val_loss = np.mean(val_losses)
                val_loss = cur_val_loss
                # cur_val_loss = torch.tensor(cur_val_loss).to(device)
                # tensor_list = [cur_val_loss.new_empty(cur_val_loss.size()) for _ in range(self.world_size)]
                # torch.distributed.all_gather(tensor_list, cur_val_loss)
                # cur_val_loss = torch.stack(tensor_list).mean().item()
                all_val_loss.append(cur_val_loss)
                val_acc = accuracy_score(labels, (np.array(predictions) > 0.5) if classifier_data["num_classes"] == 1 else predictions)
                # val_acc = torch.tensor(val_acc).to(device)
                # tensor_list = [val_acc.new_empty(val_acc.size()) for _ in range(self.world_size)]
                # torch.distributed.all_gather(tensor_list, val_acc)
                # val_acc = torch.stack(tensor_list).mean().item()
                all_val_acc.append(val_acc)

                # Test

            torch.distributed.barrier()
        predictions = []
        if hasattr(model, "no_sync") and rank == 0 and self.hpo is None:
            model = model.eval()
            inner_model = model.module
            if stored_state is not None:
                stored_state = {k.replace("module.", ""): v for k, v in stored_state.items()}
                inner_model.load_state_dict(stored_state, strict=True)
            for step, batch in enumerate(tqdm(classifier_data["test"], desc="%s test" % dataset_key)):
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
                _ = batch.pop("label", None)
                with torch.no_grad():
                    with model.no_sync():
                        output = inner_model(**batch, label=None)
                test_preds = output["predictions"].cpu().tolist()
                test_preds = test_preds if isinstance(test_preds, (list, tuple)) else [test_preds]
                predictions.extend(test_preds)
        elif rank == 0 and self.hpo is None:
            val_acc = 0.0
            model = model.eval()
            inner_model = model
            if stored_state is not None:
                stored_state = {k.replace("module.", ""): v for k, v in stored_state.items()}
                inner_model.load_state_dict(stored_state, strict=True)
            for step, batch in enumerate(tqdm(classifier_data["test"], desc="%s test" % dataset_key)):
                batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
                _ = batch.pop("label", None)
                with torch.no_grad():
                    output = inner_model(**batch, label=None)
                test_preds = output["predictions"].cpu().tolist()
                test_preds = test_preds if isinstance(test_preds, (list, tuple)) else [test_preds]
                predictions.extend(test_preds)

        del model
        clean_memory()
        if rank != 0:
            return None
        print("For %s: Train = %.4f, Val = %.4f, stored_state_val_acc = %.4f, stored_state_val_loss = %.4f" % (dataset_key, train_acc, val_acc, stored_state_val_acc, stored_state_val_loss))
        print("For %s: all_val_loss = %s, all_val_accuracy = %s" % (dataset_key, all_val_loss, all_val_acc))
        return dict(val_acc=val_acc, train_acc=train_acc, predictions=predictions, all_val_loss=all_val_loss, all_val_acc=all_val_acc,
                    all_train_acc=all_train_acc, epochs=epochs, broken=broken, val_loss=val_loss)

    def mnli(self, model, mnli, device, dataset_key, rank):
        mnli = load_dataset("multi_nli")
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        mnli["validation"] = concatenate_datasets([mnli["validation_matched"], mnli["validation_mismatched"]])
        mnli["test"] = mnli["validation_mismatched"]
        test_labels = [m["label"] for m in mnli["test"]]
        mnli = mnli.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        classifier_data = self.prepare_classifier(model_dict, mnli, device, 3, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            test_accuracy = accuracy_score(test_labels, classifier_results["predictions"])
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"],
                              test_accuracy=test_accuracy)
        test_accuracy = accuracy_score(test_labels, classifier_results["predictions"])
        return None, dict(dataset="mnli", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                          epochs=classifier_results["epochs"], test_acc=test_accuracy,
                          val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                          model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def swag(self, model, swag, device, dataset_key, rank):
        swag = load_dataset("swag", "regular")
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        swag_c1 = swag.map(
            lambda x: dict(text=x["startphrase"] + f" {tokenizer.sep_token} " + x["ending0"], label=x["label"] == 0,
                           choice=0),
            remove_columns=["startphrase", 'ending0', "ending1", "ending2", "ending3", "sent1", "sent2"])
        swag_c2 = swag.map(
            lambda x: dict(text=x["startphrase"] + f" {tokenizer.sep_token} " + x["ending1"], label=x["label"] == 1,
                           choice=1),
            remove_columns=["startphrase", 'ending0', "ending1", "ending2", "ending3", "sent1", "sent2"])
        swag_c3 = swag.map(
            lambda x: dict(text=x["startphrase"] + f" {tokenizer.sep_token} " + x["ending2"], label=x["label"] == 2,
                           choice=2),
            remove_columns=["startphrase", 'ending0', "ending1", "ending2", "ending3", "sent1", "sent2"])
        swag_c4 = swag.map(
            lambda x: dict(text=x["startphrase"] + f" {tokenizer.sep_token} " + x["ending3"], label=x["label"] == 3,
                           choice=3),
            remove_columns=["startphrase", 'ending0', "ending1", "ending2", "ending3", "sent1", "sent2"])
        swag = DatasetDict({k: concatenate_datasets([v, swag_c2[k], swag_c3[k], swag_c4[k],]) for k, v in swag_c1.items()})
        classifier_data = self.prepare_classifier(model_dict, swag, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        choices = [swag["test"][i]["choice"] for i in range(len(swag["test"]))]
        final_predictions = [dict(idx=idx, label=pred, choice=ch) for idx, pred, ch in zip(test_idx, classifier_results["predictions"], choices)]
        final_predictions = pd.DataFrame.from_records(final_predictions).groupby("idx", group_keys=False).apply(
            lambda x: x[x.label >= x.label.max()][["idx", "choice"]].rename(columns={"choice": "label"})).to_dict('records')
        return final_predictions, dict(dataset="copa", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def hellaswag(self):
        pass

    def anli(self):
        pass

    def esnli(self):
        pass

    def snli(self):
        pass

    def boolq(self, model, boolq, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        boolq = boolq.map(lambda x: dict(text=x["passage"] + f" {tokenizer.sep_token} " + x["question"]), remove_columns=['question', 'passage'])
        classifier_data = self.prepare_classifier(model_dict, boolq, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        # print(classifier_results["predictions"])
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="boolq", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def wic(self, model, wic, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        wic_reversed = wic.map(lambda x: dict(text=x["sentence2"] + f" {tokenizer.sep_token} " + x["sentence1"] + f" {tokenizer.sep_token} " + x["word"]),
                remove_columns=['sentence1', 'sentence2', "word"])
        wic = wic.map(lambda x: dict(text=x["sentence1"] + f" {tokenizer.sep_token} " + x["sentence2"] + f" {tokenizer.sep_token} " + x["word"]), remove_columns=['sentence1', 'sentence2', "word"])
        wic["train"] = concatenate_datasets((wic["train"], wic_reversed["train"]))
        classifier_data = self.prepare_classifier(model_dict, wic, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="wic", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"], val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)
    
    def cb(self, model, cb, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        cb = cb.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        classifier_data = self.prepare_classifier(model_dict, cb, device, 3, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["cb"][pred]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="cb", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def rte_axb_axg(self, model, rte, axb, axg, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        rte = rte.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        classifier_data = self.prepare_classifier(model_dict, rte, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None, None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"]), None, None
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["rte"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]

        rte_res = dict(dataset="rte", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)
        axb = axb.map(lambda x: dict(text=x["sentence1"] + f" {tokenizer.sep_token} " + x["sentence2"]), remove_columns=["sentence1", "sentence2"])
        model_dict["model"] = classifier_data["model"]
        classifier_data = self.prepare_classifier(model_dict, axb, device, 1, dataset_key, rank, reinit=False)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data, predict_only=True)
        test_idx = classifier_data["test_idx"]
        final_predictions_axb = [dict(idx=idx, label=self.num_to_word["rte"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]

        axg = axg.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        classifier_data = self.prepare_classifier(model_dict, axg, device, 1, dataset_key, rank, reinit=False)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data, predict_only=True)
        test_idx = classifier_data["test_idx"]
        final_predictions_axg = [dict(idx=idx, label=self.num_to_word["rte"][int(pred > 0.5)]) for idx, pred in
                                 zip(test_idx, classifier_results["predictions"])]

        return final_predictions, rte_res, final_predictions_axb, final_predictions_axg

    def multirc(self, model, multirc, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        multirc = multirc.map(lambda x: dict(text=x["paragraph"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer"]), remove_columns=["paragraph", "question", "answer"])
        classifier_data = self.prepare_classifier(model_dict, multirc, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=pred > 0.5) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        mrcp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for pd in final_predictions:
            mrcp[pd["idx"]["paragraph"]][pd["idx"]["question"]][pd["idx"]["answer"]] = pd["label"]
        mrcp = [
            {"idx": k, "passage": {"questions": [{"idx": m, "answers": [{"idx": o, "label": p} for o, p in sorted(n.items())]} for m, n in sorted(v.items())]}}
            for k, v in sorted(mrcp.items())]
        final_predictions = mrcp
        return final_predictions, dict(dataset="multirc", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"], 
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def copa(self, model, copa, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        copa_c1 = copa.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice1"], label=x["label"] == 0, choice=0),
                           remove_columns=["premise", 'question', "choice1", "choice2"])
        copa_c2 = copa.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice2"], label=x["label"] == 1, choice=1),
                           remove_columns=["premise", 'question', "choice1", "choice2"])
        copa = DatasetDict({k: concatenate_datasets([v, copa_c2[k]]) for k, v in copa_c1.items()})

        classifier_data = self.prepare_classifier(model_dict, copa, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        choices = [copa["test"][i]["choice"] for i in range(len(copa["test"]))]
        final_predictions = [dict(idx=idx, label=pred, choice=ch) for idx, pred, ch in zip(test_idx, classifier_results["predictions"], choices)]
        final_predictions = pd.DataFrame.from_records(final_predictions).groupby("idx", group_keys=False).apply(lambda x: x[x.label >= x.label.max()][["idx", "choice"]].rename(columns={"choice": "label"})).to_dict('records')
        return final_predictions, dict(dataset="copa", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def record(self, model, record, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        rtest = record["test"]
        record = record.map(rproc(tokenizer), batched=True, batch_size=1, remove_columns=["answers", "passage", "query"])
        classifier_data = self.prepare_classifier(model_dict, record, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data, predict_only=False)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        choices = [record["test"][i]["choice"] for i in range(len(record["test"]))]
        final_predictions = [dict(idx=idx, label=pred, choice=ch) for idx, pred, ch in zip(test_idx, classifier_results["predictions"], choices)]
        final_predictions = pd.DataFrame.from_records(final_predictions).groupby("idx", group_keys=False).apply(
            lambda x: x[x.label >= x.label.max()][["idx", "choice"]].head(1)).to_dict('records')

        entities = [rtest[i]["entities"] for i in range(len(rtest))]
        for fp, en in zip(final_predictions, entities):
            if len(en) - 1 < fp["choice"]:
                print(en, fp["choice"], fp["idx"])
        final_predictions = [dict(idx=fp["idx"], label=en[fp["choice"]]) for fp, en in zip(final_predictions, entities)]
        return final_predictions, dict(dataset="record", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def wsc(self, model, wsc, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        wsc = wsc.map(wsc_proc(tokenizer), remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"])
        classifier_data = self.prepare_classifier(model_dict, wsc, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"], val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="wsc.fixed", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def __call__(self, generate_test_predictions=True):
        tokenizer = self.tokenizer
        model = self.model.to(self.device).eval() if not isinstance(self.model, str) else self.model
        size_dicts = self.size_dicts
        pred_datas = []

        # glue = dict()
        # for gl in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:
        #     glue[gl] = load_dataset("glue", gl)

        super_glue, _ = superglue_test(test_only=False, pet_dataset=False)
        keys = ['cb', 'copa', 'multirc', 'record', 'wsc.fixed', 'rte', 'boolq', 'wic',]  # 'axb', 'axg'
        if self.hpo is not None:
            assert self.dataset_key is not None
            dataset = super_glue[self.dataset_key]
            hpo = self.hpo
            hpo_keys = list(hpo.keys())
            hpo_combinations = np.array(np.meshgrid(*hpo.values())).T.reshape(-1, len(hpo_keys))
            dk = self.dataset_key
            results = []
            for idx, c in enumerate(hpo_combinations):
                for k, v in zip(hpo_keys, c):
                    setattr(self, k, v)
                if dk == "boolq":
                    _, pred_data = self.boolq(model, dataset, self.device, dk, self.rank)
                elif dk == "cb":
                    _, pred_data = self.cb(model, dataset, self.device, dk, self.rank)
                elif dk == "copa":
                    _, pred_data = self.copa(model, dataset, self.device, dk, self.rank)
                elif dk == "multirc":
                    _, pred_data = self.multirc(model, dataset, self.device, dk, self.rank)
                elif dk == "record":
                    _, pred_data = self.record(model, dataset, self.device, dk, self.rank)
                elif dk == "wic":
                    _, pred_data = self.wic(model, dataset, self.device, dk, self.rank)
                elif dk == "wsc.fixed":
                    _, pred_data = self.wsc(model, dataset, self.device, dk, self.rank)
                elif dk == "rte":
                    _, pred_data, _, _ = self.rte_axb_axg(model, dataset, super_glue["axb"], super_glue["axg"], self.device, dk, self.rank)
                elif dk == "swag":
                    _, pred_data = self.swag(model, None, self.device, dk, self.rank)
                elif dk == "mnli":
                    _, pred_data = self.mnli(model, None, self.device, dk, self.rank)
                elif dk == "hellaswag":
                    _, pred_data = self.hellaswag(model, None, self.device, dk, self.rank)
                elif dk == "anli":
                    _, pred_data = self.anli(model, dataset, self.device, dk, self.rank)
                else:
                    raise NotImplementedError
                if self.rank == 0:
                    res_dict = dict(zip(hpo_keys, c))
                    res_dict["val_acc"] = pred_data["val_acc"]
                    res_dict["train_acc"] = pred_data["train_acc"]
                    res_dict["val_loss"] = pred_data["val_loss"]
                    if "test_acc" in pred_data:
                        res_dict["test_acc"] = pred_data["test_acc"]
                    results.append(res_dict)
            if self.rank == 0:
                print(tabulate(results, headers="keys", tablefmt="grid"))
            return None

        if self.dataset_key is not None:
            keys = [self.dataset_key]

        elif os.path.exists(os.path.join(os.getcwd(), 'validation.txt')):
            with open('validation.txt') as f:
                my_list = [eval(x.rstrip()) for x in f if len(x.rstrip()) > 0]
                if self.rank == 0:
                    print(my_list)
            processed_datasets = [one['dataset'] for one in my_list]
            keys = [k for k in keys if k not in processed_datasets]

        for idx, dk in enumerate(keys):
            print("[SUPERGLUE]: Time = %s, Train for Rank = %s/%s, dataset = %s, device = %s, idx = %s" % (get_time_string(), self.rank, self.world_size, dk, self.device, idx))
            dataset = super_glue[dk] if dk in super_glue else None
            torch.distributed.barrier()
            if dk == "boolq":
                final_predictions, pred_data = self.boolq(model, dataset, self.device, dk, self.rank)
            elif dk == "cb":
                final_predictions, pred_data = self.cb(model, dataset, self.device, dk, self.rank)
            elif dk == "copa":
                final_predictions, pred_data = self.copa(model, dataset, self.device, dk, self.rank)
            elif dk == "multirc":
                final_predictions, pred_data = self.multirc(model, dataset, self.device, dk, self.rank)
            elif dk == "record":
                final_predictions, pred_data = self.record(model, dataset, self.device, dk, self.rank)
            elif dk == "wic":
                final_predictions, pred_data = self.wic(model, dataset, self.device, dk, self.rank)
            elif dk == "wsc.fixed":
                final_predictions, pred_data = self.wsc(model, dataset, self.device, dk, self.rank)
            elif dk == "rte":
                final_predictions, pred_data, final_predictions_axb, final_predictions_axg = self.rte_axb_axg(model, dataset, super_glue["axb"], super_glue["axg"], self.device, dk, self.rank)
            elif dk == "mnli":
                final_predictions, pred_data = self.mnli(model, None, self.device, dk, self.rank)
            elif dk == "swag":
                final_predictions, pred_data = self.swag(model, None, self.device, dk, self.rank)
            elif dk == "anli":
                final_predictions, pred_data = self.anli(model, None, self.device, dk, self.rank)
            elif dk == "hellaswag":
                final_predictions, pred_data = self.hellaswag(model, None, self.device, dk, self.rank)

            _ = gc.collect()
            if self.rank == 0:
                if "model" in pred_data:
                    torch.save(pred_data.pop("model").state_dict(), model+("" if model.endswith("."+dk) else ("."+dk)))
                print("val_acc: %s" % pred_data["val_acc"])
                print("train_acc: %s" % pred_data["train_acc"])
                print("val_loss: %s" % pred_data["val_loss_hist"][0])
                if dk in self.superglue_file_names:
                    with jsonlines.open(self.superglue_file_names[dk], mode='w') as writer:
                        writer.write_all(final_predictions)
                    if dk == "rte":
                        with jsonlines.open(self.superglue_file_names["axb"], mode='w') as writer:
                            writer.write_all(final_predictions_axb)
                        with jsonlines.open(self.superglue_file_names["axg"], mode='w') as writer:
                            writer.write_all(final_predictions_axg)
                pred_datas.append(pred_data)
                with open('validation.txt', 'a') as f:
                    print(str(pred_data), file=f)
            # import pandas as pd
            # print(pd.DataFrame.from_records(pred_datas))
        if self.rank == 0:
            print(pred_datas)
            print(tabulate(pred_datas, headers="keys", tablefmt="grid"))



class LargeValidator:
    def __init__(self, location, model, config, device, tokenizer, rank, world_size, size_dicts, no_autocast=False):
        self.location = location
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.ignore_keys = ['cnn_dailymail1024',
                            'wikihow_sep1024',
                            # 'wikihow_all1024',
                            'big_patent1024',
                            'xsum1024',
                            'scientific_papers_arxiv512',
                            'scientific_papers_pubmed512',
                            'kelm1024',
                            # 'amazon_reviews_multi1024',
                            'gigaword_qna',
                            'mrqa_v1',
                            'mrqa_v2',
                            'e2e_nlg_cleaned_qna',
                            'discovery_qna',
                            'hans_qna_v2',
                            'hans_qna_v1',
                            'wikihow_sep_qna_v2',
                            'wikihow_sep_qna_v3',
                            'wikihow_sep_qna_v1',
                            'empathetic_dialogues_qna'
                            ]
        self.includes = ['superglue_cb_v2',
                         'superglue_cb_v1',
                         'superglue_copa_v1',
                         'superglue_copa_v2',
                         'superglue_copa_v3',
                         'superglue_wsc_fixed_v2',
                         'superglue_wsc_v1',
                         'superglue_wsc_fixed_v1',
                         'superglue_wsc_v2',
                         'superglue_rte_v2',
                         'superglue_rte_v1',
                         'superglue_wic_v1',
                         'superglue_wic_v2',
                         'superglue_multirc_v3',
                         'superglue_boolq',
                         'superglue_record_v4',
                         'superglue_record_v3',
                         'superglue_multirc_v1',
                         'superglue_multirc_v2',
                         'superglue_record_v1',
                         'superglue_record_v2',
                         'snli_qna_v1',
                         'race_qna',
                         'glue_sst2_v2',
                         'glue_sst2',
                         'glue_qnli',
                         'rotten_tomatoes_qna',
                         'commonsense_qa',
                         'winogrande_qna',
                         'scitail_qna',
                         'hellaswag_qna',
                         'squad_v2_qna',
                         'squad_v2_qna_v2',
                         'swag_qna'
                         ]
        self.no_autocast = no_autocast
        self.size_dicts = size_dicts

    def __call__(self):
        # TODO: save model if val acc higher than before
        # TODO: build a full score for val set using scores from all datasets
        # TODO: WnB integration from root process
        # TODO: parallel validation
        # TODO: Resume:: save modelw with epoch number and give ability to save n_models only, save optimizer, save scheduler, num_steps already done also needs to be saved

        # TODO: Lower LR by lambda LR or step LR by step counting in a deterministic way
        # WanDB control decide if init or not and make shim
        # Better Batching
        size_dicts = self.size_dicts
        datadict = DatasetDict.load_from_disk(self.location)
        tokenizer = self.tokenizer
        model = self.model.to(self.device)
        model = model.eval()
        collate_fn = get_collate_fn(self.config.num_highway_cls_tokens, tokenizer.pad_token_id)
        results = dict()
        _ = [datadict.pop(k, None) for k in self.ignore_keys]
        datadict = {k: v for k, v in datadict.items() if k in self.includes}
        print("[Validation]: Time = %s, Rank = %s, Total Datasets for Val = %s" % (get_time_string(), self.rank, len(datadict)))
        for idx, (k, dataset) in enumerate(sorted(datadict.items())):
            while idx >= self.world_size:
                idx -= self.world_size
            if idx != self.rank:
                continue
            clean_memory()
            cns = dataset.column_names
            predictions = []
            if 'answer' in cns:
                labels = [dataset[i]["answer"] for i in range(len(dataset))]
            dataset = TokenizerDataset(self.config, tokenizer, get_char_to_id(),
                                       dict(padding="max_length", truncation=True, return_tensors="pt", max_length=self.config.tokenizer_length),
                                       dataset)
            dataset.training = False
            record_accuracy = False
            if 'answer' not in cns:
                dataset.training = True
                record_accuracy = True
            length = len(dataset)
            print("[Validation]: Time = %s, Rank = %s, Prepare-Validation-Dataset, Val for dataset = %s, length = %s, with columns = %s" % (get_time_string(), self.rank, k, len(dataset), cns))
            loader = DataLoader(dataset, sampler=None, batch_size=min(size_dicts.values()), collate_fn=collate_fn, prefetch_factor=2, num_workers=4)
            # loader = custom_batching_fn(loader, size_dicts, False)
            # loader = custom_batching_fn(tqdm(loader, desc=k, miniters=100, mininterval=30.0), size_dicts_val, False)

            for pt_batch in loader:
                pt_batch["record_accuracy"] = record_accuracy
                pt_batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in pt_batch.items()}
                # print("[Validation]: Time = %s, Rank = %s, Start-Validation, Val for dataset = %s, batch size = %s, first batch loaded" % (get_time_string(), self.rank, k, pt_batch["input_ids"].size()))
                if 'answer' in cns:
                    with torch.no_grad():

                        funnel_inputs = dict(input_ids=pt_batch["input_ids"],
                                             attention_mask=pt_batch["attention_mask"],
                                             token_type_ids=pt_batch["token_type_ids"] if "token_type_ids" in pt_batch else None,
                                             inputs_embeds=None,
                                             char_ids=pt_batch["char_ids"], char_offsets=pt_batch["char_offsets"],
                                             run_decoder=False,
                                             run_answering=True)
                        output = model.backbone(**funnel_inputs)
                        # print("[Validation]: Time = %s, Rank = %s, run-backbone-validation, Val for dataset = %s, Funnel model run" % (get_time_string(), self.rank, k))
                        answering_predictions = output["answering_logits"].detach().argmax(dim=-1)
                    # debug_answering_predictions = answer_decoder_debug(answering_predictions, tokenizer)
                    # print("[Validation]: Time = %s, Rank = %s, Mid-Validation, Val for dataset = %s, Answering preds = %s, inps = %s" % (get_time_string(), self.rank, k, debug_answering_predictions, answering_predictions[:, :8].tolist()))
                    answering_predictions = answer_decoder(answering_predictions, tokenizer)
                    predictions.extend(answering_predictions)

                else:
                    labels = pt_batch["label_mlm_input_ids"] if "label_mlm_input_ids" in pt_batch else pt_batch["input_ids"]
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        with autocast():
                            output = model(**pt_batch, labels=labels)["accuracy_hist"]
                    predictions.append(output)


            print("[Validation]: Time = %s, Rank = %s, For Dataset %s, Built predictions list, samples = %s" % (get_time_string(), self.rank, k, list(zip(labels, predictions))[:4]))
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
                most_common_label = Counter(final_labels).most_common(1)[0][0]
                most_common_label_count = Counter(final_labels).most_common(1)[0][1]
                discount_score = most_common_label_count / len(final_labels)
                results[k] = dict(accuracy=score, common_class_accuracy=discount_score)
            else:
                results[k] = pd.DataFrame.from_records(predictions).mean().to_dict()
                _ = results[k].pop("answering_lm_accuracy", None)
            print("[Validation]: Time = %s, Rank = %s, Finished-Validation, For Dataset %s, results = %s" % (get_time_string(), self.rank, k, results[k]))
            with open('validation.txt', 'a') as f:
                print(str({k: results[k]}) + os.linesep, file=f)

            clean_memory()
        model = model.train()
        return results

def cleanup():

    dist.destroy_process_group()


def build_dataloader(location, shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, size_dicts, continuous_iter=True, world_size=1, num_workers=1):
    assert max(size_dicts.values()) % min(size_dicts.values()) == 0
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    min_size = gcd_array(list(size_dicts.values()) + list(size_dicts.values()))
    prefetch_factor = 2 * (max(size_dicts.values()) // min_size)
    kwargs = dict(prefetch_factor=prefetch_factor) if num_workers > 0 else dict()
    try:
        train_dataset = Dataset.load_from_disk(location)
        train_dataset = TokenizerDataset(config, tokenizer, get_char_to_id(), dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), train_dataset)

        train_loader = DataLoader(train_dataset, sampler=None if single_node else DistributedSampler(train_dataset, shuffle=shuffle_dataset),
                                  batch_size=min_size, collate_fn=collate_fn, shuffle=shuffle_dataset and single_node,
                                  num_workers=num_workers, pin_memory=True, **kwargs)

        train_loader = custom_batching_fn(train_loader, size_dicts, continuous_iter)
    except:
        train_dataset = DatasetDict.load_from_disk(location)
        train_dataset = {k: v for k, v in train_dataset.items() if len(v) >= world_size}
        train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_dataset.items()}
        lsum = sum(train_dataset_sampling_proba.values())
        train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
        train_dataset = {k: TokenizerDataset(config, tokenizer, get_char_to_id(), dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), v) for k, v in train_dataset.items()}
        # for v in train_dataset.values():
        #     v.training = False

        train_loader = {k: DataLoader(v, sampler=None if single_node else DistributedSampler(v, shuffle=shuffle_dataset, ), shuffle=shuffle_dataset and single_node,
                                      batch_size=min_size, collate_fn=collate_fn, num_workers=num_workers, **kwargs) for k, v in train_dataset.items()}

        train_loader = {k: custom_batching_fn(dataloader, size_dicts, continuous_iter) for k, dataloader in train_loader.items()}
        train_loader = datadict_iterator(train_loader, train_dataset_sampling_proba)
    return train_loader


def train_catch_exception(local_rank, args):
    from fastformer.training.train_lm_distributed import train
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


def train_inner_loop(args, ddp_model, batch, labels, optimizer, scheduler, scaler, gradient_clipping, iter_size=1, no_sync=False, zero_grad_check=False):
    # It seems to me like the first accumulated gradients might get clipped several times hence giving more weight to last accumulated gradients :

    output = ddp_model(**batch, labels=labels)
    loss = output["loss"] / iter_size
    loss_dict = output["loss_dict"]
    loss.backward()
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
        ddp_model.clip_grad_norm_(gradient_clipping) if isinstance(ddp_model, FSDP) else torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clipping)
        optimizer.step()
        scheduler.step()

    if "loss" in loss_dict and np.isnan(loss_dict["loss"]):
        es = "[Train-Exception]: Time = %s, NAN Loss, Scale = %s, loss_dict = %s, lr = %s" % (
            get_time_string(), None, loss_dict, optimizer.param_groups[0]['lr'])
        raise ValueError(es)

    return dict(loss_dict=loss_dict, accuracy_hist=output["accuracy_hist"], preds_dict=output["preds_dict"], zero_grad=len(zgradders), inf_grad=len(inf_gradders))


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
    import warnings
    warnings.simplefilter("ignore")
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

    print("[Train]: Time = %s, Initializing Dist Process with init-method = %s for Rank = %s" % (get_time_string(), init_method, rank))
    dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
    print("[Train]: Time = %s, Initialized Dist Process for Rank = %s" % (get_time_string(), rank))
    barrier = get_barrier(True)
    rnd = torch.tensor(int(time.time())).to(device)
    dist.broadcast(rnd, 0)
    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(timezone('Asia/Kolkata')).strftime(format)
    set_seeds(args["seed"])
    model_config.model_size = args["model_config"]
    size_dicts = get_batch_size(args["model_config"], not args["no_autocast"])

    mconf = model_config.to_dict()

    config = config_dict[mconf.pop("model_size")]
    if any(config.relative_attention):
        size_dicts = {k: v - 4 for k, v in size_dicts.items()}
    size_dicts = {1024: args["batch_size"]} if "batch_size" in args and isinstance(args["batch_size"], int) else size_dicts
    tokenizer = get_tokenizer(mconf.pop("tokenizer_name"))
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = 1024
    config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens

    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)

    if args["world_size"] != 128:
        optimizer_config.lr = optimizer_config.lr * (args["world_size"]/128)
    config.eps = 1e-4
    if args["no_autocast"]:
        optimizer_config.eps = 1e-7
        config.layer_norm_eps = 1e-7
        config.eps = 1e-7
        optimizer_config.gradient_clipping = 4 * optimizer_config.gradient_clipping

    fsdp_params = configure_fsdp(not args["no_autocast"], True if not args["no_autocast"] else False, True)
    fsdp_wrapper(wrap_type=0, init=True)
    print("[Train]: Time = %s, Build Model with fsdp params = %s" % (get_time_string(), fsdp_params))

    model = None
    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]) and not args["test_only"]:
        model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)

        try:
            model.load_state_dict(state_dict, strict=False)
        except:

            pos_emb = state_dict["backbone.embeddings.position_embeddings.weight"]
            state_dict["backbone.embeddings.position_embeddings.weight"] = torch.cat((pos_emb[:4], pos_emb, pos_emb[-5:]), 0)
            model.load_state_dict(state_dict, strict=False)
        del state_dict
    elif args["pretrained_model"] is not None and args["test_only"]:
        model = args["pretrained_model"]

    if args["test_only"]:
        SuperGlueTest(None, model, config, device, tokenizer, rank, args["world_size"], size_dicts, args["epochs"], args["lr"],
                      args["seed"], args["batch_size"], args["accumulation_steps"], args["weight_decay"],
                      args["cls_tokens"], args["enable_layer_normalizers"], args["hpo"], args["dataset_key"], args["finetune"])()
        return

    if args["validate_on_start"] or args["validate_only"]:
        _ = LargeValidator(args["validation_dataset"], model, config, device, tokenizer, rank, args["world_size"], size_dicts, args["no_autocast"])()
        clean_memory()
        if args["validate_only"]:
            return

    if model is None:
        model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
    if rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(model) / 1_000_000))
        print(model)

    # ddp_model = FSDP(model, **fsdp_params)  # find_unused_parameters=True
    ddp_model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False, bucket_cap_mb=10)  # find_unused_parameters=True
    try:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
    except:
        print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))
    clean_memory()

    optc = optimizer_config.to_dict()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"]))
    optimizer.zero_grad(set_to_none=True)

    # model, optim, gradscaler, scheduler, steps

    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
    print("[Train]: Time = %s, Optimizer Created for Rank = %s, params = %s" % (get_time_string(), rank, optc))
    shuffle_dataset = args["shuffle_dataset"]
    if not args["validate_only"] and not args["test_only"]:
        train_loader = build_dataloader(args["train_dataset"], shuffle_dataset, 0.75, config, collate_fn, tokenizer, size_dicts, world_size=args["world_size"], num_workers=args["num_workers"])

    print("[Train]: Data Loaded for Rank = %s" % rank)
    validate_every_steps = args["validate_every_steps"]
    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    gradient_clipping = optc["gradient_clipping"]
    print("[Train]: Scheduler Created for Rank = %s" % rank)
    if "resume" in args and isinstance(args["resume"], str) and len(args["resume"].strip()) > 0:
        print("[Train]: Trying Resume from %s for Rank = %s" % (args["resume"], rank))
        other_load_details = load(args["resume"], ddp_model, optimizer, scheduler, None, gpu_device)

        if other_load_details is None:
            print("[Train]: No resume checkpoint from %s for Rank = %s" % (args["resume"], rank))
            args["resume"] = None
        else:
            print("[Train]: Resumed from %s for Rank = %s, other details = %s" % (args["resume"], rank, other_load_details))

    else:
        print("[Train]: No Resume for Rank = %s" % rank)
    _ = ddp_model.train()

    batch_times = []
    model_times = []
    full_times = []
    ddp_model.zero_grad(set_to_none=True)
    samples_processed = 0
    samples_processed_this_log_iter = 0
    print("[Train]: Time = %s, Start Training for Rank = %s" % (get_time_string(), rank))
    ds_name = list(filter(lambda x: len(x.strip()) > 0, args["train_dataset"].split("/")))[-1].replace("train_fastformer_resampled_", "")
    group = "%s-%s-%sN-%s" % (ds_name, args["model_config"], args["nodes"], time_string)
    if local_rank == 0:
        wandb_init_args = dict(project="fastformer", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                               config={"args":args, "model_config": mconf, "config": config, "optimizer_config": optc},
                               settings=wandb.Settings(start_method="fork"))

        time.sleep(random.random() * 5)
        wandb.init(**wandb_init_args)
        print("[Train]: WandB-watch added over model for Rank = %s" % rank)
        # wandb.watch(model, log="all", log_freq=log_every_steps * 4)
    barrier()

    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)

    def get_hook(name_of_param=None):
        if name_of_param is None:
            def hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    # print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
                    grad[is_nan_inf] = 0.0
                sign = torch.sign(grad)
                grad_rand = torch.randint_like(grad, -10, 10)
                grad_rand[grad_rand == 0] = 1
                grad = torch.where(sign == 0, 1e-3 * grad_rand, grad)
                grad = torch.clamp_(grad, -1, 1)
                return grad
            return hook
        else:
            def named_hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
                    grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-2), grad)
                    grad = torch.clamp_(grad, -1e1, 1e1)
                    return grad
                else:
                    return None
            return named_hook
    if not args["no_autocast"] and args["backward_hook"]:
        for name, param in ddp_model.named_parameters():
            if "embeddings" in name or "sent_predict_fc" in name or "embed_proj_transpose" in name or "embed_proj" in name or "lm_head" in name or "contrastive_ffn" in name or "encoder.blocks.0" in name: #
                param.register_hook(get_hook())
            else:
                param.register_hook(get_hook())

    no_sync = args["accumulation_steps"] > 1
    iter_size = args["accumulation_steps"]

    start_time = time.time()
    for step, batch in enumerate(train_loader):
        gen_batch_time = time.time() - start_time
        batch_times.append(gen_batch_time)
        bs_size = list(batch["input_ids"].size())
        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}

        # if other_load_details is not None:
        #     if step < other_load_details["step"] and args["skip_steps"]:
        #         if (step + 1) % log_every_steps == 0 or step == 0:
        #             print("[Train]: Time = %s, Skipping step = %s, due to checkpoint with details = %s, Rank = %s" % (get_time_string(), step, other_load_details, rank))
        #         continue
        #     else:
        #         step += int(other_load_details["step"] * (other_load_details["world_size"]/args["world_size"]))
        #
        electra_loss_w = float(min(1.0, ((step + 1) / ((2 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["electra_loss_w"])
        ddp_model.module.electra_loss_w = electra_loss_w
        model.electra_loss_w = electra_loss_w

        answering_lm_w = float(min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["answering_lm_w"])
        ddp_model.module.answering_lm_w = answering_lm_w
        model.answering_lm_w = answering_lm_w

        sentence_order_prediction_w = float(min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["sentence_order_prediction_w"])
        ddp_model.module.sentence_order_prediction_w = sentence_order_prediction_w
        model.sentence_order_prediction_w = sentence_order_prediction_w

        contrastive_w = float(
            min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["contrastive_w"])
        ddp_model.module.contrastive_w = contrastive_w
        model.contrastive_w = contrastive_w

        input_cls_orthogonal_w = float(max(0.0, 1.0 - ((step + 1) / ((2 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["input_cls_orthogonal_w"])
        ddp_model.module.input_cls_orthogonal_w = input_cls_orthogonal_w
        model.input_cls_orthogonal_w = input_cls_orthogonal_w

        if (step + 1) % save_every_steps == 0:
            state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
            if local_rank == 0:
                torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
                if "checkpoint" in args and isinstance(args["checkpoint"], str) and len(args["checkpoint"].strip()) > 0:
                    save(args["checkpoint"], ddp_model, optimizer, scheduler, None, {"step": step, "samples_processed": samples_processed, "world_size": args["world_size"]})
            del state_dict
        if (step + 1) % validate_every_steps == 0:
            state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
            val_model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
            val_model.load_state_dict(state_dict)
            _ = LargeValidator(args["validation_dataset"], val_model, config, device, tokenizer, rank, args["world_size"], size_dicts, args["no_autocast"])()
            del val_model
            del state_dict
            clean_memory()
            barrier()
        record_accuracy = False
        if (step + 1) % log_every_steps == 0:
            if local_rank == 0:
                record_accuracy = True

        batch["record_accuracy"] = record_accuracy
        labels = batch["label_mlm_input_ids"] if "label_mlm_input_ids" in batch else batch["input_ids"]
        model_start_time = time.time()
        samples_processed += int(batch["input_ids"].size(0))
        samples_processed_this_log_iter += int(batch["input_ids"].size(0))
        # clean_memory()
        # print("Step = %s, Before:, for Rank = %s, input_size = %s, Allocated = %.3f, Max Allocated = %.3f, Percent = %s" %
        #       (step, rank, batch["input_ids"].size(), torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() /1e6, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()))  # torch.cuda.memory_summary()

        try:
            if no_sync and (step + 1) % iter_size != 0:
                with ddp_model.no_sync():
                    output = train_inner_loop(dict(no_autocast=args["no_autocast"], cpu=args["cpu"]), ddp_model, batch, labels, optimizer, scheduler, None, gradient_clipping, iter_size=iter_size,
                                              no_sync=True, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0)
            else:
                output = train_inner_loop(dict(no_autocast=args["no_autocast"], cpu=args["cpu"]), ddp_model, batch, labels, optimizer, scheduler, None, gradient_clipping, iter_size=iter_size,
                                          no_sync=False, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0)
                optimizer.zero_grad(set_to_none=True)

        except Exception as e:
            es = "[Train-Exception]: Time = %s, Step = %s for Rank = %s, Scale = %s, input_size = %s, lr = %s" % (
            get_time_string(), step, rank, None, bs_size, optimizer.param_groups[0]['lr'])
            print(es)
            torch.save(dict(**batch, labels=labels), os.path.join(os.getcwd(), "error-input.pth"))
            reraise(e, es)  # https://stackoverflow.com/questions/9157210/how-do-i-raise-the-same-exception-with-a-custom-message-in-python/62662138#62662138

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
                preds_dict = output["preds_dict"]
                time.sleep(random.random() + 0.1)
                wandb.log(dict(lr=optimizer.param_groups[0]['lr'], step=step, samples_processed=samples_processed, samples_per_second=samples_per_second, batch_x_sequence=np.prod(bs_size[:2]),
                               input_cls_orthogonal_w=input_cls_orthogonal_w, electra_loss_w=electra_loss_w,
                               batch_times=np.mean(batch_times), model_times=np.mean(model_times),
                               **loss_dict, **acc_dict, zero_grad=output["zero_grad"], inf_grad=output["inf_grad"]))
                print("[Train]: Time = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Loss = %s, Accuracy = %s, LR = %s" %
                      (get_time_string(), rank, step, samples_processed,
                       bs_size, loss_dict, output["accuracy_hist"], optimizer.param_groups[0]['lr']))
                print("[Train-Timings]: Time = %s, Batch time = %.4f, Model Time = %.4f, samples_per_second = %s" % (get_time_string(), np.mean(batch_times), np.mean(model_times), samples_per_second))
                if ddp_model.module.electra_loss_w > 0 and ddp_model.module.sentence_order_prediction_w > 0 and ddp_model.module.contrastive_w > 0:
                    print("[Train-Timings]: Time = %s,\n sent_order = %s,\n sent_order_nih = %s,\n mx_labels = %s,\n contrastive_labels = %s,\n electra_labels = %s" % (get_time_string(),
                                                                                                                    list(zip(batch["labels_segment_index"].view(-1).tolist(), preds_dict["sent_order_preds"])),
                                                                                                                    list(zip(preds_dict["sent_order_nih_labels"], preds_dict["sent_order_nih_preds"])),
                                                                                                                    list(zip(preds_dict["mx_labels"], preds_dict["mx_label_pred"])),
                                                                                                                    list(zip(preds_dict["contrastive_actuals"][:16] if "contrastive_actuals" in preds_dict else [], preds_dict["contrastive_preds"][:16] if "contrastive_actuals" in preds_dict else [])),
                                                                                                                    list(zip(preds_dict["electra_logits"][:16], preds_dict["electra_preds"][:16], preds_dict["electra_labels"][:16]))))
                del acc_dict
                del loss_dict

            batch_times = []
            model_times = []
            full_times = []
            samples_processed_this_log_iter = 0

            clean_memory()
            barrier()
        del batch
        del labels
        del output
        del bs_size
        start_time = time.time()

    print("Time = %s, Finished Training for Rank = %s" % (get_time_string(), rank))
    state_dict = ddp_model.state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
        if "checkpoint" in args and isinstance(args["checkpoint"], str) and len(args["checkpoint"].strip()) > 0:
            save(args["checkpoint"], ddp_model, optimizer, scheduler, None,
                 {"step": step, "samples_processed": samples_processed, "world_size": args["world_size"], "loss": loss_dict, "accuracy_dict": acc_dict})

# I've been tracking an ema of sample training loss during training and using that to guide weighted data sampling (rather than the typical uniform sampling). Seems to help with a variety of real world datasets where the bulk of the data is often very similar and easy to learn but certain subpopulations are much more challenging.


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        # start_processes(train_catch_exception, (args,), args["gpus_per_node"], True, False, start_method='spawn')
        mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)


