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

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')
    parser.add_argument('--batch_size', required=False, type=int,
                        help='Batch Size')
    parser.add_argument('--seed', required=False, type=int, default=3123,
                        help='seed')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--finetune', action="store_true", default=False,
                        help='finetune')

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
    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    return vars(args)


class FastFormerForClassification(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, num_classes, model, tokenizer=None,
                 additive_margin_softmax_w=0.3, reinit_backbone=False, train_backbone=True):
        if isinstance(config, FastFormerConfig):
            super().__init__(config)
        elif model is not None and hasattr(model, "config"):
            super().__init__(model.config)
        else:
            raise ValueError

        self.backbone = model
        
        if num_classes == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = CrossEntropyLoss(ignore_index=-100)
        
        self.num_features = config.block_channel_size[-1] if isinstance(config, FastFormerConfig) else (model.config.hidden_size if hasattr(model, "config") and hasattr(model.config, "hidden_size") else 768) * 4
        self.head = nn.Linear(self.num_features, num_classes)
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.train_backbone = train_backbone
        if reinit_backbone:
            self.init_weights()

        init_weights(self.head)

    def get_representations(self, input_ids, attention_mask, char_ids=None, char_offsets=None, label=None, token_type_ids=None):

        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = self.backbone(**inputs)["hidden_states"]
        funnel_outputs = torch.cat((hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), -1)
        return funnel_outputs

    def forward(self, input_ids, attention_mask, char_ids=None, char_offsets=None, label=None, token_type_ids=None, **kwargs):
        with torch.set_grad_enabled(self.train_backbone and self.training):
            funnel_outputs = self.get_representations(input_ids, attention_mask, char_ids, char_offsets, label, token_type_ids)
        if not self.train_backbone:
            funnel_outputs = funnel_outputs.detach()
        logits = self.head(funnel_outputs)
        loss = 0.0
        if label is not None and label.min() >= 0:
            loss = self.ce(logits.squeeze(-1) if logits.ndim > 2 or self.num_classes == 1 else logits, label.float() if self.num_classes == 1 else label.long())

        logits = logits.detach()
        if self.num_classes > 1:
            predictions = logits.argmax(-1)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 or self.num_classes == 1 else predictions
        else:
            predictions = torch.sigmoid(logits)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 or self.num_classes == 1 else predictions
        return dict(predictions=predictions, loss=loss)


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
        text = x["text"] + f" {tokenizer.sep_token} " + modified_text + f" {tokenizer.sep_token} " + words[x["span1_index"]] + f" {tokenizer.sep_token} " + \
               words[x["span2_index"]]
        return dict(text=text)


class SuperGlueTest:
    def __init__(self, location, model, device, rank, world_size, epochs, lr,
                 seed, batch_size, accumulation_steps,
                 weight_decay, hpo=None, dataset_key=None, finetune=True):
        self.location = location
        self.model = model

        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.finetune = finetune
        self.hpo = eval(hpo) if hpo is not None else None
        self.seed = seed

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
                                  rte=dict(agree="entailment", entailment="entailment", entail="entailment", contradiction="not_entailment",
                                           contradict="not_entailment", disagree="not_entailment", neutral="not_entailment"))
        self.task_word_map["wic"] = self.task_word_map["boolq"]
        self.task_word_map["axg"] = self.task_word_map["rte"]
        self.task_word_map["axb"] = self.task_word_map["rte"]
        self.task_word_map["wsc.fixed"] = self.task_word_map["boolq"]
        # self.epoch_per_dataset = {"boolq": 35, 'cb': 150, 'copa': 200, 'multirc': 19, 'record': 2, 'rte': 200, 'wic': 100, 'wsc.fixed': 200}
        self.epochs = epochs
        # self.lr_per_dataset = {"boolq": lr, 'cb': lr, 'copa': lr, 'multirc': lr, 'record': lr, 'rte': lr, 'wic': lr, 'wsc.fixed': lr}

        self.num_to_word = dict(boolq={0: "false", 1: "true"}, cb={0: "entailment", 1: "contradiction", 2: "neutral"},
                                rte={0: "entailment", 1: "not_entailment"})

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
            from transformers import AutoTokenizer, AutoModel
            if os.path.exists(model):
                model_name = model.split("/")[-1].split(".")[0]
                try:
                    main_model, tokenizer = get_mtt_backbone(model_name, 1, self.enable_layer_normalizers, None, reinit=False,
                                                             train_layer_normalizers=False, enable_layer_normalizers_statistics=self.enable_layer_normalizers,
                                                             dropout_prob=0.1)
                except:
                    main_model, tokenizer = get_mtt_backbone(model, 1, self.enable_layer_normalizers, None, reinit=False,
                                                             train_layer_normalizers=False, enable_layer_normalizers_statistics=self.enable_layer_normalizers,
                                                             dropout_prob=0.1)
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
        print("batch_size = %s, iter size = %s" % (batch_size, self.iter_size))
        print(model)
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
            classifier = FastFormerForClassification(model.config if hasattr(model, "config") else None, num_classes, model, tokenizer,
                                                     train_backbone=train_backbone)
            classifier.backbone = copy.deepcopy(model.backbone if hasattr(model, "backbone") else model)
            classifier = classifier.to(device)
            del model
            model = classifier
            ddp_model = DDP(model, device_ids=None if self.device == torch.device("cpu") else [self.device], find_unused_parameters=True,
                            bucket_cap_mb=10)  # find_unused_parameters=True
            try:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
            except:
                print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))
            clean_memory()
            optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=self.lr, eps=optc["eps"], weight_decay=self.weight_decay,
                                          betas=(optc["beta_1"], optc["beta_2"]))
            optimizer.zero_grad(set_to_none=True)

        collate_fn = get_collate_fn(0, tokenizer.pad_token_id)

        train = None
        if "train" in dataset:
            train = MTTDataset(1, len(tokenizer), tokenizer,
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
                               dataset["train"])
            train.training = False
            train = DataLoader(train, sampler=None if self.world_size == 1 else DistributedSampler(train, shuffle=True), batch_size=batch_size,
                               collate_fn=collate_fn, num_workers=num_workers, shuffle=self.world_size == 1, **dataloader_params)

            iter_size = self.iter_size
            steps_per_epoch = int(np.ceil(len(train.sampler) / (batch_size * iter_size)) if train.sampler is not None else (len(train) / iter_size))
            print("epochs = ", int(self.epochs), " steps_per_epoch=", steps_per_epoch, " lr=", self.lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, epochs=int(self.epochs), steps_per_epoch=steps_per_epoch, div_factor=1e2,
                                                            three_phase=False, pct_start=0.1, anneal_strategy="linear")

        validation = None
        if "validation" in dataset:
            validation = MTTDataset(1, len(tokenizer), tokenizer,
                                    dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
                                    dataset["validation"])
            validation.training = False
            validation = DataLoader(validation, sampler=None, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                                    shuffle=False, **dataloader_params)

        test = None
        test_idx = None
        if rank == 0:
            test = MTTDataset(1, len(tokenizer), tokenizer,
                              dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
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
        max_allowed_epochs = int(self.epochs)

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
                model = model.eval()

                epochs += 1

            torch.distributed.barrier()
            if rank == 0:
                pbar.close()

            if rank == 0:
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
            model = model.eval()
            inner_model = model
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
        print("For %s: Train = %.4f, Val = %.4f, stored_state_val_acc = %.4f, stored_state_val_loss = %.4f" % (
        dataset_key, train_acc, val_acc, stored_state_val_acc, stored_state_val_loss))
        print("For %s: all_val_loss = %s, all_val_accuracy = %s" % (dataset_key, all_val_loss, all_val_acc))
        return dict(val_acc=val_acc, train_acc=train_acc, predictions=predictions, all_val_loss=all_val_loss, all_val_acc=all_val_acc,
                    all_train_acc=all_train_acc, epochs=epochs, val_loss=val_loss, broken=False)

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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"],
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
        swag = DatasetDict({k: concatenate_datasets([v, swag_c2[k], swag_c3[k], swag_c4[k], ]) for k, v in swag_c1.items()})
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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        # print(classifier_results["predictions"])
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="boolq", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def wic(self, model, wic, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        wic_reversed = wic.map(lambda x: dict(text=x["sentence2"] + f" {tokenizer.sep_token} " + x["sentence1"] + f" {tokenizer.sep_token} " + x["word"]),
                               remove_columns=['sentence1', 'sentence2', "word"])
        wic = wic.map(lambda x: dict(text=x["sentence1"] + f" {tokenizer.sep_token} " + x["sentence2"] + f" {tokenizer.sep_token} " + x["word"]),
                      remove_columns=['sentence1', 'sentence2', "word"])
        wic["train"] = concatenate_datasets((wic["train"], wic_reversed["train"]))
        classifier_data = self.prepare_classifier(model_dict, wic, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="wic", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"], val_loss_hist=classifier_results["all_val_loss"][-3:],
                                       broken=classifier_results["broken"],
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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["cb"][pred]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="cb", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"]), None, None
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
        final_predictions_axb = [dict(idx=idx, label=self.num_to_word["rte"][int(pred > 0.5)]) for idx, pred in
                                 zip(test_idx, classifier_results["predictions"])]

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
        multirc = multirc.map(lambda x: dict(text=x["paragraph"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer"]),
                              remove_columns=["paragraph", "question", "answer"])
        classifier_data = self.prepare_classifier(model_dict, multirc, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=pred > 0.5) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        mrcp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for pd in final_predictions:
            mrcp[pd["idx"]["paragraph"]][pd["idx"]["question"]][pd["idx"]["answer"]] = pd["label"]
        mrcp = [
            {"idx": k, "passage": {"questions": [{"idx": m, "answers": [{"idx": o, "label": p} for o, p in sorted(n.items())]} for m, n in sorted(v.items())]}}
            for k, v in sorted(mrcp.items())]
        final_predictions = mrcp
        return final_predictions, dict(dataset="multirc", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def copa(self, model, copa, device, dataset_key, rank):
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        copa_c1 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice1"], label=x["label"] == 0,
                           choice=0),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa_c2 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice2"], label=x["label"] == 1,
                           choice=1),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa = DatasetDict({k: concatenate_datasets([v, copa_c2[k]]) for k, v in copa_c1.items()})

        classifier_data = self.prepare_classifier(model_dict, copa, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        choices = [copa["test"][i]["choice"] for i in range(len(copa["test"]))]
        final_predictions = [dict(idx=idx, label=pred, choice=ch) for idx, pred, ch in zip(test_idx, classifier_results["predictions"], choices)]
        final_predictions = pd.DataFrame.from_records(final_predictions).groupby("idx", group_keys=False).apply(
            lambda x: x[x.label >= x.label.max()][["idx", "choice"]].rename(columns={"choice": "label"})).to_dict('records')
        return final_predictions, dict(dataset="copa", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
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
        return final_predictions, dict(dataset="record", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
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
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in zip(test_idx, classifier_results["predictions"])]
        return final_predictions, dict(dataset="wsc.fixed", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def __call__(self, generate_test_predictions=True):
        model = self.model.to(self.device).eval() if not isinstance(self.model, str) else self.model
        pred_datas = []
        print("[SUPERGLUE]: call to superglue class")

        # glue = dict()
        # for gl in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:
        #     glue[gl] = load_dataset("glue", gl)

        super_glue, _ = superglue_test(test_only=False, pet_dataset=False)
        keys = ['cb', 'copa', 'multirc', 'record', 'wsc.fixed', 'rte', 'boolq', 'wic', ]  # 'axb', 'axg'

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
            print("[SUPERGLUE]: Time = %s, Train for Rank = %s/%s, dataset = %s, device = %s, idx = %s" % (
            get_time_string(), self.rank, self.world_size, dk, self.device, idx))
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
                final_predictions, pred_data, final_predictions_axb, final_predictions_axg = self.rte_axb_axg(model, dataset, super_glue["axb"],
                                                                                                              super_glue["axg"], self.device, dk, self.rank)
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
                    torch.save(pred_data.pop("model").state_dict(), model + ("" if model.endswith("." + dk) else ("." + dk)))
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

def cleanup():
    dist.destroy_process_group()


def train_test(local_rank, args):
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
    print("[Train]: Time = %s, ------------------ Prepare to init Dist Process for Rank = %s" % (get_time_string(), rank))
    if args["nr"] == 0:
        args["master_addr"] = "0.0.0.0"
    init_method = "tcp://%s:%s" % (args["master_addr"], args["master_port"])

    print("[Train]: Time = %s, ---------- Initializing Dist Process with init-method = %s for Rank = %s" % (get_time_string(), init_method, rank))
    dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
    set_seeds(args["seed"])


    model = args["pretrained_model"]
    print("[Train]: Time = %s, Inside if call, Superglue call" % (get_time_string()))
    SuperGlueTest(None, model, device, rank, args["world_size"], args["epochs"], args["lr"],
                  args["seed"], args["batch_size"], args["accumulation_steps"], args["weight_decay"],
                args["hpo"], args["dataset_key"], args["finetune"])()
    return

# I've been tracking an ema of sample training loss during training and using that to guide weighted data sampling (rather than the typical uniform sampling).
# Seems to help with a variety of real world datasets where the bulk of the data is often very similar and easy to learn but certain subpopulations are much more challenging.


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    print("[TRAIN]: superglue copa train start")
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_test(0, args)
    else:
        # start_processes(train_catch_exception, (args,), args["gpus_per_node"], True, False, start_method='spawn')
        mp.spawn(train_test, nprocs=args["gpus_per_node"], args=(args,), join=True)

