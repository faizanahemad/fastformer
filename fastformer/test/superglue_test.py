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
from transformers import optimization
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
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

optimizer_config = dict(eps=1e-7, beta_1=0.9, beta_2=0.98, gradient_clipping=0.1)

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


class TextDataset(Dataset):
    def __init__(self, tokenizer,
                 tokenizer_args: dict, dataset: Dataset):
        try:
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.dataset = dataset

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        item = self.dataset[item]
        label = item["label"] if "label" in item else 0.0

        text = item["text"]
        text = clean_text(text)

        results = dict()
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
        input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
        inp = dict(input_ids=input_ids, attention_mask=attention_mask)
        if "label" in item:
            results["label"] = label
        results.update(inp)

        return results

    def __len__(self):
        return len(self.dataset)


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, model):
        super().__init__()
        self.backbone = model
        if num_classes == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = CrossEntropyLoss(ignore_index=-100)

        self.num_features = (model.config.hidden_size if hasattr(model, "config") and hasattr(model.config, "hidden_size") else 768) * 1
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.num_features, num_classes))
        self.num_classes = num_classes
        init_weights(self.head)

    def get_representations(self, input_ids, attention_mask, token_type_ids=None):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = self.backbone(**inputs)["hidden_states"]
        funnel_outputs = hidden_states[-1][:, 0] # + hidden_states[-2][:, 0] + hidden_states[-3][:, 0] + hidden_states[-4][:, 0]
        return funnel_outputs

    def forward(self, input_ids, attention_mask, label=None, token_type_ids=None, **kwargs):
        with torch.set_grad_enabled(self.training):
            funnel_outputs = self.get_representations(input_ids, attention_mask, token_type_ids)
        logits = self.head(funnel_outputs)
        loss = 0.0
        if label is not None and label.min() >= 0:
            loss = self.ce(logits.squeeze(-1) if logits.ndim > 2 or self.num_classes == 1 else logits, label.float() if self.num_classes == 1 else label.long())

        logits = logits.detach()
        if self.num_classes > 1:
            # predictions = logits.argmax(-1)
            predictions = torch.softmax(logits, dim=-1)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 else predictions
        else:
            predictions = torch.sigmoid(logits)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 else predictions
        return dict(predictions=predictions, loss=loss)


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='lr')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Epochs')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight_decay')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout')

    parser.add_argument('--hpo', required=False, type=str,
                        help='hpo dict with lr, epochs, batch_size, weight_decay, iter_size')

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

    parser.add_argument('--scheduler_policy', required=False, type=str, default="olr",
                        help='scheduler_policy')

    parser.add_argument('--scheduler_warmup', required=False, type=float, default=0.1,
                        help='scheduler_warmup')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--init_method', required=False, type=str, default="tcp",
                        help='init_method')

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
    def __init__(self, tokenizer, dataset, version=1):
        self.tokenizer = tokenizer
        self.version = version
        if "wsc" in dataset:
            self.span1_text = "span1_text"
            self.span2_text = "span2_text"
            self.span1_index = "span1_index"
            self.span2_index = "span2_index"
            self.index_type = "word"
        elif "dpr" in dataset:
            self.span1_text = "noun"
            self.span2_text = "Pronoun"
            self.span1_index = "offset"
            self.span2_index = "Pronoun-offset"
            self.index_type = "char"

    def __call__(self, x):
        tokenizer = self.tokenizer
        text = x["text"]
        words = x["text"].split()
        span1_text = x[self.span1_text]
        span2_text = x[self.span2_text]
        span1_index = x[self.span1_index]
        span2_index = x[self.span2_index]
        if span1_index > span2_index:
            span1_index, span2_index = span2_index, span1_index
            span1_text, span2_text = span2_text, span1_text

        if self.index_type == "word":
            words = words[:span1_index] + ["[%s]" % (span1_text)] + words[(span1_index + len(span1_text.split())):span2_index] + [
                "[%s]" % (span2_text)] + words[span2_index + len(span2_text.split()):]
            modified_text = " ".join(words)
            modified_text_2 = words[:span2_index] + ["[%s]" % (span2_text)] + words[span2_index + len(span2_text.split()):]
            modified_text_2 = " ".join(modified_text_2)
            modified_text_3 = words[:span2_index] + [tokenizer.mask_token] + words[span2_index + len(span2_text.split()):]
            modified_text_3 = " ".join(modified_text_3)
            modified_text_4 = words[:span2_index] + [span1_text] + words[span2_index + len(span2_text.split()):]
            modified_text_4 = " ".join(modified_text_4)
        elif self.index_type == "char":
            modified_text = text[:span1_index] + "[%s]" % (span1_text) + text[(span1_index + len(span1_text)):span2_index] + \
                "[%s]" % (span2_text) + text[span2_index + len(span2_text):]
            modified_text_2 = text[:span2_index] + "[%s]" % (span2_text) + text[span2_index + len(span2_text):]
            modified_text_3 = text[:span2_index] + tokenizer.mask_token + text[span2_index + len(span2_text):]
            modified_text_4 = text[:span2_index] + span1_text + text[span2_index + len(span2_text):]
        else:
            raise ValueError("Index type = %s not supported" % self.index_type)
        clues = span1_text + (" [%s] " % span1_text) + f" {tokenizer.sep_token} " + \
               span2_text + (" [%s]" % span2_text)
        if self.version == 1:
            text = x["text"] + f" {tokenizer.sep_token} " + modified_text + f" {tokenizer.sep_token} " + clues
        elif self.version == 2:
            text = clues + f" {tokenizer.sep_token} " + modified_text + f" {tokenizer.sep_token} " + x["text"]
        elif self.version == 3:
            text = modified_text + f" {tokenizer.sep_token} " + clues + f" {tokenizer.sep_token} " + x["text"]
        elif self.version == 4:
            text = x["text"] + f" {tokenizer.sep_token} " + clues + f" {tokenizer.sep_token} " + modified_text
        elif self.version == 5:
            text = modified_text + f" {tokenizer.sep_token} " + x["text"] + f" {tokenizer.sep_token} " + clues
        elif self.version == 6:
            text = clues + f" {tokenizer.sep_token} " + x["text"] + f" {tokenizer.sep_token} " + modified_text
        elif self.version == 7:
            text = clues + f" {tokenizer.sep_token} " + modified_text
        elif self.version == 8:
            text = modified_text + f" {tokenizer.sep_token} " + clues
        elif self.version == 9:
            text = modified_text_2 + f" {tokenizer.sep_token} " + clues
        elif self.version == 10:
            text = clues + f" {tokenizer.sep_token} " + modified_text_2
        elif self.version == 11:
            text = modified_text + f" {tokenizer.sep_token} " + span1_text
        elif self.version == 12:
            text = modified_text_2 + f" {tokenizer.sep_token} " + span1_text
        elif self.version == 13:
            text = modified_text_3 + f" {tokenizer.sep_token} " + span1_text
        elif self.version == 14:
            text = modified_text_4
        elif self.version == 15:
            pass
        elif self.version == 16:
            pass

        return dict(text=text, process_version=self.version)


class SuperGlueTest:
    def __init__(self, location, model, device, tokenizer, rank, world_size, epochs, lr,
                 seed, batch_size, accumulation_steps,
                 weight_decay, dropout, scheduler_policy, scheduler_warmup, hpo=None, dataset_key=None, finetune=True):
        self.location = location
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.finetune = finetune
        self.hpo = eval(hpo) if hpo is not None else None
        self.seed = seed
        self.scheduler_warmup = scheduler_warmup
        self.scheduler_policy = scheduler_policy

        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout

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
        if isinstance(model, ClassificationModel):
            model = model.train()
            model.config.eps = 1e-7
            tokenizer = model.tokenizer
        elif isinstance(model, str):
            if "deberta" in model.lower() or "large" in model.lower():
                batch_size = batch_size // 2
                self.iter_size *= 2
            from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining, CTRLConfig, CTRLPreTrainedModel
            from transformers.models.deberta import DebertaModel
            if os.path.exists(model):
                model_name = model.split("/")[-1].split(".")[0]
                try:
                    main_model, tokenizer, _ = get_backbone(model_name, reinit=False, dropout_prob=None)
                except:
                    main_model, tokenizer, _ = get_backbone(model, reinit=False, dropout_prob=None)
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
            change_dropout(model, self.dropout)
            model = model.train()
            for p in model.parameters():
                p.requires_grad = self.finetune
            if not self.finetune:
                model = model.eval()
        elif isinstance(model, DDP):
            tokenizer = model.module.tokenizer
            model = model.module
            assert isinstance(model, ClassificationModel)
        else:
            print(type(model))
            raise ValueError
        return dict(model=model, tokenizer=tokenizer, dataloader_params=dataloader_params, batch_size=batch_size)

    def prepare_classifier(self, model_dict, dataset, device, num_classes, dataset_key, rank, reinit=False, max_epochs=None):
        set_seeds(self.seed)
        train_backbone = self.finetune
        num_workers = 2
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        dataloader_params = model_dict["dataloader_params"]
        batch_size = model_dict["batch_size"]
        max_allowed_epochs = int(self.epochs) if max_epochs is None else max_epochs

        # rnd = torch.tensor(random.randint(0, 2**32 - 1)).to(device)
        # dist.broadcast(rnd, 0)

        optc = optimizer_config
        optimizer = None
        scheduler = None
        ddp_model = model
        if reinit or not isinstance(model, (ClassificationModel, DDP)):
            classifier = ClassificationModel(num_classes, model)
            classifier.backbone = copy.deepcopy(model.backbone if hasattr(model, "backbone") else model)
            classifier = classifier.to(device)
            del model
            model = classifier
            ddp_model = DDP(model, device_ids=None if self.device == torch.device("cpu") else [self.device], find_unused_parameters=True,
                            bucket_cap_mb=10)  # find_unused_parameters=True
            # try:
            #     from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
            #     ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
            # except:
            #     print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))
        clean_memory()
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=self.lr, eps=optc["eps"], weight_decay=self.weight_decay,
                                      betas=(optc["beta_1"], optc["beta_2"]))
        optimizer.zero_grad(set_to_none=True)

        collate_fn = get_collate_fn(0, tokenizer.pad_token_id)

        train = None
        train_idx = None
        train_version = None
        if "train" in dataset:
            train = TextDataset(tokenizer,
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
                               dataset["train"])
            train = DataLoader(train, sampler=None if self.world_size == 1 else DistributedSampler(train, shuffle=True), batch_size=batch_size,
                               collate_fn=collate_fn, num_workers=num_workers, shuffle=self.world_size == 1, **dataloader_params)

            iter_size = self.iter_size
            steps_per_epoch = int(np.ceil(len(train.sampler) / (batch_size * iter_size)) if train.sampler is not None else (len(train) / iter_size))
            total_steps = int(max_allowed_epochs) * steps_per_epoch
            print("epochs = ", int(max_allowed_epochs), " steps_per_epoch=", steps_per_epoch, " lr=", self.lr, " total steps =", total_steps)
            if self.scheduler_policy == "olr":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, epochs=int(max_allowed_epochs), steps_per_epoch=steps_per_epoch, div_factor=1e2,
                                                                three_phase=False, pct_start=self.scheduler_warmup, anneal_strategy="linear")
            else:
                from transformers import get_constant_schedule_with_warmup
                scheduler = get_constant_schedule_with_warmup(optimizer, self.scheduler_warmup * total_steps)
            if "idx" in dataset["train"][0]:
                train_idx = dataset["train"]["idx"]
            else:
                train_idx = list(range(len(dataset["train"])))

            if "process_version" in dataset["train"].column_names:
                train_version = dataset["train"]["process_version"]


        validation = None
        validation_idx = None
        validation_version = None
        if "validation" in dataset:
            validation = TextDataset(tokenizer,
                                    dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
                                    dataset["validation"])
            validation = DataLoader(validation, sampler=None, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                                    shuffle=False, **dataloader_params)
            if "idx" in dataset["validation"][0]:
                validation_idx = dataset["validation"]["idx"]
            else:
                validation_idx = list(range(len(dataset["validation"])))

            if "process_version" in dataset["validation"].column_names:
                validation_version = dataset["validation"]["process_version"]


        test = None
        test_idx = None
        if rank == 0 and "test" in dataset and dataset["test"] is not None:
            test = TextDataset(tokenizer,
                              dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512),
                              dataset["test"])
            if "idx" in dataset["test"][0]:
                test_idx = [dataset["test"][i]["idx"] for i in range(len(dataset["test"]))]
            else:
                test_idx = list(range(len(dataset["test"])))
            test = DataLoader(test, sampler=None, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers,
                              shuffle=False, **dataloader_params)

        return dict(model=ddp_model, optimizer=optimizer, scheduler=scheduler, train=train, tokenizer=tokenizer,
                    validation=validation, test=test, optc=optc, test_idx=test_idx, num_classes=num_classes,
                    dataset_key=dataset_key, rank=rank, train_backbone=train_backbone,
                    validation_idx=validation_idx, train_idx=train_idx, validation_version=validation_version)

    def train_classifier(self, model, device, classifier_data, predict_only=False, max_epochs=None):
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
        max_allowed_epochs = int(self.epochs) if max_epochs is None else max_epochs

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

                train_predictions = np.array(train_predictions)
                train_predictions = (train_predictions > 0.5) if classifier_data["num_classes"] == 1 else np.argmax(train_predictions, -1)
                train_acc = accuracy_score(train_labels, train_predictions)
                all_train_acc.append(train_acc)
                model = model.eval()
                epochs += 1

            if rank == 0:
                pbar.close()

            if stored_state is not None:
                model.load_state_dict(stored_state)
                stored_state = {k.replace("module.", ""): v for k, v in stored_state.items()}
                model.module.load_state_dict(stored_state, strict=True)

            if rank == 0 and "validation" in classifier_data and classifier_data["validation"] is not None:
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
                if isinstance(classifier_data["validation_idx"][0], int):
                    if classifier_data["validation_version"] is not None:
                        individual_preds = np.array(predictions)
                        vals = pd.DataFrame(list(zip(classifier_data["validation_idx"], labels,
                                                     individual_preds > 0.5 if classifier_data["num_classes"] == 1 else np.argmax(individual_preds, -1),
                                                     classifier_data["validation_version"])), columns=["idx", "label", "prediction", "version"])
                        vals["label"] = vals["label"].astype(int)
                        vals["prediction"] = vals["prediction"].astype(int)
                        vals["correct"] = vals["label"] == vals["prediction"]
                        version_wise_correct = vals.groupby("version")[["correct"]].mean()
                        print("For %s: version_wise_correct = \n%s" % (dataset_key, version_wise_correct))
                    labels = pd.DataFrame(list(zip(classifier_data["validation_idx"], labels)), columns=["idx", "label"]).groupby("idx").head(1)["label"].values
                    if classifier_data["num_classes"] == 1:
                        predictions = pd.DataFrame(list(zip(classifier_data["validation_idx"], predictions)), columns=["idx", "predictions"]).groupby("idx")["predictions"].mean().values
                    else:
                        p = pd.DataFrame(predictions)
                        p["idx"] = classifier_data["validation_idx"]
                        predictions = p.groupby("idx").mean().values
                predictions = np.array(predictions)
                val_acc = accuracy_score(labels, (predictions > 0.5) if classifier_data["num_classes"] == 1 else np.argmax(predictions, -1))
                # val_acc = torch.tensor(val_acc).to(device)
                # tensor_list = [val_acc.new_empty(val_acc.size()) for _ in range(self.world_size)]
                # torch.distributed.all_gather(tensor_list, val_acc)
                # val_acc = torch.stack(tensor_list).mean().item()
                all_val_acc.append(val_acc)

                # Test

            torch.distributed.barrier()
        predictions = []
        if hasattr(model, "no_sync") and rank == 0 and self.hpo is None and "test" in classifier_data and classifier_data["test"] is not None:
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
        elif rank == 0 and self.hpo is None and "test" in classifier_data and classifier_data["test"] is not None:
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
        print("For %s: Train = %.4f, Val = %.4f, stored_state_val_acc = %.4f, stored_state_val_loss = %.4f" % (
        dataset_key, train_acc, val_acc, stored_state_val_acc, stored_state_val_loss))
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

        mnli_copa_rte_cb, _, _ = self.get_mnli_copa_rte_cb(tokenizer)
        scitail = self.get_scitail(tokenizer)
        cosmos_qa = self.get_cosmos_qa(tokenizer)
        hellaswag = self.get_hellaswag(tokenizer)
        swag = self.get_swag(tokenizer)
        commonsense_qa = self.get_commonsense_qa(tokenizer)
        mnli_copa_rte_cb = merge_datasets_as_df([scitail, mnli_copa_rte_cb, hellaswag, cosmos_qa, swag, commonsense_qa], ["train", "validation"], ["label", "text"])
        for split in ["train", "validation"]:
            labels = np.array(mnli_copa_rte_cb[split]["label"]).clip(0, 1).astype(int)
            labels[labels==0], labels[labels==1] = 1, 0
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].remove_columns(['label'])
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].add_column("label", labels)

        boolq = boolq.map(lambda x: dict(text=x["passage"] + f" {tokenizer.sep_token} " + x["question"]), remove_columns=['question', 'passage'])
        mnli_copa_rte_cb = merge_datasets_as_df([mnli_copa_rte_cb, boolq], ["train", "validation"], ["label", "text"])
        del mnli_copa_rte_cb["validation"]
        classifier_data = self.prepare_classifier(model_dict, mnli_copa_rte_cb, device, 1, "mnli_copa_rte_cb", rank, max_epochs=2)
        _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
        model_dict["model"] = classifier_data["model"]
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
        qa_srl = self.get_qa_srl(tokenizer)
        qqp = self.get_qqp(tokenizer)
        scitail = self.get_scitail(tokenizer)
        cosmos_qa = self.get_cosmos_qa(tokenizer)
        hellaswag = self.get_hellaswag(tokenizer)
        swag = self.get_swag(tokenizer)
        commonsense_qa = self.get_commonsense_qa(tokenizer)
        mnli_copa_rte_cb, _, _ = self.get_mnli_copa_rte_cb(tokenizer)
        mnli_copa_rte_cb = merge_datasets_as_df([scitail, mnli_copa_rte_cb, hellaswag, cosmos_qa, swag, commonsense_qa], ["train", "validation"],
                                                ["label", "text"])

        for split in ["train", "validation"]:
            labels = np.array(mnli_copa_rte_cb[split]["label"]).clip(0, 1).astype(int)
            labels[labels == 0], labels[labels == 1] = 1, 0
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].remove_columns(['label'])
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].add_column("label", labels)

        wic_reversed = wic.map(lambda x: dict(text=x["sentence2"] + f" {tokenizer.sep_token} " + x["sentence1"] + f" {tokenizer.sep_token} " + x["word"]),
                               remove_columns=['sentence1', 'sentence2', "word"])
        wic = wic.map(lambda x: dict(text=x["sentence1"] + f" {tokenizer.sep_token} " + x["sentence2"] + f" {tokenizer.sep_token} " + x["word"]),
                      remove_columns=['sentence1', 'sentence2', "word"])
        wic["train"] = concatenate_datasets((wic["train"], wic_reversed["train"]))
        mnli_copa_rte_cb = merge_datasets_as_df([mnli_copa_rte_cb, qa_srl, wic, qqp], ["train", "validation"], ["label", "text"])
        del mnli_copa_rte_cb["validation"]
        classifier_data = self.prepare_classifier(model_dict, mnli_copa_rte_cb, device, 1, "mnli_copa_rte_cb", rank, max_epochs=2)
        _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
        model_dict["model"] = classifier_data["model"]
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

    def get_mnli(self, tokenizer):
        mnli = load_dataset("multi_nli")
        mnli["validation"] = concatenate_datasets([mnli["validation_matched"], mnli["validation_mismatched"]])
        mnli_labels = np.array(mnli["train"]["label"]).astype(int)
        mnli_validation_labels = np.array(mnli["validation"]["label"]).astype(int)
        mnli_labels[mnli_labels == 1], mnli_labels[mnli_labels == 2] = 2, 1
        mnli_validation_labels[mnli_validation_labels == 1], mnli_validation_labels[mnli_validation_labels == 2] = 2, 1
        mnli["train"] = mnli["train"].remove_columns(['label']).add_column("label", mnli_labels)
        mnli["validation"] = mnli["validation"].remove_columns(['label']).add_column("label", mnli_validation_labels)
        mnli = mnli.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]),
                        remove_columns=["hypothesis", "premise", 'promptID', 'pairID', 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse',
                                        'hypothesis_parse', 'genre', ])
        mnli = mnli.filter(lambda x: len(x["text"].split()) > 32)
        return mnli

    def get_copa(self, tokenizer):
        copa = load_dataset("super_glue", "copa")
        copa_c1 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice1"],
                           label=int(not x["label"] == 0)),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa_c2 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice2"],
                           label=int(not x["label"] == 1)),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa = DatasetDict({k: concatenate_datasets([v, copa_c2[k]]) for k, v in copa_c1.items()})
        for split in ["train", "validation", "test"]:
            copa_labels = np.array(copa[split]["label"])
            copa[split] = copa[split].remove_columns(['label'])
            copa[split] = copa[split].add_column("label", copa_labels)
        return copa


    def get_qa_srl(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        qa_srl = load_dataset("qa_srl", script_version="master")
        qa_srl_pos = qa_srl.map(lambda x: dict(text=x["sentence"] + f" {tokenizer.sep_token} " + (" ".join([q for q in x["question"] if q!="_"])) + f" {tokenizer.sep_token} " + x["answers"][0], label=1),
                                remove_columns=['sentence', 'sent_id', 'predicate_idx', 'predicate', 'question', 'answers'])
        list_of_answers = [b for a in list(qa_srl["train"]["answers"]) for b in a]
        qa_srl_n1 = qa_srl.map(lambda x: dict(
            text=x["sentence"] + f" {tokenizer.sep_token} " + (" ".join([q for q in x["question"] if q != "_"])) + f" {tokenizer.sep_token} " + random.sample(list_of_answers, 1)[0],
            label=0),
                                remove_columns=['sentence', 'sent_id', 'predicate_idx', 'predicate', 'question', 'answers'])
        qa_srl_n2 = qa_srl.map(lambda x: dict(
            text=x["sentence"] + f" {tokenizer.sep_token} " + (" ".join([q for q in x["question"] if q != "_"])) + f" {tokenizer.sep_token} " +
                 random.sample(list_of_answers, 1)[0],
            label=0),
                               remove_columns=['sentence', 'sent_id', 'predicate_idx', 'predicate', 'question', 'answers'])
        qa_srl_n3 = qa_srl.map(lambda x: dict(
            text=x["sentence"] + f" {tokenizer.sep_token} " + (" ".join([q for q in x["question"] if q != "_"])) + f" {tokenizer.sep_token} " +
                 random.sample(list_of_answers, 1)[0],
            label=0),
                               remove_columns=['sentence', 'sent_id', 'predicate_idx', 'predicate', 'question', 'answers'])
        qa_srl = merge_datasets_as_df([qa_srl_pos, qa_srl_n1, qa_srl_n2, qa_srl_n3], ["train", "validation"], ["label", "text"])
        return qa_srl

    def get_qqp(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        quora = load_dataset("quora")
        quora["validation"] = Dataset.from_dict(quora["train"][0:10])
        quora_1 = quora.map(lambda x: dict(text=x["questions"]["text"][0] + f" {tokenizer.sep_token} " + x["questions"]["text"][1], label=int(x["is_duplicate"])), remove_columns=['questions', 'is_duplicate'])
        quora_2 = quora.map(
            lambda x: dict(text=x["questions"]["text"][1] + f" {tokenizer.sep_token} " + x["questions"]["text"][0], label=int(x["is_duplicate"])),
            remove_columns=['questions', 'is_duplicate'])
        quora = merge_datasets_as_df([quora_1, quora_2], ["train", "validation"], ["label", "text"])
        return quora



    def get_rte(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        rte = load_dataset("super_glue", "rte")
        rte = rte.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])

        for split in ["train", "validation", "test"]:
            rte_labels = np.array(rte[split]["label"])
            rte[split] = rte[split].remove_columns(['label'])
            rte[split] = rte[split].add_column("label", rte_labels)
        return rte

    def get_cb(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        cb = load_dataset("super_glue", "cb")
        cb1 = cb.map(lambda x: dict(text="premise: " + x["premise"] + f" {tokenizer.sep_token} " + "hypothesis: " + x["hypothesis"]),
                     remove_columns=["hypothesis", "premise"])
        cb2 = cb.map(lambda x: dict(text="hypothesis: " + x["hypothesis"] + f" {tokenizer.sep_token} " + "premise: " + x["premise"]),
                     remove_columns=["hypothesis", "premise"])
        cb3 = cb.map(lambda x: dict(text=x["hypothesis"] + f" {tokenizer.sep_token} " + x["premise"]), remove_columns=["hypothesis", "premise"])
        cb4 = cb.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        cb5 = cb.map(lambda x: dict(text="premise: " + x["premise"] + f" {tokenizer.sep_token} " + "hypothesis: " + x[
            "hypothesis"] + f" {tokenizer.sep_token} " + "Do the premise and hypothesis entail or contradict with each other?"),
                     remove_columns=["hypothesis", "premise"])
        cb6 = cb.map(lambda x: dict(text="hypothesis: " + x["hypothesis"] + f" {tokenizer.sep_token} " + "premise: " + x[
            "premise"] + f" {tokenizer.sep_token} " + "Do the premise and hypothesis agree?"),
                     remove_columns=["hypothesis", "premise"])

        for split in ["train", "validation", "test"]:
            cb1[split] = cb1[split].add_column("process_version", [1] * len(cb1[split]))
            cb2[split] = cb2[split].add_column("process_version", [2] * len(cb2[split]))
            cb3[split] = cb3[split].add_column("process_version", [3] * len(cb3[split]))
            cb4[split] = cb4[split].add_column("process_version", [4] * len(cb4[split]))
            cb5[split] = cb5[split].add_column("process_version", [5] * len(cb5[split]))
            cb6[split] = cb6[split].add_column("process_version", [6] * len(cb6[split]))

        dsets = [cb1, cb2, cb3, cb4, cb5, cb6]
        cb = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        return cb
    
    def get_mnli_copa_rte_cb(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        mnli = self.get_mnli(tokenizer)
        cb = self.get_cb(tokenizer)
        copa = self.get_copa(tokenizer)
        rte = self.get_rte(tokenizer)
        mnli_copa_rte_cb = merge_datasets_as_df([mnli, cb, copa, rte], ["train", "validation"], ["label", "text"])
        copa_rte = merge_datasets_as_df([copa, rte, cb], ["train", "validation"], ["label", "text"])
        return mnli_copa_rte_cb, copa_rte, mnli

    def cb(self, model, cb, device, dataset_key, rank):
        # MNLI || Scitail / RTE / COPA / MultiRC
        from datasets import Dataset
        import pandas as pd
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        enable_mnli = True
        enable_rte = True
        enable_copa = True
        mnli = self.get_mnli(tokenizer)
        cb = self.get_cb(tokenizer)
        copa = self.get_copa(tokenizer)
        rte = self.get_rte(tokenizer)
        mnli_copa_rte_cb, copa_rte, mnli = self.get_mnli_copa_rte_cb(tokenizer)
        if rank == 0:
            print("COPA", copa)
            print("RTE", rte)
            print("MNLI", mnli)
        if enable_rte and enable_copa and enable_mnli:
            copa_rte = {split: concatenate_datasets([copa[split], rte[split]]) for split in ["train", "validation", "test"]}
            for split in ["train", "validation", "test"]:
                copa_rte[split] = copa_rte[split].remove_columns(['idx'])
            del copa_rte["test"]
            mnli_copa_rte = dict()
            for split in ["train", "validation"]:
                d1p = copa_rte[split].to_pandas()[["label", "text"]]
                d2p = mnli[split].to_pandas()[["label", "text"]]
                mnli_copa_rte[split] = Dataset.from_pandas(pd.concat([d1p, d2p]))
            mnli_copa_rte = DatasetDict(mnli_copa_rte)

            # mnli_copa_rte = DatasetDict({split: concatenate_datasets([copa_rte[split], mnli[split]]) for split in ["train", "validation"]})
            classifier_data = self.prepare_classifier(model_dict, copa_rte, device, 3, "copa_rte", rank, max_epochs=3)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=3)
            model_dict["model"] = classifier_data["model"]
            classifier_data = self.prepare_classifier(model_dict, mnli_copa_rte, device, 3, "mnli_copa_rte", rank, max_epochs=2)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
            model_dict["model"] = classifier_data["model"]
        elif enable_rte and enable_copa:
            copa_rte = DatasetDict({split: concatenate_datasets([copa[split], rte[split]]) for split in ["train", "validation", "test"]})
            del copa_rte["test"]
            classifier_data = self.prepare_classifier(model_dict, copa_rte, device, 3, "copa_rte", rank, max_epochs=7)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=7)
            model_dict["model"] = classifier_data["model"]
        elif enable_rte:
            classifier_data = self.prepare_classifier(model_dict, rte, device, 3, "rte", rank, max_epochs=7)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=7)
            model_dict["model"] = classifier_data["model"]
            
        elif enable_copa:
            classifier_data = self.prepare_classifier(model_dict, copa, device, 3, "copa", rank, max_epochs=5)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=5)
            model_dict["model"] = classifier_data["model"]
        elif enable_mnli:
            classifier_data = self.prepare_classifier(model_dict, mnli, device, 3, "mnli", rank, max_epochs=2)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
            model_dict["model"] = classifier_data["model"]
            

        classifier_data = self.prepare_classifier(model_dict, cb, device, 3, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]

        p = pd.DataFrame(classifier_results["predictions"])
        p["idx"] = test_idx
        p = p.groupby("idx").mean().reset_index().values
        predictions = np.argmax(p[:, 1:], -1)
        test_idx = p[:, 0]

        final_predictions = [dict(idx=idx, label=self.num_to_word["cb"][pred]) for idx, pred in zip(test_idx, predictions)]
        return final_predictions, dict(dataset="cb", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def get_rte_extended(self, tokenizer):
        cb = load_dataset("super_glue", "rte")
        cb1 = cb.map(lambda x: dict(text="premise: " + x["premise"] + f" {tokenizer.sep_token} " + "hypothesis: " + x["hypothesis"]),
                     remove_columns=["hypothesis", "premise"])
        cb2 = cb.map(lambda x: dict(text="hypothesis: " + x["hypothesis"] + f" {tokenizer.sep_token} " + "premise: " + x["premise"]),
                     remove_columns=["hypothesis", "premise"])
        cb3 = cb.map(lambda x: dict(text=x["hypothesis"] + f" {tokenizer.sep_token} " + x["premise"]), remove_columns=["hypothesis", "premise"])
        cb4 = cb.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]), remove_columns=["hypothesis", "premise"])
        cb5 = cb.map(lambda x: dict(text="premise: " + x["premise"] + f" {tokenizer.sep_token} " + "hypothesis: " + x[
            "hypothesis"] + f" {tokenizer.sep_token} " + "Do the premise and hypothesis entail or contradict with each other?"),
                     remove_columns=["hypothesis", "premise"])
        cb6 = cb.map(lambda x: dict(text="hypothesis: " + x["hypothesis"] + f" {tokenizer.sep_token} " + "premise: " + x[
            "premise"] + f" {tokenizer.sep_token} " + "Do the premise and hypothesis agree?"),
                     remove_columns=["hypothesis", "premise"])

        for split in ["train", "validation", "test"]:
            cb1[split] = cb1[split].add_column("process_version", [1] * len(cb1[split]))
            cb2[split] = cb2[split].add_column("process_version", [2] * len(cb2[split]))
            cb3[split] = cb3[split].add_column("process_version", [3] * len(cb3[split]))
            cb4[split] = cb4[split].add_column("process_version", [4] * len(cb4[split]))
            cb5[split] = cb5[split].add_column("process_version", [5] * len(cb5[split]))
            cb6[split] = cb6[split].add_column("process_version", [6] * len(cb6[split]))

        dsets = [cb1, cb2, cb3, cb4, cb5, cb6]
        cb = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        return cb
    
    def get_scitail(self, tokenizer):
        scitail = load_dataset("scitail", "tsv_format").map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["hypothesis"]),
                                                            remove_columns=["hypothesis", "premise"])
        for split in ["train", "validation", "test"]:
            labels = np.array([0 if lbl == "entails" else 1 for lbl in list(scitail[split]["label"])]).astype(int)
            scitail[split] = scitail[split].remove_columns(['label'])
            scitail[split] = scitail[split].add_column("label", labels)
        return scitail


    def get_swag(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        swag = load_dataset("swag")  # .filter(lambda x: x["gold-source"]=="gold")
        swag1 = swag.map(
            lambda x: dict(text=x["sent1"] + f" {tokenizer.sep_token} " + x["sent2"] + " " + x["ending0"],
                           label=int(not x["label"] == 0)),
            remove_columns=['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3',])
        swag2 = swag.map(
            lambda x: dict(text=x["sent1"] + f" {tokenizer.sep_token} " + x["sent2"] + " " + x["ending1"],
                           label=int(not x["label"] == 1)),
            remove_columns=['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', ])
        swag3 = swag.map(
            lambda x: dict(text=x["sent1"] + f" {tokenizer.sep_token} " + x["sent2"] + " " + x["ending2"],
                           label=int(not x["label"] == 2)),
            remove_columns=['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', ])
        swag4 = swag.map(
            lambda x: dict(text=x["sent1"] + f" {tokenizer.sep_token} " + x["sent2"] + " " + x["ending3"],
                           label=int(not x["label"] == 3)),
            remove_columns=['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', ])
        hellaswag = DatasetDict({k: concatenate_datasets([v, swag2[k], swag3[k], swag4[k]]) for k, v in swag1.items()})
        for split in ["train", "validation", "test"]:
            labels = np.array(hellaswag[split]["label"]).astype(int)
            hellaswag[split] = hellaswag[split].remove_columns(['label'])
            hellaswag[split] = hellaswag[split].add_column("label", labels)
        return hellaswag


    def get_hellaswag(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        hellaswag = load_dataset("hellaswag")
        hellaswag_c1 = hellaswag.map(
            lambda x: dict(text=x["ctx_a"] + f" {tokenizer.sep_token} " + x["ctx_b"] + " " + x["endings"][0],
                           label=int(not x["label"] == 0)),
            remove_columns=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type',])
        hellaswag_c2 = hellaswag.map(
            lambda x: dict(text=x["ctx_a"] + f" {tokenizer.sep_token} " + x["ctx_b"] + " " + x["endings"][1],
                           label=int(not x["label"] == 1)),
            remove_columns=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type',])
        hellaswag_c3 = hellaswag.map(
            lambda x: dict(text=x["ctx_a"] + f" {tokenizer.sep_token} " + x["ctx_b"] + " " + x["endings"][2],
                           label=int(not x["label"] == 2)),
            remove_columns=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type',])
        hellaswag_c4 = hellaswag.map(
            lambda x: dict(text=x["ctx_a"] + f" {tokenizer.sep_token} " + x["ctx_b"] + " " + x["endings"][3],
                           label=int(not x["label"] == 3)),
            remove_columns=['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type',])
        hellaswag = DatasetDict({k: concatenate_datasets([v, hellaswag_c2[k], hellaswag_c3[k], hellaswag_c4[k]]) for k, v in hellaswag_c1.items()})
        for split in ["train", "validation", "test"]:
            labels = np.array(hellaswag[split]["label"]).astype(int)
            hellaswag[split] = hellaswag[split].remove_columns(['label'])
            hellaswag[split] = hellaswag[split].add_column("label", labels)
        return hellaswag

    def get_cosmos_qa(self, tokenizer):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        qa = load_dataset("cosmos_qa")
        qa1 = qa.map(
            lambda x: dict(text=x["context"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer0"],
                           label=int(not x["label"] == 0)),
            remove_columns=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3',])
        qa2 = qa.map(
            lambda x: dict(text=x["context"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer1"],
                           label=int(not x["label"] == 1)),
            remove_columns=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', ])
        qa3 = qa.map(
            lambda x: dict(text=x["context"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer2"],
                           label=int(not x["label"] == 2)),
            remove_columns=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', ])
        qa4 = qa.map(
            lambda x: dict(text=x["context"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["answer3"],
                           label=int(not x["label"] == 3)),
            remove_columns=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', ])
        qa = DatasetDict({k: concatenate_datasets([v, qa2[k], qa3[k], qa4[k]]) for k, v in qa1.items()})
        for split in ["train", "validation", "test"]:
            labels = np.array(qa[split]["label"]).astype(int)
            qa[split] = qa[split].remove_columns(['label'])
            qa[split] = qa[split].add_column("label", labels)
        return qa

    def get_commonsense_qa(self, tokenizer):
        commonsense_qa = load_dataset("commonsense_qa")
        commonsense_qa = commonsense_qa.map(lambda x: dict(label=(ord(x["answerKey"]) - ord('A')) if len(x["answerKey"]) > 0 else 0), remove_columns=["answerKey"])
        ca1 = commonsense_qa.map(
            lambda x: dict(text=x["question"] + f" {tokenizer.sep_token} " + x["choices"]["text"][0],
                           label=int(not x["label"] == 0)),
            remove_columns=['question', 'choices'])
        ca2 = commonsense_qa.map(
            lambda x: dict(text=x["question"] + f" {tokenizer.sep_token} " + x["choices"]["text"][1],
                           label=int(not x["label"] == 1)),
            remove_columns=['question', 'choices'])
        ca3 = commonsense_qa.map(
            lambda x: dict(text=x["question"] + f" {tokenizer.sep_token} " + x["choices"]["text"][2],
                           label=int(not x["label"] == 2)),
            remove_columns=['question', 'choices'])
        ca4 = commonsense_qa.map(
            lambda x: dict(text=x["question"] + f" {tokenizer.sep_token} " + x["choices"]["text"][3],
                           label=int(not x["label"] == 3)),
            remove_columns=['question', 'choices'])
        ca5 = commonsense_qa.map(
            lambda x: dict(text=x["question"] + f" {tokenizer.sep_token} " + x["choices"]["text"][4],
                           label=int(not x["label"] == 4)),
            remove_columns=['question', 'choices'])

        dsets = [ca1, ca2, ca3, ca4, ca5]
        cb = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        return cb

    def rte_axb_axg(self, model, rte, axb, axg, device, dataset_key, rank):
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        mnli_copa_rte_cb, _, _ = self.get_mnli_copa_rte_cb(tokenizer)
        scitail = self.get_scitail(tokenizer)
        cosmos_qa = self.get_cosmos_qa(tokenizer)
        hellaswag = self.get_hellaswag(tokenizer)
        swag = self.get_swag(tokenizer)
        commonsense_qa = self.get_commonsense_qa(tokenizer)
        mnli_copa_rte_cb = merge_datasets_as_df([scitail, mnli_copa_rte_cb, hellaswag, cosmos_qa, swag, commonsense_qa], ["train", "validation"], ["label", "text"])
        rte = self.get_rte_extended(tokenizer)
        mnli_copa_rte_cb = merge_datasets_as_df([rte, mnli_copa_rte_cb], ["train", "validation"], ["label", "text"])
        for split in ["train", "validation"]:
            labels = np.array(mnli_copa_rte_cb[split]["label"]).clip(0, 1).astype(int)
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].remove_columns(['label'])
            mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].add_column("label", labels)
        del mnli_copa_rte_cb["validation"]
        classifier_data = self.prepare_classifier(model_dict, mnli_copa_rte_cb, device, 1, "mnli_copa_rte_cb", rank, max_epochs=2)
        _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
        model_dict["model"] = classifier_data["model"]
        classifier_data = self.prepare_classifier(model_dict, rte, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None, None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"]), None, None
        test_idx = classifier_data["test_idx"]
        results = pd.DataFrame(list(zip(test_idx, classifier_results["predictions"])), columns=["id", "predictions"]).groupby("id").mean().reset_index().values
        final_predictions = [dict(idx=idx, label=self.num_to_word["rte"][int(pred > 0.5)]) for idx, pred in results]

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
        # TODO: can we include label=1 examples of multirc from train-set only into MLM data itself?
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
        # TODO: Is the option even plausible as a training task. NS are options from other examples.
        # TODO: Multi-View Choice, Question, Premise || Question, Premise, Choice || Premise, Choice (No question)
        # TODO: Given Premise and one option, select between {'cause', 'effect'} as correct question.
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]

        # copa_options = list(copa["train"]["choice1"]) + list(copa["train"]["choice2"]) + list(copa["validation"]["choice1"]) + list(copa["validation"]["choice2"]) + list(copa["test"]["choice1"]) + list(copa["test"]["choice2"])
        # copa_aux1 = copa.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["choice1"], label=1),
        #     remove_columns=["premise", 'question', "choice1", "choice2"])
        # copa_aux2 = copa.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["choice2"], label=1),
        #                      remove_columns=["premise", 'question', "choice1", "choice2"])
        # copa_ns = None
        # for i in range(10):
        #     random.seed(3413 + i)
        #     copa_ns1 = copa.map(lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + random.sample(copa_options, 1)[0], label=0),
        #                          remove_columns=["premise", 'question', "choice1", "choice2"], load_from_cache_file=False)
        #     if copa_ns is None:
        #         copa_ns = copa_ns1
        #     else:
        #         copa_ns = DatasetDict({split: concatenate_datasets([copa_ns[split], copa_ns1[split]]) for split in copa_ns.keys()})
        # copa_ns = DatasetDict({split: concatenate_datasets([copa_aux1[split], copa_ns[split], copa_aux2[split]]) for split in copa_ns.keys()}).shuffle()
        # copa_ns["train"] = concatenate_datasets([copa_ns["train"], copa_ns["validation"], copa_ns["test"]])
        # del copa_ns["test"]
        # classifier_data = self.prepare_classifier(model_dict, copa_ns, device, 1, "copa_ns", rank, max_epochs=5)
        # _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=5)
        # model_dict["model"] = classifier_data["model"]


        copa_c1 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice1"], label=x["label"] == 0,
                           choice=0),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa_c2 = copa.map(
            lambda x: dict(text=x["premise"] + f" {tokenizer.sep_token} " + x["question"] + f" {tokenizer.sep_token} " + x["choice2"], label=x["label"] == 1,
                           choice=1),
            remove_columns=["premise", 'question', "choice1", "choice2"])
        copa = DatasetDict({k: concatenate_datasets([v, copa_c2[k]]) for k, v in copa_c1.items()})

        if False:
            mnli_copa_rte_cb, copa_rte_cb, _ = self.get_mnli_copa_rte_cb(tokenizer)
            scitail = self.get_scitail(tokenizer)
            cosmos_qa = self.get_cosmos_qa(tokenizer)
            hellaswag = self.get_hellaswag(tokenizer)
            swag = self.get_swag(tokenizer)
            commonsense_qa = self.get_commonsense_qa(tokenizer)
            mnli_copa_rte_cb = merge_datasets_as_df([scitail, copa_rte_cb, hellaswag, cosmos_qa, swag, commonsense_qa], ["train", "validation"], ["label", "text"])
            for split in ["train", "validation"]:
                labels = np.array(mnli_copa_rte_cb[split]["label"]).clip(0, 1).astype(int)
                labels[labels == 0], labels[labels == 1] = 1, 0
                mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].remove_columns(['label'])
                mnli_copa_rte_cb[split] = mnli_copa_rte_cb[split].add_column("label", labels)


            del mnli_copa_rte_cb["validation"]
            swag_hellaswag = merge_datasets_as_df([hellaswag, swag, copa], ["train", "validation"], ["label", "text"])
            classifier_data = self.prepare_classifier(model_dict, swag_hellaswag, device, 1, "swag_hellaswag", rank, max_epochs=2)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
            model_dict["model"] = classifier_data["model"]
            classifier_data = self.prepare_classifier(model_dict, mnli_copa_rte_cb, device, 1, "mnli_copa_rte_cb", rank, max_epochs=3)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=3)
            model_dict["model"] = classifier_data["model"]
            classifier_data = self.prepare_classifier(model_dict, swag, device, 1, "swag", rank, max_epochs=2)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=2)
            model_dict["model"] = classifier_data["model"]
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
        # TODO: test gap before DPR, gap after DPR, gap with DPR
        # TODO: Test wsc with both [Yes]
        # TODO: test using test-set of both [No]
        # TODO: MLM tuning of model before classification training on only true labels of train set.
        # TODO: Use val + train for 2 epochs in end before test predictions?

        # TODO: Pretrain with large MLM set.
        from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
        from collections import defaultdict
        model_name = model
        model_dict = self.build_model(model)
        tokenizer = model_dict["tokenizer"]
        caching = False
        versions = [2, 3, 4, 5, 6, 7, 8, 10]
        enable_wiki_dpr = True
        enable_wsc = True
        enable_dpr = True
        if enable_wiki_dpr:
            dsets = [wsc.map(wsc_proc(tokenizer, "wsc", i), remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"],
                             load_from_cache_file=caching) for i in [13, 14]]
            wsc_wiki = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
            for split in ["train", "validation", "test"]:
                wsc_wiki[split] = wsc_wiki[split].remove_columns(['process_version'])
                wsc_wiki[split] = wsc_wiki[split].remove_columns(['idx'])
                labels = np.array(wsc_wiki[split]['label'])
                wsc_wiki[split] = wsc_wiki[split].remove_columns(['label'])
                wsc_wiki[split] = wsc_wiki[split].add_column("label", labels)
            del wsc_wiki["test"]
            del wsc_wiki["validation"]
            wiki_dpr = build_wiki_dpr(tokenizer)
            del wiki_dpr["test"]
            del wiki_dpr["validation"]
            if rank == 0:
                print(wiki_dpr["train"].features, "\n", wsc["train"].features)
            wiki_dpr["train"] = concatenate_datasets([wsc_wiki["train"], wiki_dpr["train"], wsc_wiki["train"]])
            classifier_data = self.prepare_classifier(model_dict, wiki_dpr, device, 1, "wiki_dpr", rank, max_epochs=1)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=1)
            model_dict["model"] = classifier_data["model"]

        dsets = [wsc.map(wsc_proc(tokenizer, "wsc", i), remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"],
                         load_from_cache_file=caching) for i in versions]
        wsc = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        if rank == 0:
            print(wsc["validation"])
        wsc = wsc.map(lambda x: dict(label=int(x["label"])), load_from_cache_file=caching)
        for split in ["train", "validation", "test"]:
            labels = np.array(wsc[split]['label'])
            wsc[split] = wsc[split].remove_columns(['label'])
            wsc[split] = wsc[split].add_column("label", labels)

            idx = np.array(wsc[split]['idx'])
            wsc[split] = wsc[split].remove_columns(['idx'])
            wsc[split] = wsc[split].add_column("idx", idx)

            process_version = np.array(wsc[split]['process_version'])
            wsc[split] = wsc[split].remove_columns(['process_version'])
            wsc[split] = wsc[split].add_column("process_version", process_version)

        if enable_wsc:
            classifier_data = self.prepare_classifier(model_dict, wsc, device, 1, dataset_key, rank, max_epochs=3)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=3)
            model_dict["model"] = classifier_data["model"]

        dpr = load_dataset('csv', data_files={'train': "dpr/winograd_train.csv", "validation": "dpr/winograd_dev.csv", 'test': "dpr/winograd_test.csv"})
        dprA = dpr.remove_columns(['B', 'B-offset', 'B-coref']).rename_column("Text", "text").rename_column("A", "noun").rename_column('A-coref', "label").rename_column('A-offset', "offset")
        dprB = dpr.remove_columns(['A', 'A-offset', 'A-coref']).rename_column("Text", "text").rename_column("B", "noun").rename_column('B-coref', "label").rename_column('B-offset', "offset")
        dpr = DatasetDict({split: concatenate_datasets([d[split] for d in [dprA, dprB]]) for split in ["train", "validation", "test"]})
        # dpr["train"] = concatenate_datasets([dpr["train"], dpr["test"]])
        # dpr["validation"] = dpr["test"]
        dpr["train"] = dpr["train"].add_column("idx", list(range(len(dpr["train"]))))
        dpr["validation"] = dpr["validation"].add_column("idx", list(range(len(dpr["validation"]))))
        dpr["test"] = dpr["test"].add_column("idx", list(range(len(dpr["test"]))))
        dpr = dpr.map(lambda x: dict(label=int(x["label"])), load_from_cache_file=caching)


        dsets = [dpr.map(wsc_proc(tokenizer, "dpr", i), remove_columns=['Pronoun', 'Pronoun-offset', 'noun', 'offset'],
                         load_from_cache_file=caching) for i in versions]
        dpr = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})

        dpr = DatasetDict({split: concatenate_datasets([dpr[split], wsc[split].map(lambda x: dict(idx=x["idx"]+len(dpr[split])), load_from_cache_file=caching)]) for split in ["train", "validation", "test"]})
        dpr1 = dpr

        #
        gap = load_dataset("gap")
        gapA = gap.remove_columns(['B', 'B-offset', 'B-coref']).rename_column("Text", "text").rename_column("A", "noun").rename_column('A-coref',"label").rename_column('A-offset', "offset")
        gapB = gap.remove_columns(['A', 'A-offset', 'A-coref']).rename_column("Text", "text").rename_column("B", "noun").rename_column('B-coref', "label").rename_column('B-offset', "offset")
        gap = DatasetDict({split: concatenate_datasets([d[split] for d in [gapA, gapB]]) for split in ["train", "validation", "test"]})

        dpr=gap

        # dpr["train"] = concatenate_datasets([dpr["train"], dpr["test"]])
        dpr["train"] = dpr["train"].add_column("idx", list(range(len(dpr["train"]))))
        dpr["validation"] = dpr["validation"].add_column("idx", list(range(len(dpr["validation"]))))
        dpr["test"] = dpr["test"].add_column("idx", list(range(len(dpr["test"]))))
        dpr = dpr.map(lambda x: dict(label=int(x["label"])))

        for split in ["train", "validation", "test"]:
            labels = np.array(dpr[split]['label'])
            dpr[split] = dpr[split].remove_columns(['label'])
            dpr[split] = dpr[split].add_column("label", labels)

            idx = np.array(dpr[split]['idx'])
            dpr[split] = dpr[split].remove_columns(['idx'])
            dpr[split] = dpr[split].add_column("idx", idx)

        dsets = [dpr.map(wsc_proc(tokenizer, "dpr", i), remove_columns=['Pronoun', 'Pronoun-offset', 'noun', 'offset', 'URL', 'ID'],
                         load_from_cache_file=caching) for i in versions]
        dpr = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        if rank == 0:
            print(dpr["train"].features, "\n", wsc["train"].features)
            print("#" * 80)
            print(dpr["validation"].features, "\n", wsc["validation"].features)
            print("#" * 80)
            print(dpr["test"].features, "\n", wsc["test"].features)
            print("#" * 80)
        
        # dpr = DatasetDict({split: concatenate_datasets([dpr[split], wsc[split].map(lambda x: dict(idx=x["idx"] + len(dpr[split])), load_from_cache_file=caching)]) for split in ["train", "validation", "test"]})
        dpr = DatasetDict(
            {split: concatenate_datasets([dpr[split], dpr1[split]]) for split
             in ["train", "validation", "test"]})
        del dpr["test"]
        if enable_dpr:
            classifier_data = self.prepare_classifier(model_dict, dpr, device, 1, "gap", rank, max_epochs=6)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=6)
            model_dict["model"] = classifier_data["model"]

            self.seed = self.seed + 17
            classifier_data = self.prepare_classifier(model_dict, wsc, device, 1, dataset_key, rank, max_epochs=1)
            _ = self.train_classifier(classifier_data["model"], device, classifier_data, max_epochs=1)
            model_dict["model"] = classifier_data["model"]

        #
        # dsets = [wsc.map(wsc_proc(tokenizer, "wsc", i), remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"]) for i in range(1, 7)]
        # wsc = DatasetDict({split: concatenate_datasets([d[split] for d in dsets]) for split in ["train", "validation", "test"]})
        self.seed = self.seed + 343
        classifier_data = self.prepare_classifier(model_dict, wsc, device, 1, dataset_key, rank)
        classifier_results = self.train_classifier(classifier_data["model"], device, classifier_data)
        if rank != 0:
            return None, None
        elif self.hpo is not None:
            return None, dict(train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"], epochs=classifier_results["epochs"],
                              val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                              val_loss=classifier_results["val_loss"])
        test_idx = classifier_data["test_idx"]
        results = pd.DataFrame(list(zip(test_idx, classifier_results["predictions"])), columns=["id", "predictions"]).groupby("id").mean().reset_index().values
        final_predictions = [dict(idx=idx, label=self.num_to_word["boolq"][int(pred > 0.5)]) for idx, pred in results]
        return final_predictions, dict(dataset="wsc.fixed", train_acc=classifier_results["train_acc"], val_acc=classifier_results["val_acc"],
                                       epochs=classifier_results["epochs"],
                                       val_loss_hist=classifier_results["all_val_loss"][-3:], broken=classifier_results["broken"],
                                       model=getattr(classifier_data["model"], "module", classifier_data["model"]).backbone)

    def __call__(self, generate_test_predictions=True):
        from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
        tokenizer = self.tokenizer
        model = self.model.to(self.device).eval() if not isinstance(self.model, str) else self.model
        pred_datas = []

        # glue = dict()
        # for gl in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:
        #     glue[gl] = load_dataset("glue", gl)

        super_glue = dict()
        for gl in ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed', 'axb', 'axg']:  # 'wsc',
            super_glue[gl] = load_dataset("super_glue", gl)
        keys = ['cb', 'copa', 'multirc', 'record', 'wsc.fixed', 'rte', 'boolq', 'wic', ]  # 'axb', 'axg'
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


def train(local_rank, args):
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ["NCCL_DEBUG"] = "WARN"
    gpu_device = local_rank
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import warnings
    warnings.simplefilter("ignore")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    torch.backends.cudnn.benchmark = True
    rank = args["nr"] if args["cpu"] else (args["nr"] * args["gpus_per_node"] + local_rank)
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
        init_method = "tcp://%s:%s" % (args["master_addr"], args["master_port"])
    elif args["init_method"] == "file":
        init_method = 'file://%s/%s' % (args["master_addr"], args["master_port"])
    else:
        raise ValueError
    print("[Train]: Time = %s, Initializing Dist Process with init-method = %s for Rank = %s" % (get_time_string(), init_method, rank))
    dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
    print("[Train]: Time = %s, Initialized Dist Process for Rank = %s" % (get_time_string(), rank))
    set_seeds(args["seed"])

    model = args["pretrained_model"]
    _, tokenizer, _ = get_backbone(args["pretrained_model"])
    SuperGlueTest(None, model, device, tokenizer, rank, args["world_size"], args["epochs"], args["lr"],
                  args["seed"], args["batch_size"], args["accumulation_steps"],
                  args["weight_decay"], args["dropout"], args["scheduler_policy"], args["scheduler_warmup"],
                  args["hpo"], args["dataset_key"])()
    return


def build_wiki_dpr(tokenizer):
    from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
    from collections import defaultdict
    if os.path.exists("wiki_dpr"):
        wiki_dpr = DatasetDict.load_from_disk("wiki_dpr")
        return wiki_dpr
    if os.path.exists("MaskedWiki_sample.txt"):
        with open("MaskedWiki_sample.txt") as f:
            lines = f.readlines()

        examples = defaultdict(list)
        for i in range(0, len(lines), 5):
            cur_ex = lines[i:i + 5]

            cur_ex = list(map(lambda x: x.replace("\n", ""), cur_ex))
            options = cur_ex[2].split(",")
            correct_option = cur_ex[3]
            text = cur_ex[0].replace("[MASK]", tokenizer.mask_token)
            for opt in options:
                ctext = text + f" {tokenizer.sep_token} " + opt
                label = int(opt == correct_option)
                examples["text"].append(ctext)
                examples["label"].append(label)

                ctext2 = text.replace(tokenizer.mask_token, opt)
                examples["text"].append(ctext2)
                examples["label"].append(label)
        wiki_dpr = Dataset.from_dict(examples).train_test_split(test_size=0.1)
        split2 = wiki_dpr["test"].train_test_split(test_size=0.5)
        wiki_dpr["validation"] = split2["train"]
        wiki_dpr["test"] = split2["test"]
        wiki_dpr = wiki_dpr.filter(lambda x: len(x["text"].split()) > 40)
        wiki_dpr.save_to_disk("wiki_dpr")
        return wiki_dpr
    raise ValueError




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train(0, args)
    else:
        mp.spawn(train, nprocs=args["gpus_per_node"], args=(args,), join=True)
        # start_processes(train, (args,), args["gpus_per_node"], True, False, start_method='spawn')







