import copy
import sys
import traceback

import numpy as np
import torch
from more_itertools import windowed
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
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from fastformer.data import *
from fastformer.config import *
from fastformer.optimizers import Novograd, RangerLars
from fastformer.data.dataset import datadict_iterator, MTTDataset, get_simple_collate_fn, get_next
from fastformer.utils import *
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from collections import defaultdict
from transformers import optimization
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import wandb
from pytz import timezone
from datetime import datetime, timedelta
from torch.utils.data.dataloader import DataLoader
import multiprocessing
import signal
from torch.multiprocessing.spawn import _prctl_pr_set_pdeathsig
import nltk

from tabulate import tabulate
from torch.multiprocessing import Process, ProcessContext

try:
    from torch.cuda.amp import GradScaler, autocast
except:
    pass

optimizer_config = OptimizerConfig(5e-5, 1e-4, 1e-4, 0.9, 0.98,
                                   8, 8, 0.1)


def get_valid_sentences(text, sent_detector, tokenizer, required_length_min, required_length_max):
    text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', text)
    sents = sent_detector.tokenize(text)
    tokenizer_args = dict(padding="none", truncation=True, return_tensors="pt", max_length=512)
    sent_lengths = [tokenizer(s, return_offsets_mapping=False, **tokenizer_args)["attention_mask"].squeeze().sum() for s in sents]
    valid_pairs = []
    valid_lengths = []
    sents_n_lengths = list(zip(sents, sent_lengths))
    for wlen in range(1, len(sents_n_lengths)):
        for ws in windowed(sents_n_lengths, wlen):
            current_sents, current_lengths = zip(*ws)
            tl = sum(current_lengths)
            if tl >= required_length_min and tl < required_length_max - 2:
                valid_pairs.append(" ".join(current_sents))
                valid_lengths.append(tl)
    if len(valid_pairs) > 0:
        text = random.choices(valid_pairs, np.array(valid_lengths)/sum(valid_lengths), k=1)[0]
    else:
        raise ValueError

    return text


class sample_random_token:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        probas = np.ones(len(tokenizer))
        for i in tokenizer.all_special_ids:
            probas[i] = 0
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
            if len(token) <= 3:
                probas[i] = 0
        probas = probas / np.sum(probas)
        self.probas = probas
        self.length = len(tokenizer)

    def __call__(self):
        t_id = random.choices(range(self.length), self.probas)[0]
        return t_id


def segment(text, n_segments, sent_detector, pad_token):
    text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', text)
    sents = sent_detector.tokenize(text)
    sent_wc = list(map(lambda x: len(x.split()), sents))
    twc = len(text.split())
    segments = defaultdict(str)
    tol = 0.1
    while len(segments) < n_segments and tol <= (n_segments/2):
        segments = defaultdict(str)
        expected_wc = max(twc // (n_segments + tol), 16)  # Each segment is atleast 16 words
        tol += 0.2
        cwc = 0
        sidx = 0
        for s, wc in zip(sents, sent_wc):
            segments[sidx] = (segments[sidx] + " " + s).strip()
            cwc += wc
            if cwc >= expected_wc and sidx < n_segments - 1:
                cwc = 0
                sidx += 1

    return list(segments.values()) + [pad_token] * (n_segments - len(segments))


def token_id_masking(tokens, tokenizer, probability: float, sampler=None) -> str:

    if probability == 0 or len(tokens) <= 2:
        return tokens

    tokens = np.array(tokens.tolist())
    original_tokens = tokens.copy()
    special_tokens_idx = np.in1d(original_tokens, tokenizer.all_special_ids)
    full_length = np.logical_not(special_tokens_idx).sum()
    probas = np.random.random(len(tokens))
    masked = probas <= probability
    rand_replace = probas < (probability * 0.15)
    tokens[masked] = tokenizer.mask_token_id
    if sampler is not None:
        rand_tokens = np.array([sampler() for _ in range(np.sum(rand_replace))])
    else:
        rand_tokens = np.array([random.sample(range(len(tokenizer)), 1)[0] for _ in range(np.sum(rand_replace))])
    tokens[rand_replace] = rand_tokens

    if full_length > 64:
        for i, t in enumerate(tokens):
            if t == tokenizer.mask_token_id and i > 1 and random.random() <= 0.125:
                tokens[i - 1] = tokenizer.mask_token_id
            if t == tokenizer.mask_token_id and i < len(tokens) - 1 and random.random() <= 0.125:
                tokens[i + 1] = tokenizer.mask_token_id


    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    return torch.tensor(list(tokens))


class MaskedLanguageSentenceOrderModelDataset(Dataset):
    def __init__(self, tokenizer, tokenizer_args: dict, dataset: Dataset, word_mask_proba=0.15):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        try:
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.dataset = dataset
        self.word_mask_proba = word_mask_proba
        self.vocab = list(tokenizer.get_vocab())
        self.allowed_raw_length = self.tokenizer_args["max_length"] - (self.tokenizer_args["max_length"] // 8)
        self.token_sampler = sample_random_token(self.tokenizer)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        item = self.dataset[item]

        text = item["text"]
        text = clean_text(text)
        length = len(text.strip().split())

        if length > self.allowed_raw_length:
            try:
                text = get_valid_sentences(text, self.sent_detector, tokenizer, int(self.tokenizer_args["max_length"] * 0.8), self.tokenizer_args["max_length"])
            except:
                text = " ".join(text.split()[:self.allowed_raw_length])
            length = len(text.strip().split())

        seg_sep_token = f" {tokenizer.sep_token} "
        num_segments = 2
        segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
        num_segments = sum(segments != tokenizer.pad_token)

        label_sentence_order = 0
        if num_segments > 1:
            if random.random() < 0.5:
                label_sentence_order = 0
            else:
                label_sentence_order = 1
                segments[0], segments[1] = segments[1], segments[0]

        results = dict(label_sentence_order=label_sentence_order)
        text = seg_sep_token.join(segments)  # Training Labels for MLM
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
        input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
        results["label_mlm_input_ids"] = input_ids.squeeze()

        input_ids = token_id_masking(results["label_mlm_input_ids"], self.tokenizer, self.word_mask_proba, sampler=self.token_sampler)
        results.update(dict(input_ids=input_ids, attention_mask=attention_mask, ))

        return results

    def __len__(self):
        return len(self.dataset)


class MaskedLanguageSentenceOrderModel(PreTrainedModel):
    def __init__(self, backbone: PreTrainedModel, tokenizer, mlm_w, sentence_order_w, reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        hidden_size = backbone.config.hidden_size
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.config = backbone.config
        self.backbone = backbone
        self.mlm_w = mlm_w
        self.sentence_order_w = sentence_order_w

        if reinit:
            self.backbone.init_weights()

        self.sent_order_nn = nn.Linear(hidden_size, 1)
        init_weights(self.sent_order_nn, 0.01)

        self.lm_head = RobertaLMHead(backbone.config)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.backbone.embeddings.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask, label_mlm_input_ids: torch.Tensor, label_sentence_order: torch.Tensor, validation_iter=False):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
        label_mlm_input_ids = label_mlm_input_ids.view(-1)
        masked_lm_loss = self.loss_ce(prediction_scores, label_mlm_input_ids)

        cls_token_output = sequence_output[:, 0]
        label_sentence_order = label_sentence_order.float()
        sent_order_logits = self.sent_order_nn(cls_token_output).squeeze(-1)
        sentence_order_loss = self.loss_bce(sent_order_logits, label_sentence_order)

        sentence_order_accuracy = None
        mask_proportion = None
        mlm_accuracy = None
        non_mlm_accuracy = None
        if validation_iter:
            input_ids = input_ids.view(-1)
            mask_indices = (input_ids.int() != label_mlm_input_ids.int())
            mask_proportion = (mask_indices.sum() / attention_mask.sum()).item()
            lm_predictions = prediction_scores.detach().argmax(dim=-1)
            lm_accuracy = (lm_predictions == label_mlm_input_ids).float()
            mlm_accuracy = lm_accuracy[mask_indices].mean().item()
            non_mlm_accuracy = lm_accuracy[torch.logical_and(torch.logical_not(mask_indices), label_mlm_input_ids != self.tokenizer.pad_token_id)].mean().item()

            sent_order_preds = (torch.sigmoid(sent_order_logits.detach()) > 0.5).type(sent_order_logits.dtype)
            sentence_order_accuracy = (sent_order_preds == label_sentence_order).type(sent_order_logits.dtype).mean().item()

        return dict(loss=self.mlm_w * masked_lm_loss + self.sentence_order_w * sentence_order_loss,
                    mlm_loss=masked_lm_loss, sentence_order_loss=sentence_order_loss, mlm_accuracy=mlm_accuracy, non_mlm_accuracy=non_mlm_accuracy,
                    sentence_order_accuracy=sentence_order_accuracy, mask_proportion=mask_proportion)


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--model_config', required=True, type=str,
                        help='model config')

    parser.add_argument('--wandb_name', required=False, type=str, default="",
                        help='wandb_name')

    parser.add_argument('--total_steps', type=int, required=False,
                        help='total_steps')

    parser.add_argument('--batch_size', required=True, type=int,
                        help='Batch Size')

    parser.add_argument('--lr', default=optimizer_config.lr, type=float,
                        help='lr')
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

    parser.add_argument('--sentence_order_w', type=float, required=False, default=1.0,
                        help='sentence_order weight')

    parser.add_argument('--mlm_w', type=float, required=False, default=1.0,
                        help='generator_wmlm_w weight')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--optimizer', required=False, type=str, default="adamw",
                        help='optimizer')

    parser.add_argument('--num_workers', required=False, type=int, default=2,
                        help='Dataloader workers')

    parser.add_argument('--master_addr', type=str, required='MASTER_ADDR' not in os.environ,
                        default=None if 'MASTER_ADDR' not in os.environ else os.environ['MASTER_ADDR'],
                        help='Master ADDR')
    parser.add_argument('--master_port', type=str, required='MASTER_PORT' not in os.environ,
                        default=None if 'MASTER_PORT' not in os.environ else os.environ['MASTER_PORT'],
                        help='Master PORT')
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

    seed = 61567
    args.seed = seed
    return vars(args)


def build_dataloader(location, shuffle_dataset, batch_size, tokenizer, world_size=1, num_workers=None, max_length=512):
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    import os
    num_workers = min(max(os.cpu_count() // 2, 1), 4) if num_workers is None else num_workers
    dataset = Dataset.load_from_disk(location)
    dataset_args = dict(tokenizer=tokenizer, dataset=dataset,
                        tokenizer_args=dict(padding="max_length", truncation=True, return_tensors="pt", max_length=max_length),
                        word_mask_proba=0.15)

    kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else dict()

    dataset = MaskedLanguageSentenceOrderModelDataset(**dataset_args)
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
        assert args["world_size"] == 1
        device = torch.device("cpu")

        # init_method = "tcp://%s:%s" % ("127.0.0.1", "9999")
    else:
        device = torch.device(f'cuda:{gpu_device}')  # Unique only on individual node.
        torch.cuda.set_device(device)
    init_method="tcp://%s:%s" % (args["master_addr"], args["master_port"])

    rnd = torch.tensor(0.0, device="cpu")
    if args["world_size"] > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args["world_size"], init_method=init_method)
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
    optimizer_config.eps = 1e-7
    eps = 1e-7

    reinit = args["pretrained_model"] is None or "pretrained_model" not in args or args["pretrained_model"] == ""
    backbone, tokenizer = get_backbone(args["model_config"], reinit, dropout_prob=0.01)
    batch_size = args["batch_size"] if "batch_size" in args and isinstance(args["batch_size"], int) else batch_size
    mlm_w = args["mlm_w"] if "mlm_w" in args else 1.0
    sentence_order_w = args["sentence_order_w"] if "sentence_order_w" in args else 1.0

    model = MaskedLanguageSentenceOrderModel(backbone, tokenizer, mlm_w, sentence_order_w, reinit).to(device)
    trainable_model = model.backbone
    if local_rank == 0 and rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(model) / 1_000_000))

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

        print("[Train]: Time = %s, Loaded Pretrained model with Load type = %s, Torch Version = %s" % (get_time_string(), load_type, torch.__version__))
        del state_dict
    model = model.train()

    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in model.named_parameters() if v.requires_grad}))
    if args["world_size"] > 1:
        # model = FSDP(model, **fsdp_params)  # find_unused_parameters=True

        model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=True, bucket_cap_mb=10)  # find_unused_parameters=True

    clean_memory()
    barrier()
    optc = optimizer_config.to_dict()
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"],
                                      betas=(optc["beta_1"], optc["beta_2"]))
    elif args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=optc["lr"], momentum=0.9, weight_decay=optc["weight_decay"], nesterov=True)
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

    dataloader = build_dataloader(args["dataset"], args["shuffle_dataset"], batch_size, tokenizer,
                                  world_size=args["world_size"], num_workers=args["num_workers"], max_length=512)

    iter_size = max(args["accumulation_steps"], 1)
    no_sync = iter_size > 1
    steps_per_epoch = int(np.ceil(len(dataloader.sampler) / (batch_size * iter_size)) if dataloader.sampler is not None else (len(dataloader) / iter_size))
    if local_rank == 0:
        print("[Train]: Time = %s, Optimizer and Scheduler Initialised, max lr = %.5f, steps_per_epoch = %s, batch size = %s, dataloader length = %s, Sampler Present = %s, Sampler Length = %s" %
              (get_time_string(), optc["lr"], steps_per_epoch, batch_size, len(dataloader), dataloader.sampler is not None, len(dataloader.sampler) if dataloader.sampler is not None else -1))

    dataloader = get_next(dataloader)
    log_every_steps = args["log_every_steps"] * iter_size
    save_every_steps = args["save_every_steps"]

    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    # scheduler = optimization.get_linear_schedule_with_warmup(optimizer, optc["warmup_steps"], args["epochs"] * len(dataloader))
    div_factor = optc["lr"]/1e-6
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], total_steps=args["total_steps"],
                                                    div_factor=div_factor, three_phase=False, pct_start=0.06, anneal_strategy="linear", cycle_momentum=False)

    barrier()

    gradient_clipping = optc["gradient_clipping"]

    group = "%s-%s-%s-%sN-%s" % (args["wandb_name"], ds_name, args["model_config"], args["nodes"], time_string)
    wandb_init_args = dict(project="de_lm", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                           config={"args":args, "optimizer_config": optc},
                           settings=wandb.Settings(start_method="fork"))

    time.sleep(random.random())
    wandb.init(**wandb_init_args)

    full_times = []
    batch_times = []
    model_times = []
    model.zero_grad(set_to_none=True)
    samples_processed = 0
    samples_processed_this_log_iter = 0
    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)


    total_steps = args["total_steps"]
    steps_done = 0
    step = 0

    start_time = time.time()
    while steps_done < total_steps:
        random.seed(step)
        batch = dataloader()

        epoch = dataloader.epoch

        key = list(batch.keys())[0]
        bs_size = list(batch[key].size())
        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}

        gen_batch_time = time.time() - start_time

        batch_times.append(gen_batch_time)
        if (steps_done + 1) % save_every_steps == 0 or (args["total_steps"] is not None and (steps_done + 1) >= args["total_steps"]):
            state_dict = (getattr(model, "module", model).backbone).state_dict()
            if local_rank == 0:
                torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
            del state_dict
            clean_memory()
            barrier()
            if args["total_steps"] is not None and (steps_done + 1) >= args["total_steps"]:
                return

        samples_processed += int(batch[key].size(0))
        samples_processed_this_log_iter += int(batch[key].size(0))
        validation_iter = (step + 1) % log_every_steps == 0 or step == 0
        model_start = time.time()
        if no_sync and (step + 1) % iter_size != 0 and hasattr(model, "no_sync"):
            with model.no_sync():
                output = train_inner_loop(model, batch, optimizer,
                                          scheduler, gradient_clipping, iter_size=iter_size,
                                          no_sync=True, validation_iter=validation_iter)
            model_times.append(time.time() - model_start)
        else:
            output = train_inner_loop(model, batch, optimizer,
                                      scheduler, gradient_clipping, iter_size=iter_size,
                                      no_sync=False, validation_iter=validation_iter)
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
            model_times.append(time.time() - model_start)

        step += 1
        del batch

        full_time = time.time() - start_time
        full_times.append(full_time)
        if step == 0 and local_rank == 0:
            print("[Train]: Time = %s, First Batch Training for Rank = %s" % (get_time_string(), rank))
        if validation_iter:
            steps_remaining = total_steps - steps_done
            # print({k for k, v in output.items() if isinstance(v, torch.Tensor)})
            output = {k: float(v) if v else v for k, v in output.items()}
            samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
            wandb_log = dict(lr=optimizer.param_groups[0]['lr'], step=step, samples_processed=samples_processed, samples_per_second=samples_per_second,
                             batch_times=np.mean(batch_times), full_times=np.mean(full_times), model_times=np.mean(model_times),
                             steps_remaining=steps_remaining, pct_complete=(100 * steps_done / total_steps),
                             epoch=epoch,
                             **{k: v for k, v in output.items() if v is not None})

            wandb.log(wandb_log)
            if local_rank == 0:
                print("[Train]: Time = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Details = %s, LR = %s" %
                      (get_time_string(), rank, step, samples_processed, bs_size, output, optimizer.param_groups[0]['lr']))
                print("[Train-Timings]: Time = %s, Batch time = %.4f, Full Time = %.4f, Model Time = %.4f, samples_per_second = %s, steps_remaining = %s, pct_complete = %.4f" % (
                    get_time_string(), np.mean(batch_times), np.mean(full_times), np.mean(model_times), samples_per_second, steps_remaining, (100 * steps_done / total_steps),))
                # print("Step = %s, Steps Done = %s, log_every_steps = %s, total_steps = %s, steps_remaining = %s, validation_iter = %s, %s" % (step, steps_done, log_every_steps, total_steps, steps_remaining, validation_iter, (step + 1) % log_every_steps == 0))
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
    state_dict = (getattr(model, "module", model).backbone).state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
    del model


def forward_backward(batch, validation_iter, ddp_model, iter_size):

    output = ddp_model(**batch, validation_iter=validation_iter)

    loss = output.pop("loss") / iter_size
    loss.backward()
    _ = output.pop("predictions", None)
    loss_float = loss.item()
    del loss
    return output, loss_float


def train_inner_loop(ddp_model, batch, optimizer, scheduler, gradient_clipping, iter_size=1,
                     no_sync=False, validation_iter=False):

    if no_sync:
        with ddp_model.no_sync():
            output, loss_float = forward_backward(batch, validation_iter, ddp_model, iter_size)
    else:
        output, loss_float = forward_backward(batch, validation_iter, ddp_model, iter_size)

    # print([name for name, params in (ddp_model.student if isinstance(ddp_model, (PatchCLR, MultiTaskHighwayCLSPretraining)) else ddp_model).named_parameters() if params.grad is None])
    if not no_sync:

        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clipping)
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



