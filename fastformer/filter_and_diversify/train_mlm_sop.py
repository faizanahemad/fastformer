import copy
import sys
import traceback
from typing import Optional, Iterator

import numpy as np
import torch
from more_itertools import windowed
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
import traceback
from torch.multiprocessing import Process
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from torch.cuda.amp import GradScaler, autocast
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from fastformer.config import *
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
from torch.distributed.optim import ZeroRedundancyOptimizer

from tabulate import tabulate
from torch.multiprocessing import Process, ProcessContext

try:
    from torch.cuda.amp import GradScaler, autocast
except:
    pass

optimizer_config = dict(lr=5e-5, eps=1e-4, weight_decay=1e-4, beta_1=0.9, beta_2=0.98, gradient_clipping=1.0)

class get_next:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.epoch = 0

    def __call__(self):
        try:
            return next(self.iter)
        except StopIteration as st:
            self.epoch += 1
            dataloader = self.dataloader
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(self.epoch)
            else:
                print("Time = %s: Unable to set Epoch = %s" % (get_time_string(), self.epoch))
            self.iter = iter(dataloader)
            return next(self.iter)


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

    two_remove_proba = 0.15
    three_remove_proba = 0.05
    probability = probability / (1 + two_remove_proba + 2 * three_remove_proba)
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
    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    if full_length > 64:
        for i, t in enumerate(tokens):
            if t == tokenizer.mask_token_id and i > 1 and i < len(tokens) - 1:
                proba = random.random()
                if proba < two_remove_proba:
                    if random.random() < 0.5:
                        tokens[i - 1] = tokenizer.mask_token_id
                    else:
                        tokens[i + 1] = tokenizer.mask_token_id
                elif proba < two_remove_proba + three_remove_proba:
                    tokens[i - 1] = tokenizer.mask_token_id
                    tokens[i + 1] = tokenizer.mask_token_id
                else:
                    pass


    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    return torch.tensor(list(tokens))


class MaskedLanguageSentenceOrderModelDataset(Dataset):
    def __init__(self, tokenizer, tokenizer_args: dict, dataset: Dataset, word_mask_proba=0.15, mlm_sop_enabled=True):
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
        self.mlm_sop_enabled = mlm_sop_enabled

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.dataset[item]
        item = self.dataset[item]
        tokenizer = self.tokenizer

        text = item["text"]
        text = clean_text(text)
        length = len(text.strip().split())

        if length > self.allowed_raw_length:
            try:
                text = get_valid_sentences(text, self.sent_detector, tokenizer, int(self.tokenizer_args["max_length"] * 0.6), self.tokenizer_args["max_length"])
            except:
                text = " ".join(text.split()[:self.allowed_raw_length])
            length = len(text.strip().split())

        if self.mlm_sop_enabled:
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
            input_ids, attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs["attention_mask"].squeeze()
            results["label_mlm_input_ids"] = input_ids
            input_ids = token_id_masking(results["label_mlm_input_ids"], self.tokenizer, self.word_mask_proba, sampler=self.token_sampler)
            results.update(dict(input_ids=input_ids, attention_mask=attention_mask, ))
        else:
            tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
            results = dict(input_ids=tokenizer_outputs["input_ids"].squeeze(), attention_mask=tokenizer_outputs["attention_mask"].squeeze())

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
        self.config.gradient_checkpointing = True
        if hasattr(backbone, "pooler"):
            backbone.pooler = None
        self.backbone = backbone
        self.mlm_w = mlm_w
        self.sentence_order_w = sentence_order_w
        self.tokenizer = tokenizer

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

        return dict(loss=(self.mlm_w * masked_lm_loss) + (self.sentence_order_w * sentence_order_loss),
                    mlm_loss=masked_lm_loss.item(), sentence_order_loss=sentence_order_loss.item(),
                    mlm_accuracy=mlm_accuracy, non_mlm_accuracy=non_mlm_accuracy,
                    sentence_order_accuracy=sentence_order_accuracy, mask_proportion=mask_proportion)


class RTDMLMModel(PreTrainedModel):
    def __init__(self, backbone: PreTrainedModel, masking_model: nn.Module, tokenizer, mlm_w, sentence_order_w, reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        hidden_size = backbone.config.hidden_size
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.config = backbone.config
        self.config.gradient_checkpointing = True
        if hasattr(backbone, "pooler"):
            backbone.pooler = None
        self.masking_model = masking_model.eval()
        for p in self.masking_model.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.momentum_backbone = copy.deepcopy(backbone)
        self.momentum_backbone.load_state_dict(backbone.state_dict())
        self.copy_momentum = 0.9
        for p in self.momentum_backbone.parameters():
            p.requires_grad = False
        self.mlm_w = mlm_w
        self.tokenizer = tokenizer
        self.rtd_nn = nn.Linear(hidden_size, 1)
        init_weights(self.rtd_nn, 0.01)

        if reinit:
            self.backbone.init_weights()

        self.lm_head = RobertaLMHead(backbone.config)
        self.tie_weights()
        self.momentum_lm_head = copy.deepcopy(self.lm_head)
        self.momentum_lm_head.load_state_dict(self.lm_head.state_dict())
        for p in self.momentum_lm_head.parameters():
            p.requires_grad = False

    def do_masking(self, input_ids, attention_mask, validation_iter=False):
        label_mlm_input_ids = input_ids.clone()
        b, s = input_ids.shape[:2]
        ss = attention_mask.sum(1).float().mean().item()
        with torch.no_grad():
            mlm_rtd_hints = self.masking_model(input_ids, attention_mask)
        word_ce = mlm_rtd_hints["word_ce"]
        top_k = mlm_rtd_hints["top_k_alternatives"]
        word_wise_accuracy = mlm_rtd_hints["word_accuracy"]
        non_mask_locations = torch.logical_or(input_ids == self.tokenizer.eos_token_id, torch.logical_or(torch.logical_not(attention_mask.bool()), input_ids == self.tokenizer.bos_token_id))
        word_ce[non_mask_locations] = 0.0
        indices = torch.multinomial(word_ce, int(0.2 * ss), False)
        indices = indices[:, torch.randperm(indices.size()[1])]
        word_mask, rtd_mask = torch.chunk(indices, 2, dim=1)
        word_mask = [torch.arange(b, device=word_mask.device).repeat_interleave(word_mask.size(1)), word_mask.reshape(-1)]
        rtd_mask = [torch.arange(b, device=rtd_mask.device).repeat_interleave(rtd_mask.size(1)), rtd_mask.reshape(-1)]
        top_k = top_k[:, :, torch.randint(0, top_k.size(2), (1,))].squeeze(-1)
        input_ids[word_mask[0], word_mask[1]] = self.tokenizer.mask_token_id
        input_ids[rtd_mask[0], rtd_mask[1]] = top_k[rtd_mask[0], rtd_mask[1]]

        mask_locations = input_ids == self.tokenizer.mask_token_id
        mlm_input_ids = label_mlm_input_ids.clone()
        selected_mask_locations = torch.logical_and(mask_locations, torch.rand(input_ids.size(), device=input_ids.device) < 0.5)
        mlm_input_ids[selected_mask_locations] = self.tokenizer.mask_token_id
        with torch.no_grad():
            outputs = self.momentum_backbone(
                mlm_input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            sequence_output = outputs[0]
            teacher_prediction_scores = self.momentum_lm_head(sequence_output)[selected_mask_locations]
            lm_predictions = teacher_prediction_scores.detach().argmax(dim=-1)
            mlm_teacher_accuracy = (lm_predictions == label_mlm_input_ids[selected_mask_locations]).float().mean().item()

        rtd_locations = torch.logical_and(input_ids != label_mlm_input_ids, torch.logical_not(mask_locations))
        extra_masks = torch.logical_or(rtd_locations, torch.logical_and(torch.rand(input_ids.size(), device=input_ids.device) < 0.1, torch.logical_not(non_mask_locations)))
        rtd_input_ids = input_ids.clone()
        rtd_input_ids[extra_masks] = self.tokenizer.mask_token_id
        with torch.no_grad():
            outputs = self.backbone(
                rtd_input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)[rtd_locations]
            lm_predictions = prediction_scores.detach().argmax(dim=-1)
        rtd_replacement_accuracy = (lm_predictions == label_mlm_input_ids[rtd_locations]).float().mean().item()
        sampled_replacements = temperature_sampling(prediction_scores, 1.0).view(-1)
        input_ids[rtd_locations] = sampled_replacements
        rtd_post_replacement_accuracy = (sampled_replacements == label_mlm_input_ids[rtd_locations]).float().mean().item()


        mask_accuracy = None
        rtd_accuracy = None
        if validation_iter:
            mask_accuracy = word_wise_accuracy[word_mask[0], word_mask[1]].float().mean().item()
            rtd_accuracy = word_wise_accuracy[rtd_mask[0], rtd_mask[1]].float().mean().item()
        accuracy = mlm_rtd_hints["accuracy"]
        rtd_labels = torch.logical_and(input_ids != label_mlm_input_ids, input_ids != self.tokenizer.mask_token_id).float()
        return input_ids, label_mlm_input_ids, rtd_labels, mask_accuracy, rtd_accuracy, accuracy, selected_mask_locations, rtd_locations, teacher_prediction_scores, mlm_teacher_accuracy, rtd_replacement_accuracy, rtd_post_replacement_accuracy

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.backbone.embeddings.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask, validation_iter=False):
        input_ids, label_mlm_input_ids, rtd_labels, only_mask_accuracy_masking_model, only_rtd_accuracy_masking_model, accuracy_masking_model, selected_mask_locations, rtd_locations, teacher_prediction_scores, mlm_teacher_accuracy, rtd_replacement_accuracy, rtd_post_replacement_accuracy = self.do_masking(input_ids, attention_mask, validation_iter)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        attention_mask = attention_mask.bool()
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        distillation_loss = ((prediction_scores[selected_mask_locations] - teacher_prediction_scores) ** 2).mean()
        rtd_scores = self.rtd_nn(sequence_output).squeeze(-1)[attention_mask].view(-1)
        rtd_labels = rtd_labels[attention_mask].view(-1)
        rtd_loss = 10.0 * self.loss_bce(rtd_scores, rtd_labels)
        prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
        label_mlm_input_ids = label_mlm_input_ids.view(-1)
        masked_lm_loss = self.loss_ce(prediction_scores, label_mlm_input_ids)

        mask_proportion = None
        mlm_accuracy = None
        copy_token_lm_accuracy = None
        rtd_accuracy = None
        only_rtd_accuracy = None
        only_mask_lm_accuracy = None
        only_mask_proportion = None
        only_rtd_proportion = None
        non_rtd_accuracy = None
        only_rtd_lm_accuracy = None
        mlm_student_accuracy = None
        if validation_iter:
            momentum_param_copy(self.backbone, self.momentum_backbone, self.copy_momentum)
            momentum_param_copy(self.lm_head, self.momentum_lm_head, self.copy_momentum)
            rtd_labels = rtd_labels.bool()
            not_rtd_labels = torch.logical_not(rtd_labels)
            not_rtd_labels_mean = not_rtd_labels.float().mean().item()  # majority class prediction
            input_ids = input_ids.view(-1).int()
            mask_indices = (input_ids != label_mlm_input_ids.int())
            only_mask_indices = (input_ids == self.tokenizer.mask_token_id)
            rtd_binary = rtd_scores > 0.0
            only_rtd_accuracy = rtd_binary[rtd_labels].float().mean().item()
            rtd_accuracy = ((rtd_binary == rtd_labels).float().mean().item() - not_rtd_labels_mean) / (1 - not_rtd_labels_mean)
            rtd_accuracy = max(0, rtd_accuracy)
            non_rtd_accuracy = torch.logical_not(rtd_binary)[not_rtd_labels].float().mean().item()
            mask_proportion = (mask_indices.sum() / attention_mask.sum()).item()
            only_mask_proportion = (only_mask_indices.sum() / attention_mask.sum()).item()
            only_rtd_proportion = (rtd_labels.sum() / attention_mask.sum()).item()
            lm_predictions = prediction_scores.detach().argmax(dim=-1)
            lm_accuracy = (lm_predictions == label_mlm_input_ids).float()
            mlm_accuracy = lm_accuracy[mask_indices].mean().item()
            mlm_student_accuracy = lm_accuracy[selected_mask_locations.view(-1)].mean().item()
            only_rtd_lm_accuracy = lm_accuracy[attention_mask.view(-1)]
            only_rtd_lm_accuracy = only_rtd_lm_accuracy.reshape(-1)[rtd_labels].mean().item()
            only_mask_lm_accuracy = lm_accuracy[only_mask_indices].mean().item()
            copy_token_lm_accuracy = lm_accuracy[torch.logical_and(torch.logical_not(mask_indices), label_mlm_input_ids != self.tokenizer.pad_token_id)].mean().item()

        return dict(loss=self.mlm_w * (masked_lm_loss + rtd_loss) + distillation_loss, rtd_loss=rtd_loss.item(), distillation_loss=distillation_loss.item(),
                    mlm_loss=masked_lm_loss.item(), only_rtd_lm_accuracy=only_rtd_lm_accuracy,
                    mlm_teacher_accuracy=mlm_teacher_accuracy, mlm_student_accuracy=mlm_student_accuracy, rtd_replacement_accuracy=rtd_replacement_accuracy, rtd_post_replacement_accuracy=rtd_post_replacement_accuracy,
                    only_mask_lm_accuracy=only_mask_lm_accuracy, only_rtd_accuracy=only_rtd_accuracy,
                    mlm_accuracy=mlm_accuracy, copy_token_lm_accuracy=copy_token_lm_accuracy, rtd_accuracy=rtd_accuracy, non_rtd_accuracy=non_rtd_accuracy,
                    accuracy_masking_model=accuracy_masking_model, only_mask_accuracy_masking_model=only_mask_accuracy_masking_model, only_rtd_accuracy_masking_model=only_rtd_accuracy_masking_model,
                    mask_proportion=mask_proportion, only_mask_proportion=only_mask_proportion, only_rtd_proportion=only_rtd_proportion)


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

    parser.add_argument('--lr', default=optimizer_config["lr"], type=float,
                        help='lr')
    parser.add_argument('--weight_decay', default=optimizer_config["weight_decay"], type=float,
                        help='weight_decay')
    parser.add_argument('--gradient_clipping', default=optimizer_config["gradient_clipping"], type=float,
                        help='gradient_clipping')
    parser.add_argument('--beta_1', default=optimizer_config["beta_1"], type=float,
                        help='beta_1')
    parser.add_argument('--beta_2', default=optimizer_config["beta_2"], type=float,
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

    parser.add_argument('--sampling_column', required=False, type=str,
                        help='sampling_column')

    parser.add_argument('--mlm_w', type=float, required=False, default=1.0,
                        help='mlm_w weight')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--hard_mlm', action="store_true", default=False,
                        help='hard_mlm Position aware MLM and Multi-Segment Order Prediction + Coherence prediction')
    parser.add_argument('--hard_mlm_model', required=False, type=str,
                        help='hard_mlm_model storage location')

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

    seed = 61526837
    args.seed = seed
    return vars(args)


class DistributedWeightedSampler(DistributedSampler):
    def __init__(self, weights, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        # https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/18
        # https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.weights = torch.as_tensor(weights)
        self.replacement = True

    def __iter__(self) -> Iterator:
        indices = list(super().__iter__())
        weights = self.weights[indices]
        assert len(weights) == self.num_samples
        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        dataset_indices = torch.as_tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())


def build_dataloader(location, shuffle_dataset, batch_size, tokenizer, mlm_sop_enabled, sampling_column=None, world_size=1, num_workers=None, max_length=512):
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    import os
    num_workers = min(max(os.cpu_count() // 2, 1), 4) if num_workers is None else num_workers
    dataset = Dataset.load_from_disk(location)
    dataset_args = dict(tokenizer=tokenizer, dataset=dataset, mlm_sop_enabled=mlm_sop_enabled,
                        tokenizer_args=dict(padding="max_length", truncation=True, return_tensors="pt", max_length=max_length),
                        word_mask_proba=0.15)
    print("[Train]: Time = %s, Initializing Dataloader with dataset args = %s" % (dataset_args, get_time_string()))

    kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else dict()
    dataset = MaskedLanguageSentenceOrderModelDataset(**dataset_args)

    weights = None
    if sampling_column is not None:
        if "sbert" in sampling_column and "perplexity" in sampling_column and "tfidf" in sampling_column:
            ppl = np.log1p(dataset["perplexity"])
            ppl = 3 + ((ppl - ppl.mean()) / ppl.std()).clip(-3, 3) + 1e-5
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            sbert = 3 + ((sbert - sbert.mean()) / sbert.std()).clip(-3, 3) + 1e-5
            tfidf_top = np.array(dataset["tfidf_top_k_128"])
            tfidf_top = 3 + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-3, 3) + 1e-5
            weights = (ppl + sbert + tfidf_top) / 3.0
        elif "sbert" in sampling_column and "perplexity" in sampling_column:
            ppl = np.log1p(dataset["perplexity"])
            ppl = 3 + ((ppl - ppl.mean()) / ppl.std()).clip(-3, 3) + 1e-5
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            sbert = 3 + ((sbert - sbert.mean()) / sbert.std()).clip(-3, 3) + 1e-5
            weights = (ppl + sbert) / 2.0
        elif sampling_column == "perplexity":
            ppl = np.log1p(dataset["perplexity"])
            weights = 3 + ((ppl - ppl.mean()) / ppl.std()).clip(-3, 3) + 1e-5
        elif "tfidf" in sampling_column:
            tfidf_top = np.array(dataset["tfidf_top_k_128"])
            weights = 3 + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-3, 3) + 1e-5
        elif "tfidf_truncated" in sampling_column:
            tfidf_top = np.array(dataset["tfidf_truncated_average"])
            mean = tfidf_top[~np.isnan(tfidf_top)].mean()
            tfidf_top[np.isnan(tfidf_top)] = mean
            weights = 3 + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-3, 3) + 1e-5
        elif "sbert" in sampling_column:
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            weights = 3 + ((sbert - sbert.mean()) / sbert.std()).clip(-3, 3) + 1e-5
        else:
            raise ValueError("Sampling column = %s is not valid" % (sampling_column))

    if weights is not None:
        sampler = None if single_node else DistributedWeightedSampler(weights, dataset, shuffle=shuffle_dataset)
    else:
        sampler = None if single_node else DistributedSampler(dataset, shuffle=shuffle_dataset)

    train_loader = DataLoader(dataset, sampler=sampler,
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

    optimizer_config["lr"] = args["lr"]
    optimizer_config["weight_decay"] = args["weight_decay"]
    optimizer_config["gradient_clipping"] = args["gradient_clipping"]
    optimizer_config["beta_1"] = args["beta_1"]
    optimizer_config["beta_2"] = args["beta_2"]
    optimizer_config["eps"] = 1e-7
    eps = 1e-7

    reinit = args["pretrained_model"] is None or "pretrained_model" not in args or args["pretrained_model"] == ""
    backbone, tokenizer = get_backbone(args["model_config"], reinit, dropout_prob=0.01)
    batch_size = args["batch_size"] if "batch_size" in args and isinstance(args["batch_size"], int) else batch_size
    mlm_w = args["mlm_w"] if "mlm_w" in args else 1.0
    sentence_order_w = args["sentence_order_w"] if "sentence_order_w" in args else 1.0

    if isinstance(backbone, (CoOccurenceModel, MixerCoOccurenceModel, SMixerCoOccurenceModel)):
        model = backbone.to(device)
        trainable_model = model
        mlm_sop_enabled = False
    elif args["hard_mlm"]:
        masking_model, _ = get_backbone("co-oc-7-roberta-large", False, dropout_prob=0.0)
        state_dict = torch.load(args["hard_mlm_model"], map_location='cpu')
        masking_model.load_state_dict(state_dict, strict=True)
        masking_model = masking_model.eval()
        if hasattr(masking_model, "model"):
            masking_model.model = None
            del masking_model.model
        model = RTDMLMModel(backbone, masking_model, tokenizer, mlm_w, sentence_order_w, reinit).to(device)
        mlm_sop_enabled = False
    else:
        model = MaskedLanguageSentenceOrderModel(backbone, tokenizer, mlm_w, sentence_order_w, reinit).to(device)
        trainable_model = model.backbone
        mlm_sop_enabled = True
    if local_rank == 0 and rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(model) / 1_000_000))

    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]):
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)
        trainable_model.load_state_dict(state_dict, strict=True)
        print("[Train]: Time = %s, Loaded Pretrained model, Torch Version = %s" % (get_time_string(), torch.__version__))
        del state_dict
    model = model.train()

    # print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), {k for k, v in model.named_parameters() if v.requires_grad}))
    if args["world_size"] > 1:
        # model = FSDP(model, **fsdp_params)  # find_unused_parameters=True

        model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False, bucket_cap_mb=10, gradient_as_bucket_view=True)  # find_unused_parameters=True

    clean_memory()
    barrier()
    optc = copy.deepcopy(optimizer_config)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args["optimizer"] == "adamw":
        optimizer_class = torch.optim.AdamW
        optimizer = dict(lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"]))
    elif args["optimizer"] == "sgd":
        optimizer_class = torch.optim.SGD
        optimizer = dict(lr=optc["lr"], momentum=0.9, weight_decay=optc["weight_decay"], nesterov=True)
    else:
        raise ValueError
    if args["world_size"] > 1 and False:
        optimizer = ZeroRedundancyOptimizer(trainable_params, optimizer_class=optimizer_class, parameters_as_bucket_view=True, **optimizer)
    else:
        optimizer = optimizer_class(trainable_params, **optimizer)
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

    dataloader = build_dataloader(args["dataset"], args["shuffle_dataset"], batch_size, tokenizer, mlm_sop_enabled, args["sampling_column"],
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
    div_factor = optc["lr"]/1e-7
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], total_steps=args["total_steps"],
                                                    div_factor=div_factor, three_phase=False, pct_start=0.06, anneal_strategy="linear", cycle_momentum=False)

    barrier()

    gradient_clipping = optc["gradient_clipping"]

    group = "%s-%s-%sN-%s" % (args["wandb_name"], args["model_config"], args["nodes"], time_string)
    wandb_init_args = dict(project="fnd", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
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

            state_dict = getattr(getattr(model, "module", model), "backbone", getattr(model, "module", model)).state_dict()
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
            output = {k: float(v) for k, v in output.items() if try_float(v)}
            samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
            wandb_log = dict(lr=optimizer.param_groups[0]['lr'], step=step, updates_done=steps_done, samples_processed=samples_processed, samples_per_second=samples_per_second,
                             batch_times=np.mean(batch_times), full_times=np.mean(full_times), model_times=np.mean(model_times),
                             steps_remaining=steps_remaining, pct_complete=(100 * steps_done / total_steps),
                             epoch=epoch,
                             **{k: v for k, v in output.items() if v is not None})

            if local_rank <= (16 // args["world_size"]) or args["world_size"] <= 8:
                wandb.log(wandb_log)
            if local_rank == 0:
                print("[Train]: Time = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Details = %s, LR = %s" %
                      (get_time_string(), rank, step, samples_processed, bs_size, output, optimizer.param_groups[0]['lr']))
                print("[Train-Timings]: Time = %s, Batch time = %.4f, Full Time = %.4f, Model Time = %.4f, samples_per_second = %s, steps_remaining = %s, pct_complete = %.4f" % (
                    get_time_string(), np.mean(batch_times), np.mean(full_times), np.mean(model_times), samples_per_second, steps_remaining, (100 * steps_done / total_steps),))
                print("[Train]: Time = %s, wandb_log = %s" % (get_time_string(), wandb_log))
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
    state_dict = getattr(getattr(model, "module", model), "backbone", getattr(model, "module", model)).state_dict()
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

    # print([name for name, params in ddp_model.named_parameters() if params.grad is None])
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



