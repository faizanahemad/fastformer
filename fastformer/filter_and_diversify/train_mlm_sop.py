import copy
import sys
import traceback
from typing import Optional, Iterator, Tuple

import numpy as np
import torch
from more_itertools import windowed
from torch import nn, Tensor
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
from transformers.modeling_utils import get_parameter_dtype
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaEncoder

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
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('precision', 2)
pd.set_option('display.width', 1000)

from tabulate import tabulate
from torch.multiprocessing import Process, ProcessContext
from pprint import pprint
import json
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead



try:
    from torch.cuda.amp import GradScaler, autocast
except:
    pass

optimizer_config = dict(lr=1e-4, eps=1e-6, weight_decay=1e-3, beta_1=0.9, beta_2=0.98, gradient_clipping=1.0)


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
    def __init__(self, tokenizer, tokenizer_args: dict, dataset: Dataset, word_mask_proba=0.15, mlm_sop_enabled=True,
                 hard_lm_enabled=False,):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        try:
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.hard_lm_enabled = hard_lm_enabled
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
                text = get_valid_sentences(text, self.sent_detector, tokenizer, int(self.tokenizer_args["max_length"] * 0.75), self.tokenizer_args["max_length"])
            except:
                splits = text.split()
                text = " ".join(splits[:self.allowed_raw_length]) if random.random() < 0.5 else " ".join(splits[len(splits) - self.allowed_raw_length:])

        seg_sep_token = f" {tokenizer.sep_token} "
        if self.mlm_sop_enabled:
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
        elif self.hard_lm_enabled:
            # TODO: if hard tasks are given reduce mask rate
            task = random.randint(0, 7)

            if task == 0:
                task_labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
                noise_type = 0
            elif task == 1:
                # Do MSOP
                num_segments = random.randint(1, 4)
                segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
                num_segments = sum(segments != tokenizer.pad_token)
                seg_idxs = random.sample(range(num_segments), num_segments)
                in_order = all([i < j for i, j in windowed(seg_idxs, 2, fillvalue=1e8)])
                task_labels = list(seg_idxs) + [-100] * (11 - num_segments)
                text = seg_sep_token.join(segments)
                noise_type = 0 if in_order else 1
            elif task == 2:
                # Do place of insert
                sents = self.sent_detector.tokenize(text)
                sent_idx = random.randint(0, len(sents) - 2)
                masked_sent = sents[sent_idx]
                position = tokenizer.encode(" ".join(sents[:sent_idx]), return_offsets_mapping=False, **self.tokenizer_args)["attention_mask"].sum()
                if len(masked_sent.split()) > 16:
                    sents = sents[:sent_idx] + sents[sent_idx + 1:]
                else:
                    masked_sent_2 = sents[sent_idx + 1]
                    masked_sent = masked_sent + " " + masked_sent_2
                    sents = sents[:sent_idx] + sents[sent_idx + 2:]
                text = " ".join(sents) + seg_sep_token + masked_sent
                noise_type = 1
                task_labels = [-100, -100, -100, -100, position, -100, -100, -100, -100, -100, -100]
            elif task == 3:
                # Detect place of random insert
                random_item = self.dataset[random.randint(0, len(self.dataset))]
                random_text = random_item["text"]
                random_text = clean_text(random_text)
                random_sents = self.sent_detector.tokenize(random_text)
                random_sent_idx = random.randint(0, len(random_sents) - 2)
                noise_sent = random_sents[random_sent_idx]
                if len(noise_sent.split()) < 16:
                    noise_sent = random_sents[random_sent_idx] + " " + random_sents[random_sent_idx + 1]

                start_position = -100
                end_position = -100
                sents = self.sent_detector.tokenize(text)
                if len(sents) > 3:
                    sent_idx = random.randint(1, len(sents) - 2)
                    removed_sent = sents[sent_idx]
                    start_position = tokenizer.encode(" ".join(sents[:sent_idx]), return_offsets_mapping=False, **self.tokenizer_args)["attention_mask"].sum()
                    end_position = tokenizer.encode(" ".join(sents[:sent_idx] + [noise_sent]), return_offsets_mapping=False, **self.tokenizer_args)["attention_mask"].sum()
                    if len(removed_sent.split()) > 16:
                        sents = sents[:sent_idx] + [noise_sent] + sents[sent_idx + 1:]
                    else:
                        sents = sents[:sent_idx] + [noise_sent] + sents[sent_idx + 2:]
                    end_position = min(end_position, self.tokenizer_args["max_length"])

                    noise_type = 2
                else:
                    noise_type = 0
                text = " ".join(sents)
                task_labels = [-100, -100, -100, -100, -100, start_position, end_position, -100, -100, -100, -100]

            elif task == 4:
                # Detect deletion position
                sents = self.sent_detector.tokenize(text)
                sent_idx = random.randint(0, len(sents) - 2)
                removed_sent = sents[sent_idx]
                position = tokenizer.encode(" ".join(sents[:sent_idx]), return_offsets_mapping=False, **self.tokenizer_args)["attention_mask"].sum()
                if len(sents) > 3:
                    if len(removed_sent.split()) > 32:
                        sents = sents[:sent_idx] + sents[sent_idx + 1:]
                    else:
                        sents = sents[:sent_idx] + sents[sent_idx + 2:]
                    noise_type = 3
                else:
                    position = -100
                    noise_type = 0
                text = " ".join(sents)
                task_labels = [-100, -100, -100, -100, -100, -100, -100, position, -100, -100, -100]

            elif task == 5:
                # Reversed order without SEP token, detect where reversal has happened
                num_segments = 2
                segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
                num_segments = sum(segments != tokenizer.pad_token)
                position = -100
                if num_segments > 1:
                    segments[0], segments[1] = segments[1], segments[0]
                    position = tokenizer.encode(" ".join(segments[0]), return_offsets_mapping=False, **self.tokenizer_args)["attention_mask"].sum()
                    noise_type = 1
                else:
                    noise_type = 0
                text = seg_sep_token.join(segments)
                task_labels = [-100, -100, -100, -100, -100, -100, -100, -100, position, -100, -100]

            elif task == 6:
                num_segments = 2
                segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
                num_segments = sum(segments != tokenizer.pad_token)
                if num_segments > 1:
                    if random.random() < 0.3:
                        noise_type = 0
                        label = 0
                    else:
                        rn = random.random()
                        if rn < 0.3:
                            label = 1
                            segments[0], segments[1] = segments[1], segments[0]
                            noise_type = 1
                        elif rn < 0.7:
                            label = 2
                            segments[0], segments[1] = segments[1], segments[0]
                            segments[0] = " ".join(self.sent_detector.tokenize(segments[0])[1:]) if random.random() < 0.25 else segments[0]
                            segments[0] = " ".join(self.sent_detector.tokenize(segments[0])[:-1]) if random.random() < 0.25 else segments[0]
                            segments[1] = " ".join(self.sent_detector.tokenize(segments[1])[1:]) if random.random() < 0.25 else segments[1]
                            segments[1] = " ".join(self.sent_detector.tokenize(segments[1])[:-1]) if random.random() < 0.25 else segments[1]
                            noise_type = 4  # Order change + Deletion
                        else:
                            label = 3
                            random_item = self.dataset[random.randint(0, len(self.dataset))]
                            random_text = random_item["text"]
                            random_text = clean_text(random_text)
                            if random.random() < 0.5:
                                segments[1] = " ".join(random_text.split()[:len(segments[1].split())])
                            else:
                                segments[0] = segments[1]
                                segments[1] = " ".join(random_text.split()[:len(segments[1].split())])
                            noise_type = 2
                else:
                    label = 0
                    noise_type = 0
                task_labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, label, -100]
                text = seg_sep_token.join(segments)

            elif task == 7:
                # Select correct option, 1st and last sent are negatives, need atleast 6 sents
                sents = self.sent_detector.tokenize(text)
                ns_possible = False
                if len(sents) > 7:
                    ns_possible = True
                    if random.random() < 0.5:
                        ns, sents = (sents[0], sents[1:]) if len(sents[0].split()) > 16 else (sents[0] + sents[1], sents[2:])
                    else:
                        ns, sents = (sents[-1], sents[:-1]) if len(sents[-1].split()) > 16 else (sents[-2] + sents[-1], sents[:-2])


                sent_idx = random.randint(0, len(sents) - 2)
                masked_sent = sents[sent_idx]
                if len(masked_sent.split()) > 16:
                    sents = sents[:sent_idx] + seg_sep_token + sents[sent_idx + 1:]
                else:
                    masked_sent_2 = sents[sent_idx + 1]
                    masked_sent = masked_sent + " " + masked_sent_2
                    sents = sents[:sent_idx] + seg_sep_token + sents[sent_idx + 2:]
                if ns_possible and random.random() < 0.6:
                    text = " ".join(sents) + seg_sep_token + ns
                    label = 0
                    noise_type = 2
                else:
                    text = " ".join(sents) + seg_sep_token + masked_sent
                    label = 1
                    noise_type = 1
                task_labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, label]
            else:
                raise ValueError
            task_labels.append(noise_type)

            tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
            results = dict(input_ids=tokenizer_outputs["input_ids"].squeeze(),
                           attention_mask=tokenizer_outputs["attention_mask"].squeeze(),
                           task_labels=task_labels, noise_type=noise_type)


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


class RTDMLMModelOld(PreTrainedModel):
    def __init__(self, backbone: PreTrainedModel, lm_head: nn.Module, masking_model: nn.Module, tokenizer, mlm_w, sentence_order_w, reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        hidden_size = backbone.config.hidden_size
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="none")
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.config = backbone.config
        self.config.gradient_checkpointing = True
        self.rtd_temperature = 1.5
        self.word_ce_schedule = 1.0
        if hasattr(backbone, "pooler"):
            backbone.pooler = None
        self.masking_model = masking_model.eval()
        for p in self.masking_model.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.mlm_w = mlm_w
        self.tokenizer = tokenizer
        self.rtd_nn = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))
        init_weights(self.rtd_nn, 0.01)
        self.lm_head = lm_head
        if reinit:
            self.backbone.init_weights()
            init_weights(self.lm_head, 0.01)

        self.tie_weights()
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id

    def do_masking(self, input_ids, attention_mask, validation_iter=False):
        label_mlm_input_ids = input_ids.clone()
        b, s = input_ids.shape[:2]
        ss = attention_mask.sum(1).float().mean().item()
        with torch.no_grad():
            mlm_rtd_hints = self.masking_model(input_ids, attention_mask, validation_iter=validation_iter)
        attention_mask = attention_mask.bool()
        word_ce = mlm_rtd_hints["word_ce"]
        co_oc_teacher_prediction_scores = None
        if self.word_ce_schedule < 0.0:
            co_oc_teacher_prediction_scores = mlm_rtd_hints["prediction_scores"]
            word_ce_max = word_ce.max()
            word_ce = (word_ce ** self.word_ce_schedule).clip(0, word_ce_max)
        else:
            word_ce = (word_ce ** self.word_ce_schedule)
        non_mask_locations = torch.logical_or(input_ids == self.eos_token_id, torch.logical_or(torch.logical_not(attention_mask), input_ids == self.bos_token_id))
        word_ce[non_mask_locations] = 0.0
        decided_noise_proportion = (0.15 + random.random() * 0.05)
        average_tokens_per_sample = ss
        indices = torch.multinomial(word_ce, int(decided_noise_proportion * average_tokens_per_sample), False)
        indices = indices[:, torch.randperm(indices.size()[1])]

        word_mask, rtd_mask_model = torch.chunk(indices, 2, dim=1)
        word_mask = [torch.arange(b, device=word_mask.device).repeat_interleave(word_mask.size(1)), word_mask.reshape(-1)]
        rtd_mask_model = [torch.arange(b, device=rtd_mask_model.device).repeat_interleave(rtd_mask_model.size(1)), rtd_mask_model.reshape(-1)]

        input_ids[word_mask[0], word_mask[1]] = self.tokenizer.mask_token_id
        rtd_locations_model = torch.zeros_like(input_ids, dtype=torch.bool)
        rtd_locations_model[rtd_mask_model[0], rtd_mask_model[1]] = True
        rtd_input_ids = label_mlm_input_ids.clone()
        rtd_input_ids[rtd_locations_model] = self.tokenizer.mask_token_id
        with torch.no_grad():
            outputs = self.backbone(
                rtd_input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            sequence_output = outputs[0][rtd_locations_model]
            prediction_scores = self.lm_head(sequence_output)
            if validation_iter:
                lm_predictions = prediction_scores.detach().argmax(dim=-1)
        sampled_replacements = temperature_sampling(prediction_scores, self.rtd_temperature).view(-1)
        input_ids[rtd_locations_model] = sampled_replacements
        rtd_labels = torch.logical_and(input_ids != label_mlm_input_ids, input_ids != self.tokenizer.mask_token_id).float()
        mask_locations = input_ids == self.tokenizer.mask_token_id
        all_noise_locations = torch.logical_or(mask_locations, rtd_locations_model)
        mask_accuracy = None
        rtd_masking_accuracy = None
        rtd_model_post_replacement_accuracy = None
        rtd_model_accuracy = None
        accuracy = None
        all_rtd_proportion = None
        all_rtd_fraction = None
        if validation_iter:
            attention_sum = attention_mask.sum()
            word_wise_accuracy = mlm_rtd_hints["word_accuracy"]
            mask_accuracy = word_wise_accuracy[mask_locations].float().mean().item()
            rtd_masking_accuracy = word_wise_accuracy[rtd_locations_model].float().mean().item()
            labels_rtd = label_mlm_input_ids[rtd_locations_model]
            rtd_model_accuracy = (lm_predictions == labels_rtd).float().mean().item()
            rtd_input_ids = input_ids[rtd_locations_model]
            all_rtd_proportion = ((rtd_input_ids != labels_rtd).sum() / attention_sum).item()
            rtd_model_post_replacement_accuracy = (rtd_input_ids == labels_rtd).float().mean().item()
            accuracy = mlm_rtd_hints["accuracy"]
        return dict(input_ids=input_ids, label_mlm_input_ids=label_mlm_input_ids, rtd_labels=rtd_labels,
                    mask_accuracy=mask_accuracy, rtd_accuracy=rtd_masking_accuracy,
                    accuracy=accuracy, rtd_post_replacement_accuracy=rtd_model_post_replacement_accuracy, rtd_model_accuracy=rtd_model_accuracy,
                    rtd_model_post_replacement_accuracy=rtd_model_post_replacement_accuracy,
                    decided_noise_proportion=decided_noise_proportion, average_tokens_per_sample=average_tokens_per_sample,
                    co_oc_teacher_prediction_scores=co_oc_teacher_prediction_scores, all_noise_locations=all_noise_locations, all_rtd_proportion=all_rtd_proportion, all_rtd_fraction=all_rtd_fraction)

    def get_output_embeddings(self):
        if isinstance(self.lm_head, RobertaLMHead):
            return self.lm_head.decoder
        elif isinstance(self.lm_head,(BertOnlyMLMHead, DebertaV2OnlyMLMHead, DebertaOnlyMLMHead)):
            return self.lm_head.predictions.decoder
        else:
            raise ValueError("Only Roberta & Deberta LM heads supported")

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.backbone.embeddings.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask, validation_iter=False):
        mask_dict = self.do_masking(input_ids, attention_mask, validation_iter)
        input_ids, label_mlm_input_ids, rtd_labels = dict_get(mask_dict, "input_ids", "label_mlm_input_ids", "rtd_labels")
        only_mask_accuracy_masking_model, only_rtd_accuracy_masking_model, all_rtd_proportion = dict_get(mask_dict, "mask_accuracy", "rtd_accuracy", "all_rtd_proportion")
        accuracy_masking_model, rtd_post_replacement_accuracy, rtd_model_accuracy = dict_get(mask_dict, "accuracy", "rtd_post_replacement_accuracy", "rtd_model_accuracy")
        decided_noise_proportion, average_tokens_per_sample, all_rtd_fraction = dict_get(mask_dict, "decided_noise_proportion", "average_tokens_per_sample", "all_rtd_fraction")
        mask_stats = dict(decided_noise_proportion=decided_noise_proportion, average_tokens_per_sample=average_tokens_per_sample,
                          all_rtd_fraction=all_rtd_fraction, accuracy_masking_model=accuracy_masking_model, rtd_post_replacement_accuracy=rtd_post_replacement_accuracy,
                          rtd_model_accuracy=rtd_model_accuracy, only_mask_accuracy_masking_model=only_mask_accuracy_masking_model,
                          only_rtd_accuracy_masking_model=only_rtd_accuracy_masking_model, all_rtd_proportion=all_rtd_proportion)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        attention_mask = attention_mask.bool()
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        co_oc_guidance_loss = None
        co_oc_teacher_prediction_scores, all_noise_locations = dict_get(mask_dict, "co_oc_teacher_prediction_scores", "all_noise_locations")
        if self.word_ce_schedule < 0:
            co_oc_guidance_loss = 10 * ((F.softmax(prediction_scores, dim=-1) - F.softmax(co_oc_teacher_prediction_scores, dim=-1)) ** 2)[all_noise_locations].sum(-1).mean()
        rtd_scores = self.rtd_nn(sequence_output[attention_mask]).view(-1)
        rtd_labels = rtd_labels[attention_mask].view(-1)
        rtd_loss = 10.0 * self.loss_bce(rtd_scores, rtd_labels)
        prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
        label_mlm_input_ids = label_mlm_input_ids.view(-1)
        masked_lm_loss = self.loss_ce(prediction_scores, label_mlm_input_ids)
        masked_lm_loss = masked_lm_loss.mean() + masked_lm_loss[all_noise_locations.view(-1)].mean()
        # All token MLM averages over all tokens and copy tokens have low average.

        val_stats = dict()
        if validation_iter:
            only_rtd_proportion = rtd_labels.mean().item()
            am_sum = attention_mask.sum()
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
            mask_proportion = (mask_indices.sum() / am_sum).item()
            only_mask_proportion = (only_mask_indices.sum() / am_sum).item()
            lm_predictions = prediction_scores.detach().argmax(dim=-1)
            lm_accuracy = (lm_predictions == label_mlm_input_ids).float()
            mlm_accuracy = lm_accuracy[mask_indices].mean().item()
            only_rtd_lm_accuracy = lm_accuracy[attention_mask.view(-1)]
            only_rtd_lm_accuracy = only_rtd_lm_accuracy[rtd_labels].mean().item()
            only_mask_lm_accuracy = lm_accuracy[only_mask_indices].mean().item()
            copy_token_lm_accuracy = lm_accuracy[torch.logical_and(torch.logical_not(mask_indices), label_mlm_input_ids != self.tokenizer.pad_token_id)].mean().item()
            val_stats = dict(mask_proportion=mask_proportion, mlm_accuracy=mlm_accuracy, copy_token_lm_accuracy=copy_token_lm_accuracy, rtd_accuracy=rtd_accuracy,
                             only_rtd_accuracy=only_rtd_accuracy, only_mask_lm_accuracy=only_mask_lm_accuracy, only_mask_proportion=only_mask_proportion, only_rtd_proportion=only_rtd_proportion,
                             non_rtd_accuracy=non_rtd_accuracy, only_rtd_lm_accuracy=only_rtd_lm_accuracy)

        return dict(loss=self.mlm_w * (masked_lm_loss + rtd_loss) + (co_oc_guidance_loss if co_oc_guidance_loss is not None else 0.0),
                    rtd_loss=rtd_loss.item(), co_oc_guidance_loss=(co_oc_guidance_loss.item() if co_oc_guidance_loss is not None else None),
                    masked_lm_loss=masked_lm_loss.item(), **val_stats, **mask_stats)


class CEPredictor(nn.Module):
    def __init__(self, config, in_dims=768):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_size = config.hidden_size
        self.config = config
        self.ce_pred = RobertaEncoder(config)
        self.in_fc1 = nn.Linear(self.in_dims, self.hidden_size)
        self.in_fc2 = nn.Linear(self.in_dims, self.hidden_size)
        self.in_fc3 = nn.Linear(self.in_dims, self.hidden_size)
        self.in_fc4 = nn.Linear(64, self.hidden_size)
        self.out_fc = nn.Linear(self.hidden_size, 1)
        init_weights(self.in_fc1, 0.01)
        init_weights(self.in_fc2, 0.01)
        init_weights(self.in_fc3, 0.01)
        init_weights(self.in_fc4, 0.01)
        init_weights(self.out_fc, 0.01)
        init_weights(self.ce_pred, 0.01)

    @property
    def dtype(self) -> torch.dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, masked_embeddings, word_embeddings, model_outputs, mask_ce, attention_mask):
        # print("CEPredictor attention_mask = ", attention_mask.size(), attention_mask.dtype)
        attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.size(), attention_mask.device)
        masked_embeddings = self.in_fc1(masked_embeddings)
        word_embeddings = self.in_fc2(word_embeddings)
        model_outputs = self.in_fc3(model_outputs)
        mask_ce = self.in_fc4(encode_scalar_column(mask_ce))
        input_embeddings = masked_embeddings + word_embeddings + model_outputs + mask_ce
        last_hidden_state = self.ce_pred(input_embeddings, attention_mask, return_dict=False)[0]
        outputs = self.out_fc(last_hidden_state)
        return outputs.squeeze(-1)


class RTDMLMModel(PreTrainedModel):
    def __init__(self, backbone: PreTrainedModel, lm_head: nn.Module, masking_model: nn.Module, tokenizer, mlm_w, sentence_order_w, weight_decay, reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        hidden_size = backbone.config.hidden_size
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="mean")
        self.config = backbone.config
        self.config.gradient_checkpointing = False
        self.rtd_temperature = 1.5
        self.word_ce_schedule = 1.5
        if hasattr(backbone, "pooler"):
            backbone.pooler = None
        self.masking_model = masking_model.eval()
        for p in self.masking_model.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.mlm_w = mlm_w
        self.tokenizer = tokenizer
        encoder_config = copy.deepcopy(backbone.config)
        encoder_config.hidden_size = 256
        encoder_config.num_hidden_layers = 4
        encoder_config.num_attention_heads = 8
        encoder_config.intermediate_size = 1024
        encoder_config.hidden_dropout_prob = 0.0
        encoder_config.attention_probs_dropout_prob = 0.0

        self.lm_head = lm_head
        self.initial_backbone_parameters = None
        if reinit:
            self.backbone.init_weights()
            init_weights(self.lm_head, 0.01)
        # else:
        #     self.weight_decay = weight_decay
        #     self.initial_backbone_parameters = torch.cat([p.detach().data.clone().reshape(-1) for p in self.backbone.parameters()])
        #     self.initial_backbone_parameters.requires_grad = False

        self.tie_weights()
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id

    def get_hard_mask(self, input_ids, attention_mask, non_mask_locations):
        b, s = input_ids.shape[:2]
        mask_token_id = self.tokenizer.mask_token_id
        with torch.no_grad():
            mask_probas = torch.rand((b, s), device=input_ids.device)
            mask_probas[non_mask_locations] = 1.0
            mask_locations = mask_probas < 0.15
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_locations] = mask_token_id
            masked_embeddings = self.backbone.embeddings.word_embeddings(masked_input_ids)
            word_embeddings = self.backbone.embeddings.word_embeddings(input_ids)
            model_outputs = self.backbone(inputs_embeds=masked_embeddings, attention_mask=attention_mask, return_dict=False)[0]
            prediction_scores = self.lm_head(model_outputs)
            mask_ce = self.loss_ce(prediction_scores.view(-1, prediction_scores.size(-1)), input_ids.view(-1)).reshape(b, s)
        predicted_ce = self.ce_pred(masked_embeddings, word_embeddings, model_outputs, mask_ce, attention_mask)
        predicted_ce_use = predicted_ce.detach()
        predicted_ce_use[non_mask_locations] = 0.0
        predicted_ce_use[mask_locations] = mask_ce[mask_locations]
        asum = attention_mask.sum(1)
        hard_masks = [torch.topk(pce, int(0.4 * asum[ix])).indices[int(0.2 * asum[ix]):] for ix, pce in enumerate(predicted_ce_use)]
        hard_masks = [hm[torch.randperm(hm.size(0))[:int(0.08 * asum[ix])]] for ix, hm in enumerate(hard_masks)]
        hard_masks_batches = [torch.tensor([ix]*hm.size(0), device=input_ids.device) for ix, hm in enumerate(hard_masks)]
        hard_mask_locations = torch.zeros_like(input_ids, dtype=torch.bool)
        hard_masks_batches = torch.cat(hard_masks_batches)
        hard_masks = torch.cat(hard_masks)
        hard_mask_locations[hard_masks_batches.long(), hard_masks.long()] = True
        return dict(hard_mask_locations=hard_mask_locations, predicted_ce=predicted_ce)

    def get_hard_mask_v2(self, input_ids, attention_mask, non_mask_locations):
        b, s = input_ids.shape[:2]
        mask_token_id = self.tokenizer.mask_token_id
        mask_probas = torch.rand((b, s), device=input_ids.device)
        mask_probas[non_mask_locations] = 0.0
        probas_sections = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
        ce = torch.zeros_like(input_ids, dtype=torch.float)
        for p_start, p_end in zip(probas_sections[:-1], probas_sections[1:]):
            mask_locations = p_start < mask_probas
            mask_locations = torch.logical_and(mask_locations, mask_probas < p_end)
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_locations] = mask_token_id
            with torch.no_grad():
                model_outputs = self.backbone(input_ids=masked_input_ids, attention_mask=attention_mask, return_dict=False)[0][mask_locations]
                prediction_scores = self.lm_head(model_outputs)
                mask_ce = self.loss_ce(prediction_scores, input_ids[mask_locations])
                mask_ce[prediction_scores.argmax(dim=-1) == input_ids[mask_locations]] = 0.0
            ce[mask_locations] = mask_ce
        ce[non_mask_locations] = 0.0
        asum = attention_mask.sum(1)
        hard_masks = [torch.topk(pce, int(0.3 * asum[ix])).indices[int(0.2 * asum[ix]):] for ix, pce in enumerate(ce)]
        hard_masks = [hm[torch.randperm(hm.size(0))[:int(0.08 * asum[ix])]] for ix, hm in enumerate(hard_masks)]
        hard_masks_batches = [torch.tensor([ix]*hm.size(0), device=input_ids.device) for ix, hm in enumerate(hard_masks)]
        hard_mask_locations = torch.zeros_like(input_ids, dtype=torch.bool)
        hard_masks_batches = torch.cat(hard_masks_batches)
        hard_masks = torch.cat(hard_masks)
        hard_mask_locations[hard_masks_batches.long(), hard_masks.long()] = True
        return dict(hard_mask_locations=hard_mask_locations, predicted_ce=ce)

    def get_hard_mask_v3(self, input_ids, attention_mask, non_mask_locations):
        b, s = input_ids.shape[:2]
        with torch.no_grad():
            model_outputs = self.backbone(inputs_embeds=self.drop(self.backbone.embeddings.word_embeddings(input_ids)), attention_mask=attention_mask, return_dict=False)[0]
            prediction_scores = self.lm_head(model_outputs)
            mask_ce = self.loss_ce(prediction_scores.view(-1, prediction_scores.size(-1)), input_ids.view(-1)).reshape(b, s)
            top_confs, _ = prediction_scores.topk(2, -1)
            top_confs = F.softmax(top_confs, dim=-1)
            confidences = top_confs[:, :, 0] - top_confs[:, :, 1]
            under_confidence_scores = (1 - confidences)  # 1/confidences

            predictions = prediction_scores.argmax(dim=-1)
            correctness = predictions == input_ids
            mask_ce = mask_ce + under_confidence_scores
            mask_ce[correctness] = -1e-3
        ce = mask_ce
        ce[non_mask_locations] = -1e-3
        asum = attention_mask.sum(1)
        hard_masks = [torch.topk(pce, int(0.3 * asum[ix])).indices[int(0.1 * asum[ix]):] for ix, pce in enumerate(ce)]
        hard_masks = [hm[torch.randperm(hm.size(0))[:int(0.08 * asum[ix])]] for ix, hm in enumerate(hard_masks)]
        hard_masks_batches = [torch.tensor([ix]*hm.size(0), device=input_ids.device) for ix, hm in enumerate(hard_masks)]
        hard_mask_locations = torch.zeros_like(input_ids, dtype=torch.bool)
        hard_masks_batches = torch.cat(hard_masks_batches)
        hard_masks = torch.cat(hard_masks)
        hard_mask_locations[hard_masks_batches.long(), hard_masks.long()] = True
        # print("hard_mask_locations correctness = %s" % correctness[hard_mask_locations].float().mean().item())
        return dict(hard_mask_locations=hard_mask_locations, predicted_ce=ce)

    def get_hard_mask_v4(self, input_ids, attention_mask, non_mask_locations):
        b, s = input_ids.shape[:2]
        mask_probas = torch.rand((b, s), device=input_ids.device)
        mask_probas[non_mask_locations] = 1.0
        mask_probas[:, 0] = 1.0
        mask_locations = mask_probas < 0.05
        return dict(hard_mask_locations=mask_locations, predicted_ce=mask_probas)

    def do_masking(self, input_ids, attention_mask, validation_iter=False):
        label_mlm_input_ids = input_ids.clone()
        b, s = input_ids.shape[:2]
        ss = attention_mask.sum(1).float().mean().item()
        mask_token_id = self.tokenizer.mask_token_id

        non_mask_locations = torch.logical_or(input_ids == self.eos_token_id,
                                              torch.logical_or(torch.logical_not(attention_mask), input_ids == self.bos_token_id))
        with torch.no_grad():
            mlm_rtd_hints = self.masking_model(input_ids, attention_mask, validation_iter=validation_iter)

        random_mask_locations, predicted_ce = dict_get(self.get_hard_mask_v4(input_ids, attention_mask, non_mask_locations), "hard_mask_locations", "predicted_ce")

        word_ce = mlm_rtd_hints["word_ce"]
        word_ce_max = max(1e-2, word_ce.max().item())
        word_ce = (word_ce ** self.word_ce_schedule).clip(1e-5, 0.98 * (word_ce_max ** self.word_ce_schedule))
        word_ce[torch.logical_or(non_mask_locations, random_mask_locations)] = 1e-5
        decided_noise_proportion = (0.09 + random.random() * 0.03)
        average_tokens_per_sample = ss
        word_mask = torch.multinomial(word_ce, int(decided_noise_proportion * average_tokens_per_sample), False)
        word_mask = [torch.arange(b, device=word_mask.device).repeat_interleave(word_mask.size(1)), word_mask.reshape(-1)]

        input_ids[word_mask[0], word_mask[1]] = mask_token_id
        co_oc_mask_locations = input_ids == self.tokenizer.mask_token_id
        input_ids[random_mask_locations] = mask_token_id
        mask_locations = input_ids == self.tokenizer.mask_token_id
        all_noise_locations = mask_locations

        mask_accuracy = None
        accuracy = None
        co_oc_mask_accuracy = None
        random_mask_accuracy = None
        if validation_iter:
            word_wise_accuracy = mlm_rtd_hints["word_accuracy"]
            mask_accuracy = word_wise_accuracy[mask_locations].float().mean().item()
            co_oc_mask_accuracy = word_wise_accuracy[co_oc_mask_locations].float().mean().item()
            random_mask_accuracy = word_wise_accuracy[random_mask_locations].float().mean().item()
            accuracy = mlm_rtd_hints["word_accuracy"].float().mean().item()
        return dict(input_ids=input_ids, label_mlm_input_ids=label_mlm_input_ids,
                    mask_accuracy=mask_accuracy, co_oc_mask_accuracy=co_oc_mask_accuracy,
                    accuracy=accuracy, random_mask_accuracy=random_mask_accuracy,
                    random_mask_locations=random_mask_locations, co_oc_mask_locations=co_oc_mask_locations,
                    decided_noise_proportion=decided_noise_proportion, average_tokens_per_sample=average_tokens_per_sample,
                    all_noise_locations=all_noise_locations)

    def get_output_embeddings(self):
        if isinstance(self.lm_head, RobertaLMHead):
            return self.lm_head.decoder
        elif isinstance(self.lm_head,(BertOnlyMLMHead, DebertaV2OnlyMLMHead, DebertaOnlyMLMHead)):
            return self.lm_head.predictions.decoder
        else:
            raise ValueError("Only Roberta & Deberta LM heads supported")

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.backbone.embeddings.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask, validation_iter=False):
        mask_dict = self.do_masking(input_ids, attention_mask, validation_iter)
        input_ids, label_mlm_input_ids = dict_get(mask_dict, "input_ids", "label_mlm_input_ids")
        only_mask_accuracy_masking_model, only_co_oc_mask_accuracy_masking_model, only_random_mask_accuracy_masking_model = dict_get(mask_dict, "mask_accuracy", "co_oc_mask_accuracy", "random_mask_accuracy")
        accuracy_masking_model = dict_get(mask_dict, "accuracy")
        average_tokens_per_sample = dict_get(mask_dict,"average_tokens_per_sample")
        mask_stats = dict(average_tokens_per_sample=average_tokens_per_sample,
                          accuracy_masking_model=accuracy_masking_model,
                          only_mask_accuracy_masking_model=only_mask_accuracy_masking_model,
                          only_co_oc_mask_accuracy_masking_model=only_co_oc_mask_accuracy_masking_model, only_random_mask_accuracy_masking_model=only_random_mask_accuracy_masking_model)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        attention_mask = attention_mask.bool()
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        all_noise_locations, co_oc_mask_locations, random_mask_locations = dict_get(mask_dict, "all_noise_locations", "co_oc_mask_locations", "random_mask_locations")
        prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
        label_mlm_input_ids = label_mlm_input_ids.view(-1)
        all_noise_locations = all_noise_locations.view(-1)
        masked_lm_loss = self.loss_ce(prediction_scores[all_noise_locations], label_mlm_input_ids[all_noise_locations])
        loss = masked_lm_loss
        # All token MLM averages over all tokens and copy tokens have low average.
        backbone_divergence_loss = None
        if self.initial_backbone_parameters is not None:
            backbone_divergence_loss = ((self.initial_backbone_parameters - torch.cat([p.reshape(-1) for p in self.backbone.parameters()])) ** 2).mean()
            loss = loss + self.weight_decay * backbone_divergence_loss
            backbone_divergence_loss = backbone_divergence_loss.item()

        val_stats = dict()
        if validation_iter:
            random_mask_locations = random_mask_locations.view(-1)
            co_oc_mask_locations = co_oc_mask_locations.view(-1)
            am_sum = attention_mask.sum()
            mask_proportion = (all_noise_locations.sum() / am_sum).item()
            co_oc_mask_proportion = (co_oc_mask_locations.sum() / am_sum).item()
            random_mask_proportion = (random_mask_locations.sum() / am_sum).item()

            lm_predictions = prediction_scores.detach().argmax(dim=-1)
            lm_accuracy = (lm_predictions == label_mlm_input_ids).float()
            mlm_accuracy = lm_accuracy[all_noise_locations].mean().item()
            co_oc_mask_lm_accuracy = lm_accuracy[co_oc_mask_locations].mean().item()
            random_mask_lm_accuracy = lm_accuracy[random_mask_locations].mean().item()

            copy_token_lm_accuracy = lm_accuracy[torch.logical_and(torch.logical_not(all_noise_locations), label_mlm_input_ids != self.tokenizer.pad_token_id)].mean().item()
            val_stats = dict(mask_proportion=mask_proportion, mlm_accuracy=mlm_accuracy, copy_token_lm_accuracy=copy_token_lm_accuracy,
                             co_oc_mask_lm_accuracy=co_oc_mask_lm_accuracy, random_mask_lm_accuracy=random_mask_lm_accuracy,
                             co_oc_mask_proportion=co_oc_mask_proportion, random_mask_proportion=random_mask_proportion)

        return dict(loss=loss, backbone_divergence_loss=backbone_divergence_loss,
                    masked_lm_loss=masked_lm_loss.item(), **val_stats, **mask_stats)



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

    seed = 39529837
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
    # print("[Train]: Time = %s, Initializing Dataloader with dataset args = %s" % (dataset_args, get_time_string()))

    kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else dict()
    dataset = MaskedLanguageSentenceOrderModelDataset(**dataset_args)

    weights = None
    sm_wr = 2
    if sampling_column is not None:
        bigram_tfidf_average_column = "bigram_tfidf_average" if "bigram_tfidf_average" in dataset.dataset.column_names else "tfidf_average"
        if "sbert" in sampling_column and "perplexity" in sampling_column and "tfidf" in sampling_column:
            ppl = np.log1p(dataset["perplexity"])
            ppl = sm_wr + ((ppl - ppl.mean()) / ppl.std()).clip(-sm_wr, sm_wr) + 1
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            sbert = sm_wr + ((sbert - sbert.mean()) / sbert.std()).clip(-sm_wr, sm_wr) + 1
            tfidf_top = np.array(dataset["tfidf_average"])
            tfidf_top = sm_wr + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-sm_wr, sm_wr) + 1
            bigram_tfidf_average = np.array(dataset[bigram_tfidf_average_column])
            bigram_tfidf_average = sm_wr + ((bigram_tfidf_average - bigram_tfidf_average.mean()) / bigram_tfidf_average.std()).clip(-sm_wr, sm_wr) + 1
            weights = (ppl + sbert + ((tfidf_top + bigram_tfidf_average)/2.0)) / 3.0
        elif "sbert" in sampling_column and "perplexity" in sampling_column:
            ppl = np.log1p(dataset["perplexity"])
            ppl = sm_wr + ((ppl - ppl.mean()) / ppl.std()).clip(-sm_wr, sm_wr) + 1
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            sbert = sm_wr + ((sbert - sbert.mean()) / sbert.std()).clip(-sm_wr, sm_wr) + 1
            weights = (ppl + sbert) / 2.0
        elif sampling_column == "perplexity":
            ppl = np.log1p(dataset["perplexity"])
            weights = sm_wr + ((ppl - ppl.mean()) / ppl.std()).clip(-sm_wr, sm_wr) + 1
        elif "tfidf" in sampling_column:
            tfidf_top = np.array(dataset["tfidf_average"])
            tfidf_top = sm_wr + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-sm_wr, sm_wr) + 1
            bigram_tfidf_average = np.array(dataset[bigram_tfidf_average_column])
            bigram_tfidf_average = sm_wr + ((bigram_tfidf_average - bigram_tfidf_average.mean()) / bigram_tfidf_average.std()).clip(-sm_wr, sm_wr) + 1
            weights = tfidf_top + bigram_tfidf_average
        elif "tfidf_truncated" in sampling_column:
            tfidf_top = np.array(dataset["tfidf_truncated_average"])
            mean = tfidf_top[~np.isnan(tfidf_top)].mean()
            tfidf_top[np.isnan(tfidf_top)] = mean
            weights = sm_wr + ((tfidf_top - tfidf_top.mean()) / tfidf_top.std()).clip(-sm_wr, sm_wr) + 1
        elif "sbert" in sampling_column:
            sbert = 1 - np.array(dataset["sbert_top_128_avg"])
            weights = sm_wr + ((sbert - sbert.mean()) / sbert.std()).clip(-sm_wr, sm_wr) + 1
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
    set_seeds(args["seed"])
    batch_size = 8

    optimizer_config["lr"] = args["lr"]
    optimizer_config["weight_decay"] = args["weight_decay"]
    optimizer_config["gradient_clipping"] = args["gradient_clipping"]
    optimizer_config["beta_1"] = args["beta_1"]
    optimizer_config["beta_2"] = args["beta_2"]
    optimizer_config["eps"] = 1e-6

    reinit = args["pretrained_model"] is None or "pretrained_model" not in args or args["pretrained_model"] == ""
    optimizer_config["weight_decay"] = optimizer_config["weight_decay"]  # if reinit else 0.0
    backbone, tokenizer, lm_head = get_backbone(args["model_config"], reinit, dropout_prob=0.0)
    batch_size = args["batch_size"] if "batch_size" in args and isinstance(args["batch_size"], int) else batch_size
    mlm_w = args["mlm_w"] if "mlm_w" in args else 1.0
    sentence_order_w = args["sentence_order_w"] if "sentence_order_w" in args else 1.0

    if isinstance(backbone, (CoOccurenceModel, MixerCoOccurenceModel)):
        model = backbone.to(device)
        trainable_model = model
        mlm_sop_enabled = False
    elif args["hard_mlm"]:
        masking_model, _ , _ = get_backbone(args["hard_mlm_model"].split("/")[-1], False, dropout_prob=0.0)
        state_dict = torch.load(args["hard_mlm_model"], map_location='cpu')
        masking_model.load_state_dict(state_dict, strict=True)
        masking_model = masking_model.eval()
        if hasattr(masking_model, "model"):
            masking_model.model = None
            del masking_model.model
        model = RTDMLMModel(backbone, lm_head, masking_model, tokenizer, mlm_w, sentence_order_w, args["weight_decay"], reinit).to(device)
        trainable_model = model.backbone
        if hasattr(model, "initial_backbone_parameters") and model.initial_backbone_parameters is not None:
            model.initial_backbone_parameters = model.initial_backbone_parameters.to(device)
        mlm_sop_enabled = False
    else:
        model = MaskedLanguageSentenceOrderModel(backbone, tokenizer, mlm_w, sentence_order_w, reinit).to(device)
        trainable_model = model.backbone
        mlm_sop_enabled = True
    if local_rank == 0 and rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s, reinit = %s" % (get_time_string(), numel(model) / 1_000_000, reinit))

    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]):
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)
        trainable_model.load_state_dict(state_dict, strict=False)
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
    div_factor = optc["lr"]/1e-9
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], total_steps=args["total_steps"],
                                                    div_factor=div_factor, three_phase=False, pct_start=0.04, anneal_strategy="linear", cycle_momentum=False)

    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, int(0.06 * args["total_steps"]))
    barrier()

    gradient_clipping = optc["gradient_clipping"]

    group = "%s-%s-%sN-%s" % (args["wandb_name"], args["model_config"], args["nodes"], time_string)
    wandb_init_args = dict(project="fnd", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                           config={"args":args, "optimizer_config": optc},
                           settings=wandb.Settings(start_method="fork"))

    time.sleep(random.random())
    activate_wandb_log = local_rank <= (8 // args["world_size"]) or args["world_size"] <= 8
    if activate_wandb_log:
        wandb.init(**wandb_init_args)

    full_times = []
    batch_times = []
    model_times = []
    logs_save = []
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
        random.seed(args["seed"] + step)
        batch = dataloader()

        if hasattr(getattr(model, "module", model), "distillation_loss_w"):
            getattr(model, "module", model).distillation_loss_w = np.interp(steps_done,
                                                                            [0, 10, total_steps // 5, total_steps // 4],
                                                                            [1.0, 0.0, 0.0, 1.0])
            if steps_done == (total_steps // 5):
                momentum_param_copy(getattr(model, "module", model).backbone,
                                    getattr(model, "module", model).momentum_backbone, 0.0)
                momentum_param_copy(getattr(model, "module", model).lm_head,
                                    getattr(model, "module", model).momentum_lm_head, 0.0)


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
        printable_iter = (step + 1) % (4 * log_every_steps) == 0
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
            model.zero_grad(set_to_none=True)
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
            wandb_log = {k: float(v) for k, v in wandb_log.items()}
            logs_save.append(pd.DataFrame.from_records([wandb_log]).T)
            if activate_wandb_log:
                time.sleep(random.random() * 0.1)
                wandb.log(wandb_log)
            if printable_iter:
                printed = pd.concat(logs_save, axis=1)
                printed["mean"] = printed.mean(1)
                logs_save = []
                if local_rank == 0:
                    # print(json.dumps(dict(time=get_time_string(), **wandb_log), skipkeys=True, indent=2, sort_keys=True))
                    print("-" * 80)
                    print(printed)
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



