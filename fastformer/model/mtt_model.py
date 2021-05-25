import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import traceback


from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast

from fairscale.nn.misc import checkpoint_wrapper
from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap

import numpy as np
import math
import random
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import AutoModel, RobertaTokenizerFast

from fastformer.model.fastformer_vision_model import PatchCLR

try:
    from performer_pytorch import SelfAttention, FastAttention
except:
    pass
from collections import defaultdict
from torch.nn import TransformerDecoder
import time

from torch.utils.data import DataLoader

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fastformer.data import *
from fastformer.model.AdMSLoss import AdMSoftmaxLoss, BCELossFocal
from fastformer.model.lib import *
from fastformer.utils import *
from fastformer.model import FastFormerPreTrainedModel


def get_mtt_backbone(model_name, cls_tokens, reinit=False):
    model = AutoModel.from_pretrained(model_name)
    vocab_size, dims = model.embeddings.word_embeddings.weight.size(0), model.embeddings.word_embeddings.weight.size(1)
    if reinit:
        model.init_weights()

    if "roberta" in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif "bert" in model_name:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


class MTTModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, tokenizer, num_features=768, cls_tokens=1,
                 generator_w=0.0, discriminator_w=0.0, dino_w=1.0, sentence_order_prediction_w=1.0, input_cls_orthogonal_w=0.1,
                 dropout=0.1,
                 reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.cls_tokens = cls_tokens
        self.backbone = backbone
        self.pad_token_id = tokenizer.pad_token_id
        self.ce = CrossEntropyLoss(ignore_index=-100)
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.ignore_zero_ce = CrossEntropyLoss(ignore_index=0)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.tokenizer = tokenizer
        self.dino_dims = 2 ** 16
        norm_last_layer = True
        bottleneck_dim = 256
        self.input_cls_orthogonal_w = input_cls_orthogonal_w
        assert generator_w > 0 or dino_w > 0
        if discriminator_w > 0:
            assert generator_w > 0
        self.generator_w = generator_w
        self.discriminator_w = discriminator_w
        self.dino_w = dino_w
        self.vocab_size = self.backbone.embeddings.word_embeddings.weight.size(0) + (cls_tokens - 1)
        self.lm_head = nn.Linear(num_features, self.vocab_size)
        self.sentence_order_prediction_w = sentence_order_prediction_w
        if reinit:
            self.init_weights()

        if dino_w > 0:
            last_layer = nn.Linear(bottleneck_dim, self.dino_dims, bias=False)
            init_weights(last_layer, 0.02)
            last_layer = nn.utils.weight_norm(last_layer)
            last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                last_layer.weight_g.requires_grad = False

            self.ffn = nn.Sequential(nn.Linear(num_features, 2048), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(2048, 2048), nn.GELU(),
                                     nn.Linear(2048, bottleneck_dim), nn.GELU(),
                                     Norm(),
                                     last_layer)  # weight_norm
            init_weights(self.ffn[0], 0.02)
            init_weights(self.ffn[2], 0.02)
            init_weights(self.ffn[3], 0.02)
            last_layer.weight_g.data.fill_(1)

        if generator_w > 0:
            self.generator_ffn = nn.Sequential(nn.LayerNorm(num_features),
                                               nn.Dropout(dropout),
                                               nn.Linear(num_features, num_features * 2),
                                               nn.GELU(),
                                               nn.Linear(num_features * 2, num_features),  # nn.Tanh()
                                               )
            init_weights(self.generator_ffn, 0.01)

        if discriminator_w > 0:
            self.discriminator_ffn = nn.Sequential(nn.LayerNorm(num_features),
                                                   nn.Dropout(dropout),
                                                   nn.Linear(num_features, num_features * 2),
                                                   nn.GELU(),
                                                   nn.Linear(num_features * 2, 1))

            init_weights(self.discriminator_ffn, 0.01)

        if sentence_order_prediction_w > 0:
            self.sent_order_nn = nn.Sequential(nn.Linear(num_features, num_features),
                                               nn.Dropout(dropout),
                                               nn.GELU(),
                                               nn.Linear(num_features, 1))
            init_weights(self.sent_order_nn, 0.01)

    def get_input_embeddings(self):
        return self.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.backbone.embeddings.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            labels_segment_index=None,
            char_ids=None, char_offsets=None,
    ):
        backbone_inputs = dict(input_ids=input_ids, attention_mask=attention_mask, char_ids=char_ids, char_offsets=char_offsets, output_hidden_states=True)
        backbone_inputs = {k: v for k, v in backbone_inputs.items() if v is not None}
        outputs = self.backbone(**backbone_inputs)
        masked_lm_loss = None
        lm_accuracy = None
        discriminator_label_mean = None
        discriminator_loss = None
        dino = None
        discriminator_accuracy = None
        discriminator_positive_accuracy = None
        discriminator_negative_accuracy = None
        sent_order_accuracy = None
        sent_order_loss = None
        input_cls_orthogonal_loss = None
        b = input_ids.size(0)

        if self.input_cls_orthogonal_w > 0 and self.training and self.cls_tokens > 1:
            inputs_embeds_cls = outputs["hidden_states"][-12][:, :self.cls_tokens]
            inputs_embeds_cls = inputs_embeds_cls / (inputs_embeds_cls.norm(2, -1, True).detach() + self.config.layer_norm_eps)
            inputs_embeds_cls = inputs_embeds_cls.bmm(inputs_embeds_cls.transpose(1, 2))
            input_cls_orthogonal_loss = self.input_cls_orthogonal_w * (inputs_embeds_cls ** 2).mean()

        if self.sentence_order_prediction_w and labels_segment_index is not None:
            labels_segment_index = labels_segment_index.float()
            sent_order_logits = self.sent_order_nn(outputs["hidden_states"][-1][:, self.cls_tokens - 1]).squeeze(-1)
            sent_order_loss = self.sentence_order_prediction_w * self.loss_bce(sent_order_logits, labels_segment_index)
            sent_order_preds = (torch.sigmoid(sent_order_logits.detach()) > 0.5).type(torch.float)
            sent_order_accuracy = (sent_order_preds == labels_segment_index).float().mean()

        if self.dino_w > 0:
            dino = self.ffn(outputs["pooler_output"] if "pooler_output" in outputs else outputs["hidden_states"][-1][:, 0])

        if (self.generator_w > 0 or self.discriminator_w > 0) and labels is not None:
            generator_output = self.generator_ffn(outputs["hidden_states"][-6 if self.discriminator_w > 0 else -1][:, self.cls_tokens - 1:])
            lm_logits = self.lm_head(generator_output)
            new_input_ids = lm_logits.detach().argmax(dim=-1)

            if self.generator_w > 0:
                active_labels = labels.reshape(-1)
                active_prediction_logits = lm_logits.reshape(-1, self.vocab_size)
                masked_lm_loss = self.generator_w * self.loss_ce(active_prediction_logits, active_labels)

            lm_accuracy = (new_input_ids == labels).float().mean().item()
            if self.discriminator_w > 0:
                # TODO: Gradually sample more from our lm
                # TODO: sample from lm such that we sample high confident samples which are wrong.
                tol = min(0.75 - lm_accuracy, 0) / (1 - lm_accuracy)
                mask = (torch.randn(new_input_ids.shape[:2], device=new_input_ids.device) >= tol).type(new_input_ids.dtype)
                new_input_ids = new_input_ids * mask + (1 - mask) * labels
                discriminator_labels = (new_input_ids == labels).float()
                discriminator_label_mean = discriminator_labels.mean()
                discriminator_outputs = self.backbone(input_ids=new_input_ids, attention_mask=attention_mask[:, self.cls_tokens - 1:], output_hidden_states=True)["hidden_states"][-1]
                discriminator_outputs = self.discriminator_ffn(discriminator_outputs)

                discriminator_outputs = discriminator_outputs.squeeze(-1).reshape(b, -1)
                discriminator_labels = discriminator_labels.reshape(b, -1)
                discriminator_loss = self.discriminator_w * self.loss_bce(discriminator_outputs, discriminator_labels)
                discriminator_preds = (torch.sigmoid(discriminator_outputs.detach()) > 0.5).type(torch.float)
                sample_accuracies = (discriminator_preds == discriminator_labels).type(torch.float)
                discriminator_labels = discriminator_labels.bool()
                discriminator_positive_accuracy = sample_accuracies[discriminator_labels].mean().item()
                discriminator_negative_accuracy = sample_accuracies[torch.logical_not(discriminator_labels)].mean().item()
                discriminator_accuracy = torch.mean(sample_accuracies).item()


        return dict(masked_lm_loss=masked_lm_loss, lm_accuracy=lm_accuracy, dino=dino, discriminator_accuracy=discriminator_accuracy, sent_order_accuracy=sent_order_accuracy,
                    discriminator_label_mean=discriminator_label_mean, discriminator_loss=discriminator_loss, sent_order_loss=sent_order_loss, input_cls_orthogonal_loss=input_cls_orthogonal_loss,
                    discriminator_positive_accuracy=discriminator_positive_accuracy, discriminator_negative_accuracy=discriminator_negative_accuracy)


class MultiTaskHighwayCLSPretraining(PatchCLR):
    def __init__(self, student: MTTModel, teacher: MTTModel, eps=1e-7):
        super().__init__(student, teacher, eps, 0.04, 0.1, 0.9)
        self.cls_tokens = student.cls_tokens
        self.dino_dims = student.dino_dims
        self.student = student
        self.teacher = teacher
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.generator_w = 0.0
        teacher.discriminator_w = 0.0
        teacher.sentence_order_prediction_w = 0.0
        teacher.input_cls_orthogonal_w = 0.0
        self.generator_w = student.generator_w
        self.discriminator_w = student.discriminator_w

        self.eps = eps
        self.teacher_contrastive_temperature = 0.04
        self.student_contrastive_temperature = 0.1
        self.dino_w = student.dino_w
        self.dino_cw = 0.9

    def __call__(
            self,
            input_ids,
            attention_mask,
            labels=None,
            labels_segment_index=None,
            char_ids=None, char_offsets=None,

            input_ids_teacher=None,
            attention_mask_teacher=None,
            char_ids_teacher=None, char_offsets_teacher=None,
            dino_center=None,
    ):
        student_rep = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, labels_segment_index=labels_segment_index,
                                   char_ids=char_ids, char_offsets=char_offsets)
        with torch.no_grad():
            teacher_rep = self.teacher(input_ids=input_ids_teacher, attention_mask=attention_mask_teacher,
                                       char_ids=char_ids_teacher, char_offsets=char_offsets_teacher)
        dino_loss = None
        losses = [v for k, v in student_rep.items() if "_loss" in k and v is not None]
        loss = sum(losses)
        if self.dino_w > 0:
            dino_results = self.dino_loss(student_rep.pop("dino").unsqueeze(0), teacher_rep.pop("dino").detach().unsqueeze(0), dino_center, 1, 1)
            dino_center = dino_results["dino_center"]
            dino_loss = dino_results["dino_loss"]
            loss += dino_loss
        return dict(loss=loss, dino_center=dino_center, dino_loss=dino_loss, **student_rep)



