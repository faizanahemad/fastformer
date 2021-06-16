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
from transformers import AutoModel, RobertaTokenizerFast, ConvBertTokenizer, ConvBertTokenizerFast, RobertaTokenizer, RobertaConfig, RobertaModel

from fastformer.model.fast_convbert import ConvBertModel, ConvBertConfig
from fastformer.model.fastformer_vision_model import PatchCLR
from fastformer.model.roberta_prenorm import PreNormRobertaModel


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

def log(t, eps=1e-8):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.0):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


def temperature_sampling(logits, temperature=1.0):
    if temperature is None or temperature == 0.0:
        return torch.argmax(logits)
    probs = F.softmax(logits / temperature)
    pred_ids = probs.view(-1, probs.size(-1)).multinomial(1, replacement=False)
    if logits.ndim == 3:
        pred_ids = pred_ids.view(*probs.shape[:2])
    return pred_ids


def get_mtt_backbone(model_name, cls_tokens, approximate_unused_layers, sampling_alpha, reinit=False):
    # TODO: Later also add a QnA boolean / fixed number of options question
    # TODO: Add extra CLS attr and tokens in embedding

    if "prenorm-roberta" in model_name:
        if "large" in model_name:
            model_name = "roberta-large"
        else:
            model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        config = RobertaConfig.from_pretrained(model_name)
        # config.gradient_checkpointing = True
        # config.vocab_size = 30522
        config.approximate_unused_layers = approximate_unused_layers
        if sampling_alpha is not None:
            config.sampling_alpha = sampling_alpha
        model = PreNormRobertaModel(config)
    elif "roberta" in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig.from_pretrained(model_name)
        # config.gradient_checkpointing = True
        model = RobertaModel(config)
    elif "bert" in model_name:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained(model_name)
    elif "fast-conv" in model_name:
        # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # tokenizer = ConvBertTokenizerFast.from_pretrained("YituTech/conv-bert-base")
        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer = ConvBertTokenizer.from_pretrained("YituTech/conv-bert-base")  # 30522, 50265
        model = ConvBertModel(ConvBertConfig(vocab_size=30522))
    elif "conv" in model_name:
        tokenizer = ConvBertTokenizer.from_pretrained("YituTech/conv-bert-base")
        model = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    if reinit:
        model.init_weights()

    with torch.no_grad():
        if cls_tokens > 1:
            std = model.embeddings.word_embeddings.weight.std()
            dims = model.embeddings.word_embeddings.weight.shape[1]
            mean = model.embeddings.word_embeddings.weight.mean()
            extras = nn.Parameter(torch.randn(cls_tokens - 1, dims) * std + mean)
            extras.requires_grad = True
            model.embeddings.word_embeddings.weight = nn.Parameter(torch.cat((model.embeddings.word_embeddings.weight, extras)))
            setattr(model, "cls_tokens", cls_tokens)
            model.embeddings.word_embeddings.weight.requires_grad = True
    return model, tokenizer


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, hidden_dropout, layer_norm_eps):
        super().__init__()

        d_inner = d_model * 4
        self.d_model = d_model
        activation_dropout = Dropout(hidden_dropout)

        self.layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_inner, bias=True), nn.GELU(), activation_dropout, nn.Linear(d_inner, d_model, bias=False))

    def forward(self, hidden):
        h = self.layer_norm(hidden)
        h = self.ffn(h)
        h = hidden + h
        return h


class MTTModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, tokenizer, cls_tokens=1,
                 generator_w=0.0, discriminator_w=0.0, dino_w=1.0, sentence_order_prediction_w=1.0,
                 attention_penalty_w=0.0,
                 dropout=0.1, lm_layers=4, electra_layers=8, lm_layers_total=6, electra_layers_total=12,
                 drop_unused_layers=None, approximate_unused_layers=None, exclude_layers=None, keep_last_layer=False,
                 reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.cls_tokens = cls_tokens
        self.checkpointing = getattr(backbone.config, "gradient_checkpointing", False) if hasattr(backbone, "config") else False
        self.backbone = backbone
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.tokenizer = tokenizer
        self.dino_dims = 2 ** 12
        norm_last_layer = True
        self.exclude_layers = exclude_layers
        bottleneck_dim = 256
        self.n_layers = len(self.backbone.encoder.layer)
        self.sampling_alpha = getattr(self.config, "sampling_alpha", 1.0)
        self.attention_penalty_w = attention_penalty_w
        self.lm_layers = lm_layers
        self.electra_layers = electra_layers
        self.lm_layers_total = lm_layers_total
        self.electra_layers_total = electra_layers_total
        self.layer_ratio = int(self.electra_layers_total / self.lm_layers_total)
        self.drop_unused_layers = drop_unused_layers
        self.approximate_unused_layers = approximate_unused_layers
        self.keep_last_layer = keep_last_layer
        self.start_from_proba = 0.0
        assert drop_unused_layers is None or approximate_unused_layers is None or (approximate_unused_layers ^ drop_unused_layers) or (not drop_unused_layers and not approximate_unused_layers)
        if attention_penalty_w > 0:
            attention_penalty = get_rolling_diagonal_weights(tokenizer.model_max_length, 
                                                             backbone.config.conv_kernel_size if hasattr(backbone.config, "conv_kernel_size") else 9)
            attention_penalty.requires_grad = False
            self.register_buffer("attention_penalty", attention_penalty)
        assert generator_w > 0 or dino_w > 0

        self.generator_w = generator_w
        self.discriminator_w = discriminator_w
        self.dino_w = dino_w
        self.vocab_size = self.backbone.embeddings.word_embeddings.weight.size(0)
        embedding_dims = self.backbone.embeddings.word_embeddings.weight.size(1)
        num_features = self.backbone.embeddings.position_embeddings.weight.size(1)
        self.sentence_order_prediction_w = sentence_order_prediction_w
        if self.generator_w > 0 or self.discriminator_w > 0:
            self.lm_head = nn.Linear(embedding_dims, self.vocab_size, bias=False)
            self.tie_weights()
        if reinit:
            self.init_weights()

        if dino_w > 0:
            last_layer = nn.Linear(bottleneck_dim, self.dino_dims, bias=False)
            init_weights(last_layer, 0.02)
            last_layer = nn.utils.weight_norm(last_layer)
            last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                last_layer.weight_g.requires_grad = False

            self.ffn = nn.Sequential(nn.Linear(num_features, num_features), nn.GELU(),
                                     nn.Linear(num_features, bottleneck_dim),
                                     Norm(),
                                     last_layer)  # weight_norm
            init_weights(self.ffn[0], 0.02)
            init_weights(self.ffn[2], 0.02)
            last_layer.weight_g.data.fill_(1)

        if generator_w > 0:
            self.generator_ffn = nn.Sequential(nn.Linear(num_features, num_features),
                                               nn.GELU(),
                                               nn.Linear(num_features, num_features),  # nn.Tanh()
                                               )
            init_weights(self.generator_ffn, 0.01)

        if discriminator_w > 0:
            self.discriminator_ffn = nn.Sequential(nn.Linear(num_features, num_features),
                                                   nn.GELU(),
                                                   nn.Linear(num_features, 1))

            init_weights(self.discriminator_ffn, 0.01)

        if sentence_order_prediction_w > 0:
            self.sent_order_nn = nn.Sequential(nn.Linear(num_features, num_features),
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
            rng_seed=None, num_layers_lm=None, num_layers_total_lm=None,
            num_layers_electra=None, num_layers_total_electra=None,
            discriminator_inputs=None,
    ):
        rng_seed = None
        g_cpu = None
        if rng_seed is not None:
            g_cpu = torch.Generator()
            g_cpu = g_cpu.manual_seed(rng_seed)
        gen = np.random.default_rng(rng_seed) if rng_seed is not None else random
        start_from_proba = gen.random() < self.start_from_proba
        backbone_inputs = dict(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True, output_attentions=self.attention_penalty_w > 0,
                               )
        start_sampling_from_lm = None
        start_sampling_from_electra = None
        lm_layers = None
        electra_layers = None
        electra_layers_total = None
        lm_layers_total = None
        no_grad_embedding = False
        if isinstance(self.backbone, PreNormRobertaModel):
            if self.lm_layers is not None or num_layers_lm is not None:
                backbone_inputs["num_layers"] = self.lm_layers if num_layers_lm is None else num_layers_lm
                lm_layers = backbone_inputs["num_layers"]
                backbone_inputs["rng_seed"] = rng_seed
            if self.lm_layers_total is not None or num_layers_total_lm is not None:
                backbone_inputs["num_layers_total"] = self.lm_layers_total if num_layers_total_lm is None else num_layers_total_lm
                lm_layers_total = backbone_inputs["num_layers_total"]
            backbone_inputs["drop_unused_layers"] = self.drop_unused_layers
            backbone_inputs["approximate_unused_layers"] = self.approximate_unused_layers
            backbone_inputs["start_sampling_from"] = 0
            if self.training and start_from_proba and lm_layers is not None and torch.is_grad_enabled():
                lm_layers_total = lm_layers_total if lm_layers_total is not None else self.n_layers
                probas = torch.tensor([((lm_layers_total - i) / lm_layers_total) if (i < lm_layers_total - lm_layers) else 0.0 for i in
                                       range(lm_layers_total)]) ** self.sampling_alpha
                start_sampling_from_lm = torch.multinomial(probas, 1, replacement=False, generator=g_cpu).long().item()
                backbone_inputs["start_sampling_from"] = start_sampling_from_lm
            backbone_inputs["keep_last_layer"] = self.keep_last_layer
        backbone_inputs = {k: v for k, v in backbone_inputs.items() if v is not None}
        outputs = self.backbone(**backbone_inputs)
        approx_loss = outputs["approx_loss"] if "approx_loss" in outputs else None
        sent_hidden = outputs["pooler_output"] if "pooler_output" in outputs else outputs["hidden_states"][-1][:, 0]
        exclude_layers = outputs["selected_layers"] if "selected_layers" in outputs else []
        n_grad_forward_layers_lm = outputs["n_grad_forward_layers"] if "n_grad_forward_layers" in outputs else None
        n_forward_layers_lm = outputs["n_forward_layers"] if "n_forward_layers" in outputs else None
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
        input_cls_orthogonal = None
        lm_input_accuracy = None
        discriminator_extra_accuracy = None
        mask_indices_mean = None
        masked_accuracy = None
        attention_penalty_loss = None
        discriminator_dino = None
        active_locations = attention_mask.bool()

        if self.attention_penalty_w > 0:
            attentions = outputs["attentions"]
            # attentions = sum([a/(len(attentions) - i) for i, a in enumerate(attentions)]).mean(0).mean(0)
            attentions = sum([a for i, a in enumerate(attentions[1:-1])]).mean(0).mean(0)
            penalty = self.attention_penalty[:attentions.size(0), :attentions.size(1)]
            attention_penalty_loss = self.attention_penalty_w * attentions[penalty != 0].mean()

        if self.cls_tokens > 1:
            inputs_embeds_cls = outputs["hidden_states"][0][:, :self.cls_tokens].detach()
            inputs_embeds_cls = inputs_embeds_cls / (inputs_embeds_cls.norm(2, -1, True) + self.config.layer_norm_eps)
            inputs_embeds_cls = inputs_embeds_cls.bmm(inputs_embeds_cls.transpose(1, 2))
            inputs_embeds_cls = inputs_embeds_cls * (1 - torch.eye(inputs_embeds_cls.size(-1), device=inputs_embeds_cls.device).unsqueeze(0))
            input_cls_orthogonal = ((inputs_embeds_cls ** 2) ** 0.5).mean()

        if self.dino_w > 0:
            dino_hidden = outputs["hidden_states"][-1][:, self.cls_tokens - 1]
            dino = self.ffn(dino_hidden)

        if self.generator_w > 0 or self.discriminator_w > 0 or discriminator_inputs is not None:
            if self.generator_w > 0 or discriminator_inputs is None:
                mask_indices = (input_ids.int() != labels.int())
                mask_indices_mean = mask_indices[active_locations].float().mean().item()
                lm_input_accuracy = (input_ids == labels)[active_locations].type(torch.int32).float().mean().item()
                generator_output = outputs["hidden_states"][-7 if self.discriminator_w > 0 and lm_layers is None else -1]
                generator_output = self.generator_ffn(generator_output)
                if hasattr(self.backbone, "embeddings_project"):
                    generator_output = self.backbone.embeddings_reverse_project(generator_output)
                lm_mask = mask_indices.unsqueeze(-1).expand(-1, -1, generator_output.size(-1))
                if no_grad_embedding:
                    self.lm_head.requires_grad_(False)
                lm_logits = self.lm_head(generator_output[lm_mask].reshape(-1, generator_output.size(-1)))
                if no_grad_embedding:
                    self.lm_head.requires_grad_(True)

            if self.generator_w > 0 and labels is not None:
                active_labels = labels[mask_indices].reshape(-1)
                active_prediction_logits = lm_logits.reshape(-1, self.vocab_size)
                masked_lm_loss = 0.0
                if active_prediction_logits.ndim > 1 and active_prediction_logits.shape[0] > 0 and active_prediction_logits.shape[1] > 0:
                    masked_lm_loss = self.generator_w * self.loss_ce(active_prediction_logits, active_labels)
                    masked_accuracy = (active_prediction_logits.detach().argmax(dim=-1) == active_labels).type(active_prediction_logits.dtype).mean().item()

            if self.discriminator_w > 0 or discriminator_inputs is not None:
                if discriminator_inputs is None:
                    new_input_ids = input_ids.clone()
                    new_input_ids[mask_indices] = temperature_sampling(lm_logits.detach()).view(-1)
                    discriminator_labels = (new_input_ids.int() == labels.int()).type(lm_logits.dtype)
                    discriminator_inputs = dict(input_ids=new_input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    discriminator_labels = discriminator_labels[active_locations].reshape(-1)
                    discriminator_inputs["drop_unused_layers"] = self.drop_unused_layers
                    discriminator_inputs["approximate_unused_layers"] = self.approximate_unused_layers
                    discriminator_inputs["start_sampling_from"] = 0
                else:
                    discriminator_labels = discriminator_inputs.pop("discriminator_labels")
                if isinstance(self.backbone, PreNormRobertaModel):
                    if self.electra_layers is not None or num_layers_electra is not None:
                        discriminator_inputs["num_layers"] = self.electra_layers if num_layers_electra is None else num_layers_electra
                        electra_layers = discriminator_inputs["num_layers"]
                        discriminator_inputs["rng_seed"] = rng_seed
                    if self.electra_layers_total is not None or num_layers_total_electra is not None:
                        discriminator_inputs["num_layers_total"] = self.electra_layers_total if num_layers_total_electra is None else num_layers_total_electra
                    discriminator_inputs["drop_unused_layers"] = self.drop_unused_layers
                    discriminator_inputs["approximate_unused_layers"] = self.approximate_unused_layers
                    discriminator_inputs["start_sampling_from"] = 0
                    if self.training and start_from_proba and electra_layers is not None and torch.is_grad_enabled():
                        electra_layers_total = electra_layers_total if electra_layers_total is not None else self.n_layers
                        probas = torch.tensor([((electra_layers_total - i) / electra_layers_total) if (i < electra_layers_total - electra_layers) else 0.0 for i in
                                               range(electra_layers_total)]) ** self.sampling_alpha
                        start_sampling_from_electra = torch.multinomial(probas, 1, replacement=False, generator=g_cpu).long().item()
                        discriminator_inputs["start_sampling_from"] = start_sampling_from_electra
                    if self.exclude_layers:
                        discriminator_inputs["exclude_layers"] = exclude_layers
                    discriminator_inputs["keep_last_layer"] = self.keep_last_layer
                discriminator_inputs["output_hidden_states"] = True
                discriminator_outputs = self.backbone(**discriminator_inputs)
                n_grad_forward_layers_electra = discriminator_outputs["n_grad_forward_layers"] if "n_grad_forward_layers" in discriminator_outputs else None
                n_forward_layers_electra = discriminator_outputs["n_forward_layers"] if "n_forward_layers" in discriminator_outputs else None
                disc_approx_loss = discriminator_outputs["approx_loss"] if "approx_loss" in discriminator_outputs else None
                discriminator_outputs = discriminator_outputs["hidden_states"][-1]
                approx_loss = (approx_loss + disc_approx_loss) if approx_loss is not None else disc_approx_loss
                _ = discriminator_inputs.pop("num_layers", None)
                _ = discriminator_inputs.pop("num_layers_total", None)
                _ = discriminator_inputs.pop("rng_seed", None)
                _ = discriminator_inputs.pop("output_hidden_states", None)
                _ = discriminator_inputs.pop("start_sampling_from", None)
                _ = discriminator_inputs.pop("exclude_layers", None)

                if self.dino_w > 0:
                    discriminator_dino = self.ffn(discriminator_outputs[:, self.cls_tokens - 1])

                sent_hidden = discriminator_outputs[:, 0]
                if self.discriminator_w > 0:
                    discriminator_outputs = self.discriminator_ffn(discriminator_outputs)

                    discriminator_outputs = discriminator_outputs.squeeze(-1)[active_locations].reshape(-1)
                    discriminator_inputs["discriminator_labels"] = discriminator_labels
                    discriminator_label_mean = discriminator_labels.mean()
                    discriminator_loss = self.discriminator_w * self.loss_bce(discriminator_outputs, discriminator_labels)
                    discriminator_preds = (torch.sigmoid(discriminator_outputs.detach()) > 0.5).type(discriminator_outputs.dtype)
                    sample_accuracies = (discriminator_preds == discriminator_labels).type(discriminator_preds.dtype)
                    discriminator_labels = discriminator_labels.bool()
                    discriminator_positive_accuracy = sample_accuracies[discriminator_labels].mean().item()
                    discriminator_negative_accuracy = sample_accuracies[torch.logical_not(discriminator_labels)].mean().item()
                    discriminator_accuracy = torch.mean(sample_accuracies).item()
                    discriminator_extra_accuracy = max(0.0, float((discriminator_accuracy - discriminator_label_mean) / (1.0 - discriminator_label_mean)))

        if self.sentence_order_prediction_w and labels_segment_index is not None:
            labels_segment_index = labels_segment_index.float()
            sent_order_logits = self.sent_order_nn(sent_hidden).squeeze(-1)
            sent_order_loss = self.sentence_order_prediction_w * self.loss_bce(sent_order_logits, labels_segment_index)
            sent_order_preds = (torch.sigmoid(sent_order_logits.detach()) > 0.5).type(sent_order_logits.dtype)
            sent_order_accuracy = (sent_order_preds == labels_segment_index).type(sent_order_logits.dtype).mean().item()

        if approx_loss is not None:
            approx_loss = approx_loss * 0.01
        return dict(masked_lm_loss=masked_lm_loss, lm_accuracy=lm_accuracy, lm_input_accuracy=lm_input_accuracy,
                    dino=dino, discriminator_accuracy=discriminator_accuracy, sent_order_accuracy=sent_order_accuracy,
                    discriminator_extra_accuracy=discriminator_extra_accuracy, masked_accuracy=masked_accuracy,
                    discriminator_label_mean=discriminator_label_mean, discriminator_loss=discriminator_loss,
                    sent_order_loss=sent_order_loss, input_cls_orthogonal=input_cls_orthogonal, attention_penalty_loss=attention_penalty_loss,
                    discriminator_positive_accuracy=discriminator_positive_accuracy, discriminator_negative_accuracy=discriminator_negative_accuracy,
                    mask_indices_mean=mask_indices_mean, discriminator_dino=discriminator_dino, discriminator_inputs=discriminator_inputs,
                    n_grad_forward_layers_electra=n_grad_forward_layers_electra, n_forward_layers_electra=n_forward_layers_electra,
                    n_forward_layers_lm=n_forward_layers_lm, n_grad_forward_layers_lm=n_grad_forward_layers_lm,
                    approx_loss=approx_loss, start_from_proba=self.start_from_proba, sampling_alpha=self.sampling_alpha,
                    start_sampling_from_lm=start_sampling_from_lm, start_sampling_from_electra=start_sampling_from_electra,
                    electra_to_lm_layer_ratio=n_forward_layers_electra/n_forward_layers_lm)


class MultiTaskHighwayCLSPretraining(PatchCLR):
    def __init__(self, student: MTTModel, teacher: MTTModel, eps=1e-7, device=None):
        super().__init__(student, teacher, eps, 0.04, 0.1, 0.9)
        self.cls_tokens = student.cls_tokens
        self.dino_dims = student.dino_dims
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        # print([n for n, p in student.named_parameters()])
        for n, p in student.named_parameters():
            if "layer_normalizers" not in n:
                p.requires_grad = True
        teacher.generator_w = 0.0
        teacher.discriminator_w = 0.0
        teacher.sentence_order_prediction_w = 0.0
        self.generator_w = student.generator_w
        self.discriminator_w = student.discriminator_w
        self.device = device

        self.eps = eps
        self.teacher_contrastive_temperature = 0.04
        self.student_contrastive_temperature = 0.1
        self.dino_w = student.dino_w

    def __call__(
            self,
            input_ids,
            attention_mask,
            labels=None,
            labels_segment_index=None,
            input_ids_teacher=None,
            attention_mask_teacher=None,
            char_ids_teacher=None, char_offsets_teacher=None,
            dino_center=None,
            discriminator_dino_center=None,
            rng_seed=None,
            validation_iter=False,
    ):
        # TODO: Do we need to guide both students (MLM/ELECTRA)
        all_discriminator_extra_accuracy = None
        all_masked_accuracy = None
        student_rep = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, labels_segment_index=labels_segment_index,
                                   rng_seed=rng_seed)
        discriminator_inputs = student_rep.pop("discriminator_inputs", None)
        if discriminator_inputs is not None:
            _ = discriminator_inputs.pop("drop_unused_layers", None)
            _ = discriminator_inputs.pop("approximate_unused_layers", None)
        all_stats = dict()
        if validation_iter:
            with torch.no_grad():
                all_layers_accuracy = self.student(input_ids=input_ids, attention_mask=attention_mask,
                                                   labels=labels, labels_segment_index=labels_segment_index,
                                                   rng_seed=rng_seed,
                                                   num_layers_lm=getattr(self.student, "module", self.student).lm_layers_total, num_layers_total_lm=getattr(self.student, "module", self.student).lm_layers_total,
                                                   num_layers_electra=getattr(self.student, "module", self.student).electra_layers_total, num_layers_total_electra=getattr(self.student, "module", self.student).electra_layers_total,
                                                   discriminator_inputs=discriminator_inputs)
                all_stats = dict(
                    all_discriminator_extra_accuracy=all_layers_accuracy["discriminator_extra_accuracy"],
                    all_masked_accuracy=all_layers_accuracy["masked_accuracy"],
                    all_n_forward_layers_lm=all_layers_accuracy["n_forward_layers_lm"],
                    all_n_grad_forward_layers_lm=all_layers_accuracy["n_grad_forward_layers_lm"],
                    all_n_grad_forward_layers_electra=all_layers_accuracy["n_grad_forward_layers_electra"],
                    all_n_forward_layers_electra=all_layers_accuracy["n_forward_layers_electra"],
                    all_start_sampling_from_lm=all_layers_accuracy["start_sampling_from_lm"],
                    all_start_sampling_from_electra=all_layers_accuracy["start_sampling_from_electra"],
                )

        if self.dino_w > 0:
            with torch.no_grad():
                # print("teacher layers = ", self.teacher.lm_layers_total, self.teacher.electra_layers_total)
                teacher = self.teacher  # .eval()
                if self.device is not None:
                    teacher = self.teacher.to(self.device)
                teacher_rep = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                      num_layers_lm=self.teacher.lm_layers_total, num_layers_total_lm=self.teacher.lm_layers_total,
                                      num_layers_electra=self.teacher.electra_layers_total,
                                      num_layers_total_electra=self.teacher.electra_layers_total,
                                      discriminator_inputs=discriminator_inputs)
                if self.device is not None:
                    self.teacher = self.teacher.to(torch.device("cpu"))
        dino_loss = None
        losses = [v for k, v in student_rep.items() if "_loss" in k and v is not None]
        loss = sum(losses)
        if self.dino_w > 0:
            dino_results = self.dino_loss(student_rep.pop("dino").unsqueeze(0), teacher_rep.pop("dino").detach().unsqueeze(0), dino_center, 1, 1)
            dino_center = dino_results["dino_center"]
            dino_loss = dino_results["dino_loss"]
            student_dino = student_rep.pop("discriminator_dino", None)
            dino_results = self.dino_loss(student_dino.unsqueeze(0), teacher_rep.pop("discriminator_dino").detach().unsqueeze(0), discriminator_dino_center, 1, 1)
            discriminator_dino_center = dino_results["dino_center"]
            dino_loss = (dino_loss + dino_results["dino_loss"]) / 2.0
            loss += dino_loss
        student_rep = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in student_rep.items()}
        return dict(loss=loss,
                    dino_center=dino_center.detach() if isinstance(dino_center, torch.Tensor) else dino_center,
                    discriminator_dino_center=discriminator_dino_center.detach() if isinstance(discriminator_dino_center, torch.Tensor) else discriminator_dino_center,
                    dino_loss=dino_loss.detach() if isinstance(dino_loss, torch.Tensor) else dino_loss, **student_rep, **all_stats)



