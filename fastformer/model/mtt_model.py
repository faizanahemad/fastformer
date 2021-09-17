import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import traceback


from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast


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



def get_mtt_backbone(model_name, cls_tokens, enable_layer_normalizers, sampling_alpha, reinit=False, train_layer_normalizers=True, enable_layer_normalizers_statistics=False, dropout_prob=0.05):
    # TODO: Later also add a QnA boolean / fixed number of options question
    # TODO: Add extra CLS attr and tokens in embedding

    if "prenorm-roberta" in model_name:
        if "xlarge" in model_name:
            config = RobertaConfig.from_pretrained("roberta-large")
            config.num_hidden_layers = 48
            model_name = "roberta-large"
        elif "large" in model_name:
            model_name = "roberta-large"
            config = RobertaConfig.from_pretrained(model_name)
        elif "small" in model_name:
            model_name = "roberta-base"
            config = RobertaConfig.from_pretrained(model_name)
            config.hidden_size = 256
        else:
            model_name = "roberta-base"
            config = RobertaConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        config.enable_layer_normalizers_statistics = enable_layer_normalizers_statistics
        # config.gradient_checkpointing = not enable_layer_normalizers and not enable_layer_normalizers_statistics
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # config.gradient_checkpointing = True
        # config.vocab_size = 30522
        config.enable_layer_normalizers = enable_layer_normalizers
        config.train_layer_normalizers = train_layer_normalizers
        if sampling_alpha is not None:
            config.sampling_alpha = sampling_alpha
        model = PreNormRobertaModel(config)
    elif "roberta" in model_name:

        if "large" in model_name:
            model_name = "roberta-large"
        elif "base" in model_name or "small" in model_name:
            model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        config = RobertaConfig.from_pretrained(model_name)
        # config.gradient_checkpointing = True
        model = AutoModel.from_pretrained(model_name)
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
    change_dropout(model)
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
                 lm_temperature=1.0,
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
        self.dino_dims = 2 ** 14
        norm_last_layer = True
        self.lm_temperature = lm_temperature
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
        num_features_small = self.backbone.small_config.hidden_size
        self.sentence_order_prediction_w = sentence_order_prediction_w
        if self.generator_w > 0 or self.discriminator_w > 0:
            self.lm_head = nn.Linear(embedding_dims, self.vocab_size, bias=False)
            self.tie_weights()
        if reinit:
            self.init_weights()

        # if dino_w > 0:
        #     last_layer = nn.Linear(bottleneck_dim, self.dino_dims, bias=False)
        #     init_weights(last_layer, 0.02)
        #     last_layer = nn.utils.weight_norm(last_layer)
        #     last_layer.weight_g.data.fill_(1)
        #     if norm_last_layer:
        #         last_layer.weight_g.requires_grad = False
        #
        #     self.ffn = nn.Sequential(nn.Linear(num_features, num_features), nn.GELU(),
        #                              nn.Linear(num_features, bottleneck_dim),
        #                              # Norm(),
        #                              last_layer)
        #     init_weights(self.ffn[0], 0.02)
        #     init_weights(self.ffn[2], 0.02)
        #     last_layer.weight_g.data.fill_(1)

        if generator_w > 0:
            self.generator_ffn = nn.Sequential(nn.Linear(num_features_small, num_features),
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
            self.sent_order_nn = nn.Sequential(nn.Linear(num_features + num_features_small, num_features),
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
            validation_iter=False,
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
                probas = torch.tensor([((lm_layers_total - i) / lm_layers_total) if (i < lm_layers) else 0.0 for i in
                                       range(lm_layers_total)]) ** max(self.sampling_alpha, 0.01)
                start_sampling_from_lm = torch.multinomial(probas, 1, replacement=False, generator=g_cpu).long().item()
                backbone_inputs["start_sampling_from"] = start_sampling_from_lm
            backbone_inputs["keep_last_layer"] = self.keep_last_layer
        backbone_inputs = {k: v for k, v in backbone_inputs.items() if v is not None}
        outputs = self.backbone(**backbone_inputs)
        layer_scales_loss = outputs["layer_scales_loss"] if "layer_scales_loss" in outputs else None
        sent_hidden = outputs["last_hidden_state"][:, 0]
        exclude_layers = outputs["selected_layers"] if "selected_layers" in outputs else []
        n_grad_forward_layers_lm = outputs["n_grad_forward_layers"] if "n_grad_forward_layers" in outputs else None
        n_forward_layers_lm = outputs["n_forward_layers"] if "n_forward_layers" in outputs else None
        masked_lm_loss = None
        lm_accuracy = None
        discriminator_label_mean = None
        discriminator_loss = None
        dino = None
        lm_replace_possible = False
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
        discriminator_selected_layers = None
        generator_output = None
        discriminator_outputs = None
        discriminator_ffn_inputs = None
        discriminator_ffn_outputs = None
        generator_selected_layers = outputs["selected_layers"] if "selected_layers" in outputs else []
        active_locations = attention_mask.bool()

        if self.generator_w > 0 or self.discriminator_w > 0 or discriminator_inputs is not None:
            if self.generator_w > 0 or discriminator_inputs is None or validation_iter or self.dino_w > 0:
                mask_indices = (input_ids.int() != labels.int())
                generator_output = outputs["last_hidden_state"]
                generator_output = self.generator_ffn(generator_output)
                if hasattr(self.backbone, "embeddings_project"):
                    generator_output = self.backbone.embeddings_reverse_project(generator_output)
                lm_mask = mask_indices.unsqueeze(-1).expand(-1, -1, generator_output.size(-1))
                lm_logits = generator_output[lm_mask].reshape(-1, generator_output.size(-1))
                dino = lm_logits
                lm_logits = self.lm_head(lm_logits)

            if (self.generator_w > 0 or validation_iter) and labels is not None:
                active_labels = labels[mask_indices].reshape(-1)
                active_prediction_logits = lm_logits.reshape(-1, self.vocab_size)
                masked_lm_loss = 0.0
                if active_prediction_logits.ndim > 1 and active_prediction_logits.shape[0] > 0 and active_prediction_logits.shape[1] > 0:
                    lm_replace_possible = True
                    if (self.training and torch.is_grad_enabled()) or validation_iter:
                        masked_lm_loss = self.generator_w * self.loss_ce(active_prediction_logits, active_labels)
                    if validation_iter:
                        masked_accuracy = (active_prediction_logits.detach().argmax(dim=-1) == active_labels).type(active_prediction_logits.dtype).mean().item()
                        mask_indices_mean = mask_indices[active_locations].float().mean().item()
                        lm_input_accuracy = (input_ids == labels)[active_locations].type(torch.int32).float().mean().item()

            if self.discriminator_w > 0 or discriminator_inputs is not None:
                if discriminator_inputs is None:
                    if lm_replace_possible:
                        new_input_ids = input_ids.clone()
                        new_input_ids[mask_indices] = temperature_sampling(lm_logits.detach(), self.lm_temperature).view(-1)
                    else:
                        new_input_ids = input_ids
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
                    if self.training and start_from_proba and electra_layers is not None and torch.is_grad_enabled() and start_sampling_from_lm is not None and lm_layers is not None and electra_layers is not None:
                        electra_layers_total = electra_layers_total if electra_layers_total is not None else self.n_layers
                        probas = torch.tensor([((electra_layers_total - i) / electra_layers_total) if (i < electra_layers_total - electra_layers) else 0.0 for i in
                                               range(electra_layers_total)]) ** max(self.sampling_alpha, 0.01)
                        start_sampling_from_electra = start_sampling_from_lm * (electra_layers // lm_layers)
                        discriminator_inputs["start_sampling_from"] = start_sampling_from_electra
                    if self.exclude_layers:
                        discriminator_inputs["exclude_layers"] = exclude_layers
                    discriminator_inputs["keep_last_layer"] = self.keep_last_layer
                discriminator_inputs["output_hidden_states"] = True

                discriminator_inputs["run_large_encoder"] = True
                discriminator_outputs = self.backbone(**discriminator_inputs)

                discriminator_layer_scales_loss = discriminator_outputs["layer_scales_loss"] if "layer_scales_loss" in discriminator_outputs else None
                if layer_scales_loss is not None:
                    layer_scales_loss = layer_scales_loss + discriminator_layer_scales_loss
                discriminator_selected_layers = discriminator_outputs["selected_layers"] if "selected_layers" in discriminator_outputs else []
                n_grad_forward_layers_electra = discriminator_outputs["n_grad_forward_layers"] if "n_grad_forward_layers" in discriminator_outputs else None
                n_forward_layers_electra = discriminator_outputs["n_forward_layers"] if "n_forward_layers" in discriminator_outputs else None
                discriminator_outputs = discriminator_outputs["last_hidden_state"]

                _ = discriminator_inputs.pop("num_layers", None)
                _ = discriminator_inputs.pop("num_layers_total", None)
                _ = discriminator_inputs.pop("rng_seed", None)
                _ = discriminator_inputs.pop("output_hidden_states", None)
                _ = discriminator_inputs.pop("start_sampling_from", None)
                _ = discriminator_inputs.pop("exclude_layers", None)

                # if self.dino_w > 0:
                #     discriminator_dino = self.ffn(discriminator_outputs[:, self.cls_tokens - 1])

                sent_hidden = torch.cat((discriminator_outputs[:, 0], sent_hidden), -1)
                if self.discriminator_w > 0 or validation_iter or self.dino_w > 0:
                    discriminator_ffn_inputs = discriminator_outputs
                    discriminator_dino = discriminator_outputs.reshape(-1, discriminator_outputs.size(-1))
                    discriminator_outputs = self.discriminator_ffn(discriminator_outputs)
                    discriminator_ffn_outputs = discriminator_outputs
                    discriminator_outputs = discriminator_outputs.squeeze(-1)[active_locations].reshape(-1)
                    discriminator_inputs["discriminator_labels"] = discriminator_labels
                    if (self.training and torch.is_grad_enabled()) or validation_iter:
                        discriminator_loss = self.discriminator_w * self.loss_bce(discriminator_outputs, discriminator_labels)
                    if validation_iter:
                        discriminator_label_mean = discriminator_labels.mean()
                        discriminator_labels = discriminator_labels.bool()
                        discriminator_preds = (torch.sigmoid(discriminator_outputs.detach()) > 0.5).type(discriminator_outputs.dtype)
                        sample_accuracies = (discriminator_preds == discriminator_labels).type(discriminator_preds.dtype)
                        discriminator_accuracy = torch.mean(sample_accuracies).item()
                        discriminator_extra_accuracy = max(0.0, float((discriminator_accuracy - discriminator_label_mean) / (1.0 - discriminator_label_mean)))
                        discriminator_positive_accuracy = sample_accuracies[discriminator_labels].mean().item()
                        discriminator_negative_accuracy = sample_accuracies[torch.logical_not(discriminator_labels)].mean().item()

        if self.sentence_order_prediction_w and labels_segment_index is not None and ((self.training and torch.is_grad_enabled()) or validation_iter):
            labels_segment_index = labels_segment_index.float()
            sent_order_logits = self.sent_order_nn(sent_hidden).squeeze(-1)
            if (self.training and torch.is_grad_enabled()) or validation_iter:
                sent_order_loss = self.sentence_order_prediction_w * self.loss_bce(sent_order_logits, labels_segment_index)
            if validation_iter:
                sent_order_preds = (torch.sigmoid(sent_order_logits.detach()) > 0.5).type(sent_order_logits.dtype)
                sent_order_accuracy = (sent_order_preds == labels_segment_index).type(sent_order_logits.dtype).mean().item()

        if layer_scales_loss is not None:
            layer_scales_loss = 0.001 * layer_scales_loss
        return dict(masked_lm_loss=masked_lm_loss, lm_accuracy=lm_accuracy, lm_input_accuracy=lm_input_accuracy,
                    dino=dino, discriminator_accuracy=discriminator_accuracy, sent_order_accuracy=sent_order_accuracy,
                    discriminator_extra_accuracy=discriminator_extra_accuracy, masked_accuracy=masked_accuracy,
                    discriminator_label_mean=discriminator_label_mean, discriminator_loss=discriminator_loss,
                    sent_order_loss=sent_order_loss, input_cls_orthogonal=input_cls_orthogonal, attention_penalty_loss=attention_penalty_loss,
                    discriminator_positive_accuracy=discriminator_positive_accuracy, discriminator_negative_accuracy=discriminator_negative_accuracy,
                    mask_indices_mean=mask_indices_mean, discriminator_dino=discriminator_dino, discriminator_inputs=discriminator_inputs,
                    n_grad_forward_layers_electra=n_grad_forward_layers_electra, n_forward_layers_electra=n_forward_layers_electra,
                    n_forward_layers_lm=n_forward_layers_lm, n_grad_forward_layers_lm=n_grad_forward_layers_lm,
                    layer_scales_loss=layer_scales_loss, start_from_proba=self.start_from_proba, sampling_alpha=self.sampling_alpha,
                    start_sampling_from_lm=start_sampling_from_lm, start_sampling_from_electra=start_sampling_from_electra,
                    electra_to_lm_layer_ratio=n_forward_layers_electra/n_forward_layers_lm,
                    generator_output=generator_output, discriminator_outputs=discriminator_outputs,
                    discriminator_ffn_inputs=discriminator_ffn_inputs, discriminator_ffn_outputs=discriminator_ffn_outputs,
                    discriminator_selected_layers=discriminator_selected_layers, generator_selected_layers=generator_selected_layers)


class MultiTaskHighwayCLSPretraining(PatchCLR):
    def __init__(self, student: MTTModel, teacher: MTTModel, eps=1e-7, device=None):
        super().__init__(student, teacher, eps, 10, 10, 0.9)
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
        teacher.generator_w = student.generator_w
        teacher.discriminator_w = student.discriminator_w
        teacher.sentence_order_prediction_w = student.sentence_order_prediction_w
        self.generator_w = student.generator_w
        self.discriminator_w = student.discriminator_w
        self.device = device

        self.eps = eps
        self.teacher_contrastive_temperature = 10.0
        self.student_contrastive_temperature = 10.0
        self.dino_w = student.dino_w

    def get_last_dino_layer(self):
        return None

    def dino_loss(self, x1_dino, x2_dino, dino_center, student_crops, teacher_crops):
        dino_loss = ((x1_dino - x2_dino) ** 2).sum(dim=-1).mean()
        dino_loss = self.dino_w * dino_loss
        return dict(dino_loss=dino_loss, dino_center=dino_center)

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
        # self.student = self.student.eval()
        # TODO: Do we need to guide both students (MLM/ELECTRA)
        # temp_input_ids = input_ids.clone()
        # temp_labels = labels.clone()
        # temp_attention_mask = attention_mask.clone()
        student_rep = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   labels_segment_index=labels_segment_index,
                                   rng_seed=rng_seed, validation_iter=validation_iter)
        discriminator_inputs = student_rep.pop("discriminator_inputs", None)
        if discriminator_inputs is not None:
            _ = discriminator_inputs.pop("drop_unused_layers", None)
            _ = discriminator_inputs.pop("approximate_unused_layers", None)
        all_stats = dict()
        if validation_iter:
            with torch.no_grad():
                # print(torch.all(input_ids == temp_input_ids), torch.all(temp_labels == labels), torch.all(attention_mask == temp_attention_mask))
                # print(discriminator_inputs)
                all_layers_accuracy = self.student(input_ids=input_ids, attention_mask=attention_mask,
                                                   labels=labels, labels_segment_index=labels_segment_index,
                                                   rng_seed=rng_seed,
                                                   num_layers_lm=getattr(self.student, "module", self.student).lm_layers_total, num_layers_total_lm=getattr(self.student, "module", self.student).lm_layers_total,
                                                   num_layers_electra=getattr(self.student, "module", self.student).electra_layers_total, num_layers_total_electra=getattr(self.student, "module", self.student).electra_layers_total,
                                                   discriminator_inputs=discriminator_inputs, validation_iter=validation_iter)
                all_stats = dict(
                    all_discriminator_extra_accuracy=all_layers_accuracy["discriminator_extra_accuracy"],
                    all_discriminator_accuracy=all_layers_accuracy["discriminator_accuracy"],
                    all_masked_accuracy=all_layers_accuracy["masked_accuracy"],
                    all_n_forward_layers_lm=all_layers_accuracy["n_forward_layers_lm"],
                    all_n_grad_forward_layers_lm=all_layers_accuracy["n_grad_forward_layers_lm"],
                    all_n_grad_forward_layers_electra=all_layers_accuracy["n_grad_forward_layers_electra"],
                    all_n_forward_layers_electra=all_layers_accuracy["n_forward_layers_electra"],
                    all_start_sampling_from_lm=all_layers_accuracy["start_sampling_from_lm"],
                    all_start_sampling_from_electra=all_layers_accuracy["start_sampling_from_electra"],
                    all_generator_selected_layers=all_layers_accuracy["generator_selected_layers"],
                    all_discriminator_selected_layers=all_layers_accuracy["discriminator_selected_layers"],

                )

        teacher_stats = dict()
        if self.dino_w > 0:
            with torch.no_grad():
                # print("teacher layers = ", self.teacher.lm_layers_total, self.teacher.electra_layers_total)
                teacher = self.teacher  # .eval()
                if self.device is not None:
                    teacher = self.teacher.to(self.device)
                teacher_rep = teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels, labels_segment_index=labels_segment_index,
                                      num_layers_lm=self.teacher.lm_layers_total, num_layers_total_lm=self.teacher.lm_layers_total,
                                      num_layers_electra=self.teacher.electra_layers_total,
                                      num_layers_total_electra=self.teacher.electra_layers_total,
                                      discriminator_inputs=discriminator_inputs, validation_iter=validation_iter)

                if validation_iter:
                    teacher_stats = dict(
                        teacher_masked_lm_loss=teacher_rep["masked_lm_loss"].item(),
                        teacher_discriminator_loss=teacher_rep["discriminator_loss"].item(),
                        teacher_masked_accuracy=teacher_rep["masked_accuracy"],
                        teacher_discriminator_extra_accuracy=teacher_rep["discriminator_extra_accuracy"],
                        teacher_sent_order_accuracy=teacher_rep["sent_order_accuracy"],
                    )
                if self.device is not None:
                    self.teacher = self.teacher.to(torch.device("cpu"))
        dino_loss = None
        losses = [v for k, v in student_rep.items() if "_loss" in k and v is not None]
        loss = sum(losses)
        if getattr(self.student, "module", self.student).dino_w > 0:
            student_dino = student_rep.pop("dino")
            dino_results = self.dino_loss(student_dino, teacher_rep.pop("dino").detach(), dino_center, 1, 1)
            dino_center = dino_results["dino_center"]
            # dino_loss = 0
            # if teacher_stats["teacher_masked_lm_loss"] < student_rep["masked_lm_loss"].item():
            dino_loss = dino_results["dino_loss"]

            student_dino = student_rep.pop("discriminator_dino", None)
            dino_results = self.dino_loss(student_dino, teacher_rep.pop("discriminator_dino").detach(), discriminator_dino_center, 1, 1)
            discriminator_dino_center = dino_results["dino_center"]
            # if teacher_stats["teacher_discriminator_loss"] < student_rep["discriminator_loss"].item():
            dino_loss = (dino_loss + dino_results["dino_loss"]) / 2.0
            loss = loss + getattr(self.student, "module", self.student).dino_w * dino_loss
        student_rep = get_loggable_dict(student_rep)
        all_stats = get_loggable_dict(all_stats)
        teacher_stats = get_loggable_dict(teacher_stats)
        return dict(loss=loss,
                    dino_center=dino_center.detach() if isinstance(dino_center, torch.Tensor) else dino_center,
                    discriminator_dino_center=discriminator_dino_center.detach() if isinstance(discriminator_dino_center, torch.Tensor) else discriminator_dino_center,
                    dino_loss=dino_loss.detach() if isinstance(dino_loss, torch.Tensor) else dino_loss, **student_rep, **all_stats, **teacher_stats)



