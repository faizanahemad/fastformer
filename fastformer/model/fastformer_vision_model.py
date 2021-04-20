import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import traceback
from attrdict import AttrDict


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


try:
    from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
except:
    pass

from fastformer.config import *
from fastformer.model import FastFormerPreTrainedModel
from itertools import repeat
import collections.abc

logger = logging.get_logger(__name__)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_highway_cls_tokens=1, hidden_dropout=0.1, layer_norm_eps=1e-4,
                 relative_attention=False, pos_dim=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.relative_attention = relative_attention
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.height = img_size[0] // patch_size[0]
        self.width = img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, num_highway_cls_tokens, embed_dim))
        if relative_attention:
            self.first_pos_embed = nn.Parameter(torch.zeros(1, embed_dim))
            self.row_embed = nn.Parameter(torch.zeros(self.height * 2 + 1, pos_dim))
            self.column_embed = nn.Parameter(torch.zeros(self.width * 2 + 1, pos_dim))

        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.dropout = Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.relative_attention:
            x[:, 0] = x[:, 0] + self.first_pos_embed
            row_embed = self.row_embed
            column_embed = self.column_embed
        else:
            x = x + self.pos_embed
            row_embed = None
            column_embed = None
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x, row_embed, column_embed


class MultiheadAttention(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_encoder_layer, layer_index):
        super().__init__()

        self.config = config
        self.relative_attention = config.relative_attention[block_index]
        self.block_index = block_index
        d_model, all_head, d_head = config.block_channel_size[block_index], config.n_head[block_index], config.d_head[block_index]
        d_model_initial = d_model
        n_head = all_head[0]
        total_heads = sum(all_head)
        assert d_model % 16 == 0
        remaining_d_model = d_model
        self.n_head = n_head
        d_model = remaining_d_model
        self.block_index = block_index
        self.layer_index = layer_index
        self.is_encoder_layer = is_encoder_layer

        assert d_model % d_head == 0
        assert d_model % n_head == 0
        assert d_model % (n_head * d_head) == 0
        self.attention_dropout = Dropout(config.attention_dropout)
        self.d_model = d_model
        self.cls_tokens = self.config.num_highway_cls_tokens

        self.q_head = nn.Linear(d_model, n_head * d_head, bias=True)
        self.k_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_head = nn.Linear(d_model_initial, d_model_initial)

        if self.relative_attention:
            self.pos_q_head = nn.Linear(config.embedding_size, n_head * d_head // 2, bias=True)
            self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head // 2, bias=False)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(config.block_channel_size[block_index], eps=config.layer_norm_eps)
        self.scale_factor = 1 + (4 if self.relative_attention else 0)
        self.scale = 1.0 / ((d_head ** 0.5)*(self.scale_factor ** 0.5))

    def self_attention(self, query, key, value, attention_inputs):
        batch_size, seq_len, dim = query.shape
        initial_seq_len = seq_len
        (row_embed_q, column_embed_q,), attention_mask = attention_inputs
        context_len = key.shape[1]
        n_head, d_head = self.n_head, self.config.d_head[self.block_index]
        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = value.view(batch_size, context_len, n_head, self.d_model // n_head)

        scale = self.scale
        q_head = q_head + self.r_w_bias
        q_head = q_head * scale
        content_score = torch.einsum("bind,bjnd->bnij", q_head, k_head)
        attn_score = content_score
        # Shape n_head x d_head
        # Shapes batch_size x n_head x seq_len x context_len
        if self.relative_attention:
            row_embed_k, column_embed_k = self.pos_k_head(row_embed_q), self.pos_q_head(column_embed_q)
            row_embed_q, column_embed_q = self.pos_q_head(row_embed_q), self.pos_q_head(column_embed_q)
            # print(query.size(), key.size(), position_embed_of_query.size(), position_embed_of_key.size())
            nc_score = disentangled_att_bias_2d(q_head[:, self.cls_tokens:].transpose(1, 2), k_head[:, self.cls_tokens:].transpose(1, 2),
                                                row_embed_q, column_embed_q, row_embed_k, column_embed_k,
                                                self.scale_factor,
                                                self.config.max_position_embeddings,
                                                self.config.max_position_embeddings,
                                                n_head // 2,
                                                query_stride=self.config.stride ** self.block_index,
                                                key_stride=(self.config.stride ** self.block_index) if self.layer_index != 0 else (self.config.stride ** max(self.block_index - 1, 0) if self.is_encoder_layer else self.config.stride ** (len(self.config.block_sizes) - 1)))

            nc_score = F.pad(nc_score, (self.cls_tokens, 0, self.cls_tokens, 0, 0, 0, 0, 0))
            attn_score = attn_score + nc_score

        # precision safe in case of mixed precision training
        # attn_score = attn_score.float()
        # perform masking
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=attn_score.dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)
        attn_out = attn_vec.reshape(batch_size, seq_len, self.d_model)
        # Shape shape batch_size x seq_len x d_model

        return attn_out

    def forward(self, query, key, value, attention_inputs):
        assert query.size(1) > 0
        batch_size, seq_len, _ = query.shape
        query_temp = query
        value = value if value is not None else query
        value = self.v_head(value)
        if key is None and value is None:
            key = query
            value = query
        attn_out = self.self_attention(query, key, value, attention_inputs)
        output = query_temp + self.layer_norm(attn_out)
        return output


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, hidden_dropout, layer_norm_eps):
        super().__init__()
        
        d_inner = d_model * 4
        self.d_model = d_model
        activation_dropout = Dropout(hidden_dropout)

        self.layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_inner, bias=True), nn.GELU(), nn.Linear(d_inner, d_model, bias=False), activation_dropout)

    def forward(self, hidden):
        h = hidden
        h = self.ffn(h)
        h = self.layer_norm(h)
        h = hidden + h
        return h


class TransformerLayer(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_encoder_layer, layer_index):
        super().__init__()
        self.attention = MultiheadAttention(config, block_index, is_encoder_layer, layer_index)
        self.ffn = PositionwiseFFN(config.block_channel_size[block_index], config.hidden_dropout, config.layer_norm_eps)
        self.block_index = block_index
        self.block_size = config.block_sizes[block_index]

    def forward(self, query, key, value, attention_inputs):
        assert query.size(1) > 0
        attn = self.attention(query, key, value, attention_inputs)
        output = self.ffn(attn)
        return output


class FastFormerVisionModel(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, reinit=True):
        super().__init__(config)
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens

        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_chans, config.block_channel_size[0],
                                      config.num_highway_cls_tokens, config.hidden_dropout, config.layer_norm_eps,
                                      config.relative_attention, config.embedding_size)
        self.encoder_block_one = nn.ModuleList()
        for i in range(config.block_sizes[0]):
            self.encoder_block_one.append(TransformerLayer(config, 0, True, i))

        self.encoder_block_two = nn.ModuleList()
        for i in range(config.block_sizes[1]):
            self.encoder_block_two.append(TransformerLayer(config, 1, True, i))

        self.stride = config.stride
        self.decoder_block = nn.ModuleList()
        self.dim_match = nn.Identity()
        if config.block_channel_size[0] != config.block_channel_size[1]:
            self.dim_match = nn.Sequential(nn.Linear(config.block_channel_size[0],config.block_channel_size[1], bias=False),
                                           nn.LayerNorm(config.block_channel_size[1], eps=config.layer_norm_eps))

        self.has_decoder = config.has_decoder
        if config.has_decoder:
            for i in range(config.num_decoder_layers):
                self.decoder_block.append(TransformerLayer(config, 0, False, i))
            self.dim_match_decoder_linear = nn.Linear(config.block_channel_size[1], config.block_channel_size[0], bias=False)
            self.dim_match_decoder_ln = nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)
        if reinit:
            self.init_weights()

    def forward(self, x, run_decoder=False):
        config = self.config
        B, C, H, W = x.shape
        x, row_embed, column_embed = self.patch_embed(x)
        H, W = self.patch_embed.height, self.patch_embed.width
        initail_attention = None
        second_block_attention = initail_attention
        hidden = x
        for layer in self.encoder_block_one:
            hidden = layer(hidden, hidden, hidden, ((row_embed, column_embed,), initail_attention))
        first_block_hidden = hidden
        hidden = self.dim_match(hidden)
        second_block_init = hidden
        if self.stride > 1:
            assert H % self.stride == 0 and W % self.stride == 0
            cls, hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            hidden = hidden.reshape(B, H, W, config.block_channel_size[1]).permute(0, 3, 1, 2)
            hidden = F.avg_pool2d(hidden, self.stride, self.stride).flatten(2).permute(0, 2, 1)
            hidden = torch.cat((cls, hidden), 1)

        for i, layer in enumerate(self.encoder_block_two):
            if i == 0:
                hidden = layer(hidden, second_block_init, second_block_init, ((row_embed, column_embed,), second_block_attention))
            else:
                hidden = layer(hidden, hidden, hidden, ((row_embed, column_embed,), second_block_attention))
        second_block_hidden = hidden

        upsampled_hidden = None
        if run_decoder and self.has_decoder:
            hidden = self.dim_match_decoder_ln(self.dim_match_decoder_linear(hidden))
            upsampled_hidden = hidden
            if self.stride > 1:
                cls, upsampled_hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
                stride_factor = (config.stride ** (len(config.block_sizes) - 1))
                upsampled_hidden = upsampled_hidden.reshape(B, H // stride_factor, W // stride_factor, config.block_channel_size[0])
                upsampled_hidden = upsampled_hidden.repeat_interleave(stride_factor, -2).repeat_interleave(stride_factor, -3)
                upsampled_hidden = upsampled_hidden.view(B, -1, config.block_channel_size[0])
                upsampled_hidden = torch.cat((cls, upsampled_hidden), 1)
                upsampled_hidden = upsampled_hidden + first_block_hidden

            for i, layer in enumerate(self.decoder_block):
                if i == 0:
                    upsampled_hidden = layer(upsampled_hidden, hidden, hidden,
                                             ((row_embed, column_embed,), second_block_attention))
                else:
                    upsampled_hidden = layer(upsampled_hidden, upsampled_hidden, upsampled_hidden,
                                             ((row_embed, column_embed,), initail_attention))
        return dict(first_block_hidden=first_block_hidden, second_block_hidden=second_block_hidden, third_block_hidden=upsampled_hidden)


class ClassificationModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_classes, num_features=768, train_backbone=False, reinit_backbone=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.backbone = backbone
        self.num_features = num_features
        self.head = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.GELU(), nn.Linear(self.num_features, num_classes))
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)
        self.train_backbone = train_backbone
        if reinit_backbone:
            self.init_weights()

        for module in self.head:
            if hasattr(module, "weight"):
                fan_out, fan_in = module.weight.shape
                std = np.sqrt(1.0 / float(fan_in + fan_out))
                nn.init.normal_(module.weight, std=std)
            if hasattr(module, "bias"):
                nn.init.constant_(module.bias, 0.0)

    def get_representations(self, x):
        if isinstance(self.backbone, FastFormerVisionModel):
            output = self.backbone(x, run_decoder=True)
            representation = torch.cat((output["second_block_hidden"][:, 0], output["third_block_hidden"][:, 0]), 1)
        else:
            representation = self.backbone(x)
            if len(representation.size()) == 3:
                representation = representation[:, 0]
        return representation

    def forward(self, x, labels=None):
        if self.train_backbone:
            representation = self.get_representations(x)
        else:
            with torch.no_grad():
                representation = self.get_representations(x)

        logits = self.head(representation)
        loss = 0
        predictions = logits.detach().argmax(dim=-1)
        accuracy = None
        if labels is not None:
            loss = self.loss_ce(logits, labels)
            accuracy = (predictions == labels).float().mean().item()
        return dict(loss=loss, logits=logits, predictions=predictions, accuracy=accuracy)


class PatchCLR(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_features=384, eps=1e-4,
                 patchclr_w=1.0, contrastive_temperature=5e-2,
                 simclr_w=1.0, clustering_w=1.0, gap_bias_w=0.1, reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.backbone = backbone
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)
        self.ffn = nn.Linear(num_features, 128)
        self.num_features = 128
        self.eps = eps
        self.contrastive_temperature = contrastive_temperature
        self.simclr_w = simclr_w
        self.clustering_w = clustering_w
        self.patchclr_w = patchclr_w
        self.gap_bias_w = gap_bias_w
        if reinit:
            self.init_weights()

    def calculate_contrastive_loss(self, contrastive_matrix, label_lengths, extra_negatives=None):
        contrastive_matrix = contrastive_matrix / self.contrastive_temperature
        mask = contrastive_matrix.new_zeros(contrastive_matrix.size(), requires_grad=False).fill_diagonal_(1e3)
        contrastive_matrix = contrastive_matrix - mask
        del mask
        rnd_idx = 0
        if extra_negatives is not None:
            ens = extra_negatives.size(1)
            rnd_idx = random.randint(0, ens)
            en1, en2 = extra_negatives.split([rnd_idx, ens - rnd_idx])
            contrastive_matrix = torch.cat((en1, contrastive_matrix, en2), 1)

        labels = torch.cat((torch.arange(label_lengths, device=contrastive_matrix.device) + label_lengths + rnd_idx, torch.arange(label_lengths, device=contrastive_matrix.device) + rnd_idx))
        loss = self.loss_ce(contrastive_matrix, labels)
        predictions = contrastive_matrix.detach().argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return loss, accuracy

    def build_representations(self, x1, x2, eval=False):
        if eval:
            _ = self.eval()
            _ = self.backbone.eval()
            _ = self.ffn.eval()
        if isinstance(self.backbone, FastFormerVisionModel):
            b1 = self.backbone(x1, run_decoder=True)
            b2 = self.backbone(x2, run_decoder=True)
        else:
            b1 = self.ffn(self.backbone(x1))
            b2 = self.ffn(self.backbone(x2))
        if isinstance(b1, dict):
            b1 = self.ffn(b1["third_block_hidden"] if b1["third_block_hidden"] is not None else b1["second_block_hidden"])  # B,S,D
            b2 = self.ffn(b2["third_block_hidden"] if b2["third_block_hidden"] is not None else b2["second_block_hidden"])  # B,S,D
        b1 = b1 / (b1.norm(2, -1, True) + self.eps)
        b2 = b2 / (b2.norm(2, -1, True) + self.eps)
        return b1, b2

    def forward(self, x1, x2, patch_clr_or_not, extra_negative_repr_patchclr=None, extra_negative_repr_simclr=None):
        b1, b2 = self.build_representations(x1, x2)
        b, s = b1.shape[:2]
        bs = b * s

        patchclr_loss = 0.0
        patchclr_accuracy = None
        if self.patchclr_w > 0 and patch_clr_or_not.sum() > 0:
            b1p = b1[patch_clr_or_not]
            b2p = b2[patch_clr_or_not]
            out_1 = b1p.reshape(-1, self.num_features)  # BxS , D
            out_2 = b2p.reshape(-1, self.num_features)  # BxS , D

            c1 = torch.cat((out_1, out_2), 0)

            # b2 = torch.cat((out_2, out_1), 0)
            contrastive_matrix = c1.mm(c1.t()) * (1 - torch.eye(c1.size(0), c1.size(0), device=c1.device))
            contrastive_matrix_store = contrastive_matrix

            patchclr_negative=None
            if extra_negative_repr_patchclr is not None:
                if extra_negative_repr_patchclr.size(0) > 8 * bs:
                    extra_negative_repr_patchclr = extra_negative_repr_patchclr[bs:]
                extra_negative_repr_patchclr = extra_negative_repr_patchclr.to(c1.device)
                if extra_negative_repr_patchclr.size(0) > 4 * bs:
                    c1_det = out_1.detach()
                    selector_mat = c1_det.mm(extra_negative_repr_patchclr.t())
                    topk_indices_argmax = selector_mat.argmax(1)
                    topk_indices_max = torch.topk(selector_mat.max(0).values, bs, dim=0).indices
                    topk_indices_mean = torch.topk(selector_mat.mean(0), bs, dim=0).indices
                    topk_indices = torch.unique(torch.cat((topk_indices_argmax, topk_indices_max, topk_indices_mean)))
                    patchclr_negative = c1.mm(extra_negative_repr_patchclr[topk_indices].contiguous().t())
                    del topk_indices_mean
                    del topk_indices_max
                    del topk_indices_argmax
                    del selector_mat
                    del topk_indices
                    extra_negative_repr_patchclr = torch.cat((extra_negative_repr_patchclr, c1_det), 0)
                else:
                    patchclr_negative = c1.mm(extra_negative_repr_patchclr.t())

            else:
                extra_negative_repr_patchclr = out_1.detach()
            extra_negative_repr_patchclr = extra_negative_repr_patchclr.detach().cpu()

            patchclr_loss, patchclr_accuracy = self.calculate_contrastive_loss(contrastive_matrix, out_1.shape[0], patchclr_negative)
            patchclr_loss = self.patchclr_w * patchclr_loss
        clustering_loss = 0.0
        if self.clustering_w > 0 and self.patchclr_w > 0 and self.gap_bias_w == 0:
            cmm = contrastive_matrix_store.reshape(2, bs, 2, bs).transpose(1,2).reshape(4, bs, bs)
            cmm2 = cmm.reshape(4, b, s, b, s).transpose(2, 3).reshape(4, b, b, -1).mean(-1)
            should_be_similar = torch.diagonal(cmm2, dim1=1, dim2=2)
            clustering_loss = self.clustering_w * ((4*b - (2*b/s)) + cmm2.sum() - 2 * should_be_similar.sum()) / (math.prod(cmm2.size()) * 0.5)

        simclr_loss = 0.0
        simclr_accuracy = None
        simclr_or_not = ~patch_clr_or_not.detach()
        if self.simclr_w > 0:
            b1s = b1[:, 0]
            b2s = b2[:, 0]
            sc1 = torch.cat((b1s, b2s), 0)

            contrastive_matrix = sc1.mm(sc1.t()) * (1 - torch.eye(sc1.size(0), sc1.size(0), device=sc1.device))
            simclr_negative = None
            if extra_negative_repr_simclr is not None:
                if extra_negative_repr_simclr.size(0) > 256 * b:
                    extra_negative_repr_simclr = extra_negative_repr_simclr[b:]
                extra_negative_repr_simclr = extra_negative_repr_simclr.to(sc1.device)
                sc1_det = b1s.detach()
                if extra_negative_repr_simclr.size(0) >= 64 * b:
                    selector_mat = sc1_det.mm(extra_negative_repr_simclr.t())
                    topk_indices_argmax = selector_mat.argmax(1)
                    topk_indices_max = torch.topk(selector_mat.max(0).values, 16 * b, dim=0).indices
                    topk_indices_mean = torch.topk(selector_mat.mean(0), 16 * b, dim=0).indices
                    topk_indices_mean_select = torch.topk(torch.topk(selector_mat, 4, dim=0).values.mean(0), 32 * b, dim=0).indices
                    most_recent_indices = torch.arange(extra_negative_repr_simclr.size(0) - 32 * b, extra_negative_repr_simclr.size(0), device=sc1.device)
                    rand_indices = torch.randint(0, extra_negative_repr_simclr.size(0), (16 * b,), device=sc1.device)

                    topk_indices = torch.unique(torch.cat((topk_indices_argmax, topk_indices_max, topk_indices_mean, rand_indices, topk_indices_mean_select, most_recent_indices)))
                    simclr_negative = sc1.mm(extra_negative_repr_simclr[topk_indices].contiguous().t())
                    del topk_indices_mean
                    del topk_indices_max
                    del topk_indices_argmax
                    del selector_mat
                    del topk_indices
                    del rand_indices
                    del topk_indices_mean_select
                    del most_recent_indices
                else:
                    simclr_negative = sc1.mm(extra_negative_repr_simclr.t())
                extra_negative_repr_simclr = torch.cat((extra_negative_repr_simclr, sc1_det), 0)
            else:
                extra_negative_repr_simclr = b1s.detach()

            simclr_loss, simclr_accuracy = self.calculate_contrastive_loss(contrastive_matrix, b1s.shape[0], simclr_negative)
            simclr_loss = self.simclr_w * simclr_loss

        gap_bias_loss = 0.0
        gap_bias_accuracy = 0.0
        if self.gap_bias_w > 0 and self.simclr_w > 0 and self.patchclr_w > 0:
            p1s = b1[:, 1:].mean(1)
            p2s = b2[:, 1:].mean(1)
            b1s = b1[:, 0]
            b2s = b2[:, 0]
            gap_bias = torch.cat((b1s, b2s, p1s, p2s), 0)
            contrastive_matrix = gap_bias.mm(gap_bias.t()) * (1 - torch.eye(gap_bias.size(0), gap_bias.size(0), device=b1s.device))
            simclr_negative = None
            if extra_negative_repr_simclr is not None:
                simclr_negative = gap_bias.mm(extra_negative_repr_simclr.t())

            gap_bias_loss, gap_bias_accuracy = self.calculate_contrastive_loss(contrastive_matrix, p1s.shape[0] + p2s.shape[0], simclr_negative)
            gap_bias_loss = self.gap_bias_w * gap_bias_loss

        # TODO: GAP bias SIMCLR
        # SimCLR Loss subset of patch clr
        # TODO: patch clr with previous batch vectors
        # TODO: Reduce SIMCLR weight to zero slowly
        # TODO: Additive Margin Softmax to make the task harder.

        loss = patchclr_loss + clustering_loss + simclr_loss + gap_bias_loss
        return dict(loss=loss, patchclr_loss=patchclr_loss, clustering_loss=clustering_loss, simclr_loss=simclr_loss, gap_bias_loss=gap_bias_loss,
                    patchclr_accuracy=patchclr_accuracy, simclr_accuracy=simclr_accuracy, gap_bias_accuracy=gap_bias_accuracy, 
                    extra_negative_repr_patchclr=extra_negative_repr_patchclr.detach().cpu() if extra_negative_repr_patchclr is not None else None,
                    extra_negative_repr_simclr=extra_negative_repr_simclr.detach().cpu() if extra_negative_repr_simclr is not None else None)


if __name__ == '__main__':
    import time
    import argparse
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW
    import timm
    from PIL import Image
    import torchvision.transforms as transforms

    torch.backends.cudnn.benchmark = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default='cpu',
                    help="Device")
    ap.add_argument("--config", type=str, default='vision_md_rel_funnel_config',
                    help="Config")

    ap.add_argument("--forward_only", type=str2bool, default=False)
    ap.add_argument("--deit", action="store_true", default=False,)
    ap.add_argument("--classification", action="store_true", default=False,)
    ap.add_argument("--fp16", type=str2bool, default=False)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    args = vars(ap.parse_args())
    forward_only = args["forward_only"]
    device = args["device"]
    fp16 = args["fp16"]
    lr = args["lr"]
    epochs = args["epochs"]
    config = args["config"]
    batch_size = args["batch_size"]
    config = vision_config_dict[config]

    device = torch.device(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    dog = to_tensor(Image.open("dog.jpg"))
    cat = to_tensor(Image.open("cat.jpg"))
    fox = to_tensor(Image.open("fox.jpg"))
    grasshopper = to_tensor(Image.open("grasshopper.jpg"))
    x = torch.stack([dog, cat, fox, grasshopper])
    

    # x = torch.randn(batch_size, 3, 224, 224, device=device)

    if args["deit"]:
        from timm.models.vision_transformer import VisionTransformer
        if args["classification"]:
            model = get_pretrained_deit(False)
        else:
            model = get_pretrained_deit()
            model = PatchCLR(model, 768, 1e-7, simclr_w=1.0)
        print(model)
    else:
        model = FastFormerVisionModel(config)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable Params = %s" % (numel(model) / 1_000_000))
        print(model)
        if args["classification"]:
            pass
        else:
            model = PatchCLR(model, config.block_channel_size[0] if config.has_decoder else config.block_channel_size[1], 1e-7, simclr_w=1.0)
    if "pretrained_model" in args and args["pretrained_model"] is not None:
        if not os.path.exists(args["pretrained_model"]):
            args["pretrained_model"] = os.path.normpath(os.path.join(os.getcwd(), args["pretrained_model"]))
        if os.path.exists(args["pretrained_model"]):
            state_dict = torch.load(args["pretrained_model"], map_location=device)
            model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if args["classification"]:
        output = model(x)
        print(output.argmax(-1))
        exit()

    output = model(x, x)
    check_patch_clr_acc(model, "clr", "cpu", args["pretrained_model"], config)

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(all_params, lr=lr, eps=1e-6, weight_decay=1e-2)
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()

    try:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
    except:
        pass
    if forward_only:
        _ = model.eval()
    else:
        _ = model.train()

    def get_unused_params(model):
        for name, params in model.named_parameters():
            if params.grad is None:
                print(name)

    def run():
        if not forward_only:
            if fp16:
                with autocast():
                    output = model(x, x)
                    loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                    scaler.scale(loss).backward()
                    get_unused_params(model)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(x, x)
                loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                loss.backward()
                get_unused_params(model)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        else:
            if fp16:
                with autocast():
                    with torch.no_grad():
                        output = model(x, x)

            else:
                with torch.no_grad():
                    output = model(x, x)
            return output
        return output


    _ = [run() for _ in range(1)]
    times = []
    accuracy = []
    for _ in trange(epochs):
        st = time.time()
        output = run()
        et = time.time() - st
        times.append(et)
        output = {k: float(v) for k, v in output.items() if v is not None}
        accuracy.append(output)
    print("Time Taken = %.4f, Lowest = %.4f, Highest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.max(times), np.std(times)))
    print(accuracy[:5])
    print(accuracy[-5:])















