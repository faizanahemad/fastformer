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
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import weight_norm
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
import traceback

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
        self.cls_token = None
        self.extra_cls_token = None
        if num_highway_cls_tokens > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if num_highway_cls_tokens - 1 > 0:
            self.extra_cls_token = nn.Parameter(torch.zeros(1, num_highway_cls_tokens - 1, embed_dim))
        if relative_attention:
            self.first_pos_embed = nn.Parameter(torch.zeros(1, embed_dim))
            self.row_embed = nn.Parameter(torch.zeros(self.height * 2 + 1, pos_dim))
            self.column_embed = nn.Parameter(torch.zeros(self.width * 2 + 1, pos_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.dropout = Dropout(hidden_dropout)
        # self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
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
        if self.extra_cls_token is not None:
            extra_cls_token = self.extra_cls_token.expand(B, -1, -1)
            x = torch.cat((extra_cls_token, x), dim=1)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        # x = self.layer_norm(x)
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
        query = self.layer_norm(query)
        value = self.layer_norm(value) if value is not None else query
        key = self.layer_norm(key) if key is not None else query
        value = self.v_head(value)
        attn_out = self.self_attention(query, key, value, attention_inputs)
        output = query_temp + attn_out
        return output


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
                                      config.relative_attention[0], config.embedding_size)
        self.encoder_block_one = nn.ModuleList()
        for i in range(config.block_sizes[0]):
            self.encoder_block_one.append(TransformerLayer(config, 0, True, i))

        self.encoder_block_two = nn.ModuleList()
        for i in range(config.block_sizes[1]):
            self.encoder_block_two.append(TransformerLayer(config, 1, True, i))

        self.stride = config.stride
        self.decoder_block = nn.ModuleList()
        self.dim_match = nn.Identity()
        self.dim_match_ln = nn.Identity()
        if config.block_channel_size[0] != config.block_channel_size[1]:
            self.dim_match = nn.Linear(config.block_channel_size[0], config.block_channel_size[1], bias=False)
            self.dim_match_ln = nn.LayerNorm(config.block_channel_size[1], eps=config.layer_norm_eps)

        self.has_decoder = config.has_decoder
        if config.has_decoder:
            for i in range(config.num_decoder_layers):
                self.decoder_block.append(TransformerLayer(config, 0, False, i))
            self.dim_match_decoder_linear = nn.Linear(config.block_channel_size[1], config.block_channel_size[0], bias=False)
            self.dim_match_decoder_ln = nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)
        if reinit:
            self.init_weights()

    def forward_first_block(self, x):
        x, row_embed, column_embed = self.patch_embed(x)
        hidden = x
        for layer in self.encoder_block_one:
            hidden = layer(hidden, None, None, ((row_embed, column_embed,), None))
        return hidden, row_embed, column_embed

    def forward_second_block(self, x, first_block_out):
        hidden, row_embed, column_embed = first_block_out
        B, C, H, W = x.shape
        H, W = self.patch_embed.height, self.patch_embed.width
        config = self.config
        hidden = self.dim_match(hidden)
        second_block_init = hidden
        if self.stride > 1:
            assert H % self.stride == 0 and W % self.stride == 0
            cls, hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            hidden = hidden.reshape(B, H, W, config.block_channel_size[1]).permute(0, 3, 1, 2)
            hidden = F.avg_pool2d(hidden, self.stride, self.stride).flatten(2).permute(0, 2, 1)
            hidden = torch.cat((cls, hidden), 1)
        hidden = self.dim_match_ln(hidden)

        for i, layer in enumerate(self.encoder_block_two):
            if i == 0:
                hidden = layer(hidden, second_block_init, second_block_init, ((row_embed, column_embed,), None))
            else:
                hidden = layer(hidden, None, None, ((row_embed, column_embed,), None))
        return hidden

    def forward(self, x, run_decoder=True):
        config = self.config
        B, C, H, W = x.shape
        first_block_out = self.forward_first_block(x)
        first_block_hidden, row_embed, column_embed = first_block_out
        hidden = self.forward_second_block(x, first_block_out)
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
                                             ((row_embed, column_embed,), None))
                else:
                    upsampled_hidden = layer(upsampled_hidden, None, None,
                                             ((row_embed, column_embed,), None))
        hidden = second_block_hidden if upsampled_hidden is None else upsampled_hidden
        return hidden


class ClassificationModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_classes, num_features=768, train_backbone=False, reinit_backbone=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.backbone = backbone
        if isinstance(self.backbone, FastFormerVisionModel):
            num_features = num_features * (self.backbone.config.num_highway_cls_tokens + 1)
        self.num_features = num_features
        self.head = nn.Sequential(nn.LayerNorm(self.num_features), nn.Linear(self.num_features, num_classes))  # nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.GELU(), nn.Linear(self.num_features, num_classes))
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)
        self.train_backbone = train_backbone
        if not train_backbone:
            for v in self.backbone.parameters():
                v.requires_grad = False
        if reinit_backbone:
            self.init_weights()

        init_weights(self.head, 0.01)

    def get_representations(self, x):
        b = x.size(0)
        if isinstance(self.backbone, FastFormerVisionModel):
            output = self.backbone(x, run_decoder=False)
            representation = torch.cat((output[:, 0:self.backbone.config.num_highway_cls_tokens].view(b, -1), output[:, self.backbone.config.num_highway_cls_tokens:].mean(1)), 1)
        else:
            representation = self.backbone(x)
            if len(representation.size()) == 3:
                representation = representation[:, 0]
        return representation

    def forward(self, x, labels=None):
        # TODO: GAP
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


class PatchModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_features=768,
                 generator_w=0.0, discriminator_w=0.0, simclr_w=1.0, dino_w=1.0,
                 reinit=False):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.cls_tokens = backbone.config.num_highway_cls_tokens if hasattr(backbone, "config") and hasattr(backbone.config, "num_highway_cls_tokens") else 1
        self.backbone = backbone
        self.ffn_input_features = num_features * self.cls_tokens
        self.num_moco_features = 128
        self.dino_dims = 2 ** 16
        norm_last_layer = True
        bottleneck_dim = 512
        assert generator_w > 0 or simclr_w > 0 or dino_w > 0
        if discriminator_w > 0:
            assert generator_w > 0
        self.generator_w = generator_w
        self.discriminator_w = discriminator_w
        self.simclr_w = simclr_w
        self.dino_w = dino_w
        self.fixed_tolerance_discriminator = True
        self.discriminator_tol = 0.1
        self.loss_bce = nn.BCEWithLogitsLoss()
        if reinit:
            self.init_weights()

        self.moco_ffn = nn.Sequential(nn.Linear(self.ffn_input_features, 2048), nn.GELU(),
                                      nn.Linear(2048, 256), nn.GELU(),
                                      nn.Linear(256, self.num_moco_features),
                                      Norm())
        last_layer = nn.Linear(bottleneck_dim, self.dino_dims, bias=False)
        init_weights(last_layer, 0.02)
        last_layer = nn.utils.weight_norm(last_layer)
        last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            last_layer.weight_g.requires_grad = False

        self.ffn = nn.Sequential(nn.Linear(self.ffn_input_features, 2048), nn.GELU(),
                                 nn.Linear(2048, 2048), nn.GELU(),
                                 nn.Linear(2048, bottleneck_dim), nn.GELU(),
                                 Norm(),
                                 last_layer)  # weight_norm

        self.generator_ffn = nn.Sequential(nn.LayerNorm(num_features),
                                           nn.Linear(num_features, num_features * 2),
                                           nn.GELU(),
                                           nn.Linear(num_features * 2, num_features * 2),
                                           nn.GELU(),
                                           nn.Linear(num_features * 2, num_features),  # nn.Tanh()
                                           )
        self.pool_kernel = (3, 2, 2)
        assert num_features % np.prod(self.pool_kernel) == 0
        self.discriminator_ffn = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, num_features * 2),
                                               nn.GELU(),
                                               nn.Linear(num_features * 2, num_features // np.prod(self.pool_kernel)))

        init_weights(self.ffn[0], 0.02)
        init_weights(self.ffn[1], 0.02)
        init_weights(self.ffn[2], 0.02)
        last_layer.weight_g.data.fill_(1)
        init_weights(self.generator_ffn, 0.01)
        init_weights(self.discriminator_ffn, 0.01)
        init_weights(self.moco_ffn, 0.01)

    def forward(self, x1_noised, x1_label=None):
        b = x1_noised.size(0)
        assert torch.isfinite(x1_noised).all().item()
        first_block_out = self.backbone.forward_first_block(x1_noised)
        assert torch.isfinite(first_block_out[0]).all().item()
        x1_reconstruct = None
        label_for_discriminator = None
        reconstruction_loss = None
        discriminator_label_mean = None
        absolute_reconstruction_loss = None
        extra_reconstruction_loss = None
        extra_reconstruction_small_pool_loss = None
        if self.generator_w > 0 and x1_label is not None:
            x1_reconstruct = self.generator_ffn(first_block_out[0][:, self.cls_tokens:])  # * 3
            assert torch.isfinite(x1_reconstruct).all().item()
            assert torch.isfinite(x1_label).all().item()

            x1_label_copy = x1_label
            x1_label = x1_label.view(b, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, 14 * 14, -1)
            reconstruction_loss = torch.abs(x1_reconstruct - x1_label)
            x1_reconstruct = x1_reconstruct.detach()
            reconstruction_loss_by_averaging = torch.abs(x1_label - x1_label.mean(-1).unsqueeze(-1))
            extra_reconstruction_loss = (reconstruction_loss_by_averaging - reconstruction_loss.detach()).mean()
            x1_reconstruct = x1_reconstruct.permute(0, -1, -2).reshape(b, 3, 16, 16, 14, 14).permute(0, 1, 4, 2, 5, 3).reshape(b, 3, 224, 224)
            reconstruction_loss_by_averaging = torch.abs(F.max_pool2d(x1_label_copy, kernel_size=3, stride=1, padding=1) - x1_label_copy)
            extra_reconstruction_small_pool_loss = (reconstruction_loss_by_averaging - torch.abs(x1_label_copy - x1_reconstruct)).mean()
            reconstruction_loss = reconstruction_loss.reshape(b, 14 * 14, 3, 16, 16)
            reconstruction_loss = F.max_pool3d(reconstruction_loss, kernel_size=self.pool_kernel, stride=self.pool_kernel).reshape(b, 14 * 14, -1)
            absolute_reconstruction_loss = reconstruction_loss.detach()
            assert torch.isfinite(reconstruction_loss).all().item()
            if self.fixed_tolerance_discriminator:
                label_for_discriminator = (absolute_reconstruction_loss < self.discriminator_tol).float()
            else:
                losses_per_region = -1 * reconstruction_loss.detach().mean(-1)
                highest_losses = torch.topk(losses_per_region, int(self.discriminator_pos_frac * 196), dim=1).indices
                label_for_discriminator = torch.zeros_like(losses_per_region)
                label_for_discriminator[torch.arange(highest_losses.size(0)).repeat(highest_losses.size(-1), 1).t(), highest_losses] = 1.0
            reconstruction_loss = reconstruction_loss.mean(-1).mean(-1).mean()
            absolute_reconstruction_loss = reconstruction_loss.detach()
            discriminator_label_mean = label_for_discriminator.mean().item()

            # reconstruction_loss = (x1_reconstruct - x1_label) ** 2

            reconstruction_loss = self.generator_w * reconstruction_loss

        discriminator_loss = None
        discriminator_accuracy = None
        discriminator_positive_accuracy = None
        discriminator_negative_accuracy = None
        if self.discriminator_w > 0 and x1_label is not None:
            x1_disc = self.discriminator_ffn(self.backbone(x1_reconstruct)[:, self.cls_tokens:])


            # x1_disc = x1_disc.reshape(b, 14 * 14, 3, 16, 16)
            # x1_disc = F.avg_pool3d(x1_disc, kernel_size=self.pool_kernel, stride=self.pool_kernel).reshape(b, 14 * 14, -1)
            assert torch.isfinite(x1_disc).all().item()
            logits = x1_disc.squeeze(-1).reshape(b, -1)
            label_for_discriminator = label_for_discriminator.reshape(b, -1)
            discriminator_loss = self.discriminator_w * self.loss_bce(logits, label_for_discriminator)
            predictions = (torch.sigmoid(logits.detach()) > 0.5).type(torch.float)
            sample_accuracies = (predictions == label_for_discriminator).type(torch.float)
            label_for_discriminator = label_for_discriminator.bool()
            discriminator_positive_accuracy = sample_accuracies[label_for_discriminator].mean().item()
            discriminator_negative_accuracy = sample_accuracies[torch.logical_not(label_for_discriminator)].mean().item()
            discriminator_accuracy = torch.mean(sample_accuracies).item()

        x1_simclr = None
        x1_dino = None
        if self.simclr_w > 0 or self.dino_w > 0:
            x1_repr = self.backbone.forward_second_block(x1_noised, first_block_out)
            if self.simclr_w > 0:
                x1_simclr = self.moco_ffn(x1_repr[:, :self.cls_tokens].view(b, -1))
            if self.dino_w > 0:
                x1_dino = self.ffn(x1_repr[:, :self.cls_tokens].view(b, -1))
        return dict(reconstruction_loss=reconstruction_loss, discriminator_label_mean=discriminator_label_mean, absolute_reconstruction_loss=absolute_reconstruction_loss,
                    extra_reconstruction_small_pool_loss=extra_reconstruction_small_pool_loss, extra_reconstruction_loss=extra_reconstruction_loss,
                    discriminator_loss=discriminator_loss, discriminator_accuracy=discriminator_accuracy,
                    discriminator_positive_accuracy=discriminator_positive_accuracy, discriminator_negative_accuracy=discriminator_negative_accuracy,
                    simclr=x1_simclr, dino=x1_dino, reconstruct=x1_reconstruct)


class PatchCLR:
    def __init__(self, student: PatchModel, teacher: PatchModel, eps=1e-7,
                 teacher_contrastive_temperature=0.04, student_contrastive_temperature=0.1,
                 dino_cw=0.9):

        self.cls_tokens = student.cls_tokens
        self.dino_dims = student.dino_dims
        self.student = student
        self.teacher = teacher
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.generator_w = 0.0
        teacher.discriminator_w = 0.0
        self.generator_w = student.generator_w
        self.discriminator_w = student.discriminator_w
        self.discriminator_pos_frac = 0.1
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)

        self.simclr_temperature = 0.2
        self.pool_kernel = student.pool_kernel



        self.eps = eps
        self.teacher_contrastive_temperature = teacher_contrastive_temperature
        self.student_contrastive_temperature = student_contrastive_temperature
        self.simclr_w = student.simclr_w
        self.dino_w = student.dino_w
        self.dino_cw = dino_cw

    def get_last_dino_layer(self):
        if isinstance(self.student, DistributedDataParallel):
            return self.student.module.ffn[-1]
        return self.student.ffn[-1]

    def to(self, device):
        self.student.to(device)
        self.teacher.to(device)
        return self

    def eval(self):
        self.student.eval()
        self.teacher.eval()
        return self

    def train(self):
        self.student.train()
        self.teacher.eval()
        return self

    def zero_grad(self, set_to_none=True):
        self.student.zero_grad(set_to_none=set_to_none)
        self.teacher.zero_grad(set_to_none=set_to_none)

    def calculate_contrastive_loss(self, contrastive_matrix, label_lengths, calculate_accuracy=False):
        labels = torch.arange(label_lengths, device=contrastive_matrix.device)
        contrastive_matrix = contrastive_matrix / self.simclr_temperature
        loss = self.loss_ce(contrastive_matrix, labels)
        accuracy = 0
        if calculate_accuracy:
            predictions = contrastive_matrix.detach().argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        return loss, accuracy

    def dino_loss(self, x1_dino, x2_dino, dino_center, x2_dino_dash=None):
        x1_dino = torch.log_softmax(x1_dino / self.student_contrastive_temperature, 1)
        dino_center = self.dino_cw * dino_center + (1 - self.dino_cw) * (torch.cat((x2_dino, x2_dino_dash)) if x2_dino_dash is not None else x2_dino).mean(dim=0)
        x2_dino = (x2_dino - dino_center) / self.teacher_contrastive_temperature
        x2_dino = torch.softmax(x2_dino, 1)
        dino_loss = -1 * (x2_dino * x1_dino).sum(dim=1).mean()
        dino_loss = self.dino_w * dino_loss

        if x2_dino_dash is not None:
            x2_dino_dash = (x2_dino_dash - dino_center) / self.teacher_contrastive_temperature
            x2_dino_dash = torch.softmax(x2_dino_dash, 1)
            dino_loss_dash = -1 * (x2_dino_dash * x1_dino).sum(dim=1).mean()
            dino_loss_dash = self.dino_w * dino_loss_dash
            dino_loss = (dino_loss + dino_loss_dash) / 2.0

        return dict(dino_loss=dino_loss, dino_center=dino_center)

    def simclr_loss(self, b1s, b2s, b2s_dash=None, extra_negative_repr_simclr=None, calculate_accuracy=False):
        eye = (1 - torch.eye(b1s.size(0), b1s.size(0), device=b1s.device))
        own_negatives = b1s.mm(b1s.t()) * eye
        b2s_matrix = b1s.mm(b2s.t())
        extra_neg_matrix = None
        if extra_negative_repr_simclr is not None:
            extra_neg_matrix = b1s.mm(extra_negative_repr_simclr.t())
            contrastive_matrix = torch.cat((b2s_matrix, extra_neg_matrix, own_negatives), 1)
            extra_negative_repr_simclr = torch.cat((extra_negative_repr_simclr, b2s.detach()), 0)
        else:
            contrastive_matrix = torch.cat((b2s_matrix, own_negatives), 1)
            extra_negative_repr_simclr = b2s.detach()

        simclr_loss, simclr_accuracy = self.calculate_contrastive_loss(contrastive_matrix, b1s.shape[0], calculate_accuracy=calculate_accuracy)

        if b2s_dash is not None:
            b2s_matrix = b2s_matrix * eye
            b2sd_matrix = b1s.mm(b2s_dash.t())
            if extra_neg_matrix is not None:
                contrastive_matrix = torch.cat((b2sd_matrix, b2s_matrix, extra_neg_matrix, own_negatives), 1)
            else:
                contrastive_matrix = torch.cat((b2sd_matrix, b2s_matrix, own_negatives), 1)

            simclr_loss_dash, simclr_accuracy_dash = self.calculate_contrastive_loss(contrastive_matrix, b1s.shape[0], calculate_accuracy=calculate_accuracy)
            simclr_loss = (simclr_loss + simclr_loss_dash)/2.0
            if calculate_accuracy:
                simclr_accuracy = (simclr_accuracy + simclr_accuracy_dash)/2.0
        simclr_loss = self.simclr_w * simclr_loss
        return dict(simclr_loss=simclr_loss, simclr_accuracy=simclr_accuracy, extra_negative_repr_simclr=extra_negative_repr_simclr)

    def __call__(self, x1_noised, x1_label, x2, x2_dash=None, x2_small=None, extra_negative_repr_simclr=None, dino_center=None, calculate_accuracy=False):
        student_rep = self.student(x1_noised, x1_label)
        with torch.no_grad():
            teacher_rep = self.teacher(x2, None)
            dino_dash = None
            simclr_dash = None
            if x2_dash is not None:
                teacher_rep_dash = self.teacher(x2_dash, None)
                if self.dino_w > 0:
                    dino_dash = teacher_rep_dash["dino"].detach()
                if self.simclr_w > 0:
                    simclr_dash = teacher_rep_dash["simclr"].detach()

        loss = 0.0
        dino_loss = None
        if self.dino_w > 0:
            dino_results = self.dino_loss(student_rep["dino"], teacher_rep["dino"].detach(), dino_center, dino_dash)
            dino_center = dino_results["dino_center"]
            dino_loss = dino_results["dino_loss"]
            loss += dino_loss

        reconstruction_loss = student_rep["reconstruction_loss"]
        if reconstruction_loss is not None:
            loss += reconstruction_loss

        discriminator_loss = student_rep["discriminator_loss"]
        if discriminator_loss is not None:
            loss += discriminator_loss

        simclr_loss = None
        simclr_accuracy = None
        if self.simclr_w > 0:
            simclr_results = self.simclr_loss(student_rep["simclr"], teacher_rep["simclr"].detach(), simclr_dash, extra_negative_repr_simclr=extra_negative_repr_simclr, calculate_accuracy=calculate_accuracy)
            simclr_loss = simclr_results["simclr_loss"]
            simclr_accuracy = simclr_results["simclr_accuracy"]
            extra_negative_repr_simclr = simclr_results["extra_negative_repr_simclr"]
            loss += simclr_loss

        return dict(loss=loss, reconstruction_loss=reconstruction_loss, discriminator_loss=discriminator_loss, simclr_loss=simclr_loss, dino_loss=dino_loss,
                    simclr_accuracy=simclr_accuracy, discriminator_accuracy=student_rep["discriminator_accuracy"],
                    discriminator_label_mean=student_rep["discriminator_label_mean"],
                    extra_negative_repr_simclr=extra_negative_repr_simclr, dino_center=dino_center, absolute_reconstruction_loss=student_rep["absolute_reconstruction_loss"],
                    discriminator_negative_accuracy=student_rep["discriminator_negative_accuracy"], discriminator_positive_accuracy=student_rep["discriminator_positive_accuracy"],
                    extra_reconstruction_loss=student_rep["extra_reconstruction_loss"], extra_reconstruction_small_pool_loss=student_rep["extra_reconstruction_small_pool_loss"])


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
    ap.add_argument("--config", type=str, default='vision_base_config',
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
    to_tensor = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), normalize])
    dog = to_tensor(Image.open("dog.jpg"))
    cat = to_tensor(Image.open("cat.jpg"))
    fox = to_tensor(Image.open("fox.jpg"))
    grasshopper = to_tensor(Image.open("grasshopper.jpg"))
    x1 = torch.stack([dog, cat, fox, grasshopper])

    dog = to_tensor(Image.open("dog.jpg"))
    cat = to_tensor(Image.open("cat.jpg"))
    fox = to_tensor(Image.open("fox.jpg"))
    grasshopper = to_tensor(Image.open("grasshopper.jpg"))
    x2 = torch.stack([dog, cat, fox, grasshopper])
    x = x1

    # x = torch.randn(batch_size, 3, 224, 224, device=device)

    dino_center = None
    if args["deit"]:
        from timm.models.vision_transformer import VisionTransformer
        if args["classification"]:
            model = get_pretrained_deit(False)
        else:
            model = get_pretrained_deit()
            student = PatchModel(model, 768, simclr_w=1.0, dino_w=1.0, generator_w=1.0, discriminator_w=1.0,)
            teacher = PatchModel(copy.deepcopy(model), 768, simclr_w=1.0, dino_w=1.0, generator_w=1.0, discriminator_w=1.0,)
            model = PatchCLR(student, teacher, 1e-7, )
            dino_center = torch.zeros(model.dino_dims, device=device) if model.dino_w > 0 else None
    else:
        model = FastFormerVisionModel(config)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable Params = %s" % (numel(model) / 1_000_000))
        print(model)
        if args["classification"]:
            pass
        else:
            dims = config.block_channel_size[0] if config.has_decoder else config.block_channel_size[1]
            student = PatchModel(model, dims, simclr_w=1.0, dino_w=1.0, generator_w=1.0, discriminator_w=1.0,)
            teacher = PatchModel(copy.deepcopy(model), dims, simclr_w=1.0, dino_w=1.0, generator_w=1.0, discriminator_w=1.0,)
            model = PatchCLR(student, teacher, 1e-7)
            dino_center = torch.zeros(model.dino_dims, device=device) if model.dino_w > 0 else None
    if "pretrained_model" in args and args["pretrained_model"] is not None:
        if not os.path.exists(args["pretrained_model"]):
            args["pretrained_model"] = os.path.normpath(os.path.join(os.getcwd(), args["pretrained_model"]))
        if os.path.exists(args["pretrained_model"]):
            state_dict = torch.load(args["pretrained_model"], map_location=device)
            try:
                model.student.load_state_dict(state_dict, strict=True)
            except:
                try:
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    model.student.load_state_dict(state_dict, strict=False)
                    load_type = "strict-from-ddp"
                except Exception as e:
                    print("WARN: State dict load error")
                    traceback.print_exc()

    print(model)
    model = model.to(device)
    if args["classification"]:
        output = model(x)
        print(output.argmax(-1))
        exit()

    if model.generator_w > 0:
        from fastformer.utils import get_image_augmetations
        dog = Image.open("cat.jpg")
        image_transforms = get_image_augmetations("clr", False)
        dog_noised = image_transforms["non_shape_transforms"](dog)
        dog = image_transforms["crop_224"](dog)
        dog_noised = image_transforms["crop_224"](dog_noised)
        dog_pt = image_transforms["to_pytorch"](dog)
        dog_noised_pt = image_transforms["to_pytorch"](dog_noised)

        with torch.no_grad():
            reconstructed = model.student(dog_noised_pt.unsqueeze(0), dog_pt.unsqueeze(0))["reconstruct"].squeeze(0)
            reconstructed = image_transforms["from_pytorch"](reconstructed)

        dog.show()
        dog_noised.show()
        reconstructed.show()
    output = model(x1, x1, x2, x2, calculate_accuracy=True, dino_center=dino_center)

    all_params = list(filter(lambda p: p.requires_grad, model.student.parameters()))
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
        for name, params in model.student.named_parameters():
            if params.grad is None:
                print(name)

    def run():
        if not forward_only:
            if fp16:
                with autocast():
                    output = model(x, x1, x2, calculate_accuracy=True, dino_center=dino_center)
                    loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                    scaler.scale(loss).backward()
                    get_unused_params(model)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(x, x1, x2, calculate_accuracy=True, dino_center=dino_center)
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
                        output = model(x, x1, x2, calculate_accuracy=True, dino_center=dino_center)

            else:
                with torch.no_grad():
                    output = model(x, x1, x2, calculate_accuracy=True, dino_center=dino_center)
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
        output = {k: float(v) for k, v in output.items() if v is not None and not isinstance(v, torch.Tensor)}
        accuracy.append(output)
    print("Time Taken = %.4f, Lowest = %.4f, Highest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.max(times), np.std(times)))
    print(accuracy[:5])
    print(accuracy[-5:])















