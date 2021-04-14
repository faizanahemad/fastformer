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
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_highway_cls_tokens=1, hidden_dropout=0.1, layer_norm_eps=1e-4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.height = img_size[0] // patch_size[0]
        self.width = img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, num_highway_cls_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_highway_cls_tokens, embed_dim))
        self.dropout = Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x




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
        self.cls_tokens = self.config.num_highway_cls_tokens + 1

        self.q_head = nn.Linear(d_model, n_head * d_head, bias=True)
        self.k_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_head = nn.Linear(d_model_initial, d_model_initial)

        if self.relative_attention:
            self.pos_q_head = nn.Linear(config.embedding_size, n_head * d_head, bias=True)
            self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head, bias=False)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(config.block_channel_size[block_index], eps=config.layer_norm_eps)
        self.scale_factor = 1 + (2 if self.relative_attention else 0)
        self.scale = 1.0 / ((d_head ** 0.5)*(self.scale_factor ** 0.5))

    def self_attention(self, query, key, value, attention_inputs):
        batch_size, seq_len, dim = query.shape
        initial_seq_len = seq_len
        position_embeds, attention_mask = attention_inputs
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
            position_embeds = position_embeds[self.block_index]
            position_embed_of_key, position_embed_of_query = position_embeds

            pos_k_head = self.pos_k_head(position_embed_of_key)
            pos_q_head = self.pos_q_head(position_embed_of_query)
            # print(query.size(), key.size(), position_embed_of_query.size(), position_embed_of_key.size())
            nc_score = disentangled_att_bias(q_head.transpose(1, 2), k_head.transpose(1, 2), None, pos_q_head, pos_k_head, self.scale_factor,
                                             self.config.max_position_embeddings // (self.config.stride ** self.block_index),
                                             (self.config.max_position_embeddings // (self.config.stride ** self.block_index)) if self.layer_index > 0 else ((self.config.max_position_embeddings // (self.config.stride ** max(self.block_index - 1, 0))) if self.is_encoder_layer else (self.config.max_position_embeddings // (self.config.stride ** (len(self.config.block_channel_size) - 1)))), n_head)
            nc_score = torch.cat((nc_score.new_zeros(nc_score.size(0), nc_score.size(1), self.cls_tokens, nc_score.size(-1)),
                                  torch.cat((nc_score.new_zeros(nc_score.size(0), nc_score.size(1), nc_score.size(-2) - self.cls_tokens, self.cls_tokens),
                                             nc_score[..., self.cls_tokens:, self.cls_tokens:]), -1)), -2)
            attn_score = attn_score + nc_score

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        # attn_score = attn_score.float()
        # perform masking
        if attention_mask is not None:
            # TODO: handle attention mask's pooling for qk pooling
            if len(attention_mask.size()) == 2:
                attention_mask = attention_mask[:, None, None].float()
            attn_score = attn_score - INF * (1 - attention_mask)
        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
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
    def __init__(self, config: FastFormerConfig):
        super().__init__(config)
        self.config = config
        self.cls_tokens = config.num_highway_cls_tokens
        self.patch_embed = PatchEmbed(config.img_size, config.patch_size, config.in_chans, config.block_channel_size[0],
                                      config.num_highway_cls_tokens, config.hidden_dropout, config.layer_norm_eps)
        self.encoder_block_one = nn.ModuleList()
        for i in range(config.block_sizes[0]):
            self.encoder_block_one.append(TransformerLayer(config, 0, True, i))

        self.encoder_block_two = nn.ModuleList()
        for i in range(config.block_sizes[1]):
            self.encoder_block_two.append(TransformerLayer(config, 1, True, i))

        self.stride = config.stride
        self.decoder_block = nn.ModuleList()

        if config.block_channel_size[0] != config.block_channel_size[1]:
            self.dim_match = nn.Sequential(nn.Linear(config.block_channel_size[0],config.block_channel_size[1], bias=False),
                                           nn.LayerNorm(config.block_channel_size[1], eps=config.layer_norm_eps))

        self.has_decoder = config.has_decoder
        if config.has_decoder:
            for i in range(config.num_decoder_layers):
                self.decoder_block.append(TransformerLayer(config, 0, False, i))
            self.dim_match_decoder_linear = nn.Linear(config.block_channel_size[1],config.block_channel_size[0], bias=False)
            self.dim_match_decoder = nn.ConvTranspose2d(config.block_channel_size[1], config.block_channel_size[0], config.stride ** (len(config.block_sizes) - 1), self.config.stride ** (len(self.config.block_sizes) - 1))
            self.dim_match_decoder_ln = nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)
        self.init_weights()

    def forward(self, x, run_decoder=False):
        config = self.config
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        H, W = self.patch_embed.height, self.patch_embed.width
        initail_attention = torch.ones(x.shape[:2], device=x.device)
        hidden = x
        for layer in self.encoder_block_one:
            hidden = layer(hidden, hidden, hidden, (None, initail_attention))
        first_block_hidden = hidden
        if config.block_channel_size[0] != config.block_channel_size[1]:
            hidden = self.dim_match(hidden)
        second_block_attention = initail_attention
        if self.stride > 1:
            assert H % self.stride == 0 and W % self.stride == 0
            cls, hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            hidden = hidden.reshape(B, H, W, config.block_channel_size[1]).permute(0, 3, 1, 2)
            cls_attention, second_block_attention = second_block_attention.split([config.num_highway_cls_tokens, second_block_attention.shape[1] - config.num_highway_cls_tokens], 1)
            second_block_attention = second_block_attention.reshape(B, H, W).unsqueeze(-1).permute(0, 3, 1, 2)

            hidden = F.avg_pool2d(hidden, self.stride, self.stride).flatten(2).permute(0, 2, 1)
            hidden = torch.cat((cls, hidden), 1)
            second_block_attention = F.max_pool2d(second_block_attention, self.stride, self.stride).permute(0, 2, 3, 1).flatten(1)
            second_block_attention = torch.cat((cls_attention, second_block_attention), 1)

        for layer in self.encoder_block_two:
            hidden = layer(hidden, hidden, hidden, (None, second_block_attention))
        second_block_hidden = hidden

        upsampled_hidden = None
        if run_decoder and self.has_decoder:
            cls, upsampled_hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            cls = self.dim_match_decoder_linear(cls)
            upsampled_hidden = upsampled_hidden.reshape(B, H // (config.stride ** (len(config.block_sizes) - 1)), W // (config.stride ** (len(config.block_sizes) - 1)), config.block_channel_size[1]).permute(0, 3, 1, 2)
            upsampled_hidden = self.dim_match_decoder(upsampled_hidden).flatten(2).permute(0, 2, 1)
            upsampled_hidden = self.dim_match_decoder_ln(torch.cat((cls, upsampled_hidden), 1))
            upsampled_hidden = upsampled_hidden + first_block_hidden
            hidden = self.dim_match_decoder_ln(self.dim_match_decoder_linear(hidden))

            for i, layer in enumerate(self.decoder_block):
                if i == 0:
                    upsampled_hidden = layer(upsampled_hidden, hidden, hidden, (None, second_block_attention))
                else:
                    upsampled_hidden = layer(upsampled_hidden, upsampled_hidden, upsampled_hidden, (None, initail_attention))
        return dict(first_block_hidden=first_block_hidden, second_block_hidden=second_block_hidden, third_block_hidden=upsampled_hidden)


class ClassificationModel(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_classes, num_features=768, ):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.backbone = backbone
        self.num_features = num_features
        self.head = nn.Linear(self.num_features, num_classes)
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)
        self.init_weights()

    def forward(self, x, labels=None):
        if isinstance(self.backbone, FastFormerVisionModel):
            output = self.backbone(x, run_decoder=True)
            representation = torch.cat((output["second_block_hidden"][:, :self.backbone.cls_tokens].mean(1), output["third_block_hidden"][:, :self.backbone.cls_tokens].mean(1)), 2)
        else:
            representation = self.backbone(x)
            if len(representation.size()) == 3:
                representation = representation[:, 0]
        logits = self.head(representation)
        loss = 0
        predictions = logits.detach().argmax(dim=-1)
        accuracy = None
        if labels is not None:
            loss = self.loss_ce(logits, labels)
            accuracy = (predictions == labels).float().mean().item()
        return dict(loss=loss, logits=logits, predictions=predictions, accuracy=accuracy)


class PatchCLR(FastFormerPreTrainedModel):
    def __init__(self, backbone, num_features=384, eps=1e-4, patchclr_w=1.0, contrastive_temperature=1e-2, simclr_w=1.0, clustering_w=1.0, gap_bias_w=1.0):
        super().__init__(backbone.config if hasattr(backbone, "config") else PretrainedConfig(initializer_std=1.0))
        self.backbone = backbone
        self.num_features = num_features
        self.loss_ce = CrossEntropyLoss(ignore_index=-100)
        self.ffn = nn.Sequential(nn.LayerNorm(num_features, eps=eps), nn.GELU(), nn.Linear(num_features, num_features))
        self.eps = eps
        self.contrastive_temperature = contrastive_temperature
        self.simclr_w = simclr_w
        self.clustering_w = clustering_w
        self.patchclr_w = patchclr_w
        self.gap_bias_w = gap_bias_w
        self.init_weights()

    def calculate_contrastive_loss(self, contrastive_matrix, label_lengths):
        contrastive_matrix = contrastive_matrix / self.contrastive_temperature
        mask = contrastive_matrix.new_zeros(contrastive_matrix.size(), requires_grad=False).fill_diagonal_(1e3)
        contrastive_matrix = contrastive_matrix - mask
        labels = torch.cat((torch.arange(label_lengths, device=contrastive_matrix.device) + label_lengths, torch.arange(label_lengths, device=contrastive_matrix.device)))
        loss = self.loss_ce(contrastive_matrix, labels)
        predictions = contrastive_matrix.detach().argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return loss, accuracy

    def forward(self, x1, x2, extra_negative_repr_patchclr=None, extra_negative_repr_simclr=None):
        if isinstance(self.backbone, FastFormerVisionModel):
            b1 = self.backbone(x1, run_decoder=True)
            b2 = self.backbone(x2, run_decoder=True)
        else:
            b1 = self.ffn(self.backbone(x1))
            b2 = self.ffn(self.backbone(x2))
        if isinstance(b1, dict):
            b1 = self.ffn(b1["third_block_hidden"] if b1["third_block_hidden"] is not None else b1["second_block_hidden"])  # B,S,D
            b2 = self.ffn(b2["third_block_hidden"] if b2["third_block_hidden"] is not None else b2["second_block_hidden"])  # B,S,D

        b, s = b1.shape[:2]
        bs = b * s

        patchclr_loss = 0.0
        patchclr_accuracy = None
        if self.patchclr_w > 0:
            out_1 = b1.reshape(-1, self.num_features)  # BxS , D
            out_2 = b2.reshape(-1, self.num_features)  # BxS , D

            c1 = torch.cat((out_1, out_2), 0)
            c1 = c1 / (c1.norm(2, -1, True).detach() + self.eps)
            # b2 = torch.cat((out_2, out_1), 0)
            contrastive_matrix = c1.mm(c1.t()) * (1 - torch.eye(c1.size(0), c1.size(0), device=c1.device))
            contrastive_matrix_store = contrastive_matrix

            if extra_negative_repr_patchclr is not None:
                patchclr_negative = c1.mm(extra_negative_repr_patchclr.t())
                contrastive_matrix = torch.cat((contrastive_matrix, patchclr_negative), 1)
                if extra_negative_repr_patchclr.size(0) >= 2 * c1.size(0):
                    extra_negative_repr_patchclr = torch.cat((extra_negative_repr_patchclr[c1.size(0):], c1.detach()), 0)
                else:
                    extra_negative_repr_patchclr = torch.cat((extra_negative_repr_patchclr, c1.detach()), 0)
            else:
                extra_negative_repr_patchclr = c1.detach()

            patchclr_loss, patchclr_accuracy = self.calculate_contrastive_loss(contrastive_matrix, out_1.shape[0])
            patchclr_loss = self.patchclr_w * patchclr_loss
        clustering_loss = 0.0
        if self.clustering_w > 0 and self.patchclr_w > 0:
            cmm = contrastive_matrix_store.reshape(2, bs, 2, bs).transpose(1,2).reshape(4, bs, bs)
            cmm2 = cmm.reshape(4, b, s, b, s).transpose(2, 3).reshape(4, b, b, -1).mean(-1)
            should_be_similar = torch.diagonal(cmm2, dim1=1, dim2=2)
            clustering_loss = self.clustering_w * ((4*b - (2*b/s)) + cmm2.sum() - 2 * should_be_similar.sum()) / (math.prod(cmm2.size()) * 0.5)

        simclr_loss = 0.0
        simclr_accuracy = None
        if self.simclr_w > 0:
            if hasattr(self.backbone, "cls_tokens"):
                b1s = b1[:, :self.backbone.cls_tokens].mean(1)  # B, D
                b2s = b2[:, :self.backbone.cls_tokens].mean(1)  # B, D
            else:
                b1s = b1[:, 0]
                b2s = b2[:, 0]
            sc1 = torch.cat((b1s, b2s), 0)
            sc1 = sc1 / (sc1.norm(2, -1, True).detach() + self.eps)
            contrastive_matrix = sc1.mm(sc1.t()) * (1 - torch.eye(sc1.size(0), sc1.size(0), device=sc1.device))

            if extra_negative_repr_simclr is not None:
                simclr_negative = sc1.mm(extra_negative_repr_simclr.t())
                contrastive_matrix = torch.cat((contrastive_matrix, simclr_negative), 1)
                if extra_negative_repr_simclr.size(0) >= 8 * sc1.size(0):
                    extra_negative_repr_simclr = torch.cat((extra_negative_repr_simclr[sc1.size(0):], sc1.detach()), 0)
                else:
                    extra_negative_repr_simclr = torch.cat((extra_negative_repr_simclr, sc1.detach()), 0)
            else:
                extra_negative_repr_simclr = sc1.detach()

            simclr_loss, simclr_accuracy = self.calculate_contrastive_loss(contrastive_matrix, b1s.shape[0])
            simclr_loss = self.simclr_w * simclr_loss

        gap_bias_loss = 0.0
        if self.gap_bias_w > 0 and self.simclr_w > 0 and self.patchclr_w > 0:
            if hasattr(self.backbone, "cls_tokens"):
                p1s = b1[:, self.backbone.cls_tokens:].mean(1)  # B, D
                p2s = b2[:, self.backbone.cls_tokens:].mean(1)  # B, D
            else:
                p1s = b1[:, 1:].mean(1)
                p2s = b2[:, 1:].mean(1)

            pc1 = torch.cat((p1s, p2s), 0)
            pc1 = pc1 / (pc1.norm(2, -1, True).detach() + self.eps)

            gap_bias = torch.cat((sc1, pc1), 0)
            contrastive_matrix = gap_bias.mm(gap_bias.t()) * (1 - torch.eye(gap_bias.size(0), gap_bias.size(0), device=sc1.device))
            if extra_negative_repr_simclr is not None:
                simclr_negative = gap_bias.mm(extra_negative_repr_simclr.t())
                contrastive_matrix = torch.cat((contrastive_matrix, simclr_negative), 1)

            gap_bias_loss, gap_bias_accuracy = self.calculate_contrastive_loss(contrastive_matrix, pc1.shape[0])
            gap_bias_loss = self.gap_bias_w * gap_bias_loss

        # TODO: GAP bias SIMCLR
        # SimCLR Loss subset of patch clr
        # TODO: patch clr with previous batch vectors
        # TODO: Reduce SIMCLR weight to zero slowly
        # TODO: Additive Margin Softmax to make the task harder.

        loss = patchclr_loss + clustering_loss + simclr_loss + gap_bias_loss
        return dict(loss=loss, patchclr_loss=patchclr_loss, clustering_loss=clustering_loss, simclr_loss=simclr_loss, gap_bias_loss=gap_bias_loss,
                    patchclr_accuracy=patchclr_accuracy, simclr_accuracy=simclr_accuracy, gap_bias_accuracy=gap_bias_accuracy, extra_negative_repr_patchclr=extra_negative_repr_patchclr.detach(), extra_negative_repr_simclr=extra_negative_repr_simclr.detach())


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
            model = PatchCLR(model, 768, 1e-7, simclr_w=0.0)
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
            model = PatchCLR(model, config.block_channel_size[0] if config.has_decoder else config.block_channel_size[1], 1e-7, simclr_w=0.0)
    model = model.to(device)
    if args["classification"]:
        output = model(x)
        print(output.argmax(-1))
        exit()

    output = model(x, x)

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















