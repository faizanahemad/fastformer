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

        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
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
    def __init__(self, d_model):
        super().__init__()
        
        d_inner = d_model * 4
        self.d_model = d_model
        activation_dropout = Dropout(config.hidden_dropout)

        self.layer_norm = nn.LayerNorm(d_model, config.layer_norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_inner, bias=True), activation_dropout, nn.GELU(), nn.Linear(d_inner, d_model, bias=False), activation_dropout)

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
        self.ffn = PositionwiseFFN(config.block_channel_size[block_index])
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
        for i in range(config.num_decoder_layers):
            self.decoder_block.append(TransformerLayer(config, 0, False, i))
        if config.block_channel_size[0] != config.block_channel_size[1]:
            self.dim_match = nn.Sequential(nn.Linear(config.block_channel_size[0],config.block_channel_size[1], bias=False),
                                           nn.LayerNorm(config.block_channel_size[1], eps=config.layer_norm_eps))
        self.dim_match_decoder_linear = nn.Linear(config.block_channel_size[1],config.block_channel_size[0], bias=False)
        self.dim_match_decoder = nn.ConvTranspose2d(config.block_channel_size[1], config.block_channel_size[0], config.stride ** (len(config.block_sizes) - 1), self.config.stride ** (len(self.config.block_sizes) - 1))
        self.dim_match_decoder_ln = nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)

    def forward(self, x, run_decoder=False):
        config = self.config
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        H = W = x.shape[1]
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
            cls_attention, second_block_attention = second_block_attention.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            second_block_attention = second_block_attention.reshape(B, H, W).unsqueeze(-1).permute(0, 3, 1, 2)

            hidden = F.avg_pool2d(hidden, self.stride, self.stride).permute(0, 2, 3, 1).flatten(2)
            hidden = torch.cat((cls, hidden), 1)
            second_block_attention = F.max_pool2d(second_block_attention, self.stride, self.stride).permute(0, 2, 3, 1).flatten(2).squeeze(-1)
            second_block_attention = torch.cat((cls_attention, second_block_attention), 1)

        for layer in self.encoder_block_two:
            hidden = layer(hidden, hidden, hidden, (None, second_block_attention))
        second_block_hidden = hidden

        upsampled_hidden = None
        if run_decoder:
            cls, upsampled_hidden = hidden.split([config.num_highway_cls_tokens, hidden.shape[1] - config.num_highway_cls_tokens], 1)
            cls = self.dim_match_decoder_linear(cls)
            upsampled_hidden = upsampled_hidden.reshape(B, H // (config.stride ** (len(config.block_sizes) - 1)), W // (config.stride ** (len(config.block_sizes) - 1)), config.block_channel_size[1]).permute(0, 3, 1, 2)
            upsampled_hidden = self.dim_match_decoder(upsampled_hidden).permute(0, 2, 3, 1).flatten(2)
            upsampled_hidden = self.dim_match_decoder_ln(torch.cat((cls, hidden), 1))
            upsampled_hidden = upsampled_hidden + first_block_hidden
            hidden = self.dim_match_decoder_ln(self.dim_match_decoder_linear(hidden))

            for i, layer in enumerate(self.decoder_block):
                if i == 0:
                    upsampled_hidden = layer(upsampled_hidden, hidden, hidden, (None, second_block_attention))
                else:
                    upsampled_hidden = layer(upsampled_hidden, upsampled_hidden, upsampled_hidden, (None, initail_attention))
        return dict(first_block_hidden=first_block_hidden, second_block_hidden=second_block_hidden, third_block_hidden=upsampled_hidden)


if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224)
    config = vision_md_config
    model = FastFormerVisionModel(config)
    output = model(x)
    print(output)













