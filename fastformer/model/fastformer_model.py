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

logger = logging.get_logger(__name__)


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: FastFormerConfig):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        hidden_size = config.block_channel_size[0]
        self.hidden_size = hidden_size
        self.embedding_size = config.embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings * (2 if config.relative_attention[0] else 1) + (2 if config.relative_attention[0] else 0), self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if config.char_rnn:
            char_div = 1
            char_rnn_layers = config.char_rnn_layers
            char_rnn_vocab_size = config.char_rnn_vocab_size
            self.char_embeddings = nn.Embedding(char_rnn_vocab_size, self.embedding_size // (2 * char_div), padding_idx=pad_token_id)
            self.char_rnn = ShortSeqRNN(config, self.embedding_size // (2 * char_div), 1, self.embedding_size // (2 * char_div),
                                        config.char_rnn_window_size, config.char_rnn_window_overlap, char_rnn_layers, maintain_dim=True)
            self.char_proj = nn.Linear(self.embedding_size // (2 * char_div), self.hidden_size, bias=False)
            self.char_div = char_div

        self.embed_proj = nn.Identity()
        if self.embedding_size != hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, hidden_size, bias=False)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.LayerNormPosEmb = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps) if config.separate_content_and_position_attention else nn.Identity()
        self.dropout = Dropout(config.hidden_dropout)
        self.output_to_half = False
        self.config = config
        self.highway_embeds = nn.Parameter(torch.zeros(1, config.num_highway_cls_tokens, self.hidden_size))

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, inputs_embeds=None, token_type_ids=None, position_ids=None, char_ids=None, char_offsets=None,
                use_embed_proj=True, use_highway_embeds=True):
        if inputs_embeds is None:
            input_shape = input_ids.size()
            input_shape = list(input_shape)
            initial_seq_len = input_shape[1]
            if use_highway_embeds:
                input_shape[1] = input_shape[1] + self.config.num_highway_cls_tokens
            input_shape = tuple(input_shape)
            inputs_embeds = self.word_embeddings(input_ids)
            input_shape = inputs_embeds.size()
        seq_length = input_shape[1]
        embeddings = inputs_embeds
        if use_embed_proj:
            embeddings = self.embed_proj(embeddings)

        if self.config.char_rnn and char_ids is not None:
            char_offsets = char_offsets.flatten(1, 2).unsqueeze(-1).expand(input_shape[0], -1, self.embedding_size // (2 * self.char_div))
            char_embeds = self.char_rnn(self.char_embeddings(char_ids))
            char_embeds = torch.gather(char_embeds, 1, char_offsets).view(input_shape[0], initial_seq_len, 2, self.embedding_size // (2 * self.char_div)).mean(2)
            char_embeds = self.char_proj(char_embeds) if use_embed_proj else char_embeds
            embeddings = embeddings + char_embeds

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids.long())
        if self.position_biased_input:
            embeddings += (self.embed_proj(position_embeddings) if use_embed_proj else position_embeddings)

        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += (self.embed_proj(token_type_embeddings) if use_embed_proj else token_type_embeddings)

        if self.config.num_highway_cls_tokens > 0 and use_highway_embeds:
            embeddings = torch.cat((self.highway_embeds.expand(embeddings.size(0), -1, -1)[:, :embeddings.size(1)], embeddings), 1)
        embeddings = self.LayerNorm(embeddings) if use_embed_proj else embeddings

        embeddings = self.dropout(embeddings) if use_embed_proj else embeddings
        return embeddings, self.LayerNormPosEmb(self.position_embeddings.weight if self.config.relative_attention else None) if position_embeddings is not None else None


class MultiheadAttention(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                 last_layer_index=None, force_no_sdconv=False):
        super().__init__()
        if last_layer_index is None:
            last_layer_index = layer_index
        self.config = config
        self.relative_attention = config.relative_attention[block_index]
        self.block_index = block_index
        d_model, all_head, d_head = config.block_channel_size[block_index], config.n_head[block_index], config.d_head[block_index]
        d_model_initial = d_model
        n_head = all_head[0]
        total_heads = sum(all_head)
        assert d_model % 16 == 0
        remaining_d_model = d_model
        self.sdconv = config.sdconv[block_index] and not force_no_sdconv
        self.full_channel_separation = config.full_channel_separation[block_index]
        if self.sdconv:
            self.n_conv_head = all_head[1]
            if self.full_channel_separation:
                self.conv_dims = (self.n_conv_head * d_model) // total_heads
                remaining_d_model -= self.conv_dims
            else:
                self.conv_dims = d_model
            self.sdconv = SDConv(config, self.conv_dims, self.n_conv_head, d_head, config.sdconv_kernel_size[block_index])

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

        self.approximate_attention = self.config.approximate_attention[block_index]

        qkv_transform_groups = self.config.qkv_transform_groups
        if qkv_transform_groups > 1:
            # assert n_head % qkv_transform_groups == 0 and n_head >= qkv_transform_groups
            self.q_head = Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)
            self.k_head = Conv1d(
                in_channels=d_model, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)

            self.v_head = Conv1d(in_channels=d_model_initial, out_channels=d_model_initial, kernel_size=1, groups=qkv_transform_groups)

        else:
            self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
            self.k_head = nn.Linear(d_model, n_head * d_head, bias=False)


            self.v_head = nn.Linear(d_model_initial, d_model_initial)

        if config.no_v_head:
            self.v_head = nn.Identity()

        if self.approximate_attention:
            self.attn = FastAttention(dim_heads=d_head, nb_features=n_head * d_head, )
        if self.relative_attention:
            if qkv_transform_groups > 1:
                self.pos_q_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1,
                                         groups=qkv_transform_groups, stride=1, bias=True)
                self.pos_k_head = Conv1d(in_channels=config.embedding_size, out_channels=n_head * d_head, kernel_size=1, groups=qkv_transform_groups, bias=False)
            else:
                self.pos_q_head = nn.Linear(config.embedding_size, n_head * d_head, bias=True)
                self.pos_k_head = nn.Linear(config.embedding_size, n_head * d_head, bias=False)

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.layer_norm = nn.LayerNorm(config.block_channel_size[block_index], eps=config.layer_norm_eps)
        self.scale_factor = 1 + (2 if self.relative_attention else 0)
        self.scale = 1.0 / ((d_head ** 0.5)*(self.scale_factor ** 0.5))

    def self_attention(self, query, key, value, attention_inputs, layer_index):
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
        key_stride = (self.config.stride ** self.block_index) if layer_index != 0 else (
            self.config.stride ** max(self.block_index - 1, 0) if self.is_encoder_layer else self.config.stride ** (len(self.config.block_sizes) - 1))
        # Shape n_head x d_head
        # Shapes batch_size x n_head x seq_len x context_len
        if self.relative_attention:
            position_embed_of_query = position_embed_of_key = position_embeds
            if self.relative_attention:
                pos_k_head = self.pos_k_head(position_embed_of_key)
                pos_q_head = self.pos_q_head(position_embed_of_query)
            else:

                pos_k_head = self.pos_k_head(position_embed_of_key).view(context_len, n_head, d_head)
                pos_q_head = self.pos_q_head(position_embed_of_query).view(seq_len, n_head, d_head)
                # print(query.size(), key.size(), position_embed_of_query.size(), position_embed_of_key.size())

        if self.approximate_attention:
            # TODO: how to handle attention masks
            v_head = v_head.permute(0, 2, 1, 3)
            attn_vec = self.attn(q_head.permute(0, 2, 1, 3), k_head.permute(0, 2, 1, 3), v_head).permute(0, 2, 1, 3)
            assert not self.relative_attention
            if self.relative_attention:
                pos_q_head = pos_q_head / math.sqrt(pos_q_head.size(-1) * self.scale_factor)
                c2p_score = self.attn((q_head).permute(0, 2, 1, 3), pos_k_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                                                       3)
                p2c_score = self.attn(pos_q_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), (k_head).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                                                       3)
                p2p_score = self.attn(pos_q_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3),
                                      pos_k_head.expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3), v_head).permute(0, 2, 1,
                                                                                                                     3)
                nc_score = c2p_score + p2c_score + p2p_score
                nc_score = torch.cat((nc_score.new_zeros(nc_score.size(0), nc_score.size(1), self.cls_tokens, nc_score.size(-1)),
                                      torch.cat((nc_score.new_zeros(nc_score.size(0), nc_score.size(1), nc_score.size(-2) - self.cls_tokens, self.cls_tokens),
                                                 nc_score[..., self.cls_tokens:, self.cls_tokens:]), -1)), -2)
                attn_vec = attn_vec + nc_score
                # TODO: try adaptive weighting

        else:
            content_score = torch.einsum("bind,bjnd->bnij", q_head, k_head)
            attn_score = content_score
            if self.relative_attention:
                nc_score = disentangled_att_bias(q_head[:, self.cls_tokens:].transpose(1, 2), k_head[:, self.cls_tokens:].transpose(1, 2),
                                                 None,
                                                 pos_q_head, pos_k_head, self.scale_factor,
                                                 self.config.max_position_embeddings,
                                                 self.config.max_position_embeddings,
                                                 n_head,
                                                 query_stride=self.config.stride ** self.block_index,
                                                 key_stride=key_stride)

                nc_score = F.pad(nc_score, (self.cls_tokens, 0, self.cls_tokens, 0, 0, 0, 0, 0))
                attn_score = attn_score + nc_score

            # precision safe in case of mixed precision training
            dtype = attn_score.dtype
            # attn_score = attn_score.float()
            # perform masking
            if attention_mask is not None:
                if key_stride > 1:
                    attention_mask = pool_tensor(attention_mask, self.cls_tokens, mode='min', stride=key_stride)
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

        return attn_out, attn_prob

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        assert query.size(1) > 0
        batch_size, seq_len, _ = query.shape
        query_temp = query
        value = value if value is not None else query
        value = self.v_head(value)
        if self.sdconv:

            if self.full_channel_separation:
                if key is None:
                    qp1, query = query.split([self.conv_dims, query.size(-1) - self.conv_dims], -1)
                    kp1, _ = qp1, query
                else:
                    qp1, query = query.split([self.conv_dims, query.size(-1) - self.conv_dims], -1)
                    kp1, key = key.split([self.conv_dims, key.size(-1) - self.conv_dims], -1)
                vp1, value = value.split([self.conv_dims, value.size(-1) - self.conv_dims], -1)
                sdconv_out = self.sdconv(qp1, kp1, vp1)
            else:
                if key is None:
                    key = query
                sdconv_out = self.sdconv(query, key, value)


        try:
            if key is None and value is None:
                key = query
                value = query
            attn_out, attn_prob = self.self_attention(query, key, value, attention_inputs, layer_index)
        except Exception as e:
            print("[Exception-in-train]: Query Shape = %s, Key shape = %s, Value shape = %s, layer_index = %s, block index = %s" % (query.size(), key.size(), value.size(), layer_index, self.block_index))
            # traceback.print_exc()
            # traceback.print_exception(*sys.exc_info())
            raise e

        if self.sdconv:
            attn_out = torch.cat([sdconv_out, attn_out], dim=-1)
        if self.config.identity_preserving_norm:
            output = query_temp + self.layer_norm(attn_out)
        else:
            output = self.layer_norm(query_temp + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class TransformerLayer(nn.Module):
    def __init__(self, config: FastFormerConfig, block_index, is_last_layer_of_block, is_first_layer_of_block,
                 is_encoder_layer, layer_index, last_layer_index=None):
        super().__init__()
        self.attention = MultiheadAttention(config, block_index, is_last_layer_of_block, is_first_layer_of_block, is_encoder_layer, layer_index,
                                            last_layer_index)
        self.ffn = PositionwiseFFN(config, block_index, is_last_layer_of_block, is_encoder_layer)
        self.block_index = block_index
        self.block_size = config.block_sizes[block_index]
        self.is_last_layer_of_block = is_last_layer_of_block

    def forward(self, query, key, value, attention_inputs, layer_index, output_attentions=False):
        assert query.size(1) > 0
        attn = self.attention(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
        pre_ffn, output = self.ffn(attn[0], layer_index)
        pre_ffn = pre_ffn if self.block_size - 1 == layer_index else None
        return (output, pre_ffn, attn[1]) if output_attentions else (output, pre_ffn)


class TransformerEncoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
        self.config = config

        block_channel_size = config.block_channel_size
        self.blocks = nn.ModuleList()
        self.repeats = []
        for block_index, block_size in enumerate(config.block_sizes):
            cur_channels = block_channel_size[block_index]
            next_channels = block_channel_size[min(block_index + 1, len(block_channel_size) - 1)]
            self.blocks.append(nn.ModuleList())
            self.repeats.append([])
            i = 0
            while i < block_size:
                if config.block_repeats:

                    if i == 0 and config.separate_compressiion_layer and block_index > 0:
                        inext = i + 1
                        self.blocks[block_index].append(fsdp_wrapper(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i)))
                        self.repeats[block_index].append(1)
                        i = inext
                    elif i < block_size - 1:
                        reps = ((block_size - 1) - (i)) if cur_channels != next_channels else ((block_size - 1) - i)
                        inext = i + reps
                        self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i + reps))
                        self.repeats[block_index].append(reps)
                        i = inext
                    else:
                        inext = i + 1
                        self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
                        self.repeats[block_index].append(1)
                        i = inext

                else:
                    inext = i + 1
                    self.blocks[block_index].append(TransformerLayer(config, block_index, (inext - 1) == block_size - 1, i == 0, True, i, i))
                    self.repeats[block_index].append(1)
                    i = inext
        self.pool = None
        if config.pooling_type == 'learn_sdconv':
            CompressionClass = CompressSeqSDConv
        elif config.pooling_type == 'mean':
            CompressionClass = CompressSeqMeanPooling
        if config.pooling_type in ['mean', 'learn_sdconv'] and config.stride > 1:
            pool = nn.ModuleDict()
            for block_index, _ in enumerate(config.block_sizes[1:]):
                bi = block_index + 1
                pool[str(block_index + 1)] = fsdp_wrapper(CompressionClass(config, bi, config.block_channel_size[bi], sum(config.n_head[bi]), use_in_funnel=True))
            self.pool = pool

    def forward_one_block(self, block_index, hidden, attention_inputs,
                          all_hidden_states, pre_ffn_states, all_attentions,
                          output_attentions=False, output_hidden_states=False):
        assert hidden.size(1) > 0
        (block, repeat_block) = self.blocks[block_index], self.repeats[block_index]
        pooling_flag = hidden.size(1) > 2
        pooling_flag = pooling_flag and block_index > 0 and self.config.stride > 1
        if pooling_flag:
            if self.pool:
                pooled_hidden = self.pool[str(block_index)](hidden)
        layer_index = 0
        for (_, (layer, repeats)) in enumerate(zip(block, repeat_block)):
            for repeat_index in range(repeats):
                do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                if do_pooling:
                    query = pooled_hidden
                    key = value = hidden
                else:
                    query = key = value = hidden
                # print("Block = ", block_index, ", Layer = ", layer_index, ", Sizes = ", query.size(), key.size(), value.size())
                layer_output = layer(query, key, value, attention_inputs, layer_index, output_attentions=output_attentions)
                layer_index += 1
                hidden = layer_output[0]
                pre_ffn = layer_output[1]

                if output_attentions:
                    all_attentions = all_attentions + layer_output[1:]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden,)
                    pre_ffn_states = pre_ffn_states + (pre_ffn,)

        return hidden, all_hidden_states, pre_ffn_states, all_attentions, attention_inputs

    def forward_first_block(self,
                            inputs_embeds,
                            position_embeds,
                            attention_mask,):
        assert inputs_embeds.size(1) > 0
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = position_embeds, attention_mask
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,)
        pre_ffn_states = (inputs_embeds,)
        all_attentions = None
        one_block_res = self.forward_one_block(0, hidden, attention_inputs, all_hidden_states, pre_ffn_states, all_attentions, False,
                                                   True)
        hidden, all_hidden_states, pre_ffn_states, all_attentions, attention_inputs = one_block_res

        # attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs, block_index) if self.config.stride > 1 else attention_inputs
        # block_attention_masks.append(attention_inputs[1])
        return hidden, pre_ffn_states[-1]

    def forward(
            self,
            inputs_embeds,
            position_embeds,
            attention_mask,
            output_attentions=False,
            output_hidden_states=False,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        assert inputs_embeds.size(1) > 0
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = position_embeds, attention_mask
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        pre_ffn_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        block_attention_masks = []
        block_attention_inputs = []
        for block_index, (_, _) in enumerate(zip(self.blocks, self.repeats)):
            # print("Block = ", block_index, ", Sizes = ", hidden.size(), attention_mask[0].size(), attention_mask[1].size(), all_hidden_states[-1].size())
            one_block_res = self.forward_one_block(block_index, hidden, attention_inputs, all_hidden_states, pre_ffn_states, all_attentions, output_attentions, output_hidden_states)
            hidden, all_hidden_states, pre_ffn_states, all_attentions, attention_inputs = one_block_res
            block_attention_masks.append(attention_inputs[1])
            block_attention_inputs.append(attention_inputs)

        # attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs, block_index) if self.config.stride > 1 else attention_inputs
        # block_attention_masks.append(attention_inputs[1])
        return tuple(v for v in [hidden, all_hidden_states, pre_ffn_states, all_attentions, block_attention_inputs, block_attention_masks] if v is not None)


def shift_right(input_ids, decoder_start_token_id, pad_token_id):

    assert (
        decoder_start_token_id is not None
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    # assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


def shift_right_fast(input_ids, decoder_start_token_id, pad_token_id):
    return torch.cat((input_ids.new_zeros((input_ids.shape[0], 1)), input_ids[:, :-1]), 1)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class TransformerDecoder(nn.Module):
    def __init__(self, config: FastFormerConfig):
        super().__init__()
        config = copy.deepcopy(config)
        config.sdconv = [False] * len(config.sdconv)
        self.config = config
        self.cls_tokens = self.config.num_highway_cls_tokens + 1
        self.layers = nn.ModuleList()
        if config.num_decoder_layers > 0:
            if config.block_repeats:
                self.layers.extend([TransformerLayer(config, 0, True, True, False, 0)])
                self.repeats = [config.num_decoder_layers]
            else:

                self.layers.extend(
                    [TransformerLayer(config, 0, i == config.num_decoder_layers - 1, i == 0, False, i) for i in range(config.num_decoder_layers)])
                self.repeats = [1] * config.num_decoder_layers

        else:
            self.layers = None

        block_channel_size = self.config.block_channel_size

    def forward(
            self,
            final_hidden,
            first_block_hidden,
            attention_inputs=None,
            final_hidden_attention_inputs=None,
            output_attentions=False,
            output_hidden_states=False,
    ):

        if not self.layers:
            hidden = final_hidden
            all_hidden_states = (hidden,) if output_hidden_states else None
            all_attentions = () if output_attentions else None
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)

        upsampled_hidden = upsample(
            final_hidden,
            stride=self.config.stride ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            cls_tokens=self.cls_tokens,
        )

        hidden = upsampled_hidden + first_block_hidden

        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None


        layer_index = 0
        for layer, repeats in zip(self.layers, self.repeats):
            for _ in range(repeats):
                layer_output = layer(hidden, final_hidden, final_hidden, final_hidden_attention_inputs, layer_index, output_attentions=output_attentions)
                hidden = layer_output[0]
                layer_index += 1
                layer_output = layer(hidden, hidden, hidden, attention_inputs, layer_index, output_attentions=output_attentions)
                hidden = layer_output[0]
                layer_index += 1

                if output_attentions:
                    all_attentions = all_attentions + layer_output[1:]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden,)

        return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)


class DiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config: FastFormerConfig):
        super().__init__()
        self.config = config
        self.dense = Conv1d(config.block_channel_size[0], config.block_channel_size[0] // 2, 1, config.ffn_groups, bias=False) if config.ffn_groups > 1 else nn.Linear(config.block_channel_size[0], config.block_channel_size[0] // 2, bias=False)
        self.dense_prediction = (nn.Linear(config.block_channel_size[0] // 2, 1))
        self.act = checkpoint_wrapper(ACT2FN[self.config.hidden_act](), offload_to_cpu=False)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class FastFormerPreTrainedModel(PreTrainedModel):
    config_class = FastFormerConfig
    base_model_prefix = "backbone"

    def _init_weights(self, module):
        from timm.models.layers.weight_init import trunc_normal_
        # print("[WARN] [FastFormerPreTrainedModel]: Time = %s, Init Weights, self type = %s" % (get_time_string(), type(self)))
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                trunc_normal_(module.weight, std=std)

            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)

        if classname.find("Conv1d") != -1 or classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
            if self.config.initializer_std is None:

                if getattr(module, "weight", None) is not None:
                    fan_out, fan_in = module.weight.shape[:2]
                    fan_out, fan_in = fan_out, fan_in
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
            else:
                std = self.config.initializer_std
            trunc_normal_(module.weight, std=std)

            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        elif classname == "MultiheadAttention":
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
        elif classname == "Embeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            trunc_normal_(module.word_embeddings.weight, std=std)
            trunc_normal_(module.position_embeddings.weight, std=std)
            if hasattr(module, "token_type_embeddings"):
                trunc_normal_(module.token_type_embeddings.weight, std=std)
            if hasattr(module, "char_embeddings"):
                trunc_normal_(module.char_embeddings.weight, std=std)
            if hasattr(module, "highway_embeds"):
                trunc_normal_(module.highway_embeds, std=std)
        elif classname == "PatchEmbed":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            if hasattr(module, "cls_token"):
                trunc_normal_(module.cls_token, std=std)
            if hasattr(module, "pos_embed"):
                trunc_normal_(module.pos_embed, std=std)
            if hasattr(module, "column_embed"):
                trunc_normal_(module.column_embed, std=std)
            if hasattr(module, "row_embed"):
                trunc_normal_(module.row_embed, std=std)
            if hasattr(module, "first_pos_embed"):
                trunc_normal_(module.first_pos_embed, std=std)
            if hasattr(module, "proj"):
                if self.config.initializer_std is None:
                    if getattr(module.proj, "weight", None) is not None:
                        fan_out, fan_in = module.proj.weight.shape[:2]
                        fan_out, fan_in = fan_out, fan_in
                        std = np.sqrt(1.0 / float(fan_in + fan_out))
                    else:
                        std = self.config.initializer_std
                trunc_normal_(module.proj.weight, std=std)

        elif classname == "FastAttention":
            if not hasattr(module, 'projection_matrix'):
                projection_matrix = module.create_projection(device=next(self.parameters()).device)
                module.register_buffer('projection_matrix', projection_matrix)


class FastFormerModel(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, tokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = fsdp_wrapper(TransformerDecoder(config))
        self.cls_tokens = config.num_highway_cls_tokens + 1

        block_channel_size = self.config.block_channel_size
        ffn_groups = self.config.ffn_groups
        self.final_hidden_fc = nn.Identity()
        if block_channel_size[0] != block_channel_size[-1]:
            if ffn_groups > 1:
                assert block_channel_size[0] % ffn_groups == 0 and block_channel_size[-1] % ffn_groups == 0
                self.final_hidden_fc = Conv1d(in_channels=block_channel_size[-1], out_channels=block_channel_size[0], kernel_size=1, groups=ffn_groups, bias=False)
            else:
                self.final_hidden_fc = nn.Linear(block_channel_size[-1], block_channel_size[0], bias=False)
            self.final_hidden_fc = fsdp_wrapper(nn.Sequential(self.final_hidden_fc, nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)))

        self.embed_proj_transpose = nn.Identity() if self.config.identity_preserving_norm else nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps)
        if config.embedding_size != config.block_channel_size[0]:
            ep = nn.Linear(config.block_channel_size[0], config.embedding_size, bias=True)
            # ep.weight = nn.Parameter(self.embeddings.embed_proj.weight.transpose(0, 1))
            if self.config.identity_preserving_norm:
                self.embed_proj_transpose = ep
            else:
                self.embed_proj_transpose = nn.Sequential(nn.LayerNorm(config.block_channel_size[0], eps=config.layer_norm_eps),
                                                          ep)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size + config.num_highway_cls_tokens)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward_lm_block(self, input_ids=None,
                         attention_mask=None,
                         token_type_ids=None,
                         inputs_embeds=None,
                         char_ids=None, char_offsets=None, ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        assert input_shape[1] > 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if self.config.num_highway_cls_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_shape[0], self.config.num_highway_cls_tokens, device=device), attention_mask], dim=1)
        inputs_embeds, position_embeds = self.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets, )
        return self.encoder.forward_first_block(inputs_embeds, position_embeds, attention_mask)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            char_ids=None, char_offsets=None,
            run_decoder=True,
            run_answering=True,
            bypass_embeddings=False,
            **kwargs,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        assert input_shape[1] > 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: deal with head_mask
        if self.config.num_highway_cls_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_shape[0], self.config.num_highway_cls_tokens, device=device), attention_mask], dim=1)

        if bypass_embeddings:
            assert inputs_embeds is not None
            position_embeds = kwargs.pop("position_embeds", None)
            assert position_embeds is not None
        else:
            inputs_embeds, position_embeds = self.embeddings(input_ids, inputs_embeds, token_type_ids, char_ids=char_ids, char_offsets=char_offsets,)
        assert inputs_embeds.size(1) > 0

        encoder_outputs = self.encoder(
            inputs_embeds,
            position_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        final_hidden = self.final_hidden_fc(encoder_outputs[0])
        outputs = dict(final_hidden=final_hidden, encoder_outputs=encoder_outputs, inputs_embeds=inputs_embeds, position_embeds=position_embeds, input_shape=input_shape)

        if hasattr(self, "decoder") and (run_decoder or run_answering):
            block_attention_inputs = encoder_outputs[-2]
            final_hidden_attention_inputs = list(block_attention_inputs[-1])
            decoder_outputs = self.decoder(
                final_hidden=final_hidden,
                first_block_hidden=encoder_outputs[2][self.config.block_sizes[0]],
                attention_inputs=block_attention_inputs[0],
                final_hidden_attention_inputs=final_hidden_attention_inputs,
                output_attentions=False,
                output_hidden_states=False,
            )
            outputs["decoder_outputs"] = decoder_outputs

        if run_answering:
            assert hasattr(self, "embed_proj_transpose")
            assert hasattr(self, "decoder")
            answering_hidden = self.embed_proj_transpose(decoder_outputs[0][:, self.cls_tokens: self.cls_tokens + 128])
            answering_logits = self.lm_head(answering_hidden)
            outputs["answering_logits"] = answering_logits
            outputs["answering_hidden"] = answering_hidden

        return outputs


class FastFormerForMaskedLM(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig):
        super().__init__(config)

        self.funnel = FastFormerModel(config)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.lm_dim_match = None
        self.cls_tokens = config.num_highway_cls_tokens + 1
        self.accuracy_hist = defaultdict(list)
        self.loss_ce = CrossEntropyLoss(ignore_index=config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            char_ids=None, char_offsets=None,
    ):

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            char_ids=char_ids, char_offsets=char_offsets,
        )

        last_hidden_state = outputs["decoder_outputs"][0]
        prediction_logits = None
        active_loss = attention_mask == 1

        masked_lm_loss = None
        if labels is not None:
            if self.lm_dim_match:
                last_hidden_state = self.funnel.embed_proj_transpose(last_hidden_state)
            prediction_logits = self.lm_head(last_hidden_state)
            loss_fct = self.loss_ce  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            predictions = prediction_logits.detach().argmax(dim=-1)
            labels = (labels == predictions).float()
            self.accuracy_hist["lm"].append(float(labels[active_loss].float().mean()))
            self.accuracy_hist['lm_loss'].append(float(masked_lm_loss))

        output = (prediction_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class FastFormerForClassification(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, num_classes, model, tokenizer=None,
                 additive_margin_softmax_w=0.3, reinit_backbone=False, train_backbone=True):
        if isinstance(config, FastFormerConfig):
            super().__init__(config)
        elif model is not None and hasattr(model, "config"):
            super().__init__(model.config)
        else:
            raise ValueError

        self.backbone: FastFormerModel = FastFormerModel(config, tokenizer) if model is None else model
        # if num_classes == 1:
        #     self.ce = BCELossFocal()
        # else:
        #     self.ce = AdMSoftmaxLoss(ignore_index=-100, m=additive_margin_softmax_w)

        if num_classes == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = CrossEntropyLoss(ignore_index=-100)
        self.num_features = config.block_channel_size[-1] if isinstance(config, FastFormerConfig) else (model.config.hidden_size if hasattr(model, "config") and hasattr(model.config, "hidden_size") else 768) * 4
        if train_backbone:
            self.head = nn.Sequential(nn.LayerNorm(self.num_features), nn.Dropout(0.1),
                                      nn.Linear(self.num_features, self.num_features // 4), nn.GELU(),
                                      nn.Linear(self.num_features // 4, num_classes))
        else:
            self.head = nn.Linear(self.num_features, num_classes)
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.train_backbone = train_backbone
        self.cls_tokens = model.cls_tokens if hasattr(model, "cls_tokens") else 1
        if reinit_backbone:
            self.init_weights()

        init_weights(self.head)

    def get_representations(self, input_ids, attention_mask, char_ids=None, char_offsets=None, label=None, token_type_ids=None):
        # TODO: support extra cls_tokens
        funnel_inputs = dict(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             char_ids=char_ids, char_offsets=char_offsets,
                             run_decoder=False,
                             run_answering=False)
        if isinstance(self.backbone, (FastFormerModel)):
            funnel_outputs = self.backbone(**funnel_inputs)
            funnel_outputs = funnel_outputs["encoder_outputs"][0][:, 0]
        else:
            funnel_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            if self.cls_tokens > 1:
                funnel_outputs = torch.cat((funnel_outputs["hidden_states"][-1][:, 0], funnel_outputs["hidden_states"][-1][:, 1], funnel_outputs["hidden_states"][-2][:, 0], funnel_outputs["hidden_states"][-2][:, 1]), -1)
            else:
                funnel_outputs = torch.cat((funnel_outputs["hidden_states"][-1][:, 0], funnel_outputs["hidden_states"][-2][:, 0], funnel_outputs["hidden_states"][-3][:, 0], funnel_outputs["hidden_states"][-4][:, 0]), -1)
            # funnel_outputs = torch.cat((funnel_outputs["pooler_output"] if "pooler_output" in funnel_outputs else funnel_outputs["hidden_states"][-1][:, 0], funnel_outputs["hidden_states"][-2][:, 0], funnel_outputs["hidden_states"][-3][:, 0], funnel_outputs["hidden_states"][-4][:, 0]), -1)
        return funnel_outputs

    def forward(self, input_ids, attention_mask, char_ids=None, char_offsets=None, label=None, token_type_ids=None, **kwargs):
        with torch.set_grad_enabled(self.train_backbone and self.training):
            funnel_outputs = self.get_representations(input_ids, attention_mask, char_ids, char_offsets, label, token_type_ids)
        if not self.train_backbone:
            funnel_outputs = funnel_outputs.detach()
        logits = self.head(funnel_outputs)
        loss = 0.0
        if label is not None and label.min() >= 0:
            loss = self.ce(logits.squeeze(-1) if logits.ndim > 2 or self.num_classes == 1 else logits, label.float() if self.num_classes == 1 else label.long())

        logits = logits.detach()
        if self.num_classes > 1:
            predictions = logits.argmax(-1)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 or self.num_classes == 1 else predictions
        else:
            predictions = torch.sigmoid(logits)
            predictions = predictions.squeeze(-1) if predictions.ndim > 1 or self.num_classes == 1 else predictions
        return dict(predictions=predictions, loss=loss)


from abc import ABC, abstractmethod


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss


def hook(grad):
    is_nan_inf = torch.logical_not(torch.isfinite(grad))
    if is_nan_inf.any():
        # print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Inf" % (get_time_string(), name_of_param))
        grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-2), grad)
        grad = torch.clamp_(grad, -1e1, 1e1)
        # grad = F.normalize(grad, 2, -1, eps=config.eps)
        # grad = grad / grad.norm(2, -1, True)
        return grad
    else:
        return None


class FastFormerForFusedELECTRAPretraining(FastFormerPreTrainedModel):
    def __init__(self, config: FastFormerConfig, model: FastFormerModel = None, tokenizer = None, aitm=False, alum=False,
                 adv_lm_w=1.0, adv_ascent_steps=1, aitm_clip_min=0.1, aitm_clip_max=0.9, adv_step_size=1e-3,
                 adv_epsilon=1e-2, aitm_noise_var=0.1, adv_w=1.0, alum_aitm_alternate=False,
                 input_cls_orthogonal_w=0.5,
                 electra_loss_w=1.0, lm_loss_w=1.0, sentence_order_prediction_w=1.0, contrastive_w=1.0, contrastive_temperature=5e-2,
                answering_lm_w=1.0, additive_margin_softmax_w=0.3):
        super().__init__(config)
        self.data_parallel = False
        self.config = config
        self.tokenizer = copy.deepcopy(tokenizer)
        self.funnel: FastFormerModel = FastFormerModel(config, tokenizer) if model is None else model
        self.cls_tokens = config.num_highway_cls_tokens
        self.discriminator_predictions = fsdp_wrapper(DiscriminatorPredictions(config))
        self.pad_token_id = config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id is not None else 0
        if additive_margin_softmax_w == 0:
            self.ce = CrossEntropyLoss(ignore_index=-100)
            self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
            self.ignore_zero_ce = CrossEntropyLoss(ignore_index=0)
            self.loss_bce = nn.BCEWithLogitsLoss()
        else:
            self.ce = AdMSoftmaxLoss(ignore_index=-100, m=additive_margin_softmax_w)
            self.loss_ce = AdMSoftmaxLoss(ignore_index=self.pad_token_id, m=additive_margin_softmax_w)
            self.ignore_zero_ce = AdMSoftmaxLoss(ignore_index=0, m=additive_margin_softmax_w)
            self.loss_bce = BCELossFocal()
        self.sentence_order_prediction_w = sentence_order_prediction_w
        if sentence_order_prediction_w > 0:
            self.sent_predict_fc = nn.Linear(config.block_channel_size[0], (self.cls_tokens + 1))

        if contrastive_w > 0:
            self.contrastive_ffn = nn.Sequential(nn.Linear(config.block_channel_size[0], config.block_channel_size[0] // 2, bias=True),
                                                 checkpoint_wrapper(ACT2FN[self.config.hidden_act](), offload_to_cpu=False), nn.Linear(config.block_channel_size[0] // 2, 128, bias=False),)

        self.alum_aitm_alternate = alum_aitm_alternate
        self.lm_loss_w = lm_loss_w
        self.input_cls_orthogonal_w = input_cls_orthogonal_w
        self.electra_loss_w = electra_loss_w
        self.reccord_loss = False
        self.record_accuracy = False
        self.timing_hist = list()
        self.aitm = aitm
        self.alum = alum
        self.aitm_noise_var = aitm_noise_var
        self.aitm_clip_min = aitm_clip_min
        self.aitm_clip_max = aitm_clip_max
        self.adv_lm_w = adv_lm_w
        self.adv_step_size = adv_step_size
        self.adv_epsilon = adv_epsilon
        self.adv_ascent_steps = adv_ascent_steps
        self.adv_w = adv_w
        self.answering_lm_w = answering_lm_w
        self.contrastive_w = contrastive_w
        self.contrastive_temperature = contrastive_temperature
        self.additive_margin_softmax_w = additive_margin_softmax_w
        if alum_aitm_alternate:
            pass
        elif aitm:
            self.adv_lm_w = -1 * abs(self.adv_lm_w)
            assert not alum
        elif alum:
            assert not aitm
            self.adv_lm_w = abs(self.adv_lm_w)
            self.aitm_clip_min = self.aitm_clip_max = 1.0
        self.init_weights()

    def get_output_embeddings(self):
        return None

    def adv_project(self, grad, norm_type='inf', eps=1e-4):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def aitm_loss(self, funnel_inputs, funnel_outputs, mlm_predictions, mlm_correct,
                  sent_order_predictions, electra_predictions,
                  labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
                  contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None,
                  reverse_loss=False):
        funnel_inputs = dict(**funnel_inputs)
        funnel_inputs["input_embeds"] = funnel_outputs["input_embeds"]
        funnel_inputs["position_embeds"] = funnel_outputs["position_embeds"]
        funnel_inputs["bypass_embeddings"] = True
        active_loss = funnel_inputs["attention_mask"].bool()
        new_funnel_outputs = self.funnel(funnel_inputs)
        encoder_outputs = new_funnel_outputs["encoder_outputs"]
        first_block_hidden = encoder_outputs[1][self.config.block_sizes[0] - 1]
        first_block_hidden = self.funnel.embed_proj_transpose(first_block_hidden[:, self.cls_tokens:][active_loss].contiguous())

        if self.adv_lm_w > 0:
            clip_min = clip_max = 1.0
        else:
            clip_min, clip_max = self.aitm_clip_min, self.aitm_clip_max
        lm_pre_kl = self.adv_lm_w * (
                    mlm_correct.float().clamp(clip_min, clip_max) * KL(first_block_hidden, mlm_predictions.detach(), reduction="none").sum(
                -1)).mean(0).sum()
        if reverse_loss:
            lm_kl_rev = self.adv_lm_w * (
                    mlm_correct.float().clamp(clip_min, clip_max) * KL(first_block_hidden.detach(), mlm_predictions, reduction="none").sum(
                -1)).mean(0).sum()
            lm_pre_kl = (lm_pre_kl + lm_kl_rev) / 2.0


        sent_order_pre_kl = 0
        if self.sentence_order_prediction_w > 0 and sent_order_predictions is not None:
            sent_order_block_hidden_cls = new_funnel_outputs["final_hidden"][:, 1:self.cls_tokens + 1]  # + new_funnel_outputs["final_hidden"][:, 0].unsqueeze(1)
            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls)
            sent_order_pre_kl = KL(sent_order_logits, sent_order_predictions.detach(), reduction="batchmean")
            if reverse_loss:
                sent_order_pre_kl = (sent_order_pre_kl + KL(sent_order_logits.detach(), sent_order_predictions, reduction="batchmean")) / 2.0

        decoder_outputs = new_funnel_outputs["decoder_outputs"]
        discriminator_sequence_output = decoder_outputs[0][:, self.cls_tokens:][active_loss].contiguous()
        electra_logits = self.discriminator_predictions(discriminator_sequence_output)
        electra_pre_kl = KL(electra_logits, electra_predictions.detach(), reduction="batchmean")
        if reverse_loss:
            electra_pre_kl = (electra_pre_kl + KL(electra_logits.detach(), electra_predictions, reduction="batchmean")) / 2.0

        contrastive_kl = 0.0
        answering_lm_loss_kl = 0.0
        run_answering = labels_pet_input_ids is not None
        if run_answering:
            answering_hidden = new_funnel_outputs["answering_hidden"]
            answering_lm_loss_kl = KL(answering_hidden, funnel_outputs["answering_hidden"].detach(), reduction="batchmean")
            if reverse_loss:
                answering_lm_loss_kl = (answering_lm_loss_kl + KL(answering_hidden.detach(), funnel_outputs["answering_hidden"], reduction="batchmean")) / 2.0

            answering_lm_loss_kl = self.answering_lm_w * answering_lm_loss_kl

        return self.lm_loss_w * lm_pre_kl, self.sentence_order_prediction_w * sent_order_pre_kl, self.electra_loss_w * electra_pre_kl, answering_lm_loss_kl, self.contrastive_w * contrastive_kl

    def forward_for_aitm(self, funnel_inputs, funnel_outputs,
                         mlm_predictions, mlm_correct, sent_order_predictions, electra_predictions,
                         labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
                         contrastive_anchors=None, contrastive_positives=None, contrastive_logits=None):
        if self.alum_aitm_alternate:
            self.adv_lm_w = -1 * self.adv_lm_w
        embed = funnel_outputs["input_embeds"]
        funnel_outputs = dict(**funnel_outputs)
        noise = embed.new(embed.size()).normal_(0, 1) * self.aitm_noise_var
        noise.requires_grad_()
        for _ in range(self.adv_ascent_steps):
            newembed = embed.detach() + noise
            funnel_outputs["input_embeds"] = newembed
            lm_pre_kl, sent_order_pre_kl, electra_pre_kl, answering_lm_pre_kl, contrastive_pre_kl = self.aitm_loss(funnel_inputs, funnel_outputs,
                                                                                                                   mlm_predictions, mlm_correct,
                                                                                                                   sent_order_predictions, electra_predictions,
                                                                                                                   labels_pet_input_ids,
                                                                                                                   labels_pet_attention_mask,
                                                                                                                   labels_pet_max_length,
                                                                                                                   contrastive_anchors, contrastive_positives, contrastive_logits)

            adv_loss = electra_pre_kl + sent_order_pre_kl + lm_pre_kl + answering_lm_pre_kl + contrastive_pre_kl
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0.0

            noise = noise + delta_grad * self.adv_step_size

        noise = self.adv_project(noise, eps=self.adv_epsilon)

        #
        newembed = embed + noise.detach()
        funnel_outputs["input_embeds"] = newembed
        lm_post_kl, sent_order_post_kl, electra_post_kl, answering_lm_post_kl, contrastive_post_kl = self.aitm_loss(funnel_inputs, funnel_outputs,
                                                                                                                    mlm_predictions, mlm_correct,
                                                                                                                    sent_order_predictions, electra_predictions,
                                                                                                                    labels_pet_input_ids,
                                                                                                                    labels_pet_attention_mask,
                                                                                                                    labels_pet_max_length,
                                                                                                                    contrastive_anchors, contrastive_positives,
                                                                                                                    contrastive_logits,
                                                                                                                    reverse_loss=True)
        adv_loss = sent_order_post_kl + electra_post_kl + (lm_post_kl ** 2) + answering_lm_post_kl + contrastive_post_kl
        return self.adv_w * adv_loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            labels_segment_index=None,
            char_ids=None, char_offsets=None,
            highway_cls_ar_input_ids=None, highway_cls_ar__attention_mask=None,
            labels_pet_input_ids=None, labels_pet_attention_mask=None, labels_pet_max_length=None,
            contrastive_anchors=None, contrastive_positives=None,
            **kwargs

    ):

        timing_dict = list()
        bs = input_ids.size(0)
        accuracy_hist = defaultdict()
        record_accuracy = self.record_accuracy or kwargs.pop("record_accuracy", False)
        st = time.time()
        run_answering = labels_pet_input_ids is not None
        assert run_answering
        funnel_inputs = dict(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             inputs_embeds=inputs_embeds,
                             char_ids=char_ids, char_offsets=char_offsets,
                             run_decoder=run_answering or self.electra_loss_w > 0,
                             run_answering=run_answering, )

        assert attention_mask is not None
        tokenizer_attn_mask = attention_mask
        # with autocast(enabled=kwargs.pop("autocast", False)):

        masked_lm_loss = 0.0
        first_block_cls = None
        if self.lm_loss_w > 0 or self.electra_loss_w > 0:
            active_loss = tokenizer_attn_mask.bool()
            first_block_hidden = self.funnel.forward_lm_block(input_ids=input_ids, attention_mask=attention_mask,
                                                              token_type_ids=token_type_ids,
                                                              inputs_embeds=inputs_embeds,
                                                              char_ids=char_ids, char_offsets=char_offsets, )[1]
            first_block_cls = first_block_hidden[:, 1:self.cls_tokens + 1]
            first_block_hidden = self.funnel.embed_proj_transpose(first_block_hidden[:, self.cls_tokens:])
            prediction_logits = self.funnel.lm_head(first_block_hidden)

            input_ids = prediction_logits.detach().argmax(dim=-1)


            active_labels = labels.reshape(-1)
            active_prediction_logits = prediction_logits.reshape(-1, self.config.vocab_size)
            masked_lm_loss = self.lm_loss_w * self.loss_ce(active_prediction_logits, active_labels)
            labels = (labels == input_ids).detach().float()
            if record_accuracy:
                # predictions = prediction_logits.argmax(dim=-1)
                # self.accuracy_hist["lm_preds"].append({"predictions": "".join(self.tokenizer.decode(predictions[0, 1:21].tolist())), "actuals": "".join(self.tokenizer.decode(labels[0, 1:21].tolist()))})
                accuracy_hist["lm_accuracy"] = (float(labels.float().cpu().numpy().mean()))

            et = time.time() - st
            timing_dict.append(("lm_accuracy_loss", et))
            et = time.time() - st
            timing_dict.append(("decoder_outputs", et))

        funnel_outputs = self.funnel(**funnel_inputs)
        inputs_embeds = funnel_outputs["inputs_embeds"]
        inputs_embeds_cls = inputs_embeds[:, :self.funnel.cls_tokens]
        final_hidden = funnel_outputs["final_hidden"]
        at_cast = kwargs.pop("autocast", False)
        preds_dict = dict()
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, input_ids = %s, attention_mask = %s" % (get_time_string(), random.sample(input_ids.reshape(-1).tolist(), 8), random.sample(attention_mask.reshape(-1).tolist(), 8)))
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, char_ids = %s, char_offsets = %s" % (get_time_string(), random.sample(char_ids.reshape(-1).tolist(), 8), random.sample(char_offsets.reshape(-1).tolist(), 8)))
        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Input embeds = %s, Input embeds CLS = %s" % (get_time_string(), random.sample(inputs_embeds.reshape(-1).tolist(), 8), random.sample(inputs_embeds_cls.reshape(-1).tolist(), 8)))
        et = time.time() - st
        timing_dict.append(("prepare_encoder_input", et))
        encoder_outputs = funnel_outputs["encoder_outputs"]
        et = time.time() - st
        timing_dict.append(("encoder_outputs", et))
        answering_lm_loss = 0.0
        if run_answering:
            alen = labels_pet_input_ids.size(1)
            assert alen <= funnel_outputs["answering_logits"].size(1)
            answering_logits = funnel_outputs["answering_logits"][:, :alen]
            answering_lm_loss = self.answering_lm_w * self.loss_ce(answering_logits.reshape(-1, self.config.vocab_size), labels_pet_input_ids.reshape(-1))
            if record_accuracy and "answer" in kwargs:
                answering_predictions = answering_logits.detach().argmax(dim=-1)
                answering_predictions = answer_decoder(answering_predictions, self.tokenizer)
                final_labels, final_predictions = [], []
                for lbl, prd in zip(kwargs.pop("answer", None), answering_predictions):
                    if len(prd) > len(lbl):
                        prd = prd[:len(lbl)]
                    if len(prd) < len(lbl):
                        prd = prd + ([''] * (len(lbl) - len(prd)))
                    final_labels.extend(lbl)
                    final_predictions.extend(prd)
                score = accuracy_score(final_labels, final_predictions)
                accuracy_hist["answering_lm_accuracy"] = score

        # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, first_block_hidden = %s, first_block_cls CLS = %s" % (get_time_string(), random.sample(first_block_hidden.reshape(-1).tolist(), 8), random.sample(first_block_cls.reshape(-1).tolist(), 8)))
        et = time.time() - st
        timing_dict.append(("lm_logits", et))

        et = time.time() - st
        timing_dict.append(("contrastive_loss", et))
        cls_orthogonal_loss = 0.0
        if self.input_cls_orthogonal_w > 0 and self.training:
            inputs_embeds_cls = inputs_embeds_cls/(inputs_embeds_cls.norm(2, -1, True).detach() + self.config.layer_norm_eps)
            inputs_embeds_cls = inputs_embeds_cls.bmm(inputs_embeds_cls.transpose(1, 2))
            input_cls_orthogonal_loss = self.input_cls_orthogonal_w * (inputs_embeds_cls ** 2).mean()
            cls_orthogonal_loss += input_cls_orthogonal_loss


        et = time.time() - st
        timing_dict.append(("cls_orthogonal_loss", et))
        sentence_order_loss = 0.0
        sent_order_logits = None

        if self.sentence_order_prediction_w > 0 and labels_segment_index is not None:
            segment_lens = labels_segment_index.size(1)
            mx_labels = labels_segment_index.max(-1)[0].view(-1)
            first_cls = final_hidden[:, 0]
            lsi = labels_segment_index.view(-1)
            sent_order_block_hidden_cls = final_hidden[:, 1:self.cls_tokens + 1]  + (first_block_cls if first_block_cls is not None else 0)
            sent_order_logits = self.sent_predict_fc(sent_order_block_hidden_cls).view(-1, (self.cls_tokens + 1))
            mx_label_pred = self.sent_predict_fc(first_cls)
            sent_order_loss = 0.5 * self.ce(sent_order_logits, lsi) + 0.01 * self.ce(mx_label_pred, mx_labels) + 0.5 * self.ignore_zero_ce(sent_order_logits, lsi)
            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, sent_order_block_hidden_cls = %s" % (get_time_string(), random.sample(sent_order_block_hidden_cls.reshape(-1).tolist(), 32)))
            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Logits and Labels SOP = %s" % (get_time_string(), list(zip(sent_order_logits.detach().reshape(-1, (self.cls_tokens + 1)).tolist(), labels_segment_index.reshape(-1).tolist()))[:4]))
            if record_accuracy:
                # TODO: Get accuracy of zeros only also and see if zeros only acc is too high, if yes then we may add `0` as ignored value for ce.
                sent_order_preds = sent_order_logits.detach().argmax(dim=-1).reshape(-1, segment_lens)
                not_in_order = [not all([c < n or ls_one[i + 1:].sum().item() == 0 for i, (c, n) in enumerate(zip(ls_one[:-1].tolist(), ls_one[1:].tolist()))]) for ls_one in labels_segment_index]
                non_zero = torch.tensor([ls_one[i:].sum().item() != 0 for ls_one in labels_segment_index for i in range(len(ls_one))]).reshape(-1, segment_lens).bool()

                if sum(not_in_order) > 0:
                    nih_preds = sent_order_preds[not_in_order]
                    nih_labels = labels_segment_index[not_in_order]
                    nih_non_zero = torch.tensor([ls_one[i:].sum().item() != 0 for ls_one in nih_labels for i in range(len(ls_one))]).reshape(-1, segment_lens).bool()
                    sent_order_out_nih = nih_preds == nih_labels
                    accuracy_hist["sent_order_nih_fraction"] = sum(not_in_order) / bs
                    accuracy_hist["sent_order_nih_accuracy"] = (float(sent_order_out_nih.detach().float().mean().cpu()))
                    accuracy_hist["sent_order_nih_accuracy_non_zero"] = (float(sent_order_out_nih[nih_non_zero].detach().float().mean().cpu()))
                    preds_dict["sent_order_nih_preds"] = nih_preds.cpu().view(-1).tolist()
                    preds_dict["sent_order_nih_labels"] = nih_labels.cpu().view(-1).tolist()
                else:
                    accuracy_hist["sent_order_nih_fraction"] = 0.0
                    accuracy_hist["sent_order_nih_accuracy"] = 0.0
                    accuracy_hist["sent_order_nih_accuracy_non_zero"] = 0.0
                    preds_dict["sent_order_nih_preds"] = []
                    preds_dict["sent_order_nih_labels"] = []

                sent_order_out = sent_order_preds == labels_segment_index
                # self.accuracy_hist["sent_order"].append({"all": sent_order_out.detach().cpu(), "mean": float(sent_order_out.sum() / len(sent_order_out[labels_segment_index != 0].reshape(-1))), "alt_mean": float(sent_order_out[labels_segment_index != 0].float().mean().detach().cpu())})
                accuracy_hist["sent_order_accuracy"] = (float(sent_order_out.detach().float().mean().cpu()))
                accuracy_hist["sent_order_accuracy_non_zero"] = (float(sent_order_out[non_zero].detach().float().mean().cpu()))



                preds_dict["sent_order_preds"] = sent_order_preds.cpu().view(-1).tolist()

                

                preds_dict["mx_label_pred"] = mx_label_pred.argmax(dim=-1).detach().cpu().tolist()
                preds_dict["mx_labels"] = mx_labels.cpu().tolist()

            sentence_order_loss = self.sentence_order_prediction_w * sent_order_loss
        et = time.time() - st
        timing_dict.append(("sentence_order_loss", et))

        loss_contrastive = 0.0
        contrastive_block_matrix = None
        contrastive_anchors_copy = contrastive_positives_copy = None
        if contrastive_anchors is not None and len(contrastive_anchors) > 0 and self.contrastive_w > 0 and contrastive_positives is not None and len(contrastive_positives) > 0:
            if len(contrastive_positives) > bs and len(contrastive_positives) % bs == 0 and len(contrastive_positives) / bs == torch.cuda.device_count():
                did = torch.cuda.current_device()
                contrastive_positives = contrastive_positives[did * bs: (did + 1) * bs]
                contrastive_anchors = contrastive_anchors[did * bs: (did + 1) * bs]

            contrastive_anchors_copy, contrastive_positives_copy = copy.deepcopy(contrastive_anchors), copy.deepcopy(contrastive_positives)
            contrastive_block_hidden = funnel_outputs["decoder_outputs"][0]
            assert len(contrastive_block_hidden.size()) == 3

            # dpow = self.config.stride ** 2
            # contrastive_positives = recursive_op(contrastive_positives, lambda x: int(x / dpow))
            # contrastive_anchors = recursive_op(contrastive_anchors, lambda x: int(x / dpow))
            # contrastive_len = contrastive_block_hidden.size(1)
            # for batch_positive in contrastive_positives:
            #     for positives_for_anchor in batch_positive:
            #         for positive in positives_for_anchor:
            #             if positive[0]==positive[1]:
            #                 if positive[1] < contrastive_len:
            #                     positive[1] = positive[1] + 1
            #                 elif positive[0] > 0:
            #                     positive[0] = positive[0] - 1
            #
            #             positive[0] = min(positive[0], positive[1] - 1)
            # for batch_anchor in contrastive_anchors:
            #     for anch in batch_anchor:
            #         if anch[0] == anch[1]:
            #             if anch[1] < contrastive_len:
            #                 anch[1] = anch[1] + 1
            #             elif anch[0] > 0:
            #                 anch[0] = anch[0] - 1
            # print("Anchors Batch size = %s, Input Batch Size = %s" % (len(contrastive_anchors), input_ids.size()))

            # print(contrastive_anchors, contrastive_block_hidden.size(1))
            # print([[anchor[0], anchor[1]] for anchor_batch_pos, anchors in enumerate(contrastive_anchors) for anchor in anchors])
            # print([[anchor_pos[0], anchor_pos[1]] for pos in contrastive_positives for anchor_pos in pos])
            # print("==" * 80)
            # contrastive_anchors = [[], [], [], []]
            # contrastive_positives = [[[]], [[]], [[]], [[]]]
            # print(input_ids.size(), contrastive_block_hidden.size(), "\n", contrastive_anchors, contrastive_positives)
            # assertion_anchor = [contrastive_block_hidden.size(1) > anchor[1] for anchor_batch_pos, anchors in enumerate(contrastive_anchors) for anchor in anchors]
            # print(assertion_anchor)
            # assert all(assertion_anchor)

            anchors = [contrastive_block_hidden[anchor_batch_pos, [anchor[0], anchor[1]]].mean(0) for anchor_batch_pos, anchors in
                       enumerate(contrastive_anchors) for anchor in anchors]
            contrastive_positives = [[[*cp, batch_pos] for cp in anchor_cp] for batch_pos, anchors_cp in enumerate(contrastive_positives) for anchor_cp in
                                     anchors_cp]
            n_positives_per_anchor = max([len(a) for a in contrastive_positives])
            contrastive_positives = [[a[i]] for i in range(n_positives_per_anchor) for a in contrastive_positives if len(a) > 0]
            # contrastive_positives = torch.tensor(contrastive_positives).transpose(0, 1).tolist()
            # assertion_pos = [contrastive_block_hidden.size(1) > anchor_pos[1] for pos in contrastive_positives for anchor_pos in pos]
            # print(assertion_pos)
            # assert all(assertion_pos)
            positives = [contrastive_block_hidden[anchor_pos[-1], [anchor_pos[0], anchor_pos[1]]].mean(0) for pos in contrastive_positives for anchor_pos in
                         pos]

            n_anchors = len(anchors)
            n_positives = len(positives)
            assert n_positives == 0 or n_anchors == 0 or n_positives % n_anchors == 0
            assert n_positives == 0 or n_anchors == 0 or (n_positives / n_anchors) == n_positives_per_anchor
            if n_positives == 0 or n_anchors == 0:
                pass
            else:
                contrastive_block_hidden = self.contrastive_ffn(torch.stack(anchors + positives))
                contrastive_block_hidden = contrastive_block_hidden / (contrastive_block_hidden.norm(2, -1, True).detach() + self.config.layer_norm_eps)
                contrastive_block_matrix = contrastive_block_hidden.mm(contrastive_block_hidden.t()) / self.contrastive_temperature
                # contrastive_block_matrix = contrastive_block_matrix * (1 - torch.eye(contrastive_block_matrix.size(0), device=contrastive_block_matrix.device))
                if record_accuracy:
                    accuracy_hist["contrastive_accuracy"] = 0
                    preds_dict["contrastive_preds"] = []
                    preds_dict["contrastive_actuals"] = []

                mask1 = contrastive_block_matrix.new_ones(contrastive_block_matrix.size(), requires_grad=False)
                mask2 = contrastive_block_matrix.new_zeros(contrastive_block_matrix.size(), requires_grad=False)
                const = 1e3
                column_idxs = []
                for i in range(n_positives_per_anchor + 1):
                    idxs = (torch.tensor(list(range(n_anchors))) + (n_anchors * i)).tolist() * (n_positives_per_anchor + 1)
                    column_idxs.extend(idxs)
                row_idxs = list(range(mask2.size(0))) * (n_positives_per_anchor + 1)
                mask1[torch.tensor(row_idxs), torch.tensor(column_idxs)] = 0
                mask2[torch.tensor(row_idxs), torch.tensor(column_idxs)] = -const
                vertical_lc = 0.0
                for i in range(n_positives_per_anchor + 1):
                    labels_contrastive = torch.tensor((torch.tensor(list(range(n_anchors))) + (n_anchors * i)).tolist() * (n_positives_per_anchor + 1), device=contrastive_block_hidden.device)
                    mask_c = mask2.clone()
                    mask_ones = mask1.clone()
                    mask_c[torch.arange(mask_c.size(0)), torch.tensor((torch.tensor(list(range(n_anchors))) + (n_anchors * i)).tolist() * (n_positives_per_anchor + 1))] = 0.0
                    mask_ones[torch.arange(mask_c.size(0)), torch.tensor((torch.tensor(list(range(n_anchors))) + (n_anchors * i)).tolist() * (n_positives_per_anchor + 1))] = 1.0

                    block_matrix = (contrastive_block_matrix * mask_ones) + mask_c
                    # TODO: check this
                    l2 = self.ce(block_matrix, labels_contrastive)
                    vertical_lc += l2
                    if record_accuracy:
                        contrastive_preds = block_matrix.detach().argmax(dim=-1)
                        cacc = (contrastive_preds == labels_contrastive).sum().item()
                        accuracy_hist["contrastive_accuracy"] += cacc
                        cpp = contrastive_preds.tolist()
                        cpa = labels_contrastive.tolist()
                        preds_dict["contrastive_preds"] += (cpp[:(n_anchors * i)] + cpp[(n_anchors * (i+1)):])
                        preds_dict["contrastive_actuals"] += (cpa[:(n_anchors * i)] + cpa[(n_anchors * (i+1)):])
                if record_accuracy:
                    accuracy_hist["contrastive_accuracy"] = max(accuracy_hist["contrastive_accuracy"] - contrastive_block_matrix.size(0), 0) / (contrastive_block_matrix.size(0) * (n_positives_per_anchor + 1) - contrastive_block_matrix.size(0))
                vertical_lc /= (n_positives_per_anchor + 1)
                loss_contrastive += vertical_lc
            loss_contrastive = self.contrastive_w * loss_contrastive

        et = time.time() - st
        timing_dict.append(("highway_cls_ar_sentence_loss", et))

        loss = 0.0
        if self.electra_loss_w > 0:
            decoder_outputs = funnel_outputs["decoder_outputs"]
            # cls_tokens = decoder_outputs[0][:, :self.cls_tokens + 1]
            # decoder_outputs = (decoder_outputs[0][:, self.cls_tokens + 1:], decoder_outputs[1:])
            discriminator_sequence_output = decoder_outputs[0][:, self.cls_tokens:].contiguous()
            logits = self.discriminator_predictions(discriminator_sequence_output)
            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, discriminator_sequence_output = %s, logits = %s" % (get_time_string(), random.sample(discriminator_sequence_output.reshape(-1).tolist(), 8), random.sample(logits.reshape(-1).tolist(), 8)))

            et = time.time() - st
            timing_dict.append(("electra_discriminator_logits", et))

            # print("[FastFormerForFusedELECTRAPretraining]: Time = %s, Logits and Labels for electra = %s" % (get_time_string(), list(zip(active_logits.detach().tolist(), labels.tolist()))[:4]))

            loss = self.electra_loss_w * self.loss_bce(logits, labels)
            if record_accuracy:
                electra_logits = torch.sigmoid(logits.detach()).view(-1)
                electra_preds = (electra_logits > 0.5).type(torch.int64).view(-1)
                labels = labels.view(-1)
                preds_dict["electra_logits"] = electra_logits.tolist()
                preds_dict["electra_preds"] = electra_preds.tolist()
                preds_dict["electra_labels"] = labels.tolist()
                accuracy_hist["electra_preds_mean"] = electra_preds.type(torch.float).mean().item()
                accuracy_hist["electra_labels_mean"] = labels.type(torch.float).mean().item()
                accuracy_hist["electra_accuracy"] = (torch.mean((electra_preds == labels).type(torch.float)).item())
                # if self.record_accuracy:
                #     self.accuracy_hist.append(accuracy_hist)

        et = time.time() - st
        timing_dict.append(("electra_discriminator_accuracy", et))

        electra_loss = loss

        et = time.time() - st
        adv_loss = torch.tensor(0.0)
        timing_dict.append(("aitm_alum_start", et))
        if (self.aitm or self.alum) and self.training and self.electra_loss_w > 0 and self.lm_loss_w > 0:
            adv_loss = self.forward_for_aitm(funnel_inputs, funnel_outputs, first_block_hidden, labels, sent_order_logits, logits,
                                             labels_pet_input_ids, labels_pet_attention_mask, labels_pet_max_length,
                                             contrastive_anchors_copy, contrastive_positives_copy, contrastive_block_matrix)
            loss = loss + adv_loss

        et = time.time() - st
        timing_dict.append(("aitm_alum_end", et))

        loss = loss + masked_lm_loss + sentence_order_loss + answering_lm_loss + cls_orthogonal_loss + loss_contrastive
        et = time.time() - st
        timing_dict = [(k, 100 * (v/et)) for k, v in timing_dict]
        self.timing_hist.append(timing_dict)
        self.timing_hist = self.timing_hist[-100:]

        # TODO: return one loss
        # TODO: check various loss history and accuracies over time
        # TODO: Make a separate wrapper for AITM and ALUM vs making it here?

        # TODO: CLS correction needed

        loss_dict = dict()
        if record_accuracy:
            accuracy_hist = {k: v.detach() if hasattr(v, "detach") else v for k, v in accuracy_hist.items()}
            loss_dict = dict(masked_lm_loss=float(masked_lm_loss), sentence_order_loss=float(sentence_order_loss), answering_lm_loss=float(answering_lm_loss),
                             cls_orthogonal_loss=float(cls_orthogonal_loss),
                             loss_contrastive=float(loss_contrastive), adv_loss=float(adv_loss), electra_loss=float(electra_loss), loss=float(loss))
            loss_dict = {k: v.detach() if hasattr(v, "detach") else v for k, v in loss_dict.items()}
        results = dict(loss=loss, loss_dict=loss_dict, timing_dict=timing_dict, accuracy_hist=accuracy_hist, preds_dict=preds_dict)
        if self.data_parallel:
            results = [results]
        # TODO: return pet answer
        # decoder_output=decoder_outputs[0], decoder_cls=cls_tokens, encoder_output=encoder_outputs[0][:, self.cls_tokens + 1:], encoder_cls=encoder_outputs[0][:, :self.cls_tokens + 1], encoder_hidden_states=encoder_outputs[1])
        return results


if __name__ == "__main__":
    import time
    import argparse
    import numpy as np
    from tqdm.auto import tqdm, trange
    from torch.optim import AdamW

    torch.backends.cudnn.benchmark = True
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForMaskedLM, ElectraForPreTraining, CTRLConfig, CTRLPreTrainedModel
    from transformers.models.deberta import DebertaModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default='cpu',
                    help="Device")
    ap.add_argument("--config", type=str, default='md_config_relative',
                    help="Config")
    ap.add_argument("--texts", type=str, default='large_texts',
                    help="Text Set")
    ap.add_argument("--profile", type=str2bool, default=False)
    ap.add_argument("--sdconv", type=str2bool, default=False)
    ap.add_argument("--forward_only", type=str2bool, default=False)
    ap.add_argument("--fp16", type=str2bool, default=False)
    ap.add_argument("--aitm", type=str2bool, default=False)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", type=str, default='fastformer_fused_electra')  # fastformer_mlm, fastformer_electra, fastformer_fused_electra, fastformer, microsoft/deberta-base, roberta-base, distilroberta-base, funnel-transformer/intermediate

    args = vars(ap.parse_args())
    forward_only = args["forward_only"]
    device = args["device"]
    profile = args["profile"]
    fp16 = args["fp16"]
    model_name = args["model"]
    aitm = args["aitm"]
    sdconv = args["sdconv"]
    batch_size = args["batch_size"]
    length = args["length"]
    lr = args["lr"]
    texts = args["texts"]
    config = config_dict[args["config"]]
    epochs = args["epochs"]
    if aitm:
        assert not forward_only and model_name == "fastformer_fused_electra"
    HuggingFaceModelClass = AutoModel if forward_only else AutoModelForMaskedLM

    small_max_length = 128
    medium_max_length = 512
    large_max_length = 1024
    very_large_max_length = 1536
    texts = dict(large_texts=large_texts, very_small_texts=very_small_texts, small_texts=small_texts, hetero_texts=hetero_texts)[texts]

    tokenizer = get_tokenizer("bert")
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = length
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens
    if model_name not in ["fastformer_mlm", "fastformer_electra", "fastformer_fused_electra", "fastformer"]:
        config.tokenizer_length = min(config.tokenizer_length, 512)
        config.max_position_embeddings = min(config.tokenizer_length, 512)
        config.num_highway_cls_tokens = 0
    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)
    if batch_size > len(texts):
        for _ in range(8):
            texts += texts
    dataset = SmallTextDataset(texts)
    config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
    dataset = TokenizerDataset(config, tokenizer, get_char_to_id(),
                               dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length),
                               # sentence_jumble_proba=((1024, 0.1),), word_noise_proba=((1024, 0.1),),
                               max_jumbling_span_length=2,
                               dataset=dataset)
    dataset.training = True

    if "fastformer" in model_name:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        size_dicts_t = {128: batch_size, 256: batch_size, 512: batch_size, 768: batch_size, 1024: batch_size}
        pt_batch = next(custom_batching_fn(dataloader, size_dicts_t, True))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        iter_dataloader = iter(dataloader)
        pt_batch = next(iter_dataloader)

    if "fastformer" in model_name:
        sm_pt_batch = dict(input_ids=pt_batch["input_ids"], attention_mask=pt_batch["attention_mask"],
                        char_offsets=pt_batch["char_offsets"], char_ids=pt_batch["char_ids"])
        if model_name == "fastformer_fused_electra":
            model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, adv_step_size=1e-3, lm_loss_w=5.0, electra_loss_w=1.0,
                                                         aitm=aitm, alum=False, alum_aitm_alternate=False)
            sm_pt_batch = pt_batch
            assert not forward_only
        if model_name == "fastformer_mlm":
            model = FastFormerForMaskedLM(config)
        if model_name == "fastformer":
            model = FastFormerModel(config)

        pt_batch = sm_pt_batch

    else:
        pt_batch = dict(input_ids=pt_batch["input_ids"], attention_mask=pt_batch["attention_mask"])
        if "electra" in model_name:
            HuggingFaceModelClass = ElectraForPreTraining
        elif "deberta" in model_name:
            HuggingFaceModelClass = AutoModel
        model = HuggingFaceModelClass.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model = AutoModel.from_pretrained("funnel-transformer/intermediate")
    # tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/intermediate")
    #
    # model = AutoModel.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    #
    # model = AutoModel.from_pretrained("albert-base-v2")
    # tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    #

    # model = AutoModel.from_pretrained("microsoft/deberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    # model = AutoModel.from_pretrained("distilbert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # model = AutoModel.from_pretrained("squeezebert/squeezebert-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")

    # model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    # tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # model = AutoModelForMaskedLM.from_pretrained("funnel-transformer/small-base")
    # tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small-base")

    # model = AutoModelForMaskedLM.from_pretrained("google/electra-small-discriminator")
    # tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    # model = AutoModelForMaskedLM.from_pretrained("nlpaueb/legal-bert-small-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")

    # model = AutoModelForMaskedLM.from_pretrained("nreimers/BERT-Small-L-4_H-512_A-8")
    # tokenizer = AutoTokenizer.from_pretrained("nreimers/BERT-Small-L-4_H-512_A-8")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-8_H-256_A-4")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-8_H-256_A-4")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-768_A-12")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-768_A-12")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-512_A-8")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-512_A-8")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-6_H-256_A-4")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-6_H-256_A-4")

    # model = AutoModelForMaskedLM.from_pretrained("chiragjn/small_bert_uncased_L-4_H-768_A-12")
    # tokenizer = AutoTokenizer.from_pretrained("chiragjn/small_bert_uncased_L-4_H-768_A-12")

    # TODO: Test Longformer for long sequences as well.
    # TODO: test mobilebert

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Params = %s" % (numel(model) / 1_000_000))

    if hasattr(model, "funnel"):
        funnel = model.funnel
        model_parameters = list(filter(lambda p: p.requires_grad, funnel.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Funnel Params = %s" % (numel(funnel) / 1_000_000))
    print(model)
    # print(model.funnel.encoder.repeats if hasattr(model, "funnel") else "")

    model = model.eval()
    config = md_config

    if "google/electra-base-discriminator" in model_name:
        labels = torch.randint_like(pt_batch["input_ids"], 0, 2)
        labels = labels.to(device)
    else:
        labels = pt_batch["label_mlm_input_ids"] if "label_mlm_input_ids" in pt_batch else pt_batch["input_ids"]
        labels = labels.to(device)
    if "labels" in pt_batch:
        del pt_batch["labels"]
    print("Input Sizes = ", {k: v.size() if hasattr(v, "size") else len(v) for k, v in pt_batch.items()})

    device = torch.device(device)
    # torch.autograd.set_detect_anomaly(True)

    # checkpoint = torch.load("model/fastformer_checkpoint-step-3999.pth", map_location=str(device))
    # import torch.distributed as dist
    # from torch.nn.parallel import DistributedDataParallel as DDP
    #
    # dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://%s:%s" % ("127.0.0.1", "9999"))
    # ddp_model = DDP(model, device_ids=None, find_unused_parameters=True, bucket_cap_mb=5)
    # ddp_model.load_state_dict(checkpoint['model'])
    # model = ddp_model

    # checkpoint = torch.load("model/error-model.pth", map_location=str(device))
    # model.load_state_dict(checkpoint)
    # pt_batch = torch.load("model/error-input.pth", map_location=str(device))
    # labels = pt_batch.pop("labels", None)

    model = model.to(device)
    pt_batch["record_accuracy"] = True
    pt_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in pt_batch.items()}


    try:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
    except:
        pass
    if forward_only:
        _ = model.eval()
    else:
        _ = model.train()

    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(all_params, lr=lr, eps=1e-6, weight_decay=1e-2)
    torch.autograd.set_detect_anomaly(True)

    def get_unused_params(model):
        for name, params in model.named_parameters():
            if params.grad is None:
                print(name)

    def run():
        if not forward_only:
            if fp16:
                with autocast():
                    if isinstance(model, (AutoModel, DebertaModel)):
                        output = model(**pt_batch)
                        output = ((output['last_hidden_state'][:, 0] - random.random()).mean(),)
                    else:
                        output = model(**pt_batch, labels=labels)
                    loss = output[0] if isinstance(output, (list, tuple)) else output["loss"]
                    scaler.scale(loss).backward()
                    get_unused_params(model)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if isinstance(model, (AutoModel, DebertaModel)):
                    pt_batch.pop("record_accuracy", None)
                    output = model(**pt_batch)
                    output = ((output['last_hidden_state'][:, 0] - random.random()).mean(),)
                else:
                    output = model(**pt_batch, labels=labels)
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
                        pt_outputs = model(**pt_batch)

            else:
                with torch.no_grad():
                    pt_outputs = model(**pt_batch)
            return pt_outputs


    if profile:
        import torch.autograd.profiler as profiler

        _ = [run() for _ in range(2)]
        with profiler.profile(record_shapes=True) as prof:
            _ = [run() for _ in range(epochs)]
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
    else:
        _ = [run() for _ in range(1)]
        times = []
        for _ in trange(epochs):
            st = time.time()
            _ = run()
            et = time.time() - st
            times.append(et)
        print("Time Taken = %.4f, Lowest = %.4f, Highest = %.4f, variance = %.4f" % (np.mean(times), np.min(times), np.max(times), np.std(times)))

    if not forward_only and hasattr(model, "accuracy_hist"):
        from pprint import pprint
        pprint({k: v[-5:] for k, v in model.accuracy_hist.items()})
        pprint({k: v[:5] for k, v in model.accuracy_hist.items()})
        if hasattr(model, "timing_hist"):
            pprint(model.timing_hist[-1])
            import pandas as pd
            th = pd.DataFrame([td for tm in model.timing_hist for td in tm], columns = ["step", "cumulative"])
            th = th.groupby("step", sort=False)[["cumulative"]].mean()
            timings = [0] + list(th["cumulative"].values)[:-1]
            th["differ"] = timings
            th["absolute"] = th["cumulative"] - th["differ"]
            th.drop(columns=["differ"], inplace=True)
            pprint(th)
        # pprint(model.loss_hist)
