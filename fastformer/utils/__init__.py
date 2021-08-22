import argparse
import functools
from typing import Iterable

import dill
import numpy as np
import torch
import random
import re
import gc
import os
import datetime
import time
from datetime import datetime, timedelta

import torchvision
from pytz import timezone
import time
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from timm.models.layers.weight_init import trunc_normal_
import base64
import hashlib
from hashlib import sha3_256, md5, sha256

import pandas as pd
import random
import os
import logging
import argparse
from tqdm.auto import tqdm
import subprocess
import shlex
import shutil
import math
import sys
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
from albumentations import augmentations as alb
import imgaug.augmenters as iaa
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, RobertaTokenizerFast, ConvBertTokenizer, ConvBertTokenizerFast, RobertaTokenizer, RobertaConfig, RobertaModel, \
    PretrainedConfig, PreTrainedModel, DebertaV2Config, DebertaV2Model, DebertaV2Tokenizer
import torchvision.transforms as transforms
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

def warn(*args, **kwargs):
    pass

warnings.warn = warn


logging.basicConfig(
    format='[PID: %(process)d] [%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO"),
    datefmt='%Y-%m-%d %H:%M:%S')


def getLogger(name, level=None):
    level = os.environ.get("LOGLEVEL", "INFO") if level is None else level
    log = logging.getLogger(name)
    log.setLevel(level)
    return log




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_tokenizer(name):
    from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
    if "roberta" in name:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif "bert" in name:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    setattr(tokenizer, "_sentence_mask_token", "[MASK1]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["sentence_mask_token"]
    tokenizer.add_special_tokens({"sentence_mask_token": "[MASK1]"})

    setattr(tokenizer, "_answer_option_separator_token", "[ANSWER_OPTION_SEP]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_separator_token"]
    tokenizer.add_special_tokens({"answer_option_separator_token": "[ANSWER_OPTION_SEP]"})

    setattr(tokenizer, "_answer_option_begin_token", "[ANSWER_OPTION_BEGIN]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_begin_token"]
    tokenizer.add_special_tokens({"answer_option_begin_token": "[ANSWER_OPTION_BEGIN]"})

    setattr(tokenizer, "_answer_option_end_token", "[ANSWER_OPTION_END]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_option_end_token"]
    tokenizer.add_special_tokens({"answer_option_end_token": "[ANSWER_OPTION_END]"})

    setattr(tokenizer, "_no_question_token", "[NO_QUESTION]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["no_question_token"]
    tokenizer.add_special_tokens({"no_question_token": "[NO_QUESTION]"})

    setattr(tokenizer, "_seg_sep_token", tokenizer.sep_token)
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["seg_sep_token"]
    tokenizer.add_special_tokens({"seg_sep_token": tokenizer.sep_token})

    setattr(tokenizer, "_answer_end_token", "[ANSWER_END]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_end_token"]
    tokenizer.add_special_tokens({"answer_end_token": "[ANSWER_END]"})
    n_question_tokens = 8
    for i in range(n_question_tokens):
        setattr(tokenizer, "_question_token_%s" % i, "[QUESTION_%s]" % i)
        setattr(tokenizer, "_answer_token_%s" % i, "[ANSWER_%s]" % i)
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["question_token_%s" % i]
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["answer_token_%s" % i]
        tokenizer.add_special_tokens({"question_token_%s" % i: "[QUESTION_%s]" % i, "answer_token_%s" % i: "[ANSWER_%s]" % i})
    # {k: tokenizer.encode(v, add_special_tokens=False) for k, v in tokenizer.special_tokens_map_extended.items()}
    return tokenizer


def answer_decoder(input_ids, tokenizer):
    input_ids = input_ids.tolist()
    all_answers = []
    for answers in tokenizer.batch_decode(input_ids):
        answers = answers.split(tokenizer.answer_end_token)[0]
        answers = answers.replace(tokenizer.pad_token, '').replace(tokenizer.cls_token, '').strip()
        # count_answers = len(re.findall(r'\[QUESTION_[0-9]+\]', answers))
        answers = list(filter(lambda x: len(x.strip()), re.split(r'\[QUESTION_[0-9]+\]', answers)))
        all_answers.append([x.strip() for x in answers])
    return all_answers


def answer_decoder_debug(input_ids, tokenizer):
    input_ids = input_ids[:, :8].tolist()
    all_answers = []
    for answers in tokenizer.batch_decode(input_ids):
        answers = answers.replace(tokenizer.pad_token, '').replace(tokenizer.cls_token, '').strip()
        all_answers.append(answers)
    return all_answers


def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def clean_memory():
    _ = gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    _ = gc.collect()


def squeeze_after(x: torch.Tensor, dim):
    xs = x.size()
    xsl = len(xs)
    # print(xs, xsl, xsl - 1)
    for i in range(xsl - 1, dim, -1):
        # print(i, x.size())
        x = x.squeeze(i)
    return x


def get_time_string():
    # + timedelta(hours=5, minutes=30)
    return (datetime.fromtimestamp(time.mktime(time.gmtime(time.time())))).astimezone(timezone('Asia/Kolkata')).strftime("[%a, %d %b %Y, %H:%M:%S %Z]")


def one_run(host, cmd, arg=None, dry_run=False):
    if "scp" in cmd:
        cur_cmd = cmd % (host)
        cmd_str = shlex.split(cmd % (host))
    else:
        cur_cmd = (cmd % arg) if arg is not None else cmd
        cmd_str = shlex.split("ssh %s '%s'" % (host, cur_cmd))
    if dry_run:
        return {"host": host, "cmd": cur_cmd, "stdout": "", "stderr": ""}
    s = subprocess.run(cmd_str, shell=False, capture_output=True, text=True)
    return {"host": host, "stdout": s.stdout, "stderr": s.stderr, "cmd": " ".join(cmd_str)}


def left_justify(words, width):
    """Given an iterable of words, return a string consisting of the words
    left-justified in a line of the given width.

    >>> left_justify(["hello", "world"], 16)
    'hello world     '

    """
    return ''.join(words).ljust(width)


def justify(words, width):
    """Divide words (an iterable of strings) into lines of the given
    width, and generate them. The lines are fully justified, except
    for the last line, and lines with a single word, which are
    left-justified.

    >>> words = "This is an example of text justification.".split()
    >>> list(justify(words, 16))
    ['This    is    an', 'example  of text', 'justification.  ']

    """
    line = []             # List of words in current line.
    col = 0               # Starting column of next word added to line.
    for word in words:
        if line and col + len(word) > width:
            if len(line) == 1:
                yield left_justify(line, width)
            else:
                # After n + 1 spaces are placed between each pair of
                # words, there are r spaces left over; these result in
                # wider spaces at the left.
                n, r = divmod(width - col + 1, len(line) - 1)
                narrow = '' * (n + 1)
                if r == 0:
                    yield narrow.join(line)
                else:
                    wide = '' * (n + 2)
                    yield wide.join(line[:r] + [narrow.join(line[r:])])
            line, col = [], 0
        line.append(word)
        col += len(word) + 1
    if line:
        yield left_justify(line, width)


def get_barrier(activate):
    def barrier():
        if activate:
            torch.distributed.barrier()
    return barrier


def save(filename, model, optimizer, scheduler, scaler, other_info_dict={}, is_best=False):
    if other_info_dict is not None and "step" in other_info_dict:
        filename = filename + "-step-%s" % (other_info_dict["step"])
    filename = filename + ".pth"
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), other=other_info_dict)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load(filename, model, optimizer, scheduler, scaler, device):
    import glob
    print("[Load]: Time = %s, Loading Checkpoint from %s, cwd = %s" % (get_time_string(), filename, os.getcwd()))
    if not os.path.isfile(filename):
        fss = list(map(lambda x: (x, ''.join(filter(str.isdigit, x))), glob.glob(filename + "*")))
        print("[Load]: Time = %s, Loading Checkpoint options %s" % (get_time_string(), fss))
        if len(fss) == 0:
            return None
        fss = map(lambda x: (x[0], -1 if len(x[1]) == 0 else int(x[1])), fss)
        fss = sorted(list(fss), key=lambda x: x[1], reverse=True)[0][0]
        print("[Load]: Time = %s, Loading Checkpoint from %s, exists = %s" % (get_time_string(), fss, os.path.isfile(fss)))
        filename = fss
    assert os.path.isfile(filename)
    if torch.cuda.is_available():
        loc = 'cuda:{}'.format(device)
    else:
        loc = "cpu"
    print("[Load]: Time = %s, Prepare Read Checkpoint from %s" % (get_time_string(), filename))
    checkpoint = torch.load(filename, map_location=loc)
    print("[Load]: Time = %s, Read Checkpoint from %s" % (get_time_string(), filename))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    other = checkpoint['other']
    return other


def recursive_op(arr, op, depth_limit=None, depth=0,):
    if depth_limit is not None and depth >= depth_limit:
        return arr
    elif isinstance(arr, (list, tuple)):
        arr = list(map(lambda x: recursive_op(x, op, depth_limit, depth+1), arr))
    else:
        arr = op(arr)
    return arr


class ValidationError(Exception):
    def __init__(self, message, contents, diagnostic=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.contents = contents
        self.diagnostic = diagnostic


def reraise(e, *args):
  '''re-raise an exception with extra arguments
  :param e: The exception to reraise
  :param args: Extra args to add to the exception
  '''

  # e.args is a tuple of arguments that the exception with instantiated with.
  #
  e.args = args + e.args

  # Recreate the expection and preserve the traceback info so thta we can see
  # where this exception originated.
  #
  raise e.with_traceback(e.__traceback__)


class SkipDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, skip_first, skip_last):
        self.dataset = dataset
        self.skip_first = skip_first
        self.skip_last = skip_last

    def __getitem__(self, item):
        return self.dataset[item + self.skip_first]

    def __len__(self):
        return len(self.dataset) - self.skip_last - self.skip_first


def gcd(a,b):
    if a == 0:
        return b
    return gcd(b % a, a)


def gcd_array(x):
    x = list(x)
    gcv = 1e8
    for a, b in zip(x[1:], x[:-1]):
        gcv = min(gcv, gcd(a, b))
    return gcv


fsdp_store = dict()


def configure_fsdp(enable_autocast=False, fp32_reduce_scatter=True, init=False):

    if "fsdp_params" not in fsdp_store and init:
        def get_fsdp_params():
            fsdp_params = dict(mixed_precision=enable_autocast, flatten_parameters=True, buffer_dtype=torch.float32,
                               bucket_cap_mb=25, reshard_after_forward=False, fp32_reduce_scatter=False if not enable_autocast else fp32_reduce_scatter,
                               cpu_offload=False, move_grads_to_cpu=False, process_group=torch.distributed.group.WORLD)
            return fsdp_params

        fsdp_store["fsdp_params"] = get_fsdp_params()

    return fsdp_store["fsdp_params"]


def fsdp_wrapper(module=None, wrap_type=0, init=False):
    def get_wrapper_v2(module):
        from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
        from fairscale.nn.misc import checkpoint_wrapper
        return FullyShardedDDP(checkpoint_wrapper(module, offload_to_cpu=False), **configure_fsdp(init=False))

    def get_wrapper_v1(module):
        from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
        from fairscale.nn.misc import checkpoint_wrapper
        return checkpoint_wrapper(module, offload_to_cpu=False)

    def get_wrapper_v0(module):
        return module
    wd = {0: get_wrapper_v0, 1: get_wrapper_v1, 2: get_wrapper_v2}

    if "fsdp_wrapper" not in fsdp_store and init:
        fsdp_store["fsdp_wrapper"] = wd[wrap_type]

    if module is not None and "fsdp_wrapper" in fsdp_store:
        return fsdp_store["fsdp_wrapper"](module)
    elif module is not None:
        return module
    else:
        return wd[wrap_type]


def transpose_for_scores(x, num_attention_heads):
    new_x_shape = x.size()[:-1] + (num_attention_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


def build_relative_position(query_size, key_size, device, query_stride=1, key_stride=1):
    """
    Build relative position according to the query and key
    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    q_ids = torch.arange(0, query_size * query_stride, query_stride, dtype=torch.long, device=device)
    k_ids = torch.arange(0, key_size * key_stride, key_stride, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


# @torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


# @torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), c2p_pos.size(-1)])


# @torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


def disentangled_att_bias(query_layer, key_layer, relative_pos, query_embeddings, key_embeddings, scale_factor, query_max_relative_postions,
                          key_max_relative_positions, heads, pos_k_proj=None, pos_q_proj=None,
                          query_stride=1, key_stride=1, pos_att_type=("c2p", "p2c")):
    assert heads == query_layer.size(1) or heads == query_layer.size(1)//2
    half_head = heads == query_layer.size(1)//2
    if relative_pos is None:
        q = query_layer.size(-2)
        relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device, query_stride=query_stride, key_stride=key_stride)
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    # bxhxqxk
    elif relative_pos.dim() != 4:
        raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

    relative_pos = relative_pos.long().to(query_layer.device)
    q_att_span = min(max(query_layer.size(-2), key_layer.size(-2)), query_max_relative_postions)
    query_embeddings = query_embeddings[
        max(0, query_max_relative_postions - q_att_span): min(query_embeddings.size(0), query_max_relative_postions + q_att_span), :
    ].unsqueeze(0)
    att_span = min(max(query_layer.size(-2), key_layer.size(-2)), key_max_relative_positions)
    key_embeddings = key_embeddings[
                     max(0, key_max_relative_positions - att_span): min(key_embeddings.size(0), key_max_relative_positions + att_span), :
                     ].unsqueeze(0)

    score = 0
    # content->position
    if "c2p" in pos_att_type:
        pos_key_layer = pos_k_proj(key_embeddings) if pos_k_proj else key_embeddings
        pos_key_layer = transpose_for_scores(pos_key_layer, heads)
        assert pos_key_layer.size(-1) == query_layer.size(-1)
        c2p_att = torch.matmul(query_layer[:, :heads] if half_head else query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, min(att_span * 2 - 1, key_embeddings.size(0) - 1))
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer[:, :heads] if half_head else query_layer, relative_pos))
        score += c2p_att

    # position->content
    if "p2c" in pos_att_type:
        pos_query_layer = pos_q_proj(query_embeddings) if pos_q_proj else query_embeddings

        pos_query_layer = transpose_for_scores(pos_query_layer, heads)
        pos_query_layer = pos_query_layer / math.sqrt(pos_query_layer.size(-1) * scale_factor)
        if query_layer.size(-2) != key_layer.size(-2):
            r_pos = build_relative_position(key_layer.size(-2), query_layer.size(-2), query_layer.device, query_stride=key_stride, key_stride=query_stride)
        else:
            r_pos = relative_pos
        p2c_pos = torch.clamp(-r_pos + q_att_span, 0, min(q_att_span * 2 - 1, query_embeddings.size(0) - 1))
        assert pos_query_layer.size(-1) == key_layer.size(-1)
        p2c_att = torch.matmul(key_layer[:, heads:] if half_head else key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(
            p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer[:, heads:] if half_head else query_layer, key_layer[:, heads:] if half_head else key_layer)
        ).transpose(-1, -2)
        score += p2c_att

    return score


def build_relative_position_2d(qs, ks, device, query_stride=1, key_stride=1):
    """
    Build relative position according to the query and key
    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    rel_pos_row_ids = build_relative_position(qs, ks, device, query_stride, key_stride).view(qs * ks, 1).repeat(1, ks).view(qs, ks ** 2).repeat(1, qs).view(qs ** 2, ks ** 2)
    rel_pos_col_ids = build_relative_position(qs, ks, device, query_stride, key_stride).squeeze().repeat(1, ks).repeat(qs, 1)
    rel_pos_row_ids = rel_pos_row_ids.unsqueeze(0).unsqueeze(1)
    rel_pos_col_ids = rel_pos_col_ids.unsqueeze(0).unsqueeze(1)
    return rel_pos_row_ids, rel_pos_col_ids


def disentangled_att_bias_2d(query_layer, key_layer, row_embed_q, column_embed_q, row_embed_k, column_embed_k, scale_factor, query_max_relative_postions,
                             key_max_relative_positions, heads, query_stride=1, key_stride=1):
    # scale_factor =  5
    row_qs, column_qs, row_ks, column_ks = row_embed_q.size(), column_embed_q.size(), row_embed_k.size(), column_embed_k.size()
    assert heads == query_layer.size(1) or heads == query_layer.size(1)//2
    half_head = heads == query_layer.size(1)//2

    qs = int(math.sqrt(query_layer.size(-2)))
    assert qs*qs == query_layer.size(-2)

    ks = int(math.sqrt(key_layer.size(-2)))
    assert ks * ks == key_layer.size(-2)

    rel_pos_row_ids, rel_pos_col_ids = build_relative_position_2d(qs, ks, query_layer.device, query_stride=query_stride, key_stride=key_stride)

    q_att_span = min(max(qs, ks), query_max_relative_postions)
    row_embed_q = row_embed_q[
        max(0, query_max_relative_postions - q_att_span): min(row_embed_q.size(0), query_max_relative_postions + q_att_span), :
    ].unsqueeze(0)
    column_embed_q = column_embed_q[
                  max(0, query_max_relative_postions - q_att_span): min(column_embed_q.size(0), query_max_relative_postions + q_att_span), :
                  ].unsqueeze(0)



    att_span = min(max(qs, ks), key_max_relative_positions)
    row_embed_k = row_embed_k[
                     max(0, key_max_relative_positions - att_span): min(row_embed_k.size(0), key_max_relative_positions + att_span), :
                     ].unsqueeze(0)
    column_embed_k = column_embed_k[
                  max(0, key_max_relative_positions - att_span): min(column_embed_k.size(0), key_max_relative_positions + att_span), :
                  ].unsqueeze(0)

    score = 0
    # content->position


    r_pos_key_layer = transpose_for_scores(row_embed_k, heads)
    c_pos_key_layer = transpose_for_scores(column_embed_k, heads)
    assert r_pos_key_layer.size(-1) == query_layer.size(-1) and c_pos_key_layer.size(-1) == query_layer.size(-1)

    query_layer_early = query_layer[:, :heads] if half_head else query_layer
    c2p_att = torch.matmul(query_layer_early, r_pos_key_layer.transpose(-1, -2))
    c2p_pos = torch.clamp(rel_pos_row_ids + att_span, 0, min(att_span * 2 - 1, row_ks[0] - 1))
    c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer_early, rel_pos_row_ids))

    score += c2p_att

    c2p_att = torch.matmul(query_layer_early, c_pos_key_layer.transpose(-1, -2))
    c2p_pos = torch.clamp(rel_pos_col_ids + att_span, 0, min(att_span * 2 - 1, column_ks[0] - 1))
    c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer_early, rel_pos_col_ids))
    score += c2p_att
    # position->content


    r_pos_query_layer = transpose_for_scores(row_embed_q, heads)
    c_pos_query_layer = transpose_for_scores(column_embed_q, heads)

    r_pos_query_layer = r_pos_query_layer / math.sqrt(r_pos_query_layer.size(-1) * scale_factor)
    c_pos_query_layer = c_pos_query_layer / math.sqrt(c_pos_query_layer.size(-1) * scale_factor)
    if qs != ks:
        r_pos, c_pos = build_relative_position_2d(ks, qs, query_layer.device, query_stride=key_stride, key_stride=query_stride)
    else:
        r_pos, c_pos = rel_pos_row_ids, rel_pos_col_ids

    r_p2c_pos = torch.clamp(-r_pos + q_att_span, 0, min(q_att_span * 2 - 1, row_qs[0] - 1))
    c_p2c_pos = torch.clamp(-c_pos + q_att_span, 0, min(q_att_span * 2 - 1, column_qs[0] - 1))
    assert r_pos_query_layer.size(-1) == key_layer.size(-1) and c_pos_query_layer.size(-1) == key_layer.size(-1)

    key_layer = key_layer[:, heads:] if half_head else key_layer
    query_layer = query_layer[:, heads:] if half_head else query_layer
    r_p2c_att = torch.matmul(key_layer, r_pos_query_layer.transpose(-1, -2))
    c_p2c_att = torch.matmul(key_layer, c_pos_query_layer.transpose(-1, -2))

    r_p2c_att = torch.gather(
        r_p2c_att, dim=-1, index=p2c_dynamic_expand(r_p2c_pos, query_layer, key_layer)
    ).transpose(-1, -2)
    c_p2c_att = torch.gather(
        c_p2c_att, dim=-1,
        index=p2c_dynamic_expand(c_p2c_pos, query_layer, key_layer)
    ).transpose(-1, -2)

    p2c_att = (r_p2c_att + c_p2c_att)
    if half_head:
        score = torch.cat((score, p2c_att), 1)
    else:
        score += p2c_att

    return score


def get_pretrained_deit(features_only=True):
    import types
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    if features_only:
        model.forward = types.MethodType(forward, model)
        del model.head
    return model


def identity(x):
    return x

def get_cutout(cutout_proba, cutout_size):
    cut = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=cutout_proba, scale=(cutout_size / 2, cutout_size), ratio=(0.4, 2.5), value=0, inplace=False),  # 'random'
        transforms.ToPILImage(),
    ])
    return cut

def get_multi_cuts(n_cuts, cut):
    def multi_cut(im):
        for _ in range(n_cuts):
            im = cut(im)
        return im

    return multi_cut


class get_imgaug:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image):
        return Image.fromarray(self.aug(image=np.array(image, dtype=np.uint8)))


class get_alb:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image):
        try:
            return Image.fromarray(self.aug(image=np.array(image, dtype=np.uint8))['image'])
        except:
            print(image)
            return image


class DefinedRotation(torchvision.transforms.RandomRotation):
    def __init__(self, degrees):
        super().__init__(degrees)

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """

        angle = random.sample(list(degrees), k=1)[0]

        return angle


def get_image_augmetations(mode, teacher=True, dims=224):
    from PIL import Image
    from albumentations import augmentations as alb
    import imgaug.augmenters as iaa
    import torchvision.transforms as transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    crop_224 = transforms.Compose([transforms.Resize(dims), transforms.CenterCrop(dims)])
    to_pytorch = transforms.Compose([transforms.ToTensor(), normalize])
    from_pytorch = transforms.Compose([inv_normalize, transforms.ToPILImage()])
    to_tensor = transforms.Compose([crop_224, to_pytorch])
    shape_transforms = []
    small_shape_transforms = transforms.RandomAffine(10, (0.05, 0.05), (0.9, 1.1), 10)
    cut = transforms.Compose([get_cutout(1.0, 0.02) for _ in range(4)])
    bigcut = transforms.Compose([get_cutout(1.0, 0.02) for _ in range(8)])
    if mode == "linear_probe":
        teacher = True
    if mode == "validation":
        shape_transforms = to_tensor
    else:
        shape_transforms = transforms.Compose([
            transforms.RandomResizedCrop(dims, scale=(0.35, 1.0) if teacher else (0.1, 0.35), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomChoice([
                    transforms.RandomPerspective(distortion_scale=0.2),
                    transforms.RandomRotation(45),
                    DefinedRotation(90),
                    DefinedRotation(180),
                    transforms.RandomAffine(0, (0.0, 0.0), (1.0, 1.0), 20),
                    transforms.RandomAffine(0, (0.0, 0.0), (0.4, 0.8), 0),
                ])],
                p=0.0 if teacher else 0.25),
            # transforms.RandomAffine(0, (0.0, 0.0), (0.8, 1.0) if teacher else (0.6, 1.0), 0),
        ])

    non_shape_transforms = transforms.Compose([
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.1 if teacher else 1.0),
        get_alb(alb.transforms.Solarize(threshold=128, always_apply=False, p=0.2)),
        transforms.RandomApply(
            [transforms.RandomChoice([
                get_alb(alb.transforms.ImageCompression(5, 100, 0, p=1.0, always_apply=True)),
                get_alb(alb.transforms.Equalize(p=1.0, always_apply=True, )),
                get_alb(alb.transforms.Posterize(num_bits=4, always_apply=True, p=1.0)),
                get_imgaug(iaa.AllChannelsCLAHE()),
                get_imgaug(iaa.LogContrast(gain=(0.7, 1.3))),
                get_imgaug(iaa.pillike.Autocontrast((5, 25), per_channel=True)),
                get_alb(alb.transforms.MedianBlur(p=1.0)),
                get_alb(alb.transforms.RandomGamma(p=1.0)),
                get_alb(alb.transforms.RGBShift(p=1.0)),
                get_alb(alb.transforms.MotionBlur(11, p=1.0)),
                get_alb(alb.transforms.GaussNoise(var_limit=(5.0, 25.0), mean=0, always_apply=False, p=1.0)),
            ])],
            p=0.0 if teacher else 0.25),
        transforms.RandomApply(
            [transforms.RandomChoice([
                cut, bigcut,
                get_alb(alb.transforms.GridDropout(ratio=0.5, holes_number_x=8, holes_number_y=8, random_offset=True, p=1.0)),
                get_alb(alb.transforms.GridDropout(ratio=0.3, holes_number_x=8, holes_number_y=8, random_offset=True, p=1.0))
            ])],
            p=0.0 if teacher else 0.25),
    ])

    if mode == "full_train" or mode == "linear_probe":
        shape_transforms = transforms.Compose([shape_transforms, non_shape_transforms, to_tensor])

    return dict(to_tensor=to_tensor, non_shape_transforms=non_shape_transforms, shape_transforms=shape_transforms, small_shape_transforms=small_shape_transforms,
                inv_normalize=inv_normalize, normalize=normalize, crop_224=crop_224, to_pytorch=to_pytorch, from_pytorch=from_pytorch)


def worker_init_fn(worker_id):
    random.seed(tuple(np.array(random.getstate()[1]) + worker_id))
    np.random.seed(np.random.get_state()[1] + worker_id)


class Norm(nn.Module):
    def __init__(self, p=2.0, dim=-1, eps=1e-4):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        assert torch.isfinite(x).all().item()
        x = x / (x.norm(2, -1, True) + self.eps)
        # x = F.normalize(x, self.p, self.dim, self.eps)
        assert torch.isfinite(x).all().item()
        return x


def init_weights(module, std=None):
    if isinstance(module, nn.Sequential):
        for mod in module:
            init_weights(mod)
    else:
        if std is not None:
            pass
        elif hasattr(module, "weight") and len(module.weight.shape) >= 2 and not isinstance(module, nn.Embedding):
            fan_out, fan_in = module.weight.shape[:2]
            std = np.sqrt(1.0 / (float(fan_in + fan_out)/2.0))
        elif hasattr(module, "weight"):
            std = np.sqrt(1.0 / module.weight.shape[-1])
        if hasattr(module, "weight"):
            trunc_normal_(module.weight, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def student_teacher_param_update(student, teacher, m, device=None):
    if device is not None:
        teacher = teacher.to(device)
    named_student_params = dict(getattr(student, "module", student).named_parameters())
    named_teacher_params = teacher.named_parameters()
    with torch.no_grad():
        for name_k, param_k in named_teacher_params:
            if name_k in named_student_params:
                param_k.data.mul_(m).add_((1 - m) * named_student_params[name_k].detach().data)
        backbone = getattr(student, "module", student).backbone
        if hasattr(backbone, "layer_normalizers") and getattr(backbone, "layer_normalizers", None) is not None:
            layer_normalizers = backbone.layer_normalizers
            teacher.backbone.layer_normalizers.mul_(m).add_((1 - m) * layer_normalizers)

        if hasattr(backbone, "layer_normalizers_small") and getattr(backbone, "layer_normalizers_small", None) is not None:
            layer_normalizers_small = backbone.layer_normalizers_small
            teacher.backbone.layer_normalizers_small.mul_(m).add_((1 - m) * layer_normalizers_small)

    if device is not None:
        teacher = teacher.to(torch.device("cpu"))


def get_rolling_diagonal_weights(size=512, window=9):
    diag_mat = sum([torch.diag(torch.ones(size), diagonal=i)[:-i, :-i] for i in range(1, window // 2 + 1)] + [torch.diag(torch.ones(size), diagonal=-i)[i:, i:] for i in range(window // 2 + 1)])
    return diag_mat


def change_dropout(module: nn.Module, dropout_rate=0.0):
    if hasattr(module, "config"):
        module.config.hidden_dropout_prob = dropout_rate
        module.config.attention_probs_dropout_prob = dropout_rate
    str_attrs = dir(module)
    attrs = [(attr, getattr(module, attr, None)) for attr in str_attrs if attr not in  ["base_model", "base_model_prefix"]]
    attrs = [(str_attr, attr) for (str_attr, attr) in attrs if isinstance(attr, (nn.Module, nn.ModuleList, nn.ModuleDict))]
    attrs = [x for (str_attr, attr) in attrs for x in (attr if isinstance(attr, nn.ModuleList) else (attr.values() if isinstance(attr, nn.ModuleDict) else [attr]))]
    for attr in attrs:
        change_dropout(attr)
    if hasattr(module, "dropout"):
        module.dropout.p = dropout_rate
    if hasattr(module, "hidden_dropout"):
        module.hidden_dropout.p = dropout_rate
    if hasattr(module, "attention_dropout"):
        module.attention_dropout.p = dropout_rate


def get_loggable_dict(d):
    rd = dict()
    for k, v in d.items():
        if isinstance(v, (int, float)):
            rd[k] = v
        elif isinstance(v, torch.Tensor):
            try:
                rd[k] = v.item()
            except:
                pass
        else:
            pass
    return rd


def layer_normalizer_fn(embeddings, layer_normalizer, training, train_layer_normalizers, enable_layer_normalizers, enable_layer_normalizers_statistics):
    if layer_normalizer is not None:
        if (training and torch.is_grad_enabled() and train_layer_normalizers) or enable_layer_normalizers_statistics:
            center = embeddings.detach().mean(0).mean(0)
            layer_normalizer[0].mul_(0.9999).add_(0.0001 * center)
        if enable_layer_normalizers:
            embeddings = embeddings - layer_normalizer[0].detach().clone()

        if (training and torch.is_grad_enabled() and train_layer_normalizers) or enable_layer_normalizers_statistics:
            std = embeddings.detach().view(-1, embeddings.size(-1)).std(0)
            layer_normalizer[1].mul_(0.9999).add_(0.0001 * std)
        if enable_layer_normalizers:
            embeddings = embeddings / layer_normalizer[1].detach().clone()

        if (training and torch.is_grad_enabled() and train_layer_normalizers) or enable_layer_normalizers_statistics:
            ne = embeddings.detach() if enable_layer_normalizers else (embeddings.detach() / layer_normalizer[1].detach().clone())
            norm = (ne.norm(2, -1).mean() + 1e-5).expand(embeddings.size(-1))
            layer_normalizer[2].mul_(0.9999).add_(0.0001 * norm)
        if enable_layer_normalizers:
            embeddings = embeddings / layer_normalizer[2].detach().clone()

    return embeddings


def sync_layer_normalizers():
    pass


def get_backbone(model_name, reinit=False, dropout_prob=0.05):
    # TODO: Later also add a QnA boolean / fixed number of options question
    # TODO: Add extra CLS attr and tokens in embedding
    from transformers import AutoModel, RobertaTokenizerFast, BertTokenizerFast, RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer
    import re
    if "co-oc" in model_name:
        temp1 = re.findall(r'\d+', model_name)  # find number of digits through regular expression
        window = 3 if len(temp1) == 0 else int(temp1[0])
        if "roberta" in model_name:
            if "large" in model_name:
                model_name = "roberta-large"
            elif "base" in model_name or "small" in model_name:
                model_name = "roberta-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            model = CoOccurenceModel(window, model_name, tokenizer)
        else:
            raise ValueError("Co-Oc model only supports roberta-base and roberta-large")
    elif "roberta" in model_name:
        if "large" in model_name:
            model_name = "roberta-large"
        elif "base" in model_name or "small" in model_name:
            model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        config = RobertaConfig.from_pretrained(model_name)
        if dropout_prob is not None:
            config.hidden_dropout_prob = dropout_prob
            config.attention_probs_dropout_prob = dropout_prob

        # config.gradient_checkpointing = True
        model = AutoModel.from_pretrained(model_name)
    elif "deberta" in model_name:
        if "xlarge" in model_name:
            model_name = "microsoft/deberta-xlarge-v2"
            model = DebertaV2Model.from_pretrained(model_name)
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        elif "xxlarge" in model_name:
            model_name = "microsoft/deberta-xxlarge-v2"
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            model = DebertaV2Model.from_pretrained(model_name)
        elif "large" in model_name:
            config = DebertaV2Config()
            config.hidden_size = 1024
            config.intermediate_size = 4096
            config.num_attention_heads = 16
            model = DebertaV2Model(config)
            tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-xlarge-v2")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    if reinit:
        model.init_weights()
    if dropout_prob is not None:
        change_dropout(model, dropout_prob)
    return model, tokenizer


def clean_text(text):
    if isinstance(text, (list, tuple)):
        text = " ".join(text)
    text = str(text)
    text = text.lower()
    EMPTY = ' '

    def replace_link(match):
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]*>(.*)</a>', replace_link, text)
    text = re.sub('<.*?>', EMPTY, text)
    text.replace("\\'", "'")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)
    text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', text)
    text = re.sub('[ ]+', ' ', text)

    return text


def init_text_mapper_artifact(tokenizer, sent_detector,
                              use_sbert, use_msmarco, use_albert, use_squeezebert, use_nq):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    sbert = None
    msmarco = None
    albert = None
    squeezebert = None
    nq = None
    albert_tokenizer = None
    squeezebert_tokenizer = None

    if sent_detector is None:
        import nltk
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    if use_sbert:
        sbert = SentenceTransformer("paraphrase-mpnet-base-v2").eval()
    if use_msmarco:
        msmarco = SentenceTransformer("msmarco-distilbert-base-v4").eval()
    if use_albert:
        albert = AutoModel.from_pretrained("albert-base-v2").eval()
        albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    if use_squeezebert:
        squeezebert = AutoModel.from_pretrained("squeezebert/squeezebert-uncased").eval()
        squeezebert_tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
    if use_nq:
        nq = SentenceTransformer('nq-distilbert-base-v1').eval()
    return dict(tokenizer=tokenizer, sent_detector=sent_detector, sbert=sbert, msmarco=msmarco,
                albert=albert, albert_tokenizer=albert_tokenizer,
                squeezebert=squeezebert, squeezebert_tokenizer=squeezebert_tokenizer, nq=nq)


def get_text_mapper(text_cols, total_tokens, keep_above=None,
                    use_sbert=False, use_msmarco=False, use_albert=False, use_squeezebert=False, use_nq=False):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    from typing import Dict, List
    import base64
    import hashlib
    from hashlib import sha3_256, md5
    import re
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    artifact_cache = dict()

    def mapper(examples: Dict[str, List], indices: List[int]=None) -> Dict[str, List]:
        # TODO: remove duplicates after merging for main dataset of MLM
        # TODO: add an id column after making MLM set
        texts = []
        if "artifact" not in artifact_cache:
            artifacts = init_text_mapper_artifact(None, None, use_sbert, use_msmarco, use_albert, use_squeezebert, use_nq)
            artifact_cache["tokenizer"] = artifacts.pop("tokenizer", None)
            artifact_cache["artifact"] = artifacts

        artifact = artifact_cache["artifact"]
        sbert = artifact["sbert"]
        msmarco = artifact["msmarco"]
        albert = artifact["albert"]
        albert_tokenizer = artifact["albert_tokenizer"]
        squeezebert = artifact["squeezebert"]
        squeezebert_tokenizer = artifact["squeezebert_tokenizer"]
        nq = artifact["nq"]
        sent_detector = artifact["sent_detector"]
        tokenizer = artifact_cache["tokenizer"]

        for tcol in text_cols:
            if isinstance(tcol, str):
                texts.append(list(map(clean_text, examples[tcol])))
            elif isinstance(tcol, (list, tuple)):
                ex = examples[tcol[0]]
                for tcol2 in tcol[1:]:
                    ex = [e[tcol2] for e in ex]
                texts.append(list(map(clean_text, ex)))
            else:
                raise NotImplementedError()

        one_texts = [" ".join(one_example) for one_example in zip(*texts)]
        final_texts = []
        # final_hashes = []
        document_hash = []

        for text in one_texts:
            if text is None or len(text) == 0 or (keep_above is not None and len(text.split()) < keep_above):
                continue

            cwc = 0
            cur_seg = ""
            # hashes = []
            dh = str(base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()[:32])
            if len(text.split()) < int(0.8 * total_tokens):
                final_texts.append(text)
                document_hash.append(dh)
                continue

            sents = sent_detector.tokenize(text)

            num_sents = len(sents)
            last_sents = max(int(0.2 * num_sents), 1)
            s1 = " ".join(sents[:-last_sents])

            if num_sents > 2 and len(s1.split()) < int(0.8 * total_tokens):
                final_texts.append(s1)
                document_hash.append(dh)
                final_texts.append(" ".join(sents[-last_sents:]))
                document_hash.append(dh)
                continue

            last_sents = max(int(0.5 * num_sents), 2)
            s1s = " ".join(sents[:-last_sents])
            s2s = " ".join(sents[-last_sents:])
            if num_sents > 2 and len(s1s.split()) < int(0.8 * total_tokens) and len(s2s.split()) < int(0.8 * total_tokens):
                final_texts.extend([s1s, s2s])
                document_hash.extend([dh, dh])
                continue

            last_sents = max(int(0.35 * num_sents), 1)
            s1s = " ".join(sents[:last_sents])
            s2s = " ".join(sents[last_sents:2*last_sents])
            s3s = " ".join(sents[2*last_sents:])
            if num_sents > 2 and len(s1s.split()) < int(0.8 * total_tokens) and len(s2s.split()) < int(0.8 * total_tokens) and len(s3s.split()) < int(0.8 * total_tokens):
                final_texts.extend([s1s, s2s, s3s])
                document_hash.extend([dh, dh, dh])
                continue

            for i, s in enumerate(sents):
                # hash = base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).decode()[:16]
                wc = len(tokenizer.tokenize(s, add_special_tokens=True))
                if cwc + wc < total_tokens or cwc == 0 or i == len(sents) - 1:
                    pass
                else:
                    final_texts.append(cur_seg)
                    # final_hashes.append(hashes[:3])
                    document_hash.append(dh)
                    cwc = 0
                    cur_seg = ""
                    # hashes = []
                cwc += wc
                cur_seg = cur_seg + " " + s
                # hashes.append(hash)
            # TODO: Last sent, big sentence
            final_texts.append(cur_seg.strip())
            # final_hashes.append(hashes[:3])
            document_hash.append(dh)
        # final_lengths = [len(tokenizer.tokenize(t, add_special_tokens=True)) for t in final_texts]
        final_lengths = [int(1.15 * len(t.split())) for t in final_texts]
        assert len(final_lengths) == len(final_texts)
        assert len(final_texts) == len(document_hash)

        results = dict(text=final_texts,
                       length=final_lengths,
                       # hashes=final_hashes,
                       doc_hash=document_hash
                       )
        empty_results = dict(text=[],
                             length=[],
                             doc_hash=[]
                             )
        if albert is not None:
            empty_results["albert"] = []
        if squeezebert is not None:
            empty_results["squeezebert"] = []
        if sbert is not None:
            empty_results["sbert"] = []
        if msmarco is not None:
            empty_results["msmarco"] = []
        if nq is not None:
            empty_results["nq"] = []

            # dl = {key: [fns(item[key], sep_dict[key][0], sep_dict[key][1]) for item in ld] for key in ld[0].keys()}
        if isinstance(keep_above, int):
            dl = results
            ld = [{key: value[index] for key, value in dl.items()} for index in range(max(map(len, dl.values())))]
            ld = [d for d in ld if d["length"] >= keep_above]
            if len(ld) == 0:
                # _ = empty_results.pop("length", None)
                return empty_results
            dl2 = {key:[item[key] for item in ld] for key in ld[0].keys()}
            results = dl2
            # _ = results.pop("length", None)

        with torch.no_grad():
            if albert is not None:
                results["albert"] = albert(**albert_tokenizer.batch_encode_plus(results["text"], add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True, padding=True))["last_hidden_state"].squeeze()[:,
                        0].tolist()
            if squeezebert is not None:
                results["squeezebert"] = squeezebert(**squeezebert_tokenizer.batch_encode_plus(results["text"], add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True, padding=True))["last_hidden_state"].squeeze()[:, 0].tolist()
            if sbert is not None:
                results["sbert"] = sbert.encode(results["text"]).tolist()
            if msmarco is not None:
                results["msmarco"] = msmarco.encode(results["text"]).tolist()
            if nq is not None:
                results["nq"] = nq.encode(results["text"]).tolist()
        for k, v in results.items():
            try:
                assert isinstance(v, (list, tuple))
                assert len(v) == len(list(results.values())[0])
            except AssertionError as ae:
                print(k, len(v), list(results.keys())[0], len(list(results.values())[0]))
                raise ae
        return results
    return mapper


def get_filter_mapper(keep_above):
    def mapper_filter_by_length(examples):
        empty_results = dict(zip(examples.keys(), [[]] * len(examples.keys())))
        dl = examples
        dl["length"] = [len(t.split()) for t in dl["text"]]
        ld = [{key: value[index] for key, value in dl.items()} for index in range(max(map(len, dl.values())))]
        ld = [d for d in ld if d["length"] >= keep_above]
        if len(ld) == 0:
            # _ = empty_results.pop("length", None)
            return empty_results
        dl2 = {key: [item[key] for item in ld] for key in ld[0].keys()}
        return dl2
    return mapper_filter_by_length


class CoOccurenceModel(PreTrainedModel):
    def __init__(self, window, model_name, tokenizer: RobertaTokenizerFast):
        super().__init__(PretrainedConfig())
        model = AutoModel.from_pretrained(model_name)
        config = model.config
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.loss_ce = CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.window = window
        self.lm_head = nn.Linear(256, config.vocab_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, 256)
        self.word_embeddings.weight = nn.Parameter(model.embeddings.word_embeddings.weight[:, :256].clone())
        del model
        self.kernel_size = (2 * window + 1)
        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(1, 1))
        assert config.hidden_size % 8 == 0
        if window <= 3:
            conv1 = nn.Conv2d(256, 256, (1, window), groups=8)
            conv2 = nn.Conv2d(256, config.hidden_size // 2, (1, window+1), groups=8)
            self.conv = nn.Sequential(conv1, nn.GELU(), conv2)
            self.ffn = nn.Sequential(nn.Linear(config.hidden_size // 2, config.hidden_size),
                                     nn.GELU(),
                                     nn.Linear(config.hidden_size, 256),
                                     nn.LayerNorm(256, eps=config.layer_norm_eps))
            init_weights(conv1, 0.01)
            init_weights(conv2, 0.01)
        else:
            conv1 = nn.Conv2d(256, 256, (1, 3), groups=8)
            conv2 = nn.Conv2d(256, 384, (1, window), groups=8)
            conv3 = nn.Conv2d(384, 512, (1, window-1), groups=8)
            init_weights(conv1, 0.01)
            init_weights(conv2, 0.01)
            init_weights(conv3, 0.01)
            self.conv = nn.Sequential(conv1, nn.GELU(), conv2, nn.GELU(), conv3, nn.GELU())
            self.ffn = nn.Sequential(nn.Linear(512, config.hidden_size),
                                     nn.GELU(),
                                     nn.Linear(config.hidden_size, 256),
                                     nn.LayerNorm(256, eps=config.layer_norm_eps))

        self.ln1 = nn.LayerNorm(256, eps=config.layer_norm_eps)
        self.loss_ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
        self.init_weights()
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        b, s = input_ids.shape[:2]
        folded_inputs = self.unfold(F.pad(input_ids, (self.window, self.window), value=self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(-1).float()).type(input_ids.dtype).transpose(1, 2)
        # labels = folded_inputs[:, :, self.window].contiguous()
        # assert torch.all(labels == input_ids).item()
        folded_inputs = torch.cat((folded_inputs[:, :, :self.window], folded_inputs[:, :, self.window+1:]), -1)
        embeddings = self.ln1(self.word_embeddings(folded_inputs))
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings).squeeze(-1).transpose(1, 2)
        embeddings = self.ffn(embeddings)
        prediction_scores = self.lm_head(embeddings)  # B, S, vocab
        masked_lm_loss = self.loss_ce(prediction_scores.view(-1, self.config.vocab_size), input_ids.view(-1))
        lm_predictions = prediction_scores.detach().argmax(dim=-1)
        accuracy = (lm_predictions == input_ids)[attention_mask].float().mean().item()
        _, top_k_alternatives = prediction_scores.detach().topk(16, -1)
        return dict(loss=masked_lm_loss.mean(), accuracy=accuracy,
                    word_ce=masked_lm_loss.detach().view(b, s), top_k_alternatives=top_k_alternatives)


def try_float(v):
    try:
        v = float(v)
        return True
    except:
        return False










