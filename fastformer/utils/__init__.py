import argparse
import numpy as np
import torch
import random
import re
import gc
import datetime
import time
from datetime import datetime, timedelta
from pytz import timezone
import time

import pandas as pd
import random
import os
import argparse
from tqdm.auto import tqdm
import subprocess
import shlex
import shutil
import sys
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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

    setattr(tokenizer, "_seg_sep_token", "[SEG_SEP]")
    tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["seg_sep_token"]
    tokenizer.add_special_tokens({"seg_sep_token": "[SEG_SEP]"})

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
        count_answers = len(re.findall(r'\[QUESTION_[0-9]+\]', answers))
        answers = re.split(r'\[QUESTION_[0-9]+\]', answers)
        all_answers.append(answers)
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
    torch.backends.cudnn.deterministic = True
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
