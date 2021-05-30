from typing import List, Dict

import transformers
from seaborn import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import numpy as np
import gc
import torch
from ..config import FastFormerConfig
from torch.nn import functional as F
import nlpaug.augmenter.char as nac
import unidecode
import os

from fastformer.utils import squeeze_after, get_time_string, recursive_op, gcd_array
from collections import deque
from functools import partial

def get_char_to_id():
    char_to_id = sorted([k for k, v in AutoTokenizer.from_pretrained("bert-base-uncased").get_vocab().items() if len(k) == 1]) + [" ", "\n"]
    char_to_id = dict(zip(char_to_id, range(2, len(char_to_id) + 2)))
    return char_to_id


from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import copy
import random

from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
import time

@dataclass
class TextPretrainingRegularizationConfig:
    word_masking_proba=0.15


import nltk.data
from collections import defaultdict
import re
import random


def isnumber(text):
    try:
        float(text)
        return True
    except:
        pass
    return False


def segment(text, n_segments, sent_detector, pad_token):
    text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', text)
    sents = sent_detector.tokenize(text)
    sent_wc = list(map(lambda x: len(x.split()), sents))
    twc = len(text.split())
    segments = defaultdict(str)
    tol = 0.1
    while len(segments) < n_segments and tol <= (n_segments/2):
        segments = defaultdict(str)
        expected_wc = max(twc // (n_segments + tol), 16)  # Each segment is atleast 16 words
        tol += 0.2
        cwc = 0
        sidx = 0
        for s, wc in zip(sents, sent_wc):
            segments[sidx] = (segments[sidx] + " " + s).strip()
            cwc += wc
            if cwc >= expected_wc and sidx < n_segments - 1:
                cwc = 0
                sidx += 1

    return list(segments.values()) + [pad_token] * (n_segments - len(segments))


punctuation_list = ".,\"'?!;()"


def punctuation_continue(text):
    chars = list(text)
    new_text = []
    for i, c in enumerate(chars):
        if c in punctuation_list and not chars[i - 1].isnumeric() and ((i + 1) < len(chars) and not chars[i + 1].isnumeric()):
            if random.random() < 0.5:
                puncts = "".join([random.sample(".,\"'?!", 1)[0]] * random.randint(1, 3))
            else:
                puncts = "".join([c] * random.randint(1, 3))

            if random.random() < 0.5:
                new_text.append(c)
                new_text.append(puncts)
            else:
                new_text.append(puncts)
                new_text.append(c)
        else:
            new_text.append(c)
    return "".join(new_text)


def punctuation_replace(text):
    chars = list(text)
    for i, c in enumerate(chars):
        if c in punctuation_list and not chars[i - 1].isnumeric() and ((i + 1) < len(chars) and not chars[i + 1].isnumeric()):
            chars[i] = random.sample(punctuation_list, 1)[0]
    return "".join(chars)


def punctuation_strip(text):
    chars = list(text)
    for i, c in enumerate(chars):
        if c in punctuation_list and not chars[i - 1].isnumeric() and ((i + 1) < len(chars) and not chars[i + 1].isnumeric()):
            chars[i] = " "
    text = "".join(chars)
    text = " ".join([w.strip() for w in text.split()])
    return text


punct_augs = dict(punctuation_strip=punctuation_strip, punctuation_replace=punctuation_replace, punctuation_continue=punctuation_continue)


def word_join(w1, w2, mask_token):
    while w1 == mask_token or w2 == mask_token or isnumber(w1) or isnumber(w2):
        raise ValueError()
    return " ".join([w1, w2])


def word_space_separate(text):
    return " ".join(list(text.strip()))


def word_char_swap(text):
    swapper = nac.RandomCharAug(action="swap", aug_char_min=1, aug_char_max=1,
                      aug_word_min=1,
                      aug_word_max=1, include_numeric=False,
                      include_upper_case=False)
    text = swapper.augment(text)
    return text


def word_keyboard(text):
    kb = nac.KeyboardAug(aug_char_min=1, aug_char_max=1, aug_word_min=1,
                    aug_word_max=1, include_special_char=False,
                    include_numeric=False, include_upper_case=False)
    text = kb.augment(text)
    return text

word_augs = dict(word_space_separate=word_space_separate, word_char_swap=word_char_swap, word_keyboard=word_keyboard, word_join=word_join)
augs = list(dict(**word_augs, **punct_augs).items())


def word_level_noising(text, tokenizer, probability=0.15):
    # Avoid [MASK] tokens
    mask_token = tokenizer.mask_token
    pad_token = tokenizer.pad_token
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text
    tokens = text.split()
    new_tokens = []
    skip_next_word = False
    for idx, token in enumerate(tokens):
        if skip_next_word:
            skip_next_word = False
            continue
        if token != mask_token and token != pad_token:
            prob = random.random()
            if prob < probability:
                aug_possibilities = random.sample(augs, k=2)
                if aug_possibilities[0][0] == "word_join" and (idx == len(tokens) - 1 or (tokens[idx + 1] == mask_token or isnumber(token) or isnumber(tokens[idx + 1]))):
                    aug_possibilities = aug_possibilities[1]
                else:
                    aug_possibilities = aug_possibilities[0]
                aug_name, aug_method = aug_possibilities
                if aug_name == "word_join":
                    token = aug_method(token, tokens[idx + 1], mask_token)
                    skip_next_word = True
                else:
                    skip_next_word = False
                    token = aug_method(token)


        new_tokens.append(token)
    new_text = " ".join(new_tokens)
    return new_text


class sample_random_token:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        probas = np.linspace(1, 0.01, len(tokenizer))
        for i in tokenizer.all_special_ids:
            probas[i] = 0
        probas = probas / np.sum(probas)
        self.probas = probas
        self.length = len(tokenizer)

    def __call__(self):
        t_id = random.choices(range(self.length), self.probas)[0]
        text = self.tokenizer.decode(t_id).strip().lower()
        return text, t_id


def span_based_whole_word_masking(text: str, tokenizer, probability: float, vocab: list, max_span_length: int = 1, sampler=None) -> str:
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text
    tokens = text.split()
    new_tokens = []
    skip_next_n_words = 0
    for idx, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens_extended or token in tokenizer.all_special_tokens:
            skip_next_n_words = 0
            new_tokens.append(token)
            continue
        if skip_next_n_words > 0:
            skip_next_n_words -= 1
            continue
        prob = random.random()
        if prob < probability:
            prob /= probability
            token_lengths = len(tokenizer.tokenize(" "+tokens[idx].strip()))
            if prob < 0.9:
                span_size = min(random.sample(range(1, max_span_length + 1), 1)[0], len(tokens) - idx)
                token_lengths = sum([len(tokenizer.tokenize(" "+tokens[idx + i])) for i in range(span_size)])
                tks = [tokenizer.mask_token] * token_lengths
                skip_next_n_words = span_size - 1
            elif sampler is not None:
                tks = [sampler()[0] for _ in range(token_lengths)]
            else:
                tks = [tokenizer.decode(random.sample(range(len(tokenizer)), 1)[0]).strip() for _ in range(token_lengths)]

        else:
            tks = [token]
        new_tokens.extend(tks)
    return " ".join(new_tokens)


def token_masking(text: str, tokenizer, probability: float, vocab: list, max_span_length: int = 1, sampler=None) -> str:
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text

    tokens = np.array(tokenizer.encode(text, add_special_tokens=False))
    probas = np.random.random(len(tokens))
    masked = probas < probability
    rand_replace = probas < (probability * 0.1)
    tokens[masked] = tokenizer.mask_token_id
    if sampler is not None:
        rand_tokens = np.array([sampler()[1] for _ in range(np.sum(rand_replace))])
    else:
        rand_tokens = np.array([random.sample(range(len(tokenizer)), 1)[0] for _ in range(np.sum(rand_replace))])
    tokens[rand_replace] = rand_tokens
    return tokenizer.decode(tokens)


def token_id_masking(tokens, tokenizer, probability: float, vocab: list, max_span_length: int = 1, sampler=None) -> str:

    if probability == 0 or len(tokens) <= 2:
        return tokens

    tokens = np.array(tokens.tolist())
    original_tokens = tokens.copy()
    special_tokens_idx = np.in1d(original_tokens, tokenizer.all_special_ids)
    full_length = np.logical_not(special_tokens_idx).sum()
    probas = np.random.random(len(tokens))
    masked = probas <= probability
    rand_replace = probas < (probability * 0.15)
    tokens[masked] = tokenizer.mask_token_id
    if sampler is not None:
        rand_tokens = np.array([sampler()[1] for _ in range(np.sum(rand_replace))])
    else:
        rand_tokens = np.array([random.sample(range(len(tokenizer)), 1)[0] for _ in range(np.sum(rand_replace))])
    tokens[rand_replace] = rand_tokens

    if full_length > 64:
        for i, t in enumerate(tokens):
            if t == tokenizer.mask_token_id and i > 1 and random.random() <= 0.125:
                tokens[i - 1] = tokenizer.mask_token_id
            if t == tokenizer.mask_token_id and i < len(tokens) - 1 and random.random() <= 0.125:
                tokens[i + 1] = tokenizer.mask_token_id


    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    return torch.tensor(list(tokens))


def char_mapper(char_to_id, x):
    x = x.lower()
    # ud = list(unidecode.unidecode(x.lower()).lower())
    # if len(ud) == 0:
    #     return 1
    # return char_to_id.__getitem__(ud[0])
    if x in char_to_id:
        return char_to_id[x]
    else:
        return 1


def char_rnn_tokenize(text, tokenizer, char_to_id=None, **tokenizer_args):
    # Do padding myself
    text = " ".join([x.strip() for x in text.split()])
    text = " " + text.strip()
    if isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and char_to_id is not None:
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=True, **tokenizer_args)
        offset_mapping = tokenizer_outputs["offset_mapping"]
        offset_mapping -= 1
        offset_mapping = F.relu(offset_mapping)
        # offset_mapping[:, -1] = 1
        char_list = list(text)
        char_lists = list(map(partial(char_mapper, char_to_id), char_list))

        # assert len(char_list) == len(char_lists)
        mxlen = offset_mapping.max().item()
        # assert mxlen < len(char_list)
        tokenizer_outputs["char_ids"] = char_lists[:(mxlen+1)]
        # assert mxlen < len(char_lists)
        assert mxlen < len(tokenizer_outputs["char_ids"])
        tokenizer_outputs["char_offsets"] = offset_mapping.squeeze()
        # assert tokenizer_outputs["input_ids"].shape[1] == tokenizer_args["max_length"]
        del tokenizer_outputs["offset_mapping"]
    else:
        tokenizer_outputs = tokenizer(text, **tokenizer_args)
    tokenizer_outputs["input_ids"] = tokenizer_outputs["input_ids"].squeeze()
    tokenizer_outputs["attention_mask"] = tokenizer_outputs["attention_mask"].squeeze()
    if "token_type_ids" in tokenizer_outputs:
        tokenizer_outputs["token_type_ids"] = tokenizer_outputs["token_type_ids"].squeeze()

    return tokenizer_outputs


class TokenizerDataset(Dataset):
    def __init__(self, config: FastFormerConfig, tokenizer: PreTrainedTokenizerFast,
                 char_to_id: dict, tokenizer_args: dict, dataset: Dataset, sentence_jumble_proba=((0, 0.95), (256, 1.0), (512, 1.0), (1024, 1.0)),
                 word_jumble_proba=((0, 0.05), (128, 0.1), (256, 0.1), (512, 0.15), (1024, 0.2)),
                 word_mask_in_pet=True, word_noise_in_pet=True, sentence_jumble_in_pet=False, word_jumble_in_pet=False,
                 sentence_jumble_pet_length_threshold=192, sentence_jumble_max_segments_in_pet=6,
                 word_mask_proba: list = ((0, 0.05), (128, 0.1), (256, 0.1), (512, 0.15), (1024, 0.2)),
                 word_noise_proba: tuple = ((0, 0.1), (128, 0.1), (256, 0.1), (512, 0.15), (1024, 0.2)),
                 max_span_length: int = 1, max_jumbling_span_length: int = 2, n_anchors: int = 4, n_positives: int = 2):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.min_segments = 2
        self.cls_tokens = config.num_highway_cls_tokens if config is not None else 0
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer_args = tokenizer_args
        labels_pet_text_tokenizer_args = dict(**self.tokenizer_args)
        labels_pet_text_tokenizer_args["add_special_tokens"] = False
        labels_pet_text_max_length = 128
        labels_pet_text_tokenizer_args["max_length"] = labels_pet_text_max_length
        self.labels_pet_text_tokenizer_args = labels_pet_text_tokenizer_args
        self.dataset = dataset
        self.char_to_id = copy.deepcopy(char_to_id)
        self.word_mask_proba = word_mask_proba
        self.vocab = list(tokenizer.get_vocab())
        self.max_span_length = max_span_length
        self.word_noise_proba = word_noise_proba
        self.training = True
        self.wp_l, self.wp_p = zip(*word_mask_proba)
        self.word_mask_in_pet = word_mask_in_pet
        self.word_noise_in_pet = word_noise_in_pet
        self.wn_l, self.wn_p = zip(*word_noise_proba)
        self.wj_l, self.wj_p = zip(*word_jumble_proba)
        self.sj_l, self.sj_p = zip(*sentence_jumble_proba)
        self.max_jumbling_span_length = max_jumbling_span_length
        self.sentence_jumble_in_pet = sentence_jumble_in_pet
        self.sentence_jumble_pet_length_threshold = sentence_jumble_pet_length_threshold
        self.sentence_jumble_max_segments_in_pet = min(sentence_jumble_max_segments_in_pet, self.cls_tokens)
        self.word_jumble_in_pet = word_jumble_in_pet
        self.n_anchors = n_anchors
        self.n_positives = n_positives
        self.token_sampler = sample_random_token(self.tokenizer)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        item = self.dataset[item]
        pet_query = item["query"] if "query" in item and len(item["query"]) > 0 else []
        pet_answer = item["answer"] if "answer" in item and len(item["answer"]) > 0 else []
        label = item["label"] if "label" in item else 0.0


        # TODO: Prompt is added at end of our Seq, labels_seq is generated from an auto-regressive head

        # pet_query = ["how many queens?"] * 4
        # pet_answer = ["eight"] * 4
        assert (pet_query is None and pet_answer is None) or (isinstance(pet_query, str) and isinstance(pet_answer, str)) or (len(pet_query) == len(pet_answer) and isinstance(pet_query, list) and isinstance(pet_answer, list))
        if isinstance(pet_query, str):
            pet_query = [pet_query]
            pet_answer = [pet_answer]

        pet_query = [unidecode.unidecode(pq.strip()) for pq in pet_query if len(pq.strip().split()) > 2]
        n_queries = len(pet_query)
        if n_queries == 0:
            pet_answer = []
        else:
            pet_answer = [unidecode.unidecode(pa.strip()) for pa in pet_answer if len(pa.strip()) > 0]

        text = item["text"]
        if len(text.strip()) == 0:
            text = "empty empty no text empty"
        # assert len(text.strip()) > 0

        text = unidecode.unidecode(text)
        text = " ".join([x.strip() for x in text.split()])
        text = " " + text.strip()
        # assert len(text.strip()) > 0

        results = dict(n_pet_queries=n_queries, answer=pet_answer)

        if self.training:
            tokenizer_outputs = char_rnn_tokenize(text, tokenizer, None, **self.tokenizer_args)
            # TODO: try one in ten words / alternate sentences?
            highway_cls_ar_input_ids, highway_cls_ar__attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs["attention_mask"].squeeze()
            length = torch.sum(highway_cls_ar__attention_mask).item()

            seg_sep_token = f" {tokenizer.seg_sep_token} "

            wp = self.wp_p[np.searchsorted(self.wp_l, length) - 1]
            wn = self.wn_p[np.searchsorted(self.wn_l, length) - 1]
            sj = self.sj_p[np.searchsorted(self.sj_l, length) - 1]
            wj = self.wj_p[np.searchsorted(self.wj_l, length) - 1]

            alpha, beta = (2, 4) if length > 512 else (1, 5)

            cls_tokens_sent_seg = self.cls_tokens if n_queries == 0 else (min(self.cls_tokens, self.sentence_jumble_max_segments_in_pet) if (length > self.sentence_jumble_pet_length_threshold and self.sentence_jumble_in_pet) else 0)
            num_segments = 1 if cls_tokens_sent_seg < self.min_segments else self.min_segments
            count_pad_tokens = 100
            retries = 0
            segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
            while retries < 8 and count_pad_tokens > 0 and cls_tokens_sent_seg >= self.min_segments:
                num_segments = int(np.round(self.min_segments + random.betavariate(alpha, beta) * (cls_tokens_sent_seg - self.min_segments)))
                segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
                count_pad_tokens = sum(segments == tokenizer.pad_token)
                cls_tokens_sent_seg -= 1

            if random.random() < sj and (n_queries == 0 or self.sentence_jumble_in_pet) and count_pad_tokens <= 1 and num_segments > 1:
                seg_idxs = random.sample(range(num_segments), num_segments)
                labels_segment_index = torch.tensor(list(torch.tensor(seg_idxs) + 1) + [0] * (self.cls_tokens - num_segments))
            else:
                seg_idxs = list(range(num_segments))
                if random.random() < 0.5:
                    seg_sep_token = " "
                    labels_segment_index = torch.tensor([1] + [0] * (self.cls_tokens - 1))
                else:
                    labels_segment_index = torch.tensor(list(torch.tensor(seg_idxs) + 1) + [0] * (self.cls_tokens - num_segments))

            segments = segments[seg_idxs]

            in_order = all([c < n or labels_segment_index[i+1:].sum().item() == 0 for i, (c, n) in enumerate(zip(labels_segment_index[:-1].tolist(), labels_segment_index[1:].tolist()))])

            min_anchor_len = 32
            min_positive_len = min_anchor_len - 16
            anchors = []
            positives = []
            if in_order:
                text_len = 8 * (length // 8) - 1
                max_anchor_len = text_len // (2 * self.n_anchors)
                anchor_min_start = 0
                anchor_max_start = anchor_min_start + max_anchor_len
                while anchor_min_start < text_len - min_anchor_len and anchor_max_start <= text_len - min_anchor_len:  # len(anchors) < self.n_anchors and
                    anchor_len = int(random.betavariate(4, 2) * (max_anchor_len - min_anchor_len) + min_anchor_len)
                    anchor_len = int(np.round(anchor_len / 8) * 8)
                    anchor_start = random.randint(anchor_min_start, min(anchor_max_start, max(anchor_min_start + 1, text_len - anchor_len)))
                    anchor_end = min(anchor_start + anchor_len, text_len)
                    anchors.append([anchor_start, anchor_end])
                    positives_for_anchor = []
                    while len(positives_for_anchor) < self.n_positives:
                        positive_len = int(random.betavariate(2, 4) * (max_anchor_len - min_positive_len) + min_positive_len)
                        # positive_len = int(np.round(positive_len / 8) * 8)  # This line is only to make positives the size multiple of 8 for cuda speed
                        positive_start = random.randint(max(0, anchor_start - positive_len), min(anchor_end, text_len - positive_len))
                        positive_end = min(positive_start + positive_len, text_len)
                        positives_for_anchor.append([positive_start, positive_end])
                    positives.append(positives_for_anchor)
                    anchor_min_start = anchor_end + max_anchor_len
                    anchor_max_start = (text_len - min_anchor_len) if len(anchors) >= (self.n_anchors - 1) else (anchor_min_start + max_anchor_len)
                anchors = anchors if min_anchor_len < max_anchor_len else []
                positives = positives if min_anchor_len < max_anchor_len else [[]]
                positives = positives if len(anchors) > 0 else [[]]
            elif count_pad_tokens <= 1 and num_segments >= 2:
                anchors = []
                positives = []
                end = 0
                start = 0
                tokenizer_args = dict(**self.tokenizer_args)
                tokenizer_args["truncation"] = False
                tokenizer_args["max_length"] = 2048
                for seg in segments:
                    seg_len = tokenizer(seg, return_offsets_mapping=False, **tokenizer_args)["attention_mask"].squeeze().sum().item()
                    end += seg_len
                    end = min(end, self.tokenizer_args["max_length"])
                    seg_len = min(seg_len, self.tokenizer_args["max_length"] - start)
                    ##
                    anchor_min_start = start + 8
                    if seg_len > (2 * min_anchor_len) and ((seg_len - 16) - min_anchor_len) > 0:
                        anchor_len = int(random.betavariate(4, 2) * ((seg_len - 16) - min_anchor_len) + min_anchor_len)
                        anchor_max_start = end - anchor_len - 8
                        anchor_start = random.randint(anchor_min_start, min(anchor_max_start, max(anchor_min_start + 1, end - anchor_len)))
                        anchor_end = min(anchor_start + anchor_len, anchor_min_start + (seg_len - 16))
                        anchors.append([anchor_start, anchor_end])
                        positives_for_anchor = []
                        while len(positives_for_anchor) < self.n_positives:
                            positive_len = int(random.betavariate(2, 4) * ((seg_len - 16) - min_anchor_len) + min_anchor_len)
                            # positive_len = int(np.round(positive_len / 8) * 8)  # This line is only to make positives the size multiple of 8 for cuda speed
                            positive_start = random.randint(max(anchor_min_start, anchor_start - positive_len),
                                                            min(anchor_end, anchor_max_start, max(max(anchor_min_start, anchor_start - positive_len) + 1, end - anchor_len)))
                            positive_end = min(positive_start + positive_len, anchor_min_start + (seg_len - 16))
                            positives_for_anchor.append([positive_start, positive_end])
                        positives.append(positives_for_anchor)
                    ##
                    start += seg_len
                    start = min(start, self.tokenizer_args["max_length"])
                positives = positives if len(anchors) > 0 else [[]]

            results.update(dict(labels_segment_index=labels_segment_index, anchors=[], positives=[[]]))

            # TODO: Gap Sentence to be sentence specific not segment specific and have only 3 segments? Or 3 mode + 7 mode so that for small text also we can use sentence jumble
            # TODO: from block 2 CLS tokens remain biased since most small text don't have n_highway segments.
            # TODO: remove if blocks of GSP and word ordering. and also the embedding , have only full ar.
            # TODO: predict segment order as ar task in block 2 from CLS tokens with segments separated by [SEP]
            mlm_text = seg_sep_token.join(segments)  # Training Labels for MLM
            labels_pet_text = "" if len(pet_query) > 0 else tokenizer.no_question_token
            mlm_text = ("" if len(pet_query) > 0 else tokenizer.no_question_token) + mlm_text
            for i, (q, a) in enumerate(zip(pet_query, pet_answer)):
                q, a = q.strip(), a.strip()
                if i == 0:
                    mlm_text = mlm_text + " " + tokenizer.sep_token

                mlm_text = mlm_text + " " + getattr(tokenizer, "question_token_%s" % i) + " " + q + " " + str(len(tokenizer.tokenize(a)))
                assert a is not None
                labels_pet_text += getattr(tokenizer, "question_token_%s" % i) + " " + a
            labels_pet_text = (labels_pet_text + " " + getattr(tokenizer, "answer_end_token")).strip()
            if n_queries > 0:
                tokenizer_outputs = char_rnn_tokenize(labels_pet_text, tokenizer, None, **self.labels_pet_text_tokenizer_args)
                input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
                results.update(dict(labels_pet_input_ids=input_ids.squeeze(),
                                    labels_pet_attention_mask=attention_mask.squeeze()))

            tokenizer_outputs = char_rnn_tokenize(mlm_text, tokenizer, None, **self.tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results["label_mlm_input_ids"] = input_ids.squeeze()

            for idx, seq in enumerate(segments):
                if n_queries == 0 or self.word_mask_in_pet:
                    seq = span_based_whole_word_masking(seq, self.tokenizer, wp, self.vocab, self.max_span_length, sampler=self.token_sampler)
                if n_queries == 0 or self.word_jumble_in_pet:
                    seq = seq.split()
                    new_seq = []
                    for i in range(len(seq))[::self.max_jumbling_span_length]:
                        small_seq = seq[i:i+self.max_jumbling_span_length]
                        if random.random() <= wj and self.tokenizer.mask_token not in small_seq:
                            new_seq.extend(random.sample(small_seq, len(small_seq)))
                        else:
                            new_seq.extend(small_seq)
                    # seq = [w for i in range(len(seq))[::self.max_jumbling_span_length] for w in (random.sample(seq[i:i+self.max_jumbling_span_length], len(seq[i: i+self.max_jumbling_span_length])) if random.random() <= wj and self.tokenizer.mask_token not in seq[i:i+self.max_jumbling_span_length] else seq[i:i+self.max_jumbling_span_length])]
                    seq = " ".join(new_seq).strip()
                if n_queries == 0 or self.word_noise_in_pet:
                    seq = word_level_noising(seq, self.tokenizer, wn)
                segments[idx] = seq

            text = seg_sep_token.join(segments)
        text = ("" if len(pet_query) > 0 else (tokenizer.no_question_token if hasattr(tokenizer, "no_question_token") else "")) + text
        # assert len(text.strip()) > 0

        for i, (q, a) in enumerate(zip(pet_query, pet_answer)):
            q, a = q.strip(), a.strip()
            if i == 0:
                text = text + " " + tokenizer.sep_token

            text = text + " " + getattr(tokenizer, "question_token_%s" % i) + " " + q + " " + tokenizer.mask_token

        inp = char_rnn_tokenize(text, self.tokenizer, self.char_to_id, **self.tokenizer_args)
        results.update(inp)
        if "label" in item:
            results["label"] = label

        length = torch.sum(results["attention_mask"]).item()
        if "positives" in results:
            results["positives"] = recursive_op(results["positives"], lambda x: min(x, length))
            results["anchors"] = recursive_op(results["anchors"], lambda x: min(x, length))
            for positives_for_anchor in results["positives"]:
                for positive in positives_for_anchor:
                    positive[0] = min(positive[0], positive[1] -1)

        return results

    def __len__(self):
        return len(self.dataset)


class MTTDataset(Dataset):
    def __init__(self, cls_tokens, vocab_size, tokenizer: PreTrainedTokenizerFast,
                 tokenizer_args: dict, dataset: Dataset,
                 word_jumble_proba: tuple = ((128, 0.1), (512, 0.15)),
                 word_mask_proba: list = ((128, 0.1), (512, 0.15)),
                 word_noise_proba: tuple = ((128, 0.1), (512, 0.15)),
                 max_span_length: int = 1, max_jumbling_span_length: int = 2, jumble_sentence=True):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.cls_tokens = cls_tokens
        try:
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.word_mask_proba = word_mask_proba
        self.vocab = list(tokenizer.get_vocab())
        self.word_noise_proba = word_noise_proba
        self.training = True
        self.max_span_length = max_span_length
        self.wp_l, self.wp_p = zip(*word_mask_proba)
        self.wn_l, self.wn_p = zip(*word_noise_proba)
        self.wj_l, self.wj_p = zip(*word_jumble_proba)
        self.max_jumbling_span_length = max_jumbling_span_length
        self.jumble_sentence = jumble_sentence
        self.allowed_raw_length = self.tokenizer_args["max_length"] - 48
        self.token_sampler = sample_random_token(self.tokenizer)
        
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        item = self.dataset[item]
        label = item["label"] if "label" in item else 0.0

        text = item["text"]
        length = len(text.strip().split())
        if length == 0:
            text = "empty empty no text empty"
        if length > self.allowed_raw_length:
            text = " ".join(text.split()[:self.allowed_raw_length])
            length = len(text.strip().split())

        text = unidecode.unidecode(text)
        text = " ".join([x.strip() for x in text.split()])
        text = " " + text.strip()
        results = dict()
        acc2 = -1
        if self.training:
            seg_sep_token = f" {tokenizer.sep_token} "

            wp = self.wp_p[np.searchsorted(self.wp_l, length) - 1]
            wn = self.wn_p[np.searchsorted(self.wn_l, length) - 1]
            wj = self.wj_p[np.searchsorted(self.wj_l, length) - 1]
            num_segments = 2 if length > 32 else 1
            segments = np.array(segment(text, num_segments, self.sent_detector, tokenizer.pad_token))
            num_segments = sum(segments != tokenizer.pad_token)

            labels_segment_index = 0
            if num_segments > 1 and self.jumble_sentence:
                if random.random() < 0.5:
                    labels_segment_index = 0
                else:
                    labels_segment_index = 1
                    segments[0], segments[1] = segments[1], segments[0]
                if random.random() < 0.5:
                    seg_sep_token = " "

            results.update(dict(labels_segment_index=labels_segment_index))

            mlm_text = seg_sep_token.join(segments)  # Training Labels for MLM
            tokenizer_outputs = char_rnn_tokenize(mlm_text, tokenizer, None, **self.tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results["label_mlm_input_ids"] = input_ids.squeeze()

            # for idx, seq in enumerate(segments):
                # seq = span_based_whole_word_masking(seq, self.tokenizer, wp, self.vocab, self.max_span_length, sampler=self.token_sampler)

                # seq = seq.split()
                # new_seq = []
                # if self.max_jumbling_span_length > 1:
                #     for i in range(len(seq))[::self.max_jumbling_span_length]:
                #         small_seq = seq[i:i+self.max_jumbling_span_length]
                #         if random.random() <= wj and self.tokenizer.mask_token not in small_seq:
                #             new_seq.extend(random.sample(small_seq, len(small_seq)))
                #         else:
                #             new_seq.extend(small_seq)
                # # seq = [w for i in range(len(seq))[::self.max_jumbling_span_length] for w in (random.sample(seq[i:i+self.max_jumbling_span_length], len(seq[i: i+self.max_jumbling_span_length])) if random.random() <= wj and self.tokenizer.mask_token not in seq[i:i+self.max_jumbling_span_length] else seq[i:i+self.max_jumbling_span_length])]
                # seq = " ".join(new_seq).strip()
                # seq = word_level_noising(seq, self.tokenizer, wn)
                # segments[idx] = seq

            # text = seg_sep_token.join(segments)
            # text = token_masking(text, self.tokenizer, wp, self.vocab, self.max_span_length, sampler=self.token_sampler)
            # acc2 = (np.array(mlm_text.split()) == np.array(text.split())).mean()

            # inp = char_rnn_tokenize(text, self.tokenizer, None, **self.tokenizer_args)
            input_ids = token_id_masking(results["label_mlm_input_ids"], self.tokenizer, wp, self.vocab, self.max_span_length, sampler=self.token_sampler)
            inp = dict(input_ids=input_ids, attention_mask=attention_mask)
        else:
            tokenizer_outputs = char_rnn_tokenize(text, tokenizer, None, **self.tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            inp = dict(input_ids=input_ids, attention_mask=attention_mask)
        if "label" in item:
            results["label"] = label

        if self.cls_tokens > 1:
            dtype = inp["input_ids"].dtype
            inp["input_ids"] = torch.cat((torch.tensor([self.vocab_size + i for i in range(self.cls_tokens - 1)]).type(dtype), inp["input_ids"]))
            dtype = inp["attention_mask"].dtype
            inp["attention_mask"] = torch.cat((torch.tensor([self.vocab_size + i for i in range(self.cls_tokens - 1)]).type(dtype), inp["attention_mask"]))
            if "label_mlm_input_ids" in results:
                results["label_mlm_input_ids"] = torch.cat((torch.tensor([self.vocab_size + i for i in range(self.cls_tokens - 1)]).type(dtype), results["label_mlm_input_ids"]))

        # acc = (inp["input_ids"] != results["label_mlm_input_ids"]).float().mean()
        # print(acc.item(), acc2)
        # print(text,"\n",mlm_text)
        # print(list(zip(list(zip(inp["input_ids"].tolist(), results["label_mlm_input_ids"].tolist())), (inp["input_ids"] == results["label_mlm_input_ids"]).float().tolist())))
        # print(list(zip(self.tokenizer.decode(inp["input_ids"].tolist()).split(), self.tokenizer.decode(results["label_mlm_input_ids"].tolist()).split())))
        # print(list(zip([self.tokenizer.decode(k) for k in inp["input_ids"].tolist()], [self.tokenizer.decode(k) for k in results["label_mlm_input_ids"].tolist()])))
        results.update(inp)

        return results

    def __len__(self):
        return len(self.dataset)


class get_collate_fn:
    def __init__(self, num_cls, padding_index):
        self.num_cls = num_cls
        self.padding_index = padding_index
        self.non_reduced_keys = ["char_ids", "char_offsets", "token_type_ids",
                                 "contrastive_anchors", "contrastive_positives", 'labels_segment_index',
                                 'highway_cls_ar__attention_mask', 'attention_mask',
                                 "labels_pet_attention_mask", 'label_mlm_input_ids', 'answer']

    def __call__(self, samples):
        num_cls = self.num_cls
        padding_index = self.padding_index
        char_ids = None
        max_chars = 0
        if isinstance(samples, list) and isinstance(samples[0], dict) and "char_ids" in samples[0]:
            char_ids = [s["char_ids"] for s in samples]
            for s in samples:
                del s["char_ids"]

            max_chars = max(list(map(len, char_ids)))
            max_chars = int(32 * np.ceil(max_chars / 32))
            char_ids = [torch.tensor(cid) for cid in char_ids]
            char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=padding_index)
            padding = max_chars - char_ids.shape[1]
            char_ids = torch.cat([char_ids, char_ids.new(char_ids.shape[0], padding).fill_(padding_index)], 1)

        # print({key: [d[key].size() if isinstance(d[key], torch.Tensor) else d[key] for d in samples] for key in samples[0].keys()})
        anchors = [s["anchors"] if "anchors" in s else [] for s in samples]
        positives = [s["positives"] if "positives" in s else [[]] for s in samples]
        answer = [s["answer"] if "answer" in s else [] for s in samples]
        for s in samples:
            _ = s.pop("anchors", None)
            _ = s.pop("positives", None)
            _ = s.pop("answer", None)
            if "labels_pet_input_ids" not in s:
                s["labels_pet_input_ids"] = s["input_ids"].new(128).fill_(0)
                s["labels_pet_attention_mask"] = s["input_ids"].new(128).fill_(0)
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    if v.size() != samples[0][k].size():
                        print("Collate size mismatch Key = %s, size_zero = %s, size_cur = %s" % (k, samples[0][k].size(), v.size()))

        samples = default_collate(samples)
        samples["contrastive_anchors"] = anchors
        samples["contrastive_positives"] = positives
        # print(anchors, positives)
        samples["answer"] = answer
        if char_ids is not None:
            samples["char_ids"] = char_ids
        # TODO: reduce the batch seq length to minimum required and a multiple of 16.
        # print({k: v.size() for k, v in samples.items() if hasattr(v, "size")})
        # ['labels_segment_index', 'highway_cls_ar_input_ids', 'highway_cls_ar__attention_mask', 'label_mlm_input_ids', 'input_ids',
        # 'token_type_ids', 'attention_mask', 'char_offsets', 'labels_pet_input_ids', 'labels_pet_attention_mask', 'contrastive_anchors', 'contrastive_positives', 'char_ids']
        samples = {k: squeeze_after(v, 0) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        input_reduce_key = "label_mlm_input_ids" if "label_mlm_input_ids" in samples else 'input_ids'
        other_non_reduced_pair = "label_mlm_input_ids" if "label_mlm_input_ids" not in samples else 'input_ids'
        reduced_keys = ['highway_cls_ar_input_ids', input_reduce_key, 'labels_pet_input_ids']

        for k, v in samples.items():
            if k in self.non_reduced_keys or len(v.size()) < 2 or k not in reduced_keys:
                continue

            step_size = 8
            while len(v[:, -step_size:].unique()) == 1 and v[:, -step_size:].unique().item() == padding_index and v.shape[1] > step_size:
                v = v[:, :-step_size]
            if k != "labels_pet_input_ids":  # Because this don't get highway_cls tokens added.
                required_len = int(step_size * np.ceil(v.shape[1]/step_size))
                required_len = required_len - step_size + (step_size - num_cls)
                padding = required_len - v.shape[-1]
                if padding > 0:
                    v = torch.cat([v, v.new(v.shape[0], padding).fill_(padding_index)], 1)
                elif padding < 0:
                    v = v[:, :padding]
            samples[k] = v
        # print(type(samples), samples.keys(), {k: v.size() for k, v in samples.items() if hasattr(v, "size")})
        if other_non_reduced_pair in samples:
            samples[other_non_reduced_pair] = samples[other_non_reduced_pair][:, :samples[input_reduce_key].shape[1]]
        samples['attention_mask'] = samples['attention_mask'][:, :samples['input_ids'].shape[1]]
        if "highway_cls_ar_input_ids" in samples:
            samples['highway_cls_ar__attention_mask'] = samples['highway_cls_ar__attention_mask'][:, :samples['highway_cls_ar_input_ids'].shape[1]]
        if "labels_pet_input_ids" in samples:
            samples['labels_pet_attention_mask'] = samples['labels_pet_attention_mask'][:, :samples['labels_pet_input_ids'].shape[1]]
        if "token_type_ids" in samples:
            # samples['token_type_ids'] = samples['token_type_ids'][:, :samples['input_ids'].shape[1]]
            del samples['token_type_ids']
        if "char_offsets" in samples:
            samples['char_offsets'] = samples['char_offsets'][:, :samples['input_ids'].shape[1]]
        samples = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        # {k: (v.size() if hasattr(v, "size") else len(v), type(v)) for k, v in samples.items()}
        # mxoff = samples['char_offsets'].max().item()
        # char_ids_shape = samples["char_ids"].shape[1]
        # try:
        #     assert mxoff < char_ids_shape
        # except:
        #     print("char_offsets > char_ids, max offset = %s, char_ids max = %s, Max chars before pad = %s" % (mxoff, char_ids_shape, max_chars_temp))
        # assert samples["input_ids"].size(1) > 0
        return samples


def datadict_iterator(dict_loader, dict_probas, continuous_iter=True):
    keys, probas = zip(*list(dict_probas.items()))
    dict_loader = dict(**dict_loader)
    dict_loader_temp = dict(**dict_loader)
    dict_loader = {k: iter(v) for k, v in dict_loader.items()}

    err_count = defaultdict(int)
    while len(dict_loader) > 0:
        cur_key = random.choices(keys, probas)[0]
        try:
            yield next(dict_loader[cur_key])
        except StopIteration as st:
            if continuous_iter:
                dict_loader[cur_key] = iter(dict_loader_temp[cur_key])
            else:
                _ = dict_loader.pop(cur_key, None)
                _ = dict_probas.pop(cur_key, None)
                keys, probas = zip(*list(dict_probas.items()))
        except Exception as e:
            err_count[cur_key] = err_count[cur_key] + 1
            print(cur_key, e)
            if err_count[cur_key] > 100:
                print("Remove Dataset %s due to errors" % cur_key)
                _ = dict_loader.pop(cur_key, None)
                _ = dict_probas.pop(cur_key, None)
                keys, probas = zip(*list(dict_probas.items()))

    raise StopIteration()


def merge_one(b1v, b2v):
    if isinstance(b1v, torch.Tensor) and len(b1v.size()) > 1:
        b1vs = b1v.size(1)
        b2vs = b2v.size(1)
        if b1vs > b2vs:
            padding = b1vs - b2vs
            b2v = torch.cat([b2v, b2v.new(b2v.shape[0], padding, *b2v.shape[2:]).fill_(0)], 1)
        elif b1vs < b2vs:
            padding = b2vs - b1vs
            b1v = torch.cat([b1v, b1v.new(b1v.shape[0], padding, *b1v.shape[2:]).fill_(0)], 1)
        return torch.cat((b1v, b2v), 0).contiguous()
    elif isinstance(b1v, torch.Tensor):
        return torch.cat((b1v, b2v), 0).contiguous()
    elif isinstance(b1v, (list, tuple)):
        return b1v + b2v
    else:
        raise TypeError


def batch_merge(b1, b2):
    fb = dict()
    for b1k, b1v in b1.items():
        b2k = b1k
        b2v = b2[b2k]
        # fb[b1k] = merge_one(b1v, b2v)
        try:
            fb[b1k] = merge_one(b1v, b2v)
        except Exception as e:
            print((b1k, b2k,), b1v.size(), b2v.size(),  b1v,  b2v, e)
            raise e
    return fb


def custom_batching_fn_old(dataloader, batch_size_dict, continuous_iter=True):
    size, batch_size = zip(*list(batch_size_dict.items()))
    min_batch_size = min(batch_size)
    i = 1
    cur_iter = 1
    prev_batch = None
    # if prev is small but cur is big then keep prev and yield cur
    # if prev is big and cur is big then yield prev and cur

    # if prev is big and cur is small then yield prev keep cur
    # if prev is small and cur is small then merge
    while i > 0:
        print("%s [custom_batching_fn]: Start Epoch = %s" % (get_time_string(), cur_iter))
        start_time = time.time()
        for _, cur_batch in enumerate(dataloader):
            if prev_batch is None:
                prev_batch = cur_batch
            else:
                prev_seq_len = prev_batch["input_ids"].size(-1)
                cur_seq_len = cur_batch["input_ids"].size(-1)
                prev_mx_bt_size = batch_size[np.searchsorted(size, prev_seq_len)]
                cur_mx_bt_size = batch_size[np.searchsorted(size, cur_seq_len)]

                # TODO: can we merge or not, if we merge then dont yield else yield and make prev_batch=cur_batch
                prev_batch_size = prev_batch["input_ids"].size(0)
                cur_batch_size = cur_batch["input_ids"].size(0)
                actual_batch_size_post_merge = prev_batch_size + cur_batch_size
                can_we_merge = actual_batch_size_post_merge <= prev_mx_bt_size and actual_batch_size_post_merge <= cur_mx_bt_size
                if can_we_merge:
                    # if prev is small and cur is small then merge
                    prev_batch = batch_merge(prev_batch, cur_batch)

                elif actual_batch_size_post_merge > prev_mx_bt_size and prev_batch_size >= (prev_mx_bt_size - min_batch_size):
                    # if prev is big and cur is small then yield prev keep cur
                    yield prev_batch
                    prev_batch = cur_batch
                elif actual_batch_size_post_merge > cur_mx_bt_size and cur_batch_size >= (cur_mx_bt_size - min_batch_size):
                    # if prev is small but cur is big then keep prev and yield cur
                    yield cur_batch
                else:
                    if prev_batch_size > cur_batch_size or (prev_batch_size == cur_batch_size and prev_seq_len >= cur_seq_len):
                        yield prev_batch
                        prev_batch = cur_batch
                    else:
                        yield cur_batch

        if not continuous_iter:
            i = i - 1
        tot_time = time.time() - start_time
        print("%s [custom_batching_fn]: End Epoch = %s, Time Taken = %.0f" % (get_time_string(), cur_iter, tot_time))
        cur_iter += 1


def custom_batching_fn(dataloader, batch_size_dict, continuous_iter=True):
    size, batch_size = zip(*list(batch_size_dict.items()))
    min_batch_size = gcd_array(batch_size)
    keep_iterating = True
    cur_iter = 1
    rand_idx = 49
    batch_cache = list()
    batch_age = list()
    maxlen = 100
    # If it can't be merged and can't be yielded then store.
    while keep_iterating:
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(cur_iter + rand_idx)
            print("Time = %s [custom_batching_fn]: Distributed Sampler Epoch = %s" % (get_time_string(), cur_iter))
        else:
            print("Time = %s [custom_batching_fn]: Unable to set Epoch = %s" % (get_time_string(), cur_iter))
        start_time = time.time()
        for _, cur_batch in enumerate(dataloader):

            cur_seq_len = cur_batch["input_ids"].size(-1)
            cur_mx_bt_size = batch_size[np.searchsorted(size, cur_seq_len)]
            cur_batch_size = cur_batch["input_ids"].size(0)
            batch_cache.sort(key=lambda x: x["input_ids"].size(0) * x["input_ids"].size(-1))

            temp = []
            age_temp = []
            can_we_merge = False
            while batch_cache:
                prev_batch = batch_cache.pop()
                prev_age = batch_age.pop()
                prev_seq_len = prev_batch["input_ids"].size(-1)
                prev_batch_size = prev_batch["input_ids"].size(0)
                prev_mx_bt_size = batch_size[np.searchsorted(size, prev_seq_len)]
                if prev_batch_size > (prev_mx_bt_size - min_batch_size) or prev_age > maxlen * 10:
                    yield prev_batch
                else:
                    actual_batch_size_post_merge = prev_batch_size + cur_batch_size
                    can_we_merge = actual_batch_size_post_merge <= prev_mx_bt_size and actual_batch_size_post_merge <= cur_mx_bt_size
                    if can_we_merge and cur_batch is not None:
                        prev_batch = batch_merge(prev_batch, cur_batch)
                        cur_batch = None
                    temp.append(prev_batch)
                    age_temp.append(prev_age + 1)

            batch_cache = temp
            batch_age = age_temp
            if not can_we_merge and len(batch_cache) < maxlen and cur_batch_size <= (cur_mx_bt_size - min_batch_size) and cur_batch is not None:
                batch_cache.append(cur_batch)
                batch_age.append(1)
            elif cur_batch is not None:
                yield cur_batch

        keep_iterating = continuous_iter
        tot_time = time.time() - start_time
        print("%s [custom_batching_fn]: End Epoch = %s, Time Taken = %.0f" % (get_time_string(), cur_iter, tot_time))
        cur_iter += 1


class DummyDataset:

    def __init__(self, length):
        self.length = length

    def __getitem__(self, i):
        return {"input_ids": torch.randn(512)}

    def __len__(self):
        return self.length

"""

from torch.utils.data.dataloader import DataLoader
from fastformer.data import custom_batching_fn
dd = DummyDataset(128)
loader = DataLoader(dd, sampler=None, batch_size=8, num_workers=0)
size_dicts = {128:96, 256:48, 512: 24, 768: 16, 1024: 8}
train_loader = custom_batching_fn(loader, size_dicts, True)
samples_done = 0
for i, b in enumerate(train_loader):
    samples_done += b["input_ids"].size(1)
    print(i, samples_done)
"""



def all_datasets():
    from datasets import load_dataset
    # Done MLM
    ## Large
    bookcorpus = load_dataset("bookcorpus")
    bookcorpusopen = load_dataset("bookcorpusopen")
    openwebtext = load_dataset("openwebtext")
    wikipedia = load_dataset("wikipedia", '20200501.en')  # select the right title for article
    reddit = load_dataset("reddit")  # Dont take texts below 64? Or combine small ones with big ones for threading structure?
    big_patent = load_dataset("big_patent", 'all', script_version="master")
    amazon_us_reviews = dict()
    for ds in ['Wireless_v1_00', 'Watches_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', 'Video_v1_00', 'Toys_v1_00', 'Tools_v1_00', 'Sports_v1_00',
               'Software_v1_00',
               'Shoes_v1_00', 'Pet_Products_v1_00', 'Personal_Care_Appliances_v1_00',
               # 'PC_v1_00',
               'Outdoors_v1_00', 'Office_Products_v1_00', 'Musical_Instruments_v1_00', 'Music_v1_00', 'Mobile_Electronics_v1_00', 'Mobile_Apps_v1_00',
               'Major_Appliances_v1_00', 'Luggage_v1_00', 'Lawn_and_Garden_v1_00', 'Kitchen_v1_00', 'Jewelry_v1_00', 'Home_Improvement_v1_00',
               'Home_Entertainment_v1_00', 'Home_v1_00', 'Health_Personal_Care_v1_00', 'Grocery_v1_00', 'Gift_Card_v1_00', 'Furniture_v1_00',
               'Electronics_v1_00', 'Digital_Video_Games_v1_00', 'Digital_Video_Download_v1_00', 'Digital_Software_v1_00', 'Digital_Music_Purchase_v1_00',
               'Digital_Ebook_Purchase_v1_00', 'Camera_v1_00', 'Books_v1_00', 'Beauty_v1_00', 'Baby_v1_00', 'Automotive_v1_00', 'Apparel_v1_00',
               'Digital_Ebook_Purchase_v1_01', 'Books_v1_01', 'Books_v1_02']:
        amazon_us_reviews[ds] = load_dataset("amazon_us_reviews", ds, script_version="master")
    amazon_us_reviews = concatenate_datasets(list([t["train"] for t in amazon_us_reviews.values()]))
    amazon_us_reviews = amazon_us_reviews.filter(lambda e: len(e["review_body"].split()) > 64)

    ## Medium
    yahoo_answers_topics = load_dataset("yahoo_answers_topics")
    amazon_polarity = load_dataset("amazon_polarity", script_version="master")
    scientific_papers_pubmed = load_dataset("scientific_papers", 'pubmed')
    scientific_papers_arxiv = load_dataset("scientific_papers", 'arxiv')

    ## Small
    yahoo_answers_qa = load_dataset("yahoo_answers_qa")  # World knowledge testing rather than answer selection.
    reuters_hayes = load_dataset("reuters21578", 'ModHayes')
    reuters_lewis = load_dataset("reuters21578", 'ModLewis')
    reuters_apte = load_dataset("reuters21578", 'ModApte')
    reuters = concatenate_datasets([d[split] for d in [reuters_hayes, reuters_lewis, reuters_apte] for split in ["train", "test"]])
    ohsumed = load_dataset("ohsumed", script_version="master")
    xsum = load_dataset("xsum")
    eli5 = load_dataset("eli5")  # sentence ordering task, order answers in order of upvotes.
    cnn_dailymail = load_dataset("cnn_dailymail", '3.0.0')
    # Below 2 are same as above
    cnn_dailymail = load_dataset("cnn_dailymail", '2.0.0')
    cnn_dailymail = load_dataset("cnn_dailymail", '1.0.0')
    yelp_polarity = load_dataset("yelp_polarity")
    # Source of both these is same
    yelp_review_full = load_dataset("yelp_review_full", script_version="master")
    amazon_reviews_multi = load_dataset("amazon_reviews_multi", 'en')

    wmt14de_en = load_dataset("wmt14", 'de-en')  # ['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en']
    giga_fren = load_dataset("giga_fren", 'en-fr', script_version="master")
    wmt16ru_en = load_dataset("wmt16", 'ru-en')
    wmt16ru_en = wmt16ru_en.filter(lambda e: len(e["translation"]['en'].split()) > 64, num_proc=16)
    wmt17cs_en = load_dataset("wmt17", "cs-en")
    wmt17cs_en = wmt17cs_en.filter(lambda e: len(e["translation"]['en'].split()) > 64, num_proc=16)

    to_en = dict()
    for ds in ['af-en', 'am-en', 'an-en', 'ar-en', 'as-en', 'az-en', 'be-en', 'bg-en', 'bn-en', 'br-en', 'bs-en', 'ca-en', 'cs-en', 'cy-en', 'da-en', 'de-en',
               'dz-en', 'el-en']:
        to_en[ds] = load_dataset("opus100", ds, script_version="master")
    to_en = concatenate_datasets(
        [d[split].map(lambda x: dict(text=x["translation"]["en"]), remove_columns=['translation'], num_proc=16) for d in to_en.values() for split in
         ["train", "test", "validation"] if split in d])
    to_en = to_en.filter(lambda e: len(e["text"].split()) > 64, num_proc=16)




    # Not Done
    medal = load_dataset("medal", script_version="master")
    cc100_en = load_dataset("cc100", lang="en", script_version="master")  # http://data.statmt.org/cc-100/
    # cc100_en = load_dataset(path="cc100.py", lang="en", script_version="master")

    wmt15fr_en = load_dataset("wmt15", "fr-en")


    un_pc = load_dataset("un_pc", 'en-fr', script_version="master")  # ['ar-de', 'ar-en', 'ar-es', 'ar-fr', 'ar-ru', 'ar-zh', 'de-en', 'de-es', 'de-fr', 'de-ru', 'de-zh', 'en-es', 'en-fr', 'en-ru', 'en-zh', 'es-fr', 'es-ru', 'es-zh', 'fr-ru', 'fr-zh', 'ru-zh']
    un_pc = un_pc.filter(lambda e: len(e["translation"]['en'].split()) > 64, num_proc=8)
    un_pc = load_dataset("un_pc", 'en-ru', script_version="master")
    emea = load_dataset("emea", lang1="en", lang2="nl", script_version="master")

    un_ga_fr = load_dataset("un_ga", 'en_to_fr', script_version="master")
    menyo20k_mt = load_dataset("menyo20k_mt", script_version="master")
    hind_encorp = load_dataset("hind_encorp", script_version="master")
    opus100_en_fr = load_dataset("opus100", 'en-fr', script_version="master")
    opus100_en_ru = load_dataset("opus100", 'en-ru', script_version="master")



    opus_tedtalks = load_dataset("opus_tedtalks", script_version="master")
    capes = load_dataset("capes", script_version="master")
    multi_x_science_sum = load_dataset("multi_x_science_sum")
    app_reviews = load_dataset("app_reviews", script_version="master")
    news_commentary = load_dataset("news_commentary", "en-fr", script_version="master")
    scielo = load_dataset("scielo", 'en-pt', script_version="master")
    scb_mt_enth_2020 = load_dataset("scb_mt_enth_2020", 'enth', script_version="master")
    setimes = load_dataset("setimes", 'en-sr', script_version="master")


    imdb = load_dataset("imdb", script_version="master")

    # generics_kb = load_dataset("generics_kb",'generics_kb_best', script_version="master")
    open_subtitles = load_dataset("open_subtitles", 'en-hi', script_version="master")

    xsum_factuality = load_dataset("xsum_factuality", "xsum_faithfulness", script_version="master")
    xsum_factuality = load_dataset("xsum_factuality", "xsum_factuality", script_version="master")


    wiki_lingua = load_dataset("wiki_lingua", 'english', script_version="master")
    samsum = load_dataset("samsum", script_version="master")

    wikihow_all = load_dataset("wikihow", 'all', data_dir='/local/') # Correct order (title +  headline) || (title + text), title + text match, Correct order (title + text)
    wikihow_sep = load_dataset("wikihow", 'sep', data_dir='/local/') # headline+text match/mlm, title+text match,
    multi_news = load_dataset("multi_news")

    wiki_auto = load_dataset("wiki_auto", 'auto_acl')  # select the right summary from simple wiki
    ag_news = load_dataset("ag_news")
    gigaword = load_dataset("gigaword")  # select the correct summary from list of summaries
    kelm = load_dataset("kelm", script_version="master")
    wiki_atomic_edits_insertions = load_dataset("wiki_atomic_edits", 'english_insertions', script_version="master")
    wiki_atomic_edits_deletions = load_dataset("wiki_atomic_edits", 'english_deletions', script_version="master")
    wiki_split = load_dataset("wiki_split", script_version="master")  # Matching task?
    dbpedia_14 = load_dataset("dbpedia_14", script_version="master")
    tuple_ie = load_dataset("tuple_ie", 'all', script_version="master")


    for ds in ['el-en', 'cs-en', 'en-hu', 'en-ro', 'en-sk', 'en-uk', 'en-ja', 'en-es', 'en-fr', 'de-en', 'en-ko', 'en-zh', 'en-ru', 'en-pt']:
        ppt = load_dataset("para_pat", ds, script_version="master")

    un_multi = load_dataset("un_multi", 'en-fr', script_version="master")


    # cola, sst2, qqp, qnli
    glue = dict()
    for gl in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:
        glue[gl] = load_dataset("glue", gl)

    super_glue = dict()
    for gl in ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']:
        super_glue[gl] = load_dataset("super_glue", gl)

    health_fact = load_dataset("health_fact", script_version="master")

    google_wellformed_query = load_dataset("google_wellformed_query", script_version="master")
    per_sent = load_dataset("per_sent", script_version="master")
    selqa = load_dataset("selqa", 'answer_selection_experiments', script_version="master")  # Is the answer given correct / relevant
    selqa = load_dataset("selqa", 'answer_selection_analysis', script_version="master")  # select the text that seems like the right / relevant answer
    selqa = load_dataset("selqa", 'answer_triggering_experiments', script_version="master")
    esnli = load_dataset("esnli")  # SNLI with explanations, explanations can be used along with the 2 statements for labelling
    anli = load_dataset("anli")
    hans = load_dataset("hans")
    scitail = load_dataset("scitail", 'snli_format')
    go_emotions = load_dataset("go_emotions", 'raw', script_version="master")
    discovery = load_dataset("discovery", 'discovery', script_version="master")
    paws = load_dataset("paws", 'labeled_final', script_version="master")

    codah = load_dataset("codah", "codah")
    swag = load_dataset("swag", 'regular')
    hellaswag = load_dataset("hellaswag")
    qangaroo = load_dataset("qangaroo", 'wikihop')
    qasc = load_dataset("qasc") # 3 tasks can be built
    wiqa = load_dataset("wiqa")
    search_qa = load_dataset("search_qa", 'train_test_val')
    hotpot_qa = load_dataset("hotpot_qa", 'distractor')
    hotpot_qa = load_dataset("hotpot_qa", 'fullwiki') # Same as distractor
    inquisitive_qg = load_dataset("inquisitive_qg", script_version="master")
    squad = load_dataset("squad")
    squad_v2 = load_dataset("squad_v2")
    squad_adversarial = load_dataset("squad_adversarial", 'AddSent', script_version="master")
    ropes = load_dataset("ropes")

    tweet_qa = load_dataset("tweet_qa", script_version="master")
    trivia_qa = load_dataset("trivia_qa", "rc")
    wiki_qa = load_dataset("wiki_qa")  # Is answer correct / relevant or not
    narrativeqa = load_dataset("narrativeqa", script_version="master") # Use the summary column and treat this as a Matching task with matching question to answer.
    mc_taco = load_dataset("mc_taco", script_version="master")
    social_i_qa = load_dataset("social_i_qa", script_version="master")
    quac = load_dataset("quac", script_version="master")
    asset_simplification = load_dataset("asset", 'simplification', script_version="master")
    asset_ratings = load_dataset("asset", 'ratings', script_version="master")
    e2e_nlg_cleaned = load_dataset("e2e_nlg_cleaned", script_version="master")
    youtube_caption_corrections = load_dataset("youtube_caption_corrections", script_version="master")
    europa_eac_tm = load_dataset("europa_eac_tm", 'en2fr', script_version="master")

    sent_comp = load_dataset("sent_comp", script_version="master")

    ms_marco = load_dataset("ms_marco", 'v2.1')
    quarel = load_dataset("quarel")
    quartz = load_dataset("quartz")
    mocha = load_dataset("mocha", script_version="master")
    quail = load_dataset("quail")
    quoref = load_dataset("quoref")
    race = load_dataset("race", 'all')
    winogrande = load_dataset("winogrande", 'winogrande_xl')

    kilt = load_dataset("kilt_tasks")
    com_qa = load_dataset("com_qa")
    qed = load_dataset("qed", script_version="master")
    commonsense_qa = load_dataset("commonsense_qa")
    cosmos_qa = load_dataset("cosmos_qa")
    mrqa = load_dataset("mrqa", script_version="master")
    natural_questions = load_dataset("natural_questions")
    piqa = load_dataset("piqa")
    pubmed_qa = load_dataset("pubmed_qa", 'pqa_labeled', script_version="master")
    quora = load_dataset("quora")
    # pubmed = load_dataset("pubmed", script_version="master")

    biomrc_large_A = load_dataset("biomrc", 'biomrc_large_A')
    biomrc_large_B = load_dataset("biomrc", 'biomrc_large_B')
    med_hop = load_dataset("med_hop", 'original', script_version="master")
    covid_qa_deepset = load_dataset("covid_qa_deepset", script_version="master")
    sciq = load_dataset("sciq")
    peer_read_reviews = load_dataset("peer_read", 'reviews', script_version="master")
    # peer_read_pdf = load_dataset("peer_read", 'parsed_pdfs', script_version="master")
    conv_ai_3 = load_dataset("conv_ai_3", script_version="master")
    daily_dialog = load_dataset("daily_dialog")
    medical_questions_pairs = load_dataset("medical_questions_pairs", script_version="master")
    empathetic_dialogues = load_dataset("empathetic_dialogues")

    ai2_arc = load_dataset("ai2_arc", 'ARC-Challenge')
    ai2_arc_easy = load_dataset("ai2_arc", 'ARC-Easy')
    circa = load_dataset("circa")
    zest = load_dataset("zest", script_version="master")
    drop = load_dataset("drop")
    eraser_multi_rc = load_dataset("eraser_multi_rc")
    conceptnet5 = load_dataset("conceptnet5", 'conceptnet5', script_version="master")  # Can make cloze task by asking what is the relation between the 2 words, only when sentence field is also present
    crawl_domain = load_dataset("crawl_domain", script_version="master")
    numer_sense = load_dataset("numer_sense")

    conll2003 = load_dataset("conll2003")
    polyglot_ner = load_dataset("polyglot_ner", 'en')
    acronym_identification = load_dataset("acronym_identification", script_version="master")
    limit = load_dataset("limit", script_version="master")
    wikiann = load_dataset("wikiann", 'en', script_version="master")
    # blog_authorship_corpus = load_dataset("blog_authorship_corpus")
    ptb_text_only = load_dataset("ptb_text_only", script_version="master")
    rotten_tomatoes = load_dataset("rotten_tomatoes")
    sentiment140 = load_dataset("sentiment140")
    emotion = load_dataset("emotion")



    scitldr = load_dataset("scitldr", 'Abstract', script_version="master")

    wiki_asp = dict()
    for ds in ['album', 'animal', 'artist', 'building', 'company', 'educational_institution', 'event', 'film', 'group', 'historic_place', 'infrastructure', 'mean_of_transportation', 'office_holder', 'plant', 'single', 'soccer_player', 'software', 'television_show', 'town', 'written_work']:
        wiki_asp[ds] = load_dataset("wiki_asp", ds, script_version="master")
    wiki_asp = concatenate_datasets(list([t["train"] for t in wiki_asp.values()]))

    taskmaster2 = dict()
    for ds in ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']:
        taskmaster2[ds] = load_dataset("taskmaster2", ds, script_version="master")
    taskmaster2 = concatenate_datasets(list([t["train"] for t in taskmaster2.values()]))

    # qa4mre = dict()
    # for ds in ['2011.main.EN', '2012.main.EN', '2013.main.EN', '2013.entrance_exam.EN', '2012.alzheimers.EN', '2013.alzheimers.EN']:
    #     qa4mre[ds] = load_dataset("qa4mre", ds, script_version="master")
    # qa4mre = concatenate_datasets(list(qa4mre.values()))

    seval = load_dataset("joelito/sem_eval_2010_task_8")  # See: https://huggingface.co/datasets/joelito/sem_eval_2010_task_8


    # math_qa = load_dataset("math_qa")
    # docred = load_dataset("docred")
    # lama = load_dataset("lama", 'trex', script_version="master") # Don't train on this, fact checking dataset
    # openbookqa = load_dataset("openbookqa", 'additional') # fact checking dataset
    # aqua_rat  = load_dataset("aqua_rat", 'raw', script_version="master") # numer_sense
    # jeopardy = load_dataset("jeopardy", script_version="master") # fact checking dataset
    # common_gen = load_dataset("common_gen", script_version="master")
    # has_part  = load_dataset("has_part", script_version="master")
    # mkqa  = load_dataset("mkqa", script_version="master")
    # nq_open  = load_dataset("nq_open", script_version="master")
    # winograd_wsc  = load_dataset("winograd_wsc",'wsc285', script_version="master")

    # wiki_bio  = load_dataset("wiki_bio", script_version="master")
    # t2s = train_10_20_ds.map(get_text_mapper(["title", "text"], 512, tokenizer, sent_detector), batched=True, remove_columns=["title"])
    # train_10_20_ds = datasets.load_dataset('wikipedia', '20200501.en', split='train[10:20]')

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


def get_text_mapper(text_cols, total_tokens, tokenizer, sent_detector):
    def mapper(examples: Dict[str, List], indices: List[int]=None) -> Dict[str, List]:
        # TODO: remove duplicates after merging for main dataset of MLM
        # TODO: add an id column after making MLM set
        texts = []

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

        for text in one_texts:
            if text is None or len(text) == 0:
                continue
            text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', text)
            sents = sent_detector.tokenize(text)
            cwc = 0
            cur_seg = ""
            for i, s in enumerate(sents):
                wc = len(tokenizer.tokenize(s, add_special_tokens=True))
                if cwc + wc <= total_tokens or cwc == 0 or i == len(sents) - 1:
                    pass
                else:
                    final_texts.append(cur_seg)
                    cwc = 0
                    cur_seg = ""
                cwc += wc
                cur_seg = cur_seg + " " + s
            # TODO: Last sent, big sentence
            final_texts.append(cur_seg)
        final_lengths = [len(tokenizer.tokenize(t, add_special_tokens=True)) for t in final_texts]
        assert len(final_lengths) == len(final_texts)
        return dict(text=final_texts, length=final_lengths)
    return mapper


def get_matching_mapper(text_cols, matching_query, matching_cols, mlm_query=tuple(), mlm_ans=tuple(),
                        total_tokens=768, tokenizer=None, mcq_query=tuple(), mcq_cols=tuple(), mcq_ans=tuple(),
                        text_only_answers=False, matching_query_with_adversarial=True, n_jumbled_options=1, n_within_text_options=1):
    asep = " [ANSWER_OPTION_SEP] "
    aoptbegin = "[ANSWER_OPTION_BEGIN] "
    aoptend = " [ANSWER_OPTION_END]"
    word_choice_1 = ["select", "what is", "choose"]
    word_choice_2 = ["appropriate", "correct", "right"]
    mask = " "+tokenizer.mask_token+" "

    def mapper(examples: Dict[str, List], indices: List[int]=None) -> Dict[str, List]:
        texts = []
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
        # select/(what is) the correct X from A,B,C,D ?
        # select/(what is) the correct option for X from 1. A, 2. B, 3. C, 4. D
        # Which X from A,B,C,D seems appropriate/correct
        # Which option from 1. A, 2. B, 3. C, 4. D seems appropriate/correct for X

        # Choose / select the right options from 1. A, 2. B, 3. C, 4. D for X?


        query = []
        answer = []
        for qtxt, qmc in zip(matching_query, matching_cols):
            if isinstance(qtxt, (list, tuple)):
                qtxt = random.sample(qtxt, 1)[0]
            cq_query = []
            cq_answer = []
            query_answers = list(map(clean_text, examples[qmc]))

            if matching_query_with_adversarial:
                qaws = [qa.split() for qa in query_answers]
                qaws = [" ".join(k) for qa in qaws for k in [random.sample(qa, len(qa)) for _ in range(n_jumbled_options)]]
                qaws = [k for k in qaws if k not in query_answers]
                query_answers = query_answers + qaws
                adversarial_same_text = [" ".join(t.split()[idx:idx+random.sample([3, 4, 5], 1)[0]]) for t in one_texts for idx in (random.sample(range(0, len(t.split())), n_within_text_options) if len(t.split()) > 1 else [])]
                query_answers = query_answers + adversarial_same_text

            query_answers = np.array(query_answers)

            shuffled_idxs = random.sample(range(len(query_answers)), len(query_answers))
            query_answers_shuffle = list(query_answers[shuffled_idxs])
            query_answers_shuffle_type_1 = aoptbegin + asep.join(query_answers_shuffle) + aoptend
            query_answers_shuffle_type_2 = aoptbegin + asep.join([str(i+1)+". " + a for i, a in enumerate(query_answers_shuffle)]) + aoptend
            for idx in range(len(one_texts)):
                aidx = shuffled_idxs.index(idx)
                atext = query_answers[idx]
                # assert query_answers[shuffled_idxs[aidx]] == atext == query_answers_shuffle[aidx]

                atext_len = len(atext.split())
                rnd = random.random()
                if (rnd < 0.25 and atext_len <= 4) or text_only_answers:
                    cq_query.append("%s the correct %s from %s?" % (random.sample(word_choice_1, 1)[0], qtxt, query_answers_shuffle_type_1))
                    cq_answer.append(atext)
                elif rnd < 0.5:
                    cq_query.append("%s the correct option for %s from %s?" % (random.sample(word_choice_1, 1)[0], qtxt, query_answers_shuffle_type_2))
                    cq_answer.append(str(aidx + 1))
                elif rnd < 0.75 and atext_len <= 4:
                    cq_query.append("Which %s from %s seems %s?" % (qtxt, query_answers_shuffle_type_1, random.sample(word_choice_2, 1)[0]))
                    cq_answer.append(atext)
                else:
                    cq_query.append("Which option from %s seems %s for %s?" % (query_answers_shuffle_type_2, random.sample(word_choice_2, 1)[0], qtxt))
                    cq_answer.append(str(aidx + 1))
            query.append(cq_query)
            answer.append(cq_answer)

        for qtxt, qmc, mcol in zip(mcq_query, mcq_cols, mcq_ans):
            if isinstance(qtxt, (list, tuple)):
                qtxt = random.sample(qtxt, 1)[0]
            cq_query = []
            cq_answer = []
            candidates = list(zip(*[examples[col] for col in qmc])) if isinstance(qmc, (list, tuple)) else examples[qmc]
            candidates = [d for c in candidates for d in list(map(clean_text, c))]
            len_per_sample = len(qmc) if isinstance(qmc, (list, tuple)) else len(examples[qmc][0])

            if matching_query_with_adversarial:
                qaws = [qa.split() for qa in candidates]
                qaws = list(set([" ".join(k) for qa in qaws for k in [random.sample(qa, len(qa)) for _ in range(n_jumbled_options)]]))
                qaws = [k for k in qaws if k not in candidates]
                candidates = candidates + qaws
                adversarial_same_text = [" ".join(t.split()[idx:idx+random.sample([3, 4, 5], 1)[0]]) for t in one_texts for idx in (random.sample(range(0, len(t.split())), n_within_text_options) if len(t.split()) > 1 else [])]
                candidates = candidates + adversarial_same_text
            candidates = np.array(candidates)

            shuffled_idxs = random.sample(range(len(candidates)), len(candidates))
            candidates_shuffle = list(candidates[shuffled_idxs])
            candidates_shuffle_type_1 = aoptbegin + asep.join(candidates_shuffle) + aoptend
            candidates_shuffle_type_2 = aoptbegin + asep.join([str(i + 1) + ". " + a for i, a in enumerate(candidates_shuffle)]) + aoptend
            if qtxt in examples:
                qt = [q + " " for q in examples[qtxt]]
                qtxt = "answer"
            else:
                qt = [""] * len(one_texts)
            for idx, aidx_c, qtx in zip(range(len(one_texts)), examples[mcol], qt):
                if aidx_c != -1 and isinstance(aidx_c, int):
                    aidx = shuffled_idxs.index(idx * len_per_sample + aidx_c)
                    atext = candidates[idx * len_per_sample + aidx_c]
                    assert candidates[shuffled_idxs[aidx]] == atext == candidates_shuffle[aidx]
                elif isinstance(aidx_c, str) and len(aidx_c) > 0:
                    atext = aidx_c
                    aidx = candidates_shuffle.index(atext)
                    assert candidates[shuffled_idxs[aidx]] == atext == candidates_shuffle[aidx]
                else:
                    aidx = atext = mask
                atext_len = len(atext.split())
                rnd = random.random()
                if (rnd < 0.25 and atext_len <= 3) or text_only_answers:
                    cq_query.append("%s%s the correct %s from %s?" % (qtx, random.sample(word_choice_1, 1)[0], qtxt, candidates_shuffle_type_1))
                    if text_only_answers and len(qtx)>0:
                        cq_query[-1] = qtx
                    cq_answer.append(atext)
                elif rnd < 0.5:
                    cq_query.append("%s%s the correct option for %s from %s?" % (qtx, random.sample(word_choice_1, 1)[0], qtxt, candidates_shuffle_type_2))
                    cq_answer.append(str(aidx + 1) if type(aidx)==int else aidx)
                elif rnd < 0.75 and atext_len <= 3:
                    cq_query.append("%sWhich %s from %s seems %s?" % (qtx, qtxt, candidates_shuffle_type_1, random.sample(word_choice_2, 1)[0]))
                    cq_answer.append(atext)
                else:
                    cq_query.append("%sWhich option from %s seems %s for %s?" % (qtx, candidates_shuffle_type_2, random.sample(word_choice_2, 1)[0], qtxt))
                    cq_answer.append(str(aidx + 1) if type(aidx)==int else aidx)
            query.append(cq_query)
            answer.append(cq_answer)

        for qtxt_mlm, qmc_mlm in zip(mlm_query, mlm_ans):
            qt = qtxt_mlm
            if isinstance(qtxt_mlm, (list, tuple)):
                qt = random.sample(qtxt_mlm, 1)[0]
            if qt in examples:
                cq_query = examples[qt]
            else:
                cq_query = [qt]*len(one_texts)
            cq_answer = list(map(clean_text, examples[qmc_mlm]))
            query.append(cq_query)
            answer.append(cq_answer)

        query = list(zip(*query))
        answer = list(zip(*answer))
        qlens = [len(tokenizer.tokenize(" ".join(t) + " ".join(a), add_special_tokens=True)) + 2*len(t) for t, a in zip(query, answer)]
        remaining_len = [int((total_tokens - ll) / 1.4) for ll in qlens]
        one_texts = [" ".join(t.split()[:r]) for t, r in zip(one_texts, remaining_len)]
        final_lengths = [len(tokenizer.tokenize(t, add_special_tokens=True)) + ql for t, ql in zip(one_texts, qlens)]
        return dict(text=one_texts, length=final_lengths, query=query, answer=answer)

    return mapper


def superglue_test(test_only=True, pet_dataset=True, dataset_location=os.path.join(os.path.expanduser("~"), "processed_datasets/superglue_test")):
    import datasets
    import re
    import numpy as np
    import random
    from typing import List, Dict
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    import nltk.data
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    import os
    os.cpu_count()
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    super_glue = dict()
    for gl in ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed', 'axb', 'axg']: # 'wsc',
        super_glue[gl] = load_dataset("super_glue", gl)

    if not os.path.exists(dataset_location):
        os.makedirs(dataset_location)
    if test_only and dataset_location is not None and os.path.exists(dataset_location) and os.path.exists(os.path.join(dataset_location, "dataset_dict.json")):
        sglue_proc = DatasetDict.load_from_disk(dataset_location)
        super_glue = DatasetDict({k: v['test'] for k, v in super_glue.items()})
        return super_glue, sglue_proc
    elif not pet_dataset:
        return super_glue, None

    sglue_proc = dict()

    sglue_proc["axg"] = super_glue["axg"].map(lambda x: dict(text="premise: " + x["premise"] + " hypothesis: " + x["hypothesis"]),
                                              remove_columns=["idx", "hypothesis", "premise"], num_proc=8)

    sglue_proc["axg"] = sglue_proc["axg"].map(
        get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree or contradict?"], ["label"], 1024, tokenizer),
        batched=True, num_proc=16, batch_size=8, remove_columns=["label"])

    sglue_proc["axb"] = super_glue["axb"].map(lambda x: dict(text="premise: " + x["sentence1"] + " hypothesis: " + x["sentence2"]),
                                              remove_columns=["idx", "sentence1", "sentence2"], num_proc=8)

    sglue_proc["axb"] = sglue_proc["axb"].map(
        get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree or contradict?"], ["label"], 1024, tokenizer),
        batched=True, num_proc=16, batch_size=8, remove_columns=["label"])

    sglue_proc['boolq'] = super_glue['boolq'].map(lambda x: dict(label={1: "yes", 0: "no", -1: tokenizer.mask_token}[x["label"]]))
    sglue_proc['boolq'] = sglue_proc['boolq'].map(get_matching_mapper(["passage"], [], [], ["question"], ["label"], 1024, tokenizer), batched=True, num_proc=16,
                                                  batch_size=4, remove_columns=['question', 'passage', 'idx', 'label'])

    sglue_proc['cb'] = super_glue['cb'].map(lambda x: dict(text="premise: " + x["premise"] + " hypothesis: " + x["hypothesis"],
                                                              label=({0: "agree", 1: "contradict", 2: "neutral", -1: tokenizer.mask_token}[x["label"]])),
                                               remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
    sglue_proc['cb'] = sglue_proc['cb'].map(
        get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer),
        batched=True, num_proc=16, batch_size=8, remove_columns=["label"])

    sglue_proc['copa'] = super_glue["copa"].map(lambda x: dict(question=x["question"] + ", ")).map(
        get_matching_mapper(["premise"], [], [], [], [], 1024, tokenizer, ['question'], [["choice1", "choice2"]], ["label"], n_jumbled_options=0,
                            n_within_text_options=2), batched=True, num_proc=16, batch_size=2,
        remove_columns=['premise', 'choice1', 'choice2', 'question', 'idx', 'label'])

    sglue_proc['multirc'] = super_glue["multirc"].map(lambda x: dict(label=({0: "no", 1: "yes", -1: tokenizer.mask_token}[x["label"]])))
    sglue_proc['multirc'] = sglue_proc['multirc'].map(get_matching_mapper(["paragraph", "question", "answer"], [], [], [
        ["Is the question answered correctly?", "Does the previous statement provide valid answer to the asked question?"]], ["label"], 1024, tokenizer),
                                                            batched=True, num_proc=16, batch_size=8,
                                                            remove_columns=['paragraph', 'question', 'answer', 'idx', 'label'])

    sglue_proc['record'] = super_glue['record'].map(lambda x: dict(query=x["query"] + " Find the right entity for @placeholder from the given options.",
                                                                      label=x['entities'].index(x['answers'][0]) if len(x['answers']) > 0 else -1))
    sglue_proc['record'] = sglue_proc['record'].map(
        get_matching_mapper(["passage"], [], [], [], [], 1024, tokenizer, ["query"], ['entities'], ["label"], n_jumbled_options=0, n_within_text_options=2),
        batched=True, num_proc=16, batch_size=1, remove_columns=['passage', 'query', 'entities', 'answers', 'idx', 'label'])

    sglue_proc['rte'] = super_glue['rte'].map(lambda x: dict(text="premise: " + x["premise"] + " hypothesis: " + x["hypothesis"],
                                                                label=({0: "agree", 1: "contradict", 2: "neutral", -1: tokenizer.mask_token}[x["label"]])),
                                                 remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
    sglue_proc['rte'] = sglue_proc['rte'].map(
        get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer),
        batched=True, num_proc=16, batch_size=8, remove_columns=["label"])

    sglue_proc["wic"] = super_glue["wic"].map(lambda x: dict(text="first sentence: " + x["sentence1"] + ", second sentence: " + x["sentence2"],
                                                                query="Does the word '%s' have same meaning in first sentence and second sentence" % x["word"],
                                                                label={0: "no", 1: "yes", -1: tokenizer.mask_token}[x["label"]]))

    sglue_proc["wic"] = sglue_proc["wic"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16,
                                                    batch_size=8,
                                                    remove_columns=['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2', 'idx', 'label'])

    sglue_proc["wsc.fixed"] = super_glue["wsc.fixed"].map(lambda x: dict(label={0: "no", 1: "yes", -1: tokenizer.mask_token}[x["label"]],
                                                                            query="Do [word1] %s and [word2] %s refer to the same entity" % (
                                                                            x["span1_text"], x["span2_text"]),
                                                                            text=" ".join([defaultdict(str, {x["span1_index"]: "[word1] ",
                                                                                                             x["span2_index"]: "[word2] "})[idx] + w for idx, w
                                                                                           in enumerate(x["text"].split())])))
    sglue_proc["wsc.fixed"] = sglue_proc["wsc.fixed"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True,
                                                                num_proc=16, batch_size=8,
                                                                remove_columns=['idx', 'label', 'query', 'span1_index', 'span1_text', 'span2_index',
                                                                                'span2_text'])

    if test_only:
        sglue_proc = DatasetDict({k: v['test'] for k, v in sglue_proc.items()})

    return super_glue, sglue_proc


# https://github.com/niderhoff/nlp-datasets
# https://amitness.com/toolbox/

"""
import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bookcorpusopen512 = bookcorpusopen.map(get_text_mapper(["title", "text"], 512, tokenizer, sent_detector), batched=True, remove_columns=["title"], num_proc=24)


kelm1024 = kelm.map(get_text_mapper(["sentence"], 1024, tokenizer, sent_detector), batched=True, remove_columns=['triple', 'sentence'], num_proc=32)
kelm1024.save_to_disk("/home/ahemf/processed_datasets/kelm1024")

openwebtext256 = openwebtext.map(get_text_mapper(["text"], 256, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
openwebtext256.save_to_disk("/home/ahemf/processed_datasets/openwebtext256")
openwebtext512 = openwebtext.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
openwebtext512.save_to_disk("/home/ahemf/processed_datasets/openwebtext512")
openwebtext1024 = openwebtext.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
openwebtext1024.save_to_disk("/home/ahemf/processed_datasets/openwebtext1024")

reddit = reddit.map(lambda x: dict(text=x["normalizedBody"]), remove_columns=["author", "body", "content", "normalizedBody", "subreddit", "subreddit_id", "summary", "id"], num_proc=48)
reddit128 = reddit.map(get_text_mapper(["text"], 128, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=48)
reddit128.save_to_disk("/home/ahemf/processed_datasets/reddit128")
reddit256 = reddit.map(get_text_mapper(["text"], 256, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=48)
reddit256.save_to_disk("/home/ahemf/processed_datasets/reddit256")
reddit512 = reddit.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=48)
reddit512.save_to_disk("/home/ahemf/processed_datasets/reddit512")
reddit1024 = reddit.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=48)
reddit1024.save_to_disk("/home/ahemf/processed_datasets/reddit1024")

medal = medal.map(lambda x: dict(text=x["text"]+(" ".join(x["label"]) if isinstance(x["label"], (list, tuple)) else x["label"])), remove_columns=["abstract_id", "location"], num_proc=48)
medal128 = medal.map(get_text_mapper(["text"], 128, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
medal128.save_to_disk("/home/ahemf/processed_datasets/medal128")
medal256 = medal.map(get_text_mapper(["text"], 256, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
medal256.save_to_disk("/home/ahemf/processed_datasets/medal256")
medal512 = medal.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
medal512.save_to_disk("/home/ahemf/processed_datasets/medal512")
medal1024 = medal.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
medal1024.save_to_disk("/home/ahemf/processed_datasets/medal1024")

big_patent = big_patent.map(lambda d: dict(text=d["abstract"]+d["description"]), remove_columns=["description", "abstract"], num_proc=32)
big_patent128 = big_patent.map(get_text_mapper(["text"], 128, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
big_patent128.save_to_disk("/home/ahemf/processed_datasets/big_patent128")
big_patent256 = big_patent.map(get_text_mapper(["text"], 256, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
big_patent256.save_to_disk("/home/ahemf/processed_datasets/big_patent256")
big_patent512 = big_patent.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
big_patent512.save_to_disk("/home/ahemf/processed_datasets/big_patent512")
big_patent1024 = big_patent.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
big_patent1024.save_to_disk("/home/ahemf/processed_datasets/big_patent1024")

yahoo_answers_topics = yahoo_answers_topics.map(lambda d: dict(text=d["question_title"]+d["question_content"]+d["best_answer"]), remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'], num_proc=32)
yahoo_answers_topics128 = yahoo_answers_topics.map(get_text_mapper(["text"], 128, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
yahoo_answers_topics128.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics128")
yahoo_answers_topics256 = yahoo_answers_topics.map(get_text_mapper(["text"], 256, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
yahoo_answers_topics256.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics256")
yahoo_answers_topics512 = yahoo_answers_topics.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
yahoo_answers_topics512.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics512")
yahoo_answers_topics1024 = yahoo_answers_topics.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
yahoo_answers_topics1024.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics1024")

yahoo_answers_qa = yahoo_answers_qa.map(lambda d: dict(text=d["question"]+d["answer"]+(" ".join(d["nbestanswers"]))), remove_columns=['id', 'question', 'answer', 'nbestanswers', 'main_category'], num_proc=32)
yahoo_answers_qa512 = yahoo_answers_qa.map(get_text_mapper(["text"], 512, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=32)
yahoo_answers_qa512.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_qa512")

amazon_polarity512 = amazon_polarity.map(get_text_mapper(["content"], 512, tokenizer, sent_detector), batched=True, remove_columns=['label', 'title', 'content'], num_proc=32)
amazon_polarity512.save_to_disk("/home/ahemf/processed_datasets/amazon_polarity512")

wikihow_all1024 = wikihow_all.map(get_text_mapper(["title", "headline", "text",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['headline', 'title'], num_proc=32)
wikihow_all1024.save_to_disk("/home/ahemf/processed_datasets/wikihow_all1024")

wikihow_sep1024 = wikihow_sep.map(get_text_mapper(["title", "overview", "headline", "text",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['text', 'headline', 'title', 'overview', 'sectionLabel'], num_proc=32)
wikihow_sep1024.save_to_disk("/home/ahemf/processed_datasets/wikihow_sep1024")


scientific_papers_pubmed512 = scientific_papers_pubmed.map(get_text_mapper(["abstract","article"], 512, tokenizer, sent_detector), batched=True, remove_columns=['article', 'abstract', 'section_names'], num_proc=32)
scientific_papers_pubmed512.save_to_disk("/home/ahemf/processed_datasets/scientific_papers_pubmed512")

scientific_papers_arxiv512 = scientific_papers_arxiv.map(get_text_mapper(["abstract","article"], 512, tokenizer, sent_detector), batched=True, remove_columns=['article', 'abstract', 'section_names'], num_proc=32)
scientific_papers_arxiv512.save_to_disk("/home/ahemf/processed_datasets/scientific_papers_arxiv512")

reuters512 = reuters.map(get_text_mapper(["title","text"], 512, tokenizer, sent_detector), batched=True, remove_columns=['topics', 'lewis_split', 'cgis_split', 'old_id', 'new_id', 'places', 'people', 'orgs', 'exchanges', 'date', 'title'], num_proc=32)
reuters512.save_to_disk("/home/ahemf/processed_datasets/reuters512")

amazon_us_reviews1024 = amazon_us_reviews.map(get_text_mapper(["product_title", "product_category", "review_body"], 1024, tokenizer, sent_detector), batched=True, remove_columns=['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date'], num_proc=32)
amazon_us_reviews1024.save_to_disk("/home/ahemf/processed_datasets/amazon_us_reviews1024")

ohsumed1024 = ohsumed.map(get_text_mapper(["title", "abstract",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['seq_id', 'medline_ui', 'mesh_terms', 'title', 'publication_type', 'abstract', 'author', 'source'], num_proc=32)
ohsumed1024.save_to_disk("/home/ahemf/processed_datasets/ohsumed1024")

xsum1024 = xsum.map(get_text_mapper(["document",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['document', 'summary', 'id'], num_proc=32)
xsum1024.save_to_disk("/home/ahemf/processed_datasets/xsum1024")

eli51024 = eli5.map(get_text_mapper(["title", 'selftext', ['answers', 'text']], 1024, tokenizer, sent_detector), batched=True, remove_columns=['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls'], num_proc=32)
eli51024.save_to_disk("/home/ahemf/processed_datasets/eli51024")

cnn_dailymail1024 = cnn_dailymail.map(get_text_mapper(["article",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['article', 'highlights', 'id'], num_proc=32)
cnn_dailymail1024.save_to_disk("/home/ahemf/processed_datasets/cnn_dailymail1024")

yelp_review_full1024 = yelp_review_full.map(get_text_mapper(["text",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['text', 'label'], num_proc=32)
yelp_review_full1024.save_to_disk("/home/ahemf/processed_datasets/yelp_review_full1024")

amazon_reviews_multi1024 = amazon_reviews_multi.map(get_text_mapper(["product_category","review_title", "review_body",], 1024, tokenizer, sent_detector), batched=True, remove_columns=['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'], num_proc=32)
amazon_reviews_multi1024.save_to_disk("/home/ahemf/processed_datasets/amazon_reviews_multi1024")

wmt14de_en1024 = wmt14de_en["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['translation'], num_proc=16)
wmt14de_en1024.save_to_disk("/home/ahemf/processed_datasets/wmt14de_en1024")

wmt15fr_en1024 = wmt15fr_en["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['translation'], num_proc=16)
wmt15fr_en1024.save_to_disk("/home/ahemf/processed_datasets/wmt15fr_en1024")

wmt16ru_en1024 = wmt16ru_en["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['translation'], num_proc=16)
wmt16ru_en1024.save_to_disk("/home/ahemf/processed_datasets/wmt16ru_en1024")

wmt17cs_en1024 = wmt17cs_en["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['translation'], num_proc=16)
wmt17cs_en1024.save_to_disk("/home/ahemf/processed_datasets/wmt17cs_en1024")


giga_fren1024 = giga_fren["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['id', 'translation'], num_proc=16)
giga_fren1024.save_to_disk("/home/ahemf/processed_datasets/giga_fren1024")

to_en1024 = to_en.map(get_text_mapper(["text"], 1024, tokenizer, sent_detector), batched=True, remove_columns=[], num_proc=16)
to_en1024.save_to_disk("/home/ahemf/processed_datasets/to_en1024")

un_pc1024 = un_pc["train"].map(get_text_mapper([["translation","en"],], 1024, tokenizer, sent_detector), batched=True, remove_columns=['translation'], num_proc=16)
un_pc1024.save_to_disk("/home/ahemf/processed_datasets/un_pc1024")
unpc = Dataset.load_from_disk("processed_datasets/un_pc1024")

amazon_us_reviews1024 = amazon_us_reviews.map(get_text_mapper(["product_category", "product_title", "review_headline", "review_body"], 1024, tokenizer, sent_detector), batched=True, remove_columns=['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date'], num_proc=24)
amazon_us_reviews1024.save_to_disk("/home/ahemf/processed_datasets/amazon_us_reviews1024")


rs.map(get_matching_mapper(["normalizedBody"], ["summary", "topic"], ["summary", "subreddit"], 512, tokenizer), batched=True, remove_columns=["author", "body", "content", "normalizedBody", "subreddit", "subreddit_id", "summary", "id"])[0]
######

reddit = load_dataset("reddit") 
reddit_qna = reddit.map(get_matching_mapper(["normalizedBody"], ["summary"], ["summary"],["topic category"], ["subreddit"], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, remove_columns=["author", "body", "content", "normalizedBody", "subreddit", "subreddit_id", "summary", "id"], num_proc=16, batch_size=2)
reddit_qna.save_to_disk("/home/ahemf/processed_datasets/reddit_qna")

bookcorpusopen = load_dataset("bookcorpusopen")
bookcorpusopen = bookcorpusopen.map(lambda x: dict(title=" ".join(x["title"].replace(".epub","").replace(".txt","").split('-')), text=x["text"][4096:]), num_proc=8)
bookcorpusopen_qna = bookcorpusopen.map(get_matching_mapper(["text"], ["title"], ["title",],[],[], 1024, tokenizer), batched=True, remove_columns=["title"], num_proc=16, batch_size=4)
bookcorpusopen_qna.save_to_disk("/home/ahemf/processed_datasets/bookcorpusopen_qna")

wikipedia = load_dataset("wikipedia", '20200501.en')
wikipedia_qna = wikipedia.map(get_matching_mapper(["text"], ["title"], ["title",],[], [], 1024, tokenizer), batched=True, remove_columns=["title"], num_proc=16, batch_size=4)
wikipedia_qna.save_to_disk("/home/ahemf/processed_datasets/wikipedia_qna")

amazon_polarity = load_dataset("amazon_polarity", script_version="master")
amazon_polarity = amazon_polarity.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
amazon_polarity_qna = amazon_polarity.map(get_matching_mapper(["content"], ["title", ], ["title", ],["sentiment expressed in review?"], ["sentiment"], 1024, tokenizer, n_jumbled_options=1, n_within_text_options=1), batched=True, remove_columns=["title", "sentiment", "content"], num_proc=16, batch_size=4)
amazon_polarity_qna.save_to_disk("/home/ahemf/processed_datasets/amazon_polarity_qna")

yahoo_answers_qa = load_dataset("yahoo_answers_qa")  # World knowledge testing rather than answer selection.
yahoo_answers_qa_qna = yahoo_answers_qa.map(get_matching_mapper(["answer"], ["question"], ["question"],["category"], ["main_category"], 1024, tokenizer, n_jumbled_options=1, n_within_text_options=2), batched=True, remove_columns=["id", 'question', 'answer', 'nbestanswers', 'main_category'], num_proc=16, batch_size=2)
yahoo_answers_qa_qna.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_qa_qna")

yahoo_answers_topics = load_dataset("yahoo_answers_topics")
yahoo_answers_topics_qna = yahoo_answers_topics.map(get_matching_mapper(["best_answer"], ["question"], ["question_title"],[],[], 1024, tokenizer), batched=True, remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'], num_proc=16, batch_size=4)
yahoo_answers_topics_qna.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics_qna")

reuters_hayes = load_dataset("reuters21578", 'ModHayes')
reuters_lewis = load_dataset("reuters21578", 'ModLewis')
reuters_apte = load_dataset("reuters21578", 'ModApte')
reuters = concatenate_datasets([d[split] for d in [reuters_hayes, reuters_lewis, reuters_apte] for split in ["train", "test"]])
reuters = reuters.map(lambda x: dict(title=x["title"].replace('&lt;', ' ').replace('>', ' ')), num_proc=16, batch_size=16)
reuters_qna = reuters.map(get_matching_mapper(["text"], ["title"], ["title"],[],[], 1024, tokenizer), batched=True, remove_columns=['topics', 'lewis_split', 'cgis_split', 'old_id', 'new_id', 'places', 'people', 'orgs', 'exchanges', 'date', 'title'], num_proc=16, batch_size=4)
reuters_qna.save_to_disk("/home/ahemf/processed_datasets/reuters_qna")

ohsumed = load_dataset("ohsumed", script_version="master")
ohsumed_qna = ohsumed.map(get_matching_mapper(["abstract"], ["title"], ["title"],[],[], 1024, tokenizer), batched=True, remove_columns=['seq_id', 'medline_ui', 'mesh_terms', 'title', 'publication_type', 'abstract', 'author', 'source'], num_proc=16, batch_size=4)
ohsumed_qna.save_to_disk("/home/ahemf/processed_datasets/ohsumed_qna")

xsum = load_dataset("xsum")
xsum_qna = xsum.map(get_matching_mapper(["document"], ["summary"], ["summary"],[],[], 1024, tokenizer), batched=True, remove_columns=['document', 'summary', 'id'], num_proc=16, batch_size=4)
xsum_qna.save_to_disk("/home/ahemf/processed_datasets/xsum_qna")

eli5 = load_dataset("eli5")
eli5_qna = eli5.map(get_matching_mapper([["answers", "text"]], ["title"], ["title"],[],[], 1024, tokenizer), batched=True, remove_columns=['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls'], num_proc=16, batch_size=4)
eli5_qna.save_to_disk("/home/ahemf/processed_datasets/eli5_qna")

cnn_dailymail = load_dataset("cnn_dailymail", '3.0.0')
cnn_dailymail_qna = cnn_dailymail.map(get_matching_mapper(["article"], ["highlights"], ["highlights"],[],[],  1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['article', 'highlights', 'id'])
cnn_dailymail_qna.save_to_disk("/home/ahemf/processed_datasets/cnn_dailymail_qna")

amazon_reviews_multi = load_dataset("amazon_reviews_multi", 'en')
amazon_reviews_multi_qna = amazon_reviews_multi.map(get_matching_mapper(["review_body"], ["title"], ["review_title"], [["Predict the review rating", "What is the rating suggested by the review on a scale of 1 to 5?"]], ["stars"], 1024, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'])
amazon_reviews_multi_qna.save_to_disk("/home/ahemf/processed_datasets/amazon_reviews_multi_qna")


wiki_lingua = load_dataset("wiki_lingua", 'english', script_version="master")
wiki_lingua = wiki_lingua.map(batch_process_wiki_lingua, batched=True, num_proc=1, remove_columns=["article"])
wiki_lingua_qna = wiki_lingua.map(get_matching_mapper(["document"], ["summary", "heading"], ["summary", "section_name"],["topic",], ["url",], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['document', 'section_name', 'summary', 'url'])
wiki_lingua_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_lingua_qna")

samsum = load_dataset("samsum", script_version="master")
samsum_qna = samsum.map(get_matching_mapper(["dialogue"], ["summary"], ["summary"], [], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['dialogue', 'id', 'summary',])
samsum_qna.save_to_disk("/home/ahemf/processed_datasets/samsum_qna")

multi_news = load_dataset("multi_news")
multi_news_qna = multi_news.map(get_matching_mapper(["document"], ["summary"], ["summary"],[], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['document', 'summary'])
multi_news_qna.save_to_disk("/home/ahemf/processed_datasets/multi_news_qna")

wiki_auto = load_dataset("wiki_auto", 'auto_acl')
wiki_auto = wiki_auto.map(lambda x: dict(simple_sentence=x["simple_sentence"].replace("-RRB-", ")").replace("-LRB-", "(").replace("-PIPE-", "|"), normal_sentence=x["normal_sentence"].replace("-RRB-", ")").replace("-LRB-", "(").replace("-PIPE-", "|")), num_proc=16)
wiki_auto = wiki_auto["full"]
wiki_auto_qna = wiki_auto.map(get_matching_mapper(["normal_sentence"], ["simplified sentence"], ["simple_sentence"],[], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['normal_sentence', 'simple_sentence'])
wiki_auto_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_auto_qna")

gigaword = load_dataset("gigaword")
gigaword = gigaword.map(lambda x: dict(document=x["document"].replace("-rrb-", ")").replace("-lrb-", "(").replace("-PIPE-", "|"), summary=x["summary"].replace("-rrb-", ")").replace("-lrb-", "(").replace("-PIPE-", "|")), num_proc=16)
gigaword_qna = gigaword.map(get_matching_mapper(["document"], ["summary"], ["summary"], [], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['document', 'summary',])
gigaword_qna.save_to_disk("/home/ahemf/processed_datasets/gigaword_qna")

wiki_atomic_edits_insertions = load_dataset("wiki_atomic_edits", 'english_insertions', script_version="master")
wiki_atomic_edits_insertions_qna = wiki_atomic_edits_insertions.map(get_matching_mapper(["base_sentence"], ["closest match"], ["edited_sentence"], ["edited phrase"], ["phrase"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['id', 'base_sentence', 'phrase', 'edited_sentence'])
wiki_atomic_edits_insertions_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_atomic_edits_insertions_qna")

wiki_atomic_edits_deletions = load_dataset("wiki_atomic_edits", 'english_deletions', script_version="master")
wiki_atomic_edits_deletions_qna = wiki_atomic_edits_deletions.map(get_matching_mapper(["base_sentence"], ["closest match], ["edited_sentence", ], ["deleted phrase"], ["phrase"], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['id', 'base_sentence', 'phrase', 'edited_sentence'])
wiki_atomic_edits_deletions_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_atomic_edits_deletions_qna")

wiki_split = load_dataset("wiki_split", script_version="master") 
wiki_split_qna = wiki_split.map(get_matching_mapper(["complex_sentence"], ["first simple sentence", "second simple sentence"], ["simple_sentence_1", "simple_sentence_2"], [], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['simple_sentence_2', 'simple_sentence_1', 'complex_sentence'])
wiki_split_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_split_qna")

per_sent = load_dataset("per_sent", script_version="master")
per_sent = per_sent.map(lambda x: dict(sentiment=({0: "negative", 1: "neutral", 2:"positive", -1:"neutral"}[x["TRUE_SENTIMENT"]])), remove_columns=["TRUE_SENTIMENT"], num_proc=8)
per_sent_qna = per_sent.map(get_matching_mapper(["DOCUMENT"], ["focused entity", "title"], ["TARGET_ENTITY", "TITLE"], ["Predict the correct sentiment from positive, neutral and negative"], ["sentiment"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['DOCUMENT', 'DOCUMENT_INDEX', 'MASKED_DOCUMENT', 'Paragraph0', 'Paragraph1', 'Paragraph10', 'Paragraph11', 'Paragraph12', 'Paragraph13', 'Paragraph14', 'Paragraph15', 'Paragraph2', 'Paragraph3', 'Paragraph4', 'Paragraph5', 'Paragraph6', 'Paragraph7', 'Paragraph8', 'Paragraph9', 'TARGET_ENTITY', 'TITLE', 'sentiment'])
per_sent_qna.save_to_disk("/home/ahemf/processed_datasets/per_sent_qna")


yelp_polarity = yelp_polarity.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
yelp_polarity_qna = yelp_polarity.map(get_matching_mapper(["text"], [], [], ["Predict the correct sentiment between positive and negative"], ["sentiment"], 512, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['sentiment'])
yelp_polarity_qna.save_to_disk("/home/ahemf/processed_datasets/yelp_polarity_qna")


app_reviews_qna = app_reviews.map(get_matching_mapper(["review"], [], [], [["Predict the review rating", "What is the rating suggested by the review on a scale of 1 to 5?"]], ["stars"], 768, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['package_name', 'review', 'date', 'star'])
app_reviews_qna.save_to_disk("/home/ahemf/processed_datasets/app_reviews_qna")

imdb = imdb.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
imdb_qna = imdb.map(get_matching_mapper(["text"], [], [], ["Predict the correct sentiment between positive and negative"], ["sentiment"], 512, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['sentiment'])
imdb_qna.save_to_disk("/home/ahemf/processed_datasets/imdb_qna")


anli = load_dataset("anli")
anli = anli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "contradict", 1: "neutral", 2:"agree", -1:"neutral"}[x["label"]])), remove_columns=["hypothesis", "premise", "reason", "uid"], num_proc=8)
anli_qna_v1 = anli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
anli_qna_v1.save_to_disk("/home/ahemf/processed_datasets/anli_qna_v1")

anli = load_dataset("anli")
anli = anli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "disagree", 1: "unrelated", 2:"entail", -1:"unrelated"}[x["label"]])), remove_columns=["hypothesis", "premise", "reason", "uid"], num_proc=8)
anli_qna_v2 = anli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
anli_qna_v2.save_to_disk("/home/ahemf/processed_datasets/anli_qna_v2")

# snli, mnli

# entail, contradict, neither
snli = load_dataset("snli")
snli = snli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "agree", 1: "contradict", 2:"neutral", -1:"neutral"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
snli_qna_v1 = snli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
snli_qna_v1.save_to_disk("/home/ahemf/processed_datasets/snli_qna_v1")

snli = load_dataset("snli")
snli = snli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "entail", 1: "disagree", 2:"unrelated", -1:"unrelated"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
snli_qna_v2 = snli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
snli_qna_v2.save_to_disk("/home/ahemf/processed_datasets/snli_qna_v2")

mnli = load_dataset("multi_nli")
mnli = mnli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "agree", 1: "contradict", 2:"neutral", -1:"neutral"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
mnli_qna_v1 = mnli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
mnli_qna_v1.save_to_disk("/home/ahemf/processed_datasets/mnli_qna_v1")

mnli = load_dataset("multi_nli")
mnli = mnli.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "entail", 1: "disagree", 2:"unrelated", -1:"unrelated"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
mnli_qna_v2 = mnli.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
mnli_qna_v2.save_to_disk("/home/ahemf/processed_datasets/mnli_qna_v2")

hans = load_dataset("hans")
hans = hans.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "agree", 1: "contradict", 2:"neutral", -1:"neutral"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
hans_qna_v1 = hans.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
hans_qna_v1.save_to_disk("/home/ahemf/processed_datasets/hans_qna_v1")

hans = load_dataset("hans")
hans = hans.map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "entail", 1: "disagree", 2:"unrelated", -1:"unrelated"}[x["label"]])), remove_columns=["hypothesis", "premise"], num_proc=8)
hans_qna_v2 = hans.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
hans_qna_v2.save_to_disk("/home/ahemf/processed_datasets/hans_qna_v2")

# 
scitail = load_dataset("scitail", 'snli_format')
scitail = scitail.map(lambda x: dict(text="premise: "+ x["sentence1"]+ " hypothesis: " + x["sentence2"], label=x["gold_label"]), remove_columns=['sentence1_binary_parse', 'sentence1_parse', 'sentence1', 'sentence2_parse', 'sentence2', 'annotator_labels', 'gold_label'], num_proc=8)
scitail_qna = scitail.map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
scitail_qna.save_to_disk("/home/ahemf/processed_datasets/scitail_qna")

#
go_emotions = load_dataset("go_emotions", 'raw', script_version="master")
go_emotions = go_emotions.filter(lambda x: len(x["text"].split())>16)
go_emotions = go_emotions.map(lambda x: dict(label=[k for k, v in x.items() if v == 1]), remove_columns=['id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear', 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'])
go_emotions = go_emotions.filter(lambda x: len(x['label'])>0)
go_emotions = go_emotions.map(lambda x: dict(label=" ".join(sorted(x["label"]))))
go_emotions_qna = go_emotions.map(get_matching_mapper(["text"], [], [], ["Select the right set of emotions from admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral?"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
go_emotions_qna.save_to_disk("/home/ahemf/processed_datasets/go_emotions_qna")

#
dm = pd.read_csv("https://raw.githubusercontent.com/synapse-developpement/Discovery/master/data/markers_list.txt", header=None, names=["discourse marker"])
dm = list(map(lambda t: t.replace("_", " ").replace("[no-conn]", "no connection"),dm["discourse marker"].values))
discovery = load_dataset("discovery", 'discovery', script_version="master").filter(lambda x: len(x["sentence1"].split()) >= 16 and len(x["sentence2"].split()) >= 16)
discovery = discovery.map(lambda x: dict(label=dm[x["label"]]), remove_columns=["idx"], num_proc=8)
discovery = discovery.filter(lambda x: len(x['label'])>0)
discovery_qna = discovery.map(get_matching_mapper(["sentence1","sentence2"], [], [], ["Predict the right connective word or phrase for the two preceding sentences."], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label", "sentence1", "sentence2"])
discovery_qna.save_to_disk("/home/ahemf/processed_datasets/discovery_qna")

#
paws = load_dataset("paws", 'labeled_final', script_version="master")
paws = paws.map(lambda x: dict(label="yes" if x["label"] == 1 else "no", text="first statement: "+x["sentence1"]+" second statement: "+x["sentence2"]), remove_columns=["id", "sentence2", "sentence1"])
paws_qna = paws.map(get_matching_mapper(["text"], [], [], ["Do the two given statements mean the same or paraphrase each other?"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
paws_qna.save_to_disk("/home/ahemf/processed_datasets/paws_qna")


#
swag = load_dataset("swag", 'regular')
swag_qna = swag.map(get_matching_mapper(["startphrase"], [], [], [], [], 1024, tokenizer, [["completion phrase", "end phrase", "end"]], [['ending0', 'ending1', 'ending2', 'ending3']], ["label"]), batched=True, num_proc=16, batch_size=2, remove_columns=['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'label'])
swag_qna.save_to_disk("/home/ahemf/processed_datasets/swag_qna")

hellaswag = load_dataset("hellaswag")
hellaswag = hellaswag.map(lambda x: dict(text="activity: "+x["activity_label"]+ ", context: " + x["ctx"], label=int(x["label"]) if len(x["label"])>0 else -1), remove_columns=['ind', 'activity_label', 'ctx_a', 'ctx_b','ctx','source_id', 'split', 'split_type',], num_proc=8)
hellaswag_qna = hellaswag.map(get_matching_mapper(["text"], [], [], [], [], 1024, tokenizer, [["completion phrase", "end phrase", "end", "ending"]], ["endings"], ["label"]), batched=True, num_proc=16, batch_size=2, remove_columns=['endings', 'label'])
hellaswag_qna.save_to_disk("/home/ahemf/processed_datasets/hellaswag_qna")

hotpot_qa = load_dataset("hotpot_qa", 'distractor')
hotpot_qa = hotpot_qa.map(lambda x: dict(context=" ".join([s for p in x["context"]["sentences"] for s in p])), num_proc=8)
hotpot_qa = hotpot_qa.map(lambda x: dict(text = "question: " + x["question"] + ", context: " + x["context"]),num_proc=8, remove_columns=['context', 'id', 'level', 'question', 'supporting_facts', 'type'])
hotpot_qa = hotpot_qa.map(lambda x: dict(length=len(x["text"].split())), num_proc=8).filter(lambda x: x["length"] <= 768, num_proc=8)
hotpot_qa_qna_v1 = hotpot_qa.map(get_matching_mapper(["text"], [], [], [["What is the question's answer?", "Provide an answer to the asked question?", "Answer the question"]], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["answer"])
hotpot_qa_qna_v1 = hotpot_qa_qna_v1.filter(lambda x: x["length"] <= 1000)
hotpot_qa_qna_v1.save_to_disk("/home/ahemf/processed_datasets/hotpot_qa_qna_v1")

hotpot_qa = load_dataset("hotpot_qa", 'distractor')
hotpot_qa = hotpot_qa.map(lambda x: dict(context=" ".join([s for p in x["context"]["sentences"] for s in p])), num_proc=8)
hotpot_qa = hotpot_qa.map(lambda x: dict(text =x["context"]),num_proc=8, remove_columns=['context', 'id', 'level', 'supporting_facts', 'type'])
hotpot_qa = hotpot_qa.map(lambda x: dict(length=len(x["text"].split())), num_proc=8).filter(lambda x: x["length"] <= 1000, num_proc=8)
hotpot_qa_qna_v2 = hotpot_qa.map(get_matching_mapper(["text"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["question", "answer"])
hotpot_qa_qna_v2 = hotpot_qa_qna_v2.filter(lambda x: x["length"] <= 1000)
hotpot_qa_qna_v2.save_to_disk("/home/ahemf/processed_datasets/hotpot_qa_qna_v2")

#
qangaroo = load_dataset("qangaroo", 'wikihop')
qangaroo = qangaroo.map(lambda x: dict(text="query: "+x["query"].replace("_", ' ')+", context: "+" ".join(x["supports"])), remove_columns=["id", "query", "supports"])
qangaroo = qangaroo.filter(lambda x: x["answer"] in x["candidates"]).filter(lambda x: "!" not in x["answer"])
qangaroo_qna = qangaroo.map(get_matching_mapper(["text"], [], [], [], [], 1024, tokenizer, [["answer to query", "answer to question"]], ["candidates"], ["answer"]), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'candidates'])
qangaroo_qna.save_to_disk("/home/ahemf/processed_datasets/qangaroo_qna")

qangaroo = load_dataset("qangaroo", 'wikihop')
qangaroo = qangaroo.map(lambda x: dict(text=" ".join(x["supports"]), query=x["query"].replace("_", ' ')), remove_columns=["id", "supports"])
qangaroo = qangaroo.filter(lambda x: x["answer"] in x["candidates"]).filter(lambda x: "!" not in x["answer"])
qangaroo_qna_v2 = qangaroo.map(get_matching_mapper(["text"], [], [], ["query"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'candidates'])
qangaroo_qna_v2.save_to_disk("/home/ahemf/processed_datasets/qangaroo_qna_v2")

#
qasc = load_dataset("qasc")
qasc = qasc.map(lambda x: dict(options=x["choices"]["text"], answerKey=(ord(x["answerKey"]) - ord('A') if len(x["answerKey"]) else -1)), remove_columns=["choices", "id"])
qasc = qasc.map(lambda x: dict(text="first fact: "+x["fact1"] + ", second fact: "+x["fact2"], answer=x["options"][x["answerKey"]]), remove_columns=['fact1', 'fact2', 'combinedfact'])

qasc_qa_v1 = qasc.map(get_matching_mapper(["text", "formatted_question"], [], [], ["Answer the question with correct choice."], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'answerKey', 'formatted_question', 'options', 'question'])
qasc_qa_v1.save_to_disk("/home/ahemf/processed_datasets/qasc_qa_v1")

qasc_qa_v2 = qasc.map(get_matching_mapper(["text", "question"], [], [], [], [], 1024, tokenizer, [["answer to query", "answer to question"]], ["options"], ["answerKey"]), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'answerKey', 'formatted_question', 'options', 'question'])
qasc_qa_v2.save_to_disk("/home/ahemf/processed_datasets/qasc_qa_v2")

qasc_qa_v3 = qasc.map(get_matching_mapper(["text",], [], [], ["formatted_question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'answerKey', 'formatted_question', 'options', 'question'])
qasc_qa_v3.save_to_disk("/home/ahemf/processed_datasets/qasc_qa_v3")

qasc_qa_v4 = qasc.map(get_matching_mapper(["text",], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'answerKey', 'formatted_question', 'options', 'question'])
qasc_qa_v4.save_to_disk("/home/ahemf/processed_datasets/qasc_qa_v4")

#
squad_v2 = squad_v2.map(lambda x: dict(text = x["title"]+". "+x["context"], answer=x["answers"]["text"]), remove_columns=["id", "title", "context", "answers"],)
squad_v2 = squad_v2.map(lambda x: dict(answer=x["answer"][0] if len(x["answer"]) > 0 else "no answer"))
squad_v2_qna = squad_v2.map(get_matching_mapper(["text", "question"], [], [], ["Answer the question."], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'question'])
squad_v2_qna.save_to_disk("/home/ahemf/processed_datasets/squad_v2_qna")

#

squad_v2 = squad_v2.map(lambda x: dict(text = x["title"]+". "+x["context"], answer=x["answers"]["text"]), remove_columns=["id", "title", "context", "answers"],)
squad_v2 = squad_v2.map(lambda x: dict(answer=x["answer"][0] if len(x["answer"]) > 0 else "no answer"))
squad_v2_qna_v2 = squad_v2.map(get_matching_mapper(["text", ], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'question'])
squad_v2_qna_v2.save_to_disk("/home/ahemf/processed_datasets/squad_v2_qna_v2")


ropes = ropes.map(lambda x: dict(answer=x["answers"]["text"]), remove_columns=["answers"]).map(lambda x: dict(answer=x["answer"][0] if len(x["answer"]) > 0 else "no answer"))
ropes_qna = ropes.map(get_matching_mapper(["background", "situation", "question"], [], [], ["Answer the question."], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'background', 'id', 'question', 'situation'])
ropes_qna.save_to_disk("/home/ahemf/processed_datasets/ropes_qna")

ropes = ropes.map(lambda x: dict(answer=x["answers"]["text"]), remove_columns=["answers"]).map(lambda x: dict(answer=x["answer"][0] if len(x["answer"]) > 0 else "no answer"))
ropes_qna_v2 = ropes.map(get_matching_mapper(["background", "situation"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'background', 'id', 'question', 'situation'])
ropes_qna_v2.save_to_disk("/home/ahemf/processed_datasets/ropes_qna_v2")


wiki_qa = wiki_qa.map(lambda x: dict(label="yes" if x["label"]==1 else "no"))
wiki_qa_qna = wiki_qa.map(get_matching_mapper(["document_title", "question", "answer"], [], [], [["Is the question answered correctly?", "Does the previous statement provide valid answer to the asked question?"]], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['question_id', 'question', 'document_title', 'answer', 'label'])
wiki_qa_qna.save_to_disk("/home/ahemf/processed_datasets/wiki_qa_qna")


#
narrativeqa = load_dataset("narrativeqa", script_version="master")
narrativeqa = narrativeqa.map(lambda x: dict(text=x["document"]["summary"]["text"], answers=[a["text"] for a in x["answers"]], question=x["question"]["text"]), remove_columns=["document"])
narrativeqa_v1 = narrativeqa.map(lambda x: dict(answer=x["answers"][0]), remove_columns=["answers"])
narrativeqa_v2 = narrativeqa.filter(lambda x: len(x["answers"])>1).map(lambda x: dict(answer=x["answers"][1]), remove_columns=["answers"])

narrativeqa_v1 = narrativeqa_v1.map(get_matching_mapper(["text"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'question',])
narrativeqa_v2 = narrativeqa_v2.map(get_matching_mapper(["text"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'question',])
narrativeqa_v1.save_to_disk("/home/ahemf/processed_datasets/narrativeqa_v1")
narrativeqa_v2.save_to_disk("/home/ahemf/processed_datasets/narrativeqa_v2")

#
social_i_qa = load_dataset("social_i_qa", script_version="master")
social_i_qa_v1 = social_i_qa.map(lambda x: dict(text=x["context"] + x["question"], label=int(x["label"]) - 1), remove_columns=["context", "question"])
social_i_qa_v2 = social_i_qa.map(lambda x: dict(text=x["context"], label=int(x["label"]) - 1), remove_columns=["context"])
social_i_qa_v1 = social_i_qa_v1.map(get_matching_mapper(["text"], [], [], [], [], 1024, tokenizer, [["answer to query", "answer to question"]], [["answerA", "answerB", "answerC"]], ["label"]), batched=True, num_proc=16, batch_size=2, remove_columns=['answerA', 'answerB', 'answerC', 'label'])
social_i_qa_v2 = social_i_qa_v2.map(get_matching_mapper(["text", ], [], [], [], [], 1024, tokenizer, ["question"], [["answerA", "answerB", "answerC"]], ["label"]), batched=True, num_proc=16, batch_size=2, remove_columns=['question', 'answerA', 'answerB', 'answerC', 'label'])
social_i_qa_v3 = social_i_qa.map(lambda x: dict(text=x["context"], label=[x['answerA'], x['answerB'], x['answerC']][int(x["label"]) - 1]), remove_columns=["context"])
social_i_qa_v3 = social_i_qa_v3.map(get_matching_mapper(["text", ], [], [], ["question"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['question', 'answerA', 'answerB', 'answerC', 'label'])
social_i_qa_v1.save_to_disk("/home/ahemf/processed_datasets/social_i_qa_v1")
social_i_qa_v2.save_to_disk("/home/ahemf/processed_datasets/social_i_qa_v2")
social_i_qa_v3.save_to_disk("/home/ahemf/processed_datasets/social_i_qa_v3")

#
quac = load_dataset("quac", script_version="master")
quac = quac.map(lambda d: dict(text=[d["background"][0]+d["context"][0]]*len(d["questions"][0]), question=d["questions"][0], options=[d["orig_answers"][0]["texts"]]*len(d["questions"][0]), label=list(range(len(d["questions"][0])))), batch_size=1, batched=True, remove_columns=['dialogue_id', 'wikipedia_page_title', 'background', 'section_title', 'context', 'turn_ids', 'questions', 'followups', 'yesnos', 'answers', 'orig_answers'])
quac = quac.map(lambda x: dict(options=["no answer" if opt=="CANNOTANSWER" else opt for opt in x["options"]]))
quac_qna = quac.map(get_matching_mapper(["text", ], [], [], [], [], 1024, tokenizer, ["question"], ["options"], ["label"], text_only_answers=True), batched=True, num_proc=16, batch_size=1, remove_columns=['question', "options", 'label'])
quac_qna.save_to_disk("/home/ahemf/processed_datasets/quac_qna")

#
e2e_nlg_cleaned = load_dataset("e2e_nlg_cleaned", script_version="master")
e2e_nlg_cleaned = e2e_nlg_cleaned.map(lambda x:dict(meaning_representation=x["meaning_representation"].split(", ")), )
e2e_nlg_cleaned = e2e_nlg_cleaned.map(lambda x:dict(question=[q.split("[")[0] for q in x["meaning_representation"]], answer=[q.split("[")[-1].replace("]",'') for q in x["meaning_representation"]]), remove_columns=["meaning_representation"])
e2e_nlg_cleaned = e2e_nlg_cleaned.map(lambda x:dict(question=[re.sub("([a-z])([A-Z])","\g<1> \g<2>", q) for q in x["question"]], answer=[re.sub(r'([A-Z][a-z]+(?=[A-Z]))', r'\1 ', q) for q in x["answer"]]))
e2e_nlg_cleaned = e2e_nlg_cleaned.map(lambda d: dict(text=[d["human_reference"][0]]*len(d["question"][0]), question=d["question"][0], options=[d["answer"][0]]*len(d["question"][0]), label=list(range(len(d["question"][0])))), batch_size=1, batched=True, remove_columns=['answer', 'human_reference'])
e2e_nlg_cleaned_qna = e2e_nlg_cleaned.map(get_matching_mapper(["text", ], [], [], [], [], 1024, tokenizer, ["question"], ["options"], ["label"], text_only_answers=False), batched=True, num_proc=16, batch_size=2, remove_columns=['question', "options", 'label'])
e2e_nlg_cleaned_qna.save_to_disk("/home/ahemf/processed_datasets/e2e_nlg_cleaned_qna")

# MLM instead of topic modelling for reddit n wikipedia etc with adversarial 
# by jumbling the match, picking few options from same text (same len as answer), insert stop words within the actual match

sent_comp = load_dataset("sent_comp", script_version="master")
sent_comp = sent_comp.map(lambda x: dict(text=x["graph"]["sentence"], shortened=x["compression"]["text"]),num_proc=16, remove_columns=['graph', 'compression', 'compression_ratio', 'doc_id', 'source_tree', 'compression_untransformed'])
sent_comp_qna = sent_comp.map(get_matching_mapper(["text"], ["shortened text", "headline"], ["shortened", "headline"], [], [], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=["shortened", "headline"])
sent_comp_qna.save_to_disk("/home/ahemf/processed_datasets/sent_comp_qna")

# 
quartz = load_dataset("quartz")
quartz = quartz.map(lambda x: dict(options=x["choices"]["text"], text=x["para"], label=ord(x["answerKey"]) - ord('A'),physical_effect_direction=x["para_anno"]["effect_dir_str"], physical_effect=x["para_anno"]["effect_prop"],), remove_columns=['id', 'choices', 'answerKey', 'para', 'para_id', 'para_anno', 'question_anno'],)
quartz_qna = quartz.map(get_matching_mapper(["text"], [], [], ["Physical phenomena", "Physical phenomena Direction"], ["physical_effect", "physical_effect_direction"], 1024, tokenizer, ["question"], ["options"], ["label"]), batched=True, num_proc=16, batch_size=2, remove_columns=['label', 'options', 'physical_effect', 'physical_effect_direction', 'question'])
quartz_qna.save_to_disk("/home/ahemf/processed_datasets/quartz_qna")

# Given an answer+text, generate a plausible question 
mocha = load_dataset("mocha", script_version="master").map(lambda x: dict(label=0))
mocha_qna = mocha.map(get_matching_mapper(["context"], [], [], [], [], 1024, tokenizer, ["question"], [["reference", "candidate"]], ["label"]), batched=True, num_proc=16, batch_size=4, remove_columns=['constituent_dataset', 'id', 'context', 'question', 'reference', 'candidate', 'score', 'metadata', 'candidate2', 'score2'])
mocha_qna.save_to_disk("/home/ahemf/processed_datasets/mocha_qna")

#
quoref = load_dataset("quoref").map(lambda x: dict(answer=x["answers"]["text"][0], text=x["context"]), remove_columns=['id', 'context', 'title', 'url', 'answers'])
quoref_qna = quoref.map(get_matching_mapper(["text"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['answer', 'question'])
quoref_qna.save_to_disk("/home/ahemf/processed_datasets/quoref_qna")

#
race = load_dataset("race", 'all').map(lambda x: dict(answer=ord(x["answer"])-ord('A')))
race_qna = race.map(get_matching_mapper(["article"], [], [], [], [], 1024, tokenizer, ["question"], ["options"], ["answer"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'article', 'example_id', 'options', 'question'])
race_qna.filter(lambda x: x["length"]>1024)
race_qna.save_to_disk("/home/ahemf/processed_datasets/race_qna")
#
winogrande = load_dataset("winogrande", 'winogrande_xl').map(lambda x: dict(answer=(ord(x["answer"]) - ord('1')) if len(x["answer"]) else -1, question="Fill in the blank."))
winogrande_qna = winogrande.map(get_matching_mapper(["sentence"], [], [], [], [], 1024, tokenizer, ["question"], [["option1", "option2"]], ["answer"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'option1', 'option2', 'question', 'sentence'])
winogrande_qna.save_to_disk("/home/ahemf/processed_datasets/winogrande_qna")

#
qed = load_dataset("qed", script_version="master").map(lambda x: dict(label=0, answer=x["original_nq_answers"][0]["string"], selected_sentence=x["annotation"]["selected_sentence"]["string"]), remove_columns=['example_id', 'title_text', 'url', 'sentence_starts', 'original_nq_answers', 'annotation'])
qed_qna_v1 = qed.map(get_matching_mapper(["paragraph_text"], [], [], [], [], 1024, tokenizer, ["question"], [["selected_sentence"]], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'paragraph_text', 'question', 'selected_sentence'])
qed_qna_v1.save_to_disk("/home/ahemf/processed_datasets/qed_qna_v1")

qed_qna_v2 = qed.map(get_matching_mapper(["paragraph_text"], [], [], ["question"], ["answer"], 1024, tokenizer,), batched=True, num_proc=16, batch_size=2, remove_columns=['answer', 'paragraph_text', 'question', 'selected_sentence'])
qed_qna_v2.save_to_disk("/home/ahemf/processed_datasets/qed_qna_v2")

commonsense_qa = load_dataset("commonsense_qa").map(lambda x: dict(text=x['question'], options=x["choices"]["text"], label=(ord(x["answerKey"]) - ord('A')) if len(x["answerKey"]) else -1), remove_columns=["answerKey", "choices"])
commonsense_qa = commonsense_qa.map(get_matching_mapper(["text"], [], [], [], [], 1024, tokenizer, ["answer"], ['options'], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['label', 'options', 'question'])
commonsense_qa.save_to_disk("/home/ahemf/processed_datasets/commonsense_qa")

cosmos_qa = load_dataset("cosmos_qa")
cosmos_qa = cosmos_qa.map(get_matching_mapper(["context"], [], [], [], [], 1024, tokenizer, ["question"], [['answer0', 'answer1', 'answer2', 'answer3']], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['id', 'context', 'question', 'answer0', 'answer1', 'answer2', 'answer3', 'label'])
cosmos_qa.save_to_disk("/home/ahemf/processed_datasets/cosmos_qa")

mrqa = load_dataset("mrqa", script_version="master").map(lambda x: dict(answers=x["answers"][0]), remove_columns=['subset', 'context_tokens', 'qid', 'question_tokens', 'detected_answers'],)
mrqa_v1 = mrqa.map(lambda x: dict(label=0)).map(get_matching_mapper(["context"], [], [], [], [], 1024, tokenizer, ["question"], [['answers']], ["label"], n_jumbled_options=0, n_within_text_options=1), batched=True, num_proc=16, batch_size=2, remove_columns=['answers', 'context', 'question', 'label'])
mrqa_v1.save_to_disk("/home/ahemf/processed_datasets/mrqa_v1")

mrqa_v2 = mrqa.map(lambda x: dict(label=0)).map(get_matching_mapper(["context"], [], [], ["question"], ["answers"], 1024, tokenizer, [], [], [], n_jumbled_options=0, n_within_text_options=1), batched=True, num_proc=16, batch_size=2, remove_columns=['answers', 'context', 'question', 'label'])
mrqa_v2.save_to_disk("/home/ahemf/processed_datasets/mrqa_v2")


# 
natural_questions = load_dataset("natural_questions")
natural_questions = natural_questions.map(lambda x: dict(question=x['question']['text'], answer=" ".join([t for a in x['annotations']['short_answers'][:1] for t in a['text'][:1]]), context=" ".join(x['document']['tokens']['token'])), remove_columns=['id', 'document','annotations'], num_proc=16)
natural_questions_qna = natural_questions.map(get_matching_mapper(["context"], [], [], ["question"], ["answer"], 1024, tokenizer), batched=True, num_proc=16, batch_size=256, remove_columns=['answer', 'context', 'question'])
natural_questions_qna.save_to_disk("/home/ahemf/processed_datasets/natural_questions_qna")

#
piqa = load_dataset("piqa")
piqa = piqa.map(get_matching_mapper(["goal"], [], [], [], [], 1024, tokenizer, [["action", "procedure", "solution"]], [['sol1', 'sol2']], ["label"], n_jumbled_options=0, n_within_text_options=1), batched=True, num_proc=16, batch_size=2, remove_columns=['goal', 'sol1', 'sol2', 'label'])
piqa.save_to_disk("/home/ahemf/processed_datasets/piqa")

#
pubmed_qa = load_dataset("pubmed_qa", 'pqa_labeled', script_version="master")
pubmed_qa = pubmed_qa.map(get_matching_mapper(["context"], [], [], ["question"], ["final_decision"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['pubid', 'question', 'context', 'long_answer', 'final_decision'])
pubmed_qa.save_to_disk("/home/ahemf/processed_datasets/pubmed_qa")

#
quora = load_dataset("quora").map(lambda x:dict(question1=x["questions"]["text"][0], question2=x["questions"]["text"][1], label = "yes" if x["is_duplicate"] else "no"), remove_columns=['questions', 'is_duplicate'])
quora = quora.map(get_matching_mapper(["question1", "question2"], [], [], ["Do the two questions mean the same?", "Are the queestions asking the same thing", "Are they same?"], ["label"], 1024, tokenizer,), batched=True, num_proc=16, batch_size=2, remove_columns=['question1', 'question2', 'label'])
quora.save_to_disk("/home/ahemf/processed_datasets/quora")

#
sciq = load_dataset("sciq").map(lambda x: dict(label=0))
sciq = sciq.map(get_matching_mapper(["support"], [], [], [], [], 1024, tokenizer, ["question"], [['correct_answer', 'distractor1', 'distractor2', 'distractor3']], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['question', 'distractor3', 'distractor1', 'distractor2', 'correct_answer', 'support', 'label'])
sciq.save_to_disk("/home/ahemf/processed_datasets/sciq")

#
peer_read_reviews = load_dataset("peer_read", 'reviews', script_version="master")
peer_read_reviews_qna = peer_read_reviews.map(get_matching_mapper(["abstract"], ["title"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=1, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['id', 'conference', 'comments', 'subjects', 'version', 'date_of_submission', 'title', 'authors', 'accepted', 'abstract', 'histories', 'reviews'])
peer_read_reviews_qna.save_to_disk("/home/ahemf/processed_datasets/peer_read_reviews_qna")

#
medical_questions_pairs = load_dataset("medical_questions_pairs", script_version="master").map(lambda x: dict(label="yes" if x["label"] else "no", question_1="first question: "+x["question_1"], question_2=", second question: " + x["question_2"]))
medical_questions_pairs_qna = medical_questions_pairs.map(get_matching_mapper(["question_1", "question_2"], [], [], ["Do the two statements mean the same?", "Are the queestions asking the same thing", "Are they same?"], ["label"], 1024, tokenizer,), batched=True, num_proc=16, batch_size=2, remove_columns=['dr_id', 'question_1', 'question_2', 'label'])
medical_questions_pairs_qna.save_to_disk("/home/ahemf/processed_datasets/medical_questions_pairs_qna")

#
empathetic_dialogues = load_dataset("empathetic_dialogues")
empathetic_dialogues_qna = empathetic_dialogues.map(get_matching_mapper(["utterance"], [], [], ["emotion"], ["context"], 1024, tokenizer,), batched=True, num_proc=16, batch_size=2, remove_columns=['conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance', 'selfeval', 'tags'])
empathetic_dialogues_qna.save_to_disk("/home/ahemf/processed_datasets/empathetic_dialogues_qna")

#
ai2_arc = load_dataset("ai2_arc", 'ARC-Challenge').map(lambda x: dict(answerKey=ord(x["answerKey"]) - ord(x["choices"]["label"][0]), choices=x["choices"]['text']))
ai2_arc_qna = ai2_arc.map(get_matching_mapper(["question"], [], [], [], [], 1024, tokenizer, ["option"], ["choices"], ["answerKey"], n_jumbled_options=0, n_within_text_options=0), batched=True, num_proc=16, batch_size=2, remove_columns=['id', 'question', 'choices', 'answerKey'])
ai2_arc_qna.save_to_disk("/home/ahemf/processed_datasets/ai2_arc_qna")

ai2_arc_easy = load_dataset("ai2_arc", 'ARC-Easy').map(lambda x: dict(answerKey=ord(x["answerKey"]) - ord(x["choices"]["label"][0]), choices=x["choices"]['text']))
ai2_arc_easy_qna = ai2_arc_easy.map(get_matching_mapper(["question"], [], [], [], [], 1024, tokenizer, ["option"], ["choices"], ["answerKey"], n_jumbled_options=0, n_within_text_options=0), batched=True, num_proc=16, batch_size=2, remove_columns=['id', 'question', 'choices', 'answerKey'])
ai2_arc_easy_qna.save_to_disk("/home/ahemf/processed_datasets/ai2_arc_easy_qna")

#
eraser_multi_rc = load_dataset("eraser_multi_rc").map(lambda x: dict(query=x["query_and_answer"].split('||')[0], answer=x["query_and_answer"].split('||')[1], ql=0))
eraser_multi_rc_qna = eraser_multi_rc.map(get_matching_mapper(["passage"], [], [], [], [], 1024, tokenizer, ["query", "supporting fact"], [["answer"], "evidences"], ["ql", "label"], n_jumbled_options=1, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=[ 'evidences', 'label', 'passage', 'ql', 'query_and_answer'])
eraser_multi_rc_qna.save_to_disk("/home/ahemf/processed_datasets/eraser_multi_rc_qna")

#
rotten_tomatoes = load_dataset("rotten_tomatoes").map(lambda x: dict(label="positive" if x["label"] else "negative"))
rotten_tomatoes_qna = rotten_tomatoes.map(get_matching_mapper(["text"], [], [], ["Is the sentiment expressed positive or negative"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['label'])
rotten_tomatoes_qna.save_to_disk("/home/ahemf/processed_datasets/rotten_tomatoes_qna")

#
sentiment140 = load_dataset("sentiment140").filter(lambda x: len(x['text'].split())>=24, num_proc=16).map(lambda x: dict(label={0: "negative", 2: "neutral", 4: "positive"}[x["sentiment"]]), remove_columns=['date', 'user', 'sentiment', 'query'])
sentiment140 = sentiment140.map(lambda x: dict(text=" ".join(filter(lambda x: not x.startswith('@'),x["text"].split()))), num_proc=16)
sentiment140_qna = sentiment140.map(get_matching_mapper(["text"], [], [], ["Is the sentiment expressed positive or negative"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['label'])
sentiment140_qna.save_to_disk("/home/ahemf/processed_datasets/sentiment140_qna")

#
scitldr = load_dataset("scitldr", 'Abstract', script_version="master").map(lambda x: dict(text=" ".join(x["source"])), remove_columns=['source', 'source_labels', 'rouge_scores', 'paper_id',])
scitldr_qna = scitldr.map(get_matching_mapper(["text"], ["summary"], ["target"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=4, remove_columns=["target"])


scitldr = load_dataset("scitldr", 'Abstract', script_version="master").map(lambda x: dict(text=" ".join(x["source"]), options=x["source"], label=x["source_labels"].index(1)), remove_columns=['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'])
scitldr_qna_v2 = scitldr.map(get_matching_mapper(["text"], [], [], [], [], 1024, tokenizer, ["most important sentence"], ["options"], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=1, remove_columns=["options", "label"])

scitldr = load_dataset("scitldr", 'Abstract', script_version="master").map(lambda x: dict(text=" ".join([("sentence %s: " % i) + s for i, s in enumerate(x["source"])]), label=["sentence %s" % i for i, _ in enumerate(x["source"])][x["source_labels"].index(1)]), remove_columns=['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'])
scitldr_qna_v3 = scitldr.map(get_matching_mapper(["text"], [], [], ["most important sentence"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=1, remove_columns=["label"])

scitldr_qna.save_to_disk("/home/ahemf/processed_datasets/scitldr_qna")
scitldr_qna_v2.save_to_disk("/home/ahemf/processed_datasets/scitldr_qna_v2")
scitldr_qna_v3.save_to_disk("/home/ahemf/processed_datasets/scitldr_qna_v3")

#

wikihow_all = load_dataset("wikihow", 'all', data_dir='/local/')
wikihow_all_qna_v1 = wikihow_all.map(get_matching_mapper(["text"], ["title"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['headline', 'title'])
wikihow_all_qna_v2 = wikihow_all.map(get_matching_mapper(["headline"], ["title"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=4, remove_columns=['headline', 'title'])

wikihow_sep = load_dataset("wikihow", 'sep', data_dir='/local/')
wikihow_sep_qna_v1 = wikihow_sep.map(get_matching_mapper(["text"], ["title"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=4, remove_columns=['headline', 'title', 'overview', 'sectionLabel'])
wikihow_sep_qna_v2 = wikihow_sep.map(get_matching_mapper(["overview"], ["title"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=4, remove_columns=['headline', 'title', 'overview', 'sectionLabel'])
wikihow_sep_qna_v3 = wikihow_sep.map(get_matching_mapper(["text"], ["headline"], ["title"], [], [], 1024, tokenizer, n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=4, remove_columns=['headline', 'title', 'overview', 'sectionLabel'])

wikihow_all_qna_v1.save_to_disk("/home/ahemf/processed_datasets/wikihow_all_qna_v1")
wikihow_all_qna_v2.save_to_disk("/home/ahemf/processed_datasets/wikihow_all_qna_v2")
wikihow_sep_qna_v1.save_to_disk("/home/ahemf/processed_datasets/wikihow_sep_qna_v1")
wikihow_sep_qna_v2.save_to_disk("/home/ahemf/processed_datasets/wikihow_sep_qna_v2")
wikihow_sep_qna_v3.save_to_disk("/home/ahemf/processed_datasets/wikihow_sep_qna_v3")
#

# biomrc_large_A, biomrc_large_B, kilt, crawl_domain

#
sglue_proc = dict()
sglue_proc['boolq'] = super_glue['boolq'].map(lambda x: dict(label={1:"yes", 0:"no", -1:tokenizer.mask_token}[x["label"]]))
sglue_proc['boolq'] = sglue_proc['boolq'].map(get_matching_mapper(["passage"], [], [], ["question"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=4, remove_columns=['question', 'passage', 'idx', 'label'])
sglue_proc['boolq'].save_to_disk("/home/ahemf/processed_datasets/superglue_boolq")


sglue_proc['cb_v1'] = super_glue['cb'].map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "agree", 1: "contradict", 2:"neutral", -1:tokenizer.mask_token}[x["label"]])), remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
sglue_proc['cb_v1'] = sglue_proc['cb_v1'].map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
sglue_proc['cb_v1'].save_to_disk("/home/ahemf/processed_datasets/superglue_cb_v1")


sglue_proc['cb_v2'] = super_glue['cb'].map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "entail", 1: "disagree", 2:"unrelated", -1:tokenizer.mask_token}[x["label"]])), remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
sglue_proc['cb_v2'] = sglue_proc['cb_v2'].map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
sglue_proc['cb_v2'].save_to_disk("/home/ahemf/processed_datasets/superglue_cb_v2")

sglue_proc['copa_v1'] = super_glue["copa"].map(lambda x: dict(question=x["question"] + ", ")).map(get_matching_mapper(["premise"], [], [], [], [], 1024, tokenizer, ['question'], [["choice1", "choice2"]], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=2, remove_columns=['premise', 'choice1', 'choice2', 'question', 'idx', 'label'])
sglue_proc['copa_v2'] = super_glue["copa"].map(lambda x: dict(question=x["question"] + ", ")).map(get_matching_mapper(["premise"], [], [], [], [], 1024, tokenizer, ['question'], [["choice1", "choice2"]], ["label"], n_jumbled_options=0, n_within_text_options=0), batched=True, num_proc=16, batch_size=2, remove_columns=['premise', 'choice1', 'choice2', 'question', 'idx', 'label'])
sglue_proc['copa_v3'] = super_glue["copa"].map(lambda x: dict(question=x["question"] + ", ")).map(get_matching_mapper(["premise"], [], [], [], [], 1024, tokenizer, ['question'], [["choice1", "choice2"]], ["label"], n_jumbled_options=0, n_within_text_options=0), batched=True, num_proc=16, batch_size=1, remove_columns=['premise', 'choice1', 'choice2', 'question', 'idx', 'label'])

sglue_proc['copa_v1'].save_to_disk("/home/ahemf/processed_datasets/superglue_copa_v1")
sglue_proc['copa_v2'].save_to_disk("/home/ahemf/processed_datasets/superglue_copa_v2")
sglue_proc['copa_v3'].save_to_disk("/home/ahemf/processed_datasets/superglue_copa_v3")


sglue_proc['multirc_v1'] = super_glue["multirc"].map(lambda x: dict(label=({0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]])))
sglue_proc['multirc_v1'] = sglue_proc['multirc_v1'].map(get_matching_mapper(["paragraph", "question", "answer"], [], [], [["Is the question answered correctly?", "Does the previous statement provide valid answer to the asked question?"]], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['paragraph', 'question', 'answer', 'idx', 'label'])
sglue_proc['multirc_v1'].save_to_disk("/home/ahemf/processed_datasets/superglue_multirc_v1")

sglue_proc['multirc_v2'] = super_glue["multirc"].map(lambda x: dict(paragraph ="context: "+ x["paragraph"] + " question: "+x["question"] + " answer: "+x["answer"],  label=({0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]])))
sglue_proc['multirc_v2'] = sglue_proc['multirc_v2'].map(get_matching_mapper(["paragraph"], [], [], [["Is the question answered correctly?", "Does the previous statement provide valid answer to the asked question?"]], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['paragraph', 'question', 'answer', 'idx', 'label'])
sglue_proc['multirc_v2'].save_to_disk("/home/ahemf/processed_datasets/superglue_multirc_v2")

from datasets import DatasetDict
sglue_proc['multirc_v3'] = DatasetDict(**super_glue["multirc"])
del sglue_proc['multirc_v3']["test"]
sglue_proc['multirc_v3'] = sglue_proc['multirc_v3'].filter(lambda x: x["label"]==1).map(lambda x: dict(label=0))
sglue_proc['multirc_v3'] = sglue_proc['multirc_v3'].map(get_matching_mapper(["paragraph"], [], [], [], [], 1024, tokenizer, ["question"], [["answer"]], ["label"], n_jumbled_options=1, n_within_text_options=2), batched=True, num_proc=16, batch_size=1, remove_columns=['paragraph', 'question', 'answer', 'idx', 'label'])
sglue_proc['multirc_v3'].save_to_disk("/home/ahemf/processed_datasets/superglue_multirc_v3")

sglue_proc['record_v1'] = super_glue['record'].map(lambda x: dict(query=x["query"] + " Find the right entity for @placeholder from the given options.", label=x['entities'].index(x['answers'][0]) if len(x['answers']) > 0 else -1))
sglue_proc['record_v1'] = sglue_proc['record_v1'].map(get_matching_mapper(["passage"], [], [], [], [], 1024, tokenizer, ["query"], ['entities'], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=1, remove_columns=['passage', 'query', 'entities', 'answers', 'idx', 'label'])
sglue_proc['record_v2'] = super_glue['record'].map(lambda x: dict(query=x["query"] + " Find the right entity for @placeholder from the given options.", label=x['entities'].index(x['answers'][0]) if len(x['answers']) > 0 else -1))
sglue_proc['record_v2'] = sglue_proc['record_v2'].map(lambda x: dict(answers=x["answers"][0] if len(x["answers"]) > 0 else tokenizer.mask_token)).map(get_matching_mapper(["passage"], [], [], ["query"], ["answers"], 1024, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['passage', 'query', 'entities', 'answers', 'idx', 'label'])

sglue_proc['record_v3'] = DatasetDict(**super_glue["record"])
del sglue_proc['record_v3']["test"]
sglue_proc['record_v3'] = sglue_proc['record_v3'].filter(lambda x: len(set([i.lower() for i in x['answers']])) > 1).map(lambda x: dict(query=x["query"] + " Find the right entity for @placeholder from the given options.", label=x['entities'].index(x['answers'][1] if len(x['answers']) > 0 else -1)))
sglue_proc['record_v3'] = sglue_proc['record_v3'].map(get_matching_mapper(["passage"], [], [], [], [], 1024, tokenizer, ["query"], ['entities'], ["label"], n_jumbled_options=0, n_within_text_options=2), batched=True, num_proc=16, batch_size=1, remove_columns=['passage', 'query', 'entities', 'answers', 'idx', 'label'])
sglue_proc['record_v4'] = DatasetDict(**super_glue["record"])
del sglue_proc['record_v4']["test"]
sglue_proc['record_v4'] = sglue_proc['record_v4'].filter(lambda x: len(set([i.lower() for i in x['answers']])) > 1).map(lambda x: dict(query=x["query"] + " Find the right entity for @placeholder.", label=x['entities'].index(x['answers'][1] if len(x['answers']) > 0 else -1)))
sglue_proc['record_v4'] = sglue_proc['record_v4'].map(lambda x: dict(answers=x["answers"][1]  if len(x["answers"]) > 1 else tokenizer.mask_token)).map(get_matching_mapper(["passage"], [], [], ["query"], ["answers"], 1024, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['passage', 'query', 'entities', 'answers', 'idx', 'label'])

sglue_proc['record_v1'].save_to_disk("/home/ahemf/processed_datasets/superglue_record_v1")
sglue_proc['record_v2'].save_to_disk("/home/ahemf/processed_datasets/superglue_record_v2")
sglue_proc['record_v3'].save_to_disk("/home/ahemf/processed_datasets/superglue_record_v3")
sglue_proc['record_v4'].save_to_disk("/home/ahemf/processed_datasets/superglue_record_v4")

sglue_proc['rte_v1'] = super_glue['rte'].map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "agree", 1: "contradict", 2:"neutral", -1:tokenizer.mask_token}[x["label"]])), remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
sglue_proc['rte_v1'] = sglue_proc['rte_v1'].map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis agree, contradict or are unrelated (neutral)"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
sglue_proc['rte_v1'].save_to_disk("/home/ahemf/processed_datasets/superglue_rte_v1")


sglue_proc['rte_v2'] = super_glue['rte'].map(lambda x: dict(text="premise: "+ x["premise"]+ " hypothesis: " + x["hypothesis"], label=({0: "entail", 1: "disagree", 2:"unrelated", -1:tokenizer.mask_token}[x["label"]])), remove_columns=["idx", "hypothesis", "premise"], num_proc=8)
sglue_proc['rte_v2'] = sglue_proc['rte_v2'].map(get_matching_mapper(["text"], [], [], ["Do the premise and hypothesis entail, disagree or are unrelated"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=["label"])
sglue_proc['rte_v2'].save_to_disk("/home/ahemf/processed_datasets/superglue_rte_v2")


sglue_proc["wic_v1"] = super_glue["wic"].map(lambda x: dict(text="first sentence: " + x["sentence1"] + ", second sentence: "+x["sentence2"], query="Does the word '%s' have same meaning in first sentence and second sentence" % x["word"], label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]]))
sglue_proc["wic_v2"] = super_glue["wic"].map(lambda x: dict(query="Does the word '%s' mean the same in both sentences" % x["word"], label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]]))
sglue_proc["wic_v1"] = sglue_proc["wic_v1"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2', 'idx', 'label'])
sglue_proc["wic_v2"] = sglue_proc["wic_v2"].map(get_matching_mapper(["sentence1", "sentence2"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2', 'idx', 'label'])
sglue_proc["wic_v2"].save_to_disk("/home/ahemf/processed_datasets/superglue_wic_v2")
sglue_proc["wic_v1"].save_to_disk("/home/ahemf/processed_datasets/superglue_wic_v1")


sglue_proc["wsc_v1"] = super_glue["wsc"].map(lambda x: dict(label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]], 
                                            query="Do [word1] %s and [word2] %s refer to the same entity" % (x["span1_text"], x["span2_text"]),
                                            text=" ".join([defaultdict(str, {x["span1_index"]: "[word1] ", x["span2_index"]: "[word2] "})[idx] + w for idx, w in enumerate(x["text"].split())])))
sglue_proc["wsc_v2"] = super_glue["wsc"].map(lambda x: dict(label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]], 
                                            query="Do [%s] and [%s] refer to the same thing" % (x["span1_text"], x["span2_text"]),
                                            text=" ".join([("[" if idx in [x["span1_index"], x["span2_index"]] else "") +  w + ("]" if idx in [x["span1_index"], x["span2_index"]] else "") for idx, w in enumerate(x["text"].split())])))

sglue_proc["wsc_v1"] = sglue_proc["wsc_v1"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['idx', 'label', 'query', 'span1_index', 'span1_text', 'span2_index', 'span2_text'])
sglue_proc["wsc_v2"] = sglue_proc["wsc_v2"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['idx', 'label', 'query', 'span1_index', 'span1_text', 'span2_index', 'span2_text'])
sglue_proc["wsc_v1"].save_to_disk("/home/ahemf/processed_datasets/superglue_wsc_v1")
sglue_proc["wsc_v2"].save_to_disk("/home/ahemf/processed_datasets/superglue_wsc_v2")

#

sglue_proc["wsc.fixed_v1"] = super_glue["wsc.fixed"].map(lambda x: dict(label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]], 
                                            query="Do [word1] %s and [word2] %s refer to the same entity" % (x["span1_text"], x["span2_text"]),
                                            text=" ".join([defaultdict(str, {x["span1_index"]: "[word1] ", x["span2_index"]: "[word2] "})[idx] + w for idx, w in enumerate(x["text"].split())])))
sglue_proc["wsc.fixed_v2"] = super_glue["wsc.fixed"].map(lambda x: dict(label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]], 
                                            query="Do [%s] and [%s] refer to the same thing" % (x["span1_text"], x["span2_text"]),
                                            text=" ".join([("[" if idx in [x["span1_index"], x["span2_index"]] else "") +  w + ("]" if idx in [x["span1_index"], x["span2_index"]] else "") for idx, w in enumerate(x["text"].split())])))

sglue_proc["wsc.fixed_v1"] = sglue_proc["wsc.fixed_v1"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['idx', 'label', 'query', 'span1_index', 'span1_text', 'span2_index', 'span2_text'])
sglue_proc["wsc.fixed_v2"] = sglue_proc["wsc.fixed_v2"].map(get_matching_mapper(["text"], [], [], ["query"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['idx', 'label', 'query', 'span1_index', 'span1_text', 'span2_index', 'span2_text'])
sglue_proc["wsc.fixed_v1"].save_to_disk("/home/ahemf/processed_datasets/superglue_wsc_fixed_v1")
sglue_proc["wsc.fixed_v2"].save_to_disk("/home/ahemf/processed_datasets/superglue_wsc_fixed_v2")


glue_proc = dict()
glue_proc['cola'] = glue['cola'].map(lambda x: dict(label={0: "no", 1: "yes", -1:tokenizer.mask_token}[x["label"]]))
glue_proc['cola'] = glue_proc['cola'].map(get_matching_mapper(["sentence"], [], [], ["Is the sentence grammatically and syntactically correct?"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=8, remove_columns=['sentence', 'label', 'idx'])
glue_proc['cola'].save_to_disk("/home/ahemf/processed_datasets/glue_cola")

glue_proc['sst2'] = glue['sst2'].map(lambda x: dict(label={0: "negative", -1: tokenizer.mask_token, 1: "positive"}[x["label"]]))
glue_proc['sst2'] = glue_proc['sst2'].map(get_matching_mapper(["sentence"], [], [], ["Is the sentiment expressed positive or negative"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['sentence', 'label', 'idx'])
glue_proc['sst2'].save_to_disk("/home/ahemf/processed_datasets/glue_sst2")

glue_proc['sst2_v2'] = glue['sst2'].map(lambda x: dict(label={0: "no", -1: tokenizer.mask_token, 1: "yes"}[x["label"]]))
glue_proc['sst2_v2'] = glue_proc['sst2_v2'].map(get_matching_mapper(["sentence"], [], [], ["Is positive sentiment expressed?"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['sentence', 'label', 'idx'])
glue_proc['sst2_v2'].save_to_disk("/home/ahemf/processed_datasets/glue_sst2_v2")

glue_proc["qnli"] = glue["qnli"].map(lambda x: dict(text="context: "+x["sentence"] + ", question: "+x["question"], label={0: "yes", -1: tokenizer.mask_token, 1: "no"}[x["label"]]))
glue_proc["qnli"] = glue_proc["qnli"].map(get_matching_mapper(["text"], [], [], ["Does the context answer the question?"], ["label"], 1024, tokenizer), batched=True, num_proc=16, batch_size=2, remove_columns=['question', 'sentence', 'label', 'idx'])
glue_proc['qnli'].save_to_disk("/home/ahemf/processed_datasets/glue_qnli")

# MRC can be used as mcq and mlm


"""

"""
import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

contents = os.listdir("processed_datasets")
dataset_dict = dict()
unloadable = []
loadable = []
for c in contents:
    try:
        try:
            dataset_dict[c] = Dataset.load_from_disk("processed_datasets/%s" % c)
        except:
            dataset_dict[c] = DatasetDict.load_from_disk("processed_datasets/%s" % c)
        loadable.append(c)
    except:
        unloadable.append(c)
        
print(len(unloadable), len(loadable), len(contents))

dsets_512 = ['openwebtext512',
 'reddit512',
 'scientific_papers_arxiv512',
 'scientific_papers_pubmed512',
 'yahoo_answers_topics512',
 'bookcorpusopen512',
 'bookcorpus512',
 'big_patent512',
 'reuters512',
 'amazon_polarity512',
 'wiki512',
 'yahoo_answers_qa512']
 
 dsets = ['openwebtext512',
    'reddit512',
    'yahoo_answers_topics512',
    'bookcorpusopen512',
    'bookcorpus512',
    'reuters512',
    'wiki512',
    'yahoo_answers_qa512']

dsets = ['ai2_arc_easy_qna',
 'ai2_arc_qna',
 'amazon_polarity_qna',
 'amazon_reviews_multi1024',
 'amazon_reviews_multi_qna',
 'amazon_us_reviews1024',
 'anli_qna_v1',
 'anli_qna_v2',
 'app_reviews_qna',
 'big_patent1024',
 'bookcorpus1024',
 'bookcorpusopen1024',
 'bookcorpusopen_qna',
 'cnn_dailymail1024',
 'cnn_dailymail_qna',
 'commonsense_qa',
 'cosmos_qa',
 'discovery_qna',
 'e2e_nlg_cleaned_qna',
 'eli51024',
 'eli5_qna',
 'empathetic_dialogues_qna',
 'eraser_multi_rc_qna',
 'giga_fren1024',
 'gigaword_qna',
 'glue_cola',
 'glue_qnli',
 'glue_sst2',
 'glue_sst2_v2',
 'go_emotions_qna',
 'hans_qna_v1',
 'hans_qna_v2',
 'hellaswag_qna',
 'hotpot_qa_qna_v1',
 'hotpot_qa_qna_v2',
 'imdb_qna',
 'kelm1024',
 'medical_questions_pairs_qna',
 'mnli_qna_v1',
 'mnli_qna_v2',
 'mocha_qna',
 'mrqa_v1',
 'mrqa_v2',
 'multi_news_qna',
 'narrativeqa_v1',
 'narrativeqa_v2',
 'natural_questions_qna',
 'ohsumed1024',
 'ohsumed_qna',
 'openwebtext1024',
 'paws_qna',
 'peer_read_reviews_qna',
 'per_sent_qna',
 'piqa',
 'pubmed_qa',
 'qangaroo_qna',
 'qangaroo_qna_v2',
 'qasc_qa_v1',
 'qasc_qa_v2',
 'qasc_qa_v3',
 'qasc_qa_v4',
 'qed_qna_v1',
 'qed_qna_v2',
 'quac_qna',
 'quartz_qna',
 'quora',
 'quoref_qna',
 'race_qna',
 'reddit1024',
 'reddit_qna',
 'reuters_qna',
 'ropes_qna',
 'ropes_qna_v2',
 'rotten_tomatoes_qna',
 'samsum_qna',
 'sciq',
 'scitail_qna',
 'scitldr_qna',
 'scitldr_qna_v2',
 'scitldr_qna_v3',
 'sent_comp_qna',
 'sentiment140_qna',
 'snli_qna_v1',
 'snli_qna_v2',
 'social_i_qa_v1',
 'social_i_qa_v2',
 'social_i_qa_v3',
 'squad_v2_qna',
 'squad_v2_qna_v2',
 'superglue_boolq',
 'superglue_cb_v1',
 'superglue_cb_v2',
 'superglue_copa_v1',
 'superglue_copa_v2',
 'superglue_copa_v3',
 'superglue_multirc_v1',
 'superglue_multirc_v2',
 'superglue_multirc_v3',
 'superglue_record_v1',
 'superglue_record_v2',
 'superglue_record_v3',
 'superglue_record_v4',
 'superglue_rte_v1',
 'superglue_rte_v2',
 'superglue_wic_v1',
 'superglue_wic_v2',
 'superglue_wsc_fixed_v1',
 'superglue_wsc_fixed_v2',
 'superglue_wsc_v1',
 'superglue_wsc_v2',
 'swag_qna',
 'to_en1024',
 'un_pc1024',
 'wiki1024',
 'wiki_atomic_edits_deletions_qna',
 'wiki_atomic_edits_insertions_qna',
 'wiki_auto_qna',
 'wikihow_all1024',
 'wikihow_all_qna_v1',
 'wikihow_all_qna_v2',
 'wikihow_sep1024',
 'wikihow_sep_qna_v1',
 'wikihow_sep_qna_v2',
 'wikihow_sep_qna_v3',
 'wiki_lingua_qna',
 'wikipedia_qna',
 'wiki_qa_qna',
 'wiki_split_qna',
 'winogrande_qna',
 'wmt14de_en1024',
 'wmt15fr_en1024',
 'wmt16ru_en1024',
 'wmt17cs_en1024',
 'xsum1024',
 'xsum_qna',
 'yahoo_answers_qa_qna',
 'yahoo_answers_topics_qna',
 'yelp_polarity_qna',
 'yelp_review_full1024',
 'amazon_polarity512',
 'reuters512',
 'scientific_papers_arxiv512',
 'scientific_papers_pubmed512',
 'yahoo_answers_qa512',
 'yahoo_answers_topics512',
 ]
dataset_dict_filtered = {k:v for k, v in dataset_dict.items() if k in dsets}

train_ds = dict()
validation_ds = dict()
test_ds = dict()
for k, v in dataset_dict_filtered.items():
    train = None
    test = None
    validation = None
    if isinstance(v, Dataset):
        train = v
    else:
        splits = list(v.keys())
        train = v["train"] if "train" in v else None
        test = v["test"] if "test" in v else None
        validation = v["validation"] if "validation" in v else None
    for split in [train, validation, test]:
        if split is None:
            continue
        feats = split.column_names
        
        if "query" not in feats:
            split.map(lambda x: dict(query=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
        if "answer" not in feats:
            split.map(lambda x: dict(answer=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
    
    if train is not None:
        train_ds[k] = train
    if validation is not None:
        validation_ds[k] = validation
    if test is not None:
        test_ds[k] = test

# TODO: test for sequences of 1024 length 
# dataset_name, dataset_type, make qna and mlm datasets separate?

train_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/train_fastformer")
validation_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/validation_fastformer")
test_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/test_fastformer")

def ds_length_stats(ds, lbs=((0, 64), (64, 128), (128, 512), (512, 768), (768, 1024)), tokenizer=None):
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    from collections import defaultdict
    if isinstance(ds, Dataset):
        ds = DatasetDict(dict(ds=ds))

    splits = list(ds.keys())
    split_info = defaultdict(dict)
    len_info = defaultdict(dict)
    for key in splits:
        split = ds[key]
        remove_cols = list(set(split.column_names) - {'length'})
        split = split.map(lambda x: dict(length=x["length"]), batch_size=4096, batched=True, remove_columns=remove_cols)
        for lb in lbs:
            l = len(split.filter(lambda x: lb[0]<=x["length"]<lb[1]))
            split_info[key][lb] = l
            len_info[lb][key] = l
    aggregate_len_info = {ll: sum(vl.values()) for ll, vl in len_info.items()}
    return aggregate_len_info, len_info, split_info


ds_length_stats(train_fastformer)
"""

"""
import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


train_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/train_fastformer")
validation_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/validation_fastformer")
test_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/test_fastformer")

train_fastformer_1p = dict()
for k, v in train_fastformer.items():
    length = len(v)
    new_len = length // 100
    v = Dataset.from_dict(v[0:new_len])
    train_fastformer_1p[k] = v
    
train_fastformer_1p = DatasetDict(**train_fastformer_1p)

"""

"""
import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


train_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/train_fastformer")
sum(map(len, train_fastformer.values()))

sampling_fraction = 0.75 # 0.5, 0.65, 0.75
train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_fastformer.items()}
lsum = sum(train_dataset_sampling_proba.values())
train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
probas = {k: int(v * 10_000_000) for k, v in train_dataset_sampling_proba.items()}
sum(probas.values())
probas

def resample_dataset(ds, sample_size):
    if sample_size < len(ds):
        select_proba = sample_size / len(ds)
        return ds.filter(lambda x: random.random() <= select_proba, batch_size=16_000)
    else:
        repeats = sample_size // len(ds)
        delta = resample_dataset(ds, sample_size - (repeats * len(ds)))
        return concatenate_datasets(([ds]*repeats)+[delta])
        


def add_query_answer_columns(x):
    if "query" not in x:
        x["query"] = [[""]] * len(x["text"])
    else:
        x["query"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]),x["query"]))
    if "answer" not in x:
        x["answer"] = [[""]] * len(x["text"])
    else:
        x["answer"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]), x["answer"]))
    return x

train_fastformer_resampled = DatasetDict({k: resample_dataset(train_fastformer[k], samples) for k, samples in probas.items()})
train_fastformer_resampled = DatasetDict({k: v.map(add_query_answer_columns, batched=True, batch_size=16_384, remove_columns=["label"] if "label" in v.column_names else []) for k, v in train_fastformer_resampled.items()})
dsets = list(train_fastformer_resampled.values())
keys = list(train_fastformer_resampled.keys()) 
for k, dset in zip(keys, dsets):
    if dset.features.type != dsets[0].features.type:
        print(k," :: ", dset.features.type, " :: ", dsets[0].features.type, "\n")
        dset.cast_(dsets[0].features)
train_fastformer_resampled = concatenate_datasets(list(train_fastformer_resampled.values()))

def filter_small_text(x):
    return len(x["text"].strip()) > 10 and len(x["text"].strip().split()) > 8 and x["length"] < 1024
    
train_fastformer_resampled = train_fastformer_resampled.filter(filter_small_text, batch_size=4096)

train_fastformer_resampled = train_fastformer_resampled.sort("length")

def batched_reshuffle(x):
    xv = list(zip(*x.values()))
    xv = random.sample(xv, len(xv))
    xv = list(zip(*xv))
    shuffled = dict(zip(x.keys(), xv)) 
    return {k: list(v) for k, v in shuffled.items()}

train_fastformer_resampled = train_fastformer_resampled.map(batched_reshuffle, batched=True, batch_size=16_384)
train_fastformer_resampled_100M = train_fastformer_resampled
train_fastformer_resampled_100M.save_to_disk("/home/ahemf/processed_datasets/train_fastformer_resampled_100M")

di = DatasetInfo.from_directory("/home/ahemf/processed_datasets/train_fastformer_resampled_100M")
print({x: len(getattr(di, x)) if hasattr(getattr(di, x), "__len__") else x for x in dir(di) if not x.startswith('__')})
di.homepage=''
di.license='MIT'
di.citation=''
di.description=''
di.write_to_directory("/home/ahemf/processed_datasets/train_fastformer_resampled_100M")
"""

# checker
"""
import time
import datasets
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


from fastformer.data import TokenizerDataset, get_collate_fn, custom_batching_fn, char_to_id
from fastformer.config import md_config, model_config, size_dicts
from fastformer.utils import get_tokenizer

mconf = model_config.to_dict()
tokenizer = get_tokenizer(mconf.pop("tokenizer_name"))
mconf = model_config.to_dict()
config = md_config

tokenizer = get_tokenizer(mconf.pop("tokenizer_name"))
config.vocab_size = len(tokenizer) + 22
config.tokenizer_length = 1024
config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens
collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)

train_fastformer_resampled = Dataset.load_from_disk("/home/ahemf/processed_datasets/train_fastformer_resampled_10M")
train_dataset = TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), train_fastformer_resampled)
train_loader = DataLoader(train_dataset, sampler=None, batch_size=8, collate_fn=collate_fn, prefetch_factor=2, num_workers=64, shuffle=False)
train_loader = custom_batching_fn(tqdm(train_loader), size_dicts, False)

st = time.time()
times = []
for step, batch in enumerate(train_loader):
    if (step + 1) % 1000 == 0:
        print("Step = %s, Average Time = %s" % (step, np.mean(times)))
        times = []
    et = time.time()
    st = st
    times.append(et - st)
    
validation_fastformer = DatasetDict.load_from_disk("/home/ahemf/processed_datasets/validation_fastformer")
dataset_dict = validation_fastformer
dataset_dict = {"big_patent1024": Dataset.from_dict(validation_fastformer["big_patent1024"][328784:])}

for k, v in tqdm(dataset_dict.items()):
    dataset = TokenizerDataset(config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), v)
    loader = DataLoader(dataset, sampler=None, batch_size=8, collate_fn=collate_fn, num_workers=0, shuffle=False)
    loader = custom_batching_fn(tqdm(loader), size_dicts, False)
    
    st = time.time()
    times = []
    for step, batch in enumerate(loader):
        if (step + 1) % 1000 == 0:
            print("Step = %s, Average Time = %s" % (step, np.mean(times)))
            times = []
        et = time.time()
        st = st
        times.append(et - st)
    

"""

"""
train_fastformer_resampled = Dataset.load_from_disk("/home/ahemf/processed_datasets/train_fastformer_resampled_100M")

nplen = np.array(train_fastformer_resampled._data['length'].to_pylist())
train_fastformer_resampled._data = train_fastformer_resampled._data.filter(mask=nplen>12)
setattr(train_fastformer_resampled, "_num_rows", len(train_fastformer_resampled._data))
print(np.min(train_fastformer_resampled._data['length'].to_pylist()))
train_fastformer_resampled.save_to_disk("/home/ahemf/processed_datasets/train_fastformer_resampled_100M")

"""


"""
# QNA medium set builder
# ls | grep 'qna\|superglue_\|sciq\|quora\|piqa\|mrqa\|cosmos_qa\|commonsense_qa\|glue_' | grep -v'wikipedia_\|yahoo_answers_topics_qna\|wiki_atomic_edits_\|reddit_qna\|cnn_dailymail_qna\|amazon_polarity_qna\|wikihow_sep_\|natural_questions_qna\|ohsumed\|gigaword_\|mrqa_\|wiki_split_\|xsum_\|multi_news_\|wiki_lingua\|wiki_auto\|reuters_qna\|eli5_qna'
# ls | grep 'qna\|superglue_\|sciq\|quora\|piqa\|mrqa\|cosmos_qa\|commonsense_qa\|glue_' | grep -v'wikipedia_\|yahoo_answers_topics_qna\|wiki_atomic_edits_\|reddit_qna\|cnn_dailymail_qna\|amazon_polarity_qna\|wikihow_sep_\|natural_questions_qna\|ohsumed\|gigaword_\|mrqa_\|wiki_split_\|xsum_\|multi_news_\|wiki_lingua\|wiki_auto\|reuters_qna\|eli5_qna' | xargs -I '{}' ls -lh '{}'/train/dataset.arrow

import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


datasets = [
"ai2_arc_easy_qna",
"ai2_arc_qna",
"amazon_reviews_multi_qna",
"anli_qna_v1",
"anli_qna_v2",
"app_reviews_qna",
"bookcorpusopen_qna",
"commonsense_qa",
"cosmos_qa",
"discovery_qna",
"e2e_nlg_cleaned_qna",
"empathetic_dialogues_qna",
"eraser_multi_rc_qna",
"glue_cola",
"glue_qnli",
"glue_sst2",
"glue_sst2_v2",
"go_emotions_qna",
"hans_qna_v1",
"hans_qna_v2",
"hellaswag_qna",
"hotpot_qa_qna_v1",
"hotpot_qa_qna_v2",
"imdb_qna",
"medical_questions_pairs_qna",
"mnli_qna_v1",
"mnli_qna_v2",
"mocha_qna",
"paws_qna",
"peer_read_reviews_qna",
"per_sent_qna",
"piqa",
"qangaroo_qna",
"qangaroo_qna_v2",
"qed_qna_v1",
"qed_qna_v2",
"quac_qna",
"quartz_qna",
"quora",
"quoref_qna",
"race_qna",
"ropes_qna",
"ropes_qna_v2",
"rotten_tomatoes_qna",
"samsum_qna",
"sciq",
"scitail_qna",
"scitldr_qna",
"scitldr_qna_v2",
"scitldr_qna_v3",
"sent_comp_qna",
"sentiment140_qna",
"snli_qna_v1",
"snli_qna_v2",
"squad_v2_qna",
"squad_v2_qna_v2",
"superglue_boolq",
"superglue_cb_v1",
"superglue_cb_v2",
"superglue_copa_v1",
"superglue_copa_v2",
"superglue_copa_v3",
"superglue_multirc_v1",
"superglue_multirc_v2",
"superglue_multirc_v3",
"superglue_record_v1",
"superglue_record_v2",
"superglue_record_v3",
"superglue_record_v4",
"superglue_rte_v1",
"superglue_rte_v2",
"superglue_wic_v1",
"superglue_wic_v2",
"superglue_wsc_fixed_v1",
"superglue_wsc_fixed_v2",
"superglue_wsc_v1",
"superglue_wsc_v2",
"swag_qna",
"wikihow_all_qna_v1",
"wikihow_all_qna_v2",
"wiki_qa_qna",
"winogrande_qna",
"yahoo_answers_qa_qna",
"yelp_polarity_qna"
]

contents = datasets
dataset_dict = dict()
unloadable = []
loadable = []
for c in contents:
    try:
        try:
            dataset_dict[c] = Dataset.load_from_disk("processed_datasets/%s" % c)
        except:
            dataset_dict[c] = DatasetDict.load_from_disk("processed_datasets/%s" % c)
        loadable.append(c)
    except:
        unloadable.append(c)
        
print(len(unloadable), len(loadable), len(contents))

train_ds = dict()
validation_ds = dict()
test_ds = dict()
for k, v in dataset_dict.items():
    train = dict()
    test = dict()
    validation = dict()
    if isinstance(v, Dataset):
        train[k] = v
    else:
        splits = list(v.keys())
        for split in splits:
            if "train" in split:
                train[k + split.replace("train", '')] = v[split]
            elif "test" in split:
                test[k + split.replace("test", '')] = v[split]
            elif "validation" in split:
                validation[k + split.replace("validation", '')] = v[split]
                
        
    for split in [train, validation, test]:
        if split is None:
            continue
        for dk, dv in split.items():
            feats = dv.column_names
        
            if "query" not in feats:
                dv = dv.map(lambda x: dict(query=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
            if "answer" not in feats:
                dv = dv.map(lambda x: dict(answer=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
            split[dk] = dv
    
    train_ds.update(train)

    validation_ds.update(validation)

    test_ds.update(test)
    

train_qna_medium_fastformer = DatasetDict(train_ds)
validation_qna_medium_fastformer = DatasetDict(validation_ds)
test_qna_medium_fastformer = DatasetDict(test_ds)

train_qna_medium_fastformer.save_to_disk("processed_datasets/train_qna_medium_fastformer")
validation_qna_medium_fastformer.save_to_disk("processed_datasets/validation_qna_medium_fastformer")
test_qna_medium_fastformer.save_to_disk("processed_datasets/test_qna_medium_fastformer")

#####
train_fastformer = train_qna_medium_fastformer
#####

sampling_fraction = 0.5 # 0.5, 0.65, 0.75
train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_fastformer.items()}
lsum = sum(train_dataset_sampling_proba.values())
train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
probas = {k: int(v * 6_000_000) for k, v in train_dataset_sampling_proba.items()}
actual_lens = {k: len(v) for k, v in train_fastformer.items()}
compared_lens = {k: (probas[k], v) for k, v in actual_lens.items()}
compared_truth = {k: probas[k] >= v for k, v in actual_lens.items()}
print(sum(actual_lens.values()))
print(probas)
print(compared_lens)
print(all(compared_truth.values()))

def resample_dataset(ds, sample_size):
    if sample_size < len(ds):
        select_proba = sample_size / len(ds)
        return ds.filter(lambda x: random.random() <= select_proba, batch_size=16_000)
    else:
        repeats = sample_size // len(ds)
        delta = resample_dataset(ds, sample_size - (repeats * len(ds)))
        return concatenate_datasets(([ds]*repeats)+[delta])
        


def add_query_answer_columns(x):
    if "query" not in x:
        x["query"] = [[""]] * len(x["text"])
    else:
        x["query"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]),x["query"]))
    if "answer" not in x:
        x["answer"] = [[""]] * len(x["text"])
    else:
        x["answer"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]), x["answer"]))
    return x

train_fastformer_resampled = DatasetDict({k: resample_dataset(train_fastformer[k], samples) for k, samples in probas.items()})
train_fastformer_resampled = DatasetDict({k: v.map(add_query_answer_columns, batched=True, batch_size=16_384, remove_columns=["label"] if "label" in v.column_names else []) for k, v in train_fastformer_resampled.items()})
dsets = list(train_fastformer_resampled.values())
keys = list(train_fastformer_resampled.keys()) 
for k, dset in zip(keys, dsets):
    if dset.features.type != dsets[0].features.type:
        print(k," :: ", dset.features.type, " :: ", dsets[0].features.type, "\n")
        dset.cast_(dsets[0].features)
train_fastformer_resampled = concatenate_datasets(list(train_fastformer_resampled.values()))

def filter_small_text(x):
    return len(x["text"].strip()) > 8 and len(x["text"].strip().split()) > 4 and x["length"] < 1024
    
train_fastformer_resampled = train_fastformer_resampled.filter(filter_small_text, batch_size=4096)
train_fastformer_resampled.save_to_disk("/home/ahemf/processed_datasets/train_qna_medium_fastformer_5M")

di = DatasetInfo.from_directory("/home/ahemf/processed_datasets/train_qna_medium_fastformer_5M")
print({x: len(getattr(di, x)) if hasattr(getattr(di, x), "__len__") else x for x in dir(di) if not x.startswith('__')})
di.homepage=''
di.license='MIT'
di.citation=''
di.description=''
di.write_to_directory("/home/ahemf/processed_datasets/train_qna_medium_fastformer_5M")

"""


"""
# QNA small set builder
# ls | grep 'qna\|superglue_\|sciq\|quora\|piqa\|mrqa\|cosmos_qa\|commonsense_qa\|glue_' | grep -v'wikipedia_\|yahoo_answers_topics_qna\|wiki_atomic_edits_\|reddit_qna\|cnn_dailymail_qna\|amazon_polarity_qna\|wikihow_sep_\|natural_questions_qna\|ohsumed\|gigaword_\|mrqa_\|wiki_split_\|xsum_\|multi_news_\|wiki_lingua\|wiki_auto\|reuters_qna\|eli5_qna'
# ls | grep 'qna\|superglue_\|sciq\|quora\|piqa\|mrqa\|cosmos_qa\|commonsense_qa\|glue_' | grep -v'wikipedia_\|yahoo_answers_topics_qna\|wiki_atomic_edits_\|reddit_qna\|cnn_dailymail_qna\|amazon_polarity_qna\|wikihow_sep_\|natural_questions_qna\|ohsumed\|gigaword_\|mrqa_\|wiki_split_\|xsum_\|multi_news_\|wiki_lingua\|wiki_auto\|reuters_qna\|eli5_qna' | xargs -I '{}' ls -lh '{}'/train/dataset.arrow

import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from datasets import DatasetInfo
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


datasets = [
"anli_qna_v1",
"anli_qna_v2",
"commonsense_qa",
"eraser_multi_rc_qna",
"glue_cola",
"glue_qnli",
"glue_sst2",
"glue_sst2_v2",
"go_emotions_qna",
"hans_qna_v1",
"hans_qna_v2",
"hellaswag_qna",
"imdb_qna",
"mnli_qna_v1",
"mnli_qna_v2",
"paws_qna",
"per_sent_qna",
"piqa",
"quartz_qna",
"race_qna",
"ropes_qna",
"ropes_qna_v2",
"rotten_tomatoes_qna",
"snli_qna_v1",
"snli_qna_v2",
"squad_v2_qna",
"squad_v2_qna_v2",
"superglue_boolq",
"superglue_cb_v1",
"superglue_cb_v2",
"superglue_copa_v1",
"superglue_copa_v2",
"superglue_copa_v3",
"superglue_multirc_v1",
"superglue_multirc_v2",
"superglue_multirc_v3",
"superglue_record_v1",
"superglue_record_v2",
"superglue_record_v3",
"superglue_record_v4",
"superglue_rte_v1",
"superglue_rte_v2",
"superglue_wic_v1",
"superglue_wic_v2",
"superglue_wsc_fixed_v1",
"superglue_wsc_fixed_v2",
"superglue_wsc_v1",
"superglue_wsc_v2",
"swag_qna",
"winogrande_qna",
]


contents = datasets
dataset_dict = dict()
unloadable = []
loadable = []
for c in contents:
    try:
        try:
            dataset_dict[c] = Dataset.load_from_disk("processed_datasets/%s" % c)
        except:
            dataset_dict[c] = DatasetDict.load_from_disk("processed_datasets/%s" % c)
        loadable.append(c)
    except:
        unloadable.append(c)
        
print(len(unloadable), len(loadable), len(contents))

train_ds = dict()
validation_ds = dict()
test_ds = dict()
for k, v in dataset_dict.items():
    train = dict()
    test = dict()
    validation = dict()
    if isinstance(v, Dataset):
        train[k] = v
    else:
        splits = list(v.keys())
        for split in splits:
            if "train" in split:
                train[k + split.replace("train", '')] = v[split]
            elif "test" in split:
                test[k + split.replace("test", '')] = v[split]
            elif "validation" in split:
                validation[k + split.replace("validation", '')] = v[split]
                
        
    for split in [train, validation, test]:
        if split is None:
            continue
        for dk, dv in split.items():
            feats = dv.column_names
        
            if "query" not in feats:
                dv = dv.map(lambda x: dict(query=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
            if "answer" not in feats:
                dv = dv.map(lambda x: dict(answer=[[]] * len(x["text"])), num_proc=1, batched=True, batch_size=4096)
            split[dk] = dv
    
    train_ds.update(train)

    validation_ds.update(validation)

    test_ds.update(test)
    

train_qna_medium_fastformer = DatasetDict(train_ds)
validation_qna_medium_fastformer = DatasetDict(validation_ds)
test_qna_medium_fastformer = DatasetDict(test_ds)

train_qna_medium_fastformer.save_to_disk("processed_datasets/train_qna_small_fastformer")
validation_qna_medium_fastformer.save_to_disk("processed_datasets/validation_qna_small_fastformer")
test_qna_medium_fastformer.save_to_disk("processed_datasets/test_qna_small_fastformer")

#####
train_fastformer = train_qna_medium_fastformer
#####

sampling_fraction = 0.7 # 0.5, 0.65, 0.75
train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_fastformer.items()}
lsum = sum(train_dataset_sampling_proba.values())
train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
probas = {k: int(v * 5_000_000) for k, v in train_dataset_sampling_proba.items()}
actual_lens = {k: len(v) for k, v in train_fastformer.items()}
compared_lens = {k: (probas[k], v) for k, v in actual_lens.items()}
compared_truth = {k: probas[k] >= v for k, v in actual_lens.items()}
print(sum(actual_lens.values()))
print(probas)
print(compared_lens)
print(all(compared_truth.values()))


def resample_dataset(ds, sample_size):
    if sample_size < len(ds):
        select_proba = sample_size / len(ds)
        return ds.filter(lambda x: random.random() <= select_proba, batch_size=16_000)
    else:
        repeats = sample_size // len(ds)
        delta = resample_dataset(ds, sample_size - (repeats * len(ds)))
        return concatenate_datasets(([ds]*repeats)+[delta])
        


def add_query_answer_columns(x):
    if "query" not in x:
        x["query"] = [[""]] * len(x["text"])
    else:
        x["query"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]),x["query"]))
    if "answer" not in x:
        x["answer"] = [[""]] * len(x["text"])
    else:
        x["answer"] = list(map(lambda y:[""] if isinstance(y, (list, tuple)) and len(y) == 0 else (y if isinstance(y, (list, tuple)) else [str(y)]), x["answer"]))
    return x

train_fastformer_resampled = DatasetDict({k: resample_dataset(train_fastformer[k], samples) for k, samples in probas.items()})
train_fastformer_resampled = DatasetDict({k: v.map(add_query_answer_columns, batched=True, batch_size=16_384, remove_columns=["label"] if "label" in v.column_names else []) for k, v in train_fastformer_resampled.items()})
dsets = list(train_fastformer_resampled.values())
keys = list(train_fastformer_resampled.keys()) 
for k, dset in zip(keys, dsets):
    if dset.features.type != dsets[0].features.type:
        print(k," :: ", dset.features.type, " :: ", dsets[0].features.type, "\n")
        dset.cast_(dsets[0].features)
train_fastformer_resampled = concatenate_datasets(list(train_fastformer_resampled.values()))

def filter_small_text(x):
    return len(x["text"].strip()) > 8 and len(x["text"].strip().split()) > 4 and x["length"] < 1024

    
train_fastformer_resampled = train_fastformer_resampled.filter(filter_small_text, batch_size=4096)
train_fastformer_resampled.save_to_disk("/home/ahemf/processed_datasets/train_qna_small_fastformer_5M")


"""

def batch_process_wiki_lingua(examples: Dict[str, List])-> Dict[str, List]:
    article: List[Dict[str, List]] = examples["article"]
    url = examples["url"]
    url = [" ".join(u.split("/")[-1].split("-")) for u in url]
    dl2: Dict[str, List[List[str]]] = {key: [item[key] for item in article] for key in article[0].keys()}

    for k, v in dl2.items():
        dl2[k] = [v2 for v1 in v for v2 in v1]
        lns = [len(v1) for v1 in v]
        
    assert len(lns) == len(url)
    assert len(set([len(v) for v in dl2.values()]))
    url = [k for u, l in zip(url, lns) for k in [u]*l]
    dl2["url"] = url
    return dl2


def ds_length_stats(ds, lbs=((0, 64), (64, 128), (128, 512), (512, 768), (768, 1024)), tokenizer=None):
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    from collections import defaultdict
    if isinstance(ds, Dataset):
        ds = DatasetDict(dict(ds=ds))

    splits = list(ds.keys())
    split_info = defaultdict(dict)
    len_info = defaultdict(dict)
    for key in splits:
        split = ds[key]
        remove_cols = list(set(split.column_names) - {'length'})
        split = split.map(lambda x: dict(length=x["length"]), batch_size=4096, batched=True, remove_columns=remove_cols)
        for lb in lbs:
            l = len(split.filter(lambda x: lb[0]<=x["length"]<lb[1]))
            split_info[key][lb] = l
            len_info[lb][key] = l
    aggregate_len_info = {ll: sum(vl.values()) for ll, vl in len_info.items()}
    return aggregate_len_info, len_info, split_info


# TODO: Run the dataset processor / collator on full dataset first instead of running it with model, with and without shuffle
# TODO: Run the same for validation datasets.
#

"""
import numpy as np

udeps_catalog = ['en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pronouns', 'en_pud',]
udeps_all = {uc: load_dataset("universal_dependencies", uc, script_version="master") for uc in udeps_catalog}
udeps_all = {k: v.filter(lambda x: isinstance(x["text"], str) and len(x["text"].split()) >= 8 and len(x["text"].split()) <= 64) for k, v in udeps_all.items()}
udeps_all = {k+"_"+vk: vv for k, v in udeps_all.items() for vk, vv in v.items()}
udeps_concat = concatenate_datasets([vv for k, v in udeps_all.items() for vk, vv in v.items()])
udeps_concat = udeps_concat.filter(lambda x: len(set(x["text"])) > 6)
udeps_concat = udeps_concat.map(lambda x: x, remove_columns=['feats', 'lemmas', 'misc', 'deps'])

def build_token_info(x):
    head = np.array(list(map(int, x['head'])))
    tokens = np.array(x['tokens'])
    deprel = np.array(x['deprel'])
    upos = np.array(x['upos'])
    xpos = np.array(x['xpos'])
    
    th = tokens[head]
    dist = head - np.arange(len(tokens))
    heads_upos =  upos[head]
    heads_xpos = xpos[head]
    heads_deprel = deprel[head]
    
    return [dict(token=token, head_token=head_token, head_dist=head_dist, token_deprel=token_deprel, head_deprel=head_deprel, token_upos=token_upos, head_upos=head_upos, token_xpos=token_xpos, head_xpos=head_xpos) for token, head_token, head_dist, token_deprel, head_deprel, token_upos, head_upos, token_xpos, head_xpos, in zip(tokens, th, dist, deprel, heads_deprel, upos, heads_upos, xpos, heads_xpos)]

"""
#
#
