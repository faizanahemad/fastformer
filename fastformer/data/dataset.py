


from transformers import AutoTokenizer, PreTrainedTokenizerFast
import numpy as np
import torch
from ..config import FastFormerConfig
from torch.nn import functional as F
import nlpaug.augmenter.char as nac

char_to_id = sorted([k for k, v in AutoTokenizer.from_pretrained("bert-base-uncased").get_vocab().items() if len(k) == 1]) + [" ", "\n"]
char_to_id = dict(zip(char_to_id, range(2, len(char_to_id) + 2)))


from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import copy

from dataclasses import dataclass

@dataclass
class TextPretrainingRegularizationConfig:
    word_masking_proba=0.15


import nltk.data
from collections import defaultdict
import re
import random


def isnumber(text):
    try:
        text = int(text)
        return True
    except:
        try:
            t = float(text)
            return True
        except:
            pass
    return False


def segment(text, n_segments, sent_detector, pad_token):
    text = re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ',text)
    sents = sent_detector.tokenize(text)
    sent_wc = list(map(lambda x: len(x.split()), sents))
    twc = len(text.split())
    segments = defaultdict(str)
    tol = 0.1
    while len(segments) < n_segments and tol < 0.9:
        segments = defaultdict(str)
        expected_wc = twc // (n_segments + tol)
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
    return " ".join(new_tokens)


def span_based_whole_word_masking(text: str, tokenizer, probability: float, vocab: list, max_span_length: int = 3) -> str:
    text = str(text)
    if probability == 0 or len(text) == 0 or len(text.split()) <= 2:
        return text
    tokens = text.split()
    new_tokens = []
    skip_next_n_words = 0
    for idx, token in enumerate(tokens):
        if skip_next_n_words > 0:
            skip_next_n_words -= 1
            continue
        prob = random.random()
        if prob < probability:
            prob /= probability
            if prob < 0.9:
                span_size = min(random.sample(range(1, max_span_length + 1), 1)[0], len(tokens) - idx)
                tks = [tokenizer.mask_token] * sum([len(tokenizer.tokenize(tokens[idx + i])) for i in range(span_size)])
                skip_next_n_words = span_size - 1
            else:
                tks = random.sample(vocab, 1)
        else:
            tks = [token]
        new_tokens.extend(tks)
    return " ".join(new_tokens)


def char_rnn_tokenize(text, tokenizer, char_to_id, **tokenizer_args):
    # Do padding myself
    tokenizer_outputs = tokenizer(text, return_offsets_mapping=True, **tokenizer_args)
    offset_mapping = tokenizer_outputs["offset_mapping"]
    offset_mapping[:, -1] -= 1
    offset_mapping = F.relu(offset_mapping)
    char_list = list(text)
    char_lists = list(map(lambda x: char_to_id.__getitem__(x.lower()), char_list))
    tokenizer_outputs["char_ids"] = char_lists
    tokenizer_outputs["char_offsets"] = offset_mapping.squeeze()
    assert tokenizer_outputs["input_ids"].shape[1] == tokenizer_args["max_length"]
    tokenizer_outputs["input_ids"] = tokenizer_outputs["input_ids"].squeeze()
    assert tokenizer_outputs["input_ids"].shape[0] == tokenizer_args["max_length"]
    tokenizer_outputs["attention_mask"] = tokenizer_outputs["attention_mask"].squeeze()
    tokenizer_outputs["token_type_ids"] = tokenizer_outputs["token_type_ids"].squeeze()
    del tokenizer_outputs["offset_mapping"]
    return tokenizer_outputs


class TokenizerDataset(Dataset):
    def __init__(self, config: FastFormerConfig, tokenizer: PreTrainedTokenizerFast,
                 char_to_id: dict, tokenizer_args: dict, dataset: Dataset, sentence_jumble_proba=0.75,
                 word_mask_proba: float = 0.15, word_noise_proba: float = 0.15, max_span_length: int = 3):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.cls_tokens = config.num_highway_cls_tokens + 1
        assert self.cls_tokens > 2
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer_args = tokenizer_args
        self.dataset = dataset
        self.char_to_id = copy.deepcopy(char_to_id)
        self.mask_segment_count = 1
        self.word_mask_proba = word_mask_proba
        self.vocab = list(tokenizer.get_vocab())
        self.max_span_length = max_span_length
        self.word_jumble_segment_count = 1
        self.word_noise_proba = word_noise_proba
        self.training = True
        self.sentence_jumble_proba = sentence_jumble_proba

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        item = self.dataset[item]
        pet_query = item["pet_query"] if "pet_query" in item else None
        pet_answer = item["pet_answer"] if "pet_answer" in item else None
        if "label" in item:
            label = item["label"]
        elif "labels" in item:
            label = item["labels"]
        else:
            label = 0

        text = item["text"]
        results = dict(labels=label)

        if self.training:
            segments = np.array(segment(text, self.cls_tokens, self.sent_detector, tokenizer.pad_token))
            assert len(segments) == self.cls_tokens
            if random.random() < self.sentence_jumble_proba:
                seg_idxs = random.sample(range(self.cls_tokens), self.cls_tokens)
            else:
                seg_idxs = list(range(self.cls_tokens))
            seg_slides = seg_idxs + seg_idxs[0:1]
            two_ordered = torch.tensor([int(s1 < s2) for s1, s2 in zip(seg_slides[:-1], seg_slides[1:])])
            seg_slides = seg_idxs + seg_idxs[0:4]
            three_ordered_dilated = torch.tensor([int(s1 < s2 < s3) for s1, s2, s3 in zip(seg_slides[0:-4:1], seg_slides[2:-2:1], seg_slides[4::1])])
            results = dict(labels=label, labels_two_sentence_order=two_ordered, labels_three_sentence_dilated_order=three_ordered_dilated,
                           labels_segment_index=torch.tensor(seg_idxs))

            segments = segments[seg_idxs]
            noised_seg_idxs = random.sample(range(self.cls_tokens), self.mask_segment_count + self.word_jumble_segment_count)
            masked_seg_idxs = noised_seg_idxs[:self.mask_segment_count]
            word_jumble_seg_idxs = noised_seg_idxs[self.mask_segment_count:]
            masked_segments = segments[masked_seg_idxs][0]
            word_jumble_segments = segments[word_jumble_seg_idxs][0]

            small_segment_tokenizer_args = dict(**self.tokenizer_args)
            small_segment_tokenizer_args["max_length"] = self.tokenizer_args["max_length"] // (self.cls_tokens - 1)
            tokenizer_outputs = tokenizer(masked_segments, return_offsets_mapping=True, **small_segment_tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results.update(dict(gap_sentence_input_ids=input_ids.squeeze(), gap_sentence_attention_mask=attention_mask.squeeze()))
            tokenizer_outputs = tokenizer(word_jumble_segments, return_offsets_mapping=True, **small_segment_tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results.update(dict(jumble_sentence_input_ids=input_ids.squeeze(), jumble_sentence_attention_mask=attention_mask.squeeze()))

            for idx in word_jumble_seg_idxs:
                seq = segments[idx].split()
                random.shuffle(seq)
                segments[idx] = " ".join(seq).strip()

            segments[masked_seg_idxs] = [" ".join(seq.split()[:len(seq.split()) // 2]) + " " + self.tokenizer.sentence_mask_token for seq in segments[masked_seg_idxs]]
            mlm_text = " ".join(segments)  # Training Labels for MLM
            if pet_query is not None:
                mlm_text = mlm_text + " " + tokenizer.sep_token + " " + pet_query
                assert pet_answer is not None
                mlm_text = mlm_text + " " + tokenizer.sep_token + " " + pet_answer
            tokenizer_outputs = tokenizer(mlm_text, return_offsets_mapping=True, **self.tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results["label_mlm_input_ids"] = input_ids.squeeze()

            for idx, seq in enumerate(segments):
                if idx in noised_seg_idxs:
                    continue
                seq = span_based_whole_word_masking(seq, self.tokenizer, self.word_mask_proba, self.vocab, self.max_span_length)
                seq = word_level_noising(seq, self.tokenizer, self.word_noise_proba)
                segments[idx] = seq

            text = " ".join(segments)
        if pet_query is not None:
            text = text + " " + tokenizer.sep_token + " " + pet_query
            if self.training:
                text = text + " " + tokenizer.sep_token + " " + " ".join([tokenizer.mask_token] * len(tokenizer.tokenize(pet_answer)))
            else:
                text = text + " " + tokenizer.sep_token + " " + tokenizer.mask_token
        inp = char_rnn_tokenize(text, self.tokenizer, self.char_to_id, **self.tokenizer_args)
        results.update(inp)
        return results

    def __len__(self):
        return len(self.dataset)


def collate_fn(samples):
    char_ids = None
    padding_index = 0
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

    print({key: [d[key].size() if isinstance(d[key], torch.Tensor) else d[key] for d in samples] for key in samples[0].keys()})
    samples = default_collate(samples)
    if char_ids is not None:
        samples["char_ids"] = char_ids
    # TODO: reduce the batch seq length to minimum required and a multiple of 16.
    for k, v in samples.items():
        if len(v.size()) < 2 or k == "char_offsets" or "label" in k or k == "token_type_ids":
            continue
        step_size = 64 if k == "char_ids" else 16
        while bool(v[:, -step_size:].sum() == 0) and v.shape[1] > step_size:
            v = v[:, :-step_size]
        required_len = int(step_size * np.ceil(v.shape[1]/step_size))
        padding = required_len - v.shape[-1]
        v = torch.cat([v, v.new(v.shape[0], padding).fill_(padding_index)], 1)
        samples[k] = v
    if "label_mlm_input_ids" in samples:
        samples['label_mlm_input_ids'] = samples['label_mlm_input_ids'][:, :samples['input_ids'].shape[1]]
    if "token_type_ids" in samples:
        samples['token_type_ids'] = samples['token_type_ids'][:, :samples['input_ids'].shape[1]]
    if "char_offsets" in samples:
        samples['char_offsets'] = samples['char_offsets'][:, :samples['input_ids'].shape[1]]
    return samples



