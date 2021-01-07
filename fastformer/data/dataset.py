from typing import List, Dict

from seaborn import load_dataset
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
from datasets import load_dataset, concatenate_datasets

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
    tokenizer_outputs["char_ids"] = char_lists[:offset_mapping.max().item()]
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
        self.word_mask_proba = word_mask_proba
        self.vocab = list(tokenizer.get_vocab())
        self.max_span_length = max_span_length
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
        pet_query = item["query"] if "query" in item else None
        pet_answer = item["answer"] if "answer" in item else None

        labels_seq = item["labels_seq"] if "labels_seq" in item else None
        labels_seq_prompt = item["labels_seq_prompt"] if "labels_seq_prompt" in item else None

        # TODO: Prompt is added at end of our Seq, labels_seq is generated from an auto-regressive head

        # pet_query = ["how many queens?"] * 8
        # pet_answer = ["eight"] * 8
        assert (pet_query is None and pet_answer is None) or (isinstance(pet_query, str) and isinstance(pet_answer, str)) or (len(pet_query) == len(pet_answer) and isinstance(pet_query, list) and isinstance(pet_answer, list))
        if isinstance(pet_query, str):
            n_queries = 1
            pet_query = [pet_query]
            pet_answer = [pet_answer]
        elif isinstance(pet_query, list):
            n_queries = len(pet_query)
        else:
            n_queries = 0
            pet_query = pet_answer = []
        if "label" in item:
            label = item["label"]
        elif "labels" in item:
            label = item["labels"]
        else:
            label = 0

        text = item["text"]
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=True, **self.tokenizer_args)
        # TODO: try one in ten words
        highway_cls_ar_input_ids, highway_cls_ar__attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs["attention_mask"].squeeze()
        results = dict(labels=label, n_pet_queries=n_queries)

        if self.training:
            segments = np.array(segment(text, self.cls_tokens, self.sent_detector, tokenizer.pad_token))
            # TODO: No jumbling for cases where more than 2 of segment is PAD
            count_pad_tokens = sum(segments == tokenizer.pad_token)
            assert len(segments) == self.cls_tokens
            if random.random() < self.sentence_jumble_proba and n_queries == 0 and count_pad_tokens <= 2:
                seg_idxs = random.sample(range(self.cls_tokens), self.cls_tokens)
            else:
                seg_idxs = list(range(self.cls_tokens))
            seg_slides = seg_idxs + seg_idxs[0:1]
            two_ordered = torch.tensor([int(s1 < s2) for s1, s2 in zip(seg_slides[:-1], seg_slides[1:])])
            seg_slides = seg_idxs + seg_idxs[0:4]
            three_ordered_dilated = torch.tensor([int(s1 < s2 < s3) for s1, s2, s3 in zip(seg_slides[0:-4:1], seg_slides[2:-2:1], seg_slides[4::1])])
            results = dict(labels=label, labels_two_sentence_order=two_ordered, labels_three_sentence_dilated_order=three_ordered_dilated,
                           labels_segment_index=torch.tensor(seg_idxs),
                           highway_cls_ar_input_ids=highway_cls_ar_input_ids, highway_cls_ar__attention_mask=highway_cls_ar__attention_mask)

            segments = segments[seg_idxs]
            masked_segments = word_jumble_segments = tokenizer.pad_token
            iters = 0
            word_jumble_seg_idxs = -1
            masked_seg_idxs = self.cls_tokens - 1
            while (masked_segments == tokenizer.pad_token or word_jumble_segments == tokenizer.pad_token) and iters <= 16:
                iters += 1
                if word_jumble_segments == tokenizer.pad_token:
                    word_jumble_seg_idxs = random.sample(list(set(list(range(self.cls_tokens))) - {masked_seg_idxs}), 1)[0]
                    word_jumble_segments = segments[word_jumble_seg_idxs]
                if masked_segments == tokenizer.pad_token or seg_idxs[masked_seg_idxs] == self.cls_tokens - 1:
                    masked_seg_idxs = random.sample(list(set(list(range(self.cls_tokens))) - {word_jumble_seg_idxs}), 1)[0]
                    masked_segments = segments[masked_seg_idxs]

            small_segment_tokenizer_args = dict(**self.tokenizer_args)
            small_segment_tokenizer_args["max_length"] = self.tokenizer_args["max_length"] // (self.cls_tokens - 1)
            tokenizer_outputs = tokenizer(masked_segments, return_offsets_mapping=True, **small_segment_tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results.update(dict(gap_sentence_input_ids=input_ids.squeeze(), gap_sentence_attention_mask=attention_mask.squeeze()))
            tokenizer_outputs = tokenizer(word_jumble_segments, return_offsets_mapping=True, **small_segment_tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results.update(dict(jumble_sentence_input_ids=input_ids.squeeze(), jumble_sentence_attention_mask=attention_mask.squeeze()))

            if n_queries == 0 and count_pad_tokens <= 2:
                seq = segments[word_jumble_seg_idxs].split()
                wj_seq1, wj_seq2 = seq[:len(seq)//4], seq[len(seq)//4:]
                random.shuffle(wj_seq2)
                seq = wj_seq1 + wj_seq2
                segments[word_jumble_seg_idxs] = " ".join(seq).strip()
                segments[masked_seg_idxs] = " ".join(masked_segments.split()[:len(masked_segments.split()) // 4]) + " " + self.tokenizer.sentence_mask_token
            else:
                word_jumble_seg_idxs = masked_seg_idxs = -1
            mlm_text = " ".join(segments)  # Training Labels for MLM
            labels_pet_text = ""
            for i, (q, a) in enumerate(zip(pet_query, pet_answer)):
                if i == 0:
                    mlm_text = mlm_text + " " + tokenizer.sep_token

                mlm_text = mlm_text + " " + getattr(tokenizer, "question_token_%s" % i) + " " + q
                mlm_text = mlm_text + " " + getattr(tokenizer, "answer_token_%s" % i) + " " + str(len(tokenizer.tokenize(a)))
                assert a is not None
                labels_pet_text += getattr(tokenizer, "question_token_%s" % i) + " " + a + " " + getattr(tokenizer, "answer_token_%s" % i)
            labels_pet_text = (labels_pet_text + " " + getattr(tokenizer, "answer_end_token")).strip()
            if n_queries > 0:
                labels_pet_text_tokenizer_args = dict(**self.tokenizer_args)
                labels_pet_text_max_length = 128
                labels_pet_text_tokenizer_args["max_length"] = labels_pet_text_max_length
                tokenizer_outputs = tokenizer(labels_pet_text, return_offsets_mapping=True, **labels_pet_text_tokenizer_args)
                input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
                results.update(dict(labels_pet_input_ids=input_ids.squeeze(),
                                    labels_pet_attention_mask=attention_mask.squeeze()))

            tokenizer_outputs = tokenizer(mlm_text, return_offsets_mapping=True, **self.tokenizer_args)
            input_ids, attention_mask = tokenizer_outputs["input_ids"], tokenizer_outputs["attention_mask"]
            results["label_mlm_input_ids"] = input_ids.squeeze()

            noised_seg_idxs = [word_jumble_seg_idxs, masked_seg_idxs]
            for idx, seq in enumerate(segments):
                if idx in noised_seg_idxs or n_queries != 0:
                    continue
                seq = span_based_whole_word_masking(seq, self.tokenizer, self.word_mask_proba, self.vocab, self.max_span_length)
                seq = word_level_noising(seq, self.tokenizer, self.word_noise_proba)
                segments[idx] = seq

            text = " ".join(segments)

        for i, (q, a) in enumerate(zip(pet_query, pet_answer)):
            if i == 0:
                text = text + " " + tokenizer.sep_token

            text = text + " " + getattr(tokenizer, "question_token_%s" % i) + " " + q
            text = text + " " + getattr(tokenizer, "answer_token_%s" % i) + " " + tokenizer.mask_token

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
        if len(v.size()) < 2 or k == "char_offsets" or k == "token_type_ids":
            continue
        if "label" in k and (k not in ["labels_pet_input_ids", "labels_pet_attention_mask",]):
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
    wikihow = load_dataset("wikihow", 'all')
    wikihow = load_dataset("wikihow", 'sep')
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
    search_qa = load_dataset("search_qa", 'train_test_val')
    hotpot_qa = load_dataset("hotpot_qa", 'distractor')
    hotpot_qa = load_dataset("hotpot_qa", 'fullwiki')
    inquisitive_qg = load_dataset("inquisitive_qg", script_version="master")
    squad = load_dataset("squad")
    squad_v2 = load_dataset("squad_v2")
    squad_adversarial = load_dataset("squad_adversarial", 'AddSent', script_version="master")
    ropes = load_dataset("ropes")

    tweet_qa = load_dataset("tweet_qa", script_version="master")
    trivia_qa = load_dataset("trivia_qa", "rc")
    wiki_qa = load_dataset("wiki_qa")  # Is answer correct / relevant or not
    narrativeqa = load_dataset("narrativeqa", script_version="master")
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


def get_matching_mapper(text_cols, query_texts, query_match_cols, query_text_mlm=tuple(), query_match_cols_mlm=tuple(), total_tokens=768, tokenizer=None):
    asep = " [ANSWER_OPTION_SEP] "
    aoptbegin = "[ANSWER_OPTION_BEGIN] "
    aoptend = " [ANSWER_OPTION_END]"
    word_choice_1 = ["select", "what is", "choose"]
    word_choice_2 = ["appropriate", "correct", "right"]

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
        for qtxt, qmc in zip(query_texts, query_match_cols):
            cq_query = []
            cq_answer = []
            query_answers = np.array(list(map(clean_text, examples[qmc])))
            shuffled_idxs = random.sample(range(len(query_answers)), len(query_answers))
            query_answers_shuffle = list(query_answers[shuffled_idxs])
            query_answers_shuffle_type_1 = aoptbegin + asep.join(query_answers_shuffle) + aoptend
            query_answers_shuffle_type_2 = aoptbegin + asep.join([str(i+1)+". " + a for i, a in enumerate(query_answers_shuffle)]) + aoptend
            for idx in range(len(one_texts)):
                aidx = shuffled_idxs.index(idx)
                atext = query_answers[idx]
                atext_len = len(atext.split())
                assert query_answers[shuffled_idxs[aidx]] == atext == query_answers_shuffle[aidx]
                rnd = random.random()

                if rnd < 0.25 and atext_len <= 4:
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

        for qtxt_mlm, qmc_mlm in zip(query_text_mlm, query_match_cols_mlm):
            qt = qtxt_mlm
            if isinstance(qtxt_mlm, (list, tuple)):
                qt = random.sample(qtxt_mlm, 1)[0]

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


"""
import datasets
import re
import numpy as np
import random
from typing import List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import os
os.cpu_count()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bookcorpusopen512 = bookcorpusopen.map(get_text_mapper(["title", "text"], 512, tokenizer, sent_detector), batched=True, remove_columns=["title"], num_proc=24)


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


reddit_qna = reddit.map(get_matching_mapper(["normalizedBody"], ["summary", "topic"], ["summary", "subreddit"], 512, tokenizer), batched=True, remove_columns=["author", "body", "content", "normalizedBody", "subreddit", "subreddit_id", "summary", "id"], num_proc=16, batch_size=16)
reddit_qna.save_to_disk("/home/ahemf/processed_datasets/reddit_qna")

bookcorpusopen = bookcorpusopen.map(lambda x: dict(title=" ".join(x["title"].replace(".epub","").replace(".txt","").split('-')), text=x["text"][4096:]), num_proc=8)
bookcorpusopen_qna = bookcorpusopen.map(get_matching_mapper(["text"], ["title"], ["title",], 768, tokenizer), batched=True, remove_columns=["title"], num_proc=16, batch_size=16)
bookcorpusopen_qna.save_to_disk("/home/ahemf/processed_datasets/bookcorpusopen_qna")

wikipedia_qna = wikipedia.map(get_matching_mapper(["text"], ["title"], ["title",], 768, tokenizer), batched=True, remove_columns=["title"], num_proc=16, batch_size=16)
wikipedia_qna.save_to_disk("/home/ahemf/processed_datasets/wikipedia_qna")

amazon_polarity = amazon_polarity.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
amazon_polarity_qna = amazon_polarity.map(get_matching_mapper(["content"], ["title", "sentiment"], ["title", "sentiment"], 768, tokenizer), batched=True, remove_columns=["title", "sentiment", "content"], num_proc=16, batch_size=16)
amazon_polarity_qna.save_to_disk("/home/ahemf/processed_datasets/amazon_polarity_qna")

yahoo_answers_qa_qna = yahoo_answers_qa.map(get_matching_mapper(["answer"], ["question", "category"], ["question", "main_category"], 768, tokenizer), batched=True, remove_columns=["id", 'question', 'answer', 'nbestanswers', 'main_category'], num_proc=16, batch_size=16)
yahoo_answers_qa_qna.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_qa_qna")

yahoo_answers_topics_qna = yahoo_answers_topics.map(get_matching_mapper(["best_answer"], ["question"], ["question_title"], 768, tokenizer), batched=True, remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'], num_proc=16, batch_size=16)
yahoo_answers_topics_qna.save_to_disk("/home/ahemf/processed_datasets/yahoo_answers_topics_qna")

reuters = reuters.map(lambda x: dict(title=x["title"].replace('&lt;', ' ').replace('>', ' ')), num_proc=16, batch_size=16)
reuters_qna = reuters.map(get_matching_mapper(["text"], ["title"], ["title"], 768, tokenizer), batched=True, remove_columns=['topics', 'lewis_split', 'cgis_split', 'old_id', 'new_id', 'places', 'people', 'orgs', 'exchanges', 'date', 'title'], num_proc=16, batch_size=16)
reuters_qna.save_to_disk("/home/ahemf/processed_datasets/reuters_qna")

ohsumed_qna = ohsumed.map(get_matching_mapper(["abstract"], ["title"], ["title"], 896, tokenizer), batched=True, remove_columns=['seq_id', 'medline_ui', 'mesh_terms', 'title', 'publication_type', 'abstract', 'author', 'source'], num_proc=16, batch_size=16)
ohsumed_qna.save_to_disk("/home/ahemf/processed_datasets/ohsumed_qna")

xsum_qna = xsum.map(get_matching_mapper(["document"], ["summary"], ["summary"], 1024, tokenizer), batched=True, remove_columns=['document', 'summary', 'id'], num_proc=16, batch_size=16)
xsum_qna.save_to_disk("/home/ahemf/processed_datasets/xsum_qna")

eli5_qna = eli5.map(get_matching_mapper([["answers", "text"]], ["title"], ["title"], 896, tokenizer), batched=True, remove_columns=['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls', 'answers_urls'], num_proc=16, batch_size=16)
eli5_qna.save_to_disk("/home/ahemf/processed_datasets/eli5_qna")

cnn_dailymail_qna = cnn_dailymail.map(get_matching_mapper(["article"], ["highlights"], ["highlights"], 1024, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['article', 'highlights', 'id'])
cnn_dailymail_qna.save_to_disk("/home/ahemf/processed_datasets/cnn_dailymail_qna")


yelp_polarity = yelp_polarity.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
yelp_polarity_qna = yelp_polarity.map(get_matching_mapper(["text"], [], [], ["Predict the correct sentiment between positive and negative"], ["sentiment"], 512, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['sentiment'])
yelp_polarity_qna.save_to_disk("/home/ahemf/processed_datasets/yelp_polarity_qna")

amazon_reviews_multi_qna = amazon_reviews_multi.map(get_matching_mapper(["review_body"], ["title"], ["review_title"], [["Predict the review rating", "What is the rating suggested by the review on a scale of 1 to 5?"]], ["stars"], 768, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'])
amazon_reviews_multi_qna.save_to_disk("/home/ahemf/processed_datasets/amazon_reviews_multi_qna")

app_reviews_qna = app_reviews.map(get_matching_mapper(["review"], [], [], [["Predict the review rating", "What is the rating suggested by the review on a scale of 1 to 5?"]], ["stars"], 768, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['package_name', 'review', 'date', 'star'])
app_reviews_qna.save_to_disk("/home/ahemf/processed_datasets/app_reviews_qna")

imdb = imdb.map(lambda x: dict(sentiment="positive" if x['label']==1 else 'negative'), remove_columns=["label"], num_proc=8)
imdb_qna = imdb.map(get_matching_mapper(["text"], [], [], ["Predict the correct sentiment between positive and negative"], ["sentiment"], 512, tokenizer), batched=True, num_proc=16, batch_size=16, remove_columns=['sentiment'])
imdb_qna.save_to_disk("/home/ahemf/processed_datasets/imdb_qna")

"""




