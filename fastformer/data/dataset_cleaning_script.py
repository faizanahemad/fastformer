from fastformer.utils import squeeze_after, get_time_string, recursive_op, gcd_array, clean_text, get_text_mapper
from transformers import AutoTokenizer
import re
import nltk
from typing import Dict, List
from datasets import load_dataset
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


dataset = load_dataset("oscar", language="en")
dataset = load_dataset("c4", 'en.noblocklist', script_version="master")
dataset = load_dataset("c4", 'en', script_version="master")
mappr = get_text_mapper(["text"], 500, keep_above=384)
dataset_filtered = dataset.map(mappr, batched=True, batch_size=128, remove_columns=["id"], num_proc=32)


def filter(x):
    n_tokens = len(tokenizer.tokenize(x["text"]))
    return n_tokens > 96
