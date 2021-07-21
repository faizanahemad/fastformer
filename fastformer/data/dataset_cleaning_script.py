from fastformer.utils import squeeze_after, get_time_string, recursive_op, gcd_array, clean_text, get_text_mapper
from transformers import AutoTokenizer
import re
import nltk
from typing import Dict, List
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def filter(x):
    n_tokens = len(tokenizer.tokenize(x["text"]))
    return n_tokens > 96
