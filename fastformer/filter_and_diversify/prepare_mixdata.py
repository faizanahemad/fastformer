from datasets import load_dataset, concatenate_datasets
import re
from fastformer.utils import clean_text
import os
from fastformer.utils import get_filter_mapper

wiki = load_dataset("wikipedia", "20200501.en")
reddit = load_dataset("reddit")
openwebtext = load_dataset("openwebtext")
bookcorpus = load_dataset("bookcorpus")
yahoo_answers_topics = load_dataset("yahoo_answers_topics")
cc_news = load_dataset('cc_news', split="train")
newsroom = load_dataset("newsroom", data_dir="/home/ahemf/release")
wikihow = load_dataset("wikihow", "all", data_dir="/home/ahemf")

wiki_mapped = wiki.map(lambda x: dict(text=[tt+". "+tx for tt, tx in zip(x["title"], x["text"])]), batched=True, batch_size=256)
wiki_mapped = wiki_mapped.map(lambda x: dict(text=list(map(lambda y: re.sub(r'(?<=[.,;!?])(?=[^\s0-9])', ' ', y.replace("\n\n", " ").replace("\n\n", " ")), x["text"]))),  batched=True, batch_size=256, remove_columns=["title"])
reddit_mapped = reddit.map(lambda x: dict(text=[tt+". "+tx for tt, tx in zip(x["summary"], x["normalizedBody"])]), batched=True, batch_size=256, remove_columns=['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id', 'content', 'summary'])
reddit_mapped = reddit_mapped.map(lambda x: dict(text=list(map(lambda y: clean_text(y), x["text"]))),  batched=True, batch_size=256)
openwebtext_mapped = openwebtext.map(lambda x: dict(text=list(map(lambda y: clean_text(y), x["text"]))),  batched=True, batch_size=256)
bookcorpus_mapped = bookcorpus.map(lambda x: dict(text=list(map(lambda y: clean_text(y), x["text"]))),  batched=True, batch_size=256)
yahoo_answers_topics_mapped = yahoo_answers_topics.map(lambda x: dict(text=[clean_text(tt+" "+tx+" "+bx) for tt, tx, bx in zip(x["question_title"], x["question_content"], x["best_answer"])]), batched=True, batch_size=256, remove_columns=['id', 'topic', 'question_title', 'question_content', 'best_answer'])
cc_news_mapped = cc_news.map(lambda x: dict(text=[clean_text(tt+" "+tx+" "+bx) for tt, tx, bx in zip(x["title"], x["description"], x["text"])]), batched=True, batch_size=256, remove_columns=['title', 'domain', 'date', 'description', 'url', 'image_url'])
newsroom_mapped = newsroom.map(lambda x: dict(text=[clean_text(tt+" "+tx+" "+bx) for tt, tx, bx in zip(x["title"], x["summary"], x["text"])]), batched=True, batch_size=256, remove_columns=['summary', 'title', 'url', 'date', 'density_bin', 'coverage_bin', 'compression_bin', 'density','coverage', 'compression'])
wikihow_mapped = wikihow.map(lambda x: dict(text=[clean_text(tt+" "+tx+" "+bx) for tt, tx, bx in zip(x["title"], x["headline"], x["text"])]), batched=True, batch_size=256, remove_columns=['headline', 'title'])

wiki_mapped = wiki_mapped["train"]
reddit_mapped = reddit_mapped["train"]
openwebtext_mapped = openwebtext_mapped["train"]
bookcorpus_mapped = bookcorpus_mapped["train"]
yahoo_answers_topics_mapped = concatenate_datasets((yahoo_answers_topics_mapped["train"], yahoo_answers_topics_mapped["test"]))
newsroom_mapped = concatenate_datasets((newsroom_mapped["train"], newsroom_mapped["validation"], newsroom_mapped["test"]))
wikihow_mapped = concatenate_datasets((wikihow_mapped["train"], wikihow_mapped["validation"], wikihow_mapped["test"]))


wiki_mapped = wiki_mapped.map(lambda x: dict(dataset=["wikipedia"] * len(list(x.values())[0])), batched=True, batch_size=256)
reddit_mapped = reddit_mapped.map(lambda x: dict(dataset=["reddit"] * len(list(x.values())[0])), batched=True, batch_size=256)
openwebtext_mapped = openwebtext_mapped.map(lambda x: dict(dataset=["openwebtext"] * len(list(x.values())[0])), batched=True, batch_size=256)
bookcorpus_mapped = bookcorpus_mapped.map(lambda x: dict(dataset=["bookcorpus"] * len(list(x.values())[0])), batched=True, batch_size=256)
yahoo_answers_topics_mapped = yahoo_answers_topics_mapped.map(lambda x: dict(dataset=["yahoo_answers_topics"] * len(list(x.values())[0])), batched=True, batch_size=256)
cc_news_mapped = cc_news_mapped.map(lambda x: dict(dataset=["cc_news"] * len(list(x.values())[0])), batched=True, batch_size=256)
newsroom_mapped = newsroom_mapped.map(lambda x: dict(dataset=["newsroom"] * len(list(x.values())[0])), batched=True, batch_size=256)
wikihow_mapped = wikihow_mapped.map(lambda x: dict(dataset=["wikihow"] * len(list(x.values())[0])), batched=True, batch_size=256)

dsets = [wiki_mapped, reddit_mapped, openwebtext_mapped, bookcorpus_mapped, yahoo_answers_topics_mapped,
         cc_news_mapped, newsroom_mapped, wikihow_mapped]
dsets = concatenate_datasets(dsets)
dsets.save_to_disk("/home/ahemf/processed/dsets")

fmap = get_filter_mapper(128)
dsets_128 = dsets.map(fmap, batched=True, batch_size=256,)
dsets_128.save_to_disk("/home/ahemf/processed/dsets_128")

fmap = get_filter_mapper(256)
dsets_256 = dsets_128.map(fmap, batched=True, batch_size=256,)
dsets_256.save_to_disk("/home/ahemf/processed/dsets_256")


fmap = get_filter_mapper(448)
dsets_448 = dsets_256.map(fmap, batched=True, batch_size=256,)
dsets_448.save_to_disk("/home/ahemf/processed/dseipythts_448")

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda")
sbert = SentenceTransformer("paraphrase-mpnet-base-v2").eval().to(device)
dset = Dataset.load_from_disk("/home/ahemf/processed_datasets/dsets_448")
with torch.no_grad():
    dset_sbert = dset.map(lambda x: dict(sbert=sbert.encode(x["text"])), batched=True, batch_size=128)
dset_sbert.save_to_disk("/home/ahemf/processed_datasets/dsets_448_sbert")

# https://huggingface.co/transformers/perplexity.html
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn import CrossEntropyLoss as CELoss, functional as F
from torch import nn
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda")


class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int = -100) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        return loss.mean(1)

dset_sbert = Dataset.load_from_disk("/home/ahemf/processed_datasets/dsets_448_sbert")
model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device).eval()
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

max_length = model.config.n_positions
stride = 512
tokenizer.pad_token = tokenizer.eos_token
max_length = 128
stride = 128


def perplexity(x):
    # Batch Size = 1 only because loss from GPT2LMHeadModel gives one cross entropy value for whole input
    encoded = tokenizer.batch_encode_plus(x["text"], padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(1)
    input_ids[input_ids == tokenizer.eos_token_id] = -100

    lls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        cur_input_ids = input_ids[:, begin_loc:end_loc].contiguous()
        cur_attention_mask = attention_mask[:, begin_loc:end_loc].contiguous()
        cur_input_ids = cur_input_ids.to(device)
        cur_attention_mask = cur_attention_mask.to(device)
        target_ids = cur_input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[:, :2] = -100
        with torch.no_grad():
            outputs = model(cur_input_ids, attention_mask=cur_attention_mask, labels=target_ids)
            log_likelihood = outputs[0] * trg_len  # B x 1
        lls.append(log_likelihood)

    lls = torch.stack(lls, 0)
    ppl = torch.exp(lls.sum().cpu() / lengths)
    return dict(perplexity=ppl.tolist())

perplexity(dset_sbert[0:1])
model.zero_grad(set_to_none=True)
with torch.no_grad():
    dset_sbert = dset_sbert.map(perplexity, batched=True, batch_size=1)








from transformers import AutoTokenizer
from collections import Counter
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
dset = Dataset.load_from_disk("/home/ahemf/processed/dsets_448")
# dsets_tokenized = dset.map(lambda x: dict(tokens=tokenizer.batch_encode_plus(x["text"], add_special_tokens=False, max_length=4096, padding=False)["input_ids"]), batched=True, batch_size=1024)


def term_frequency_builder(tokens):
    raw_counts = Counter(tokens)
    distinct_tokens = list(raw_counts.keys())
    mc = raw_counts.most_common()[0]
    most_common_count = mc[1]
    most_common_token = mc[0]
    WK = 0.25
    raw_counts = {str(k): v for k, v in raw_counts.items()}
    tf = {k: WK + (1 - WK) * (v / most_common_count) for k, v in raw_counts.items()}
    return dict(tf=tf, raw_counts=raw_counts, most_common_count=most_common_count, most_common_token=most_common_token, distinct_tokens=distinct_tokens)


def batch_term_frequency_builder(x):
    tokens=tokenizer.batch_encode_plus(x["text"], add_special_tokens=False, max_length=4096, padding=False)["input_ids"]
    ld = [term_frequency_builder(tk) for tk in tokens]
    dl = {key: [item[key] for item in ld] for key in ld[0].keys()}
    dl["tokens"] = tokens
    return dl

dsets_tokenized = dset.map(batch_term_frequency_builder, batched=True, batch_size=1024, num_proc=32)



