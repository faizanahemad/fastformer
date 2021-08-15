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
from torch.nn.parallel.replicate import replicate
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda")
sbert = SentenceTransformer("paraphrase-mpnet-base-v2").eval().to(device)
dset = Dataset.load_from_disk("/home/ahemf/processed_datasets/dsets_448")
with torch.no_grad():
    dset_sbert = dset.map(lambda x: dict(sbert=sbert.encode(x["text"])), batched=True, batch_size=128)
dset_sbert.save_to_disk("/home/ahemf/processed_datasets/dsets_448_sbert")

########################################################################################################
# https://huggingface.co/transformers/perplexity.html
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn import CrossEntropyLoss, functional as F, DataParallel
from torch import nn
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda:0")


class DataParallel(nn.DataParallel):
    def __init__(self, module):
        super(DataParallel, self).__init__(module)
        self.replicas = None

    def replicate(self, module, device_ids):
        if self.replicas is None:
            from torch.nn.parallel.replicate import replicate
            self.replicas = replicate(module, device_ids, not torch.is_grad_enabled())
        return self.replicas

dset_sbert = Dataset.load_from_disk("/home/ahemf/processed_datasets/dsets_448_sbert")
model_id = "distilgpt2"  # 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).eval()
max_length = 256
stride = max_length
for p in model.parameters():
    p.requires_grad = False
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

model.to(device)
for p in model.parameters():
    p.requires_grad = False
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


def perplexity(x, device):
    # Batch Size = 1 only because loss from GPT2LMHeadModel gives one cross entropy value for whole input
    encoded = tokenizer.batch_encode_plus(x["text"], padding=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"][:, :1024].to(device)
    attention_mask = encoded["attention_mask"][:, :1024].to(device)
    lengths = attention_mask.sum(1)

    lls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        cur_input_ids = input_ids[:, begin_loc:end_loc].contiguous()
        cur_attention_mask = attention_mask[:, begin_loc:end_loc].contiguous()
        cur_input_ids = cur_input_ids
        cur_attention_mask = cur_attention_mask
        target_ids = cur_input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[:, :2] = -100
        target_ids[target_ids == tokenizer.eos_token_id] = -100
        with torch.no_grad():
            outputs = model(cur_input_ids, attention_mask=cur_attention_mask, labels=target_ids)
            lm_logits = outputs["logits"].detach()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(shift_logits.size(0), shift_logits.size(1)).mean(1).squeeze().unsqueeze(-1)
            log_likelihood = loss * trg_len  # B x 1
        lls.append(log_likelihood)

    lls = torch.cat(lls, 1).squeeze()
    ppl = torch.exp(lls.sum(1) / lengths)
    return dict(perplexity=ppl.tolist())


def perplexity_without_device(x):
    return perplexity(x, device)

with torch.no_grad():
    print(perplexity_without_device(dset_sbert[0:32]))


model.zero_grad(set_to_none=True)
with torch.no_grad():
    dset_sbert = dset_sbert.map(perplexity_without_device, batched=True, batch_size=32)

#####################################################################################################
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn import CrossEntropyLoss, functional as F, DataParallel
from torch import nn
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda:0")


def perplexity(x, device):
    # Batch Size = 1 only because loss from GPT2LMHeadModel gives one cross entropy value for whole input
    encoded = tokenizer.batch_encode_plus(x["text"], padding=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"][:, :1024].to(device)
    attention_mask = encoded["attention_mask"][:, :1024].to(device)
    lengths = attention_mask.sum(1)

    lls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        cur_input_ids = input_ids[:, begin_loc:end_loc].contiguous()
        cur_attention_mask = attention_mask[:, begin_loc:end_loc].contiguous()
        cur_input_ids = cur_input_ids
        cur_attention_mask = cur_attention_mask
        target_ids = cur_input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[:, :2] = -100
        target_ids[target_ids == tokenizer.eos_token_id] = -100
        with torch.no_grad():
            outputs = model(cur_input_ids, attention_mask=cur_attention_mask, labels=target_ids)
            lm_logits = outputs["logits"].detach()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).reshape(shift_logits.size(0), shift_logits.size(1)).mean(1).squeeze().unsqueeze(-1)
            log_likelihood = loss * trg_len  # B x 1
        lls.append(log_likelihood)

    lls = torch.cat(lls, 1).squeeze()
    ppl = torch.exp(lls.sum(1) / lengths)
    return dict(perplexity=ppl.tolist())

dset_sbert = Dataset.load_from_disk("/home/ahemf/processed_datasets/dsets_448_sbert")
model_id = "distilgpt2"  # 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id).eval()
max_length = 512
stride = max_length
for p in model.parameters():
    p.requires_grad = False

tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

from multiprocess.pool import Pool
device = []
with Pool(8) as p:
    def mapper(x, device_id):
        import torch
        if len(device) == 0:
            device.append(torch.device("cuda:%s" % device_id))
        model.to(device[0])
        return perplexity(x, device[0])

    def forkjoin(x):
        texts = x["text"]
        csz = int(np.ceil(len(texts) / 8))
        chunks = [dict(text=texts[i: i+csz]) for i in range(0, len(texts), csz)]
        assert len(chunks) == 8
        perplexities = [ppl for r in p.starmap(mapper, list(zip(chunks, range(8)))) for ppl in r["perplexity"]]
        return dict(perplexity=perplexities)

    with torch.no_grad():
        dset_sbert = dset_sbert.map(forkjoin, batched=True, batch_size=128)


#####################################################################################################################
from collections import Counter
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import os
import numpy as np
from multiprocess.pool import Pool
os.environ['TOKENIZERS_PARALLELISM'] = "true"
dset = Dataset.load_from_disk("/home/ahemf/processed/dsets_448")
cpu_count = os.cpu_count()
overall_counts = Counter()
# dsets_tokenized = dset.map(lambda x: dict(tokens=tokenizer.batch_encode_plus(x["text"], add_special_tokens=False, max_length=4096, padding=False)["input_ids"]), batched=True, batch_size=1024)


def term_frequency_builder(tokens):
    raw_counts = Counter(tokens)
    # distinct_tokens = list(raw_counts.keys())
    mc = raw_counts.most_common()[0]
    most_common_count = mc[1]
    # most_common_token = mc[0]
    # raw_counts = {str(k): v for k, v in raw_counts.items()}
    tf = {str(k): 0.25 + 0.75 * (v / most_common_count) for k, v in raw_counts.items()}
    return tf


tokenizer_module = []

def create_tokenizer():
    from transformers import RobertaTokenizerFast
    return RobertaTokenizerFast.from_pretrained("roberta-base")

with Pool(cpu_count) as p:

    def mapping(x):
        if len(tokenizer_module) == 0:
            tokenizer_module.append(create_tokenizer())
        tokens = tokenizer_module[0].batch_encode_plus(x["text"], add_special_tokens=False, max_length=4096, padding=False)["input_ids"]
        tf = [term_frequency_builder(tk) for tk in tokens]
        return tf

    def batch_term_frequency_builder(x):
        texts = x["text"]
        csz = int(np.ceil(len(texts) / cpu_count))
        chunks = [dict(text=texts[i: i + csz]) for i in range(0, len(texts), csz)]
        tf = [t for r in p.map(mapping, chunks) for t in r]
        overall_counts.update([k for t in tf for k in t.keys()])
        return dict(tf=tf)

    dsets_tokenized = dset.map(batch_term_frequency_builder, batched=True, batch_size=2048)



