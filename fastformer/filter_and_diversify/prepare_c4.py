from datasets import concatenate_datasets, DatasetDict, load_dataset, Dataset
from datasets import load_dataset
import re
import os
from fastformer.utils import get_filter_mapper
from fastformer.utils import clean_text
dataset = load_dataset("c4", 'en.noblocklist', script_version="master")
fmap = get_filter_mapper(256)
dataset_filtered = dataset.map(fmap, batched=True, batch_size=256, remove_columns=['timestamp'])
dataset_filtered = dataset_filtered.map(lambda x: dict(text=list(map(lambda y: clean_text(y), x["text"]))),  batched=True, batch_size=256)
dataset_filtered.save_to_disk("/home/ahemf/processed/c4_256")

fmap = get_filter_mapper(448)
dataset_448 = dataset_filtered.map(fmap, batched=True, batch_size=256, remove_columns=['timestamp'])
dataset_448 = dataset_448.map(lambda x: dict(text=list(map(lambda y: clean_text(y), x["text"]))),  batched=True, batch_size=256)
dataset_448.save_to_disk("/home/ahemf/processed/c4_448")


c4 = DatasetDict.load_from_disk("/home/ahemf/processed/c4_448")
dsets = Dataset.load_from_disk("/home/ahemf/processed/dsets_448")

c4['train'] = c4['train'].add_column('dataset', ['c4'] * len(c4['train']))
c4['train'] = c4['train'].remove_columns(['url', 'timestamp'])
c4['validation'] = c4['validation'].remove_columns(['url', 'timestamp'])
c4['validation'] = c4['validation'].add_column('dataset', ['c4'] * len(c4['validation']))

dataset_col = dsets['dataset']
dsets = dsets.remove_columns(["dataset"])
dsets = dsets.add_column("dataset", dataset_col)

c4["train"] = concatenate_datasets([c4["train"], dsets])
c4["train"].save_to_disk("/home/ahemf/processed/c4_extended")

c4 = Dataset.load_from_disk("/home/ahemf/processed/c4_extended")

###################################################################
## TF-IDF
###################################################################

from collections import Counter
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import os
import numpy as np
import re
from multiprocess.pool import Pool
os.environ['TOKENIZERS_PARALLELISM'] = "true"
c4 = Dataset.load_from_disk("/home/ahemf/processed/c4_extended")
cpu_count = os.cpu_count()
overall_counts = Counter()
# dsets_tokenized = dset.map(lambda x: dict(tokens=tokenizer.batch_encode_plus(x["text"], add_special_tokens=False, max_length=4096, padding=False)["input_ids"]), batched=True, batch_size=1024)


def term_frequency_builder(tokens):
    raw_counts = Counter(tokens)
    mc = raw_counts.most_common()[0]
    most_common_count = mc[1]
    tf = [[str(k), str(0.5 + 0.5 * (v / most_common_count))] for k, v in raw_counts.items()]
    return tf


with Pool(cpu_count) as p:

    def mapping(x):
        tokens = [[re.sub(r'[^\s0-9a-zA-Z]', ' ', w).strip() for w in t.split()] for t in x["text"]]
        tokens = [[w for w in t if len(w) >= 2] for t in tokens]
        tf = [term_frequency_builder(tk) if len(tk) > 0 else [["", str(1.0)]] for tk in tokens]
        return tf

    def batch_term_frequency_builder(x):
        texts = x["text"]
        csz = int(np.ceil(len(texts) / cpu_count))
        chunks = [dict(text=texts[i: i + csz]) for i in range(0, len(texts), csz)]
        tf = [t for r in p.map(mapping, chunks) for t in r]
        overall_counts.update([k for t in tf for k, _ in t])
        return dict(tf=tf)

    c4_tokenized = c4.map(batch_term_frequency_builder, batched=True, batch_size=2048)

overall_counts = {k: v for k, v in overall_counts.items()}
import pickle
with open('overall_counts.pickle', 'wb') as handle:
    pickle.dump(overall_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

count_buckets = {k:int(np.cbrt(v))+1 for k, v in overall_counts.items()}

from collections import defaultdict
freq2token = defaultdict(list)

with open('overall_counts.pickle', 'rb') as handle:
    overall_counts = pickle.load(handle)

overall_counts = {k: v for k, v in overall_counts.items() if v >= 5}
n_docs = len(c4_tokenized)
log_docs = np.log2(n_docs)
idf = {k: log_docs - np.log2(1 + v) + 1 for k, v in overall_counts.items()}

with open('idf.pickle', 'wb') as handle:
    pickle.dump(idf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('count_buckets.pickle', 'wb') as handle:
    pickle.dump(count_buckets, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('overall_counts.pickle', 'rb') as handle:
    oc = pickle.load(handle)
with open('idf.pickle', 'rb') as handle:
    idf = pickle.load(handle)


def tfidf_one(tf):
    tf = [[t, (float(v)*idf[t]) if t in idf else float(v)] for t, v in tf]
    v_only = sorted([v for t, v in tf], reverse=True)
    average = np.mean(v_only)
    top_16 = np.mean(v_only[:16])
    top_128 = np.mean(v_only[:128])
    truncated_average = np.mean(v_only[8:-8])
    truncated_average = average if np.isnan(truncated_average) else truncated_average
    return top_16, top_128, average, truncated_average


with Pool(cpu_count) as p:
    def tfidf_many(tf):
        return [tfidf_one(t) for t in tf]


    def tfidf_batch(x):
        tf = x["tf"]
        # tfidf = [tfidf_one(t) for t in tf]
        csz = int(np.ceil(len(tf) / cpu_count))
        chunks = [tf[i: i + csz] for i in range(0, len(tf), csz)]
        tfidf = [t for r in p.map(tfidf_many, chunks) for t in r]
        top_k_16, top_k_128, average, truncated_average = zip(*tfidf)
        return dict(tfidf_top_k_16=list(top_k_16), tfidf_top_k_128=list(top_k_128), tfidf_average=list(average), tfidf_truncated_average=list(truncated_average))
    print(tfidf_batch(c4_tokenized[0:128]))
    c4_tokenized = c4_tokenized.map(tfidf_batch, batched=True, batch_size=2048)

def tfidf_batch(x):
    tfidf = [np.mean([float(v)*idf.get(t, 1) for t, v in tf]) for tf in x["tf"]]
    return dict(tfidf_average=tfidf)
print(tfidf_batch(c4_tokenized[0:32]))
c4_tokenized = c4_tokenized.map(tfidf_batch, batched=True, batch_size=256)

c4_tokenized = c4_tokenized.add_column("identifier", list(range(1, len(c4_tokenized)+1)))
c4_tokenized = c4_tokenized.remove_columns(["tf"])
c4_tokenized.save_to_disk("/home/ahemf/processed/c4_extended")


###################################################################
## BiGram-TF-IDF
###################################################################

from collections import Counter
from transformers import AutoTokenizer, AutoModel, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import os
import numpy as np
import re
import gc
from multiprocess.pool import Pool
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
import nltk
import time
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
os.environ['TOKENIZERS_PARALLELISM'] = "true"
c4 = Dataset.load_from_disk("/home/ahemf/processed/c4_extended")
cpu_count = os.cpu_count() // 2
overall_counts = [Counter()]
overall_counts_faster = [Counter()]
check_before = [0]

def term_frequency_builder(tokens):
    raw_counts = Counter(tokens)
    most_common_count = raw_counts.most_common()[0][1]
    tf = [[str(k), str(0.5 + 0.5 * (v / most_common_count))] for k, v in raw_counts.items()]
    return tf


with Pool(cpu_count) as p:

    def mapping(x):
        tokens = [[re.sub(r'[^\s0-9a-zA-Z\']', ' ', w).strip() for w in t.split()] for t in x]
        tokens = [[w1+" "+w2 for w1, w2 in zip(t[:-1], t[1:]) if len(w1) >= 2 and len(w2) >= 2 and w1 not in stop_words and w2 not in stop_words] for t in tokens]
        tf = [term_frequency_builder(tk) if len(tk) > 0 else [["", "1.0"]] for tk in tokens]
        return tf

    def batch_term_frequency_builder(x):
        texts = x["text"]
        csz = int(np.ceil(len(texts) / cpu_count))
        chunks = [texts[i: i + csz] for i in range(0, len(texts), csz)]
        tf = [t for r in p.map(mapping, chunks) for t in r]
        overall_counts_faster.append(Counter([k for t in tf for k, _ in t]))

        if len(overall_counts_faster) > 128:
            collect_start = time.time()
            full_counter = None
            overalls = []
            while len(overall_counts_faster) > 0:
                ctr = overall_counts_faster.pop()
                if len(overall_counts_faster) > 0:
                    ctr.update(overall_counts_faster.pop())
                if len(overall_counts_faster) > 0:
                    ctr.update(overall_counts_faster.pop())
                if len(overall_counts_faster) > 0:
                    ctr.update(overall_counts_faster.pop())
                overalls.append(ctr)
            while len(overalls) > 0:
                ctr = overalls.pop()
                if len(overalls) > 0:
                    ctr.update(overalls.pop())
                if len(overalls) > 0:
                    ctr.update(overalls.pop())
                if len(overalls) > 0:
                    ctr.update(overalls.pop())
                if full_counter is None:
                    full_counter = ctr
                else:
                    full_counter.update(ctr)
            overall_counts[0].update(full_counter)
            del full_counter
            del overalls
            pre_len = len(overall_counts[0])
            ones = 0
            post_len = pre_len
            deletions = post_len
            if pre_len > 2 ** 26:
                ones = dict([(k, v) for k, v in overall_counts[0].items() if v > 1])
                overall_counts[0] = Counter(ones)
                ones = len(ones)
                post_len = len(overall_counts[0])
                deletions = post_len
                if post_len > 2 ** 24:
                    deletions = dict([(k, v) for k, v in overall_counts[0].most_common()[2 ** 24:]])
                    overall_counts[0] = Counter(deletions)
                    post_len = len(overall_counts[0])
                    deletions = len(deletions)
            gc_start = time.time()
            _ = gc.collect()
            gc_total = time.time() - gc_start
            collect_total = time.time() - collect_start
            print("Filtering = %s, Overall counts pre-len = %s, after filtering len = %s, num ones = %s, num_deletions = %s, gc time = %.2f, collect time = %.2f" %
                  (pre_len > 2 ** 26, pre_len, post_len, pre_len - ones, post_len - deletions, gc_total, collect_total))

        return dict(tf=tf)

    c4_tokenized = c4.map(batch_term_frequency_builder, batched=True, batch_size=2048, remove_columns=['dataset', 'length', 'text', 'tfidf_average'])

overall_counts = {k: v for k, v in overall_counts.items()}
import pickle
with open('overall_counts.pickle', 'wb') as handle:
    pickle.dump(overall_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)


from collections import defaultdict
freq2token = defaultdict(list)

with open('overall_counts.pickle', 'rb') as handle:
    overall_counts = pickle.load(handle)

overall_counts = {k: v for k, v in overall_counts.items() if v >= 5}
n_docs = len(c4_tokenized)
log_docs = np.log2(n_docs)
idf = {k: log_docs - np.log2(1 + v) + 1 for k, v in overall_counts.items()}

with open('idf.pickle', 'wb') as handle:
    pickle.dump(idf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('count_buckets.pickle', 'wb') as handle:
    pickle.dump(count_buckets, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('overall_counts.pickle', 'rb') as handle:
    oc = pickle.load(handle)
with open('idf.pickle', 'rb') as handle:
    idf = pickle.load(handle)

def tfidf_batch(x):
    tfidf = [np.mean([float(v)*idf.get(t, 1) for t, v in tf]) for tf in x["tf"]]
    return dict(bigram_tfidf_average=tfidf)
print(tfidf_batch(c4_tokenized[0:32]))
c4_tokenized = c4_tokenized.map(tfidf_batch, batched=True, batch_size=256)
#
c4_tokenized = c4_tokenized.remove_columns(["tf"])
c4_tokenized.save_to_disk("/home/ahemf/processed/c4_extended")


###################################################################
## SBERT
###################################################################

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import gc
import numpy as np
from torch.nn.parallel.replicate import replicate
import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
c4_extended = Dataset.load_from_disk("/home/ahemf/processed_datasets/c4_extended")

batch_size=256
devices = 8
sberts = [SentenceTransformer("paraphrase-mpnet-base-v2").eval().to(torch.device("cuda:%s" % i)) for i in range(devices)]
for idx, sb in enumerate(sberts):
    d = torch.device("cuda:%s" % idx)
    sb.to(d)
    print(d, next(iter(sb[-2].parameters())).device)
    for p in sb.parameters():
        p.requires_grad = False

from multiprocess.pool import Pool
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=devices) as p:
    def mapper(*args):
        x, device_id = args[0]
        # print(device_id, len(x), next(iter(sberts[device_id][-2].parameters())).device)
        with torch.no_grad():
            d = torch.device("cuda:%s" % device_id)
            torch.cuda.set_device(d)
            rep = sberts[device_id].to(d).encode(x)
        # for s in sberts:
        #     s.zero_grad(set_to_none=True)
        # print(device_id, len(x), next(iter(sberts[device_id][-2].parameters())).device)
        return rep

    def forkjoin(x):
        texts = x["text"]
        csz = min(batch_size, int(np.ceil(len(texts) / 8)))
        chunks = [texts[i: i+csz] for i in range(0, len(texts), csz)]
        dev = [i % devices for i, _ in enumerate(chunks)]
        t = torch.cat([torch.tensor(r) for r in p.map(mapper, list(zip(chunks, dev)))]).cuda()
        # print(t.size())
        t = t / t.norm(2, -1, True)
        sims = t.mm(t.t()) - 1000 * torch.eye(t.size(0), device="cuda")
        values, _ = sims.topk(128)
        t128 = values.mean(1).tolist()
        return dict(sbert_top_128_avg=t128)

    with torch.no_grad():
        c4_sbert2 = c4_extended.add_column("identifier", list(range(len(c4_extended)))).shuffle(341257, writer_batch_size=8192*4).map(forkjoin, batched=True, batch_size=8192 * 4, remove_columns=['dataset', 'length', 'text', 'tfidf_average']).sort("identifier").remove_columns(["identifier"])

    print(forkjoin(c4_extended[:1024]))

    with torch.no_grad():
        c4_sbert = c4_extended.map(forkjoin, batched=True, batch_size=8192 * 4, remove_columns=['dataset', 'length', 'text', 'tfidf_average'])



c4_sbert.save_to_disk("/home/ahemf/processed_datasets/c4_sbert")


###################################################################
## Perplexity
###################################################################

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.nn import CrossEntropyLoss, functional as F, DataParallel
from torch import nn
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import torch
import os
import numpy as np
import torch
import gc
os.environ['TOKENIZERS_PARALLELISM'] = "true"
device = torch.device("cuda:0")
model_id = "distilgpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
c4_extended = Dataset.load_from_disk("/home/ahemf/processed_datasets/c4_extended")

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
        lls.append(log_likelihood if log_likelihood.ndim == 2 else log_likelihood.unsqueeze(-1))

    lls = torch.cat(lls, -1).squeeze()
    ppl = torch.exp(lls.sum(-1) / lengths)
    return dict(perplexity=ppl.tolist())


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
    print(forkjoin(c4_extended[:128]))


    with torch.no_grad():
        c4_perplexity = c4_extended.map(forkjoin, batched=True, batch_size=128, remove_columns=['dataset', 'length', 'text', 'tfidf_average'])

c4_perplexity.save_to_disk("/home/ahemf/processed_datasets/c4_perplexity")

