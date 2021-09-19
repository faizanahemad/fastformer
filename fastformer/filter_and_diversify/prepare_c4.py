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
c4_tokenized = c4_tokenized.add_column("identifier", list(range(1, len(c4_tokenized)+1)))
c4_tokenized.save_to_disk("/home/ahemf/processed/c4_extended")


