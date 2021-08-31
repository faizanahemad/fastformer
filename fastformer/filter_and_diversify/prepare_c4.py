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

