from datasets import load_dataset
import re
import os
from fastformer.utils import get_filter_mapper
from fastformer.utils import clean_text
dataset = load_dataset("c4", 'en.noblocklist', script_version="master")
fmap = get_filter_mapper(256)
dataset_filtered = dataset.map(fmap, batched=True, batch_size=256, remove_columns=['timestamp'])

dataset_filtered.save_to_disk("/home/ahemf/processed/c4_256")

fmap = get_filter_mapper(448)
dataset_448 = dataset_filtered.map(fmap, batched=True, batch_size=256, remove_columns=['timestamp'])

