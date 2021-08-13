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
dsets_448.save_to_disk("/home/ahemf/processed/dsets_448")

