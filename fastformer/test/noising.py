from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from torch import nn
import torch
from torch.nn import functional as F
import pandas as pd
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)
pd.set_option('precision', 2)
from fastformer.utils import CoOccurenceModel, get_backbone, spearman_correlation, corr, clean_text, GaussianNoise, VectorDisplacementNoise
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

text = """Events before the invention of writing systems are considered prehistory. "History" is an umbrella term comprising past events as well as the memory, discovery, collection, organization, presentation, and interpretation of these events. Historians seek knowledge of the past using historical sources such as written documents, oral accounts, art and material artifacts, and ecological markers.

History also includes the academic discipline which uses narrative to describe, examine, question, and analyze past events, and investigate their patterns of cause and effect. Historians often debate which narrative best explains an event, as well as the significance of different causes and effects. Historians also debate the nature of history as an end in itself, as well as its usefulness to give perspective on the problems of the present.

Stories common to a particular culture, but not supported by external sources (such as the tales surrounding King Arthur), are usually classified as cultural heritage or legends. History differs from myth in that it is supported by evidence. However, ancient cultural influences have helped spawn variant interpretations of the nature of history which have evolved over the centuries and continue to change today. The modern study of history is wide-ranging, and includes the study of specific regions and the study of certain topical or thematic elements of historical investigation. History is often taught as part of primary and secondary education, and the academic study of history is a major discipline in university studies."""

text2 = """Herodotus, a 5th-century BC Greek historian, is often considered the "father of history" in the Western tradition,[13] although he has also been criticized as the "father of lies".[14][15] Along with his contemporary Thucydides, he helped form the foundations for the modern study of past events and societies. Their works continue to be read today, and the gap between the culture-focused Herodotus and the military-focused Thucydides remains a point of contention or approach in modern historical writing. In East Asia, a state chronicle, the Spring and Autumn Annals, was reputed to date from as early as 722 BC, although only 2nd-century BC texts have survived."""

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
batch = [text, text2]
inputs = tokenizer.batch_encode_plus(batch, max_length=512, padding="max_length", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

roberta = AutoModelForMaskedLM.from_pretrained("roberta-base")
roberta = roberta.eval().to(device)
cooc = "cooc_7_roberta.pth"  # "fastformer/test/cooc_7_roberta.pth"
masking_model, _, _ = get_backbone(cooc.split("/")[-1], False, dropout_prob=0.0)
state_dict = torch.load(cooc, map_location='cpu')
masking_model.load_state_dict(state_dict, strict=True)
masking_model = masking_model.eval().to(device)
with torch.no_grad():
    mlm_rtd_hints = masking_model(inputs["input_ids"], inputs["attention_mask"], validation_iter=True)
co_oc_word_ce = mlm_rtd_hints["word_ce"]
co_oc_word_ce = co_oc_word_ce.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()

with torch.no_grad():
    drop = nn.Dropout(0.85).to(device)
    out = roberta(**inputs, labels=inputs["input_ids"])
    (out["logits"].argmax(dim=-1) == inputs["input_ids"]).view(-1)[inputs["attention_mask"].view(-1)]
    (out["logits"].argmax(dim=-1) == inputs["input_ids"]).view(-1)[inputs["attention_mask"].view(-1)].float().mean()
    out2 = roberta(inputs_embeds=roberta.roberta.embeddings.word_embeddings(inputs["input_ids"]), attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
    out3 = roberta(inputs_embeds=drop(roberta.roberta.embeddings.word_embeddings(inputs["input_ids"])), attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
    (out3["logits"].argmax(dim=-1) == inputs["input_ids"]).view(-1)[inputs["attention_mask"].view(-1)].float().mean()
    ce = nn.CrossEntropyLoss(reduction='none')(out3["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["input_ids"].view(-1)).view( inputs["input_ids"].size())
    ce = ce.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()
    top_confs, _ = out3["logits"].topk(2, -1)
    top_confs = F.softmax(top_confs, dim=-1)
    confidences = top_confs[:, :, 0] - top_confs[:, :, 1]
    bt = (1 - confidences)
    bt = bt.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()

monte_carlo_stddev = []
monte_carlo_ce = []
monte_carlo_bt = []
with torch.no_grad():
    for _ in range(5):
        drop = nn.Dropout(0.85).to(device)
        out = roberta(inputs_embeds=drop(roberta.roberta.embeddings.word_embeddings(inputs["input_ids"])), attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
        monte_carlo_stddev.append(out["logits"])
        ce_loss = nn.CrossEntropyLoss(reduction='none')(out["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["input_ids"].view(-1)).view( inputs["input_ids"].size())
        monte_carlo_ce.append(ce_loss)
        top_confs, _ = out["logits"].topk(2, -1)
        top_confs = F.softmax(top_confs, dim=-1)
        confidences = top_confs[:, :, 0] - top_confs[:, :, 1]
        under_confidence_scores = (1 - confidences)  # 1/confidences
        monte_carlo_bt.append(under_confidence_scores)

    monte_carlo_stddev = torch.stack(monte_carlo_stddev, dim=-1)
    monte_carlo_stddev = monte_carlo_stddev.std(-1).mean(-1)
    monte_carlo_ce = torch.stack(monte_carlo_ce, dim=-1).mean(-1)
    monte_carlo_bt = torch.stack(monte_carlo_bt, dim=-1).mean(-1)

flattened_mc = monte_carlo_stddev.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()
flattened_mc_ce = monte_carlo_ce.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()
flattened_mc_bt = monte_carlo_bt.view(-1)[inputs["attention_mask"].view(-1).bool()].detach()

compared_values = [co_oc_word_ce, ce, bt, flattened_mc, flattened_mc_ce, flattened_mc_bt]
compared_values_names = ["co_oc", "ce", "bt", "mc_std", "mc_ce", "mc_bt"]


def get_corrs(cv, cvn):
    ranked_corr, standard_corr = [], []
    for v in cv:
        rc, nc = [], []
        for u in cv:
            rc.append(spearman_correlation(v, u).item())
            nc.append(corr(v, u).item())
        ranked_corr.append(rc)
        standard_corr.append(nc)

    rc_corr = pd.DataFrame(ranked_corr, columns=cvn,  index=cvn)
    sd_corr = pd.DataFrame(standard_corr, columns=cvn,  index=cvn)
    return rc_corr, sd_corr

ranked_corr, standard_corr = get_corrs(compared_values, compared_values_names)


from datasets import load_dataset
import copy
import numpy as np
from torch.utils.data import DataLoader
wikitext = load_dataset("wikitext", "wikitext-103-v1")
wikitext = wikitext.filter(lambda x: len(x["text"].split())>64)


def token_id_masking(tokens, tokenizer, probability: float = 0.1) -> str:
    tokens = np.array(tokens.tolist())
    original_tokens = tokens.copy()
    special_tokens_idx = np.in1d(original_tokens, tokenizer.all_special_ids)
    probas = np.random.random(len(tokens))
    masked = probas <= probability
    tokens[masked] = tokenizer.mask_token_id
    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    tokens[special_tokens_idx] = original_tokens[special_tokens_idx]
    return torch.tensor(list(tokens))


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset: torch.utils.data.Dataset, word_mask_proba=0.1):
        try:
            self.tokenizer = copy.deepcopy(tokenizer)
        except:
            self.tokenizer = tokenizer
        self.tokenizer_args = dict(max_length=512, padding="max_length", return_tensors="pt")

        self.dataset = dataset
        self.word_mask_proba = word_mask_proba

    def __getitem__(self, item):
        item = self.dataset[item]
        tokenizer = self.tokenizer

        text = item["text"]
        text = clean_text(text)
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
        input_ids, attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs["attention_mask"].squeeze()
        results = dict()
        results["label_mlm_input_ids"] = input_ids
        input_ids = token_id_masking(results["label_mlm_input_ids"], self.tokenizer, self.word_mask_proba)
        results.update(dict(input_ids=input_ids, attention_mask=attention_mask, mask_locations=input_ids==tokenizer.mask_token_id))
        return results

    def __len__(self):
        return len(self.dataset)


dataset = MLMDataset(tokenizer, wikitext["train"])
dataset, _ = torch.utils.data.random_split(dataset, [1024, len(dataset) - 1024])
dataloader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True, prefetch_factor=2, shuffle=True)


overall_ce = []
overall_bt = []
overall_mlm = []
overall_vd = []
overall_gaussian = []
overall_cooc = []
i = 0

for inputs in tqdm(dataloader):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    drop = nn.Dropout(0.85).to(device)
    gn = GaussianNoise(0.15).to(device)
    vd = VectorDisplacementNoise(0.15).to(device)
    with torch.no_grad():
        out = roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                      labels=inputs["label_mlm_input_ids"])
        ce = nn.CrossEntropyLoss(reduction='none')(out["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["label_mlm_input_ids"].view(-1)).view(
            inputs["label_mlm_input_ids"].size())
        mask_ce = ce[inputs["mask_locations"]]
        overall_mlm.append(mask_ce.detach())


        inputs["input_ids"] = inputs["label_mlm_input_ids"]
        out = roberta(inputs_embeds=drop(roberta.roberta.embeddings.word_embeddings(inputs["input_ids"])), attention_mask=inputs["attention_mask"],
                      labels=inputs["input_ids"])

        ce = nn.CrossEntropyLoss(reduction='none')(out["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["input_ids"].view(-1)).view(
            inputs["input_ids"].size())
        overall_ce.append(ce[inputs["mask_locations"]].detach())
        top_confs, _ = out["logits"].topk(2, -1)
        top_confs = F.softmax(top_confs, dim=-1)
        confidences = top_confs[:, :, 0] - top_confs[:, :, 1]
        under_confidence_scores = (1 - confidences)  # 1/confidences
        overall_bt.append(under_confidence_scores[inputs["mask_locations"]].detach())

        mlm_rtd_hints = masking_model(inputs["input_ids"], inputs["attention_mask"], validation_iter=True)
        co_oc_word_ce = mlm_rtd_hints["word_ce"][inputs["mask_locations"]].detach()
        overall_cooc.append(co_oc_word_ce)


        out = roberta(inputs_embeds=gn(roberta.roberta.embeddings.word_embeddings(inputs["input_ids"])), attention_mask=inputs["attention_mask"],
                      labels=inputs["input_ids"])
        ce = nn.CrossEntropyLoss(reduction='none')(out["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["input_ids"].view(-1)).view(
            inputs["input_ids"].size())
        overall_gaussian.append(ce[inputs["mask_locations"]].detach())

        out = roberta(inputs_embeds=vd(roberta.roberta.embeddings.word_embeddings(inputs["input_ids"])),
                      attention_mask=inputs["attention_mask"],
                      labels=inputs["input_ids"])
        ce = nn.CrossEntropyLoss(reduction='none')(out["logits"].detach().view(-1, out3["logits"].size(-1)), inputs["input_ids"].view(-1)).view(
            inputs["input_ids"].size())
        overall_vd.append(ce[inputs["mask_locations"]].detach())


    i = i + 1
    if i > 32:
        break

overall_ce = torch.cat(overall_ce)
overall_bt = torch.cat(overall_bt)
overall_mlm = torch.cat(overall_mlm)
overall_cooc = torch.cat(overall_cooc)
overall_gaussian = torch.cat(overall_gaussian)
overall_vd = torch.cat(overall_vd)


compared_values = [overall_ce, overall_bt, overall_cooc, overall_gaussian, overall_vd, overall_mlm]
compared_values_names = ["ce", "bt", "co_oc", "gaussian", "vector", "mlm"]
ranked_corr, standard_corr = get_corrs(compared_values, compared_values_names)
print(ranked_corr)
print(standard_corr)



# TODO: Can also investigate ce+bt combo, a tensor displacement based noising instead of dropout, Gaussian Noise.









