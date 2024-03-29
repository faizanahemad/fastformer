# TODO: support multiple datasets via config
# TODO: support all model types including normal BERT / Roberta
# TODO: Wandb monitoring for master node rank - 0 process
# TODO: model checkpointing for root process
# TODO: base on huggingface run_lm.py
# TODO: support passing hyperpraram config file
# TODO: CPU and GPU training support
# TODO: save model
# TODO: Resume training from saved checkpoint
# TODO: Use TQDM and progress bar as well as metrics for speed

# from fastformer.training.pickle4reducer import *
# import pickle4reducer
# import multiprocessing as mp
# ctx = mp.get_context("spawn")
# ctx.reducer = Pickle4Reducer()

import multiprocessing
import traceback
import dill
import signal
import copy
import warnings
warnings.simplefilter("ignore")
import shutil
import sys
import traceback
import jsonlines
import jsonlines as jsonlines
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate
from torch import nn
from torch.nn import functional as F
import random
import os
import argparse
import time
try:
    from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
except:
    pass

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm, trange
from torch.optim import AdamW
import torch.distributed as dist
from torch.multiprocessing import Process, ProcessContext
import torch.multiprocessing as mp
import torch.optim as optim
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from torch.cuda.amp import GradScaler, autocast
from fastformer.data import *
from fastformer.config import *
from fastformer.data.dataset import datadict_iterator, superglue_test, MTTDataset
from fastformer.utils import *
from fastformer.model import *
from transformers import optimization
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import wandb
from pytz import timezone
from datetime import datetime, timedelta
from torch.utils.data.dataloader import DataLoader
from collections import Counter
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import logging

for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--model_config', required=True, type=str,
                        help='model config')
    parser.add_argument('--lr', default=optimizer_config.lr, type=float,
                        help='lr')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Epochs')
    parser.add_argument('--weight_decay', default=0.1, type=float,
                        help='weight_decay')

    parser.add_argument('--hpo', required=False, type=str,
                        help='hpo dict with lr, epochs, warmup steps')

    parser.add_argument('--dataset_key', required=False, type=str,
                        help='dataset_key')
    parser.add_argument('--cls_tokens', default=1, type=int,
                        help='cls_tokens')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')
    parser.add_argument('--batch_size', required=False, type=int,
                        help='Batch Size')
    parser.add_argument('--seed', required=False, type=int, default=3123,
                        help='seed')
    parser.add_argument('--enable_layer_normalizers', action="store_true", default=False,
                        help='enable_layer_normalizers')

    parser.add_argument('--pretrained_model', required=False, type=str,
                        help='Pretrained Model')

    parser.add_argument('--resume', required=False, type=str,
                        help='Resume From')
    parser.add_argument('--checkpoint', required=False, type=str,
                        help='Checkpoint Location')

    parser.add_argument('--model_save_dir', required=False, type=str,
                        help='Save Dir')
    parser.add_argument('--model_save_name', required=False, type=str,
                        help='Save Name')

    parser.add_argument('--validate_on_start', action="store_true", default=False,
                        help='Validate before training')

    parser.add_argument('--skip_steps', action="store_true", default=False,
                        help='Skip already trained steps while continuing training')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--validate_only', action="store_true", default=False,
                        help='Validate Only')

    parser.add_argument('--test_only', action="store_true", default=True,
                        help='Test Only')

    parser.add_argument('--shuffle_dataset', action="store_true", default=False,
                        help='Shuffle Train')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--finetune', action="store_true", default=False,
                        help='finetune')

    parser.add_argument('--no_autocast', action="store_true", default=False,
                        help='Avoid Autocast')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--backward_hook', action="store_true", default=False,
                        help='Backward Hook for gradients')

    parser.add_argument('--train_dataset', required=False, type=str,
                        help='Train Dataset')

    parser.add_argument('--validation_dataset', required=False, type=str,
                        help='Validation Dataset')

    parser.add_argument('--test_dataset', required=False, type=str,
                        help='Test Dataset')

    parser.add_argument('--init_method', required=False, type=str, default="tcp",
                        help='init_method')

    parser.add_argument('--num_workers', required=False, type=int, default=0,
                        help='Dataloader workers')

    parser.add_argument('--master_addr', type=str, required='MASTER_ADDR' not in os.environ,
                        default=None if 'MASTER_ADDR' not in os.environ else os.environ['MASTER_ADDR'],
                        help='Master ADDR')
    parser.add_argument('--master_port', type=str, required='MASTER_PORT' not in os.environ,
                        default=None if 'MASTER_PORT' not in os.environ else os.environ['MASTER_PORT'],
                        help='Master PORT')
    parser.add_argument('--dist_backend', type=str, required=False,
                        default='nccl',
                        help='Distributed Backend')
    parser.add_argument('--log_every_steps', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validate_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    assert hasattr(args, "test_dataset") or not args["test_only"]
    assert hasattr(args, "validation_dataset") or not args["validate_only"]
    return vars(args)



class LargeValidator:
    def __init__(self, location, model, config, device, tokenizer, rank, world_size, size_dicts, no_autocast=False):
        self.location = location
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.ignore_keys = ['cnn_dailymail1024',
                            'wikihow_sep1024',
                            # 'wikihow_all1024',
                            'big_patent1024',
                            'xsum1024',
                            'scientific_papers_arxiv512',
                            'scientific_papers_pubmed512',
                            'kelm1024',
                            # 'amazon_reviews_multi1024',
                            'gigaword_qna',
                            'mrqa_v1',
                            'mrqa_v2',
                            'e2e_nlg_cleaned_qna',
                            'discovery_qna',
                            'hans_qna_v2',
                            'hans_qna_v1',
                            'wikihow_sep_qna_v2',
                            'wikihow_sep_qna_v3',
                            'wikihow_sep_qna_v1',
                            'empathetic_dialogues_qna'
                            ]
        self.includes = ['superglue_cb_v2',
                         'superglue_cb_v1',
                         'superglue_copa_v1',
                         'superglue_copa_v2',
                         'superglue_copa_v3',
                         'superglue_wsc_fixed_v2',
                         'superglue_wsc_v1',
                         'superglue_wsc_fixed_v1',
                         'superglue_wsc_v2',
                         'superglue_rte_v2',
                         'superglue_rte_v1',
                         'superglue_wic_v1',
                         'superglue_wic_v2',
                         'superglue_multirc_v3',
                         'superglue_boolq',
                         'superglue_record_v4',
                         'superglue_record_v3',
                         'superglue_multirc_v1',
                         'superglue_multirc_v2',
                         'superglue_record_v1',
                         'superglue_record_v2',
                         'snli_qna_v1',
                         'race_qna',
                         'glue_sst2_v2',
                         'glue_sst2',
                         'glue_qnli',
                         'rotten_tomatoes_qna',
                         'commonsense_qa',
                         'winogrande_qna',
                         'scitail_qna',
                         'hellaswag_qna',
                         'squad_v2_qna',
                         'squad_v2_qna_v2',
                         'swag_qna'
                         ]
        self.no_autocast = no_autocast
        self.size_dicts = size_dicts

    def __call__(self):
        # TODO: save model if val acc higher than before
        # TODO: build a full score for val set using scores from all datasets
        # TODO: WnB integration from root process
        # TODO: parallel validation
        # TODO: Resume:: save modelw with epoch number and give ability to save n_models only, save optimizer, save scheduler, num_steps already done also needs to be saved

        # TODO: Lower LR by lambda LR or step LR by step counting in a deterministic way
        # WanDB control decide if init or not and make shim
        # Better Batching
        size_dicts = self.size_dicts
        datadict = DatasetDict.load_from_disk(self.location)
        tokenizer = self.tokenizer
        model = self.model.to(self.device)
        model = model.eval()
        collate_fn = get_collate_fn(self.config.num_highway_cls_tokens, tokenizer.pad_token_id)
        results = dict()
        _ = [datadict.pop(k, None) for k in self.ignore_keys]
        datadict = {k: v for k, v in datadict.items() if k in self.includes}
        print("[Validation]: Time = %s, Rank = %s, Total Datasets for Val = %s" % (get_time_string(), self.rank, len(datadict)))
        for idx, (k, dataset) in enumerate(sorted(datadict.items())):
            while idx >= self.world_size:
                idx -= self.world_size
            if idx != self.rank:
                continue
            clean_memory()
            cns = dataset.column_names
            predictions = []
            if 'answer' in cns:
                labels = [dataset[i]["answer"] for i in range(len(dataset))]
            dataset = TokenizerDataset(self.config, tokenizer, get_char_to_id(),
                                       dict(padding="max_length", truncation=True, return_tensors="pt", max_length=self.config.tokenizer_length),
                                       dataset)
            dataset.training = False
            record_accuracy = False
            if 'answer' not in cns:
                dataset.training = True
                record_accuracy = True
            length = len(dataset)
            print("[Validation]: Time = %s, Rank = %s, Prepare-Validation-Dataset, Val for dataset = %s, length = %s, with columns = %s" % (get_time_string(), self.rank, k, len(dataset), cns))
            loader = DataLoader(dataset, sampler=None, batch_size=min(size_dicts.values()), collate_fn=collate_fn, prefetch_factor=2, num_workers=4)
            # loader = custom_batching_fn(loader, size_dicts, False)
            # loader = custom_batching_fn(tqdm(loader, desc=k, miniters=100, mininterval=30.0), size_dicts_val, False)

            for pt_batch in loader:
                pt_batch["record_accuracy"] = record_accuracy
                pt_batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in pt_batch.items()}
                # print("[Validation]: Time = %s, Rank = %s, Start-Validation, Val for dataset = %s, batch size = %s, first batch loaded" % (get_time_string(), self.rank, k, pt_batch["input_ids"].size()))
                if 'answer' in cns:
                    with torch.no_grad():

                        funnel_inputs = dict(input_ids=pt_batch["input_ids"],
                                             attention_mask=pt_batch["attention_mask"],
                                             token_type_ids=pt_batch["token_type_ids"] if "token_type_ids" in pt_batch else None,
                                             inputs_embeds=None,
                                             char_ids=pt_batch["char_ids"], char_offsets=pt_batch["char_offsets"],
                                             run_decoder=False,
                                             run_answering=True)
                        output = model.backbone(**funnel_inputs)
                        # print("[Validation]: Time = %s, Rank = %s, run-backbone-validation, Val for dataset = %s, Funnel model run" % (get_time_string(), self.rank, k))
                        answering_predictions = output["answering_logits"].detach().argmax(dim=-1)
                    # debug_answering_predictions = answer_decoder_debug(answering_predictions, tokenizer)
                    # print("[Validation]: Time = %s, Rank = %s, Mid-Validation, Val for dataset = %s, Answering preds = %s, inps = %s" % (get_time_string(), self.rank, k, debug_answering_predictions, answering_predictions[:, :8].tolist()))
                    answering_predictions = answer_decoder(answering_predictions, tokenizer)
                    predictions.extend(answering_predictions)

                else:
                    labels = pt_batch["label_mlm_input_ids"] if "label_mlm_input_ids" in pt_batch else pt_batch["input_ids"]
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        with autocast():
                            output = model(**pt_batch, labels=labels)["accuracy_hist"]
                    predictions.append(output)


            print("[Validation]: Time = %s, Rank = %s, For Dataset %s, Built predictions list, samples = %s" % (get_time_string(), self.rank, k, list(zip(labels, predictions))[:4]))
            if 'answer' in cns:
                final_labels, final_predictions = [], []
                for lbl, prd in zip(labels, predictions):
                    if len(prd) > len(lbl):
                        prd = prd[:len(lbl)]
                    if len(prd) < len(lbl):
                        prd = prd + ([''] * (len(lbl) - len(prd)))
                    final_labels.extend(lbl)
                    final_predictions.extend(prd)
                score = accuracy_score(final_labels, final_predictions)
                most_common_label = Counter(final_labels).most_common(1)[0][0]
                most_common_label_count = Counter(final_labels).most_common(1)[0][1]
                discount_score = most_common_label_count / len(final_labels)
                results[k] = dict(accuracy=score, common_class_accuracy=discount_score)
            else:
                results[k] = pd.DataFrame.from_records(predictions).mean().to_dict()
                _ = results[k].pop("answering_lm_accuracy", None)
            print("[Validation]: Time = %s, Rank = %s, Finished-Validation, For Dataset %s, results = %s" % (get_time_string(), self.rank, k, results[k]))
            with open('validation.txt', 'a') as f:
                print(str({k: results[k]}) + os.linesep, file=f)

            clean_memory()
        model = model.train()
        return results

def cleanup():

    dist.destroy_process_group()


def build_dataloader(location, shuffle_dataset, sampling_fraction, config, collate_fn, tokenizer, size_dicts, continuous_iter=True, world_size=1, num_workers=1):
    assert max(size_dicts.values()) % min(size_dicts.values()) == 0
    single_node = world_size == 1
    from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
    min_size = gcd_array(list(size_dicts.values()) + list(size_dicts.values()))
    prefetch_factor = 2 * (max(size_dicts.values()) // min_size)
    kwargs = dict(prefetch_factor=prefetch_factor) if num_workers > 0 else dict()
    try:
        train_dataset = Dataset.load_from_disk(location)
        train_dataset = TokenizerDataset(config, tokenizer, get_char_to_id(), dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), train_dataset)

        train_loader = DataLoader(train_dataset, sampler=None if single_node else DistributedSampler(train_dataset, shuffle=shuffle_dataset),
                                  batch_size=min_size, collate_fn=collate_fn, shuffle=shuffle_dataset and single_node,
                                  num_workers=num_workers, pin_memory=True, **kwargs)

        train_loader = custom_batching_fn(train_loader, size_dicts, continuous_iter)
    except:
        train_dataset = DatasetDict.load_from_disk(location)
        train_dataset = {k: v for k, v in train_dataset.items() if len(v) >= world_size}
        train_dataset_sampling_proba = {k: len(v) ** sampling_fraction for k, v in train_dataset.items()}
        lsum = sum(train_dataset_sampling_proba.values())
        train_dataset_sampling_proba = {k: v / lsum for k, v in train_dataset_sampling_proba.items()}
        train_dataset = {k: TokenizerDataset(config, tokenizer, get_char_to_id(), dict(padding="max_length", truncation=True, return_tensors="pt", max_length=config.tokenizer_length), v) for k, v in train_dataset.items()}
        # for v in train_dataset.values():
        #     v.training = False

        train_loader = {k: DataLoader(v, sampler=None if single_node else DistributedSampler(v, shuffle=shuffle_dataset, ), shuffle=shuffle_dataset and single_node,
                                      batch_size=min_size, collate_fn=collate_fn, num_workers=num_workers, **kwargs) for k, v in train_dataset.items()}

        train_loader = {k: custom_batching_fn(dataloader, size_dicts, continuous_iter) for k, dataloader in train_loader.items()}
        train_loader = datadict_iterator(train_loader, train_dataset_sampling_proba)
    return train_loader


def train_catch_exception(local_rank, args):
    from fastformer.training.train_lm_distributed import train
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    nr = args["nr"]
    try:
        train(local_rank, args)
    except Exception as e:
        print("[Exception-in-train]: Node Rank = %s, Local Rank = %s, Rank = %s, Exception = %s, Trace = %s" % (nr, local_rank, rank, e, traceback.format_exc()))
        # traceback.print_tb(e.__traceback__)
        # traceback.print_exception(*sys.exc_info())
        traceback.print_exc()
        raise e


def train_inner_loop(args, ddp_model, batch, labels, optimizer, scheduler, scaler, gradient_clipping, iter_size=1, no_sync=False, zero_grad_check=False):
    # It seems to me like the first accumulated gradients might get clipped several times hence giving more weight to last accumulated gradients :

    output = ddp_model(**batch, labels=labels)
    loss = output["loss"] / iter_size
    loss_dict = output["loss_dict"]
    loss.backward()
    zgradders = []
    inf_gradders = []
    if zero_grad_check:
        zgradders = [name for name, params in ddp_model.named_parameters() if torch.all(params.grad == 0).item()]
        if len(zgradders):
            print("[Train]: Time = %s, Zero Grads: " % get_time_string(), zgradders)
        inf_gradders = [name for name, params in ddp_model.named_parameters() if torch.any(torch.logical_not(torch.isfinite(params.grad))).item()]
        if len(inf_gradders):
            print("[Train]: Time = %s, INF/NAN Grads: " % get_time_string(), inf_gradders)
            # print([name for name, params in ddp_model.named_parameters() if params.grad is None])

    if not no_sync:
        ddp_model.clip_grad_norm_(gradient_clipping) if isinstance(ddp_model, FSDP) else torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clipping)
        optimizer.step()
        scheduler.step()

    if "loss" in loss_dict and np.isnan(loss_dict["loss"]):
        es = "[Train-Exception]: Time = %s, NAN Loss, Scale = %s, loss_dict = %s, lr = %s" % (
            get_time_string(), None, loss_dict, optimizer.param_groups[0]['lr'])
        raise ValueError(es)

    return dict(loss_dict=loss_dict, accuracy_hist=output["accuracy_hist"], preds_dict=output["preds_dict"], zero_grad=len(zgradders), inf_grad=len(inf_gradders))


def train(local_rank, args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # too many barriers / one node data parallel and multiple node DDP
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ["NCCL_DEBUG"] = "WARN"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # gpu_device = 0
    gpu_device = local_rank
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import warnings
    warnings.simplefilter("ignore")
    if args["wandb_dryrun"]:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_SILENT"] = "true"
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    torch.backends.cudnn.benchmark = True
    rank = args["nr"] if args["cpu"] else (args["nr"] * args["gpus_per_node"] + local_rank)
    nr = args["nr"]
    if args["cpu"]:
        assert local_rank == 0
        device = torch.device("cpu")
        args["dist_backend"] = "gloo"
        # init_method = "tcp://%s:%s" % ("127.0.0.1", "9999")
    else:
        device = torch.device(f'cuda:{gpu_device}')  # Unique only on individual node.
        torch.cuda.set_device(device)
    print("[Train]: Time = %s, Prepare to init Dist Process for Rank = %s" % (get_time_string(), rank))
    if args["init_method"] == "tcp":
        if args["nr"] == 0:
            args["master_addr"] = "0.0.0.0"
        init_method="tcp://%s:%s" % (args["master_addr"], args["master_port"])
    elif args["init_method"] == "file":
        init_method = 'file://%s/%s' % (args["master_addr"], args["master_port"])
    else:
        raise ValueError

    print("[Train]: Time = %s, --------------- Initializing Dist Process with init-method = %s for Rank = %s" % (get_time_string(), init_method, rank))
    dist.init_process_group(args["dist_backend"], rank=rank, world_size=args["world_size"], init_method=init_method)
    print("[Train]: Time = %s, Initialized Dist Process for Rank = %s" % (get_time_string(), rank))
    barrier = get_barrier(True)
    rnd = torch.tensor(int(time.time())).to(device)
    dist.broadcast(rnd, 0)
    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(timezone('Asia/Kolkata')).strftime(format)
    set_seeds(args["seed"])
    model_config.model_size = args["model_config"]
    size_dicts = get_batch_size(args["model_config"], not args["no_autocast"])

    mconf = model_config.to_dict()

    config = config_dict[mconf.pop("model_size")]
    if any(config.relative_attention):
        size_dicts = {k: v - 4 for k, v in size_dicts.items()}
    size_dicts = {1024: args["batch_size"]} if "batch_size" in args and isinstance(args["batch_size"], int) else size_dicts
    tokenizer = get_tokenizer(mconf.pop("tokenizer_name"))
    config.vocab_size = len(tokenizer) + 22
    config.tokenizer_length = 1024
    config.tokenizer_length = config.tokenizer_length - config.num_highway_cls_tokens
    config.max_position_embeddings = config.max_position_embeddings + config.num_highway_cls_tokens

    collate_fn = get_collate_fn(config.num_highway_cls_tokens, tokenizer.pad_token_id)

    if args["world_size"] != 128:
        optimizer_config.lr = optimizer_config.lr * (args["world_size"]/128)
    config.eps = 1e-4
    if args["no_autocast"]:
        optimizer_config.eps = 1e-7
        config.layer_norm_eps = 1e-7
        config.eps = 1e-7
        optimizer_config.gradient_clipping = 4 * optimizer_config.gradient_clipping

    fsdp_params = configure_fsdp(not args["no_autocast"], True if not args["no_autocast"] else False, True)
    fsdp_wrapper(wrap_type=0, init=True)
    print("[Train]: Time = %s, Build Model with fsdp params = %s" % (get_time_string(), fsdp_params))

    model = None
    if args["pretrained_model"] is not None and os.path.exists(args["pretrained_model"]) and not args["test_only"]:
        model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
        state_dict = torch.load(args["pretrained_model"], map_location='cpu' if args['cpu'] else 'cuda:%d' % gpu_device)

        try:
            model.load_state_dict(state_dict, strict=False)
        except:

            pos_emb = state_dict["backbone.embeddings.position_embeddings.weight"]
            state_dict["backbone.embeddings.position_embeddings.weight"] = torch.cat((pos_emb[:4], pos_emb, pos_emb[-5:]), 0)
            model.load_state_dict(state_dict, strict=False)
        del state_dict
    elif args["pretrained_model"] is not None and args["test_only"]:
        model = args["pretrained_model"]
        return

    if args["validate_on_start"] or args["validate_only"]:
        _ = LargeValidator(args["validation_dataset"], model, config, device, tokenizer, rank, args["world_size"], size_dicts, args["no_autocast"])()
        clean_memory()
        if args["validate_only"]:
            return

    if model is None:
        model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
    if rank == 0:
        print("[Train]: Time = %s, Trainable Params = %s" % (get_time_string(), numel(model) / 1_000_000))
        print(model)

    # ddp_model = FSDP(model, **fsdp_params)  # find_unused_parameters=True
    ddp_model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False, bucket_cap_mb=10)  # find_unused_parameters=True
    try:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)
    except:
        print("[Train]: Time = %s, No fp16_compress_hook present, Torch Version = %s" % (get_time_string(), torch.__version__))
    clean_memory()

    optc = optimizer_config.to_dict()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"]))
    optimizer.zero_grad(set_to_none=True)

    # model, optim, gradscaler, scheduler, steps

    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
    print("[Train]: Time = %s, Optimizer Created for Rank = %s, params = %s" % (get_time_string(), rank, optc))
    shuffle_dataset = args["shuffle_dataset"]
    if not args["validate_only"] and not args["test_only"]:
        train_loader = build_dataloader(args["train_dataset"], shuffle_dataset, 0.75, config, collate_fn, tokenizer, size_dicts, world_size=args["world_size"], num_workers=args["num_workers"])

    print("[Train]: Data Loaded for Rank = %s" % rank)
    validate_every_steps = args["validate_every_steps"]
    log_every_steps = args["log_every_steps"]
    save_every_steps = args["save_every_steps"]
    scheduler = optimization.get_constant_schedule_with_warmup(optimizer, optc["warmup_steps"])
    gradient_clipping = optc["gradient_clipping"]
    print("[Train]: Scheduler Created for Rank = %s" % rank)
    if "resume" in args and isinstance(args["resume"], str) and len(args["resume"].strip()) > 0:
        print("[Train]: Trying Resume from %s for Rank = %s" % (args["resume"], rank))
        other_load_details = load(args["resume"], ddp_model, optimizer, scheduler, None, gpu_device)

        if other_load_details is None:
            print("[Train]: No resume checkpoint from %s for Rank = %s" % (args["resume"], rank))
            args["resume"] = None
        else:
            print("[Train]: Resumed from %s for Rank = %s, other details = %s" % (args["resume"], rank, other_load_details))

    else:
        print("[Train]: No Resume for Rank = %s" % rank)
    _ = ddp_model.train()

    batch_times = []
    model_times = []
    full_times = []
    ddp_model.zero_grad(set_to_none=True)
    samples_processed = 0
    samples_processed_this_log_iter = 0
    print("[Train]: Time = %s, Start Training for Rank = %s" % (get_time_string(), rank))
    ds_name = list(filter(lambda x: len(x.strip()) > 0, args["train_dataset"].split("/")))[-1].replace("train_fastformer_resampled_", "")
    group = "%s-%s-%sN-%s" % (ds_name, args["model_config"], args["nodes"], time_string)
    if local_rank == 0:
        wandb_init_args = dict(project="fastformer", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank), group=group, id=f"{group}-worker-{nr}-{rank}-{local_rank}",
                               config={"args":args, "model_config": mconf, "config": config, "optimizer_config": optc},
                               settings=wandb.Settings(start_method="fork"))

        time.sleep(random.random() * 5)
        wandb.init(**wandb_init_args)
        print("[Train]: WandB-watch added over model for Rank = %s" % rank)
        # wandb.watch(model, log="all", log_freq=log_every_steps * 4)
    barrier()

    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)

    def get_hook(name_of_param=None):
        if name_of_param is None:
            def hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    # print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
                    grad[is_nan_inf] = 0.0
                sign = torch.sign(grad)
                grad_rand = torch.randint_like(grad, -10, 10)
                grad_rand[grad_rand == 0] = 1
                grad = torch.where(sign == 0, 1e-3 * grad_rand, grad)
                grad = torch.clamp_(grad, -1, 1)
                return grad
            return hook
        else:
            def named_hook(grad):
                is_nan_inf = torch.logical_not(torch.isfinite(grad))
                if is_nan_inf.any():
                    print("[GRAD-HOOK]: Time = %s, Param Name = %s, Detected Nan/Inf" % (get_time_string(), name_of_param))
                    grad = torch.where(is_nan_inf, torch.sign(grad) * torch.empty_like(grad).fill_(1e-2), grad)
                    grad = torch.clamp_(grad, -1e1, 1e1)
                    return grad
                else:
                    return None
            return named_hook
    if not args["no_autocast"] and args["backward_hook"]:
        for name, param in ddp_model.named_parameters():
            if "embeddings" in name or "sent_predict_fc" in name or "embed_proj_transpose" in name or "embed_proj" in name or "lm_head" in name or "contrastive_ffn" in name or "encoder.blocks.0" in name: #
                param.register_hook(get_hook())
            else:
                param.register_hook(get_hook())

    no_sync = args["accumulation_steps"] > 1
    iter_size = args["accumulation_steps"]

    start_time = time.time()
    for step, batch in enumerate(train_loader):
        gen_batch_time = time.time() - start_time
        batch_times.append(gen_batch_time)
        bs_size = list(batch["input_ids"].size())
        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}

        # if other_load_details is not None:
        #     if step < other_load_details["step"] and args["skip_steps"]:
        #         if (step + 1) % log_every_steps == 0 or step == 0:
        #             print("[Train]: Time = %s, Skipping step = %s, due to checkpoint with details = %s, Rank = %s" % (get_time_string(), step, other_load_details, rank))
        #         continue
        #     else:
        #         step += int(other_load_details["step"] * (other_load_details["world_size"]/args["world_size"]))
        #
        electra_loss_w = float(min(1.0, ((step + 1) / ((2 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["electra_loss_w"])
        ddp_model.module.electra_loss_w = electra_loss_w
        model.electra_loss_w = electra_loss_w

        answering_lm_w = float(min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["answering_lm_w"])
        ddp_model.module.answering_lm_w = answering_lm_w
        model.answering_lm_w = answering_lm_w

        sentence_order_prediction_w = float(min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["sentence_order_prediction_w"])
        ddp_model.module.sentence_order_prediction_w = sentence_order_prediction_w
        model.sentence_order_prediction_w = sentence_order_prediction_w

        contrastive_w = float(
            min(1.0, ((step + 1) / ((10 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["contrastive_w"])
        ddp_model.module.contrastive_w = contrastive_w
        model.contrastive_w = contrastive_w

        input_cls_orthogonal_w = float(max(0.0, 1.0 - ((step + 1) / ((2 if args["no_autocast"] else 20) * optc["warmup_steps"]))) * mconf["input_cls_orthogonal_w"])
        ddp_model.module.input_cls_orthogonal_w = input_cls_orthogonal_w
        model.input_cls_orthogonal_w = input_cls_orthogonal_w

        if (step + 1) % save_every_steps == 0:
            state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
            if local_rank == 0:
                torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
                if "checkpoint" in args and isinstance(args["checkpoint"], str) and len(args["checkpoint"].strip()) > 0:
                    save(args["checkpoint"], ddp_model, optimizer, scheduler, None, {"step": step, "samples_processed": samples_processed, "world_size": args["world_size"]})
            del state_dict
        if (step + 1) % validate_every_steps == 0:
            state_dict = ddp_model.state_dict() if not isinstance(ddp_model, DDP) else ddp_model.module.state_dict()
            val_model = FastFormerForFusedELECTRAPretraining(config, tokenizer=tokenizer, **mconf).to(device)
            val_model.load_state_dict(state_dict)
            _ = LargeValidator(args["validation_dataset"], val_model, config, device, tokenizer, rank, args["world_size"], size_dicts, args["no_autocast"])()
            del val_model
            del state_dict
            clean_memory()
            barrier()
        record_accuracy = False
        if (step + 1) % log_every_steps == 0:
            if local_rank == 0:
                record_accuracy = True

        batch["record_accuracy"] = record_accuracy
        labels = batch["label_mlm_input_ids"] if "label_mlm_input_ids" in batch else batch["input_ids"]
        model_start_time = time.time()
        samples_processed += int(batch["input_ids"].size(0))
        samples_processed_this_log_iter += int(batch["input_ids"].size(0))
        # clean_memory()
        # print("Step = %s, Before:, for Rank = %s, input_size = %s, Allocated = %.3f, Max Allocated = %.3f, Percent = %s" %
        #       (step, rank, batch["input_ids"].size(), torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() /1e6, torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()))  # torch.cuda.memory_summary()

        try:
            if no_sync and (step + 1) % iter_size != 0:
                with ddp_model.no_sync():
                    output = train_inner_loop(dict(no_autocast=args["no_autocast"], cpu=args["cpu"]), ddp_model, batch, labels, optimizer, scheduler, None, gradient_clipping, iter_size=iter_size,
                                              no_sync=True, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0)
            else:
                output = train_inner_loop(dict(no_autocast=args["no_autocast"], cpu=args["cpu"]), ddp_model, batch, labels, optimizer, scheduler, None, gradient_clipping, iter_size=iter_size,
                                          no_sync=False, zero_grad_check=(step + 1) % log_every_steps == 0 and local_rank == 0)
                optimizer.zero_grad(set_to_none=True)

        except Exception as e:
            es = "[Train-Exception]: Time = %s, Step = %s for Rank = %s, Scale = %s, input_size = %s, lr = %s" % (
            get_time_string(), step, rank, None, bs_size, optimizer.param_groups[0]['lr'])
            print(es)
            torch.save(dict(**batch, labels=labels), os.path.join(os.getcwd(), "error-input.pth"))
            reraise(e, es)  # https://stackoverflow.com/questions/9157210/how-do-i-raise-the-same-exception-with-a-custom-message-in-python/62662138#62662138

        # clean_memory()
        # print("Step = %s, After: , for Rank = %s, input_size = %s, Allocated = %.3f, Max Allocated = %.3f, Percent = %s" %
        #       (step, rank, batch["input_ids"].size(), torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() / 1e6,
        #        torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()))  # torch.cuda.memory_summary()

        model_end_time = time.time() - model_start_time
        model_times.append(model_end_time)
        full_time = time.time() - start_time
        full_times.append(full_time)
        if step == 0:
            print("[Train]: Time = %s, First Batch Training for Rank = %s" % (get_time_string(), rank))
        if (step + 1) % log_every_steps == 0:
            if local_rank == 0:
                samples_per_second = samples_processed_this_log_iter / np.sum(full_times)
                acc_dict = output["accuracy_hist"]
                loss_dict = output["loss_dict"]
                preds_dict = output["preds_dict"]
                time.sleep(random.random() + 0.1)
                wandb.log(dict(lr=optimizer.param_groups[0]['lr'], step=step, samples_processed=samples_processed, samples_per_second=samples_per_second, batch_x_sequence=np.prod(bs_size[:2]),
                               input_cls_orthogonal_w=input_cls_orthogonal_w, electra_loss_w=electra_loss_w,
                               batch_times=np.mean(batch_times), model_times=np.mean(model_times),
                               **loss_dict, **acc_dict, zero_grad=output["zero_grad"], inf_grad=output["inf_grad"]))
                print("[Train]: Time = %s, Rank = %s, steps = %s, samples_processed=%s, batch_size = %s, Loss = %s, Accuracy = %s, LR = %s" %
                      (get_time_string(), rank, step, samples_processed,
                       bs_size, loss_dict, output["accuracy_hist"], optimizer.param_groups[0]['lr']))
                print("[Train-Timings]: Time = %s, Batch time = %.4f, Model Time = %.4f, samples_per_second = %s" % (get_time_string(), np.mean(batch_times), np.mean(model_times), samples_per_second))
                if ddp_model.module.electra_loss_w > 0 and ddp_model.module.sentence_order_prediction_w > 0 and ddp_model.module.contrastive_w > 0:
                    print("[Train-Timings]: Time = %s,\n sent_order = %s,\n sent_order_nih = %s,\n mx_labels = %s,\n contrastive_labels = %s,\n electra_labels = %s" % (get_time_string(),
                                                                                                                    list(zip(batch["labels_segment_index"].view(-1).tolist(), preds_dict["sent_order_preds"])),
                                                                                                                    list(zip(preds_dict["sent_order_nih_labels"], preds_dict["sent_order_nih_preds"])),
                                                                                                                    list(zip(preds_dict["mx_labels"], preds_dict["mx_label_pred"])),
                                                                                                                    list(zip(preds_dict["contrastive_actuals"][:16] if "contrastive_actuals" in preds_dict else [], preds_dict["contrastive_preds"][:16] if "contrastive_actuals" in preds_dict else [])),
                                                                                                                    list(zip(preds_dict["electra_logits"][:16], preds_dict["electra_preds"][:16], preds_dict["electra_labels"][:16]))))
                del acc_dict
                del loss_dict

            batch_times = []
            model_times = []
            full_times = []
            samples_processed_this_log_iter = 0

            clean_memory()
            barrier()
        del batch
        del labels
        del output
        del bs_size
        start_time = time.time()

    print("Time = %s, Finished Training for Rank = %s" % (get_time_string(), rank))
    state_dict = ddp_model.state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, model_save_name))
        if "checkpoint" in args and isinstance(args["checkpoint"], str) and len(args["checkpoint"].strip()) > 0:
            save(args["checkpoint"], ddp_model, optimizer, scheduler, None,
                 {"step": step, "samples_processed": samples_processed, "world_size": args["world_size"], "loss": loss_dict, "accuracy_dict": acc_dict})

# I've been tracking an ema of sample training loss during training and using that to guide weighted data sampling (rather than the typical uniform sampling). Seems to help with a variety of real world datasets where the bulk of the data is often very similar and easy to learn but certain subpopulations are much more challenging.


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        # start_processes(train_catch_exception, (args,), args["gpus_per_node"], True, False, start_method='spawn')
        mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)


