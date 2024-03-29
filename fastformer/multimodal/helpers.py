import argparse
import copy
import traceback
from typing import Tuple, Dict, Any, Union, List

from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
import torch.multiprocessing as mp
import torch.distributed as dist
from pytz import timezone
from datetime import datetime, timedelta
import numpy as np
import torch
import random
import time
import re
import gc
import os
from PIL import Image
import math
import torchvision.transforms as transforms
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedModel, RobertaConfig, \
    RobertaTokenizerFast, optimization
from transformers import LongformerModel, LongformerTokenizer
from transformers.models.longformer.modeling_longformer import LongformerEncoder, LongformerIntermediate, \
    LongformerOutput, LongformerPreTrainedModel, LongformerLMHead
from transformers.models.longformer.configuration_longformer import LongformerConfig
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch32_384
from timm.models.vision_transformer import vit_large_patch16_224, vit_large_patch32_384
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
import cv2
import pandas as pd
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaModel, RobertaEncoder
from einops import rearrange
from datasets import load_dataset

from fastformer.utils import set_seeds, get_barrier, init_weights, clean_memory, get_time_string, try_float, \
    worker_init_fn, numel, MetricLogger, SmoothedValue, is_dist_avail_and_initialized

image_size = 384
max_length = 512
image_patch_size = 32
image_grid = 12
total_image_panels = 4
image_mask_proba = 0.75
per_img_patches = int((image_grid * image_grid) - (image_mask_proba * (image_grid * image_grid)))

tokenizer_args=dict(padding=True, truncation=True, return_tensors="pt", max_length=max_length)

import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        # traceback.print_exc()
        return None

# train_image_augments = transforms.Compose([
#     transforms.Resize((image_size + 64, image_size + 64)),
#     transforms.RandomChoice([
#         transforms.RandomPerspective(distortion_scale=0.2, p=1.0, ),
#         transforms.RandomRotation(15, expand=True, ),
#         transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.1), shear=[-5, 5, -5, 5], fill=255),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
#         transforms.RandomApply([transforms.GaussianBlur(7)], 1.0),
#         transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
#     ]),
#     transforms.RandomChoice([transforms.TrivialAugmentWide(),
#                                  transforms.RandomAutocontrast(p=1.0),
#                                  transforms.RandomAdjustSharpness(2, p=1.0),
#                                  transforms.RandomAdjustSharpness(4, p=1.0),
#                                  transforms.RandomAdjustSharpness(16, p=1.0),
#                                  transforms.RandomAdjustSharpness(32, p=1.0),
#                                  transforms.RandomPosterize(bits=3, p=1.0),
#                                  transforms.RandomPosterize(bits=4, p=1.0),
#                                  transforms.GaussianBlur(21, sigma=(0.5, 4.0))],),
#     transforms.Resize((image_size, image_size)),
# ])

train_image_augments = transforms.Compose([
    SquarePad(),
    transforms.RandomChoice([
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0, fill=255),
        transforms.RandomRotation(10, expand=True, fill=255),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=[-5, 5, -5, 5], fill=255),
        transforms.RandomPosterize(bits=3, p=1.0),
        transforms.Compose([transforms.Resize([image_size, image_size]), transforms.GaussianBlur(7)]),
    ]),
    SquarePad(),
    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.2)),
    # transforms.Resize([image_size, image_size])
])

# train_image_augments = transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.2))

# train_image_augments = transforms.Resize([image_size, image_size])

# inference_image_shape_augments = transforms.Compose([
#         transforms.Resize(image_size+image_patch_size),
#         transforms.CenterCrop(image_size),
# ])

# panel_combine_resize = transforms.Compose([
#     transforms.Resize([image_size//2 +32, image_size//2 +32]),
#     transforms.CenterCrop(image_size//2),
# ])

panel_combine_resize = transforms.Resize([image_size//2, image_size//2])
inference_image_shape_augments = transforms.Compose([
    SquarePad(),
    transforms.Resize([image_size, image_size]),
    ])

def build_2d_sincos_position_embedding(grid_size, embed_dim, cls_tokens=0, temperature=1000., requires_grad = False):
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)

    if cls_tokens == 1:
        pe_token = torch.zeros([1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([pe_token, pos_emb], dim=0)
    pos_embed = nn.Parameter(pos_emb.unsqueeze(0))
    pos_embed.requires_grad = requires_grad
    return pos_embed


def get_sinusoid_encoding_table(n_position, d_hid, temperature=1000.):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(temperature, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def sketch_transform(im: Image.Image) -> np.ndarray:
    """
    Converts PIL Image to a sketch transform like a pencil sketch.
    Source https://towardsdatascience.com/generate-pencil-sketch-from-photo-in-python-7c56802d8acb
    To view a conversion, `Image.fromarray((sketch_transform(im) * 255).astype(np.uint8)).show()`
    @param im:
    @type im:
    @return:
    @rtype:
    """
    img = np.array(im)
    kernel_size = max(3, max(img.shape) // 32)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert Image
    invert_img = cv2.bitwise_not(grey_img)
    # invert_img=255-grey_img

    # Blur image
    blur_img = cv2.GaussianBlur(invert_img, (kernel_size, kernel_size), 0)

    # Invert Blurred Image
    invblur_img = cv2.bitwise_not(blur_img)
    # invblur_img=255-blur_img

    # Sketch Image
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
    return sketch_img / 255.0


def canny_edge_detector(im: Image.Image) -> np.ndarray:
    """
    Converts PIL Image to One-Channel canny edge detected numpy array, divided by 255.
    To view a conversion, `Image.fromarray((canny_edge_detector(im) * 255).astype(np.uint8)).show()`
    @param im:
    @type im:
    @return: One channel edge detected np.ndarray
    @rtype: np.ndarray
    """
    img = np.array(im)
    kernel_size = max(3, max(img.shape) // 128)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    np_array = cv2.cvtColor(cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                                                    (kernel_size, kernel_size), sigmaX=0, sigmaY=0), 100, 200),
                                         cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
    return np_array / 255.0


def gray_scale(im: Image.Image) -> np.ndarray:
    """
    Converts PIL Image to one channel grayscale, divided by 255.
    To view a conversion, `Image.fromarray((gray_scale(im) * 255).astype(np.uint8)).show()`
    @param im:
    @type im:
    @return:
    @rtype:
    """
    img = np.array(im)
    kernel_size = max(3, max(img.shape) // 128)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    np_array = cv2.cvtColor(cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (kernel_size, kernel_size), sigmaX=0, sigmaY=0), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
    return np_array / 255.0

gray_scale_no_blur = transforms.Grayscale()

def save_model(model_save_dir, model, optimizer, scheduler, metric_logger, local_rank, steps_done):
    state_dict = getattr(model, "module", model).state_dict()
    encoder_state_dict = getattr(getattr(model, "module", model), "encoder",
                                 getattr(model, "module", model)).state_dict()
    if local_rank == 0:
        torch.save(state_dict, os.path.join(model_save_dir, "trainer.pth"))
        torch.save(encoder_state_dict, os.path.join(model_save_dir, "encoder.pth"))
        # torch.save(state_dict, os.path.join(model_save_dir, "trainer-%s.pth" % steps_done))
        # torch.save(encoder_state_dict, os.path.join(model_save_dir, "encoder-%s.pth" % steps_done))
        # torch.save(optimizer.state_dict(), os.path.join(model_save_dir, "optimizer-%s.pth" % steps_done))
        # torch.save(scheduler.state_dict(), os.path.join(model_save_dir, "scheduler-%s.pth" % steps_done))
        torch.save(metric_logger.meters, os.path.join(model_save_dir, "metrics.pth"))
    del state_dict
    del encoder_state_dict
    clean_memory()


def float_detect(v):
    maybe_float = v is not None and ("e" in v or "." in v)
    try:
        v = float(v)
        return maybe_float
    except:
        return False

def float_format(v):
    v = str(v)
    if float_detect(v):
        decimal_split = v.split(".")
        if len(decimal_split) == 2:
            precision = min(3, len(decimal_split[1])) if "." in v and decimal_split[1].isdigit() and sum(map(int, list(decimal_split[1]))) > 0 else 0
        else:
            precision = 0
        precision_str = "%%.%sf" % precision
        return precision_str % (float(v))
    return v

def isnan(v):
    v = str(v)
    try:
        v = float(v)
        return np.isnan(v)
    except:
        return v == "None"


def text_masking(text: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], mask_probability: float = 0.15):
    mask = tokenizer.mask_token
    words = text.split()
    masked_words = []
    no_mask = True
    for ix, w in enumerate(words):
        if random.random() < mask_probability:
            tokens = tokenizer.tokenize((w if ix == 0 else " "+w))
            ln = len(tokens)
            # print(w, ln, tokens)
            masked_words.extend([mask] * ln)
            no_mask = False
        else:
            masked_words.append(w)
    try:
        if no_mask:
            w = masked_words.pop()
            tokens = tokenizer.tokenize((w if ix == 0 else " " + w))
            ln = len(tokens)
            masked_words.extend([mask] * ln)
    except Exception as e:
        print("[text_masking]: original text = %s, masked words = %s, split = %s" % (text, masked_words, words))
        raise e
    return " ".join(masked_words), " ".join(words)


def get_image_mask(num_patches, num_mask):
    mask = np.hstack([
        np.zeros(num_patches - num_mask),
        np.ones(num_mask),
    ])
    np.random.shuffle(mask)
    return mask

class MultiModalTrainingDataset(Dataset):
    """
    Expects a header-less csv.
    Handle if a particular modality is not present at all.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, tokenizer_args, images_path, data_csv, separator,
                 columns, text_columns, tabular_columns, image_columns,
                 image_size, image_patch_size, image_augments, image_to_vector=transforms.ToTensor(),
                 training=True,
                 word_mask_proba=0.15, image_mask_proba=image_mask_proba, tabular_feature_mask_proba=0.2, tabular_feature_drop_proba=0.1,
                 total_image_panels=total_image_panels,
                 ):
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.separator = separator
        self.columns = columns
        self.data_csv = data_csv
        self.images_path = images_path
        self.text_columns = text_columns
        self.image_columns = image_columns
        self.joiner = f" {tokenizer.sep_token} "
        self.tabular_columns = tabular_columns
        self.training = training
        self.tabular_feature_drop_proba = tabular_feature_drop_proba
        self.tabular_feature_mask_proba = tabular_feature_mask_proba
        self.image_augments = image_augments
        self.image_size = image_size
        self.image_to_vector = transforms.ToTensor()
        self.imagenet_normalization = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                                        std=torch.tensor(IMAGENET_DEFAULT_STD))
        self.total_image_panels = total_image_panels
        self.word_mask_proba = word_mask_proba
        self.image_patch_size = image_patch_size
        assert self.image_size % self.image_patch_size == 0
        self.num_patches = (self.image_size // self.image_patch_size) ** 2
        assert self.num_patches == (image_grid * image_grid)
        self.image_mask_proba = image_mask_proba
        self.num_mask = int(self.image_mask_proba * self.num_patches)
        self.length = len(self)
        self.dataset = load_dataset('csv', data_files=self.data_csv)["train"]
        self.imagenet_gray_mean = np.mean(IMAGENET_DEFAULT_MEAN)
        self.imagenet_gray_std = np.mean(IMAGENET_DEFAULT_STD)
        self.inference_image_shape_augments = inference_image_shape_augments

    def __get_image_mask__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask

    def __decode__(self, x):
        tokenizer = self.tokenizer
        text_coincide = x["text_input_ids"][:x["text_attention_mask"].sum()] == x["text_masked_input_ids"][:x["text_masked_attention_mask"].sum()]
        text_zipped_ids = list(zip(x["text_input_ids"][:x["text_attention_mask"].sum()].tolist(), x["text_masked_input_ids"][:x["text_masked_attention_mask"].sum()].tolist()))
        tabular_coincide = x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()] == x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()]
        tabular_zipped_ids = list(zip(x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()].tolist(),
                           x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()].tolist()))
        mean = torch.as_tensor(self.imagenet_normalization.mean)[None, :, None, None]
        std = torch.as_tensor(self.imagenet_normalization.std)[None, :, None, None]
        input_text = tokenizer.decode(x["text_input_ids"][:x["text_attention_mask"].sum()])
        masked_text = tokenizer.decode(x["text_masked_input_ids"][:x["text_masked_attention_mask"].sum()])

        student_input_tabular = tokenizer.decode(x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()])
        student_input_masked_tabular = tokenizer.decode(
            x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()])
        # input_tabular = tokenizer.decode(x["tabular_teacher_input_ids"][:x["tabular_teacher_attention_mask"].sum()])
        image_labels = x["image_labels"]
        image_masks = x["image_masks"]
        all_patch = torch.zeros(image_masks.shape[0], image_masks.shape[1], image_patch_size * image_patch_size * 3)
        B, C = all_patch.shape[0], image_patch_size * image_patch_size * 3
        image_labels = image_labels.reshape(B, -1, image_patch_size * image_patch_size * 3)
        S = all_patch.shape[1]
        all_patch = all_patch.view(-1, C)
        all_patch[image_masks.view(-1)] = image_labels.view(-1, C)

        actual_images = x["images"]
        all_patch[~image_masks.view(-1)] = rearrange(actual_images, 'b c (h p1) (w p2) -> b (h w) p1 p2 c', p1=image_patch_size,
                                   p2=image_patch_size).view(-1, C)[~image_masks.view(-1)]
        all_patch = all_patch.reshape(B, S, C).reshape(B, S, image_patch_size, image_patch_size, 3)
        all_patch = rearrange(all_patch, 'b (h w) p1 p2 c -> b (h w) (p1 p2) c', h=image_grid, w=image_grid)
        all_patch = all_patch * x["image_patch_std"] + x["image_patch_mean"]
        all_patch = rearrange(all_patch, 'b (h w) (p1 p2) c -> b (h w) p1 p2 c', h=image_grid, w=image_grid, p1=image_patch_size, p2=image_patch_size)
        all_patch = rearrange(all_patch, 'b (h w) p1 p2 c -> b (h p1) (w p2) c', h=image_grid, w=image_grid)
        all_patch = (all_patch.permute(0, 3, 1, 2) * std + mean).permute(0, 2, 3, 1)

        actual_images = actual_images * std + mean
        actual_images = [(x.permute(1, 2, 0) * 255).clip(0, 255).numpy().astype(np.uint8) for x in actual_images]
        all_patch = [(x * 255).clip(0, 255).numpy().astype(np.uint8) for x in all_patch]
        actual_images = [Image.fromarray(x) for x in actual_images]
        all_patch = [Image.fromarray(x) for x in all_patch]
        return dict(input_text=input_text, masked_text=masked_text,
                    text_coincide=text_coincide, text_zipped_ids=text_zipped_ids, tabular_coincide=tabular_coincide, tabular_zipped_ids=tabular_zipped_ids,
                    student_input_tabular=student_input_tabular, student_input_masked_tabular=student_input_masked_tabular,
                    # input_tabular=input_tabular,
                    all_patch=all_patch, actual_images=actual_images)


    def __len__(self):
        if hasattr(self, "length") and self.length is not None:
            return self.length
        else:
            with open(self.data_csv, "r+") as f:
                # reader_file = csv.reader(f, delimiter=self.separator)
                line_count = sum(1 for line in f)
            self.length = line_count - 1
            return self.length

    def __get_raw_item__(self, item):
        item = pd.read_csv(self.data_csv, names=self.columns, sep=self.separator, low_memory=False, skiprows=item, nrows=1,
                           header=0)
        text = item[self.text_columns].values[0]
        text = self.joiner.join(text)
        tabular = list(zip(self.tabular_columns, list(item[self.tabular_columns].to_records()[0])[1:]))
        image_locations = item[self.image_columns].values[0]
        image_locations = " ".join(image_locations)
        image_locations = list(image_locations.split())
        return dict(item=item, text=text, tabular=tabular, image_locations=image_locations)


    def get_text_from_item(self, item):
        text = [str(item[col]) for col in self.text_columns]
        text = self.joiner.join(text)
        # text = [str(t) for t in item[self.text_columns].values[0]]
        # text = self.joiner.join(text)
        return text

    def get_tabular_from_item(self, item):
        tabular = list(zip(self.tabular_columns, [item[col] for col in self.tabular_columns]))
        # tabular = list(zip(self.tabular_columns, list(item[self.tabular_columns].to_records()[0])[1:]))
        return tabular

    def get_image_locations_from_item(self, item):
        image_locations = " ".join([str(item[col]) for col in self.image_columns])
        # image_locations = item[self.image_columns].values[0]
        # image_locations = [str(im) for im in image_locations]
        # image_locations = " ".join(image_locations)
        return image_locations

    def __getitem__(self, item):

        tokenizer = self.tokenizer
        mask = self.tokenizer.mask_token
        # item = pd.read_csv(self.data_csv, names=self.columns, sep=self.separator, low_memory=False, skiprows=item, nrows=1,
        #                    header=0)
        item_idx = item
        item = self.dataset[item]
        text = self.get_text_from_item(item)

        if len(text) == 0 or len(text.split()) < 2:
            text += " <empty text> <empty text>"
        if self.training:
            masked_text, text = text_masking(text, tokenizer, self.word_mask_proba)
        else:
            masked_text = text
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
        text_input_ids, text_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()

        tokenizer_outputs = tokenizer(masked_text, return_offsets_mapping=False, **self.tokenizer_args)
        text_masked_input_ids, text_masked_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()
        assert text_masked_attention_mask.sum() == text_attention_mask.sum()

        # tabular = list(zip(self.tabular_columns, list(item[self.tabular_columns].to_records()[0])[1:]))
        tabular = self.get_tabular_from_item(item)
        # tabular_to_text_for_teacher = ""
        # for k, v in tabular:
        #     k = k.replace("_", " ")
        #     tabular_to_text_for_teacher = tabular_to_text_for_teacher + " " + k + " = " + float_format(v) + " ;"
        tabular_to_text_for_student_input = ""
        tabular_to_text_for_student_output = ""
        random.shuffle(tabular)
        not_masked = True
        for ix, (k, v) in enumerate(tabular):
            k = k.replace("_", " ")
            v = float_format(v)
            if random.random() < self.tabular_feature_drop_proba and (ix < len(tabular) - 1 or not not_masked) and self.training:
                continue
            if (random.random() < self.tabular_feature_mask_proba and not isnan(v)) or (not_masked and ix == len(tabular) - 1) and self.training:
                tabular_to_text_for_student_input = tabular_to_text_for_student_input + " " + k + " = " + (" ".join([mask] * len(tokenizer.tokenize(" " + v)))) + " ;"
                not_masked = False
            else:
                tabular_to_text_for_student_input = tabular_to_text_for_student_input + " " + k + " = " + (v) + " ;"
            # tabular_to_text_for_student_input = tabular_to_text_for_student_input + " " + k + " = " + (" ".join([mask] * len(tokenizer.tokenize(" " + v))) if random.random() < self.tabular_feature_mask_proba and not isnan(v) else v) + " ;"
            tabular_to_text_for_student_output = tabular_to_text_for_student_output + " " + k + " = " + v + " ;"

        # tokenizer_outputs = tokenizer(tabular_to_text_for_teacher, return_offsets_mapping=False, **self.tokenizer_args)
        # t2t_teacher_input_ids, t2t_teacher_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs["attention_mask"].squeeze()

        tokenizer_outputs = tokenizer(tabular_to_text_for_student_output, return_offsets_mapping=False, **self.tokenizer_args)
        t2t_student_input_ids, t2t_student_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()

        tokenizer_outputs = tokenizer(tabular_to_text_for_student_input, return_offsets_mapping=False, **self.tokenizer_args)
        t2t_student_masked_input_ids, t2t_student_masked_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()
        assert t2t_student_masked_attention_mask.sum() == t2t_student_attention_mask.sum()
        # assert t2t_teacher_attention_mask.sum() >= t2t_student_attention_mask.sum()

        image_locations = self.get_image_locations_from_item(item)
        image_locations = list(image_locations.split())  # Assuming all images are separated in their columns by space
        image_locations = [os.path.join(self.images_path, im) for im in image_locations if im is not None]
        image_locations = [im for im in map(pil_loader, image_locations) if im is not None and str(im).lower() != "none"]
        if len(image_locations) > 0:
            image_std = [np.array(im).std() / np.array(im).mean() for im in image_locations]
            image_locations, image_std = zip(*sorted(zip(image_locations, image_std), key=lambda x: x[1], reverse=True))
            image_locations = list(image_locations)  # We try to give high variance images single panels
        count_images = len(image_locations)

        image_locations_new = []
        for im in image_locations:
            try:
                if self.training:
                    im = self.image_augments(im)
                else:
                    im = self.inference_image_shape_augments(im)
            except Exception as e:
                print("[ERROR][Dataset]: Failed image augmentation of item_idx = %s" % (item_idx,))
                traceback.print_exc()
            image_locations_new.append(im)
        image_locations = image_locations_new
        total_image_panels = self.total_image_panels
        num_images = len(image_locations)
        # image_attention_mask = np.zeros(image_grid * image_grid * total_image_panels, dtype=np.float)
        panel_distribution = [1] * total_image_panels
        if num_images > total_image_panels:
            extra_images = num_images - total_image_panels
            j = 0
            for i in range(extra_images):
                if panel_distribution[j] < 4:
                    panel_distribution[j] += 1
                if panel_distribution[j] >= 4:
                    j += 1
                if j > total_image_panels - 1:
                    break
            panels = []
            for p in panel_distribution:
                if p == 1:
                    panels.append(image_locations.pop())
                else:
                    sample_panel = np.zeros((image_size, image_size, 3)).astype(np.uint8)
                    for p_idx in range(p):
                        resized_im = np.array(panel_combine_resize(image_locations.pop())).astype(np.uint8)
                        if p_idx == 0:
                            sample_panel[:image_size // 2, :image_size // 2] = resized_im
                        elif p_idx == 1:
                            sample_panel[image_size // 2:, :image_size // 2] = resized_im
                        elif p_idx == 2:
                            sample_panel[:image_size // 2, image_size // 2:] = resized_im
                        elif p_idx == 3:
                            sample_panel[image_size // 2:, image_size // 2:] = resized_im
                        else:
                            raise ValueError
                    panels.append(sample_panel)
            image_locations = panels
            # image_attention_mask[:] = 1.0
            assert len(image_locations) == total_image_panels

        if len(image_locations) < self.total_image_panels:
            # image_attention_mask[: image_grid * image_grid * len(image_locations)] = 1.0
            if len(image_locations) >= 1:
                image_locations.extend([random.choice(image_locations) for _ in range(self.total_image_panels - len(image_locations))])
            else:
                image_locations.extend([Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8)) for _ in range(self.total_image_panels - len(image_locations))])
        image_locations = list(map(self.image_to_vector, image_locations))
        image_locations = list(map(self.imagenet_normalization, image_locations))
        image_inputs = torch.tensor(np.stack(image_locations))

        masks = [self.__get_image_mask__() for _ in range(len(image_locations))]
        image_masks = torch.tensor(np.stack(masks)).bool()
        image_locations = torch.tensor(np.stack(image_locations))
        try:
            # images_patch = rearrange(image_locations, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=image_patch_size, p2=image_patch_size)
            images_squeeze = rearrange(image_locations, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=image_patch_size,p2=image_patch_size)
        except Exception as e:
            print(item)
            print("[Dataset] [Error] Item index = %s" % item_idx)
            raise e
        image_patch_mean = images_squeeze.mean(dim=-2, keepdim=True)
        image_patch_std = (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        images_norm = (images_squeeze - image_patch_mean) / image_patch_std
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
        B, _, C = images_patch.shape
        image_labels = images_patch.view(-1, C)[image_masks.view(-1)].reshape(B, -1, C)  # 2D indexing isn't working so bring index to 1D

        # assert (text_attention_mask.sum() == text_masked_attention_mask.sum()).item()
        # assert (tabular_student_attention_mask.sum() == tabular_student_masked_attention_mask.sum()).item()
        return dict(num_images=num_images, image_labels=image_labels,
                    image_masks=image_masks, images=image_inputs,
                    tabular_student_masked_input_ids=t2t_student_masked_input_ids,
                    tabular_student_masked_attention_mask=t2t_student_masked_attention_mask,
                    tabular_student_input_ids=t2t_student_input_ids,
                    tabular_student_attention_mask=t2t_student_attention_mask,
                    image_patch_mean=image_patch_mean,
                    image_patch_std=image_patch_std,
                    panel_distribution=torch.tensor(panel_distribution, dtype=torch.long),
                    # tabular_teacher_input_ids=t2t_teacher_input_ids,
                    # tabular_teacher_attention_mask=t2t_teacher_attention_mask,
                    text_input_ids=text_input_ids, text_attention_mask=text_attention_mask,
                    text_masked_input_ids=text_masked_input_ids, text_masked_attention_mask=text_masked_attention_mask)


class LongformerFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)

    def forward(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output

from itertools import chain
from torch.utils.checkpoint import checkpoint

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


class MultiModalEncoder(LongformerPreTrainedModel):
    def __init__(self, model_size="large", grad_checkpointing=False):
        config = LongformerConfig.from_pretrained("allenai/longformer-large-4096" if model_size == "large" else "allenai/longformer-base-4096")
        super().__init__(config)
        # TODO: check gradient checkpointing first
        # TODO: check layer_norm_eps for fp16 support
        self.model_size = model_size
        if model_size == "large":
            embed_dim = 1024
            longformer = RobertaModel.from_pretrained("roberta-large")
            # longformer = LongformerModel.from_pretrained("allenai/longformer-large-4096")
            # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            vit = vit_large_patch32_384(True)
            mid_fusion_backbone_config = LongformerConfig.from_pretrained("allenai/longformer-large-4096")
            mid_fusion_backbone_config.num_hidden_layers = 6
        elif model_size == "base":
            longformer = RobertaModel.from_pretrained("roberta-base")
            # longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")
            # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            embed_dim = 768
            vit = vit_base_patch32_384(True)
            mid_fusion_backbone_config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
            mid_fusion_backbone_config.num_hidden_layers = 6
        else:
            raise ValueError
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer

        # self.text_ffn = LongformerFFN(mid_fusion_backbone_config)
        # self.tabular_ffn = LongformerFFN(mid_fusion_backbone_config)
        # self.image_ffn = LongformerFFN(mid_fusion_backbone_config)
        mid_fusion_backbone_config.vocab_size = 2
        mid_fusion_backbone_config.pad_token_id = 1
        mid_fusion_backbone_config.attention_window = mid_fusion_backbone_config.attention_window[:mid_fusion_backbone_config.num_hidden_layers]
        mid_fusion_backbone = LongformerModel(config=mid_fusion_backbone_config)
        mid_fusion_backbone.pooler = None
        init_weights(mid_fusion_backbone, 0.02)
        self.mid_fusion_backbone = mid_fusion_backbone
        self.panel_count_emb = nn.Embedding(4, embed_dim)
        init_weights(self.panel_count_emb)
        self.panel_id_emb = nn.Parameter(torch.zeros(1, total_image_panels, 1, embed_dim))
        init_weights(self.panel_id_emb)
        self.text_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        init_weights(self.text_seg_token)

        self.tabular_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        init_weights(self.tabular_seg_token)

        self.image_seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        init_weights(self.image_seg_token)

        self.grad_checkpointing = grad_checkpointing
        self.supports_gradient_checkpointing = True
        config.grad_checkpointing = grad_checkpointing
        self.post_init()

        self.longformer = longformer
        self.vit = vit  # TODO: checkout get_pretrained_deit from utils, forward_features from vit.

    def vit_forward(self, x, mask):
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)[:, 1: x.shape[1]]

        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask.view(B, mask.shape[-1])].reshape(B, -1, C)  # ~mask means visible
        else:
            x_vis = x

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_vis = checkpoint_seq(self.vit.blocks, x_vis)
        else:
            x_vis = self.vit.blocks(x_vis)
        x_vis = self.vit.norm(x_vis)
        return x_vis

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LongformerEncoder, MultiModalEncoder, RobertaEncoder)):
            module.gradient_checkpointing = value
        self.grad_checkpointing = True

    def forward(self, input_ids=None, attention_mask=None,
                tabular_input_ids=None, tabular_attention_mask=None,
                images=None, panel_distribution=None, mask=None, output_attentions=None):
        """
        We do image masking after image patch embedding.
        @param tabular_attention_mask:
        @type tabular_attention_mask:
        @param attention_mask:
        @type attention_mask:
        @param mask:
        @type mask:
        @param input_ids: dimensions = `Batch x Sequence`
        @type input_ids: torch.Tensor
        @param images: dimensions = `(Batch x Images Per Example=5) x H x W x C`
        @type images: torch.Tensor
        @param tabular_input_ids: dimensions = `Batch x Sequence`
        @type tabular_input_ids: torch.Tensor
        @return: dict(text_embedding [both text+tabular], images_embedding [(Batch x Images Per Example) x S x d],
        text_mem_tokens, image_mem_tokens, missing_image_generator_tokens, image_mask_locations)
        @rtype:
        """

        # Assert at least one modality present.
        assert images is not None or input_ids is not None or tabular_input_ids is not None
        text_features = None
        tabular_features = None
        if input_ids is not None and tabular_input_ids is not None:
            assert input_ids.size(1) == tabular_input_ids.size(1)
            lf_input_ids = torch.cat([input_ids, tabular_input_ids], 0)
            lf_attention_mask = torch.cat([attention_mask, tabular_attention_mask], 0)
            tabular_text_output = self.longformer(input_ids=lf_input_ids, attention_mask=lf_attention_mask, )["last_hidden_state"]
            text_features = tabular_text_output[:input_ids.size(0)]
            tabular_features = tabular_text_output[input_ids.size(0):]
            tabular_text_output = torch.cat([text_features + self.text_seg_token, tabular_features + self.tabular_seg_token], 1)
            tabular_text_output_attention_mask = torch.cat([attention_mask, tabular_attention_mask], 1)
            global_attention_positions = [0, input_ids.size(1)]
        elif input_ids is not None and tabular_input_ids is None:
            text_features = self.longformer(input_ids=input_ids, attention_mask=attention_mask, )[
                "last_hidden_state"]
            tabular_text_output = text_features + self.text_seg_token
            tabular_text_output_attention_mask = attention_mask
            global_attention_positions = [0]
        elif input_ids is None and tabular_input_ids is not None:
            tabular_features = self.longformer(input_ids=tabular_input_ids, attention_mask=tabular_attention_mask, )[
                "last_hidden_state"]
            tabular_text_output = tabular_features + self.tabular_seg_token
            tabular_text_output_attention_mask = tabular_attention_mask
            global_attention_positions = [0]
        else:
            tabular_text_output = None
            tabular_text_output_attention_mask = None
            global_attention_positions = None

        if images is not None:
            b, ex, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            image_features = self.vit_forward(images, mask)  # (B x ex), Seq, Dim
            image_features = image_features.reshape(b, ex, -1, image_features.shape[-1])
            if panel_distribution is not None:
                pdist = self.panel_count_emb(panel_distribution - 1)[:, :, None, :]
                image_mid_in = image_features + pdist
            else:
                image_mid_in = image_features.clone()
            image_mid_in = image_mid_in + self.panel_id_emb
            image_mid_in = image_mid_in.reshape(b, -1, image_mid_in.shape[-1])
            image_mid_in = image_mid_in + self.image_seg_token
        else:
            image_features = None
            image_mid_in = None

        assert image_features is not None or tabular_text_output is not None

        if image_features is None:
            features = tabular_text_output
            attention_mask = tabular_text_output_attention_mask
        elif tabular_text_output is None:
            features = image_mid_in
            attention_mask = torch.ones(features.shape[:2], dtype=torch.long, device=features.device, requires_grad=False)
        else:
            features = torch.cat([tabular_text_output, image_mid_in], 1)
            image_attention_mask = torch.ones(image_mid_in.shape[:2], dtype=torch.long, device=features.device,
                                        requires_grad=False)
            attention_mask = torch.cat([tabular_text_output_attention_mask, image_attention_mask], 1)
            global_attention_positions.extend([tabular_text_output.size(1), tabular_text_output.size(1)+1])

        s = features.size(1)
        # extra = 512 - s % 512
        # if extra > 0 and extra < 512:
        #     features = torch.cat([features, torch.zeros(features.size(0), extra, features.size(2),
        #                                                 dtype=features.dtype, device=features.device, requires_grad=False)], dim=1)

        global_attention_mask = torch.zeros_like(attention_mask)

        if images is not None and tabular_text_output is not None:
            for i in range(ex):
                ep2 = (per_img_patches * i)
                end = tabular_text_output.size(1) + ep2
                end_2 = tabular_text_output.size(1) + (per_img_patches * (i + 1)) - 1
                global_attention_positions.extend([end, end_2])
        elif images is not None:
            for i in range(ex):
                global_attention_positions.extend([per_img_patches * i, 1 + per_img_patches * i, per_img_patches * (i + 1) - 1])
        global_attention_positions = list(set(global_attention_positions))
        # gap_mod = 32
        # gap_len = len(global_attention_positions)
        # if gap_len % gap_mod != 0:
        #     extra_positions_needed = gap_mod - (gap_len % gap_mod)
        #     new_positions = [i for i in range(2, 2 + extra_positions_needed)]
        #     global_attention_positions.extend(new_positions)
        #     assert len(global_attention_positions) % gap_mod == 0
        # global_attention_positions = sorted(global_attention_positions)
        global_attention_mask[:, global_attention_positions] = 1.0

        mid_fusion_out = self.mid_fusion_backbone(attention_mask=attention_mask,
                                                  global_attention_mask=global_attention_mask,
                                                  inputs_embeds=features, output_attentions=output_attentions)
        attentions = mid_fusion_out["attentions"] if output_attentions else None
        global_attentions = mid_fusion_out["global_attentions"] if output_attentions else None
        features = mid_fusion_out[0]
        # if extra > 0 and extra < 512:
        #     features = features[:, :-extra]
        assert s == features.size(1)
        image_out = None
        if image_features is not None:
            image_out = features[:, -image_mid_in.size(1):]
            image_out = image_out.reshape(b, ex, -1, image_out.shape[2])

        text_output = None
        tabular_output = None

        if tabular_text_output is not None:
            tabular_text_final = features[:, :tabular_text_output.size(1)]
            if input_ids is not None:
                text_output = tabular_text_final[:, :input_ids.size(1)]
                assert text_output.size(1) == text_features.size(1) == input_ids.size(1)
            if tabular_input_ids is not None:
                tabular_output = tabular_text_final[:, -tabular_input_ids.size(1):]
                assert tabular_output.size(1) == tabular_features.size(1) == tabular_input_ids.size(1)

        global_tokens = features[:, global_attention_positions]
        rdict = dict(image_output=image_out, text_output=text_output, tabular_output=tabular_output,
                    unimodal_image_features=image_features, unimodal_text_features=text_features,
                    unimodal_tabular_features=tabular_features, global_tokens=global_tokens,
                    global_attention_mask=global_attention_mask)
        if output_attentions:
            rdict["attentions"] = attentions
            rdict["global_attentions"] = global_attentions
            rdict["global_attention_positions"] = global_attention_positions
        return rdict


class MultiModalSelfSupervisedTrainerModel(LongformerPreTrainedModel):
    # contains both teacher and student
    # Teacher's missing image generator part is not executed
    # Contains Image and Text MLM decoder as well.
    # contains Loss fn for missing image generator, Image and Text MLM
    # Build Mask here
    # Dropout modality rate -> used with student teacher
    # For canny, gray_scale and sketch transforms of output image
    # Weigh non-zero elements in image generator output for loss more highly by loss = loss.mean() + loss[non_zero_loc].mean(),
    def __init__(self, contrastive_loss_w, image_mlm_w, text_mlm_w, tabular_mlm_w,
                 encoder: MultiModalEncoder):
        super().__init__(encoder.longformer.config)
        self.encoder = encoder
        self.mask_token_id = self.encoder.tokenizer.mask_token_id
        self.mlm_ce = nn.CrossEntropyLoss(ignore_index=self.encoder.tokenizer.pad_token_id)

        self.lm_head = LongformerLMHead(self.encoder.longformer.config)
        init_weights(self.lm_head)
        self.lm_ffn = LongformerFFN(self.encoder.longformer.config)
        init_weights(self.lm_ffn)
        self.text_mlm_w = text_mlm_w
        self.tabular_mlm_w = tabular_mlm_w
        self.image_mlm_w = image_mlm_w
        self.contrastive_loss_w = contrastive_loss_w
        decoder_embed_dim = self.encoder.embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.image_mlm_ln1 = nn.LayerNorm(encoder.embed_dim, optimizer_config["eps"])
        init_weights(self.image_mlm_ln1)
        self.text_mlm_ln = nn.LayerNorm(encoder.embed_dim, optimizer_config["eps"])
        self.tabular_mlm_ln = nn.LayerNorm(encoder.embed_dim, optimizer_config["eps"])
        init_weights(self.text_mlm_ln)
        init_weights(self.tabular_mlm_ln)
        embed_dim = decoder_embed_dim
        self.embed_dim = embed_dim
        self.contrastive_temp = 0.25

        self.contrast_loss = nn.BCEWithLogitsLoss()
        self.contrast_ffn = nn.Sequential(nn.LayerNorm(embed_dim, optimizer_config["eps"]), LongformerFFN(self.encoder.longformer.config), nn.Dropout(0.1), nn.Linear(decoder_embed_dim, decoder_embed_dim * 2), nn.GELU(), nn.Linear(decoder_embed_dim * 2, decoder_embed_dim))
        init_weights(self.contrast_ffn)

        decoder_layer_conf = RobertaConfig()
        decoder_layer_conf.hidden_size = embed_dim
        decoder_layer_conf.num_attention_heads = 16 if self.encoder.model_size == "large" else 12
        decoder_layer_conf.add_cross_attention = False
        decoder_layer_conf.is_decoder = False
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        init_weights(self.mask_token)
        trunc_normal_(self.mask_token, std=.02)
        self.decoder = nn.ModuleList([RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf),
                                      RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf)])

        self.decoder_head = nn.Sequential(nn.Linear(decoder_embed_dim, decoder_embed_dim * 4), nn.GELU(), nn.Linear(decoder_embed_dim * 4, image_patch_size*image_patch_size*3))
        init_weights(self.decoder, std=.02)
        init_weights(self.decoder_head, std=.02)
        self.image_mlm_ln2 = nn.LayerNorm(embed_dim, optimizer_config["eps"])
        self.image_mlm_ln3 = nn.LayerNorm(embed_dim, optimizer_config["eps"])
        init_weights(self.image_mlm_ln2)
        init_weights(self.image_mlm_ln3)
        decoder_query = build_2d_sincos_position_embedding(image_grid, decoder_layer_conf.hidden_size, )
        self.decoder_inputs = torch.nn.Parameter(decoder_query, requires_grad=False)

        self.mse = nn.MSELoss()
        self.mse_reconstruct = nn.MSELoss(reduction="none")
        half_image_size = image_size // 2
        centered_reconstruction_weights = torch.tensor(np.interp(np.arange(0, half_image_size), [0, 95, half_image_size-1], [1, 2, 1])).unsqueeze(0).repeat(half_image_size, 1)
        centered_reconstruction_weights = centered_reconstruction_weights * centered_reconstruction_weights.T
        self.centered_reconstruction_weights = nn.Parameter(centered_reconstruction_weights, requires_grad=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def image_mlm_forward_with_decode(self, x_vis, mask, image_unmasked_patches, image_patch_mean, image_patch_std, imagenet_mean=IMAGENET_DEFAULT_MEAN,
                          imagenet_std=IMAGENET_DEFAULT_STD):
        B_init = x_vis.size(0)
        loss, mask, x_full, mask_count = self.image_mlm_forward(x_vis, mask, image_unmasked_patches)
        x = torch.zeros_like(x_full)
        x[~mask] = x_full[:, :-mask_count].reshape(-1, x_full.size(-1))
        x[mask] = x_full[:, -mask_count:].reshape(-1, x_full.size(-1))
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, 3)
        image_patch_mean = image_patch_mean.flatten(0, 1).unsqueeze(2)
        image_patch_std = image_patch_std.flatten(0, 1).unsqueeze(2)
        x = x * image_patch_std + image_patch_mean
        x = x.reshape(x.shape[0], 12, 12, 32, 32, 3)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B_init, -1, 384, 384, 3)

        im = x.permute(0, 1, 4, 2, 3)
        mean = torch.as_tensor(imagenet_mean)[None, None, :, None, None]
        std = torch.as_tensor(imagenet_std)[None, None, :, None, None]

        im = ((im * std + mean) * 255).clip(0, 255).detach().permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)
        images = []
        for b in range(len(im)):
            ims_arr = []
            for p in range(im.shape[1]):
                ims_arr.append(Image.fromarray(im[b][p]))
            images.append(ims_arr)

        return loss, images

    def image_mlm_forward(self, x_vis, mask, image_unmasked_patches):
        mask = mask.view(-1, *mask.shape[2:])
        x_vis = x_vis.reshape(-1, x_vis.shape[2], self.decoder_embed_dim)
        image_unmasked_patches = image_unmasked_patches.view(-1, *image_unmasked_patches.shape[2:])
        B, N, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.image_mlm_ln3(self.decoder_inputs).expand(B, -1, -1)
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x_full = self.image_mlm_ln2(x_full)
        for blk in self.decoder:
            x_full = blk(x_full)[0]
        mask_count = pos_emd_mask.shape[1]
        x_full = self.decoder_head(x_full)
        loss = self.mse(input=x_full[:, -mask_count:], target=image_unmasked_patches)
        return loss, mask, x_full, mask_count

    def forward(self, input_ids, attention_mask,
                tabular_input_ids, tabular_attention_mask,
                images=None, panel_distribution=None, image_masks=None, image_labels=None,
                label_input_ids=None, label_tabular_input_ids=None):
        if random.random() < 0.1 and image_masks is not None:
            new_masks = torch.zeros_like(image_masks)
            bs, ps = image_masks.shape[:2]
            for i in range(bs):
                for j in range(ps):
                    new_mask = get_image_mask(image_grid * image_grid, int(0.25 * (image_grid * image_grid)))
                    new_masks[i][j] = torch.tensor(new_mask, device=image_masks.device)
            image_masks = new_masks
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   tabular_input_ids=tabular_input_ids,
                                   tabular_attention_mask=tabular_attention_mask, images=images, panel_distribution=panel_distribution, mask=image_masks)
        # TODO: masked and non-masked tokens must coincide properly
        masked_lm = input_ids == self.mask_token_id
        lm_feats = self.text_mlm_ln(encoder_out["text_output"] + encoder_out["unimodal_text_features"])[masked_lm]
        label_input_ids = label_input_ids[masked_lm]
        lm_out = self.lm_head(self.lm_ffn(lm_feats))
        mlm_loss = self.text_mlm_w * self.mlm_ce(lm_out, label_input_ids)
        mlm_accuracy = (lm_out.argmax(dim=-1) == label_input_ids).float().mean().item()

        masked_tabular = tabular_input_ids == self.mask_token_id
        tabular_feats = self.tabular_mlm_ln(encoder_out["tabular_output"] + encoder_out["unimodal_tabular_features"])[masked_tabular]
        label_tabular_input_ids = label_tabular_input_ids[masked_tabular]
        tabular_lm_out = self.lm_head(self.lm_ffn(tabular_feats))
        tabular_mlm_loss = self.tabular_mlm_w * self.mlm_ce(tabular_lm_out, label_tabular_input_ids)
        tabular_mlm_accuracy = (tabular_lm_out.argmax(dim=-1) == label_tabular_input_ids).float().mean().item()
        image_mlm_features = self.image_mlm_ln1(encoder_out["image_output"] + encoder_out["unimodal_image_features"])
        image_mlm_loss = self.image_mlm_w * self.image_mlm_forward(image_mlm_features,
                                                image_masks, image_labels)[0]

        image_vec = encoder_out["image_output"][:, :, 0]
        text_vec = encoder_out["text_output"][:, 0].unsqueeze(1)
        contrastive_vec = self.contrast_ffn(torch.cat([text_vec, image_vec], 1))
        contrastive_vec = contrastive_vec / contrastive_vec.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        contrastive_loss = 0
        is_dist = is_dist_avail_and_initialized()
        if is_dist:
            ws = torch.distributed.get_world_size(group=None)
            rnk = torch.distributed.get_rank(group=None)
        else:
            ws = 1
            rnk = 0

        bs = input_ids.shape[0]
        contrast_elems = total_image_panels+1
        t = torch.zeros(ws, bs, contrast_elems, self.embed_dim, device=input_ids.device)
        t[rnk] = t[rnk] + contrastive_vec.unsqueeze(0)
        if is_dist:
            torch.distributed.all_reduce(t)
        mm = torch.einsum('w x i d, y j d -> w x y i j', t, contrastive_vec) / self.contrastive_temp
        mm = mm ** 2
        labels = torch.zeros_like(mm)
        A = torch.ones(bs, contrast_elems, contrast_elems, device=input_ids.device)
        B = torch.block_diag(*A)
        B = B.reshape(bs, contrast_elems, bs*contrast_elems).permute(0, 2, 1).reshape(bs, bs, contrast_elems, contrast_elems)
        labels[rnk] = labels[rnk] + B
        contrastive_loss = self.contrastive_loss_w * self.contrast_loss(mm, labels)
        loss = mlm_loss + tabular_mlm_loss + image_mlm_loss + contrastive_loss

        return dict(loss=loss, tabular_mlm_accuracy=tabular_mlm_accuracy, mlm_accuracy=mlm_accuracy,
                    mlm_loss=mlm_loss, tabular_mlm_loss=tabular_mlm_loss, image_mlm_loss=image_mlm_loss,
                    contrastive_loss=contrastive_loss, image_masks=image_masks)

        # TODO: to optimize tabular we need to write separate collate fn. For starters keep text size and table size = 512.


optimizer_config = dict(lr=5e-5, eps=1e-8, weight_decay=1e-3, beta_1=0.9, beta_2=0.98, gradient_clipping=1.0)


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    parser.add_argument('--model_size', required=True, type=str,
                        help='model size')

    parser.add_argument('--wandb_name', required=False, type=str, default="",
                        help='wandb_name')

    parser.add_argument('--epochs', type=int, required=True,
                        help='epochs')

    parser.add_argument('--batch_size', required=True, type=int,
                        help='Batch Size')

    parser.add_argument('--lr', default=optimizer_config["lr"], type=float,
                        help='lr')
    parser.add_argument('--weight_decay', default=optimizer_config["weight_decay"], type=float,
                        help='weight_decay')
    parser.add_argument('--gradient_clipping', default=optimizer_config["gradient_clipping"], type=float,
                        help='gradient_clipping')
    parser.add_argument('--beta_1', default=optimizer_config["beta_1"], type=float,
                        help='beta_1')
    parser.add_argument('--beta_2', default=optimizer_config["beta_2"], type=float,
                        help='beta_2')

    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient Accumulation')

    parser.add_argument('--load_trainer_model', required=False, type=str,
                        help='Load SSL trainer Model')

    parser.add_argument('--load_encoder', required=False, type=str,
                        help='Load only MultiModalEncoder Model')

    parser.add_argument('--model_save_dir', required=True, type=str,
                        help='Save Dir')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--text_mlm_w', type=float, required=False, default=1.0,
                        help='text_mlm_w weight')
    parser.add_argument('--tabular_mlm_w', type=float, required=False, default=0.2,
                        help='tabular_mlm_w weight')
    parser.add_argument('--image_mlm_w', type=float, required=False, default=10.0,
                        help='image_mlm_w weight')
    parser.add_argument('--contrastive_loss_w', type=float, required=False, default=1.0,
                        help='contrastive_loss_w weight')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--optimizer', required=False, type=str, default="adamw",
                        help='optimizer')

    parser.add_argument('--num_workers', required=False, type=int, default=8,
                        help='Dataloader workers')

    parser.add_argument('--master_addr', type=str, required='MASTER_ADDR' not in os.environ,
                        default=None if 'MASTER_ADDR' not in os.environ else os.environ['MASTER_ADDR'],
                        help='Master ADDR')
    parser.add_argument('--master_port', type=str, required='MASTER_PORT' not in os.environ,
                        default=None if 'MASTER_PORT' not in os.environ else os.environ['MASTER_PORT'],
                        help='Master PORT')
    parser.add_argument('--log_every_steps', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_every_steps', type=int, default=1_000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--dataset', required=True, type=str,
                        help='Dataset')
    parser.add_argument('--images_path', required=True, type=str,
                        help='images_path')


    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 75139563
    args.seed = seed
    return vars(args)

def build_propreitery_dataset(location, images_path, tokenizer):
    COLUMNS = ['asin', 'text', 'price', 'has_customer_reviews',
               'customer_review_count', 'customer_average_review_rating', 'browse_root_name',
               'browse_node_id', 'browse_node_ids', 'browse_node_name', 'browse_node_l2',
               'brand_name', 'gl', 'category_code', 'subcategory_code', 'has_fba_offering',
               'instock_gv_count', 'gv_count', 'glance_view_band', 'total_ordered_units', 'instock_by_total_gv',
               'num_offers'] + ["physical_id"]
    textual = ["text"]
    tabular = ['price', 'has_customer_reviews',
               'customer_review_count', 'customer_average_review_rating', 'browse_root_name',
               'browse_node_id', 'browse_node_ids', 'browse_node_name', 'browse_node_l2',
               'brand_name', 'gl', 'category_code', 'subcategory_code', 'has_fba_offering',
               'instock_gv_count', 'gv_count', 'glance_view_band', 'total_ordered_units', 'instock_by_total_gv',
               'num_offers']
    image_columns = ["physical_id"]
    dataset = MultiModalTrainingDataset(tokenizer, tokenizer_args, images_path, location, ",", COLUMNS, textual, tabular,
                                        image_columns,
                                        image_size, image_patch_size, train_image_augments)
    return dataset

def build_propreitery_dataloader(location, images_path, batch_size, tokenizer, world_size=1, num_workers=None, shuffle=True, training=True, drop_last=True,):
    single_node = world_size == 1
    num_workers = min(max(os.cpu_count() // 2, 1), 4) if num_workers is None else num_workers
    COLUMNS = ['asin', 'text', 'price', 'has_customer_reviews',
               'customer_review_count', 'customer_average_review_rating', 'browse_root_name',
               'browse_node_id', 'browse_node_ids', 'browse_node_name', 'browse_node_l2',
               'brand_name', 'gl', 'category_code', 'subcategory_code', 'has_fba_offering',
               'instock_gv_count', 'gv_count', 'glance_view_band', 'total_ordered_units', 'instock_by_total_gv',
               'num_offers'] + ["physical_id"]
    textual = ["text"]
    tabular = ['price', 'has_customer_reviews',
               'customer_review_count', 'customer_average_review_rating', 'browse_root_name',
               'browse_node_id', 'browse_node_ids', 'browse_node_name', 'browse_node_l2',
               'brand_name', 'gl', 'category_code', 'subcategory_code', 'has_fba_offering',
               'instock_gv_count', 'gv_count', 'glance_view_band', 'total_ordered_units', 'instock_by_total_gv',
               'num_offers']
    image_columns = ["physical_id"]
    dataset = MultiModalTrainingDataset(tokenizer, tokenizer_args, images_path, location, ",", COLUMNS, textual, tabular,
                                        image_columns,
                                        image_size, image_patch_size, train_image_augments,
                                        training=training)
    kwargs = dict(prefetch_factor=2, persistent_workers=False) if num_workers > 0 else dict()
    sampler = None if single_node else DistributedSampler(dataset, shuffle=shuffle)
    train_loader = DataLoader(dataset, sampler=sampler, drop_last=drop_last,
                              batch_size=batch_size, shuffle=single_node and shuffle, worker_init_fn=worker_init_fn,
                              num_workers=num_workers, pin_memory=True, **kwargs)

    return train_loader




def train(local_rank, args):
    torch.backends.cudnn.benchmark = True
    import os
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # too many barriers / one node data parallel and multiple node DDP
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # gpu_device = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args["detect_anomaly"]:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    gpu_device = local_rank
    if args["wandb_dryrun"]:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_SILENT"] = "true"
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    rank = args["nr"] if args["cpu"] else (args["nr"] * args["gpus_per_node"] + local_rank)
    nr = args["nr"]
    if args["cpu"]:
        assert local_rank == 0
        assert args["world_size"] == 1
        device = torch.device("cpu")
    else:
        print("For Node Rank = %s, setting device to %s" % (nr, gpu_device))
        device = torch.device(f'cuda:{gpu_device}')  # Unique only on individual node.
        torch.cuda.set_device(device)
    # init_method = "tcp://%s:%s" % (args["master_addr"], args["master_port"])
    init_method = "env://"
    rnd = torch.tensor(0.0, device="cpu")
    if args["world_size"] > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args["world_size"], init_method=init_method)
        rnd = torch.tensor(int(time.time())).to(device)
        dist.broadcast(rnd, 0)
    barrier = get_barrier(args["world_size"] > 1)
    format = "%Y-%m-%d %H-%M %Z"
    # + timedelta(hours=5, minutes=30)
    time_string = (datetime.fromtimestamp(time.mktime(time.gmtime(rnd.cpu().item())))).astimezone(
        timezone('Asia/Kolkata')).strftime(format)
    set_seeds(args["seed"])

    optimizer_config["lr"] = args["lr"]
    optimizer_config["weight_decay"] = args["weight_decay"]
    optimizer_config["gradient_clipping"] = args["gradient_clipping"]
    optimizer_config["beta_1"] = args["beta_1"]
    optimizer_config["beta_2"] = args["beta_2"]
    optimizer_config["eps"] = 1e-8

    encoder = MultiModalEncoder(args["model_size"])
    trainer = MultiModalSelfSupervisedTrainerModel(args["contrastive_loss_w"], args["image_mlm_w"], args["text_mlm_w"],
                                                   args["text_mlm_w"], encoder)
    encoder_param_count = numel(encoder)
    trainer_param_count = numel(trainer)
    if "load_encoder" in args and args["load_encoder"] is not None:
        encoder_weights = torch.load(args["load_encoder"], map_location='cpu')
        encoder.load_state_dict(encoder_weights)

    if "load_trainer_model" in args and args["load_trainer_model"] is not None:
        trainer_weights = torch.load(args["load_trainer_model"], map_location='cpu')
        trainer.load_state_dict(trainer_weights, strict=False)

    model = trainer.train().to(device)
    if args["world_size"] > 1:
        model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False,
                    bucket_cap_mb=10, gradient_as_bucket_view=True, static_graph=True)  # find_unused_parameters=True
    clean_memory()
    barrier()
    optc = copy.deepcopy(optimizer_config)
    model.zero_grad(set_to_none=True)
    no_decay = ['bias', 'LayerNorm.weight', "embeddings"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **dict(lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"])))
    optimizer.zero_grad(set_to_none=True)

    model_save_dir = args["model_save_dir"]

    set_seeds(args["seed"] + rank)
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)
        torch.save(args, os.path.join(model_save_dir, "args.pth"))

    batch_size = args["batch_size"]
    dataloader = build_propreitery_dataloader(args["dataset"], args["images_path"], batch_size, encoder.tokenizer, args["world_size"], args["num_workers"],)  # "/local/datasets/asin-images/combined-all.csv"
    iter_size = max(args["accumulation_steps"], 1)
    no_sync = iter_size > 1
    steps_per_epoch = int(np.floor(len(dataloader.sampler) / (batch_size * iter_size)) if dataloader.sampler is not None else (len(dataloader) / (iter_size)))
    total_steps = steps_per_epoch * args["epochs"]
    div_factor = optc["lr"] / 1e-8
    pct_start = min(0.04, 10_000 / total_steps)
    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, int(pct_start * total_steps))
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, optc["lr"], total_steps=total_steps,
                                                    div_factor=100., three_phase=False, pct_start=0.04,
                                                    base_momentum=0.75,
                                                    anneal_strategy="cos", cycle_momentum=True)

    barrier()
    if local_rank == 0:
        print("[Train]: Model initialized, encoder_param_count = %s, trainer_param_count = %s, trainer - encoder = %s" % (encoder_param_count, trainer_param_count, trainer_param_count - encoder_param_count))
        print("[Train]: Time = %s, Optimizer and Scheduler Initialised, max lr = %.5f, steps_per_epoch = %s, batch size = %s, dataloader length = %s, Sampler Present = %s, Sampler Length = %s" %
              (get_time_string(), optc["lr"], steps_per_epoch, batch_size, len(dataloader), dataloader.sampler is not None, len(dataloader.sampler) if dataloader.sampler is not None else -1))
    log_every_steps = args["log_every_steps"] * iter_size
    save_every_steps = args["save_every_steps"]
    gradient_clipping = optc["gradient_clipping"]
    group = "%s-%sN-%s" % (args["wandb_name"], args["nodes"], time_string)

    wandb_init_args = dict(project="fnd", name="%s-%s-%s-%s" % (group, args["nr"], rank, local_rank),
                           group=group,
                           id=f"{group}-{nr}X{rank}-{local_rank}",
                           config={"args": args, "optimizer_config": optc},
                           settings=wandb.Settings(start_method="fork"))
    activate_wandb_log = local_rank <= (8 // args["world_size"]) or args["world_size"] <= 8
    if activate_wandb_log:
        wandb.init(**wandb_init_args)
    metric_logger = MetricLogger(delimiter="  ", plots=["loss", "lr", "mlm_accuracy",
                                                        "tabular_mlm_accuracy",
                                                        "image_mlm_loss", "contrastive_loss"])
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    logs_save = []
    model.zero_grad(set_to_none=True)
    samples_processed = 0
    samples_processed_this_log_iter = 0
    if args["detect_anomaly"]:
        torch.autograd.set_detect_anomaly(True)
    steps_done = 0
    step = 0
    for epoch in range(args["epochs"]):
        random.seed(args["seed"] + rank + epoch)
        set_seeds(args["seed"] + rank + epoch)
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        print("[Train]: Starting epoch number = %s of %s" % (epoch + 1, args["epochs"]))
        for ix, batch in enumerate(metric_logger.log_every(dataloader, log_every_steps, header = 'Epoch: [{}]'.format(epoch))):
            key = list(batch.keys())[0]
            batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
            if (steps_done + 1) % save_every_steps == 0:
                save_model(model_save_dir, model, optimizer, scheduler, metric_logger, local_rank, steps_done)
                barrier()
            samples_processed += int(batch[key].size(0))
            samples_processed_this_log_iter += int(batch[key].size(0))
            validation_iter = (step + 1) % log_every_steps == 0 or step == 0
            if no_sync and (step + 1) % iter_size != 0 and hasattr(model, "no_sync"):
                with model.no_sync():
                    model_output = trainer(batch["text_masked_input_ids"], batch["text_masked_attention_mask"],
                                           batch["tabular_student_masked_input_ids"],
                                           batch["tabular_student_masked_attention_mask"],
                                           batch["images"], batch["panel_distribution"], batch["image_masks"], batch["image_labels"],
                                           batch["text_input_ids"], batch["tabular_student_input_ids"]
                                           )
                    loss = model_output["loss"] / iter_size
                    loss.backward()
            else:
                model_output = trainer(batch["text_masked_input_ids"], batch["text_masked_attention_mask"],
                                       batch["tabular_student_masked_input_ids"],
                                       batch["tabular_student_masked_attention_mask"],
                                       batch["images"], batch["panel_distribution"], batch["image_masks"], batch["image_labels"],
                                       batch["text_input_ids"], batch["tabular_student_input_ids"]
                                       )
                loss = model_output["loss"] / iter_size
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                if isinstance(scheduler, list):
                    for sch in scheduler:
                        sch.step()
                else:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # model.zero_grad(set_to_none=True)
                steps_done += 1
            if validation_iter or (step + 1) % iter_size == 0:
                metric_logger.update(loss=model_output["loss"])
                metric_logger.update(lr=optimizer.param_groups[0]['lr'])
                metric_logger.update(tabular_mlm_accuracy=model_output["tabular_mlm_accuracy"])
                metric_logger.update(mlm_accuracy=model_output["mlm_accuracy"])
                metric_logger.update(mlm_loss=model_output["mlm_loss"])
                metric_logger.update(tabular_mlm_loss=model_output["tabular_mlm_loss"])
                metric_logger.update(image_mlm_loss=model_output["image_mlm_loss"])
                metric_logger.update(contrastive_loss=model_output["contrastive_loss"])

            loss_float = model_output["loss"].item()
            if np.isnan(loss_float):
                es = "[Train-Exception]: Time = %s, NAN Loss, loss_dict = %s, lr = %s" % (
                    get_time_string(), model_output, optimizer.param_groups[0]['lr'])
                torch.save(batch, os.path.join(model_save_dir, "bad-batch-%s.pth" % local_rank))
                raise ValueError(es)

            step += 1
            del batch
            steps_remaining = total_steps - steps_done
            output = {k: float(v) for k, v in model_output.items() if try_float(v) and v is not None}
            wandb_log = dict(lr=optimizer.param_groups[0]['lr'], step=step, updates_done=steps_done,
                             samples_processed=samples_processed,
                             steps_remaining=steps_remaining, pct_complete=(100 * steps_done / total_steps),
                             epoch=epoch,
                             **output)
            logs_save.append(pd.DataFrame.from_records([wandb_log]).T)
            if validation_iter:
                # clean_memory()
                printed = pd.concat(logs_save, axis=1)
                printed["mean"] = printed.mean(1)
                logs_save = []
                metric_logger.synchronize_between_processes()
                if local_rank == 0:
                    # print(json.dumps(dict(time=get_time_string(), **wandb_log), skipkeys=True, indent=2, sort_keys=True))
                    print(("[Time = %s]" % get_time_string()) + ("-" * 80))
                    print(printed)
                    print("Averaged stats: \n", metric_logger)
                    metric_logger.plot()
                if activate_wandb_log:
                    time.sleep(random.random() * 0.1)
                    wandb.log(wandb_log)
            del output
            del model_output
        save_model(model_save_dir, model, optimizer, scheduler, metric_logger, local_rank, steps_done)
    print("Time = %s, Finished Training for Rank = %s" % (get_time_string(), rank))
    save_model(model_save_dir, model, optimizer, scheduler, metric_logger, local_rank, steps_done)
    metric_logger.synchronize_between_processes()
    if args["world_size"] > 1:
        dist.destroy_process_group()

def train_catch_exception(local_rank, args):
    rank = args["nr"] * args["gpus_per_node"] + local_rank
    nr = args["nr"]
    try:
        train(local_rank, args)
    except Exception as e:
        import traceback
        print("[Exception-in-train]: Node Rank = %s, Local Rank = %s, Rank = %s, Exception = %s, \n Trace = %s" % (nr, local_rank, rank, e, traceback.format_exc()))
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = training_args()
    if args["world_size"] == 1 or args["cpu"]:
        train_catch_exception(0, args)
    else:
        mp.spawn(train_catch_exception, nprocs=args["gpus_per_node"], args=(args,), join=True)
