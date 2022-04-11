import argparse
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
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedModel, RobertaConfig, \
    RobertaTokenizerFast
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
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaModel
from einops import rearrange

from fastformer.utils import set_seeds, get_barrier, init_weights

image_size = 384
max_length = 512
image_patch_size = 32
image_grid = 12

tokenizer_args=dict(padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



image_shape_augments = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5,),
        transforms.RandomRotation(15, expand=True,),
        transforms.RandomAffine(0, translate=None, scale=None, shear=[-5, 5, -5, 5],)
])

train_image_augments = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5,),
    transforms.RandomRotation(15, expand=True,),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.1), shear=[-5, 5, -5, 5], fill=120),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomApply([transforms.GaussianBlur(7)], 0.25),
    transforms.RandomChoice([transforms.TrivialAugmentWide(),
                                 transforms.RandomAutocontrast(p=1.0),
                                 transforms.RandomAdjustSharpness(2, p=1.0),
                                 transforms.RandomAdjustSharpness(4, p=1.0),
                                 transforms.RandomAdjustSharpness(16, p=1.0),
                                 transforms.RandomAdjustSharpness(32, p=1.0),
                                 transforms.RandomPosterize(bits=3, p=1.0),
                                 transforms.RandomPosterize(bits=4, p=1.0),
                                 transforms.GaussianBlur(7, sigma=(0.5, 4.0)),
                                 transforms.GaussianBlur(21, sigma=(0.5, 4.0))],)
])


inference_image_shape_augments = transforms.Compose([
        transforms.Resize(image_size+image_patch_size//2),
        transforms.CenterCrop(image_size),
])

panel_combine_resize = transforms.Compose([
        transforms.Resize((image_size+image_patch_size//2)//2),
        transforms.CenterCrop(image_size//2),
])

def build_2d_sincos_position_embedding(embed_dim, cls_tokens=0, grid_size=image_grid, temperature=10000.):
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
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=0))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

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


def text_masking(text: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], mask_probability: float = 0.15):
    mask = tokenizer.mask_token
    words = text.split()
    masked_words = []
    for ix, w in enumerate(words):
        if random.random() < mask_probability:
            tokens = tokenizer.tokenize((w if ix == 0 else " "+w))
            ln = len(tokens)
            # print(w, ln, tokens)
            masked_words.extend([mask] * ln)
        else:
            masked_words.append(w)
    return " ".join(masked_words), " ".join(words)


class MultiModalTrainingDataset(Dataset):
    """
    Expects a header-less csv.
    Handle if a particular modality is not present at all.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, tokenizer_args, data_csv, separator,
                 columns, text_columns, tabular_columns, image_columns,
                 image_size, image_patch_size, image_augments, image_to_vector=transforms.ToTensor(),
                 training=True,
                 word_mask_proba=0.15, image_mask_proba=0.75, tabular_feature_mask_proba=0.15, tabular_feature_drop_proba=0.1, save_one_image=True,
                 total_image_panels=5,
                 ):
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.separator = separator
        self.columns = columns
        self.data_csv = data_csv
        self.text_columns = text_columns
        self.image_columns = image_columns
        self.joiner = f" {tokenizer.sep_token} "
        self.tabular_columns = tabular_columns
        self.training = training
        self.tabular_feature_drop_proba = tabular_feature_drop_proba
        self.tabular_feature_mask_proba = tabular_feature_mask_proba
        self.image_augments = image_augments
        self.image_size = image_size
        self.save_one_image = save_one_image
        self.image_to_vector = transforms.ToTensor()
        self.imagenet_normalization = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                                        std=torch.tensor(IMAGENET_DEFAULT_STD))
        self.total_image_panels = total_image_panels
        self.word_mask_proba = word_mask_proba
        self.image_mask_proba = image_mask_proba
        self.image_patch_size = image_patch_size
        assert self.image_size % self.image_patch_size == 0
        self.num_patches = (self.image_size // self.image_patch_size) ** 2
        self.num_mask = int(image_mask_proba * self.num_patches)
        self.length = len(self)

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
        text_zipped_ids = list(zip(x["text_input_ids"][:x["text_attention_mask"].sum()], x["text_masked_input_ids"][:x["text_masked_attention_mask"].sum()]))
        tabular_coincide = x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()] == x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()]
        tabular_zipped_ids = list(zip(x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()],
                           x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()]))
        mean = torch.as_tensor(self.imagenet_normalization.mean)[None, :, None, None]
        std = torch.as_tensor(self.imagenet_normalization.std)[None, :, None, None]
        input_text = tokenizer.decode(x["text_input_ids"][:x["text_attention_mask"].sum()])
        masked_text = tokenizer.decode(x["text_masked_input_ids"][:x["text_masked_attention_mask"].sum()])

        student_input_tabular = tokenizer.decode(x["tabular_student_input_ids"][:x["tabular_student_attention_mask"].sum()])
        student_input_masked_tabular = tokenizer.decode(
            x["tabular_student_masked_input_ids"][:x["tabular_student_masked_attention_mask"].sum()])
        input_tabular = tokenizer.decode(
            x["tabular_teacher_input_ids"][:x["tabular_teacher_attention_mask"].sum()])
        generated_image_actual = None
        sketch_components_of_generated = None
        if x["generated_image"] is not None:
            generated_image_actual = x["generated_image"][3:] * std[0] + mean[0]
            sketch_components_of_generated = (x["generated_image"][:3].permute(1,2,0) * 255).clip(0, 255).numpy().astype(np.uint8)
            generated_image_actual = (generated_image_actual.permute(1, 2, 0) * 255).clip(0, 255).numpy().astype(np.uint8)
            generated_image_actual = Image.fromarray(generated_image_actual)
            sketch_components_of_generated = Image.fromarray(sketch_components_of_generated)
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
        all_patch = rearrange(all_patch, 'b (h w) p1 p2 c -> b (h p1) (w p2) c', h=image_grid, w=image_grid)
        all_patch = (all_patch.permute(0, 3, 1, 2) * std + mean).permute(0, 2, 3, 1)

        actual_images = actual_images * std + mean
        actual_images = [(x.permute(1, 2, 0) * 255).clip(0, 255).numpy().astype(np.uint8) for x in actual_images]
        all_patch = [(x * 255).clip(0, 255).numpy().astype(np.uint8) for x in all_patch]
        actual_images = [Image.fromarray(x) for x in actual_images]
        all_patch = [Image.fromarray(x) for x in all_patch]
        return dict(input_text=input_text, masked_text=masked_text, sketch_components_of_generated=sketch_components_of_generated,
                    text_coincide=text_coincide, text_zipped_ids=text_zipped_ids, tabular_coincide=tabular_coincide, tabular_zipped_ids=tabular_zipped_ids,
                    student_input_tabular=student_input_tabular, student_input_masked_tabular=student_input_masked_tabular,
                    input_tabular=input_tabular, generated_image_actual=generated_image_actual, all_patch=all_patch, actual_images=actual_images)


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
        item = pd.read_csv(self.data_csv, names=self.columns, sep=self.separator, low_memory=False, skiprows=1, nrows=1,
                           header=0)
        text = item[self.text_columns].values[0]
        text = self.joiner.join(text)
        tabular = list(zip(self.tabular_columns, list(item[self.tabular_columns].to_records()[0])[1:]))
        image_locations = item[self.image_columns].values[0]
        image_locations = " ".join(image_locations)
        image_locations = list(image_locations.split())
        return dict(item=item, text=text, tabular=tabular, image_locations=image_locations)


    def __getitem__(self, item):

        tokenizer = self.tokenizer
        mask = self.tokenizer.mask_token
        item = pd.read_csv(self.data_csv, names=self.columns, sep=self.separator, low_memory=False, skiprows=1, nrows=1,
                           header=0)
        text = item[self.text_columns].values[0]
        text = self.joiner.join(text)
        masked_text, text = text_masking(text, tokenizer, self.word_mask_proba)
        tokenizer_outputs = tokenizer(text, return_offsets_mapping=False, **self.tokenizer_args)
        text_input_ids, text_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()

        tokenizer_outputs = tokenizer(masked_text, return_offsets_mapping=False, **self.tokenizer_args)
        text_masked_input_ids, text_masked_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()
        assert text_masked_attention_mask.sum() == text_attention_mask.sum()


        tabular = list(zip(self.tabular_columns, list(item[self.tabular_columns].to_records()[0])[1:]))
        tabular_to_text_for_teacher = ""
        for k, v in tabular:
            tabular_to_text_for_teacher = tabular_to_text_for_teacher + " " + k + " = " + float_format(v) + " ;"
        tabular_to_text_for_student_input = ""
        tabular_to_text_for_student_output = ""
        for k, v in tabular:
            if random.random() < self.tabular_feature_drop_proba:
                continue
            tabular_to_text_for_student_input = tabular_to_text_for_student_input + " " + k + " = " + (" ".join([mask] * len(tokenizer.tokenize(" " + float_format(v)))) if random.random() < self.tabular_feature_mask_proba else float_format(v)) + " ;"
            tabular_to_text_for_student_output = tabular_to_text_for_student_output + " " + k + " = " + float_format(v) + " ;"

        tokenizer_outputs = tokenizer(tabular_to_text_for_teacher, return_offsets_mapping=False, **self.tokenizer_args)
        t2t_teacher_input_ids, t2t_teacher_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()


        tokenizer_outputs = tokenizer(tabular_to_text_for_student_output, return_offsets_mapping=False, **self.tokenizer_args)
        t2t_student_input_ids, t2t_student_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()

        tokenizer_outputs = tokenizer(tabular_to_text_for_student_input, return_offsets_mapping=False, **self.tokenizer_args)
        t2t_student_masked_input_ids, t2t_student_masked_attention_mask = tokenizer_outputs["input_ids"].squeeze(), tokenizer_outputs[
            "attention_mask"].squeeze()
        assert t2t_student_masked_attention_mask.sum() == t2t_student_attention_mask.sum()
        assert t2t_teacher_attention_mask.sum() >= t2t_student_attention_mask.sum()

        image_locations = item[self.image_columns].values[0]
        image_locations = " ".join(image_locations)
        image_locations = list(image_locations.split())  # Assuming all images are separated in their columns by space
        count_images = len(image_locations)
        random.shuffle(image_locations)
        image_locations = list(map(pil_loader, image_locations))

        one_image = None
        if self.save_one_image and count_images > 0:
            one_image = image_locations.pop()
            one_image = inference_image_shape_augments(one_image)
            one_image_p1 = torch.tensor(np.stack([canny_edge_detector(one_image), gray_scale(one_image),
                                                  sketch_transform(one_image)]))
            one_image_p2 = self.imagenet_normalization(self.image_to_vector(one_image))
            one_image = torch.cat([one_image_p1, one_image_p2], 0)
        image_locations = list(map(self.image_augments, image_locations))
        total_image_panels = self.total_image_panels
        num_images = len(image_locations)
        if num_images > total_image_panels:
            extra_images = num_images - total_image_panels
            panel_distribution = [1] * total_image_panels
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
                    sample_panel = np.random.randint(0, 255, (image_size, image_size, 3)).astype(np.uint8)
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
            assert len(image_locations) == total_image_panels

        if len(image_locations) < self.total_image_panels:
            image_locations.extend([Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3)).astype(np.uint8)) for _ in range(self.total_image_panels - len(image_locations))])
        image_locations = list(map(self.image_to_vector, image_locations))
        image_locations = list(map(self.imagenet_normalization, image_locations))
        image_inputs = torch.tensor(np.stack(image_locations))


        masks = [self.__get_image_mask__() for _ in range(len(image_locations))]
        image_masks = torch.tensor(np.stack(masks)).bool()
        image_locations = torch.tensor(np.stack(image_locations))
        images_squeeze = rearrange(image_locations, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=image_patch_size,
                                   p2=image_patch_size)
        images_norm = images_squeeze
        # images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
        B, _, C = images_patch.shape
        image_labels = images_patch.view(-1, C)[image_masks.view(-1)].reshape(B, -1, C)  # 2D indexing isn't working so bring index to 1D

        # assert (text_attention_mask.sum() == text_masked_attention_mask.sum()).item()
        # assert (tabular_student_attention_mask.sum() == tabular_student_masked_attention_mask.sum()).item()
        return dict(num_images=num_images, image_labels=image_labels,
                    image_masks=image_masks, images=image_inputs, generated_image=one_image,
                    tabular_student_masked_input_ids=t2t_student_masked_input_ids,
                    tabular_student_masked_attention_mask=t2t_student_masked_attention_mask,
                    tabular_student_input_ids=t2t_student_input_ids,
                    tabular_student_attention_mask=t2t_student_attention_mask,
                    tabular_teacher_input_ids=t2t_teacher_input_ids,
                    tabular_teacher_attention_mask=t2t_teacher_attention_mask,
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
    def __init__(self, model_size="large",):
        super().__init__(LongformerConfig.from_pretrained("allenai/longformer-large-4096" if model_size == "large" else "allenai/longformer-base-4096"))
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
            mid_fusion_backbone_config.num_hidden_layers = 3
        elif model_size == "base":
            longformer = RobertaModel.from_pretrained("roberta-base")
            # longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")
            # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            embed_dim = 768
            vit = vit_base_patch32_384(True)
            mid_fusion_backbone_config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
            mid_fusion_backbone_config.num_hidden_layers = 3
        else:
            raise ValueError
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer

        # self.text_ffn = LongformerFFN(mid_fusion_backbone_config)
        # self.tabular_ffn = LongformerFFN(mid_fusion_backbone_config)
        # self.image_ffn = LongformerFFN(mid_fusion_backbone_config)
        self.mid_fusion_backbone = LongformerEncoder(config=mid_fusion_backbone_config)
        decoder_layer_conf = RobertaConfig()
        decoder_layer_conf.hidden_size = embed_dim
        decoder_layer_conf.num_attention_heads = 16 if model_size == "large" else 12
        decoder_layer_conf.add_cross_attention = True
        decoder_layer_conf.is_decoder = True
        self.decoder = nn.ModuleList([RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf)])
        self.decoder_head = nn.Sequential(LongformerFFN(mid_fusion_backbone_config), nn.Linear(embed_dim, image_patch_size*image_patch_size*3))
        self.decoder_sketch_head = nn.Sequential(LongformerFFN(mid_fusion_backbone_config),
                                                 nn.Linear(embed_dim, image_patch_size * image_patch_size * 3))
        self._init_weights()
        decoder_query = longformer.embeddings.position_embeddings.weight[:144, :decoder_layer_conf.hidden_size].detach().clone()
        self.decoder_inputs = torch.nn.Parameter(decoder_query)
        self.grad_checkpointing = False

        self.longformer = longformer
        self.vit = vit  # TODO: checkout get_pretrained_deit from utils, forward_features from vit.

    def vit_forward(self, x, mask):
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        else:
            x_vis = x

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x_vis = checkpoint_seq(self.vit.blocks, x_vis)
        else:
            x_vis = self.vit.blocks(x_vis)
        x_vis = self.vit.norm(x_vis)
        return x_vis

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LongformerEncoder, MultiModalEncoder)):
            module.gradient_checkpointing = value
        self.grad_checkpointing = True

    def forward(self, input_ids=None, attention_mask=None,
                tabular_input_ids=None, tabular_attention_mask=None,
                images=None, mask=None, activate_missing_image_generator: bool = True):
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
        @param activate_missing_image_generator:
        @type activate_missing_image_generator: Boolean
        @return: dict(text_embedding [both text+tabular], images_embedding [(Batch x Images Per Example) x S x d],
        text_mem_tokens, image_mem_tokens, missing_image_generator_tokens, image_mask_locations)
        @rtype:
        """

        # Assert at least one modality present.
        assert images is not None or input_ids is not None or tabular_input_ids is not None

        if input_ids is not None and tabular_input_ids is not None:
            assert input_ids.size(1) == tabular_input_ids.size(1)
            lf_input_ids = torch.cat([input_ids, tabular_input_ids], 0)
            lf_attention_mask = torch.cat([attention_mask, tabular_attention_mask], 0)
            tabular_text_output = self.longformer(input_ids=lf_input_ids, attention_mask=lf_attention_mask, )["last_hidden_state"]
            text_output = tabular_text_output[:input_ids.size(0)]
            tabular_output = tabular_text_output[input_ids.size(0):]
            tabular_text_output = torch.cat([text_output, tabular_output], 1)
        elif input_ids is not None and tabular_input_ids is None:
            tabular_text_output = self.longformer(input_ids=input_ids, attention_mask=attention_mask, )[
                "last_hidden_state"]
        elif input_ids is None and tabular_input_ids is not None:
            tabular_text_output = self.longformer(input_ids=tabular_input_ids, attention_mask=tabular_attention_mask, )[
                "last_hidden_state"]
        else:
            tabular_text_output = None

        if images is not None:
            b, ex, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            image_features = self.vit_forward(images, mask)
        else:
            image_features = None

        assert image_features is not None or tabular_text_output is not None

        if image_features is None:
            features = tabular_text_output
        elif tabular_text_output is None:
            features = image_features
        else:
            features = torch.cat([tabular_text_output, image_features], 1)

        features = self.mid_fusion_backbone(features)[0]
        image_out = None
        if image_features is not None:
            image_out = features[:, -image_features.size(1):]
            image_out = image_out.reshape(b, ex, *image_out[1:])
            image_features = image_features.reshape(b, ex, *image_features[1:])

        text_output = None
        tabular_output = None
        text_features = None
        tabular_features = None
        if tabular_text_output is not None:
            tabular_text_final = features[:, :tabular_text_output.size(1)]
            if input_ids is not None:
                text_output = tabular_text_final[:, :input_ids.size(1)]
                text_features = tabular_text_output[:, :input_ids.size(1)]
                assert text_output.size(1) == text_features.size(1) == input_ids.size(1)
            if tabular_input_ids is not None:
                tabular_output = tabular_text_final[:, -tabular_input_ids.size(1):]
                tabular_features = tabular_text_output[:, -tabular_input_ids.size(1):]
                assert tabular_output.size(1) == tabular_features.size(1) == tabular_input_ids.size(1)

        full_reconstruction = None
        sketch_reconstruction = None
        if activate_missing_image_generator:
            assert images is not None
            decoder_output = self.decoder_inputs.expand(features.shape[0], -1, -1)
            for dec in self.decoder:
                decoder_output = dec(decoder_output, encoder_hidden_states=features)[0]
            # generate actual image by reshape-ing
            full_reconstruction = self.decoder_head(decoder_output)  # bx144*image_patch_size*image_patch_size*3
            # rearrange(full_reconstruction, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_grid, w=image_grid, p1=image_patch_size, p2=image_patch_size)
            full_reconstruction = full_reconstruction.reshape(b, image_grid, image_grid, image_patch_size*image_patch_size*3).reshape(b, image_grid, image_grid, image_patch_size, image_patch_size, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, image_grid*image_patch_size, image_grid*image_patch_size)
            sketch_reconstruction = self.decoder_sketch_head(decoder_output)
            sketch_reconstruction = sketch_reconstruction.reshape(b, image_grid, image_grid, image_patch_size*image_patch_size*3).reshape(b, image_grid, image_grid, image_patch_size, image_patch_size, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, image_grid*image_patch_size, image_grid*image_patch_size)
        return dict(full_reconstruction=full_reconstruction, sketch_reconstruction=sketch_reconstruction,
                    image_output=image_out, text_output=text_output, tabular_output=tabular_output,
                    unimodal_image_features=image_features, unimodal_text_features=text_features,
                    unimodal_tabular_features=tabular_features,)


class MultiModalSelfSupervisedTrainerModel(LongformerPreTrainedModel):
    # contains both teacher and student
    # Teacher's missing image generator part is not executed
    # Contains Image and Text MLM decoder as well.
    # contains Loss fn for missing image generator, Image and Text MLM
    # Build Mask here
    # Dropout modality rate -> used with student teacher
    # For canny, gray_scale and sketch transforms of output image
    # Weigh non-zero elements in image generator output for loss more highly by loss = loss.mean() + loss[non_zero_loc].mean(),
    def __init__(self, teacher_student_loss_w, image_mlm_w, text_mlm_w, tabular_mlm_w, image_generation_w,
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
        self.image_generation_w = image_generation_w
        decoder_embed_dim = self.encoder.embed_dim // 2
        self.encoder_to_decoder = nn.Linear(self.encoder.embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        init_weights(self.encoder_to_decoder)

        self.pos_embed = get_sinusoid_encoding_table(image_grid * image_grid, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        decoder_layer_conf = RobertaConfig()
        decoder_layer_conf.hidden_size = decoder_embed_dim
        decoder_layer_conf.num_attention_heads = 16 if self.encoder.model_size == "large" else 12
        decoder_layer_conf.add_cross_attention = False
        decoder_layer_conf.is_decoder = False
        self.decoder = nn.ModuleList(
            [RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf), RobertaLayer(decoder_layer_conf)])
        init_weights(self.decoder)
        self.decoder_head = nn.Linear(decoder_embed_dim, image_patch_size*image_patch_size*3)
        trunc_normal_(self.decoder_head, std=.02)
        self.mse = nn.MSELoss()

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

    def image_mlm_forward(self, x_vis, mask, image_unmasked_patches):
        mask = mask.view(-1, mask.shape[2:])
        x_vis = self.encoder_to_decoder(x_vis).reshape(-1, x_vis.shape[2:])
        image_unmasked_patches = image_unmasked_patches.view(-1, image_unmasked_patches.shape[2:])
        B, N, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        for blk in self.decoder:
            x_full = blk(x_full)[0]
        mask_count = pos_emd_mask.shape[1]
        if mask_count > 0:
            x_full = self.decoder_head(x_full[:, -mask_count:])
        else:
            x_full = self.decoder_head(x_full)
        loss = self.mse(input=x_full, target=image_unmasked_patches)
        return loss

    def forward(self, input_ids, attention_mask,
                tabular_input_ids, tabular_attention_mask,
                images=None, image_masks=None, image_labels=None,
                label_input_ids=None, label_tabular_input_ids=None, generated_image=None):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   tabular_input_ids=tabular_input_ids,
                                   tabular_attention_mask=tabular_attention_mask, images=images, mask=image_masks)
        # TODO: masked and non-masked tokens must coincide properly
        masked_lm = input_ids == self.mask_token_id
        lm_feats = (encoder_out["text_output"] + encoder_out["unimodal_text_features"])[masked_lm]
        label_input_ids = label_input_ids[masked_lm]
        lm_out = self.lm_head(self.lm_ffn(lm_feats))
        mlm_loss = self.mlm_w * self.mlm_ce(lm_out, label_input_ids)
        mlm_accuracy = (lm_out.argmax(dim=-1) == label_input_ids).float().mean().item()

        masked_tabular = tabular_input_ids == self.mask_token_id
        tabular_feats = (encoder_out["tabular_output"] + encoder_out["unimodal_tabular_features"])[masked_tabular]
        label_tabular_input_ids = label_tabular_input_ids[masked_tabular]
        tabular_lm_out = self.lm_head(self.lm_ffn(tabular_feats))
        tabular_mlm_loss = self.tabular_mlm_w * self.mlm_ce(tabular_lm_out, label_tabular_input_ids)
        tabular_mlm_accuracy = (tabular_lm_out.argmax(dim=-1) == label_tabular_input_ids).float().mean().item()

        image_mlm_loss = self.image_mlm_w * self.image_mlm_forward(encoder_out["unimodal_image_features"] + encoder_out["image_output"],
                                                image_masks, image_labels)
        reconstruction = torch.cat([encoder_out["sketch_reconstruction"], encoder_out["full_reconstruction"]], 1)
        reconstruction_loss = self.image_generation_w * self.mse(input=reconstruction, output=generated_image)
        loss = mlm_loss + tabular_mlm_loss + image_mlm_loss + reconstruction_loss

        return dict(loss=loss, tabular_mlm_accuracy=tabular_mlm_accuracy, mlm_accuracy=mlm_accuracy,
                    mlm_loss=mlm_loss, tabular_mlm_loss=tabular_mlm_loss,
                    image_mlm_loss=image_mlm_loss, reconstruction_loss=reconstruction_loss)

        # TODO: to optimize tabular we need to write separate collate fn. For starters keep text size and table size = 512.


optimizer_config = dict(lr=1e-4, eps=1e-8, weight_decay=1e-3, beta_1=0.9, beta_2=0.98, gradient_clipping=1.0)


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

    parser.add_argument('--total_steps', type=int, required=False,
                        help='total_steps')

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
    parser.add_argument('--model_save_name', required=True, type=str,
                        help='Save Name')

    parser.add_argument('--wandb_dryrun', action="store_true", default=False,
                        help='WanDB Dryrun Only')

    parser.add_argument('--sentence_order_w', type=float, required=False, default=1.0,
                        help='sentence_order weight')

    parser.add_argument('--text_mlm_w', type=float, required=False, default=1.0,
                        help='text_mlm_w weight')
    parser.add_argument('--image_generation_w', type=float, required=False, default=1.0,
                        help='image_generation_w weight')
    parser.add_argument('--tabular_mlm_w', type=float, required=False, default=1.0,
                        help='tabular_mlm_w weight')
    parser.add_argument('--image_mlm_w', type=float, required=False, default=1.0,
                        help='image_mlm_w weight')

    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Train on CPU')

    parser.add_argument('--detect_anomaly', action="store_true", default=False,
                        help='AutoGrad Anomaly detection')

    parser.add_argument('--optimizer', required=False, type=str, default="adamw",
                        help='optimizer')

    parser.add_argument('--num_workers', required=False, type=int, default=4,
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

    parser.add_argument('--dataset', required=False, type=str,
                        help='Dataset')

    args = parser.parse_args()
    args.world_size = args.nodes if args.cpu else (args.gpus_per_node * args.nodes)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    seed = 79739567
    args.seed = seed
    return vars(args)

def build_propreitery_dataset(location, tokenizer):
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
    dataset = MultiModalTrainingDataset(tokenizer, tokenizer_args, location, ",", COLUMNS, textual, tabular,
                                        image_columns,
                                        image_size, image_patch_size, train_image_augments)
    return dataset

def build_propreitery_dataloader(location, batch_size, tokenizer, world_size=1, num_workers=None):
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
    dataset = MultiModalTrainingDataset(tokenizer, tokenizer_args, location, ",", COLUMNS, textual, tabular,
                                        image_columns,
                                        image_size, image_patch_size, train_image_augments)
    kwargs = dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else dict()
    sampler = None if single_node else DistributedSampler(dataset, shuffle=True)
    train_loader = DataLoader(dataset, sampler=sampler,
                              batch_size=batch_size, shuffle=single_node,
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # gpu_device = 0
    gpu_device = local_rank
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_device = local_rank
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    trainer = MultiModalSelfSupervisedTrainerModel(0, args["image_mlm_w"], args["text_mlm_w"],
                                                   args["text_mlm_w"], args["image_generation_w"], encoder)
    if "load_encoder" in args:
        encoder_weights = torch.load(args["load_encoder"], map_location='cpu')
        encoder.load_state_dict(encoder_weights)

    if "load_trainer_model" in args:
        trainer_weights = torch.load(args["load_trainer_model"], map_location='cpu')
        trainer.load_state_dict(trainer_weights)

    model = model.train()
    if args["world_size"] > 1:
        model = DDP(model, device_ids=None if args["cpu"] else [gpu_device], find_unused_parameters=False,
                    bucket_cap_mb=10, gradient_as_bucket_view=True)  # find_unused_parameters=True
    clean_memory()
    barrier()
    optc = copy.deepcopy(optimizer_config)
    model.zero_grad(set_to_none=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    torch.optim.AdamW(optimizer_grouped_parameters, **dict(lr=optc["lr"], eps=optc["eps"], weight_decay=optc["weight_decay"], betas=(optc["beta_1"], optc["beta_2"])))
    optimizer.zero_grad(set_to_none=True)

    model_save_dir = args["model_save_dir"]
    model_save_name = args["model_save_name"]

    set_seeds(args["seed"] + rank)
    if local_rank == 0:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        assert os.path.exists(model_save_dir)









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
