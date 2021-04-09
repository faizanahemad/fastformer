from typing import List, Dict

from transformers import PretrainedConfig


# TODO: check if all heads are aligned
# TODO: check if repeats are happening properly for layers
# TODO: check if upsampling (interleaving) is happening properly
# TODO: Fix parameter initialization
# TODO: separate static model structure config from dynamic training config

from dataclasses import dataclass
from dataclasses_json import dataclass_json

# size_dicts = {128: 16, 192: 8, 256:8, 512: 4, 768: 2, 1024: 2}
# size_dicts = {128: 24, 192: 16, 256: 12, 384: 8, 512: 8, 768: 4, 1024: 4}
# size_dicts = {128: 32, 192: 32, 256: 24, 384: 16, 512: 16, 768: 8, 1024: 8}
# size_dicts = {128: 16, 192: 8, 256: 8, 384: 4, 512: 4, 768: 2, 1024: 2}
# size_dicts = {128: 36, 192: 32, 256: 28, 384: 20, 512: 16, 640: 12, 768: 8, 896: 6, 1024: 4}
# size_dicts = {128: 24, 192: 12, 256: 12, 384: 12, 512: 12, 640: 12, 768: 12, 896: 12, 928: 12, 1024: 12}


def get_batch_size(size, autocast):
    size_dicts = {1024: 16}
    if not autocast:
        size_dicts = {k: v // 2 for k, v in size_dicts.items()}
    if size == "lg_config" or size == "tg_config":
        size_dicts = {k: v // 2 for k, v in size_dicts.items()}
    return size_dicts


@dataclass_json
@dataclass
class OptimizerConfig:
    lr: float
    eps: float
    weight_decay: float
    beta_1: float
    beta_2: float

    batch_size: int
    test_batch_size: int
    warmup_steps: int
    gradient_clipping: float


optimizer_config = OptimizerConfig(1e-5, 1e-4, 1e-4, 0.9, 0.98,
                                   8, 8, 40000, 0.1)


@dataclass_json
@dataclass
class ModelConfig:
    tokenizer_name: str
    model_size: str
    aitm: bool
    alum: bool
    adv_lm_w: float
    adv_ascent_steps: int
    aitm_clip_min: float
    aitm_clip_max: float
    adv_step_size: float
    adv_epsilon: float
    aitm_noise_var: float
    adv_w: float
    alum_aitm_alternate: bool
    input_cls_orthogonal_w: float
    first_block_cls_orthogonal_w: float
    electra_loss_w: float
    lm_loss_w: float
    sentence_order_prediction_w: float
    contrastive_w: float
    contrastive_temperature: float
    answering_lm_w: float
    additive_margin_softmax_w: float


model_config: ModelConfig = ModelConfig("bert", "md_config", aitm=False, alum=False,
                                        adv_lm_w=1.0, adv_ascent_steps=1, aitm_clip_min=0.1, aitm_clip_max=0.9, adv_step_size=1e-3,
                                        adv_epsilon=1e-2, aitm_noise_var=0.1, adv_w=1.0, alum_aitm_alternate=False,
                                        input_cls_orthogonal_w=0.0, first_block_cls_orthogonal_w=0.0,
                                        electra_loss_w=0.0, lm_loss_w=0.0, sentence_order_prediction_w=0.0, contrastive_w=0.0, contrastive_temperature=1e-2,
                                        answering_lm_w=2.0, additive_margin_softmax_w=0.1)


@dataclass_json
@dataclass
class FastFormerConfig(PretrainedConfig):

    def __init__(
            self,
            vocab_size=30522 + 22,
            block_sizes=[6, 6, 6],
            block_channel_size=[576, 768, 960],  # [512, 768, 1024]
            block_repeats=True,
            separate_compressiion_layer=False,
            num_decoder_layers=1,
            n_head=[(8,), (12,), (12,)],  # 8
            use_cuda_conv=True,
            d_head=[72, 64, 80],  # 32
            hidden_act="gelu",
            hidden_dropout=0.1,
            attention_dropout=0.1,
            max_position_embeddings=512,
            type_vocab_size=0,
            initializer_range=0.1,
            initializer_std=0.1,
            layer_norm_eps=1e-4,
            pooling_type="mean",  # learn, #learn_sdconv
            pooling_kernel_size=5,
            stride=2,
            ffn_groups=1,
            ffn_layers=0,
            ffn_width=4,
            qkv_transform_groups=1,
            embedding_size=128,
            num_highway_cls_tokens=7,
            position_biased_input=True,
            separate_content_and_position_attention=False,
            relative_attention=False,
            approximate_attention=False,
            compress_query_method=None,
            compressed_query_attention_kernel_size=3,
            compressed_query_attention_stride=2,
            compressed_query_attention_layers=[],
            compressed_key_attention_layers=[],
            sdconv=[False, False, False],
            sdconv_kernel_size=[5, 7, 9],
            full_channel_separation=[False, False, False],
            conv_layer_use_dynamic_conv=False,
            no_v_head=False,
            expand_dim_before_pooling=False,
            identity_preserving_norm=True,
            char_rnn=False,
            char_rnn_layers=1,
            char_rnn_vocab_size=1024,
            char_rnn_window_size=512,
            char_rnn_window_overlap=64,
            
            img_size=224,
            patch_size=16,
            in_chans=3,
            **kwargs
    ):
        super().__init__(**kwargs)
        try:
            from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
        except:
            use_cuda_conv = False
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.tokenizer_length = max_position_embeddings - num_highway_cls_tokens
        assert all([bsz % 2 == 0 for bsz in block_sizes])
        self.block_sizes = block_sizes
        self.block_repeats = block_repeats
        self.separate_compressiion_layer = separate_compressiion_layer
        self.num_decoder_layers = num_decoder_layers
        self.n_head = [(n_head,)] * len(block_sizes) if isinstance(n_head, int) else n_head
        self.n_head = [h if isinstance(h, (list, tuple)) else (h,) for h in self.n_head]
        assert all([d % sum(h) == 0 for h, d in zip(self.n_head, block_channel_size)])
        assert len(self.n_head) == len(block_sizes)
        self.d_head = [d_head] * len(block_sizes) if isinstance(d_head, int) else d_head
        assert len(self.d_head) == len(block_sizes)
        self.char_rnn = char_rnn
        self.char_rnn_layers = char_rnn_layers
        self.char_rnn_vocab_size = char_rnn_vocab_size
        self.char_rnn_window_overlap = char_rnn_window_overlap
        self.char_rnn_window_size = char_rnn_window_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        self.ffn_width = ffn_width
        self.pooling_kernel_size = pooling_kernel_size
        assert pooling_type in [

            "mean",
            "max",
            "learn",
            "learn_sdconv",
            'learn_rnn',
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
        assert compress_query_method in [
            None,
            "mean",
        ], f"Got {pooling_type} for `compress_query_method`"
        self.pooling_type = pooling_type
        self.compress_query_method = compress_query_method
        self.expand_dim_before_pooling = expand_dim_before_pooling
        self.ffn_groups = ffn_groups
        self.ffn_layers = ffn_layers
        self.qkv_transform_groups = qkv_transform_groups
        self.embedding_size = embedding_size
        self.position_biased_input = position_biased_input
        self.num_highway_cls_tokens = num_highway_cls_tokens
        self.approximate_attention = approximate_attention
        assert compressed_query_attention_kernel_size in [3, 5, 7, 9]
        self.compressed_query_attention_kernel_size = compressed_query_attention_kernel_size
        assert compressed_query_attention_stride in [1, 2, 4]
        self.compressed_query_attention_stride = compressed_query_attention_stride
        self.compressed_query_attention_layers = compressed_query_attention_layers
        self.compressed_key_attention_layers = compressed_key_attention_layers
        self.separate_content_and_position_attention = separate_content_and_position_attention
        assert separate_content_and_position_attention or position_biased_input
        self.stride = stride
        assert len(block_channel_size) == len(block_sizes)
        self.block_channel_size = block_channel_size

        self.sdconv = [sdconv] * len(block_sizes) if isinstance(sdconv, bool) else sdconv
        self.full_channel_separation = [full_channel_separation] * len(block_sizes) if isinstance(full_channel_separation, bool) else full_channel_separation
        self.use_cuda_conv = use_cuda_conv
        self.conv_layer_use_dynamic_conv = conv_layer_use_dynamic_conv
        self.sdconv_kernel_size = [sdconv_kernel_size] * len(block_sizes) if isinstance(sdconv_kernel_size, int) else sdconv_kernel_size
        self.no_v_head = no_v_head
        self.identity_preserving_norm = identity_preserving_norm
        self.relative_attention = relative_attention if isinstance(relative_attention, (list, tuple)) else [relative_attention] * len(self.block_channel_size)
        self.approximate_attention = approximate_attention if isinstance(approximate_attention, (list, tuple)) else [approximate_attention] * len(self.block_channel_size)
        assert position_biased_input or separate_content_and_position_attention
        assert not (separate_content_and_position_attention and any(approximate_attention))
        assert (any(self.approximate_attention) and position_biased_input) or not any(self.approximate_attention)
        assert len(self.approximate_attention) == len(block_sizes)  # + 1 for decoder
        if any(self.sdconv):
            assert position_biased_input

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)


vanilla_bert_base = FastFormerConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                     ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                     separate_content_and_position_attention=False, approximate_attention=[False] * 1, block_repeats=False)
vanilla_funnel_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                       ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                       separate_content_and_position_attention=False, approximate_attention=[False] * 3, )
repeated_funnel_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                        ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                        separate_content_and_position_attention=False, approximate_attention=[False] * 3,
                                        block_repeats=True, separate_compressiion_layer=True, )
repeated_funnel_channel_expanded_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[480, 768, 960],
                                                         num_decoder_layers=2, n_head=[8, 12, 12], d_head=[48, 64, 80],
                                                         ffn_groups=4, qkv_transform_groups=4, embedding_size=128, num_highway_cls_tokens=0,
                                                         separate_content_and_position_attention=False, approximate_attention=[False] * 3,
                                                         block_repeats=True, separate_compressiion_layer=False, )
vanilla_albert_base = FastFormerConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                       ffn_groups=1, qkv_transform_groups=1, embedding_size=128, num_highway_cls_tokens=0,
                                       separate_content_and_position_attention=False, approximate_attention=[False] * 1,
                                       block_repeats=True)

sm_config = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5, use_cuda_conv=False,
                             stride=4, qkv_transform_groups=8, ffn_groups=8, block_sizes=[4, 4, 4],
                             approximate_attention=[False, False, False], max_position_embeddings=1024, d_head=[24, 32, 64],
                             separate_compressiion_layer=False,
                             sdconv=[False, False, False], full_channel_separation=True,
                             sdconv_kernel_size=[5, 7, 9],
                             compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                             compressed_query_attention_layers=[(0, 3), (0, 4),
                                                                # (1, 2), (1, 3), (1, 4),
                                                                # (2, 2), (2, 3), (2, 4)
                                                                ],
                             compressed_key_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                              # (1, 1), (1, 2), (1, 3), (1, 4),
                                                              # (2, 1), (2, 2), (2, 3), (2, 4)
                                                              ],
                             n_head=[(8, 0, 0), (8, 0, 0), (12, 0, 0)],
                             block_channel_size=[384, 512, 768], no_v_head=True, expand_dim_before_pooling=False, char_rnn=True, char_rnn_window_overlap=64,
                             char_rnn_window_size=128,
                             )

# Fasttest
md_config = FastFormerConfig(separate_content_and_position_attention=True, pooling_type="learn_sdconv", pooling_kernel_size=4, use_cuda_conv=True,
                             stride=4, qkv_transform_groups=1, ffn_groups=1,
                             approximate_attention=[False, False, False], max_position_embeddings=1024, d_head=[48, 64, 64],
                             separate_compressiion_layer=True,
                             sdconv=[True, True, False], full_channel_separation=True,
                             sdconv_kernel_size=[5, 5, 3],
                             compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                             compressed_query_attention_layers=[(0, 3), (0, 4),
                                                                # (1, 2), (1, 3), (1, 4),
                                                                # (2, 2), (2, 3), (2, 4)
                                                                ],
                             compressed_key_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                              # (1, 1), (1, 2), (1, 3), (1, 4),
                                                              # (2, 1), (2, 2), (2, 3), (2, 4)
                                                              ],
                             n_head=[(4, 4, 0), (4, 4, 0), (12, 0, 0)],
                             block_channel_size=[384, 512, 768], no_v_head=False, expand_dim_before_pooling=False, char_rnn=True, char_rnn_window_overlap=64,
                             char_rnn_window_size=128,
                             )

md_config_relative = FastFormerConfig(separate_content_and_position_attention=True, pooling_type="learn_sdconv", pooling_kernel_size=4, use_cuda_conv=True,
                                      stride=4, qkv_transform_groups=1, ffn_groups=1,
                                      approximate_attention=[False, False, False], max_position_embeddings=1024, d_head=[48, 64, 64],
                                      separate_compressiion_layer=True,
                                      sdconv=[True, True, False], full_channel_separation=True,
                                      sdconv_kernel_size=[5, 5, 3],
                                      compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                                      compressed_query_attention_layers=[(0, 3), (0, 4),
                                                                         # (1, 2), (1, 3), (1, 4),
                                                                         # (2, 2), (2, 3), (2, 4)
                                                                         ],
                                      compressed_key_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                       # (1, 1), (1, 2), (1, 3), (1, 4),
                                                                       # (2, 1), (2, 2), (2, 3), (2, 4)
                                                                       ],
                                      n_head=[(4, 4, 0), (4, 4, 0), (12, 0, 0)],
                                      block_channel_size=[384, 512, 768], no_v_head=False, expand_dim_before_pooling=False, char_rnn=True,
                                      char_rnn_window_overlap=64,
                                      char_rnn_window_size=128,
                                      relative_attention=[True, True, True],
                                      )


tg_config = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=3, use_cuda_conv=True, embedding_size=256,
                             stride=2, qkv_transform_groups=1, ffn_groups=1, block_repeats=False,
                             approximate_attention=[False, False, False], max_position_embeddings=1024, d_head=[64, 64, 64],
                             separate_compressiion_layer=True,
                             sdconv=[False, False, False], full_channel_separation=True,
                             sdconv_kernel_size=[5, 5, 3],
                             compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                             compressed_query_attention_layers=[(0, 3), (0, 4),
                                                                # (1, 2), (1, 3), (1, 4),
                                                                # (2, 2), (2, 3), (2, 4)
                                                                ],
                             compressed_key_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                              # (1, 1), (1, 2), (1, 3), (1, 4),
                                                              # (2, 1), (2, 2), (2, 3), (2, 4)
                                                              ],
                             n_head=[(8, 0, 0), (12, 0, 0), (16, 0, 0)],
                             block_channel_size=[512, 768, 1024], no_v_head=False, expand_dim_before_pooling=True, char_rnn=True, char_rnn_window_overlap=64,
                             char_rnn_window_size=128,
                             )

vision_md_config = FastFormerConfig(stride=1, d_head=[48, 64], n_head=[(8, 0, 0), (12, 0, 0)], block_channel_size=[384, 768], num_decoder_layers=2, block_sizes=[6, 6], num_highway_cls_tokens=8)

vision_lg_config = FastFormerConfig(stride=1, d_head=[64, 64], n_head=[(8, 0, 0), (16, 0, 0)], block_channel_size=[512, 1024], num_decoder_layers=2, block_sizes=[6, 6], num_highway_cls_tokens=8)

vision_md_funnel_config = FastFormerConfig(stride=2, d_head=[48, 64], n_head=[(8, 0, 0), (12, 0, 0)], block_channel_size=[384, 768], num_decoder_layers=2, block_sizes=[6, 6], num_highway_cls_tokens=8)

vision_lg_funnel_config = FastFormerConfig(stride=2, d_head=[64, 64], n_head=[(8, 0, 0), (16, 0, 0)], block_channel_size=[512, 1024], num_decoder_layers=2, block_sizes=[6, 6], num_highway_cls_tokens=8)

config_dict = dict(tg_config=tg_config, md_config=md_config, sm_config=sm_config, md_config_relative=md_config_relative)

vision_config_dict = dict(vision_md_config=vision_md_config, vision_lg_config=vision_lg_config, vision_md_funnel_config=vision_md_funnel_config, vision_lg_funnel_config=vision_lg_funnel_config)


# 20 % -> expand_dim_before_pooling=True, char_rnn=True
# 40 % -> expand_dim_before_pooling=True, char_rnn=True, pooling_type="learn_sdconv"
# 30% -> expand_dim_before_pooling=True, char_rnn=True,  pooling_type="learn_rnn"

# Try -> SDCONV may have non 8 divisible items in the compression module. same with RNN compression.
# Try -> Early Compression with 3 blocks->stride 4->3 blocks->stride 2->6 blocks->stride 2->6 blocks->end
# Try -> RNN and SDCONV
