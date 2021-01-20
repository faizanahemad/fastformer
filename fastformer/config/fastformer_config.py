from typing import List, Dict

from transformers import PretrainedConfig


# TODO: check if all heads are aligned
# TODO: check if repeats are happening properly for layers
# TODO: check if upsampling (interleaving) is happening properly
# TODO: Fix parameter initialization
# TODO: separate static model structure config from dynamic training config

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class AdversarialConfig:
    aitm: bool
    alum: bool
    adv_lm_w: float = 1.0
    adv_ascent_steps: int = 1
    aitm_clip_min: float = 0.1
    aitm_clip_max: float = 0.9
    adv_step_size: float = 1e-3
    adv_epsilon: float = 1e-2
    aitm_noise_var: float = 0.1
    adv_w: float = 1.0


@dataclass_json
@dataclass
class DatasetConfig:
    num_batches_to_train: int
    num_batches_to_validate: int
    datasets: Dict[str, float]  # Dataset name and probability of choice



@dataclass_json
@dataclass
class OptimizerConfig:
    lr: float
    epsilon: float
    beta_1: float
    beta_2: float
    batch_size: int
    test_batch_size: int
    optimizer_class: str
    scheduler_class: str
    do_eval: bool
    validate_every_steps: int
    gradient_clipping: float
    fp16: bool
    early_stopping: bool
    early_stopping_steps: int
    reduce_lr_on_plateau: bool
    reduce_lr_on_plateau_steps: int


@dataclass_json
@dataclass
class ModelConfig:
    tokenizer_name: str
    tokenizer_class: str
    model_class: str
    model_name: str



@dataclass_json
@dataclass
class TrainingConfig:
    experiment_name: str
    experiment_desc: str
    seed: int
    optimizer: OptimizerConfig
    dataset: DatasetConfig
    model: ModelConfig
    adversary: AdversarialConfig


@dataclass_json
@dataclass
class FastFormerConfig(PretrainedConfig):

    def __init__(
            self,
            vocab_size=30522 + 18,
            block_sizes=[6, 6, 6],
            block_channel_size=[576, 768, 960],  # [512, 768, 1024]
            block_repeats=True,
            separate_compressiion_layer=False,
            num_decoder_layers=2,
            n_head=[(8,), (12,), (12,)],  # 8
            use_cuda_conv=True,
            d_head=[72, 64, 80],  # 32
            hidden_act="gelu",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            max_position_embeddings=512,
            type_vocab_size=0,
            initializer_range=0.1,
            initializer_std=None,
            layer_norm_eps=1e-9,
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
            untie_cls=False,
            separate_content_and_position_attention=False,
            approximate_attention=[False, False, False],
            sequence_dependent_position_transform=False,
            qkv_squeeze_fraction=1,
            light_first_layer=False,
            light_last_layer=False,
            compress_query_method=None,
            compressed_query_attention_kernel_size=3,
            compressed_query_attention_stride=2,
            compressed_query_attention_layers=[],
            compressed_key_attention_layers=[],
            sdconv=[False, False, False],
            sdconv_kernel_size=[5, 7, 9],
            full_channel_separation=[False, False, False],
            short_rnn=[False, False, False],
            short_rnn_kernel=[128, 128, 128],
            short_rnn_overlap=[16, 16, 16],
            conv_layer_use_dynamic_conv=False,
            no_v_head=False,
            expand_dim_before_pooling=False,
            identity_preserving_norm=True,
            char_rnn=False,
            char_rnn_layers=1,
            char_rnn_vocab_size=1024,
            char_rnn_window_size=512,
            char_rnn_window_overlap=64,
            **kwargs
    ):
        super().__init__(**kwargs)
        try:
            from fairseq.modules.dynamicconv_layer.dynamicconv_layer import dynamicconvFunction
        except:
            use_cuda_conv = False
        self.vocab_size = vocab_size
        self.tokenizer_length = max_position_embeddings - num_highway_cls_tokens
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
        self.activation_dropout = activation_dropout
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
            "learn",
            "learn_sdconv",
            'learn_rnn',
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
        assert qkv_squeeze_fraction == 1 or qkv_squeeze_fraction > 2
        self.qkv_squeeze_fraction = qkv_squeeze_fraction
        self.approximate_attention = approximate_attention
        self.light_first_layer = light_first_layer
        self.light_last_layer = light_last_layer
        assert compressed_query_attention_kernel_size in [3, 5, 7, 9]
        self.compressed_query_attention_kernel_size = compressed_query_attention_kernel_size
        assert compressed_query_attention_stride in [1, 2, 4]
        self.compressed_query_attention_stride = compressed_query_attention_stride
        self.compressed_query_attention_layers = compressed_query_attention_layers
        self.compressed_key_attention_layers = compressed_key_attention_layers
        self.untie_cls = untie_cls
        self.separate_content_and_position_attention = separate_content_and_position_attention
        self.sequence_dependent_position_transform = sequence_dependent_position_transform
        assert (sequence_dependent_position_transform and separate_content_and_position_attention) or (not sequence_dependent_position_transform)
        assert separate_content_and_position_attention or position_biased_input
        self.stride = stride
        assert len(block_channel_size) == len(block_sizes)
        self.block_channel_size = block_channel_size
        self.short_rnn = [short_rnn] * len(block_sizes) if isinstance(short_rnn, bool) else short_rnn
        self.short_rnn_kernel = [short_rnn_kernel] * len(block_sizes) if isinstance(short_rnn_kernel, int) else short_rnn_kernel
        self.short_rnn_overlap = [short_rnn_overlap] * len(block_sizes) if isinstance(short_rnn_overlap, int) else short_rnn_overlap

        self.sdconv = [sdconv] * len(block_sizes) if isinstance(sdconv, bool) else sdconv
        self.full_channel_separation = [full_channel_separation] * len(block_sizes) if isinstance(full_channel_separation, bool) else full_channel_separation
        self.use_cuda_conv = use_cuda_conv
        self.conv_layer_use_dynamic_conv = conv_layer_use_dynamic_conv
        self.sdconv_kernel_size = [sdconv_kernel_size] * len(block_sizes) if isinstance(sdconv_kernel_size, int) else sdconv_kernel_size
        self.no_v_head = no_v_head
        self.identity_preserving_norm = identity_preserving_norm
        assert position_biased_input or separate_content_and_position_attention
        assert not (separate_content_and_position_attention and any(approximate_attention))
        assert (sequence_dependent_position_transform and separate_content_and_position_attention) or not sequence_dependent_position_transform
        assert (any(approximate_attention) and position_biased_input) or not any(approximate_attention)
        assert len(approximate_attention) == len(block_sizes)  # + 1 for decoder
        if light_first_layer or any(self.short_rnn) or any(self.sdconv) or light_last_layer:
            assert position_biased_input

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)


vanilla_bert_base = FastFormerConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                     ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                     untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False] * 1, block_repeats=False)
vanilla_funnel_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                       ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                       untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False] * 3, )
repeated_funnel_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[768, 768, 768], num_decoder_layers=2, n_head=12, d_head=64,
                                        ffn_groups=1, qkv_transform_groups=1, embedding_size=768, num_highway_cls_tokens=0,
                                        untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False] * 3,
                                        block_repeats=True, separate_compressiion_layer=True, )
repeated_funnel_channel_expanded_base = FastFormerConfig(vocab_size=30522, block_sizes=[6, 6, 6], block_channel_size=[480, 768, 960],
                                                         num_decoder_layers=2, n_head=[8, 12, 12], d_head=[48, 64, 80],
                                                         ffn_groups=4, qkv_transform_groups=4, embedding_size=128, num_highway_cls_tokens=0,
                                                         untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False] * 3,
                                                         block_repeats=True, separate_compressiion_layer=False, )
vanilla_albert_base = FastFormerConfig(vocab_size=30522, block_sizes=[12], block_channel_size=[768], num_decoder_layers=0, n_head=12, d_head=64,
                                       ffn_groups=1, qkv_transform_groups=1, embedding_size=128, num_highway_cls_tokens=0,
                                       untie_cls=False, separate_content_and_position_attention=False, approximate_attention=[False] * 1,
                                       block_repeats=True)

sm_config = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5,
                             sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=4, ffn_groups=4,
                             approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=[24, 32, 64], separate_compressiion_layer=True,
                             qkv_squeeze_fraction=4, light_last_layer=True, light_first_layer=True,
                             sdconv=True, full_channel_separation=True, short_rnn=True,
                             sdconv_kernel_size=[5, 7, 9], block_sizes=[3, 3, 3],
                             compress_query_method="mean", compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                             compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                (1, 1), (1, 2), (1, 3), (1, 4),
                                                                (2, 1), (2, 2), (2, 3), (2, 4)
                                                                ],
                             compressed_key_attention_layers=[(0, 3), (0, 4),
                                                              (1, 3), (1, 4),
                                                              (2, 3), (2, 4)
                                                              ],
                             n_head=[(4, 2, 2), (4, 2, 2), (4, 4, 4)],
                             block_channel_size=[384, 512, 768], no_v_head=True,
                             )
md_config = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5,
                             sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=4, ffn_groups=4,
                             approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=[24, 64, 80], separate_compressiion_layer=True,
                             qkv_squeeze_fraction=1, light_last_layer=True, light_first_layer=True,
                             sdconv=True, full_channel_separation=True, short_rnn=True,
                             sdconv_kernel_size=[5, 7, 9],
                             compress_query_method="learn_sdconv", compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                             compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                (1, 1), (1, 2), (1, 3), (1, 4),
                                                                (2, 1), (2, 2), (2, 3), (2, 4)
                                                                ],
                             compressed_key_attention_layers=[(0, 3), (0, 4),
                                                              (1, 3), (1, 4),
                                                              (2, 3), (2, 4)
                                                              ],
                             # n_head=[(1, 0, 7), (1, 0, 11), (1, 0, 11)],
                             # n_head=[(1, 7, 0), (1, 11, 0), (1, 11, 0)],
                             # n_head=[(8,), (12,), (12,)],
                             n_head=[(4, 2, 2), (4, 4, 4), (4, 4, 4)],
                             block_channel_size=[384, 768, 960], no_v_head=False, expand_dim_before_pooling=False, char_rnn=True,
                             )

md_config_no_rnn = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5,
                                    sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=1, ffn_groups=1,
                                    approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=[24, 64, 80],
                                    separate_compressiion_layer=True,
                                    qkv_squeeze_fraction=1, light_last_layer=False, light_first_layer=False,
                                    sdconv=True, full_channel_separation=True, short_rnn=False,
                                    sdconv_kernel_size=[5, 7, 9],
                                    compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                                    compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                       (1, 1), (1, 2), (1, 3), (1, 4),
                                                                       (2, 1), (2, 2), (2, 3), (2, 4)
                                                                       ],
                                    compressed_key_attention_layers=[(0, 3), (0, 4),
                                                                     (1, 3), (1, 4),
                                                                     (2, 3), (2, 4)
                                                                     ],
                                    # n_head=[(1, 0, 7), (1, 0, 11), (1, 0, 11)],
                                    # n_head=[(1, 7, 0), (1, 11, 0), (1, 11, 0)],
                                    # n_head=[(8,), (12,), (12,)],
                                    n_head=[(4, 4, 0), (6, 6, 0), (6, 6, 0)],
                                    block_channel_size=[384, 768, 960], no_v_head=False, expand_dim_before_pooling=False, char_rnn=False,
                                    )


md_config_funnel = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5,
                                    sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=4, ffn_groups=4,
                                    approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=32,
                                    separate_compressiion_layer=True,
                                    qkv_squeeze_fraction=1, light_last_layer=False, light_first_layer=False,
                                    sdconv=False, full_channel_separation=True, short_rnn=False,
                                    sdconv_kernel_size=[5, 7, 9],
                                    compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                                    compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                       (1, 1), (1, 2), (1, 3), (1, 4),
                                                                       (2, 1), (2, 2), (2, 3), (2, 4)
                                                                       ],
                                    compressed_key_attention_layers=[(0, 3), (0, 4),
                                                                     (1, 3), (1, 4),
                                                                     (2, 3), (2, 4)
                                                                     ],
                                    # n_head=[(1, 0, 7), (1, 0, 11), (1, 0, 11)],
                                    # n_head=[(1, 7, 0), (1, 11, 0), (1, 11, 0)],
                                    # n_head=[(8,), (12,), (12,)],
                                    n_head=[(12, 0, 0), (12, 0, 0), (12, 0, 0)],
                                    block_channel_size=[768, 768, 768], no_v_head=False, expand_dim_before_pooling=False, char_rnn=False,
                                    )

md_config_no_sdconv = FastFormerConfig(separate_content_and_position_attention=False, pooling_type="mean", pooling_kernel_size=5,
                                       sequence_dependent_position_transform=False, stride=2, qkv_transform_groups=1, ffn_groups=1,
                                       approximate_attention=[False, False, False], max_position_embeddings=2048, d_head=64,
                                       separate_compressiion_layer=True,
                                       qkv_squeeze_fraction=1, light_last_layer=False, light_first_layer=False,
                                       sdconv=False, full_channel_separation=True, short_rnn=True,
                                       sdconv_kernel_size=[5, 7, 9],
                                       compress_query_method=None, compressed_query_attention_stride=2, compressed_query_attention_kernel_size=3,
                                       compressed_query_attention_layers=[(0, 1), (0, 2), (0, 3), (0, 4),
                                                                          (1, 1), (1, 2), (1, 3), (1, 4),
                                                                          (2, 1), (2, 2), (2, 3), (2, 4)
                                                                          ],
                                       compressed_key_attention_layers=[(0, 3), (0, 4),
                                                                        (1, 3), (1, 4),
                                                                        (2, 3), (2, 4)
                                                                        ],
                                       # n_head=[(1, 0, 7), (1, 0, 11), (1, 0, 11)],
                                       # n_head=[(1, 7, 0), (1, 11, 0), (1, 11, 0)],
                                       # n_head=[(8,), (12,), (12,)],
                                       n_head=[(6, 0, 6), (6, 0, 6), (6, 0, 6)],
                                       block_channel_size=[768, 768, 768], no_v_head=False, expand_dim_before_pooling=False, char_rnn=True,
                                       short_rnn_kernel=[16, 16, 16], short_rnn_overlap=[4, 4, 4], char_rnn_window_overlap=8, char_rnn_window_size=32,
                                       )
