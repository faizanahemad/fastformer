from transformers import BertTokenizerFast, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


t1 = """
With the success of language pretraining, it is highly desirable to develop more
efficient architectures of good scalability that can exploit the abundant unlabeled
data at a lower cost. To improve the efficiency, we examine the much-overlooked
redundancy in maintaining a full-length token-level presentation, especially for
tasks that only require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More
importantly, by re-investing the saved FLOPs from length reduction in constructing
a deeper or wider model, we further improve the model capacity. In addition, to
perform token-level predictions as required by common pretraining objectives,
Funnel-Transformer is able to recover a deep representation for each token from
the reduced hidden sequence via a decoder. Empirically, with comparable or fewer
FLOPs, Funnel-Transformer outperforms the standard Transformer on a wide
variety of sequence-level prediction tasks, including text classification, language
understanding, and reading comprehension. Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and
longer training times. To address these problems, we present two parameterreduction techniques to lower memory consumption and increase the training
speed of BERT (Devlin et al., 2019). Comprehensive empirical evidence shows
that our proposed methods lead to models that scale much better compared to
the original BERT. We also use a self-supervised loss that focuses on modeling
inter-sentence coherence, and show it consistently helps downstream tasks with
multi-sentence inputs. As a result, our best model establishes new state-of-the-art
results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.
Self-attention is a useful mechanism to build generative models for language and images. It determines the importance of context elements by comparing each element to the current time step. In this paper, we show that a very lightweight convolution can perform competitively to the best reported self-attention results. Next, we introduce dynamic convolutions which are simpler and more efficient than self-attention. We predict separate convolution kernels based solely on the current time-step in order to determine the importance of context elements. The number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic. Experiments on large-scale machine translation, language modeling and abstractive summarization show that dynamic convolutions improve over strong self-attention models. On the WMT'14 English-German test set dynamic convolutions achieve a new state of the art of 29.7 BLEU.
"""
t2 = """
Most popular optimizers for deep learning can be broadly categorized as adaptive methods (e.g. Adam) and accelerated schemes (e.g. stochastic gradient descent (SGD) with momentum). For many models such as convolutional neural networks (CNNs), adaptive methods typically converge faster but generalize worse compared to SGD; for complex settings such as generative adversarial networks (GANs), adaptive methods are typically the default because of their stability.We propose AdaBelief to simultaneously achieve three goals: fast convergence as in adaptive methods, good generalization as in SGD, and training stability. The intuition for AdaBelief is to adapt the stepsize according to the "belief" in the current gradient direction. Viewing the exponential moving average (EMA) of the noisy gradient as the prediction of the gradient at the next time step, if the observed gradient greatly deviates from the prediction, we distrust the current observation and take a small step; if the observed gradient is close to the prediction, we trust it and take a large step. We validate AdaBelief in extensive experiments, showing that it outperforms other methods with fast convergence and high accuracy on image classification and language modeling. Specifically, on ImageNet, AdaBelief achieves comparable accuracy to SGD. Furthermore, in the training of a GAN on Cifar10, AdaBelief demonstrates high stability and improves the quality of generated samples compared to a well-tuned Adam optimizer.
"""
t3 = """
We present the Open Graph Benchmark (OGB), a diverse set of challenging and realistic benchmark datasets to facilitate scalable, robust, and reproducible graph machine learning (ML) research. OGB datasets are large-scale, encompass multiple important graph ML tasks, and cover a diverse range of domains, ranging from social and information networks to biological networks, molecular graphs, source code ASTs, and knowledge graphs. For each dataset, we provide a unified evaluation protocol using meaningful application-specific data splits and evaluation metrics. In addition to building the datasets, we also perform extensive benchmark experiments for each dataset. Our experiments suggest that OGB datasets present significant challenges of scalability to large-scale graphs and out-of-distribution generalization under realistic data splits, indicating fruitful opportunities for future research. Finally, OGB provides an automated end-to-end graph ML pipeline that simplifies and standardizes the process of graph data loading, experimental setup, and model evaluation. OGB will be regularly updated and welcomes inputs from the community. OGB datasets as well as data loaders, evaluation scripts, baseline code, and leaderboards are publicly available.
"""
large_texts = [
    t1,
    t2,
    t3,
    t2 + t3
]

very_small_texts = ["The quick brown fox jumps over the lazy dog."] * 4
small_texts = [" ".join(t.split()[:128]) for t in large_texts]
very_large_texts = [
    t1 + t2 + t3,
    t2 + t3 + t1,
    t3 + t1 + t2 + t1 + t2 + t3,
    t1 + t2 + t3 + t1,
    t1,
    t2,
    t3,
    t1 + t3
]


class SmallTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, item):
        return dict(text=self.texts[item])

    def __len__(self):
        return len(self.texts)


if __name__ == "__main__":
    from fastformer.data.dataset import TokenizerDataset, collate_fn
    from fastformer.config import md_config
    char_to_id = sorted([k for k, v in AutoTokenizer.from_pretrained("bert-base-uncased").get_vocab().items() if len(k) == 1]) + [" ", "\n"]
    char_to_id = dict(zip(char_to_id, range(2, len(char_to_id) + 2)))

    for i in range(10):
        dataset = SmallTextDataset(small_texts)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        setattr(tokenizer, "_sentence_mask_token", "[MASK1]")
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES = tokenizer.SPECIAL_TOKENS_ATTRIBUTES + ["sentence_mask_token"]
        tokenizer.add_special_tokens({"sentence_mask_token": "[MASK1]"})

        dataset = TokenizerDataset(md_config, tokenizer, char_to_id, dict(padding="max_length", truncation=True, return_tensors="pt", max_length=512), dataset)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, prefetch_factor=2, num_workers=2)
        next_batch = next(iter(dataloader))
        print(next_batch)

