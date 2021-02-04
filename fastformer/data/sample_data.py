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
With the recent success of unsupervised language pretraining [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
the power of neural self-attention models (a.k.a. Transformer) [13] has been pushed to a new level,
leading to dramatic advancements in machine learning and natural language processing (NLP). More
importantly, it has been observed that with more FLOPs invested in longer pretraining and/or larger
models, the performance of pretrained Transformer models consistently improve. However, it is
extremely expensive to pretrain or even just to finetune the state-of-the-art self-attention models, as
they require much more FLOPs and memory resources compared to traditional models in NLP. This
largely limits their applications and success in more fields.
Given this challenge, there has been an increasing amount of efforts to reduce the costs of pretraining
and finetuning self-attention models. From the perspective of post-pretraining processing, typical
approaches include distillation, pruning and quantization of various kinds, which try to derive a
lighter model from an well-pretrained model by taking advantage of the richer signals in the larger
model or learning to remove less important operations. Another line of research aims at designing
an architecture that not only has a lower resource-to-performance ratio (more efficient) but also
scales as well as the Transformer, at least in certain domains. Most of such methods build upon
the Transformer backbone and focus on redesigning its building blocks. Representative solutions
include searching for better micro operation or macro module designs [14, 15], replacing the full
pairwise attention with local operations such as convolution [16] and dynamic convolution [17], and
optimizing the hidden size combinations for existing blocks [18].
Across the wide variety of ideas mentioned above, a common strategy is to identify redundant
operations or representations and replace them with more efficient ones. Inspired by this line of
thinking, in this work, we will be focusing on the potential redundancy induced by always maintaining
a full-length sequence of hidden representations across all layers in Transformer. Intuitively, for many
sequence-level NLP tasks such as text classification and ranking, the most common use case is to
extract a single vector from the entire sequence, which does not necessarily preserve all information
down to the token-level granularity. Hence, for such tasks, the full-length sequence of hidden states
may contain significant redundancy. This is analogous to the case of image recognition, where
the convolution neural network gradually reduces the spatial resolution/size of feature maps as the
neural network goes deeper. In addition, linguistic prior also encourages gradually merging nearby
tokens (words) into larger semantic units (phrases), which naturally leads to a shorter sequence of
representations.
Concretely, we propose to gradually reduce the sequential resolution (i.e. length) of the hidden
representation in self-attention models. Immediately, the reduction in sequence length can lead
to significant savings in both FLOPs and memory. More importantly, the saved computational
resource can be directly re-invested in constructing a deeper (or wider) model to boost the model
capacity without additional computational burden. In addition, to address the challenge that common
pretraining objectives such as masked language modeling (MLM) [2] require separate representations
for each token, we design a simple strategy to decode a full-length sequence of deep representations
from the hidden state of reduced length. As a result, the proposed model can be directly trained
without modifying the pretraining objectives, as well as adopted for downstream tasks that require
token-level representations.
Empirically, with comparable or even fewer FLOPs, by trading sequential resolution for depth, our
proposed model achieves an improved performance over the standard Transformer on a wide variety
of sequence-level prediction tasks, including text classification, language understanding, and reading
comprehension.
In this work, under the pretraining-finetuning paradigm, we investigate a largely overlooked dimension
of complexity in language processing. With the proposed Funnel-Transformer, we show how
sequential resolution can be compressed in a simple form to save computation and how the saved
FLOPs can be re-invested in improving the model capacity and hence the performance. Open
challenges for future research include the better ways to improve the compression scheme, to
optimize the block layout design and to re-invest the saved FLOPs. In addition, combining FunnelTransformer with model compression techniques like knowledge distillation and quantization would
be an important direction towards the enhancement of practical impact.
As far as we know, no previous work achieves performance gain via compressing the sequence length
of the hidden states under language pretraining. Meanwhile, our proposed model is quite similar to
the bottom-up model proposed by a contemporary work [19] for causal language modeling. The key
differences include the pool-query-only design for down-sampling, how the up-sampling is performed,
and our relative attention parameterization. Another closely related idea is Power-BERT [20], which
learns to soft-eliminate word vectors that are less “significant” during finetuning. Hence, for postfinetuning inference, the sequence length can be reduced to achieve acceleration. More generally, our
work is also related to previous work on hierarchical recurrent neural networks [21] and Transformer
models [22, 23]. Different from these methods, our model does not rely on any pre-defined hierarchy
or boundary of semantic meanings and always captures the full-length dependency input with
attention.
In contrast, our work draws many inspirations from the computer vision domain. The contracting
encoder and expanding decoder framework with residual connections is conceptually similar to the
ResUNet [24] for image segmentation. The strided pooling is also widely used to construct modern
image recognition networks [25]. Despite the similarities, apart from the obvious difference in data
domain and computation modules, our encoder employs a special pool-query-only design to improve
the compression, and our decoder only requires a single up-sampling with a large expansion rate.
In addition, a line of research in graph neural networks has tries to gradually reduce the number of
nodes in different ways and obtain a single vectorial representation for supervised classification. [26,
27, 28] While these methods could potentially be plugged into our model as alternative compression
operations, it remains an open question whether compression techniques developed for supervised
graph classification can be extended the large-scale language pretraining.
"""
t2 = """
Most popular optimizers for deep learning can be broadly categorized as adaptive methods (e.g. Adam) and accelerated schemes (e.g. stochastic gradient descent (SGD) with momentum). For many models such as convolutional neural networks (CNNs), adaptive methods typically converge faster but generalize worse compared to SGD; for complex settings such as generative adversarial networks (GANs), adaptive methods are typically the default because of their stability.We propose AdaBelief to simultaneously achieve three goals: fast convergence as in adaptive methods, good generalization as in SGD, and training stability. The intuition for AdaBelief is to adapt the stepsize according to the "belief" in the current gradient direction. Viewing the exponential moving average (EMA) of the noisy gradient as the prediction of the gradient at the next time step, if the observed gradient greatly deviates from the prediction, we distrust the current observation and take a small step; if the observed gradient is close to the prediction, we trust it and take a large step. We validate AdaBelief in extensive experiments, showing that it outperforms other methods with fast convergence and high accuracy on image classification and language modeling. Specifically, on ImageNet, AdaBelief achieves comparable accuracy to SGD. Furthermore, in the training of a GAN on Cifar10, AdaBelief demonstrates high stability and improves the quality of generated samples compared to a well-tuned Adam optimizer.
"""
t3 = """
We present the Open Graph Benchmark (OGB), a diverse set of challenging and realistic benchmark datasets to facilitate scalable, robust, and reproducible graph machine learning (ML) research. OGB datasets are large-scale, encompass multiple important graph ML tasks, and cover a diverse range of domains, ranging from social and information networks to biological networks, molecular graphs, source code ASTs, and knowledge graphs. For each dataset, we provide a unified evaluation protocol using meaningful application-specific data splits and evaluation metrics. In addition to building the datasets, we also perform extensive benchmark experiments for each dataset. Our experiments suggest that OGB datasets present significant challenges of scalability to large-scale graphs and out-of-distribution generalization under realistic data splits, indicating fruitful opportunities for future research. Finally, OGB provides an automated end-to-end graph ML pipeline that simplifies and standardizes the process of graph data loading, experimental setup, and model evaluation. OGB will be regularly updated and welcomes inputs from the community. OGB datasets as well as data loaders, evaluation scripts, baseline code, and leaderboards are publicly available.
"""
t4 =  'Large pre-trained multilingual models like mBERT, XLM-R achieve state of the art results on language understanding tasks. However, they are not well suited for latency critical applications on both servers and edge devices. It’s important to reduce the memory and compute resources required by these models. To this end, we propose pQRNN, a projectionbased embedding-free neural encoder that is tiny and effective for natural language processing tasks. Without pre-training, pQRNNs significantly outperform LSTM models with pre-trained embeddings despite being 140x smaller. With the same number of parameters, they outperform transformer baselines thereby showcasing their parameter efficiency. Additionally, we show that pQRNNs are effective student architectures for distilling large pretrained language models. We perform careful ablations which study the effect of pQRNN parameters, data augmentation, and distillation settings. On MTOP, a challenging multilingual semantic parsing dataset, pQRNN students achieve 95.9% of the performance of an mBERT teacher while being 350x smaller. On mATIS, a popular parsing task, pQRNN students on average are able to get to 97.1% of the teacher while again being 350x smaller. Our strong results suggest that our approach is great for latency-sensitive applications while being able to leverage large mBERT-like models. Large pre-trained language models (Devlin et al., 2018; Lan et al., 2020; Liu et al., 2019; Yang et al., 2019; Raffel et al., 2019) have demonstrated state-of-the-art results in many natural language processing (NLP) tasks (e.g. the GLUE benchmark (Wang et al., 2018)). Multilingual variants of these models covering 100+ languages (Conneau et al., 2020; Arivazhagan et al., 2019; Fang et al., 2020; Siddhant et al.; Xue et al., 2020; Chung et al., 2020) have shown tremendous crosslingual transfer learning capability on the challenging XTREME benchmark (Hu et al., 2020). However, these models require millions of parameters and several GigaFLOPS making them take up significant compute resources for applications on servers and impractical for those on the edge such as mobile platforms. Reducing the memory and compute requirements of these models while maintaining accuracy has been an active field of research. The most commonly used techniques are quantization (Gong et al., 2014; Han et al., 2015a), weight pruning (Han et al., 2015b), and knowledge distillation (Hinton et al., 2015). In this work, we will focus on knowledge distillation (KD), which aims to transfer knowledge from a teacher model to a student model, as an approach to model compression. KD has been widely studied in the context of pre-trained language models (Tang et al., 2019; Turc et al., 2019; Sun et al., 2020). These methods can be broadly classified into two categories: taskagnostic and task-specific distillation (Sun et al., 2020). Task-agnostic methods aim to perform distillation on the pre-training objective like masked language modeling (MLM) in order to obtain a smaller pre-trained model. However, many tasks of practical interest to the NLP community are not as complex as the MLM task solved by task-agnostic approaches. This results in complexity inversion — i.e., in order to solve a specific relatively easy problem the models learn to solve a general much harder problem which entails language understanding. Task-specific methods on the other hand distill the knowledge needed to solve a specific task onto a student model thereby making the student very efficient at the task that it aims to solve. This allows for decoupled evolution of both the teacher and student models. In task-specific distillation, the most important requirement is that the student model architectures are efficient in terms of the number of training samples, number of parameters, and number of FLOPS. To address this need we propose pQRNN (projection Quasi-RNN), an embedding-free neural encoder for NLP tasks. Unlike embedding-based model architectures used in NLP (Wu et al., 2016; Vaswani et al., 2017), pQRNN learns the tokens relevant for solving a task directly from the text input similar to Kaliamoorthi et al. (2019). Specifically, they overcome a significant bottleneck of multilingual pre-trained language models where embeddings take up anywhere between 47% and 71% of the total parameters due to large vocabularies (Chung et al., 2020). This results in many advantages such as not requiring pre-processing before training a model and having orders of magnitude fewer parameters. We perform distillation from a pre-trained mBERT teacher fine-tuned for the semantic parsing task. We propose a projection-based architecture for the student. We hypothesise that since the student is task-specific, using projection would allow the model to learn the relevant tokens needed to replicate the decision surface learnt by the teacher. This allows us to significantly reduce the number of parameters that are context invariant, such as those in the embeddings, and increase the number of parameters that are useful to learn a contextual representation. We further use a multilingual teacher that helps improve the performance of low-resource languages through cross-lingual transfer learning. We propose input paraphrasing as a strategy for data augmentation which further improves the final quality. We present pQRNN: a tiny, efficient, embeddingfree neural encoder for NLP tasks. We show that pQRNNs outperform LSTM models with pretrained embeddings despite being 140x smaller. They are also parameter efficient which is proven by their gain over a comparable transformer baseline. We then show that pQRNNs are ideal student architectures for distilling large pre-trained language models (i.e., mBERT). On MTOP, a multilingual task-oriented semantic parsing dataset, pQRNN students reach 95.9% of the mBERT teacher performance. Similarly, on mATIS, a popular semantic parsing task, our pQRNN students achieve 97.1% of the teacher performance. In both cases, the student pQRNN is a staggering 350x smaller than the teacher. Finally, we carefully ablate the effect of pQRNN parameters, the amount of pivot-based paraphrasing data, and the effect of teacher logit scaling. Our results prove that it’s possible to leverage large pre-trained language models into dramatically smaller pQRNN students without any significant loss in quality. Our approach has been shown to work reliably at Google-scale for latency-critical applications. '

t1 = t1.replace('\n',' ')
t2 = t2.replace('\n',' ')
t3 = t3.replace('\n',' ')
t4 = t4.replace('\n',' ')

large_texts = [
    t1,
    t2,
    t3,
    t4
]

very_small_texts = ["The quick brown fox jumps over the lazy dog."] * 4
small_texts = [" ".join(t1.split()[:256]) for _ in range(4)]

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

