# Text embedding models for recommendation

Date: 2026-09-29

## Confirmed project direction

Compare general-purpose and instruction-based text embedding models as item representations, then measure how reducing the embedding size changes recommendation quality. Keep the recommendation model and evaluation protocol fixed so that the comparison isolates the embedding choice and size.

## Venue eligibility note

CARec is published in the CIKM 2024 proceedings. CIKM is ranked A, not A*, in the current ICORE/CORE listing. This is acceptable under the assignment as written because the assignment explicitly lists CIKM among the allowed venues. The exact research-track label for this paper remains an item to confirm with the instructor if required.

Sources: [ICORE/CORE CIKM ranking](https://portal.core.edu.au/conf-ranks/25/) · [CARec publication page](https://liangwei.yangxiaoma.org/publication/cikm2024_carec/).

## Recommended base paper

### CARec: Collaborative Alignment for Recommendation

[CIKM 2024 paper](https://doi.org/10.1145/3627673.3679535) · [open paper copy](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf) · [author page](https://liangwei.yangxiaoma.org/publication/cikm2024_carec/) · [code](https://github.com/ChenMetanoia/CARec)

CARec combines two kinds of information:

- collaborative information: who interacted with which items;
- semantic information: a text embedding of each item.

It aligns these spaces so that the item representation keeps useful text meaning while also fitting the recommendation task. It evaluates both warm items and cold items. The paper compares `instructor-xl`, `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-base-en-v1.5`, and BERT-based representations. `instructor-xl` is the instruction-tuned model in this comparison; the other encoders are used as general-purpose alternatives. The paper uses the instruction `Represent the Amazon title:` for `instructor-xl`.

This is a possible base paper because it contains an encoder ablation that includes one instruction-tuned model and several non-instruction encoders, but this is not the paper's main research question. It does not isolate instruction tuning from model architecture, pre-training data, model size, or native embedding dimension. Your project would therefore be an extension of this ablation, not a reproduction of the paper's central claim.

### What CARec does and does not study

CARec studies how to align semantic item representations with collaborative-filtering representations. The encoder comparison is a supporting experiment. It is not a controlled scientific comparison of general-purpose versus instruction-tuned embedding training.

The comparison is also limited: the paper reports it in the Electronic dataset experiment, and only `instructor-xl` is instruction-conditioned. The result can motivate your question, but it cannot by itself show that instruction tuning caused the difference. A fair extension must hold the recommender, item text, target dimension, split, and evaluation protocol fixed.

### Dataset availability for CARec

Yes. The paper uses four public datasets:

- Amazon Electronics
- Amazon Office Products
- Amazon Grocery and Gourmet Food
- Yelp

The [official repository instructions](https://github.com/ChenMetanoia/CARec) tell you where to download the Amazon metadata and review data and provide the preprocessing script. The raw data is downloaded separately rather than bundled in the repository. The paper uses Recall@K and NDCG@K and creates a cold-item split by withholding interactions for 5% of items. This gives a ready-made warm/cold evaluation setup.

The safest first dataset is Amazon Electronics. It has item titles and other metadata, the paper reports results for it, and it is directly supported by the code. Add one more dataset only after the first experiment works.

## Supporting papers

### Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization

[RecSys 2025 paper](https://doi.org/10.1145/3705328.3748038) · [open paper](https://arxiv.org/html/2507.19473) · [code](https://github.com/ArtemF42/let-it-go)

This paper compares three ways to use content embeddings in a sequential recommender: use normal ID embeddings, initialise with content and fine-tune them, or keep content embeddings mostly fixed and learn a small correction. It also shows a practical way to reduce a large text embedding to the recommender dimension: standardisation followed by PCA.

### Dataset availability for Let It Go?

Yes. It uses:

- Amazon-M2, with textual item descriptions;
- Amazon Beauty, with review text;
- Zvuk, with audio representations.

The first two are useful for this project because they are text-based. The paper states that code is available and reports the Amazon-M2 and Beauty preprocessing. Zvuk is not needed for this project. The paper uses E5 text embeddings, so it is especially useful for designing the dimension-reduction part of the experiment.

### MARec: Metadata Alignment for cold-start Recommendation

[RecSys 2024 paper](https://doi.org/10.1145/3640457.3688125) · [open paper](https://assets.amazon.science/5f/3e/d92eda32446891e927ccb0a63b06/marec-metadata-alignment-for-cold-start-recommendation) · [paper page](https://www.amazon.science/publications/marec-metadata-alignment-for-cold-start-recommendation)

MARec is a simpler hybrid recommender. It learns item similarities from metadata and aligns them with similarities from user interactions. Its purpose is to make recommendations for items with no interaction history while remaining competitive for warm items. An ablation compares TF-IDF, Sentence-Transformer embeddings, and Falcon embeddings.

### Dataset availability for MARec

Yes. The paper uses public datasets and describes the splits:

- Amazon Video Games;
- Netflix;
- MovieLens10M;
- MovieLens Hetrec for cold-start experiments;
- MovieLens1M and Pinterest for warm-start experiments.

The paper gives the public dataset links and split details. It is a good source for a simple baseline and for a second cold-start dataset, but it is less directly aligned with the general-purpose versus instruction-tuned comparison than CARec.

## Model-background papers

### INSTRUCTOR: One Embedder, Any Task

[ACL Findings 2023](https://aclanthology.org/2023.findings-acl.71/) · [arXiv](https://arxiv.org/abs/2212.09741)

INSTRUCTOR explains instruction-tuned embeddings. The same text is embedded together with a short task instruction, such as `Represent the Amazon title:`. The instruction tells the encoder what kind of similarity matters. This paper is useful for the theory section, not as the recommendation base paper.

It does not provide the recommendation datasets needed for this project.

### E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training

[E5 paper](https://arxiv.org/abs/2212.03533)

E5 is a general-purpose text embedding model trained with weakly supervised contrastive learning. It is a reasonable general-purpose encoder to include if the Let It Go? pipeline is used. This paper is useful for explaining the encoder, not as the recommendation base paper.

It does not provide a recommendation dataset. Use the public recommendation datasets from CARec or Let It Go? instead.

## Proposed extension

### Research question

Do instruction-tuned item embeddings improve recommendation quality over general-purpose item embeddings, and how much of that benefit remains after reducing the embedding size?

### Minimal experiment

Use one CARec-supported dataset first, preferably Amazon Electronics. Compare:

| Encoder group | Example encoder |
| --- | --- |
| General-purpose | `all-MiniLM-L6-v2` or E5 |
| Instruction-tuned | `instructor-xl` |

Use the same CARec pipeline, same text fields, same train/validation/test split, and same random seeds. Reduce every encoder to the same target dimensions, for example 64, 128, and 256, using PCA fitted only on the training-side item embeddings. If the selected encoders share a 768-dimensional native output, add 768 as the native-dimension condition. Do not expand a smaller encoder to 768 and treat that as an equivalent comparison. Report Recall@10 and NDCG@10 separately for warm and cold items.

The improvement over CARec is a controlled study of the encoder family and dimension. CARec supplies the recommendation framework and a useful initial comparison; your work must make the comparison fair enough to ask whether instruction tuning itself remains useful at smaller dimensions.

### Important fairness rule

Do not compare a 384-dimensional MiniLM vector with a 768-dimensional Instructor vector and attribute every difference to instruction tuning. Either project both down to common dimensions or report native dimensions separately. The main comparison should use equal dimensions. If you want to include 768, choose two encoders that natively produce 768-dimensional vectors.

### Suggested extensions, in order

1. General-purpose versus instruction-tuned embeddings on Amazon Electronics.
2. Dimension ablation at 64, 128, and 256, with 768 only when both encoders naturally support it.
3. Warm versus cold-item results.
4. A second dataset, preferably Yelp or Amazon Office Products.
5. Optional: compare PCA with a small learned linear projection.

Do not train an LLM. Generate the item embeddings once, cache them, and train only the recommender and small projection layers. This keeps the project feasible with limited computational resources.
