# Working topic

## Title

Compare general-purpose and instruction-based text embedding models as item representations, and study how the embedding size affects recommendation quality.

## Research question

Do instruction-based item embeddings improve recommendation quality over general-purpose item embeddings, and how does the embedding size affect that comparison?

## Source paper

[CARec: Collaborative Alignment for Recommendation](https://doi.org/10.1145/3627673.3679535)

[CARec code repository](https://github.com/ChenMetanoia/CARec)

CARec is the source paper and experimental starting point. It uses text embeddings as item representations and aligns them with collaborative-filtering information. It also includes an encoder ablation containing `instructor-xl` and several other text encoders. Its main contribution is collaborative alignment, not instruction-based embedding models. The project will extend this part of CARec with a fair comparison and an embedding-size study.

## Planned extension

Keep the recommender, item text, data split, and evaluation fixed. Change only:

1. the text encoder family;
2. the final embedding dimension.

First experiment:

- dataset: Amazon Electronics;
- general-purpose encoder: `all-MiniLM-L6-v2` or E5;
- instruction-based encoder: `instructor-xl`;
- target dimensions: 64, 128, and 256; optionally 768 when both selected encoders naturally produce 768-dimensional vectors;
- metrics: Recall@10 and NDCG@10 for warm and cold items.

Use PCA or a learned linear projection to put all encoders into the same target dimension. Cache the embeddings and do not train an LLM.

## Main supporting papers

- [Let It Go? Not Quite](https://doi.org/10.1145/3705328.3748038) for content initialization, PCA, and cold-start evaluation.
- [MARec](https://doi.org/10.1145/3640457.3688125) for a simple metadata-to-collaborative alignment baseline.
- [INSTRUCTOR](https://aclanthology.org/2023.findings-acl.71/) for instruction-tuned embedding background.
- [E5](https://arxiv.org/abs/2212.03533) for general-purpose embedding background.

Detailed notes and dataset links: [`docs/ai/RESEARCH.md`](docs/ai/RESEARCH.md).
