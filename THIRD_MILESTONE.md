# Third milestone: Proposed methodology and evaluation

Date: 2026-10-12

## Research question

Do instruction-based item-title embeddings improve CARec recommendations over general-purpose embeddings? Does reducing their dimensions change the result?

## Setup

- Dataset: Amazon Electronics, with CARec's preprocessing and warm/cold protocol. Verify the exact split in the code.
- Recommender: CARec. Keep its architecture, item titles, data split, and training settings fixed. Change only the encoder and final embedding size.
- Popularity snapshot: Hugging Face "downloads last month," checked 2026-10-12. Counts are a rough visibility signal, not a measure of quality or active users.

General-purpose candidates:

| Model | Native dimensions | Downloads last month |
| --- | ---: | ---: |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | 221,180,273 |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | 60,146,924 |
| [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | 18,171,463 |
| [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) | 768 | 824,115 |

Instruction-based candidates:

| Model | Native dimensions / size | Downloads last month |
| --- | --- | ---: |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Up to 1024 dimensions; 0.6B parameters | 8,521,952 |
| [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 1024 dimensions; 0.6B parameters | 1,555,439 |
| [`hkunlp/instructor-large`](https://huggingface.co/hkunlp/instructor-large) | 1024 | 253,153 |

The shortlist is provisional; the full experiment may use fewer than seven models. The smallest native output is 384 dimensions, so that caps the shared size sweep. Use each model's documented input format; E5-base-v2 uses the `query:` prefix for item features.

CARec uses [`instructor-xl`](https://huggingface.co/hkunlp/instructor-xl) with `Represent the Amazon title:`. INSTRUCTOR-large is related, not identical. The XL checkpoint is an optional reference if the Mac handles it; its Hub page showed 9,245 downloads last month. [INSTRUCTOR-base](https://huggingface.co/hkunlp/instructor-base) is a smaller fallback.

## Method

1. Download and preprocess Amazon Electronics with the [CARec code](https://github.com/ChenMetanoia/CARec). Reuse its split if possible; otherwise define one split and use it for every run.
2. Test selected models on a small title batch on MPS. Follow and record each model's documented prefix or instruction format. Generate frozen embeddings once and cache them. Do not fine-tune the encoders.
3. Fit the same PCA procedure on training-item embeddings only. Compare 64, 128, 256, and 384 dimensions across the shortlist. Report 768 separately for compatible models; MiniLM and BGE-small cannot take part in that condition.
4. Train the same CARec setup for every selected encoder and dimension. Tune on validation data only and use the same seeds across conditions. Aim for three seeds if runtime permits.
5. Optionally run ID-only LightGCN as a separate interaction baseline. It is not an embedder and has no learned item vector for unseen items.

## Evaluation

- Report Recall@10 and NDCG@10 for warm and cold items. Record the cold-item definition and which items have no training interactions.
- Report embedding generation time, CARec training and scoring time, and embedding storage.
- Report the mean and standard deviation across runs. Compare encoders at each shared size, then compare sizes within each encoder.

## Compute plan

The M4 Pro uses Apple's GPU through PyTorch MPS, not CUDA. Check model fit and runtime with a small batch first. Whether INSTRUCTOR-XL fits depends on the Mac's memory.

## Decisions still to confirm

- Which encoders fit the Mac's memory and runtime budget?
- What exact CARec split and cold-item definition does the code use?
- Should LightGCN be rerun on the same split?
- Confirm the final encoder subset, prompts, and dimensions after the pilot.

## Sources

- [CARec paper](https://doi.org/10.1145/3627673.3679535) · [code and data instructions](https://github.com/ChenMetanoia/CARec)
- General-purpose encoders: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- Instruction-based encoders: [Qwen3](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [multilingual E5-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct), [INSTRUCTOR-large](https://huggingface.co/hkunlp/instructor-large) · [INSTRUCTOR-large config](https://huggingface.co/hkunlp/instructor-large/raw/main/config.json) · [INSTRUCTOR paper](https://aclanthology.org/2023.findings-acl.71/)
- CARec's [INSTRUCTOR-XL](https://huggingface.co/hkunlp/instructor-xl) · [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/mps.html)
