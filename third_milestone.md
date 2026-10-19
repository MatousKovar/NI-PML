# Proposed methodology and evaluation

- Use one dataset only, Amazon Electronics from Amazon Review Data 2018.
- Use CARec as the starting recommendation method and take results with encoders in paper as baselines.
- Keep text encoders frozen and cache their outputs. Do not train an LLM.
- Compare shared target dimensions of encoders or use PCA on the embeddings.
- Evaluate warm and cold items with Recall@10 and NDCG@10 and record resource use.


## Dataset and preprocessing
-  Sample smller subset of data for easier computation
-  Select 5% of the remaining items at random as designated cold items.
-  Apply iterative 5-core filtering to users and items
-  Split warm interactions randomly per user into training, validation, and test sets in an 80/10/10 ratio.
- item text will be same as in CARec:  title, brand, and category


## Encoders and baselines

We will include at least three instruction-conditioned and three non-instruction encoders. The proposed six-model set below will be checked for compatibility and runtime on the M4 Pro. If a checkpoint cannot run within the available budget, we will replace it within the same group rather than silently reduce the group below three.

| Encoder | Group and role | Native dimension |
| --- | --- | ---: |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Non-instruction; overlap baseline with CARec. | 384 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Non-instruction; second encoder also present in CARec. | 768 |
| [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | Non-instruction; additional general-purpose encoder. | 768 |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Instruction-aware; supports a task instruction and Matryoshka output dimensions. | 1024 |
| [Multilingual E5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | Instruction-conditioned; task instruction in the input. | 1024 |
| [INSTRUCTOR-base](https://huggingface.co/hkunlp/instructor-base) | Instruction-conditioned; separate instruction/text input. Related to CARec's INSTRUCTOR-XL, but a different checkpoint. | 768 |

