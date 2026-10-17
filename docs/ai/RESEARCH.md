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

Use one CARec-supported dataset first, preferably Amazon Electronics. The current encoder shortlist is recorded in [`THIRD_MILESTONE.md`](../../THIRD_MILESTONE.md). Use the same CARec pipeline, item text, split, and random seeds. Compare at shared dimensions 64, 128, 256, and 384 using PCA fitted only on training-item embeddings. Report 768 separately for encoders that support it. Do not expand a smaller encoder and treat that as an equivalent comparison. Report Recall@10 and NDCG@10 separately for warm and cold items.

The improvement over CARec is a controlled study of the encoder family and dimension. CARec supplies the recommendation framework and a useful initial comparison; your work must make the comparison fair enough to ask whether instruction tuning itself remains useful at smaller dimensions.

### Important fairness rule

Do not compare a 384-dimensional MiniLM vector with a 768-dimensional Instructor vector and attribute every difference to instruction tuning. Compare selected encoders at equal dimensions, up to 384 for the full shortlist. Report 768 as a separate condition for encoders that support it.

### Suggested extensions, in order

1. General-purpose versus instruction-tuned embeddings on Amazon Electronics.
2. Dimension ablation at 64, 128, 256, and 384, with 768 reported separately for compatible encoders.
3. Warm versus cold-item results.
4. A second dataset, preferably Yelp or Amazon Office Products.
5. Optional: compare PCA with a small learned linear projection.

Do not train an LLM. Generate the item embeddings once, cache them, and train only the recommender and small projection layers. This keeps the project feasible with limited computational resources.

## Milestone: related work, motivation, novelty, and baselines

Date: 2026-10-07

### Related work to cover

Keep the review focused on the connection between text encoders and recommendation:

1. **Text embeddings in recommendation and cold start.** CARec aligns semantic item vectors with collaborative signals and evaluates warm and cold items. Its encoder ablation includes `instructor-xl`, `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, BGE, and BERT, but the paper's main contribution is collaborative alignment. The encoder ablation is not a controlled study of instruction conditioning or vector dimension. [CARec, CIKM 2024](https://doi.org/10.1145/3627673.3679535) · [paper PDF](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf)
2. **Metadata encoders in cold-start recommendation.** MARec compares TF-IDF, Sentence-Transformer, and Falcon-derived metadata representations. This shows that pretrained text embeddings have already been tested in recommendation, so the project should not claim that simply adding an LLM embedding is new. [MARec, RecSys 2024](https://doi.org/10.1145/3640457.3688125) · [paper PDF](https://assets.amazon.science/5f/3e/d92eda32446891e927ccb0a63b06/marec-metadata-alignment-for-cold-start-recommendation.pdf)
3. **Instruction-conditioned encoders.** INSTRUCTOR embeds text together with a task and domain instruction and evaluates the resulting vectors on general embedding tasks. It explains what an instruction-based encoder does, but does not evaluate recommender datasets. [INSTRUCTOR, Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.71/)
4. **General-purpose encoders.** E5 learns general text embeddings with weakly supervised contrastive pretraining and evaluates them on retrieval and embedding benchmarks. It is useful background for the general-purpose side of the comparison. [E5 paper](https://arxiv.org/abs/2212.03533)
5. **Content initialization and compression.** Let It Go? Not Quite uses E5 item text embeddings, standardization, and PCA to fit content vectors to the recommender's embedding dimension, then evaluates cold and warm items. This supports the feasibility of the proposed compression pipeline. [RecSys 2025 paper](https://doi.org/10.1145/3705328.3748038) · [open paper](https://arxiv.org/html/2507.19473)
6. **Embedding-size work in recommendation.** Existing work such as PEP and CIESS studies the size of trainable recommender feature or ID embedding tables, largely to reduce model memory. That is related background, but it is a different object from compressing a frozen, pretrained text vector used to represent an item. [PEP, ICLR 2021](https://openreview.net/forum?id=vQzcqQWIS0q) · [CIESS, SIGIR 2023](https://doi.org/10.1145/3539618.3591653)

### Motivation

Item text can provide information about an item when its interaction history is sparse or absent. Prior recommendation papers use pretrained text encoders, but general embedding benchmarks do not establish which encoder works best for personalized ranking. The choice also has a resource cost: larger item vectors require more storage and scoring memory. This motivates measuring recommendation quality and resource use together, including a cold-item evaluation.

### Provisional novelty claim

Do not claim that text embeddings, instruction-based embeddings, cold-start recommendation, or embedding-size optimization are individually new. The working contribution is a controlled empirical extension of CARec: compare selected general-purpose and instruction-conditioned item encoders at the same final dimensions within the same recommender and data split, then report warm-item and cold-item ranking quality alongside embedding storage and runtime.

This is a candidate contribution, not a claim that no prior paper has done it. Before the final report, search for closer work that directly compares instruction-conditioned and general-purpose text encoders as recommender item vectors across matched dimensions. If the encoder backbones or training recipes differ, describe the result as a comparison of selected models, not as a causal estimate of instruction tuning alone.

### Baseline plan

Use a small set tied directly to the research question:

- An ID-only collaborative model, preferably the LightGCN baseline used by CARec, gives an interaction-only reference. It cannot represent unseen item IDs without an explicit cold-item handling rule, so report that limitation clearly.
- The same text-aware recommender with selected general-purpose and instruction-conditioned encoders is the main comparison. Keep the model, text fields, split, and training settings fixed.
- Repeat the selected encoder conditions at common final dimensions of 64, 128, 256, and 384, using the same PCA procedure. Report 768 separately for compatible encoders.
- Treat the full CARec result as a reference or reproduction target if implementation time permits. CARec compares many recommender baselines, including NeuMF, DirectAU, NGCF, PinSage, LightGCN, SimpleX, and NCL; reproducing all of them is not needed to answer the narrower encoder question.

Report Recall@10 and NDCG@10 separately for warm and cold items, plus vector storage and embedding-generation/scoring time. Keep model choice and dimension as separate factors in the results table.

## Milestone draft status

Date: 2026-10-12

The manuscript now contains a shorter introduction and concise related-work, motivation, novelty, and baseline sections. The novelty is framed as a controlled matched-dimension extension, since CARec already compares several text encoders. The open decisions above remain unresolved.

## Encoder candidates for Apple Silicon

Date: 2026-10-12

- `intfloat/e5-base-v2` is a general-purpose English encoder with 768-dimensional output and about 0.1B parameters. Its model card recommends the `query:` prefix when embeddings are used as model features. [Model card](https://huggingface.co/intfloat/e5-base-v2)
- `hkunlp/instructor-base` accepts task instructions and has a 768-dimensional T5-base configuration. It is a smaller alternative to CARec's `instructor-xl`, so using it changes the exact encoder while retaining the CARec recommendation framework. [Model card](https://huggingface.co/hkunlp/instructor-base) · [configuration](https://huggingface.co/hkunlp/instructor-base/blob/main/config.json)
- PyTorch provides an MPS backend for Apple GPUs through Metal. This supports local inference without CUDA, but actual model/library compatibility and runtime should be checked with a small pilot. [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/mps.html)

The earlier E5-base-v2/INSTRUCTOR-base pair remains a possible compact comparison, but no final encoder pair has been selected. The expanded candidate shortlist and dimension plan are in [`THIRD_MILESTONE.md`](../../THIRD_MILESTONE.md).

## Hugging Face popularity snapshot

Date checked: 2026-10-12. Counts below are the Hugging Face model-card "downloads last month" values. They are a visibility signal only, not evidence of quality or active users.

| Group | Model | Downloads last month | Notes |
| --- | --- | ---: | --- |
| General-purpose | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 221,180,273 | 384-dimensional output; 22.7M parameters. |
| General-purpose | [`bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 60,146,924 | 384-dimensional output; 33.4M parameters. BGE v1.5 is designed to work without an instruction, though the card also gives an optional search prompt. |
| General-purpose | [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 18,171,463 | 768-dimensional output; about 0.1B parameters. |
| General-purpose | [`e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) | 824,115 | 768-dimensional output; about 0.1B parameters. The model card recommends `query:` for embeddings used as features. |
| Instruction-aware | [`Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 8,521,952 | 0.6B parameters; supports user-defined output dimensions from 32 to 1024. |
| Instruction-based | [`multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 1,555,439 | 0.6B parameters; 1024-dimensional output; its card requires an instruction for query inputs. |
| Instruction-based | [`INSTRUCTOR-large`](https://huggingface.co/hkunlp/instructor-large) | 253,153 | Instruction-finetuned encoder; provide a task-specific instruction with each item title. |

CARec's exact `INSTRUCTOR-XL` checkpoint is an optional reference, not part of the popularity shortlist. Its model card showed 9,245 downloads last month at the same check. `INSTRUCTOR-base` remains a smaller fallback. All counts can change; recheck them before writing the final report.

## CARec code and experiment structure

Date checked: 2026-10-12.

The [official CARec repository](https://github.com/ChenMetanoia/CARec) documents an environment based on Python 3.8.5, PyTorch 1.12.1, RecBole 1.1.1, and CUDA 11.3.1. Its README says to download Amazon data separately. The [runner](https://github.com/ChenMetanoia/CARec/blob/main/run_collabcontex.py) maps a short, hardcoded list of encoder names to 384- or 768-dimensional inputs, then sets CARec's `embedding_size` from that input size. Encoder choice and recommender dimension are therefore coupled in the published runner.

The repository's visible top-level file list has no license file. Do not copy or modify its source inside this project until reuse terms are confirmed. The CARec source is not present locally; GitHub was unreachable from the shell during this check, so no code was downloaded.

**Implementation recommendation, not yet a project decision:** keep experiment runs in a small Python runner with explicit encoder, target dimension, seed, and data-split settings. Cache frozen item embeddings, reduce them to the selected target dimension, and pass them through one embedding-loading interface to CARec. Use a notebook only to inspect results and make plots. Keep downloaded CARec files separate and unchanged if reuse permission is confirmed. Unit tests should check embedding shape, item-ID alignment, and adapter behavior, not launch full experiments.

## Notebook scaffold

Date: 2026-10-12.

The requested starter notebook is [`carec_encoder_comparison.ipynb`](carec_encoder_comparison.ipynb). It has the project introduction, CARec links, workflow outline, imports, and MPS-first device selection. It does not load data or run a model. The final choice between notebook-based experiments and a separate runner remains open; revisit it after the first baseline.

## Pinned CARec source audit

Date checked: 2026-10-15. Read-only source inspection of commit [`bbfc7f385e796840f6eb5c967dfc8714284958ce`](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce). No training, dependency installation, or upstream source reuse was performed.

- Electronics uses a 0.35 interaction sample, 5-core filtering, random 5% cold-item selection, another warm 5-core pass, and filtering of cold users before the per-user 80/10/10 warm split. The input text combines title, brand, and category. These are code facts, not a claim of reproducing the published dataset counts. [Configuration](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml), [preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py).
- The first training stage learns user embeddings with the MLP bypassed; the second freezes that user table and learns the MLP. Graph propagation can still change resulting user vectors. The cold runner disables the MLP before recomputing graph representations. [Model](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/model/collabcontex.py), [runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py).
- Cold evaluation masks candidates outside the cold split. Its mask is applied before reshaping flat scores, which needs a batch-shape runtime check. Other pending checks include Trainer state between phases and the `predict` attribute path. [Evaluator](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/utils/util.py#L393).
- The encoder path hardcodes CUDA device 2 for MiniLM, MPNet, and INSTRUCTOR. MPS requires more than changing the local notebook device. The recursive tree contains no obvious license file; the existing reuse question remains open. [Encoder implementation](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L163), [tree](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce).

See [`CAREC_OVERVIEW_CS.md`](CAREC_OVERVIEW_CS.md) for the full Czech guide and proposed next steps, and [`CAREC_PAPER_CHECK.md`](CAREC_PAPER_CHECK.md) for paper statements and the PDF access limitation. The guide's Python snippets passed syntax parsing only; they were not run against data or scientific dependencies.

## Dataset access and reduced shortlist, 2026-10-15

The project now uses one dataset only, Amazon Electronics. The [official 2018 data page](https://nijianmo.github.io/amazon/index.html) linked by CARec routes full per-category reviews and metadata through an [access form](https://forms.gle/UEkkJs69e7Z5A5Ps9). The form was read but not submitted. File downloads and sizes were not verified. Use full `Electronics.json.gz` plus `meta_Electronics.json.gz`; substituting a prefiltered 5-core file changes the audited input protocol.

The proposed reduced shortlist and distinction between published references and controlled local baselines are in [`DATA_AND_MODELS_CS.md`](DATA_AND_MODELS_CS.md). Current model cards confirm [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and [multilingual E5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) as candidates; none has been downloaded or piloted. Older suggestions to add a second dataset are superseded by the confirmed single-dataset scope.

## Installed environment and local source integration, 2026-10-17

The user explicitly requested local CARec imports. An ignored checkout at `external/CARec` pins [`bbfc7f385e796840f6eb5c967dfc8714284958ce`](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce), with no tracked-source changes. This supersedes the earlier project-level deferral of local source use, not the unresolved redistribution terms.

`requirements.txt` now records the installed library versions. On 2026-10-17 the user requested standard venv/pip setup, replacing the original uv files without changing the benchmark dependencies. The setup uses Python 3.11, PyTorch 2.6.0, RecBole 1.1.1, PyG 2.6.1, NumPy 1.23.5, Transformers 4.57.1, and Sentence Transformers 5.1.2. The CPU model smoke check passed both phase losses/backward and full-sort scoring on an in-memory graph. The original LightGCN layer passed forward/backward on CPU and MPS. Imports of the original model, cold evaluator, data helper, RecBole Config/Trainer, and SentenceTransformer succeeded. This is runtime evidence for this machine, not a full CARec reproduction or MPS training claim. [Original requirements](https://github.com/ChenMetanoia/CARec#requirements), [RecBole 1.1.1](https://pypi.org/project/recbole/1.1.1/), [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

The raw files are already uncompressed and total 22,504,102,103 bytes. A ten-record sample per file parsed successfully; initial fields match the expected Amazon interaction/metadata schema. No complete dataset scan, preprocessing, encoder download, or benchmark was run. `README.md` documents restore commands and remaining checkpoint, Trainer, and evaluation checks.

The requirements-based setup passed on the existing environment with package downloads disabled. `pip check` reported no broken requirements; the environment smoke check passed on CPU and read ten records per dataset file. MPS was unavailable inside this verification sandbox; the earlier MPS result remains separate. No fresh installation was tested.
