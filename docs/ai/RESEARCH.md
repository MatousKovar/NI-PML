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

Use one CARec-supported dataset first, preferably Amazon Electronics. The current encoder shortlist is recorded in [`third_milestone.md`](../../third_milestone.md). Use the same CARec pipeline, item text, split, and random seeds. Compare at shared dimensions 64, 128, 256, and 384 using PCA fitted only on training-item embeddings. Report 768 separately for encoders that support it. Do not expand a smaller encoder and treat that as an equivalent comparison. Report Recall@10 and NDCG@10 separately for warm and cold items.

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

The earlier E5-base-v2/INSTRUCTOR-base pair remains a possible compact comparison, but no final encoder pair has been selected. The expanded candidate shortlist and dimension plan are in [`third_milestone.md`](../../third_milestone.md).

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

## Alternatives to PCA, 2026-10-19

The [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) explicitly lists Matryoshka representation learning support and output dimensions from 32 to 1024. This allows shorter representations without fitting PCA. The [MiniLM card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes a 384-dimensional model, and the [E5-base-v2 card](https://huggingface.co/intfloat/e5-base-v2) describes a 768-dimensional model; neither card documents Matryoshka support for those checkpoints. Arbitrary vector slicing should therefore not be treated as their equivalent trained dimension-reduction feature.

PCA is a proposed common reduction procedure, not an assignment requirement. Using model-specific compression is also a valid study, but would compare encoder-plus-compression systems. The user asked about alternatives; no replacement protocol or new encoder selection has been agreed, and the milestone draft has not been changed on this basis.

## Six-model candidate set, 2026-10-19

The user confirmed a minimum of three instruction-conditioned and three non-instruction models. The methodology proposes [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [MPNet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), and [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) without a free-form task instruction. E5 retains its documented `query:` feature prefix.

Instruction candidates are [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [multilingual E5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct), and [INSTRUCTOR-base](https://huggingface.co/hkunlp/instructor-base). The INSTRUCTOR card documents instruction/text pairs through `InstructorEmbedding`; its [output configuration](https://huggingface.co/hkunlp/instructor-base/blob/main/2_Dense/config.json) specifies 768 output features. This differs from the XL checkpoint used by CARec. No new dependency installation, weight download, or inference test was performed.

Model cards were checked for input conventions and documented output dimensions. Grouping is operational: task instructions versus standard model formatting. It does not establish that all other training differences are controlled. The specific checkpoint choices remain proposed pending compatibility and resource pilots; the minimum group sizes are confirmed.

## Instructed and uninstructed variants, 2026-10-26

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) explicitly discusses use with and without instructions and demonstrates prompted queries and unprompted text. This supports adding paired inference conditions; its retrieval findings do not establish a recommendation gain.
- [Multilingual E5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct#faq) recommends task instructions for query inputs and allows unprompted document inputs. A plain-text recommendation feature condition is technically an ablation proposal, not a documented guarantee of quality or equivalent intended usage.
- The [INSTRUCTOR tokenizer source](https://github.com/xlang-ai/instructor-embedding/blob/main/InstructorEmbedding/instructor.py) accepts two string fields and does not require a nonempty instruction. It tokenizes instructions separately to construct a pooling mask. Thus `["", item_text]` is a candidate way to omit the instruction while preserving the same path. This is a source-code inference, not a runtime result; checkpoint/library compatibility and empty-instruction pooling must be tested.

The user requested these variants. [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md#paired-instruction-variants-added-2026-10-26) records the conditions and comparison controls. Removing an instruction from an instruction-trained checkpoint does not create a new non-instruction-trained model. No models were downloaded or executed during verification.

## Input-format check, 2026-11-01

Rechecked the official cards while preparing notebook inputs: [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2#faq) recommends `query: ` for feature inputs; [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) shows `Instruct: {task}\nQuery:{text}`; [E5-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) shows `Instruct: {task}\nQuery: {text}`; [INSTRUCTOR](https://huggingface.co/hkunlp/instructor-base) shows instruction/text pairs through `InstructorEmbedding`. These document formatting, not recommendation quality or local compatibility. The chosen recommendation instruction is our pilot proposal.

The pinned local [CARec preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L338) concatenates title, brand, and category, removes the dataset category name, and calls `clean_text`. That function also removes quotes and line breaks, alters terminal periods, and empties texts of at least 2000 characters. Our preview formatter instead preserves punctuation, normalizes whitespace, removes only the exact `Electronics` category entry, and does not apply that character-length cutoff. Record these differences before freezing final preprocessing.

On 2026-11-02, directly executed unchanged `RawDataProcessor._preprocess_text_feature` on the first five local metadata records. It succeeded without encoding or training. All five output strings differed from our notebook formatter; the first upstream string retains two spaces after the brand and adds a final period. `clean_text` also returned an empty string for 2,000 plain characters and removed quotes/newlines in a bounded synthetic check. Direct reuse is feasible through the pinned checkout. The method needs `data_name = "Electronics"` and `text_feature_list = ["title", "brand", "category"]`; there is no need to invoke the full `preprocess()` pipeline, which also starts embedding generation.

Later the same day, replaced the notebook formatter with direct upstream method execution and upstream YAML text settings. Executed its dependencies, the current two-product preview, and nine input conditions in IPython. The returned DataFrame matches a direct method call, and the saved preview preserves the original spacing and punctuation. No weights, encoder inference, full-data processing, or upstream edits were involved.

## First-cache planning checks, 2026-11-02

Notebook preparation subsequently completed with the original [Electronics settings](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml) and unchanged [preprocessing methods](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py). Evidence is the executed notebook and `data/processed/electronics_carec_035_seed2023/manifest.json`. The streamed NumPy row selection was compared with pandas sampling, and the adapter plus original method matched full-input upstream preprocessing and splits on a bounded differential fixture. Actual saved data passed checksums and complete partition/text reconstruction from the retained sample. No encoders ran.

Observed raw counts were 20,994,353 reviews and 786,445 metadata rows. Selected 7,348,024 reviews and retained needed metadata for 519,565 sampled products. Final results: 81,520 warm users, 34,220 total products, 600,733 warm interactions and 23,163 cold interactions. Warm split sizes are 452,606/14,853/133,274; cold data cover 18,680 users and 1,797 products. The original selection designated 1,802 cold products before subsequent filtering. All saved artifacts total 263.7 MB. Both raw files remain on disk; a no-raw-file preview/environment-check simulation passed.

Source-row partitions are disjoint but distinct-pair overlap is nonzero: 419 train/validation, 3,844 train/test, and 91 validation/test pairs. Original floor rounding leaves 69,134 users without validation interactions; all 81,520 warm users have test interactions. Seven warm products have no training edge, and one item has an empty text under original cleaning. These are runtime findings that must inform the final evaluation protocol; faithful code reuse alone does not make the benchmark leakage-free. The saved zero-based ID maps are preparation artifacts, not RecBole's final padded indexing.

A full binary line-count/SHA-256 scan of local `data/Electronics.json` found 20,994,353 rows, 11,508,470,738 bytes, SHA-256 `d8b67a799c281fb4111ba4def1c638f34a7bdb38a60deaaa20056156f09efe3f`. At upstream fraction 0.35, pandas selects 7,348,024 rows before filtering. A subsequent preparation attempt was interrupted at the user's request, and its incomplete artifacts and standalone implementation were removed. These counts are source-scan evidence, not a verified processed dataset or confirmation of author-reported split identity. Both original raw files remain present.

After the user requested closer sampling compatibility, re-read the local pinned [Electronics configuration](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml) and [preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py). They specify row sampling at fraction 0.35 with seed 2023, 5-core before metadata matching, cold ratio 0.05, another warm 5-core, cold-user filtering, and random per-user 80/10/10 warm splitting. Metadata IDs are deduplicated; user/product interactions are not deduplicated there, and core degree uses row counts. These are source facts, not an executed reproduction. A smaller fraction, a distinct-pair core, or a different streaming RNG would be a protocol difference.

Official cards reconfirm output dimensions of 384 for [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 768 for [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), and 768 for [E5-base](https://huggingface.co/intfloat/e5-base-v2), including E5's required feature prefix. For N products, three float32 matrices require `N * (384 + 768 + 768) * 4` bytes: 76.8 MB at 10,000 and 153.6 MB at 20,000. This excludes weights, IDs, manifests, and preprocessing outputs; it is a calculation, not a measured cache.

Local Python 3.11 reported Sentence Transformers 5.1.2. Inspection of its installed `SentenceTransformer.encode` signature confirms batch-size, device, prompt, normalization, precision, and NumPy-output options. Current [official API documentation](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html) also describes device, revision, and model-cache controls; implementation must follow the installed version. The same read-only session reported 25,769,803,776 physical RAM bytes through psutil and approximately 117 GiB free disk through `df -h .`. `sysctl` was unavailable under sandbox restrictions. No weight download, encoder inference, full-data scan, throughput, or MPS availability measurement was performed.

## INSTRUCTOR pilot API, 2026-11-03

The installed Sentence Transformers 5.1.2 explicitly recognizes `hkunlp/instructor-base` and disables prompt inclusion in pooling. Its `encode(texts, prompt=instruction)` path tracks prompt token length; `prompt=""` suppresses automatic prompts. The pilot therefore uses this existing API instead of adding the older `InstructorEmbedding` dependency. This updates the notebook input table's API representation while retaining instruction conditioning. Both instruction modes subsequently passed the four-product MPS pilot. Sources: [Sentence Transformers 5.1.2 implementation](https://github.com/UKPLab/sentence-transformers/blob/v5.1.2/sentence_transformers/SentenceTransformer.py), [prompt documentation](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html).

## Six-encoder smoke results, 2026-11-03

All six checkpoints passed real four-product MPS inference in the comparison notebook: MiniLM 384, MPNet 768, E5-base 768, Qwen 1024, E5-instruct 1024, and INSTRUCTOR-base 768 dimensions. Both instruction modes passed for the last three models, giving nine conditions. Outputs are finite, float32, and L2-normalized; saved notebook outputs show parameters, exact sample inputs, and results. The ignored local report `artifacts/encoder_pilots/pilot_results.json` records checkpoint revisions, attention implementations, token limits, backend, load time, and first-batch time. This is a compatibility smoke check, not a throughput or recommendation benchmark.

Qwen's default SDPA implementation on the installed PyTorch 2.6.0/MPS stack returned non-finite vectors for the four-product batch. A one-product probe also aborted inside MPS grouped-query matmul. Changing only attention to eager produced finite vectors for the original batch; both final Qwen modes passed with eager attention. The notebook makes this setting explicit and does not silently fall back to CPU. These are local observations, not a claim that all Qwen/MPS versions fail. Native embeddings for the full catalogue and reduced caches have not been generated.

## Full-catalogue generation measurements, 2026-11-04

Generated and verified the remaining eight native embedding files for 34,220 Amazon Electronics products on the local M4 Pro/MPS environment. The existing MiniLM file was reused. The nine NPZ files total 1,046,056,215 bytes. All condition matrices are float32 with finite values and unit norms; stored product IDs match the frozen catalogue exactly. Full wall time was 2,553.62 seconds. No recommendation quality has been measured.

| Encoder | Conditions generated | Batch | Load, encoding, and save time, seconds |
| --- | --- | --- | --- |
| MiniLM-L6-v2 | 0, existing cache reused | Unknown | Unknown |
| MPNet-base-v2 | 1 | 32 | 106.98 |
| E5-base-v2 | 1 | 32 | 102.19 |
| Qwen3-Embedding-0.6B | 2 | 16 | 1,371.27 |
| Multilingual-E5-large-instruct | 2 | 32 | 706.26 |
| INSTRUCTOR-base | 2 | 32 | 241.53 |

Times for instruction models include both modes and are not per-condition measurements. Qwen used eager attention in float32, with left padding; INSTRUCTOR excluded prompt tokens from pooling. The shared instruction is exactly the current notebook value, recorded with its wrappers in each NPZ and the local report. `artifacts/embeddings/electronics_carec_035_seed2023/generation_report.json` records software versions, checkpoint revisions, backend, token limits, catalogue checksum, and every file checksum. The full notebook rerun succeeded with model construction and saving blocked; existing cache hashes and timestamps stayed unchanged. These are one-run local measurements, not cross-hardware benchmarks.

## Training settings comparison, 2026-11-05

The [paper, section 4.1.4](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf), available through indexed text although direct retrieval returned 404, specifies Adam, batch size 1024, and early stopping on NDCG@10 with patience 30. It searches learning rates 0.0001/0.001/0.01 and weight decay 1e-4/1e-5/1e-6; these are search ranges, not a confirmed single best Electronics configuration. The [pinned training YAML](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/properties/train.yaml) sets learning rate 0.001, batch size 1024, and a 200-epoch limit. Its `reg_weight: 1e-5` is not optimizer `weight_decay`; installed RecBole defaults optimizer weight decay to zero. Our notebook Adam also defaults to zero. The current one-epoch phase-one loop matches the released learning rate and batch size but does not reproduce paper tuning, two-phase completion, validation, or early stopping. No notebook settings were changed and no training was run for this comparison.

## Evaluation source audit, 2026-11-05

The [paper, section 4.1.3](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf), inspected through indexed primary-source text because direct retrieval previously returned 404, defines full ranking and per-user Recall/NDCG at 10/50. The [released runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py) uses full-catalogue warm scores and bypasses MLP for cold evaluation. Its [ColdEvaluator](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/utils/util.py) limits candidates to cold-split products, but applies an item-sized mask to flattened scores before reshaping. The notebook wrapper fixes mask order while retaining original scoring and RecBole metrics.

The same upstream YAML enables repeatable sampling. Installed RecBole 1.1.1 [RepeatableSampler](https://github.com/RUCAIBox/RecBole/blob/v1.1.1/recbole/sampler/sampler.py) returns empty history sets. Its [FullSortEvalDataLoader](https://github.com/RUCAIBox/RecBole/blob/v1.1.1/recbole/data/dataloader/general_dataloader.py) builds history as used products minus current positives. Our notebook supplies cumulative sets through ordinary Sampler so earlier nonpositive history is actually masked. Positives repeated across splits remain unmasked under this loader rule; the known overlap requires a separate benchmark-protocol decision. FullSortEvalDataLoader evaluates only users with positives in the selected split and deduplicates positive products per user. These are static source findings. New metric-example and one-epoch evaluation cells were prepared but not run by the agent.

## MPS uniformity adapter, 2026-11-05

PyTorch's [native distance implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Distance.cpp) excludes MPS from pdist forward/backward supported-device checks. The pinned CARec uniformity loss uses torch.pdist. The notebook therefore delegates this original loss computation to CPU for MPS model instances, with differentiable tensor transfers, while retaining graph propagation on MPS. This source-based compatibility adaptation has not been executed or timed in the notebook.

## External GPU options, 2026-11-05

The user asked about moving long CARec training off their Mac. No provider or spend is authorized yet. Options verified against official sources:

- [ClusterFIT](https://help.fit.cvut.cz/%40master/cs/cluster-fit/index.html) provides FIT compute including GPU nodes; individual access and allocation are unconfirmed. [HelpFIT GPU guidance](https://help.fit.cvut.cz/%40master/en/cloud-fit/processes/gpu.html) directs GPU use to ClusterFIT.
- [MetaCentrum terms](https://docs.metacentrum.cz/en/docs/access/terms) provide free academic/noncommercial compute for Czech institutions. Its [GPU inventory](https://docs.metacentrum.cz/en/docs/computing/gpu-comput/clusters) includes A40, A100, and L40S nodes. Registration, scheduling, and available allocation need checking; do not assume immediate access or DGX grant eligibility.
- [Runpod pricing](https://www.runpod.io/pricing), marked updated 2026-11-04, lists Pods with A40 48GB at USD0.59/hour, RTX A6000 48GB at USD0.59/hour, and RTX4090 24GB at USD0.89/hour. These are displayed rates, not a reserved quote or available-instance guarantee; storage and other charges are additional. A proposed one-hour A40/A6000 trial can measure runtime and peak memory before budgeting nine conditions. No CUDA speedup or fit has been measured.
- [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) confirms varying GPU availability and session/resource limits, including paid plans. It is a candidate for a short pilot; uninterrupted long runs are not guaranteed.

Migration would reuse processed splits and existing embedding caches rather than regenerate encoders or transfer raw datasets. The current training notebook is not yet CUDA-ready end to end: it needs explicit CUDA device selection, a compatible Linux/CUDA environment, and a user-run integration check. The MPS-only CPU uniformity adapter should not be applied on CUDA. No uploads, provisioning, dependency installation, or computations were performed.

## Local graph performance diagnosis, 2026-11-06

Explicitly authorized benchmarks on the real 905,212-directed-edge MiniLM graph reproduce the saved user-run MPS throughput. The pinned [LightGCNConv source](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/recbole_gnn/model/layers.py) materializes gathered/weighted edge vectors. With 384 float32 dimensions a single edge-feature tensor is approximately 1.30 GiB. A cached CPU CSR adjacency with torch.sparse.mm preserves normalized propagation, including duplicate-edge contribution, and reduced measured batch time from roughly 1.34 to 0.45 seconds in phase one and 1.32 to 0.51 seconds in phase two. Loss/gradient checks passed in both phases and a complete isolated phase-one epoch took approximately 98 seconds. The [local performance report](CAREC_PERFORMANCE.md) records methods, timings, source-independent correctness checks, limitations, and reproduction commands. These are local throughput findings; no notebook implementation or trained result was replaced and no larger-dimension speedup is claimed.

## Published encoder NDCG and current-run comparison, 2026-11-06

Verified Table 4 of the final CIKM paper through indexed text from the [author-hosted PDF](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf). Electronics CARec NDCG@10 warm/cold by encoder: INSTRUCTOR-XL 0.0414/0.0383; all-MiniLM-L6-v2 0.0414/0.0343; all-mpnet-base-v2 0.0419/0.0341; bge-base-en-v1.5 0.0414/0.0339; BERT-base 0.0411/0.0194. These are final test results under the authors' protocol, not training targets or directly controlled comparisons with our dataset/masking. The earlier arXiv preprint has a different title/version; use the final CIKM table for this comparison.

Read-only inspection of the current hybrid run, `20261106T073655Z-d4d38c`, found 37 completed phase-one epochs, best validation NDCG@10 0.0433 at epoch 36. The saved notebook shows KeyboardInterrupt during epoch 38; no phase-two history exists in that run. First thirty validation NDCG@10 values exactly match the prior original-MPS run at saved precision. The final-results table still displayed in the notebook belongs to the older completed `20261105T100623Z-36bfb9` run, not the interrupted current run. An absolute NDCG@10 near 0.04 is consistent with the scale of the published Electronics results and alone does not establish a bug. Existing ID, loss/gradient, split-overlap, and protocol limitations still apply.
