# CARec experiment plan for AI assistance

Updated: 2026-10-26. This is the complete working plan for implementation and experiment handoffs. It records both confirmed requirements and proposals; an open choice is not authorization to silently treat it as a final scientific decision.

## Read and maintain this plan

Read this file before changing preprocessing, encoder selection, embedding reduction, CARec training, evaluation, or benchmark reporting. Follow the current user request and repository [AGENTS.md](../../AGENTS.md). Keep the user's personal [third_milestone.md](../../third_milestone.md) unchanged unless they explicitly request an edit to that file. Do not synchronize this plan back into that document automatically.

Use [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for stable facts, [PROJECT_TODO.md](PROJECT_TODO.md) for task status, and [RESEARCH.md](RESEARCH.md) for verified findings and sources. Update only affected documents after meaningful progress. Record completed work with its evidence and date. Resolve conflicting older notes using the latest explicit user decision, and preserve unresolved disagreements as open questions.

The user wants to implement and understand the project incrementally. Handle environment and repetitive preparation when requested; teach the experimental logic in small steps unless the user asks for direct implementation. This plan documents future work and does not start a full experiment by itself.

## Objective and confirmed scope

Extend [CARec: Collaborative Alignment for Recommendation](https://doi.org/10.1145/3627673.3679535) by comparing pretrained item-text encoders and embedding dimensions within the same recommender. Measure recommendation quality and resource use. The course requires an evaluated extension of an eligible 2024–2026 research-track paper; CARec's exact track label remains an administrative check in the project notes.

Confirmed requirements:

- Use one dataset only, Amazon Electronics from Amazon Review Data 2018.
- Include at least three instruction-conditioned and three non-instruction encoders. Specific checkpoints remain subject to compatibility and resource pilots.
- Keep text encoders frozen and cache their vectors. Do not train an LLM.
- Use CARec as the starting recommendation method and keep data, text, and evaluation fixed across local comparison conditions.
- The user's current methodology uses title, brand, and category as item text.
- Report warm and cold Recall@10 and NDCG@10, plus storage and runtime.
- Work on the user's M4 Pro MacBook with MPS where supported and a verified CPU path where needed.
- Use Python 3.11, venv, pip, and [requirements.txt](../../requirements.txt). The user requested this simpler setup instead of uv configuration.
- The personal milestone is the user's writing space; this file holds the detailed AI implementation plan.

The study compares selected models. Different architectures, pretraining, size, and language coverage prevent treating every score difference as a causal effect of instruction tuning. A negative quality result or a useful quality/storage tradeoff is a valid outcome.

## Verified starting point

| Area | Recorded evidence | Still unverified |
| --- | --- | --- |
| Raw data | `data/Electronics.json` and `data/meta_Electronics.json` exist, totaling 22,504,102,103 bytes. Ten records per file parsed with expected initial fields. | Full parsing, record counts, duplicates, missing fields, ID coverage, and final split. |
| Environment | Existing `.venv` passed dependency checks and pip-based setup. | Fresh installation on another machine. |
| Upstream | Separate ignored checkout at `external/CARec`, pinned to `bbfc7f385e796840f6eb5c967dfc8714284958ce`. Wrapper imports model and helpers. | Full training and evaluator correctness. |
| Computation | Tiny CARec CPU example passed loss/backward for both phases and scoring. Original graph layer passed CPU/MPS forward/backward. | Complete CARec loss, Trainer, checkpointing, and evaluation on MPS. |
| Encoders | Candidate model cards and input conventions were inspected. | Weight downloads, inference compatibility, memory, runtime, and recommendation quality. |

These are recorded checks from earlier sessions, not checks rerun whenever this file is read. Inspect the current code and task logs before relying on a changed environment. Run `.venv/bin/python -m benchmarks.check_environment` when environment or integration changes justify it. [README.md](../../README.md) documents setup. No complete baseline or benchmark has been measured.

## Open choices before expensive runs

| Choice | Current proposal or known constraint | Required resolution |
| --- | --- | --- |
| Sampling | User wants a smaller sample. Upstream Electronics uses 35%. | Fix the fraction and sampling procedure; record any departure from upstream. |
| Checkpoints | Six candidates below satisfy the intended 3 + 3 design. | Pilot each; replace failures within their group. |
| Reduction | Earlier proposal was common PCA. User has also asked about native dimension reduction. | Choose common PCA or explicitly compare encoder-plus-reduction systems. |
| Dimensions | Earlier shared sweep was 64, 128, 256, 384; 768 was optional. | Confirm the sweep for the chosen reduction protocol and resource budget. |
| Baselines | User prefers using published CARec results to avoid reproducing existing encoders. A local MiniLM overlap run is recommended, not agreed. | Choose published-reference-only reporting or local controlled comparisons, with corresponding claim limits. |
| Seeds and tuning | Three seeds and a common validation-based training configuration were proposed. | Measure one run before committing to total compute. |
| Evaluation | Upstream cold evaluator restricts candidates to cold items. | Audit masks, warm catalogue, histories, users included, and inference before finalizing metrics. |
| Resources | M4 Pro is known. | Record RAM, usable disk, deadline, and acceptable run duration before a large sweep. |

Routine implementation can proceed within confirmed scope. Ask about a scientific choice when the next dependent step needs it; continue independent inspections while awaiting an answer. Do not reinterpret a proposed parameter as a user-confirmed result.

## Candidate encoders and input contracts

| Group | Hugging Face checkpoint | Native dimensions | Input convention |
| --- | --- | ---: | --- |
| Non-instruction | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | Standard item text; overlap with CARec. |
| Non-instruction | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | Standard item text; also present in CARec. |
| Non-instruction | [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 768 | Keep its documented `query:` prefix for feature inputs. |
| Instruction-conditioned | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1024 | Compare with task instruction and without instruction. |
| Instruction-conditioned | [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 1024 | Compare with instruction and with a no-instruction ablation; pilot required. |
| Instruction-conditioned | [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base) | 768 | Compare instruction/text pairs with empty-instruction pairs; pilot required. |

“Non-instruction” means no free-form task instruction in this experiment. E5's required formatting prefix still applies. INSTRUCTOR-base differs from CARec's INSTRUCTOR-XL. Its compatibility with the installed libraries is untested; isolate any required dependency change rather than break the shared environment.

Qwen documents Matryoshka support with dimensions 32–1024. Instruction conditioning and dimension flexibility are separate properties. The MiniLM and E5-base-v2 cards do not document equivalent trained truncation support. Do not treat arbitrary slicing as their supported replacement for PCA. Model-source verification and the INSTRUCTOR output configuration are linked in [RESEARCH.md](RESEARCH.md#six-model-candidate-set-2026-10-19).

### Paired instruction variants, added 2026-10-26

The user confirmed on 2026-10-26 that all three instruction candidates will be compared with and without instructions. Retain the six distinct checkpoints above and add these input conditions:

| Checkpoint | With instruction | Without instruction | Evidence and status |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | A fixed recommendation-related instruction in the documented wrapper. | Plain item text with no instruction wrapper or automatically supplied prompt. | Both modes are documented by the [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B); local inference is untested. |
| Multilingual E5-large-instruct | A fixed task instruction using its `Instruct:`/`Query:` format. | Plain item text, omitting that wrapper and instruction. | Add as an ablation. Its [model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct#faq) recommends instructions on query inputs; an unprompted document path exists, but that does not validate the recommendation setting. |
| INSTRUCTOR-base | The documented `[instruction, item_text]` pair. | `["", item_text]`, preserving the same API and pooling path. | The [official tokenizer implementation](https://github.com/xlang-ai/instructor-embedding/blob/main/InstructorEmbedding/instructor.py) accepts string instruction fields without requiring nonempty content. Empty-instruction feasibility is inferred from code and still needs a local pilot. |

These are input ablations of instruction-trained checkpoints, not additional non-instruction-trained models. Preserve the minimum of three distinct checkpoints in each original group. Removing the instruction does not reverse instruction tuning; this tests instruction use at inference within the recommendation pipeline.

For each pair, fix checkpoint revision, source item text, item IDs, dimensions, pooling, precision, normalization, split, CARec architecture, and paired training seeds. Audit prompt defaults so the no-instruction condition really receives no task instruction. Keep required E5-base-v2 `query:` formatting in its original non-instruction condition; it is not part of these ablations. Select the task instruction before test evaluation and verify truncation so prompt length does not silently change the item content seen by the two conditions.

Cache the two prompt modes separately and train CARec separately for each. Include `prompt_mode` and the actual formatted-input hash in cache and run identities. If using PCA, a comparison intended to hold the projection fixed needs the same fitted transform for both prompt modes. The training-only fit policy for such a shared transform is still open; record it before generating paired reduced caches.

With all three added no-instruction variants, there are six checkpoints and nine encoder/input conditions. At four dimensions and three seeds this gives 108 main CARec training runs, plus three native MiniLM reference runs if retained. This supersedes the earlier 72-run estimate for the six original conditions. A pilot may identify infeasible ablations. Record the failure and resolve the affected condition with the user before reducing the confirmed paired-comparison scope; do not claim all variants have been validated.

## Implementation sequence

### 1. Inspect and define the data contract

Stream or chunk the JSONL files; the raw data are already uncompressed. Retain only fields needed for interaction processing and the agreed text. Never require a single DataFrame containing all 22.5 GB of raw text.

Define explicit handling of absent metadata fields, invalid JSON, duplicate metadata IDs, repeated user/item interactions, and unusable text. Report counts for each exclusion. Keep stable raw product IDs and user IDs alongside any internal integer mappings.

Completion criterion: bounded inspection works, required schemas and ID mappings are defined, and the chosen sampling/text rules are recorded. A preview sample is labeled as a pilot dataset, separate from the final evaluation dataset.

### 2. Generate and freeze one interaction split

For upstream-compatible preprocessing, use this audited order as the reference:

1. Sample interactions, with upstream fraction 0.35 and preprocessing seed 2023 as proposed defaults.
2. Iteratively remove users/items with fewer than five interaction records until stable.
3. Match items to metadata and retain supported interactions.
4. Randomly designate 5% of available items as cold and move all their interactions out of warm data.
5. Repeat 5-core filtering on the warm remainder. Keep cold interactions only for users remaining in warm data.
6. Restrict metadata to retained items and split warm interactions per user into 80/10/10 training, validation, and test partitions.

The user's personal milestone summarizes this more briefly. If implementing a different order, record it as a changed protocol rather than claim exact CARec preprocessing. Five interaction records only imply five distinct neighbors after a corresponding deduplication rule. The final cold proportion can change after filtering.

Keep the split seed distinct from model-training seeds. Save partition files, raw-to-internal ID maps, text construction settings, checksums, and counts. Audit empty per-user partitions caused by rounding and nominally warm evaluation items with no training edges. State how those cases enter metrics. Ensure held-out edges never enter the training graph.

Completion criterion: all encoders can consume the same immutable split and item catalogue; designated cold items have zero train interactions; row/ID mappings and overlap rules pass checks. Final counts are recorded instead of copied from the paper.

Sources: [upstream configuration](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml), [preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py).

### 3. Pilot the six encoders and paired input modes

Start with a small fixed batch, then a representative pilot subset. Use inference mode and evaluate MPS compatibility, batch size, precision, output shape, finite values, elapsed time, and memory. Follow each checkpoint's pooling and input formatting; write down exact instructions, token limits, tokenizer/model revisions, and normalization.

Use one fixed recommendation-related instruction intent for the instruction group, adapted to each API. Choose it before test evaluation. Any prompt comparison uses validation only. Record token truncation rates where practical because equal source text does not imply equal retained tokens.

Completion criterion: at least three candidates per group have a working inference path and measured resource estimate, or documented replacements are selected. Each added prompt mode is verified or marked infeasible, including empty INSTRUCTOR inputs and disabled prompt defaults. A library import alone does not count as an inference pilot.

### 4. Generate and cache native embeddings

Encode the frozen item catalogue once per checkpoint/input configuration. Store stable item IDs with each vector matrix and a manifest. Include model revision, text hash, prompt, pooling, token limit, dtype, normalization, and relevant library versions. Detect stale caches when these inputs change.

Keep native vectors reusable across dimensions and training seeds. A separate pilot cache must not accidentally become the final catalogue. Validate vectors after mapping into RecBole's internal item IDs, including its padding row.

Completion criterion: complete catalogue coverage, unique IDs, expected shapes, finite values, stable row alignment, and cache reload checks. Actual generation time and storage are recorded.

### 5. Apply the chosen dimension protocol

If common PCA is selected, fit one PCA per encoder using unique items with actual training interactions only. Apply the fitted transform to the entire catalogue, including cold items. Save fit IDs, transform parameters, dimensions, standardization/whitening choices, and normalization. Ordinary centered PCA without standardization or whitening was proposed; post-reduction L2 normalization remains open.

If native reduction is selected, use documented functionality where available. Record the exact method for every encoder. A mix of native truncation and PCA compares complete encoder-plus-reduction systems; it does not hold compression method constant. A learned projection is another option but introduces trainable parameters and needs a separate protocol.

Keep native no-reduction references distinguishable from PCA at the same dimension. PCA can change centering even when dimensions are not reduced. Changing CARec dimensions also changes user embeddings and MLP widths, so report model capacity alongside compression effects.

Completion criterion: the reduction decision is recorded, each matrix has the agreed shape and ID order, and any fitted transform excludes evaluation-only items. Infeasible dimensions are marked unavailable rather than created by padding vectors.

### 6. Verify CARec integration and evaluation

Use [benchmarks/carec.py](../../benchmarks/carec.py) to load the pinned checkout. Keep upstream tracked source unchanged by default and document any necessary fixes in our integration. The wrapper currently rejects tracked-source changes. The original preprocessing expects gzip paths and has CUDA device assumptions; importing it is not a working JSONL/MPS pipeline.

Resolve the known checks from [CAREC_OVERVIEW_CS.md](CAREC_OVERVIEW_CS.md#8-místa-která-potřebují-ověřit-před-baseline):

- Correct loaded-vector dimensions, internal IDs, padding, and graph edges.
- Stage-one user learning with frozen item vectors and bypassed MLP; stage-two MLP learning with frozen user table.
- Trainer epoch counters, best validation score, optimizer state, early stopping, and checkpoints at the stage transition.
- Old RecBole checkpoint loading under PyTorch 2.6.
- Single-pair prediction versus full-sort behavior.
- Cold mask shape before/after score reshape and history/padding masking.
- Cold inference recomputation of both graph representations when MLP is disabled.
- Full-model MPS operations, not just the graph convolution. Use a recorded CPU path if needed.

Verify ranking metrics on a tiny manually checkable example before full training. State candidate sets and per-user averaging explicitly. The upstream cold pool contains cold items only; the exact warm catalogue requires inspection. Compare warm and cold metrics as different evaluation conditions. Prevent test results from selecting prompts, hyperparameters, or checkpoints.

Completion criterion: one tiny end-to-end run has verified phase behavior, checkpoint reload, ranking candidates, and expected metrics. Imports and finite loss alone do not complete this step.

### 7. Establish baseline scope and measure one real run

Respect the user's preference to avoid reproducing every published encoder. Published CARec values can be cited as author-reported references. They are not interchangeable with controlled local baselines when sampling, split, reduction, or implementation differs. Exact agreement is unverified even if the dataset name and seed match.

Recommend one local MiniLM overlap baseline for comparing new encoders at the same dimensions; this is still a recommendation, not a confirmed requirement. An ID-only LightGCN baseline is optional and needs a cold-item policy. If using published references only, frame results as an evaluation of our pipeline with external context rather than isolate encoder improvements over the paper's numbers.

Proposed starting hyperparameters are upstream learning rate 0.001 and batch size 1024, validation NDCG@10 for checkpoint selection, and the audited two-phase structure. Confirm effective epoch limits and regularization/dropout in code; YAML fields do not prove use. If batch size must change, use consistent comparison settings or report the difference.

Completion criterion: a real condition completes with validation-selected checkpoints, saved configuration, resource measurements, and documented baseline interpretation. Use its duration to agree on the sweep budget.

### 8. Run the experiment matrix

The expanded proposal uses six checkpoints and nine encoder/input conditions, four shared dimensions, and three paired training seeds, 2023/2024/2025, with one fixed split. That is 108 main training runs if all prompt ablations pass their pilots. A three-seed native MiniLM reference adds three. The matrix and seeds remain provisional until reduction and compute decisions are made.

Prefer retaining the required six-model coverage while negotiating the number of dimensions/seeds if the budget is too high. Confirm scope changes with the user. Reuse native embeddings across conditions, resume completed runs by configuration ID, and save logs/checkpoints/results outside versioned source. Treat failures as recorded outcomes rather than silently omit them.

Completion criterion: every agreed condition has results or an explained failure; all reported scores identify split, model revision, reduction, seed, and training configuration. No test-guided tuning is mixed into the final table.

### 9. Analyze and write the findings

Report warm/cold Recall@10 and NDCG@10. Specify users with positive held-out items and how their metrics are averaged. Show mean and standard deviation when multiple seeds exist; single-run results carry no invented uncertainty estimate.

Compare models at fixed dimensions, then dimension effects within each encoder. Report each encoder as well as group summaries so a group claim does not hide mixed results. Separate native and compressed conditions and author-reported values from local measurements.

Measure embedding generation, reduction, training, and scoring separately. Synchronize MPS for elapsed GPU timings. Record hardware, backend, precision, batch sizes, item count, vector storage, and memory with the measurement method. Distinguish vector bytes from manifests and PCA parameters. Flag timings measured on different backends.

Produce a quality-versus-dimension plot, per-condition results table, resource table, and a limitations section covering one-dataset scope, model heterogeneity, changed recommender capacity, and protocol differences from the paper. A broader novelty claim requires an updated literature check; do not infer absence of prior work from this plan.

Completion criterion: every table entry is traceable to a saved run or an explicitly cited published result, comparisons match the chosen protocol, and conclusions cover both positive and negative findings. Update AI notes; edit the personal milestone or manuscript only when requested.

## Planned artifacts and next action

The following are planned outputs, not files already implemented:

| Artifact | Contents |
| --- | --- |
| Dataset manifest | Source identity, sampling/filtering, text rules, counts, split seed, checksums, ID maps. |
| Encoder pilot report | At least six checkpoints, input formats, working backend, batch/precision, measured memory and time. |
| Embedding cache | Native/reduced matrices, stable IDs, model revisions, configuration, fitted transforms where applicable. |
| Run configuration and results | Split and cache IDs, training seed/settings, device, checkpoints, warm/cold metrics, timings, failures. |
| Analysis outputs | Reproducible tables and figures tied to saved results. |

Keep large data, caches, downloaded sources, and checkpoints ignored by Git; commit code, small configurations, and manifests without raw records. Reuse terms for redistributing upstream code remain unresolved, while local imports were explicitly requested by the user.

Next implementation action: define bounded JSONL loading and the item/interaction data contract in step 1. Settle sampling before freezing the split, reduction before generating reduced caches, and baseline scope before interpreting quality differences. Confirm remaining resource limits before scheduling long runs.
