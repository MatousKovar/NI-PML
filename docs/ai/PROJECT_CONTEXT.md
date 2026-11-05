# NI-PML project context

Last updated: 2026-11-05

## Assignment

Extend a personalised machine-learning method from a 2024-2026 research-track paper and evaluate the extension in a short scientific paper. The assignment explicitly lists CIKM as an eligible venue. See [`assignment.md`](../../assignment.md).

## Working topic and paper

Compare general-purpose and instruction-based text embedding models as item representations, and study how embedding size affects recommendation quality.

The working source paper is [CARec: Collaborative Alignment for Recommendation](https://doi.org/10.1145/3627673.3679535), published at CIKM 2024. CARec provides a text-aware recommender and an encoder ablation, but it does not isolate instruction conditioning or embedding size. The project will extend that experiment. Confirm the paper's research-track label with the instructor before treating eligibility as final.

## Initial experiment

- Use Amazon Electronics as the only dataset, confirmed on 2026-10-15. Use one fixed split following the audited CARec protocol; exact identity with the authors' split is not yet established.
- Keep the recommender, item text, split, and training settings fixed while changing the encoder and target dimension.
- Compare shared dimensions 64, 128, 256, and 384. Report 768 separately for compatible encoders.
- Fit the same PCA procedure on training-item embeddings only. Freeze and cache the text embeddings.
- Report Recall@10 and NDCG@10 on warm and cold items, plus storage and runtime.
- Read [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) before implementing preprocessing, encoders, reduction, training, evaluation, or reporting. It holds the complete AI working plan. `third_milestone.md` is the user's personal draft and stays unchanged unless explicitly requested. Research and source links are in [`RESEARCH.md`](RESEARCH.md).

These are working plans. The final encoder subset, exact cold-item protocol, and whether to use CARec code directly remain open.

## Source audit, 2026-10-15

The read-only audit pins CARec to commit `bbfc7f385e796840f6eb5c967dfc8714284958ce`. Its Electronics preprocessing uses title, brand, and category together, rather than title alone. A title-only experiment would be a deliberate protocol change. Its cold evaluator restricts candidates to items present in the cold split. These details must be fixed and recorded before comparing encoders. The source has not been executed or copied into this project.

The Czech walkthrough, source links, illustrative snippets, and proposed implementation order are in [`CAREC_OVERVIEW_CS.md`](CAREC_OVERVIEW_CS.md). Paper verification and its source-access limits are in [`CAREC_PAPER_CHECK.md`](CAREC_PAPER_CHECK.md).

## Compute and source-code status

The target machine is the user's MacBook Pro with an M4 Pro chip. Use PyTorch MPS where supported, not CUDA. On 2026-10-17, the upstream graph layer passed forward/backward checks on CPU and MPS, and a tiny CARec model passed both phase losses and scoring on CPU. Encoder inference, full training, memory use, and runtime still need pilots.

CARec's public repository is [`ChenMetanoia/CARec`](https://github.com/ChenMetanoia/CARec). On 2026-10-17 the user explicitly requested importing CARec, superseding the earlier project-level deferral of local use. A separate ignored checkout at `external/CARec` pins commit `bbfc7f385e796840f6eb5c967dfc8714284958ce`; upstream tracked source is unchanged. The tree has no obvious license file, so terms for redistributing upstream code remain unconfirmed. See [`RESEARCH.md`](RESEARCH.md#carec-code-and-experiment-structure).

## Working rules

- As clarified on 2026-11-04, prepare notebook code only. The user runs all cells, computations, and tests themselves. Do not execute them without a new explicit request; earlier generation authorization does not authorize future runs.
- Follow [`AGENTS.md`](../../AGENTS.md) and keep confirmed facts separate from proposals and open questions.
- Do not train an LLM. Cache frozen item embeddings and keep experiments small enough for local MPS testing.
- Update the relevant `docs/ai/` note after meaningful project decisions or experiments.

## Implementation workflow, 2026-11-01

The user wants to begin experiments through small, reviewable implementation steps. Explain the overall plan, break the next part into smaller steps, and implement only the step the user explicitly requests. After each step, explain the changes and verification, then stop before implementing the next step. The user asks the assistant to write the requested code while retaining visibility into what happens. This workflow does not authorize starting implementation or experiments from a planning request alone.

## Scope and immediate goal, 2026-10-15

The user confirmed a single dataset and wants to download and inspect it and settle encoder choices next. Use Amazon Review Data 2018 Electronics reviews plus metadata, reached through the official source linked by CARec. Published results may be cited as external references; the dataset name alone does not make them controlled baselines for our PCA experiments. A MiniLM overlap run and a reduced encoder shortlist are recommendations awaiting agreement. See [`DATA_AND_MODELS_CS.md`](DATA_AND_MODELS_CS.md).

## Benchmark environment, 2026-10-17

- Uncompressed raw files exist at `data/Electronics.json` and `data/meta_Electronics.json`, totaling 22,504,102,103 bytes. Ten JSON records from each parsed successfully with the expected first-record fields. This is a bounded sample, not a complete dataset validation.
- `requirements.txt` and `scripts/setup.sh` restore an isolated Python 3.11 environment using standard venv and pip. The user requested this simpler setup on 2026-10-17; the earlier uv configuration was removed. Installed versions include PyTorch 2.6.0, RecBole 1.1.1, PyG 2.6.1, NumPy 1.23.5, Sentence Transformers 5.1.2, and Transformers 4.57.1. This differs from the authors' original CUDA stack.
- `benchmarks.carec.load_carec()` checks commit and tracked-source changes, then imports the original model and data/evaluation helpers. The notebook includes these imports. Select `.venv/bin/python` in the editor.
- `benchmarks.check_environment` verifies imports, bounded data reads, graph operations, and a tiny CPU model. It does not verify the complete training/evaluation pipeline or download encoder weights. Setup and verification passed; see [`../../README.md`](../../README.md).

The pip-based setup was verified against the existing `.venv` after conversion. Dependency checks, CARec imports, CPU smoke checks, and bounded dataset reads passed. This did not rebuild the environment from scratch or rerun GPU checks.

## Methodology draft workflow, 2026-10-19

The user wants to build [`third_milestone.md`](../../third_milestone.md) collaboratively, then shorten and rewrite it themselves. The expanded English working draft follows the existing milestone language and marks unconfirmed settings as proposals. The older shortlist is preserved in an appendix. It does not confirm a final encoder subset or claim completed experiments.

## Minimum encoder coverage, 2026-10-19

The user requires at least three instruction-conditioned and three non-instruction models. The proposed six checkpoints in `third_milestone.md` are MiniLM-L6-v2, MPNet-base-v2, E5-base-v2, Qwen3-Embedding-0.6B, multilingual E5-large-instruct, and INSTRUCTOR-base. The minimum group sizes are confirmed; specific checkpoint feasibility is not. Required format prefixes such as E5's `query:` are distinct from free-form task instructions. Native dimension reduction remains a separate open protocol choice.

## Document ownership, 2026-10-19

The user requested a separate complete plan for AI assistance. [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) is now the implementation-plan reference. Keep [`third_milestone.md`](../../third_milestone.md) as the user's personal document; do not automatically edit it when planning or updating AI documentation. The new plan consolidates confirmed scope, proposed parameters, open choices, known integration issues, and completion criteria. Creating it did not run experiments or settle the remaining protocol choices.

## Instruction-use ablations, 2026-10-26

The user requested suitable encoders in both instructed and uninstructed modes. The AI plan now lists paired variants for Qwen3-Embedding-0.6B, multilingual E5-large-instruct, and INSTRUCTOR-base. Qwen documents both modes; the other two uninstructed variants are ablations requiring a pilot, with INSTRUCTOR empty-instruction support inferred from the official tokenizer code. Six distinct checkpoints remain, with nine input conditions if all ablations work. These variants do not replace the minimum of three distinct models in each original group. See [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md#paired-instruction-variants-added-2026-10-26). No inference or training was run for this update.

The user confirmed on 2026-10-26 that all three instruction candidates, Qwen3-Embedding-0.6B, multilingual E5-large-instruct, and INSTRUCTOR-base, are planned in both instruction modes. This is now also stated in the manuscript methodology. Pilot feasibility remains unverified; it does not make the paired conditions optional without a scope discussion.

## Content-only comparison, 2026-10-26

The user confirmed a local content-only baseline using the same embeddings as CARec. The planned simple method averages vectors of items in each user's training history and ranks unseen candidates by cosine similarity. It shares the split, input vectors, and candidate masks with CARec and uses no MLP or graph propagation. The comparison will test whether CARec adds value, without assuming a gain. The manuscript methodology and AI plan now include it; no implementation or experiment has run.

## Product input preparation, 2026-11-01

The user explicitly requested title, brand, and category as encoder input and a brief explanation in the notebook. Its working formatter joins these fields in that order, decodes HTML entities, collapses whitespace, omits empty fields and the exact redundant `Electronics` category, and preserves case and punctuation. Entirely empty texts raise an error. Product IDs remain separate for vector alignment; no user IDs, ratings, or review text enter the encoder. This differs from upstream's exact cleaning and is not a claim of identical preprocessing.

The notebook prepares five preview products for the existing six candidate checkpoints and nine input conditions. The shared instruction, `Represent this product for personalized recommendation.`, is a pilot choice, not a validated prompt. E5-base retains `query: `; Qwen and E5-instruct use their respective instruction wrappers; INSTRUCTOR uses instruction/text pairs. Inference compatibility, automatic prompt defaults, token truncation, and final dataset preparation remain unverified. No models were downloaded or executed.

## Embedding-generation budget, 2026-11-02

The user wants reusable disk caches and to start with the non-instruction encoders. The current plan interprets this first batch as MiniLM, MPNet, and E5-base, retaining E5's required prefix. Later instruction conditions remain in scope. The user confirmed a maximum of two hours for the first complete three-encoder generation, excluding data preparation and model downloads. Model loading and cache writes count toward that budget. Future CARec training and sweep budgets remain open.

Read-only checks reported 24 GiB of physical RAM through `psutil.virtual_memory().total` and about 117 GiB of available disk through `df -h .`. These are observations from this session, not inference benchmarks. Detailed subset, timing-pilot, and resumable-cache proposals are recorded in the notebook and [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md#proposed-subset-and-embedding-cache-run-2026-11-02). Dataset size and sampling remain unconfirmed.

The user subsequently required sampling as close to CARec as possible. Preserve upstream interaction-row sampling and preprocessing order; the active-user sampling proposal is superseded. Upstream's 35% fraction and seed 2023 are reference settings. The local fraction remains open pending retained counts and the two-hour timing estimate; about 10,000 products is not a confirmed target or cap. Upstream 5-core counts interaction records and does not first deduplicate user/product pairs. Audit duplicates and leakage before choosing any deviation. See the updated notebook plan.

## Notebook-only preparation and cleanup, 2026-11-02

The user set today's objective to prepare and save CARec-like sampled data, then remove the original datasets to free disk space. They then interrupted the standalone implementation and required all sampling/preparation to live directly in `src/carec_encoder_comparison.ipynb`. Preserve the existing jointly prepared preview/text/input cells. Do not create a separate sampling module or test suite without a new request. The large future-work plan cell was removed from the notebook at their request; its proposal remains in the AI plan for reference.

The external preparation process was stopped. Its newly added module, tests, bytecode, and two incomplete sample artifacts were removed. No verified dataset was produced and neither original file was deleted. Continue with small notebook cells under the user's direction. The earlier authorization to remove original datasets applies only after saved replacements have been verified.

## Direct upstream product text, 2026-11-02

The user requested direct reuse of CARec's text preparation. The notebook now calls the pinned `RawDataProcessor._preprocess_text_feature` with `data_name` and `text_feature_list` from upstream's Electronics YAML, replacing the custom formatter described above. Its returned text is preserved without additional cleaning; model-specific input wrappers remain separate. The text-only processor bypasses constructor directory creation and does not call `preprocess()`. The current two-product preview passed and its output is saved in the notebook. This establishes the original text-preparation path for valid matching metadata, not equivalence of the full dataset/split or encoder pipelines.

## Notebook kernel correction, 2026-11-02

The user's saved traceback showed `ModuleNotFoundError: No module named 'torch_geometric'` under system Python 3.13. Reproduced that failure with the same interpreter; the existing Python 3.11 `.venv` already contains the dependency and imports pinned CARec successfully. The notebook now declares kernel `ni-pml`, displayed as `NI-PML benchmarks`, and its data-paths cell checks `sys.prefix` before CARec imports. An already open editor session may still require selecting that kernel and restarting. All current notebook code cells executed successfully in `.venv`; no dependency installation or upstream edit was needed.

Folder inspection on 2026-11-02 confirmed that `.venv/bin/python` links to the Python 3.11.17 runtime inside `.python`; `sys.base_prefix` points there. Both directories are required for the current environment. `.tools` is a separate Python 3.13 environment containing only pip and uv from the older setup; current setup code does not reference it. Approximate allocated sizes from `du -sh` were 986 MB for `.venv`, 77 MB for `.python`, and 39 MB for `.tools`. No directories were removed during this inspection.

VS Code log inspection later that day showed its environment locator discovering `.venv`, while its Python extension retained `.tools` and Jupyter launched `/usr/local/bin/python3`. The `ni-pml` kernelspec was visible only from `.venv`; the actual system Python 3.13 could list only `python3`. Registered the existing kernel with `ipykernel install --user`, and changed `scripts/setup.sh` to repeat that registration. System Python now resolves `ni-pml` to this project's `.venv/bin/python`, which successfully imports ipykernel and torch_geometric. No packages were installed. The live VS Code picker could not be inspected through computer use, so refreshing the window and selecting the registered Jupyter kernel remains a UI handoff.

## Verified CARec sample, 2026-11-02

The user resumed notebook-based preparation after resolving the kernel issue. The notebook now implements and has executed the 35% sample, seed 2023, direct upstream `_preprocess_inter_data`, direct upstream per-user splitting, and original product-text preparation. A small adapter supplies the streamed sample in exactly pandas sampling order instead of sampling that subset again. A differential fixture matched the unchanged full-input upstream method, including repeated pairs and absent metadata. This verifies the implementation against the pinned code, not identity with the authors' raw files or published split.

Saved replacement: `data/processed/electronics_carec_035_seed2023/`, 263.7 MB. It retains 7,348,024 sampled review rows with needed fields, metadata for 519,565 sampled products, final train/validation/test/cold partitions, original product metadata and texts, and explicit zero-based ID maps. The final dataset has 81,520 users, 34,220 products, and 623,896 interaction rows. Partition counts are 452,606 train, 14,853 validation, 133,274 test, and 23,163 cold. Cold evaluation data include 18,680 users and 1,797 products.

The manifest is `verified`. File hashes, sample reload, partition and text reconstruction entirely from saved data, row separation, cold separation, catalogue coverage, ID maps, and warm 5-core all passed. The runtime manifest is stored alongside the data at `data/processed/electronics_carec_035_seed2023/manifest.json`. The redundant documentation copy was removed at the user's request on 2026-11-02; notebooks do not use that copy. Notebook previews and the existing environment check also passed with raw files simulated as unavailable. Both originals remain present; the user may now remove them for this fixed-sample workflow. A different selection from the full population requires the original data again; unused raw fields are not retained.

The original protocol has 419 train/validation, 3,844 train/test, and 91 validation/test overlapping user/product pairs despite disjoint source rows. It also leaves 7 warm products without train edges, 69,134 users without validation positives, and one empty upstream item text. These cases were preserved, not silently corrected. Resolve their evaluation treatment before claiming a leakage-free benchmark or tuning against validation. No encoders or recommender training ran.

## Separate preparation and experiment notebooks, 2026-11-02

The user requested two adjacent notebooks. [`src/data_exploration.ipynb`](../../src/data_exploration.ipynb) owns raw/saved-data previews, original CARec product-text preparation, sampling, splitting, and saving. The sampling cells and their completed outputs were moved unchanged. [`src/carec_encoder_comparison.ipynb`](../../src/carec_encoder_comparison.ipynb) loads the verified final catalogue, four partitions, and ID maps and checks their hashes. It has no raw-file loading, sampling, or splitting code. Preparation remains in notebook cells, with no separate preparation module. The user's current data notebook ends at saving; the existing verified manifest records checks completed earlier.

Verified all experiment code cells with raw files and staging-sample paths blocked. Loaded 34,220 products and all fixed partitions; data file sizes and modification times stayed unchanged. The preparation notebook's imports, paths, previews, original text method, and cache configuration also run independently. Every cell compiles; sampling and saved datasets were not regenerated during this reorganization. Both notebooks select the existing `ni-pml` kernel. Earlier instructions naming the experiment notebook as the preparation location are superseded by this split.

In the subsequent cleanup requested on 2026-11-02, reduced the comparison notebook to four cells containing short bullet notes, paths/environment selection, and read-only saved-data loading. Removed device selection, unused NumPy/PyTorch and model imports, encoder descriptions, and input-format examples from this current stage. Encoder coverage and planned ablations remain in the AI plan for later implementation. All remaining cells passed against the saved catalogue and splits with raw/staging paths blocked; preparation and stored datasets were unchanged.

After the user found the loading cell hard to read, split it into five short stages: manifest, file checks, product catalogue/ID maps, interaction partitions, and summary. Use explicit loops, named intermediate values, and direct assignments for train/valid/test/cold. The comparison notebook still only loads saved data. Differential execution confirmed identical DataFrames, dtypes, row order, manifest, and checked-file list; preparation and stored files were unchanged.

## Comparison notebook cleanup, 2026-11-03

At the user's request, reviewed the comparison notebook again and removed repeated SHA-256/size scanning and the empty trailing cell. It now contains four loading stages: manifest, product catalogue/ID maps, interaction partitions, and summary. Preserve the project-environment guard, verified-manifest requirement, string IDs, float precision, and partition row-count checks. This is ordinary loading of previously verified artifacts, not a fresh full integrity verification. All current cells executed successfully and a differential check confirmed identical catalogue, maps, split DataFrames, types, and row order. Preparation code and dataset files were unchanged. The repository was clean and `main` matched its `origin/main` tracking ref at session start; yesterday's work was already committed as `14507f2`.

The user then explicitly requested removing verification entirely because they will keep the dataset fixed. The comparison notebook now only locates the project, loads product records/maps and interaction partitions, and shows counts/previews. It does not read the manifest, check its status, enforce a specific environment, compare checksums/sizes, or assert row counts. Counts come from loaded frames. This supersedes the earlier instruction to retain loading guards. String IDs and float precision remain unchanged; old/new frames matched exactly with manifest access blocked. The dataset manifest stays on disk as preparation metadata and is no longer required by this loader.

## Encoder smoke pilots, 2026-11-03

The comparison notebook now has separate parameter/input-preview and inference cells for all six candidate encoders. All nine conditions passed a shared four-product batch on MPS with native dimensions and normalized float32 outputs. INSTRUCTOR uses installed Sentence Transformers prompt support with prompt tokens excluded from pooling. Qwen uses explicit eager attention after the default SDPA path produced non-finite values locally. Model weights are cached under the ignored `.cache/huggingface/hub`; each model is released after its section. The notebook retains actual pilot outputs. No full-catalogue generation, PCA, or recommender training ran.

The user then requested smaller implementation steps. MiniLM now has a standalone minimal pilot using only loaded `items` and `PROJECT_ROOT`, with no shared helper. Its two code cells passed on CPU using the cached checkpoint. Other model sections were left unchanged; the user had removed the shared setup cell, so those earlier helper-based sections are not currently standalone. Revise them only when requested.

MPNet now also has a standalone minimal section matching MiniLM, verified on CPU with four cached product texts and `(4, 768)` normalized output. The remaining encoder sections are unchanged. Both minimal pilots explicitly normalize outputs; the original CARec encode call leaves normalization disabled by default, so full-generation normalization remains to be settled.

Multilingual E5-large-instruct now also has a direct minimal pilot, with both instruction modes verified on CPU at `(4, 1024)`. It reuses the user's current `PILOT_INSTRUCTION` unchanged. Model memory cleanup was discussed but is explicitly not implemented at the user's request.

All six encoder sections now use direct minimal pilot code without the removed helper. Qwen and INSTRUCTOR run both instruction modes, with explicit eager attention for Qwen and prompt-excluding pooling for INSTRUCTOR. E5-base now includes its required feature prefix and keeps separate variables from MPNet. The final summary reads current embedding shapes instead of the obsolete helper results. Full current notebook execution passed on CPU for all nine conditions, including finite-value and normalization checks. This supersedes the earlier note that remaining sections require the shared helper. Memory cleanup remains unimplemented.

## MiniLM full-catalogue code, 2026-11-04

At the user's request, MiniLM now encodes the entire loaded catalogue at batch size 32, keeping normalized native 384-dimensional vectors. A separate cell saves ordered string product IDs and vectors plus model identity/revision, empty prompt, normalization, and token limit to `artifacts/embeddings/electronics_carec_035_seed2023/all-MiniLM-L6-v2.npz`. A temporary file is renamed after saving, preserving an older complete file if writing fails. The cache uses a dedicated MiniLM model ID so later encoder cells cannot corrupt its metadata. Real CPU inference on 128 products and exact save/reload in a temporary directory passed. No permanent vector file or full-catalogue run was created. Other encoder pilots and the user's shared setup remain unchanged. Resume and memory cleanup are separate future steps.

## MiniLM cache verified, 2026-11-04

A complete `artifacts/embeddings/electronics_carec_035_seed2023/all-MiniLM-L6-v2.npz` already existed when the user requested the full run. Verified all 34,220 IDs against catalogue order and uniqueness, vector shape `(34220, 384)`, float32 dtype, finite values, unit norms, empty prompt, normalization, and 256-token limit. The first four vectors agree with fresh CPU inference at the stored revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` within `2e-4` absolute/relative tolerance. File size: 53,932,811 bytes. SHA-256: `c76a66a2ac31bc0d2512ea3733ff7bbccf727b448348b0bf38372d2806a60933`. The existing cache was reused; full generation was not repeated, so its original runtime/backend were not measured here. Other encoder caches have not been generated in this step.

The save operation is now a reusable notebook-local `save_embeddings` function, called from MiniLM's save cell. It accepts the output path, ordered IDs, vectors, model/model ID, and optional prompt/normalization metadata. It retains the same NPZ fields and temporary-file replacement, and rejects mismatched ID/vector row counts. Round-trip checks used temporary files and preserved the full cache. Other encoders have not been expanded or wired to save in this refactor.

## Per-condition cache reuse, 2026-11-04

At the user's request, every notebook encoder section now checks its expected NPZ path before loading model weights. Existing files load through `load_embeddings`, which checks catalogue ID order, model ID, prompt, normalization, native dimensions, dtype, finite values, and unit norms. Invalid caches raise an error instead of being overwritten. Instruction models load lazily and generate only missing variants. MiniLM's separate save cell skips an already present file. Verified all nine cache-hit branches with model construction and file writes blocked, and checked that a single missing instruction variant is the only one encoded. This resumes complete condition files, not interrupted batches within one condition.

## All native embedding caches complete, 2026-11-04

Nine NPZ conditions now cover the same 34,220 ordered product IDs: MiniLM at 384 dimensions; MPNet and E5-base at 768; Qwen and E5-instruct at 1024 in each instruction mode; INSTRUCTOR at 768 in each instruction mode. All are float32, finite, and unit-normalized. The files total 1,046,056,215 bytes under `artifacts/embeddings/electronics_carec_035_seed2023/`. The existing MiniLM cache was reused; the other eight files were generated on MPS, one model per process. Qwen used batch 16 and eager float32 attention; other generated models used batch 32. The user's shared instruction was preserved exactly. Total wall time including loading, saving, and verification was 42.56 minutes.

The ignored `generation_report.json` records catalogue/file hashes, exact prompts, model revisions, token limits, dimensions, software versions, device, batch sizes, and per-model total generation times. MiniLM's original generation backend/time remain unknown. Every file passed full ID/order/shape/dtype/finite/norm checks. A real full notebook rerun loaded all nine caches with model construction and saving blocked; timestamps and SHA-256 hashes remained unchanged. Full notebook output now shows cache loading and catalogue shapes. Native generation is complete; PCA and recommender evaluation have not run.

## Recommender notebook scaffold, 2026-11-04

`src/carec_training.ipynb` is the separate notebook for using cached embeddings with CARec. It currently contains architecture notes, configurable selection/loading of one NPZ condition, and saved interaction loading. Its cells have no outputs or execution counts. Subsequent steps will connect RecBole IDs and padding, instantiate the pinned original CARec implementation, inspect the training-only graph, then add the two phases and evaluation. Native MiniLM is the initial input selection; this does not settle the cross-encoder dimension protocol. No code, training, graph construction, or tests were executed when creating this scaffold.

The user subsequently ran the loading cells themselves. At their request, added numbered explanations and unexecuted code for RecBole atomic-file export, configuration through the pinned CARec class, dataset construction, original-ID lookup, embedding row alignment, and a small preview. `benchmark_filename` preserves all four partitions; they share ID mappings, but only the training partition may later enter the graph. Numeric normalization is disabled. The item catalogue exports IDs only; `recbole_item_embeddings` remains a separate NumPy matrix with a zero padding row until the next feature-attachment/model step. Existing user outputs and kernel metadata were preserved. No new code or tests were run.

The user's saved `Dataset(config)` traceback subsequently showed `ValueError: not enough values to unpack` while parsing the interaction header. An existing exported header included untyped `source_row`; the installed RecBole parser splits all header fields on `:` before applying `load_col`. The notebook export now explicitly selects `user_id:token`, `item_id:token`, and `timestamp:float`. Source partitions retain their audit column. Stale export/error outputs were cleared, but no cells, file exports, or tests were executed; the user must rerun the export and dataset cells.

At the user's request, added only the item-vector attachment step after the alignment preview. It calls RecBole's `set_field_property` for `item_emb` as an item-source FLOAT_SEQ of the embedding dimension and assigns the aligned NumPy rows to the current pandas `item_feat` table. A small table preview is included. The installed `_dataframe_to_interaction` implementation will convert this field to a two-dimensional float tensor during the later build step. The new cell is unexecuted; saved-split construction and the CARec model remain subsequent steps.

Step 9 now calls `recbole_dataset.build()` and names its four outputs `train_dataset`, `valid_dataset`, `test_dataset`, and `cold_dataset`, matching the configured benchmark file order. The notebook explains conversion to tensor-backed `Interaction` objects, shared feature/ID mappings, and the need to recreate the original dataset before repeating this in-place conversion. It displays split counts and the training dataset's item-vector shape. The code was not executed; original model construction remains the next step, using `train_dataset` only.

The user ran step 9 and saved the expected partition counts and item tensor shape `(34221, 384)`. The notebook now contains unexecuted configuration, construction, and graph-summary cells for `carec.CollabContex(config, train_dataset)`. MLP widths follow the selected embedding dimension; graph depth 2, MLP settings, and dropout 0.2 follow the original runner. Initialization seed is 2023. The current configuration remains on CPU for inspection. No held-out dataset is passed to the model; no training or propagation code is added in this step, and the agent did not execute the new cells.

On 2026-11-05, the saved notebook graph summary shows the user's successful CPU construction: 81,520 real users, 34,220 real products, 452,606 training interactions, and 905,212 directed entries. Added unexecuted code for one phase-one forward preview over four requested training pairs. The original LightGCN still propagates across the full training graph before selecting those outputs; the model forward normalizes the returned vectors. Eval/no-grad mode and the absence of an optimizer preserve parameter values. The preview shows shapes, finiteness checks, and norms. No agent-run computations or tests occurred; training is still not implemented.

On 2026-11-05, the user requested moving directly into training. Step 15 now prepares one full phase-one epoch with shuffled training rows, Adam at 0.001, batch size 1024, original CARec loss, and only the user embedding table trainable. It prints batch progress and an interaction-weighted mean of batch losses. The model remains on CPU. Rerunning continues current parameters with a fresh optimizer; reconstruct the model for a fresh start. Phase two, validation, early stopping, and checkpoint saving are not yet implemented. All new code remains unexecuted by the agent.
