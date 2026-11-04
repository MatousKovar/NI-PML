# Project TODO

Last updated: 2026-11-04

## Confirmed direction

- [x] Working topic: compare general-purpose and instruction-based item embeddings, including embedding size.
- [x] Only dataset: Amazon Electronics, confirmed 2026-10-15.
- [x] Draft the related-work, motivation, novelty, and baseline milestone.
- [x] Draft the proposed methodology and evaluation in [`third_milestone.md`](../../third_milestone.md).
- [x] Restore the requested CARec encoder-comparison notebook scaffold at [`carec_encoder_comparison.ipynb`](carec_encoder_comparison.ipynb).

## Open decisions and next steps

- [x] 2026-11-04: create `src/carec_training.ipynb` with concise architecture bullets, selection/loading of one cached embedding condition, and loading of the saved interaction splits. Default to MiniLM for the first integration. Leave RecBole ID mapping, model construction, training, and evaluation as subsequent user-requested steps. No notebook cells or tests were run.
- [x] 2026-11-04: user explicitly requires code-only notebook edits from now on. The user will run all cells, computations, and tests themselves; do not execute these without a new explicit request.
- [x] 2026-11-04: add per-condition cache checks to all notebook encoder sections. Load existing NPZ files before constructing any model; lazily load instruction models only for missing variants. Keep MiniLM's separate save cell, which now skips existing files. Check cached ID order, model/prompt settings, shape/dtype, finite values, and unit norms; mismatches raise rather than overwrite. Verified all nine cache-hit branches without model construction or writes, plus a single missing instruction variant and reordered-ID rejection.
- [x] 2026-11-04: all nine full-catalogue conditions are saved and verified, 34,220 products each, totaling 1,046,056,215 bytes. Reused MiniLM and generated the other eight files on MPS, one model per process. Full run took 42.56 minutes. The ignored `generation_report.json` records revisions, exact prompts, catalogue checksum, software, backend, batch sizes, timings, and file checksums. The full current notebook rerun passed with model construction and saving blocked; all file timestamps/checksums stayed unchanged.
- [x] 2026-11-04: extract MiniLM saving into notebook-local `save_embeddings(path, item_ids, embeddings, model, model_id, prompt="", normalized=True)`. Keep temporary-file replacement and the existing NPZ schema; MiniLM now calls the function. Accept per-condition paths and prompt settings for reuse by other encoders. Validate matching ID/vector row counts before writing. Verified exact round-trip compatibility against the existing cache in a temporary directory, custom metadata, and mismatch rejection; the full cache is unchanged.
- [x] 2026-11-04: on the user's request to run full MiniLM generation, found its complete cache already on disk and verified it instead of repeating the full encoding. Loaded 34,220 ordered IDs and `(34220, 384)` float32 vectors; all values finite and unit-normalized, IDs match the saved catalogue exactly. The first four vectors match fresh CPU inference using the stored checkpoint revision. File size is 53,932,811 bytes. No other encoder generation ran.
- [x] 2026-11-04: expand only MiniLM to encode all loaded products with batch size 32, a progress bar, and finite/shape checks. Add a separate atomic `.npz` save cell with ordered product IDs, embeddings, checkpoint revision, prompt, normalization, and token limit. Verified real inference and exact save/reload for 128 products on CPU; full-catalogue generation has not run. Other encoder pilots remain unchanged. This first save step restarts interrupted encoding; resumable chunks and model cleanup remain deferred.
- [x] 2026-11-03: complete the remaining minimal sections. Qwen and INSTRUCTOR now have fact bullets, exact previews, and direct paired inference without helpers. Qwen explicitly uses eager float32 attention and left padding; INSTRUCTOR excludes prompt tokens from pooling. Fix E5-base's missing `query: ` prefix and separate model/vector names. Replace the obsolete helper-based summary with actual output shapes. The entire current notebook passed offline on CPU, all nine conditions finite and unit-normalized at expected dimensions. Preserve the shared instruction and existing MiniLM, MPNet, and E5-instruct cells. No memory cleanup or bulk generation added.
- [x] 2026-11-03: simplify only multilingual E5-large-instruct into model-fact bullets, exact input previews, and a direct pilot using the user's existing shared instruction. Both modes passed offline on CPU with finite normalized `(4, 1024)` vectors. Preserve all other cells and the user's instruction wording. The user asked about releasing models after pilots but explicitly deferred implementing memory cleanup.
- [x] 2026-11-03: add only E5-base Markdown bullets covering approximate parameter count, layers, output dimension, required feature prefix, language, and token limit, with its official model card. All code cells, outputs, and other notebook cells remain unchanged.
- [x] 2026-11-03: simplify only MPNet to match the minimal MiniLM section: four fact bullets, input preview, and direct four-product encode call without helpers. Exact cells passed offline on CPU with finite normalized `(4, 768)` vectors; measured 109,486,464 parameters and 12 layers. Retain the same explicit normalization as MiniLM; upstream normalization equivalence is not claimed. All other notebook cells are unchanged.
- [x] 2026-11-03: simplify only MiniLM into four model-fact bullets and two direct code cells: select/preview four texts, then load and encode. Remove its helper dependency. Verified the exact new cells on CPU with cached weights in offline mode: finite normalized `(4, 384)` output; measured 22,713,216 parameters and six layers. Other notebook cells are unchanged.
- [x] 2026-11-03: implement six separate encoder sections with model ID, native dimension, instruction flag, exact sample input, and four-product inference pilots. Instruction encoders have both modes, giving nine conditions. Use existing Sentence Transformers prompt support for INSTRUCTOR and exclude its prompt from pooling. Models load sequentially and release memory; no bulk generation or embedding cache implementation. All six checkpoints and nine input conditions passed on MPS, four products each. Saved outputs in the notebook and local `artifacts/encoder_pilots/pilot_results.json`; checked shapes, finite values, and unit norms. Qwen requires eager attention here because SDPA produced non-finite vectors.
- [x] 2026-11-03: add a six-model candidate table and documented input formats to the comparison notebook. Retain paired instruction/no-instruction conditions for Qwen, E5-instruct, and INSTRUCTOR; instruction wording and local pilots remain pending. No inference code or data changed.
- [x] 2026-11-03: review and trim the comparison notebook to manifest, catalogue/maps, partitions, and summary. Remove repeated checksum scanning and the empty cell; retain verified-status, environment, ID types, float precision, and row-count checks. All cells pass and old/new DataFrames match exactly. No changes to preparation or stored data; no encoder work started.
- [x] 2026-11-03: remove all comparison-notebook verification at the user's explicit request. Remove manifest reading/status checks, environment enforcement, and row-count assertions; display actual loaded counts. Catalogue, maps, and all four splits match the previous loader exactly; manifest access was blocked during verification. Preparation and data files are unchanged.

For implementation details and completion criteria, read [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). Keep the user's `third_milestone.md` unchanged unless they request changes to it.

## Incremental experiment start, 2026-11-01

- [x] Record the user's workflow: implement one explicitly requested small step, explain and verify it, then wait for the next instruction. No experiment code was added or run in this planning session.
- [x] 2026-11-01, step 1.1: add file-path checks, a bounded JSONL preview reader, and separate five-record review/metadata tables to [`src/carec_encoder_comparison.ipynb`](../../src/carec_encoder_comparison.ipynb). The data section runs independently of device/CARec imports; later CARec imports reuse its project root. Verified the four new code cells in IPython with Python 3.11 from both the repository root and `src/`: five dictionaries per file and two table displays. Notebook JSON parses and all code cells compile. Preview outputs remain unsaved; raw files are unchanged. This checks only file beginnings, not full data quality or an evaluation split.
- [x] 2026-11-01, step 1.2: add a short field glossary and an eight-field type/missing-value summary in the notebook. Verified preview and summary cells in IPython: all selected fields are present and nonempty in the first five records of each file; category values are lists. JSON syntax errors still stop the existing reader. These findings apply only to the preview; no filtering or join was performed. Existing saved outputs were preserved; new outputs remain unsaved.
- [x] 2026-11-01, step 1.3: add product-text rules, the six-checkpoint input table, a short CARec explanation, and prepared inputs to the notebook. Verified five product texts and nine conditions with matching item order in IPython, without the field-summary or CARec cells. HTML/whitespace cleanup and empty-text failure also passed. The instruction is a pilot choice; no weights, tokenization, embedding generation, or training ran. Existing cells and outputs were preserved.
- [ ] Proposed next block: agree sampling and exclusion rules, prepare and validate one fixed warm/cold split, then pilot MiniLM and evaluate content-only and CARec on the same inputs. A tiny data preview is not an evaluation split.

The unfinished steps above are proposals awaiting the user's command. Encoder feasibility, reduction protocol, dimension sweep, and total compute remain open as recorded in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

## Earlier open decisions

- [x] 2026-11-02: write the subset/cache proposal in the AI experiment plan. Its long notebook copy was later removed at the user's request. The user confirmed a two-hour first-generation ceiling, excluding preparation/downloads; no encoder inference ran.
- [x] 2026-11-02: user requires sampling close to CARec, superseding the proposed active-user selection and fixed product/user targets. Updated the notebook and AI plan; no preprocessing ran.
- [x] 2026-11-02: use upstream fraction 0.35 and seed 2023, inspect retained core/cold counts, and record duplicate-pair and split-rounding issues without altering the original method. Encoder throughput remains unmeasured.
- [ ] Implement directly in notebook cells, in separate user-requested increments: bounded interaction staging, subset/split/catalogue freezing, MiniLM inference and resumable cache, MPNet/E5 pilots, then bulk generation within the confirmed budget. Proposed inference is full-dimensional L2-normalized float32, with token settings and truncation recorded.
- [x] 2026-11-02: stop the standalone preparation process after the user's correction; remove its new module/tests and incomplete outputs, and remove only the long plan cell from the notebook. Preserve every other current cell, output, and user edit. Both original datasets remain present.
- [x] 2026-11-02: complete sampling, original CARec preparation, saving, full reconstruction, and verification directly in notebook cells. Verified output is `data/processed/electronics_carec_035_seed2023/`, 263.7 MB; its runtime `manifest.json` records the configuration and checks. Original files remain present for the user to remove; notebook previews and environment inspection also work without them.
- [x] 2026-11-02: split notebooks at the user's request. Move exploration, original text preparation, sampling, split, save, and verification cells/output to [`data_exploration.ipynb`](../../src/data_exploration.ipynb). The encoder notebook loads only verified final artifacts and retains its input examples/model imports. All experiment cells passed with raw and staging inputs blocked; preparation setup/previews run independently. Dataset files were unchanged and sampling was not rerun.
- [x] 2026-11-02: apply `unslop` to all seven data-exploration Markdown cells and rewrite explanations as short bullet points. Verified that code cells, outputs, execution counts, and notebook metadata are unchanged. No data processing ran.
- [x] 2026-11-02: revise the comparison notebook into a four-cell saved-data loader with concise Markdown bullets. Remove premature device/model imports and encoder-input examples. All remaining cells pass with raw/staging inputs blocked, loading 34,220 products and four splits; data exploration and stored artifacts are unchanged. The research scope and encoder plan remain intact.
- [x] 2026-11-02: improve the loader's readability at the user's request. Split the dense cell into five stages with short bullet notes, explicit loops, named intermediate values, and direct partition assignments. Differential execution of old/new loaders confirms identical frames, dtypes, order, manifest, and file checks; all current cells execute and stored data remain unchanged.
- [x] 2026-11-02: remove the unused duplicate manifest snapshot from `docs/ai/` at the user's request and update documentation references. Keep the runtime manifest alongside the dataset because notebook loading uses it.
- [x] 2026-11-02: prepare the user-requested data-exploration commit. Both current notebooks parse and their code compiles without saved errors; setup shell syntax and all dataset hashes in the manifest pass. Preserve the user's current 18-cell data notebook, which ends at saving. Existing dataset verification is recorded in the manifest; a fresh rebuild still needs verification before raw-file deletion.
- [ ] Before benchmark training, decide how to handle 3,844 overlapping train/test user/product pairs, 419 train/validation pairs, 91 validation/test pairs, 7 warm products without train edges, 69,134 users without validation positives, and one empty original product text. Preserve the verified upstream-compatible dataset as a reference and version any changed protocol separately.
- [x] 2026-11-02: replace the custom product-text formatter in the notebook with a direct call to pinned `RawDataProcessor._preprocess_text_feature`, using Electronics settings from upstream YAML. Verified the user's current two-record metadata preview and all nine formatted conditions in IPython. Saved the original-method text output in the notebook and cleared stale encoder-input output. No upstream edits, encoder execution, full preprocessing, or deletion ran.
- [x] 2026-11-02: diagnose and fix notebook kernel selection for the saved `torch_geometric` import error. System Python 3.13 reproduces the failure; project Python 3.11 succeeds. Set the existing `ni-pml` kernelspec, add an early environment check, and verify every notebook code cell under `.venv`. The wrong-interpreter check gives a kernel-selection message. No new tests/module files or dependency installation were added.
- [x] 2026-11-02: investigate VS Code discovery using its Python/Jupyter logs. `.venv` is discovered, but the notebook retains a system interpreter and the named kernel was registered only within `.venv`. Register `ni-pml` in user Jupyter, update setup/README, and verify visibility from system Python and the resolved kernel's imports. UI refresh/selection must still be verified in the user's editor.

- [x] 2026-10-15: Prepare a source-backed CARec walkthrough with repository map, training stages, split analysis, and illustrative snippets. See [`CAREC_OVERVIEW_CS.md`](CAREC_OVERVIEW_CS.md).
- [x] 2026-11-01: user explicitly requested title + brand + category for encoder inputs, superseding older title-only proposals. The notebook documents its preview cleaning differences from upstream. Final preprocessing still requires the sampling and exclusion decisions; the personal milestone remains unchanged.
- [ ] Audit a tiny cold evaluation: candidate mask shape, cold-only candidates, user-vector recomputation, and history masking. Check the two-stage Trainer state and single-pair prediction path before a baseline.
- [ ] Suggested first shared implementation step: define the item-ID/text and embedding-cache contract, then pilot one small encoder. This is proposed, not implemented.
- [ ] Confirm with the instructor that CARec's CIKM 2024 paper is a research-track paper.
- [ ] Confirm redistribution terms before publishing upstream source. The user explicitly authorized local CARec imports on 2026-10-17; the separate ignored checkout is unchanged. No obvious license file is present. See [`RESEARCH.md`](RESEARCH.md#carec-code-and-experiment-structure).
- [ ] Decide whether to use CARec directly or implement a smaller compatible recommender if code reuse is unavailable.
- [ ] Record available RAM, disk space, deadline, and API restrictions. The M4 Pro machine is known; MPS compatibility and runtime still need a pilot.
- [ ] Pilot a small encoder subset and choose the final models based on memory and runtime.
- [ ] Verify CARec's exact data split and cold-item definition. Decide whether to rerun LightGCN on the same split.
- [ ] Run the first Amazon Electronics baseline, then compare selected encoders at 64, 128, 256, and 384 dimensions. Report 768 separately only for compatible models.
- [ ] Report warm/cold Recall@10 and NDCG@10, embedding storage, and generation/training/scoring time.
- [x] Remove the second dataset from the current scope, per user instruction on 2026-10-15.

## Immediate session, 2026-10-15

- [x] 2026-10-17: Raw Electronics reviews and metadata are present as uncompressed JSON files in `data/`.
- [x] 2026-10-17: Read ten records per raw JSON file. Expected first-record fields, including `asin`, are present. Full ID coverage and data quality remain unchecked.
- [x] 2026-10-19: User requires at least three instruction-conditioned and three non-instruction encoders, superseding the earlier three-model minimum.
- [ ] Separate author-reported numbers from own results. Decide on one local overlap baseline rather than reproducing every model in the paper.

## Environment setup, 2026-10-17

- [x] Ignore raw data, local environments, caches, upstream checkout, and runtime outputs. Neither dataset file was tracked.
- [x] Install an isolated Python 3.11 environment and lock dependencies.
- [x] Pin an unchanged CARec checkout and expose model, cold evaluator, and data preparation imports in the notebook.
- [x] Verify setup restoration, imports, ten-record raw-data samples, CPU model losses/scoring, and CPU/MPS graph forward/backward.
- [ ] Implement bounded data loading and freeze one split without loading the 22.5 GB raw data into a single DataFrame.
- [ ] Before a full baseline, check two-stage Trainer state, cold candidate masking, MPS model-loss support, and old RecBole checkpoint loading with PyTorch 2.6.

- [x] 2026-10-17: Replace uv configuration with `requirements.txt`, standard venv/pip setup, and updated notebook root detection. Preserve the installed environment and pinned library versions.
- [x] Verify the pip-based setup against the existing environment, including dependency consistency and the environment smoke check.

## Methodology milestone, 2026-10-19

- [x] Expand and rename the working document to [`third_milestone.md`](../../third_milestone.md), preserving the earlier candidate pool in an appendix. It separates confirmed scope from proposed model selection, preprocessing, PCA, training, evaluation, and resource reporting. No experiments were run for this edit.
- [ ] Review the draft together, starting with the encoder set and one local overlap baseline. The smaller shortlist, metadata text, PCA normalization, seeds, and detailed candidate pools remain proposals.
- [ ] Resolve dimension-dependent recommender capacity, nominally warm items absent from training, and candidate masking before making controlled-comparison claims.
- [ ] Discuss whether to keep common PCA, compare model-specific compression, or choose encoders with documented Matryoshka support. Qwen supports smaller outputs; MiniLM and E5-base-v2 do not document equivalent support. See the 2026-10-19 research note. No protocol change has been confirmed.

- [ ] Pilot the six proposed checkpoints in `third_milestone.md`, including INSTRUCTOR-base instruction/text API compatibility. Replace infeasible models within their group to maintain the confirmed three-plus-three minimum. The proposed four-dimension/three-seed plan now totals 72 main training runs, plus three native MiniLM reference runs.

## AI plan handoff, 2026-10-19

- [x] Create [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) with the complete ordered plan, confirmed constraints, open choices, candidate checkpoints, artifacts, and verification criteria.
- [ ] Start with bounded JSONL loading and data contracts, then settle sampling before freezing the split. Native reduction versus PCA and published-reference versus local-baseline interpretation remain open.

## Manuscript draft, 2026-10-22

- [x] Add a short plain-English proposed-methodology-and-evaluation section to `6ac65ba1a33e4f01bca06b79/kovarm46.tex`. Personal `third_milestone.md` was left unchanged. Reduction method remains open; no experiments were run.
- [ ] Before final submission, reconcile the older Baselines paragraph, which still specifies INSTRUCTOR-XL, ID-only LightGCN, and a narrower dimension sweep, with the evolving experiment plan.

The new section passed basic brace and citation-key checks. PDF compilation was unavailable because neither `pdflatex` nor `latexmk` is installed in the current shell.

## Paired prompt conditions, 2026-10-26

- [x] Add instructed/uninstructed conditions to the AI model list for Qwen3-Embedding-0.6B, multilingual E5-large-instruct, and INSTRUCTOR-base, distinguishing documented use from ablation proposals. Personal milestone is unchanged.
- [ ] Pilot both modes, including explicit prompt-default removal and INSTRUCTOR empty-instruction handling. Record exact inputs and independent caches.
- [ ] Fix a shared training-only PCA fit policy for prompt pairs if the comparison is meant to hold the fitted projection constant.
- [ ] Rebudget the sweep for nine input conditions, provisionally 108 main runs at four dimensions and three seeds, plus three native MiniLM runs. Earlier six-condition estimates remain historical.

- [x] 2026-10-26: Add all three instruction candidates with and without task instructions to the LaTeX methodology, including separate CARec training for each input setting. User confirmed this comparison scope. The personal milestone is unchanged.

## Content-only baseline, 2026-10-26

- [x] Add a simple content-only comparison to the LaTeX methodology/evaluation and AI plan, using the same embedding conditions as CARec.
- [ ] Implement training-history mean profiles and cosine scoring, verify zero/empty profiles and tie handling, then evaluate with exactly the same masks and metrics as CARec.
- [ ] Report content-only results for each encoder, prompt mode, and dimension. The baseline has no recommender-training seed variation when its input vectors are fixed.
