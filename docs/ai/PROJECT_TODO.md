# Project TODO

Last updated: 2026-10-19

## Confirmed direction

- [x] Working topic: compare general-purpose and instruction-based item embeddings, including embedding size.
- [x] Only dataset: Amazon Electronics, confirmed 2026-10-15.
- [x] Draft the related-work, motivation, novelty, and baseline milestone.
- [x] Draft the proposed methodology and evaluation in [`third_milestone.md`](../../third_milestone.md).
- [x] Restore the requested CARec encoder-comparison notebook scaffold at [`carec_encoder_comparison.ipynb`](carec_encoder_comparison.ipynb).

## Open decisions and next steps

For implementation details and completion criteria, read [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). Keep the user's `third_milestone.md` unchanged unless they request changes to it.

- [x] 2026-10-15: Prepare a source-backed CARec walkthrough with repository map, training stages, split analysis, and illustrative snippets. See [`CAREC_OVERVIEW_CS.md`](CAREC_OVERVIEW_CS.md).
- [ ] Choose whether to preserve upstream title + brand + category text or deliberately use titles only. Current title-only milestone wording does not describe upstream preprocessing.
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
