# Project TODO

Last updated: 2026-10-12

## Confirmed direction

- [x] Working topic: compare general-purpose and instruction-based item embeddings, including embedding size.
- [x] First dataset: Amazon Electronics.
- [x] Draft the related-work, motivation, novelty, and baseline milestone.
- [x] Draft the proposed methodology and evaluation in [`THIRD_MILESTONE.md`](../../THIRD_MILESTONE.md).
- [x] Restore the requested CARec encoder-comparison notebook scaffold at [`carec_encoder_comparison.ipynb`](carec_encoder_comparison.ipynb).

## Open decisions and next steps

- [ ] Confirm with the instructor that CARec's CIKM 2024 paper is a research-track paper.
- [ ] Confirm permission to reuse CARec code. Its visible GitHub root listing had no license file when checked on 2026-10-12. Do not vendor or modify it until reuse terms are clear. See [`RESEARCH.md`](RESEARCH.md#carec-code-and-experiment-structure).
- [ ] Decide whether to use CARec directly or implement a smaller compatible recommender if code reuse is unavailable.
- [ ] Record available RAM, disk space, deadline, and API restrictions. The M4 Pro machine is known; MPS compatibility and runtime still need a pilot.
- [ ] Pilot a small encoder subset and choose the final models based on memory and runtime.
- [ ] Verify CARec's exact data split and cold-item definition. Decide whether to rerun LightGCN on the same split.
- [ ] Run the first Amazon Electronics baseline, then compare selected encoders at 64, 128, 256, and 384 dimensions. Report 768 separately only for compatible models.
- [ ] Report warm/cold Recall@10 and NDCG@10, embedding storage, and generation/training/scoring time.
- [ ] Add a second dataset only if the first experiment is stable.
