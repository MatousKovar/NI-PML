# Project TODO

Last updated: 2026-10-12

## Confirmed direction

- [x] Select the working topic: general-purpose versus instruction-based item embeddings, with an embedding-size study.
- [x] Draft the related-work, motivation, novelty, and baseline milestone in the manuscript; shorten the introduction.
- [x] Draft the proposed methodology and evaluation plan in [THIRD_MILESTONE.md](../../THIRD_MILESTONE.md); encoder and baseline choices remain open.
- [ ] Confirm CARec as the base paper; its encoder comparison is relevant but is not a controlled instruction-tuning study.
- [ ] Confirm with the instructor whether the CIKM 2024 proceedings entry satisfies the required research-track classification.
- [x] Identify a public first dataset: Amazon Electronics.

## Open decisions

- [ ] Confirm available CPU/GPU, RAM, storage, deadline, and whether external model APIs are allowed.
- [ ] Confirm CARec's exact research-track classification with the course requirements or instructor.
- [ ] Decide whether the final study uses CARec directly or a smaller compatible recommender implementation.
- [ ] Pilot the seven encoder candidates in [THIRD_MILESTONE.md](../../THIRD_MILESTONE.md), then choose the final subset after checking MPS compatibility, runtime, and memory.
- [ ] Confirm CARec's cold-item split and whether to rerun LightGCN; see [THIRD_MILESTONE.md](../../THIRD_MILESTONE.md).

## Planned experiments

- [ ] Reproduce an ID-based and a text-embedding baseline on Amazon Electronics.
- [ ] Compare the selected general-purpose and instruction-based encoders using the same item text and fixed CARec setup.
- [ ] Evaluate shared dimensions 64, 128, 256, and 384 using the same PCA procedure; report 768 separately only for encoders that support it.
- [ ] Report Recall@10 and NDCG@10 for warm and cold items.
- [ ] Record embedding storage, preprocessing time, training time, and inference/scoring time.
- [ ] Add a second public dataset only if the first experiment is stable.

## Possible final contribution

A controlled analysis of whether instruction tuning still helps when item embeddings are compressed to small dimensions. The result should include accuracy, cold-start behaviour, and resource cost.
