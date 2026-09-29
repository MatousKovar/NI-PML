# Project TODO

Last updated: 2026-09-29

## Confirmed direction

- [x] Select the working topic: general-purpose versus instruction-based item embeddings, with an embedding-size study.
- [ ] Confirm CARec as the base paper; its encoder comparison is relevant but is not a controlled instruction-tuning study.
- [ ] Confirm with the instructor whether the CIKM 2024 proceedings entry satisfies the required research-track classification.
- [x] Identify a public first dataset: Amazon Electronics.

## Open decisions

- [ ] Confirm available CPU/GPU, RAM, storage, deadline, and whether external model APIs are allowed.
- [ ] Confirm CARec's exact research-track classification with the course requirements or instructor.
- [ ] Decide whether the final study uses CARec directly or a smaller compatible recommender implementation.
- [ ] Choose the final encoder pair and target dimensions after checking model download and memory requirements.

## Planned experiments

- [ ] Reproduce an ID-based and a text-embedding baseline on Amazon Electronics.
- [ ] Compare one general-purpose encoder with `instructor-xl` using the same item text.
- [ ] Evaluate target dimensions 64, 128, and 256 using the same projection procedure; add 768 only for a pair of encoders with native 768-dimensional output.
- [ ] Report Recall@10 and NDCG@10 for warm and cold items.
- [ ] Record embedding storage, preprocessing time, training time, and inference/scoring time.
- [ ] Add a second public dataset only if the first experiment is stable.

## Possible final contribution

A controlled analysis of whether instruction tuning still helps when item embeddings are compressed to small dimensions. The result should include accuracy, cold-start behaviour, and resource cost.
