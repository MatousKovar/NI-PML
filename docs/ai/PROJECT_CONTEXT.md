# NI-PML project context

Last updated: 2026-10-12

## Assignment

Extend a personalised machine-learning method from a 2024-2026 research-track paper and evaluate the extension in a short scientific paper. The assignment explicitly lists CIKM as an eligible venue. See [`assignment.md`](../../assignment.md).

## Working topic and paper

Compare general-purpose and instruction-based text embedding models as item representations, and study how embedding size affects recommendation quality.

The working source paper is [CARec: Collaborative Alignment for Recommendation](https://doi.org/10.1145/3627673.3679535), published at CIKM 2024. CARec provides a text-aware recommender and an encoder ablation, but it does not isolate instruction conditioning or embedding size. The project will extend that experiment. Confirm the paper's research-track label with the instructor before treating eligibility as final.

## Initial experiment

- Start with Amazon Electronics using one fixed CARec split.
- Keep the recommender, item text, split, and training settings fixed while changing the encoder and target dimension.
- Compare shared dimensions 64, 128, 256, and 384. Report 768 separately for compatible encoders.
- Fit the same PCA procedure on training-item embeddings only. Freeze and cache the text embeddings.
- Report Recall@10 and NDCG@10 on warm and cold items, plus storage and runtime.
- The model shortlist and detailed protocol are in [`THIRD_MILESTONE.md`](../../THIRD_MILESTONE.md). Research and source links are in [`RESEARCH.md`](RESEARCH.md).

These are working plans. The final encoder subset, exact cold-item protocol, and whether to use CARec code directly remain open.

## Compute and source-code status

The target machine is the user's MacBook Pro with an M4 Pro chip. Use PyTorch MPS where supported, not CUDA. No encoder or recommender pilot has been run; memory use and runtime still need measurement.

CARec's public repository is [`ChenMetanoia/CARec`](https://github.com/ChenMetanoia/CARec). The visible repository listing checked on 2026-10-12 had no license file, and no CARec source is in this project. Keep reuse permission unresolved. Do not copy or modify the source until its terms are confirmed. See [`RESEARCH.md`](RESEARCH.md#carec-code-and-experiment-structure).

## Working rules

- Follow [`AGENTS.md`](../../AGENTS.md) and keep confirmed facts separate from proposals and open questions.
- Do not train an LLM. Cache frozen item embeddings and keep experiments small enough for local MPS testing.
- Update the relevant `docs/ai/` note after meaningful project decisions or experiments.
