# NI-PML project context

Last updated: 2026-09-29

## Assignment

The project must extend an existing personalised machine-learning method and present the results as a short scientific paper. The main focus is on recommender systems.

The base paper must be a 2024, 2025, or 2026 research-track full or short paper from RecSys, KDD, CIKM, WWW, or another A* machine-learning conference. The extension must be original and evaluated against the base paper.

Source: [`assignment.md`](../../assignment.md).

## Selected project topic

Compare general-purpose and instruction-based text embedding models as item representations, and study how the embedding size affects recommendation quality.

## Candidate base paper

The source paper and current candidate base paper is **CARec: Collaborative Alignment for Recommendation**, CIKM 2024. It is a useful starting point because it includes an encoder ablation with `instructor-xl`, evaluates warm and cold items, uses public recommendation datasets, and provides code and preprocessing instructions. However, its main contribution is collaborative alignment, not a controlled study of instruction-based embeddings. The project will extend that ablation.

CIKM is ranked **A**, not A*, in the current ICORE/CORE listing. The assignment explicitly names CIKM as an accepted venue, so the paper does not need CIKM to be A* to satisfy the listed-venue requirement. The paper is published in the CIKM 2024 proceedings; the exact research-track label should still be confirmed if the instructor requires it.

Sources: [ICORE/CORE CIKM ranking](https://portal.core.edu.au/conf-ranks/25/) · [CARec publication page](https://liangwei.yangxiaoma.org/publication/cikm2024_carec/).

The proposed extension is to add a controlled comparison of encoder families and embedding dimensions. The project should not be considered final until the paper's research-track eligibility, the fairness of the comparison, and the student's available hardware have been confirmed.

## Confirmed data direction

Start with Amazon Electronics from the CARec experiment. The dataset is public, contains item text, and is supported by the authors' preprocessing code. A second dataset can be added after the first pipeline is working.

## Resource constraints

- Use frozen, cached item embeddings.
- Do not train a large language model.
- Use a small recommender or the lightest reproducible CARec configuration.
- Compare equal target dimensions across encoders.
- Record runtime and embedding memory in addition to Recall and NDCG.

## Working rules

- Keep confirmed facts, proposals, and open questions separate.
- Verify a paper's year, venue, and track before treating it as the final base paper.
- Record sources for claims about papers, code, data, metrics, and compute.
- Update the AI notes after a meaningful decision or experiment. Use dates and avoid cosmetic edits.

Detailed paper, dataset, and extension notes are in [`RESEARCH.md`](RESEARCH.md). Active decisions and next steps are in [`PROJECT_TODO.md`](PROJECT_TODO.md).
