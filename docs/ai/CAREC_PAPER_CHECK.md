# CARec paper check

Date checked: 2026-10-15.

## Confirmed from the paper

CARec uses frozen text embeddings for items and randomly initialized trainable user embeddings. Item text combines title, category, and brand. A LightGCN-style aggregator propagates information through the interaction graph. [Paper, Sections 3.1–3.2](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf).

Training has two stages. Semantic alignment trains users toward frozen item representations, with user uniformity regularization. Collaborative refinement freezes users and trains an item MLP, with item uniformity regularization. Alignment pulls interacting user/item pairs together; uniformity spreads representations apart. The language model stays frozen. [Paper, Section 3.3](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf).

Warm scoring uses a user/item dot product with transformed item representations. Cold scoring bypasses the MLP and uses the semantic item representation. [Paper, Section 3.4](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf).

The paper describes 5-core Amazon filtering, random selection of 5% cold items, removal of their interactions, and an 80/10/10 warm train/validation/test split. It reports full-ranking Recall and NDCG at 10 and 50. Amazon text is truncated to 512 tokens. [Paper, Section 4.1](https://yangliangwei.github.io/publication/cikm2024_carec/CIKM2024_CARec.pdf).

## Code differences found during the companion audit

These observations concern commit `bbfc7f385e796840f6eb5c967dfc8714284958ce`, read without importing or modifying upstream source.

- The preprocessing includes a 0.35 interaction sample, repeated filtering after selecting cold items, and user filtering of cold interactions. The implemented split therefore requires more detail than the paper's 5% and 80/10/10 description. [Preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py).
- The cold runner disables the MLP and recomputes graph representations for both users and items. A reproduction should specify this behavior rather than assume that it reuses the warm-scoring user vectors. [Runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py) and [model directory](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce/model).

## Open checks before experiments

- Confirm the research-track label. The author page establishes CIKM 2024 publication but does not itself identify the track. [Author page](https://liangwei.yangxiaoma.org/publication/cikm2024_carec/).
- Verify cold ranking candidates, masks, and treatment of users with no cold positives by running a tiny evaluation example. The paper's full-ranking wording does not settle these implementation details.
- Check whether the paper's published dataset counts can be reproduced from the pinned preprocessing and seed.

## Source access limitation

The PDF URL returned 404 when opened directly on this date. Its indexed primary-source text exposed the cited sections through targeted web searches. No full PDF rendering or table-value reproduction was completed in this check. The findings above distinguish paper statements from the separate source-code audit.
