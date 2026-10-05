# NI-PML handoff

Date: 2026-10-02

## Project decision

The selected project topic is exactly:

> Compare general-purpose and instruction-based text embedding models as item representations, and study how the embedding size affects recommendation quality.

The project concerns recommender systems. The intended experiment is to keep the recommender, item text, data split, and evaluation fixed while changing the text encoder and final embedding size.

## Source paper and its role

The current source/base-paper candidate is **CARec: Collaborative Alignment for Recommendation**, CIKM 2024:

- DOI: https://doi.org/10.1145/3627673.3679535
- Author page: https://liangwei.yangxiaoma.org/publication/cikm2024_carec/
- Code: https://github.com/ChenMetanoia/CARec

Important interpretation: CARec is not primarily a paper about general-purpose versus instruction-based embeddings. Its main contribution is aligning text-based item representations with collaborative-filtering representations, especially for cold-start items. It contains a limited encoder ablation with `instructor-xl`, MiniLM, MPNet, BGE, and BERT. This ablation motivates the project but does not isolate instruction tuning causally.

CIKM is ranked A, not A*, in the ICORE/CORE listing: https://portal.core.edu.au/conf-ranks/25/. The course assignment explicitly lists CIKM as an accepted venue, so A* status is not required for this listed venue. The exact research-track label for this paper still needs confirmation from the instructor.

## Data and compute direction

Start with the public Amazon Electronics dataset used by CARec. The authors provide download and preprocessing instructions. A second dataset should be added only after the first experiment works.

Use frozen, cached embeddings. Do not train an LLM. Report Recall@10 and NDCG@10 separately for warm and cold items, together with embedding memory and runtime.

Suggested dimensions are 64, 128, and 256. Add 768 only when both chosen encoders naturally produce 768-dimensional outputs. Do not compare native 384-dimensional and 768-dimensional vectors and attribute the difference to encoder type.

## Supporting literature

See `/Users/matouskovar/FIT/NI-PML/docs/ai/RESEARCH.md` for the researched details. The main supporting papers are:

- **Let It Go? Not Quite**, RecSys 2025: content initialization, PCA, frozen embeddings, and a small trainable correction; public Amazon-M2 and Beauty data.
- **MARec**, RecSys 2024: simple metadata/collaborative alignment for cold-start recommendation; multiple public datasets.
- **INSTRUCTOR**: background on instruction-based embedding generation.
- **E5**: background on general-purpose text embeddings.

## Project artifacts

The working notes are already updated:

- `/Users/matouskovar/FIT/NI-PML/topic.md` contains the exact title, research question, source paper, and initial experiment.
- `/Users/matouskovar/FIT/NI-PML/docs/ai/PROJECT_CONTEXT.md` contains assignment constraints, venue-ranking caveat, data direction, and resource constraints.
- `/Users/matouskovar/FIT/NI-PML/docs/ai/PROJECT_TODO.md` contains open decisions and planned experiments.
- `/Users/matouskovar/FIT/NI-PML/docs/ai/RESEARCH.md` contains paper, dataset, venue, and extension research.
- `/Users/matouskovar/FIT/NI-PML/assignment.md` is the original course brief and was intentionally left unchanged.

No implementation or experiment has started yet.

## Next actions

1. Confirm CARec's exact research-track classification with the instructor.
2. Record available CPU/GPU, RAM, storage, deadline, and API restrictions.
3. Choose the encoder pair and check whether the models fit the available machine.
4. Reproduce the simplest ID-based and text-based baseline on Amazon Electronics.
5. Run the equal-dimension encoder comparison and warm/cold evaluation.

## Suggested skills

- `research` for verifying paper eligibility, datasets, model details, and reproducibility.
- `unslop` for cleaning human-facing research notes and final prose.
