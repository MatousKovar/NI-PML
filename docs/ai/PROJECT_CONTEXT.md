# NI-PML project context

Last updated: 2026-10-26

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

- Follow [`AGENTS.md`](../../AGENTS.md) and keep confirmed facts separate from proposals and open questions.
- Do not train an LLM. Cache frozen item embeddings and keep experiments small enough for local MPS testing.
- Update the relevant `docs/ai/` note after meaningful project decisions or experiments.

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
