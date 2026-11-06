# CARec performance measurements

Measured 2026-11-06 on the user's M4 Pro Mac, PyTorch 2.6.0, eight CPU threads. All runs use the real Electronics training graph, native MiniLM vectors at dimension 384, batch size 2048, and fresh isolated models. No existing notebook outputs, checkpoints, or dataset files were changed.

## Finding

The upstream LightGCN message-passing implementation is the main bottleneck. Its forward gathers and weights a vector per directed edge. At 905,212 edges and 384 float32 components, one edge-feature tensor contains 1,390,405,632 bytes, about 1.30 GiB. Both graph layers repeat this work for every batch; backward adds more edge operations. The graph has 115,742 nodes including padding, so this is considerably more work than the small MLP suggests.

A cached normalized sparse CSR adjacency performs the same propagation with `torch.sparse.mm`. On this machine it is faster on CPU than upstream message passing on MPS. A hybrid with parameters/MLP on MPS and sparse propagation on CPU also improves throughput. The benchmark initially used isolated adapters. The CPU CSR adapter was subsequently integrated into the notebook at the user's explicit request; upstream source remains unchanged.

## Measurements

Times below are warmed median milliseconds per optimization step, including loss forward, backward, and Adam. Baseline and hybrid series have three to five measured repetitions; CPU sparse series have twenty. Timings synchronize MPS at stage boundaries and exclude setup, validation, and checkpoint writes.

| Configuration | Phase 1 ms | Phase 2 ms |
| --- | ---: | ---: |
| Original graph propagation, MPS | 1339.3 | 1315.6 |
| Original graph propagation, CPU, 8 threads | 1015.2 | not measured |
| Sparse CSR propagation, CPU, 8 threads | 452.3 | 511.3 |
| Sparse CSR on CPU, remaining model on MPS | 453.4 | 478.1 |

Reducing CPU threads from eight to one made sparse phase one slower, at 1348.5 ms per step. Disabling graph layers only as a diagnostic reduced original MPS phase one to 113.5 ms. That changes the model and is not a proposed training setting. It isolates the large cost introduced by graph propagation and its backward pass.

For original MPS phase one, median stage times were approximately 571 ms for graph forward, 15 ms for uniformity forward including device transfer, 739 ms for the complete backward, and 10 ms for Adam. Uniformity backward is part of the complete backward figure and was not separately profiled. Replacing just uniformity was therefore not an adequate fix for the dominant graph cost. User-run validation previously took about 1.5-1.7 seconds per epoch, against about 283 seconds for training.

A complete phase-one training pass over all 452,606 interactions, 221 shuffled batches, took **97.76 seconds** with CPU sparse propagation. This excludes setup and validation and uses a fresh benchmark model with one warmup step before the measured epoch. No model weights were saved. The user's earlier original MPS epochs took approximately 283-297 seconds, consistent with the reproduced MPS step times. This is about a threefold reduction in training-pass time on this dataset and dimension.

## Correctness and limits

CPU sparse propagation was compared with original message passing on identical parameters and batches in both phases. Evaluation mode disabled dropout for this comparison; timed runs used training mode. Phase-one loss absolute error was zero and relative user-table gradient error was 2.12e-7. Phase-two loss absolute error was 2.98e-8; the largest relative MLP parameter gradient error was 2.32e-7. The MPS/CPU hybrid also passed the same loss and gradient checks. Sparse coalescing sums repeated edge weights after the original degree normalization, preserving their contribution rather than changing the interaction protocol.

This establishes local loss/gradient agreement to floating-point tolerance, not identical long-run model trajectories or recommendation metrics. Larger embedding dimensions, peak memory, the full nine-condition sweep, and end-to-end evaluation after this optimization have not been benchmarked. The CPU-only adapter is the simpler candidate; the hybrid adds transfers for a modest phase-two timing difference. The user subsequently selected the CPU-only adapter for notebook integration.

## Reproduction

Raw samples, correctness checks, a setup snapshot, and the benchmark script are saved under the ignored directory `artifacts/performance/carec_20261106/`. The script loads only notebook setup code from that snapshot, constructs fresh models, and never loads or overwrites a user's trained checkpoint. On the current machine MPS requires running outside the execution sandbox.

```sh
.venv/bin/python artifacts/performance/carec_20261106/benchmark.py --device mps --phase 1 --max-step-ms 500
.venv/bin/python artifacts/performance/carec_20261106/benchmark.py --device cpu --phase 1 --variant sparse_cpu --verify --repeats 20 --max-step-ms 650
.venv/bin/python artifacts/performance/carec_20261106/benchmark.py --device cpu --phase 2 --variant sparse_cpu --verify --repeats 20 --max-step-ms 650
```

The first command reproduced the slow batch and failed its speed threshold. The sparse variants passed the 650 ms threshold and their numerical checks. Thresholds are local diagnostic targets, not portable performance guarantees.
