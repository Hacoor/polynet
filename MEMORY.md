# Project Memory

Status: **curated durable research and engineering memory**

Last curated: 2026-09-13

This file records durable lessons for agents and maintainers working on PolyNet. It is not a source of live experiment results, implementation status, benchmark numbers, hardware state, or deployment state. Those must always be checked from current code, artifacts, experiment logs, and measurements.

## Source hierarchy

When sources disagree, prefer:

1. Reproducible current experimental evidence for measured behavior.
2. Current repository code, tests, benchmark harnesses, and generated artifacts for implemented behavior.
3. `PolyNet_paper.md` for the research thesis, intended architecture, and proposed experiments.
4. `AGENTS.md` for agent working rules.
5. This file for durable lessons and failure patterns.

A proposal in the paper is not evidence that it has been implemented or validated. Never turn a theoretical claim into a measured result without an experiment that supports it.

## What this repository is

PolyNet explores graph-level re-representation of frozen neural networks for more efficient inference. The research direction goes beyond local operator fusion by replacing meaningful subgraphs with heterogeneous optimized nodes in a compiled DAG.

The paper describes several complementary ideas:

- `PackedPolyFFN`: approximate activation-heavy FFN blocks with structured low-degree polynomial evaluation.
- Clustered Query Attention: reduce attention work by grouping semantically similar queries before expensive query-key computation.
- Macro-Operator Distillation: learn a lightweight shortcut that approximates a deeper teacher subgraph.
- VQ-PolyNet: learn discrete representations and routing so inference can exploit codebooks, retrieval, and predictable sparse computation.
- A two-stage compiler/runtime concept: Python tracing/profiling/partitioning followed by hardware-oriented backend generation and scheduling.

These are research components, not interchangeable buzzwords. Each changes a different part of the accuracy/latency/cost tradeoff and must be evaluated independently before combinations are credited with gains.

## Approximation is allowed only with measured fidelity

PolyNet intentionally changes computation, so correctness cannot be assumed from graph shape alone.

For every approximate replacement:

- define the exact teacher/subgraph being replaced;
- define the calibration/training data used to fit the approximation;
- measure approximation error at the node output and task-level quality after substitution;
- test inputs outside the calibration distribution and observed activation range;
- preserve a safe fallback to the standard/exact node when approximation confidence is insufficient;
- never hide accuracy loss behind aggregate speedup numbers.

Polynomial degree, codebook size, clustering thresholds, routing policy, and low-rank/expert structure are accuracy-performance controls and must be reported as such.

## Benchmark discipline

The central claim of PolyNet is a better quality/latency tradeoff, so benchmarks must measure end-to-end outcomes rather than isolated theoretical FLOPs alone.

At minimum, preserve enough experiment metadata to reproduce:

- source model and exact revision/weights;
- task/dataset and evaluation split;
- hardware and software stack;
- numerical precision/quantization;
- batch size and sequence length;
- warmup and measurement procedure;
- baseline configuration;
- compiled PolyNet configuration;
- end-to-end latency/throughput and task quality.

Compare against strong relevant baselines such as eager execution, `torch.compile`, and hardware-specific inference engines when available. A microbenchmark win is not an end-to-end win until the scheduler, data movement, compilation overhead, and fallback paths are included.

## Keep proposed and verified claims separate

Use explicit language in code, docs, and papers:

- **proposed** for an idea not yet implemented;
- **implemented** when code exists but the claim is not yet experimentally established;
- **measured** only when a reproducible experiment supports the number;
- **hypothesis** when the expected benefit remains unverified.

Do not preserve obsolete benchmark numbers or one-off measurements in this memory. Store durable experimental lessons here and keep current results in versioned experiment artifacts or reports.

## Compiler and runtime boundaries

Graph partitioning must preserve dependency and tensor-shape semantics. A fused or distilled macro-node must expose a contract precise enough for the scheduler to substitute it safely.

Hardware-aware optimization should be isolated behind backend/runtime interfaces. Do not entangle the research definition of a PolyNet node with one CUDA/Triton/kernel implementation so tightly that alternative backends cannot be compared.

Routing and cache/retrieval nodes introduce stateful behavior. Their keys, invalidation behavior, collision/error handling, and fallback path must be explicit; retrieval must not silently return a semantically invalid approximation merely because a nearby code exists.

## Research integrity and reproducibility

Prefer a small reproducible experiment that falsifies an idea over an impressive but non-reproducible result. Record failed approaches when the failure teaches a reusable boundary, such as an activation range where a polynomial approximation becomes unstable or a clustering threshold that destroys task quality.

Do not cherry-pick only favorable inputs, batches, or tasks. Separate calibration/training data from evaluation data where the method could otherwise overfit the benchmark.

## Repository boundaries

The `titishop-import/` directory contains data snapshots/import material and is not evidence that PolyNet is part of the TitiShop production architecture. Do not infer a live commerce integration from the presence of those files. Treat any future integration as a separate explicit contract.

## Memory maintenance

Add a memory entry when an experiment, implementation, or failed approach produces a reusable lesson that should change how future agents work. Generalize the lesson and record the invariant or boundary it revealed.

Do not store transient benchmark numbers, current branches, temporary experiment status, machine-specific state, credentials, or claims that an unverified proposal is complete.