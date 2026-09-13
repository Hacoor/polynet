# Agent Instructions

Before substantive work in this repository, read both `MEMORY.md` and `PolyNet_paper.md`.

Use relevant durable lessons from `MEMORY.md` to shape the research plan, implementation, benchmark design, and interpretation of results. Memory is guidance, not live evidence: verify current code, experiments, artifacts, and measurements fresh.

If current reproducible evidence conflicts with memory, current evidence wins. Update stale memory when the durable lesson has changed. Do not repeat a documented failed approach without new evidence explaining why the situation differs.

When verified work produces a reusable research or engineering lesson, add the generalized lesson to `MEMORY.md` in the same change when practical.

A substantive task is not fully investigated until the relevant project memory has been consulted and applied.

## Research rules

- Keep proposed, implemented, measured, and hypothetical claims explicitly separate.
- Do not present theoretical FLOP reduction as measured end-to-end speedup.
- Every approximate node must be evaluated for both local approximation error and downstream task quality.
- Preserve exact/standard fallbacks when approximation confidence or applicability is insufficient.
- Record enough benchmark metadata to reproduce model revision, task, hardware/software stack, precision, batch/sequence shape, baseline, and measurement procedure.
- Compare against strong relevant baselines rather than only an unoptimized implementation.
- Do not cherry-pick favorable batches or evaluation inputs; separate calibration/training data from evaluation data when overfitting is possible.
- Keep hardware-specific implementation details behind backend/runtime boundaries where practical.
- Treat `titishop-import/` as repository data material, not evidence of a live TitiShop production integration.
- Update tests, experiments, and documentation together when changing the meaning or contract of a PolyNet component.