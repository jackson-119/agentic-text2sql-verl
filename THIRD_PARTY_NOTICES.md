# Source provenance and third-party acknowledgments

This document records observed source provenance. It is not a complete
license audit or a claim of ownership over every migrated file.

## verl-derived importance sampling component

`src/agentic_text2sql/algorithms/importance_sampling.py` adapts the token-level
importance-weight calculation from `verl/trainer/ppo/rollout_corr_helper.py`.

- Upstream project: [verl](https://github.com/volcengine/verl).
- Copyright notice observed in the source:
  Copyright 2025 Bytedance Ltd. and/or its affiliates.
- Source license: Apache License, Version 2.0.
- The original copyright and license header is retained in the adapted module.
- Adaptation scope: token-level weights, upper truncation, and no batch
  normalization; input validation is stricter than the source.
- This is an adaptation of an existing implementation, not a claim of a new
  importance-sampling algorithm.

Observed historical working-tree source SHA-256:

`cc7cd16dde46e122fa267a2b7319910d2b1558fdd9724d3cc9eae248a2822f86`

Observed source LICENSE SHA-256:

`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`

The inspected training checkout reported Git HEAD
`e003163181731412595257a72ec173071efb125f`.
That revision does not identify all local modifications or untracked files.
Per-file hashes and migration manifests are therefore required in addition
to the Git revision.

No `NOTICE`, `NOTICE.md`, or `THIRD_PARTY_NOTICES.md` was found at the root
of that inspected source checkout. This observation is limited to those
locations; it is not a statement about every upstream version or dependency.

See [the IS migration manifest](experiments/manifests/importance_sampling_source.json)
and [the repository license](LICENSE).

## Historical experiment-workspace components

The following components were migrated from the project's historical
experiment workspace:

- Score-centering core and auxiliary log-probability computation.
- SQLite agent tools.
- SQL extraction and execution comparison.
- Tool-process metrics and correctness-gated reward composition.
- Assistant-only SFT masking.

The inspected SC, database-tool, tool-metric, and reward source files were
untracked in the historical training checkout. Their inspected headers did
not contain an upstream copyright notice.

Neither an untracked Git status nor the absence of a copyright header proves
authorship, ownership, or unrestricted reuse. These files should not be
described collectively as unmodified upstream verl code. Maintainer
confirmation of their authorship and any incorporated third-party material
remains a separate publication responsibility.

The component manifests under
[`experiments/manifests`](experiments/manifests) record source paths, hashes,
migration scope, and available validation evidence.

## Dependencies versus distributed source

The project uses external software, including verl, PyTorch, Transformers,
and vLLM in its historical training environment. Dependency use does not
mean that those projects' complete source trees are distributed here.

This document does not audit all transitive dependencies or grant rights
to datasets, model weights, or external artifacts. Consult their respective
licenses and terms separately.

## Reproducibility boundary

Source attribution and passing component tests do not establish end-to-end
training reproducibility.

See [reproducibility status](docs/reproducibility_status.md) for implemented
components, historical evidence, and remaining integration work.
