# Gold-SQL-guided SFT data construction

The builder is available as `python -m agentic_text2sql.sft.builder`.
Run it with the repository's `src` directory on PYTHONPATH, or after installation.

Required arguments:

- `--raw-dir`: directory containing train_spider.json, tables.json and database/.
- `--model-path`: trusted local tokenizer/model directory.
- `--output-dir`: new or empty directory outside the input directories.

The builder loads a tokenizer, not model weights. It retains the historical
`trust_remote_code=True` setting, so only trusted local model directories should
be used. Optional datasets/transformers imports occur inside main().

## Construction semantics

Relevant tables are obtained from gold SQL's structured table-unit annotations.
The builder creates scripted demonstrations, not autonomous model rollouts.
Normal list-tables, schema and execution observations come from read-only SQLite
queries. Execution output is row-limited and may be truncated.

For selected training examples, a synthetic missing-table call and error
observation are inserted. That Assistant call has trainable=False; it remains
context but is excluded from SFT supervision. Validation examples do not receive
this augmentation. The augmentation is Bernoulli sampling, not an exact quota.

Databases are shuffled and partitioned before examples are sampled.
The same seeded RNG is used for database order, example order and augmentation,
preserving the original implementation's sequence of RNG calls.
Failed or overlong examples are skipped. Insufficient output counts fail the run.

The output contains train.parquet, validation.parquet, previews and manifest.json.
Rows include messages_json, tools_json and trajectory_id for the SFT Trainer.
Runtime manifests and previews may contain local paths or dataset content:
keep them outside the public repository unless separately reviewed.

## Parameters and provenance

The source defaults are preserved; they are not a reconstruction of the expanded
historical experiment's launch command. In particular, train-size defaults to
1000, validation-size to 100, and recovery-ratio to 0.15.
The historical expanded manifest recorded 5000/500 and a requested ratio of 0.1.

The source file and archived snapshot were byte-identical. This establishes
source correspondence, not a byte-for-byte reproduction of historical data.

## Validation boundary

Synthetic tests cover trajectory generation, recovery masking, SQLite read-only
behavior, table extraction, deterministic splitting and output guards.
The main-path tests use a toy tokenizer and a Parquet-writer test double.
Real tokenizer and real Parquet end-to-end validation remain pending.

This split only separates this builder's train/validation databases.
It does not establish disjointness from downstream Pilot200 or other evaluation
sets. Data licensing, evaluation overlap and input trust require separate checks.
