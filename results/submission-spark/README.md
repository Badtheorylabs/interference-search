# Spark submission results

These receipts were produced on `spark-d580` under `~/interference-search-submission-20261003` and copied into this repository. The compressed JSONL files in `raw/` contain one row per problem × method × seed × budget for the completed Countdown suites. Decompress with `gzip -dc` when loading them.

The completed suites are:

- `main.jsonl.gz`: 1,020,000 rows, 1,000 problems, 17 methods, 10 seeds, six state-work budgets.
- `width4.jsonl.gz`: the same suite with frontier width 4.
- `proposal-budget.jsonl.gz`: the same suite charged by proposed states.
- `six-number.jsonl.gz`: 68,000 rows on 100 six-number problems and four larger budgets.
- `judge-transfer.jsonl.gz`: 4,719 threshold rows from 470 attempted transfer/hard problems; exact graph failures are in the feasibility report.
- `judge-hard-six.jsonl.gz`: 1,100 threshold rows on 100 rare-solution six-number problems.

`reports/` contains the regenerated summaries, confidence intervals, factorial contrasts, budget curves, feasibility records, and preserved historical negative experiments. `checkpoints/` contains the five retrained judge checkpoints and their receipts.

The MBPP file is an in-progress Spark receipt: `mbpp-main.jsonl.gz` contains the rows completed when this snapshot was copied. The full run remains resumable on Spark and does not have a `complete.json` receipt yet. The `judge-seed-*` files likewise preserve partial multi-seed judge replay receipts while those jobs continue on Spark.

The expanded state-prediction caches are intentionally omitted because they are redundant multi-gigabyte intermediates. Their hashes and regeneration commands are in the Spark output configurations.
