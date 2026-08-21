# Evaluation reports

`src.models.evaluate` writes reproducible evaluation artefacts here:

- `predictions_test.jsonl`: LoRA predictions (ignored by Git because it is bulky).
- `predictions_test_base.jsonl`: base-model predictions (ignored by Git).
- `eval_test.json`: LoRA aggregate metrics; commit this file after evaluation.
- `eval_test_base.json`: base-model aggregate metrics; commit this file after evaluation.
- `judge_test*.jsonl`: per-image LLM-as-judge details; review before publishing.

Reports are intentionally absent until the held-out test run has actually
completed. Do not copy training loss into the evaluation table: training loss,
CIDEr, CLIPScore and LLM-as-judge measure different things.
