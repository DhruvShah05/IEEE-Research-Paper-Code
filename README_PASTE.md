# Drop-in fixes for A2 + A5 (+ B1, compile default, `make tune`)

Copy every file in this folder over the same path in your repo (overwrite).
Then run `make test` — all of TestReorderToCanonical and TestBaselines must pass.

## What changed and why

| File | Change |
|---|---|
| `data/loaders.py` | **A2** — `reorder_to_canonical` now uses the real crypto layout (price/vol alternate within each side): level i -> `[20+2i, 21+2i, 2i, 2i+1]`. Exposes `CRYPTO_TO_CANONICAL_PERM`, `reorder_from_canonical`, works on 2-D and 3-D arrays. Everything else in the file is unchanged. |
| `models/structured_transformer.py` | **A2** — `level`/`grouped` tokenisation now reads the canonical layout (`reshape(B, tokens, group)` on contiguous per-level blocks). `in_features` respects `data.feature_set` (raw40 vs full144). |
| `models/transformer.py` | `in_features` respects `data.feature_set` (temporal mode was wrong for raw40). |
| `models/baselines.py` | **A5** — `build_model(config)` factory; persistence uses realised `mid[t]-mid[t-H]` (no labels, no future); `predict == argmax(predict_proba)` for every baseline. |
| `main.py` | **A5** — new `model: baseline` branch (fits, passes `mid_test` to persistence, saves the same artefacts as every other model). Manifest `model` key is `baseline_<name>` so aggregate/significance/backtest don't lump them. Tree branch now **raises** on null params. |
| `configs/*_baseline_*.yaml` | 7 new configs: crypto × {majority_class, random, persistence, logistic_regression}, fi2010 × {majority_class, random, logistic_regression}. (No FI-2010 persistence: Z-score data has no mid-price.) |
| `configs/*deeplob/transformer/structured_transformer*.yaml` | `gpu.compile: false` (turn on with `compile_mode: default` only after a full run works). |
| `scripts/tune_all.py` (new) + `Makefile` | `make tune` runs Optuna once for XGBoost + RF on **both** markets and prints the `model_params:` block to paste into each YAML. `make run_baselines`, `make run_trees`, `make run_neural` added. `install_gpu` now installs torch>=2.7 / cu128 (RTX 50-series) and verifies a CUDA matmul. |
| `requirements.txt` | torch removed (so `pip install -r` can't downgrade your GPU build). |
| `tests/test_models.py` | Reorder tests rewritten against the correct layout (the old ones validated the bug), plus a tokenizer-layout test and A5 baseline tests. |

## Then, in order
```
make install_gpu && make gpu_info      # must show a CUDA matmul succeeding
make test
make prepare
make threshold
make tune TRIALS=30                    # paste printed params into the 4 tree YAMLs
make smoke
make run_baselines
make run_trees
```
Send me: `make test` output, prepare logs + `crypto_quality_report.json`, one `split_manifest.json`, the two threshold-sweep CSVs, the tune output, and `results/aggregated_metrics.csv` after `run_trees`.
