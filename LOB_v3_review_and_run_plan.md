# LOB repo v3 — Review, Remaining Fixes, and Run Order

Verdict: the big bugs from round 1 are fixed correctly (labels from raw mid-price, FI-2010 remap, explicit CF file names, timestamp slicing, tuning split out, probabilities/timestamps/mid-prices saved, embargo gap, early stopping/AdamW/scheduler, baselines, backtester, significance runner, tests). But there are **five blockers that will either crash or silently produce wrong numbers**, plus a handful of smaller things. Fix Part A before running anything expensive.

---

## Part A — Blockers (fix before any GPU run)

### A1. DeepLOB and the temporal Transformer are not wired in — DeepLOB runs will crash
- `WindowedDataset` exists in `data/loaders.py` but is **never used**. `main.py` still builds `TensorDataset` of single `(B, F)` snapshots for every neural model.
- `models/deeplob.py::DeepLOB.forward` expects `(B, 1, T, 40)`. With `(B, 40)` input `Conv2d` raises a shape error. Both `crypto_deeplob.yaml` and `fi2010_deeplob.yaml` set `variant: windowed`, so `make run` will fail on all 10 DeepLOB runs (and `failed_runs.json` will record it, but you'd lose the time).
- `StandardTransformer(token_mode='temporal')` has the same problem — no config or code path feeds it `(B, T, F)`.
- **Fix:** in `main.py`, when `model_params.variant == 'windowed'` or `token_mode == 'temporal'`, build train/val/test loaders from `WindowedDataset(X, y, T)` instead of `TensorDataset`; add `unsqueeze(1)` (channel dim) inside `DeepLOB.forward` so it accepts `(B, T, 40)`. Slice `ts_test / mid_test / ret_test / y_test` by `[T-1:]` so saved arrays stay aligned with predictions. The `--smoke-test` truncation (`X_train[:100]`) must keep ≥ T+1 rows or windowed models get zero samples.
- Add a test that builds a `WindowedDataset` + `DataLoader` and runs one DeepLOB forward pass end-to-end (the current `test_models.py` only tests the module in isolation with a hand-shaped tensor).

### A2. Feature column layout differs between markets — DeepLOB and `token_mode='level'` are wrong on one of them
- Crypto 40-col layout: `[bid_p1, bid_v1, …, bid_v10, ask_p1, ask_v1, …, ask_v10]` (bid block, then ask block).
- FI-2010 raw-40 layout: `[ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, …]` (ask/bid **interleaved per level**).
- DeepLOB's second `(1,2)` stride-2 conv is designed to pair ask/bid at the same level — correct for FI-2010, but on crypto it pairs bid level 1 with bid level 2. `StructuredTransformer` `'level'` mode does the opposite: it assumes the crypto layout (`bid_p = cols 0:20:2`) and is therefore wrong on FI-2010.
- **Fix:** add a `reorder_to_canonical(X, market)` step in the loaders that puts both markets into one documented layout (recommend FI-2010's interleaved layout since DeepLOB is defined on it), and have `'level'`/`'grouped'` tokenization index that canonical layout. Add a test that checks the reorder on a synthetic frame.

### A3. XGBoost / RF configs still have `n_estimators: null` → tuning still runs inside every seeded run
- `train_tree_model` still calls `tune_tree_model` per seed when params are null, so fix 0.5 is only half-done. `make tune_xgboost` / `make tune_rf` exist but are **crypto-only** and their output is never written into the YAMLs.
- **Fix:** run tuning once per (model, market) — 4 studies — then hard-code every winning param into the 4 tree YAMLs and delete the `null` defaults. Make `train_tree_model` **raise** if `n_estimators` is None instead of tuning silently. Add `make tune` that runs all four and prints a YAML block to paste.
- Tuning objective double-weights RF: the Optuna objective passes `sample_weight='balanced'` to every model, and RF *also* has `class_weight='balanced'` in the constructor. Final RF training uses only `class_weight`. Make the objective match the final training path exactly.

### A4. Backtest has look-ahead bias with latency > 0, and gross return is mis-computed
- `experiments/backtest.py::_simulate_strategy`: `new_signal = signals[min(t + latency, n-1)]` acts at time `t` on a signal that will only exist `latency` observations **in the future**. That makes the "latency" sweep improve results instead of degrading them.
- **Fix:** act at `t` on `signals[t - latency]` (no trade for `t < latency`).
- Each trade pays cost twice (entry line and inside `net_ret_bps` at exit), but `gross_return_bps` adds back only `n_trades × cost`. Should add back `2 × n_trades × cost` (or track gross separately).
- Entry uses `mid[t+1]` plus a flat `slippage_bps`; the bid/ask spread is never charged. Either enter at `best_ask`/`best_bid` (save `test_best_bid.npy`/`test_best_ask.npy` from the loader — you already have those columns) or add half-spread from the saved data to the cost. Reviewer 2 will look for this.
- Sharpe is computed on per-250 ms PnL (mostly zeros) and annualized with √(31.5M/0.25) ≈ 11,000 — meaningless. Aggregate PnL to fixed buckets (per minute or per trade) before computing Sharpe/Sortino, and state the bucket in the paper.
- Positions are never force-closed at the label horizon H, so "hold for H" in the spec isn't implemented. Fine, but document the actual rule (hold until signal flips).

### A5. `PersistenceBaseline` leaks future information
- It predicts `y_prev` = the **true label** of the previous observation. At H=40 that label already encodes the mid-price 39 observations ahead of "now". Any number it produces on crypto is contaminated.
- **Fix:** persistence must use information available at `t`: sign of the realized mid-price change over the last H observations (`mid[t] − mid[t−H]`, thresholded the same way). Remove the `y_prev` argument entirely.
- Also: **baselines are not wired into the pipeline at all** — no `configs/*_majority.yaml` etc., and `train_tree.py::MODEL_MODULE_MAP` has no entry for them, so `make run` never produces baseline rows. Add one config per (market × baseline) and a `models/baselines.py::build_model(config)` dispatch, or a `model: baseline` + `model_params.name` convention.

---

## Part B — Should fix (won't crash, will bite in review or on the GPU)

1. **RTX 5080 / PyTorch version.** `requirements.txt` says "PyTorch 2.5 is the first stable release with RTX 5080 support" and the Makefile installs `cu126` wheels. Blackwell consumer GPUs (sm_120) need **PyTorch ≥ 2.7 with the cu128 wheels**; 2.5.1/cu126 gives "CUDA capability sm_120 is not compatible" and silently falls back to CPU (or errors). Change `install_gpu` to `--index-url https://download.pytorch.org/whl/cu128` with `torch>=2.7`, and pin what you actually end up with. Also `pip install -r requirements.txt` after the GPU install will re-pin `torch==2.5.1` (CPU) on top — move torch out of requirements.txt or put it on a separate `requirements-gpu.txt`.
2. **`torch.compile(mode='reduce-overhead')`** uses CUDA graphs and recompiles/breaks on the smaller last batch and on `model.eval()` switches. Start with `compile: false`; enable `mode: default` only after a full run works. Also `DataParallel` + `compile` + `deepcopy(state_dict)` is a known source of `_orig_mod.` key mismatches — you have one GPU, delete the `DataParallel` branch.
3. `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler('cuda')` (deprecated in 2.5+, removed later).
4. **`split_manifest.json` is never written** — `ds.save_split_manifest(run_dir)` is defined but not called in `main.py`. Call it.
5. `smoothed_mean` labeling is an O(N·H) Python list comprehension — on 1M rows this takes many minutes. Replace with `mid.rolling(H).mean().shift(-H)`.
6. `compute_microstructure_features` leaves NaN in `ret_5/10/40` for the first rows — fill or drop before scaling; `StandardScaler` will propagate NaN.
7. `RandomBaseline.predict` and `predict_proba` draw from the RNG separately, so saved argmax and probabilities disagree. Derive `predict` from `argmax(predict_proba)`.
8. `LogisticRegression(multi_class='multinomial')` is deprecated in sklearn ≥ 1.5; fine with the 1.4.1 pin, but if you bump sklearn it warns/breaks.
9. `results/aggregated_metrics.csv` currently checked in is from the **smoke run** (50-row test sets, `± 0.0000`, structured_transformer "87.6 %" accuracy). Delete it before anyone sees it; the real one gets regenerated.
10. `multi_asset.py` lists ETH/SOL parquet paths that don't exist and there is no `scripts/collect_lob.py` — the multi-asset study cannot run until you have data. Bybit/OKX aren't in the list at all. See Part C step 0.
11. `experiments/ablation_structured_transformer.py` header says 5 seeds; the OFAT grid is ~14 combos × 2 markets × 5 seeds = 140 neural runs. Fine on a 5080 but budget ~1–2 days; consider 3 seeds for the depth/width axis.
12. FI-2010 docstring in `loaders.py` says test files are "days 7, 8, 9" — they are days 8, 9, 10. Cosmetic, but it'll end up in the paper.
13. `build.md` and `LOB_code_change_list.md` are in the repo. Remove both before the code link goes to reviewers (they describe bugs and the prior rejection).

---

## Part C — Run order (after Part A)

### Step 0 — start now, in parallel (slowest)
- Collect multi-asset LOB data: ETHUSDT + SOLUSDT on Binance, and BTCUSDT on one second exchange (Bybit or OKX), 10-level depth snapshots at 250 ms, ≥ 3 days each, ideally two separate windows with different volatility. Write `scripts/collect_lob.py` (WebSocket depth stream → parquet with the same 42-column schema). Log exact UTC start/end.

### Step 1 — verification (cheap, CPU is fine). Send me all of this.
```
make test                        # every test must pass; send full output
make prepare                     # send logs + data/processed/crypto_quality_report.json
python -c "...CryptoDataset()..." # send split_manifest.json (after fix B4)
make threshold                   # send results/threshold_sweep_balance.csv + _f1.csv
```
Plus the label eyeball check: dump ~300 consecutive crypto rows of `(timestamp, mid_price, return_40, label)` to CSV and send it. And the class distribution table for both markets (train/val/test).

### Step 2 — smoke + first real numbers (hours)
```
make smoke                       # all configs, seed 0, tiny data — confirms nothing crashes (incl. DeepLOB after A1)
make tune                        # 4 tuning studies → paste params into YAMLs (A3)
make run_models MODELS="majority_class persistence logistic_regression random xgboost random_forest"
```
Send `results/aggregated_metrics.csv`. Decision point: if crypto XGBoost/RF are still at chance vs the baselines with correct labels, the paper's framing becomes "rigorous negative result" and we tune the horizon/threshold next rather than sinking GPU time into neural runs.

### Step 3 — neural headline runs (GPU, ~1 day)
```
make run_models MODELS="deeplob transformer structured_transformer"
```
Sanity target: full DeepLOB on FI-2010 k=10 should land in the high-70s to low-80s accuracy. If it's still ~60 %, the windowing/layout (A1/A2) is still wrong — stop and send me `training_history.json` + `run_manifest.json`.

### Step 4 — sweeps (GPU, ~1 day)
```
make horizon_sweep
make feature_set
make imbalance_ablation
```

### Step 5 — Transformer ablation (GPU, 1–2 days)
```
make ablation
```

### Step 6 — multi-asset (once Step 0 data exists)
```
make multi_asset
```

### Step 7 — analysis (CPU, minutes)
```
make backtest
make microstructure
make significance
make plots
make aggregate
```

### What to send me after each step
- Step 1: everything listed there.
- Steps 2–3: `results/aggregated_metrics.csv`, `results/failed_runs.json`, and for one crypto run and one FI-2010 run the full `seed_0/` directory minus `model.pt` and the `.npy` files.
- Steps 4–6: the per-experiment aggregated CSVs.
- Step 7: `results/backtest_*.csv`, `results/chance_tests.json`, significance tables, and the figures folder.

And tell me the GPU/RAM/PyTorch version you actually end up with after `make install_gpu` (send `make gpu_info` output).
