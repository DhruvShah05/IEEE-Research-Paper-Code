# LOB_Prediction_Comparison — Code Change List for Resubmission

Read every file in the repo. Below is everything that needs to change, ordered by severity. Section 0 is not optional — it changes the paper's numbers.

---

## 0. CRITICAL BUGS (fix before anything else — these invalidate current results)

### 0.1 Crypto labels are computed from the *relative-price* columns, not raw prices → labels are garbage
- **Where:** `main.py` (crypto branch), `data/loaders.py` (`CryptoDataset`), `experiments/threshold_sweep.py`.
- **What happens:** `to_relative_price(df)` overwrites columns `'2'` (best bid) and `'22'` (best ask) with `bid − mid` and `ask − mid`. Then `apply_horizon_labeling(df, ...)` recomputes `mid = (df['2'] + df['22']) / 2`, which is now `(bid − mid + ask − mid)/2 = 0` (or ±1 floating-point ulp). Future return = `(0 − 0)/0` → NaN (falls into "Stationary") or `tiny/tiny` → random huge values (random Up/Down). The labels have essentially no relationship to price movement.
- **Why this matters:** This fully explains the paper's crypto results — every model at ~34.4% balanced accuracy (chance = 33.3%) and MCC ≈ 0. Reviewer 2's whole complaint is downstream of this bug. The threshold sweep that "justified" H=40 / ±1bp was also run on garbage labels, so that justification is void.
- **Fix:** Compute mid-price and labels from the **raw** price columns *before* calling `to_relative_price`, or make `apply_horizon_labeling` take an explicit `mid_price` series as an argument. Make `to_relative_price` return a new frame and never mutate the columns the labeler reads. Add an assertion that `returns` contains no NaN/inf and that `mid_price.min() > 0`.
- **Consequence:** Every crypto number in the paper must be re-run. Expect materially different (and hopefully above-chance) results.

### 0.2 FI-2010 label semantics are inverted relative to crypto
- **Where:** `main.py` comment "1 (down), 2 (stationary), 3 (up)", `scripts/prepare_fi2010.py` docstring, `data/loaders.py`.
- **What happens:** In the official FI-2010 release, label **1 = Up, 2 = Stationary, 3 = Down** (Ntakaris et al. 2018; the DeepLOB repo README states the same). Your code subtracts 1 and then `eval/metrics.py` reports index 0 as `f1_down` and index 2 as `f1_up`. For FI-2010 those columns are swapped; for crypto (`label_by_threshold`: 0=Down, 2=Up) they are correct.
- **Why this matters:** Per-class F1 and confusion matrices for FI-2010 are mislabeled; any backtest built on FI-2010 predictions would trade in the wrong direction.
- **Fix:** Add an explicit label-remap for FI-2010 so both markets use the same convention (0=Down, 1=Stationary, 2=Up). Add a unit test that loads a small FI-2010 slice, recomputes direction from the raw mid-price (see 1.3) and asserts agreement with the remapped label.

### 0.3 The "Improved Transformer" in the paper is not what the code trains
- **Where:** `models/structured_transformer.py`, `models/transformer.py`, paper §IV-B.
- **What happens:** The paper says the improved model has "a deeper encoder … greater model capacity." In code both `StandardTransformer` and `StructuredTransformer` use `d_model=64, nhead=4, num_layers=2`. The only differences are grouped tokens and attention pooling. The claim of extra depth/capacity is false, and Reviewer 1's "cannot attribute lower performance to architecture" is partly caused by describing an architecture that doesn't exist.
- **Fix:** Either (a) rewrite the paper description to match the code (grouped level-pair tokens + attention pooling, same depth), or (b) actually expose depth/width in config and run the depth ablation (see 2.4). Also rename it — `build.md §8.8` in your own repo says not to call it "Improved" unless ablation shows it wins. Use "Structured Transformer" (or "LOB-Structured Transformer") in the paper.

### 0.4 FI-2010 loader may concatenate overlapping files
- **Where:** `scripts/prepare_fi2010.py::load_fi2010_folder` — `glob('*.txt')` then `hstack` everything.
- **What happens:** The official FI-2010 release ships `Train_Dst_NoAuction_ZScore_CF_1.txt … CF_9.txt` (CF_k = first k days) and `Test_Dst_…_CF_1 … CF_9.txt`. If more than the four standard files (`Train CF_7`, `Test CF_7/8/9`) are present, days get duplicated and the train/test boundary is broken (leakage). The paper's stated 354,825 observations does not match either the standard train (254,750) or test (139,587) counts, so something is off in what's being loaded or reported.
- **Fix:** Hard-code the exact filenames (Train `CF_7`; Test `CF_7`, `CF_8`, `CF_9`) instead of globbing. Log and save the exact row counts for train/val/test and report them in the paper. Add an assertion on expected row counts.

### 0.5 Hyperparameter tuning is re-run per seed → "seed variance" is actually hyperparameter variance
- **Where:** `train/train_tree.py` — Optuna `RandomSampler(seed=config['seed'])`, `n_trials=5`, called inside every seeded run.
- **What happens:** Each seed gets different tuned `n_estimators`/`max_depth`, so XGBoost's ±3.09% accuracy std on FI-2010 reflects different models, not training noise. That's why XGBoost (a nearly deterministic tree model) shows the *largest* variance in the paper. Reviewers will read it as instability or a bug.
- **Fix:** Tune once (single seed, more trials — 30–50), freeze the winning params into the YAML (`n_estimators: 200`, etc.), and run the 5 seeds with fixed params. Log the tuning study separately under `results/tuning/`. Report search space, trials, and chosen params in the paper.

### 0.6 Crypto day-window slicing uses row counts, not timestamps
- **Where:** `main.py` and `data/loaders.py` — `rows_per_day = 345_600`.
- **What happens:** Assumes perfectly uniform 250 ms sampling with no gaps. Any missing snapshots shift the window; the "three-day window" in the paper is then not three calendar days.
- **Fix:** Slice on column `'0'` (UNIX ms) or `'1'` (datetime): `start_ts <= t < end_ts`. Log and save the actual start/end timestamps and row count; report them in the paper.

---

## 1. DATA PIPELINE (`data/`, `scripts/`)

### 1.1 `data/features.py`
- `to_relative_price`: rename the operation honestly. Currently it is `price − mid` (absolute $ difference), not a relative/fractional representation as the paper claims. Add a config switch: `absolute_diff` | `fractional` (`(price − mid)/mid`) | `tick_units` (`(price − mid)/tick_size`). Default to fractional or tick units so the representation is transferable across assets (needed for multi-asset runs).
- Add optional volume transform (`log1p`) selectable in config — raw volumes are heavy-tailed; document the choice.
- Add a function that builds **derived microstructure features** (spread, order-flow imbalance at levels 1/5/10, cumulative depth imbalance, micro-price, recent mid-price returns over 5/10/40 events). This is needed for (a) the "why does crypto fail" analysis and (b) a fair comparison to FI-2010's engineered features.
- Make `TrainOnlyScaler` also record and save `mean_`/`scale_` to the run directory (traceability).

### 1.2 `data/labeling.py`
- Fix 0.1: `apply_horizon_labeling(df, horizon, threshold, mid_price=None)` — take an explicit mid-price series computed from raw columns; never derive it from possibly-transformed feature columns.
- Add a second labeling scheme: **FI-2010-style smoothed labeling** (compare mean of the next k mid-prices to the current mid, threshold α). Right now FI-2010 uses smoothed labels with α=0.002 while crypto uses point-to-point labels with 1 bp. That is a labeling confound in the cross-market claim; run crypto with both schemes and say so in the paper.
- Add **adaptive / volatility-scaled threshold** option (e.g., threshold = c × rolling std of returns, or percentile-based so classes are roughly balanced). Reviewer 2 and the "future work" section both promise this.
- Return the raw `returns` array alongside labels (needed for the backtest and for label-distribution plots).
- Add a function that emits the class distribution for a given (horizon, threshold) and saves it — this table must appear in the paper.

### 1.3 `scripts/prepare_fi2010.py`
- Fix 0.4 (explicit filenames, assert counts).
- Also download and prepare the **`NoAuction_DecPre`** (decimal-precision) variant alongside Z-score. Z-score data cannot recover the mid-price; DecPre can (prices are scaled by a constant, so returns are recoverable). You need the mid-price series to (a) verify labels, (b) run an FI-2010 backtest, (c) build the raw-40-feature FI-2010 variant.
- Save a `raw40` feature file: FI-2010 columns 0–39 are the raw 10-level bid/ask price/volume block. Add a `feature_set: full144 | raw40` config option so FI-2010 can be run on the same 40-dim input as crypto (removes the feature-set confound — both markets should be compared on identical inputs at least once).
- Fix the README/docstring attribution: FI-2010 is Ntakaris et al. (2018); Kercheval & Zhang (2015) is the feature taxonomy.

### 1.4 `scripts/prepare_crypto.py`
- Reconcile the data source: README/script cite Kaggle `martinsn/high-frequency-lob-btcusdt-binance`; the paper cites S. Raz `siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data`. Pick one and make paper, README, and script agree. State the exact date range used.
- Verify the `huggingface_hub` calls (`download_bucket_files`, `sync_bucket` with `hf://buckets/...`) actually exist in the pinned version; if they don't, replace with `hf_hub_download` / `snapshot_download` or drop the auto-download. A reviewer running `prepare_crypto.py` and hitting an `ImportError` is a bad first impression.
- Add a **data-quality report** step: gaps in timestamps, duplicate timestamps, crossed books (bid ≥ ask), zero volumes, non-monotone levels. Save as `data/processed/crypto_quality_report.json` and mention in the paper.
- Generalize to `prepare_crypto.py --symbol BTCUSDT --exchange binance --input path` so the same script prepares ETH/SOL and other exchanges (see 5.4).

### 1.5 `data/loaders.py`
- Currently dead code — `main.py` re-implements everything inline. Make `main.py` call `FI2010Dataset` / `CryptoDataset` and delete the duplicate logic. Two copies of split logic will drift.
- Add a **`WindowedDataset`** (PyTorch `Dataset`) that returns `(T, F)` sequences of the last T snapshots for each sample, with T configurable (default 100). Required for real DeepLOB (2.1) and for giving the Transformers temporal context. Must not cross split boundaries (first T−1 samples of each split are dropped or padded, documented).
- Add a `MultiAssetCryptoDataset` (or a loop in `run_all.py`) that runs the identical pipeline per `(exchange, symbol, window)`.
- Add a **purge/embargo gap** between train/val/test equal to the label horizon so that a training sample's label window does not overlap the validation period (currently the last 40 training rows have labels that look into the validation set). Small effect, but reviewers in finance check for it.
- Save per-split: start/end timestamps, row counts, class distribution → `split_manifest.json`.

---

## 2. MODELS (`models/`)

### 2.1 `models/deeplob.py` — replace with a real DeepLOB
- The current class applies `Conv1d` **across the feature axis of a single snapshot** and takes "the last time step" of the LSTM, which is the last *feature position*. It is not DeepLOB in any sense and produces 60% on FI-2010 vs ~84% (k=10) in the original paper — a reviewer will read this as a broken reproduction.
- Implement the Zhang et al. (2019) architecture: input `(B, 1, T=100, 40)`; conv blocks with `(1,2)` stride-2 kernels across price/volume pairs → `(1,2)` across bid/ask → `(1,10)` across levels; Inception module (1×1, 3×1, 5×1, max-pool branches); LSTM(64); linear head. Keep the old single-snapshot model under a clearly separate name (`deeplob_snapshot`) only if you want it as an extra row.
- Requires the `WindowedDataset` from 1.5 and `raw40` FI-2010 features from 1.3 (DeepLOB uses the 40 raw features, not 144).
- Log parameter count.

### 2.2 `models/transformer.py`
- Expose `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout` via `config['model_params']` — currently hard-coded, which makes the requested ablations impossible from config.
- Add a `temporal` mode that consumes `(T, F)` windows (tokens = time steps, features projected to `d_model`) so the Transformer gets the same information as DeepLOB. The current "each scalar feature is a token" design gives the model no temporal context at all.
- Log parameter count.

### 2.3 `models/structured_transformer.py`
- Same config exposure as 2.2.
- Grouped tokens on FI-2010's 144 features pair *arbitrary adjacent columns* (only the first 40 are price/vol pairs). Either restrict grouping to the raw-40 block or define groups from the FI-2010 feature taxonomy. Right now the FI-2010 result for this model is meaningless as a "level-structured" model.
- Add `token_mode: level` for crypto = one token per LOB level containing `(bid_p, bid_v, ask_p, ask_v)` — that is what "grouped by LOB level" in the paper actually implies; the current `group_size=2` is price/volume pairs, not levels.
- Add `depth`/`width` parameters so the ablation can produce a **parameter-matched** standard Transformer (same param count, no structure) — this is the control Reviewer 1 asked for.

### 2.4 `models/xgboost_model.py`, `models/random_forest.py`
- Expose and tune `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda` for XGBoost; `min_samples_leaf`, `max_features` for RF. Depth + n_estimators alone is not a credible search.
- Add `early_stopping_rounds` on the validation split for XGBoost.
- Add a `feature_importance()` export (gain-based + optional SHAP) saved per run — needed for the interpretability/"why" section.

### 2.5 New: `models/baselines.py`
- `MajorityClassBaseline`, `PersistenceBaseline` (predict last observed move direction), `LogisticRegressionBaseline`, `RandomBaseline` (seeded). These must appear as rows in every table; without them the paper cannot claim any model beats chance.

---

## 3. TRAINING (`train/`)

### 3.1 `train/train_neural.py`
- Add **early stopping** with patience on validation Macro-F1 (currently trains all 15 epochs and keeps the best — fine, but with 15 epochs the Transformers may be under-trained; make `max_epochs: 50, patience: 5`).
- Add `weight_decay` (AdamW), a learning-rate scheduler (cosine or ReduceLROnPlateau), and gradient clipping — all configurable, all logged. Needed both for fair comparison and for the regularization ablation.
- Save **test probabilities** (`test_probs.npy`, shape `(N, 3)`), not just argmax. Required for the confidence-thresholded backtest, calibration plots, and probability-based significance tests.
- Save the model checkpoint (`model.pt`) and `param_count` into `run_manifest.json`.
- Record wall-clock training time and per-batch inference latency (ms) → `run_manifest.json`. Latency matters for any HFT plausibility argument.
- The `focal_loss` path ignores class weights (`alpha=None`); either pass the computed weights as `alpha` or document that focal is un-weighted.
- Add "no mitigation" runs (`imbalance.strategy: none`) for at least the headline models — `build.md §7.2` says results with/without mitigation must be reported; they aren't.
- Make the DataLoader deterministic (`generator` seeded, `worker_init_fn`) so seeds are actually reproducible.

### 3.2 `train/train_tree.py`
- Fix 0.5: separate `tune.py` (run once) from seeded training (fixed params).
- Increase Optuna trials (≥30) and widen the space (see 2.4); use `TPESampler`.
- RF and XGBoost use different imbalance mechanisms (`class_weight='balanced'` vs `sample_weight`) — mathematically similar but state it; the sweep script uses `sample_weight` for RF instead, so the sweep and the main run don't even match each other. Unify.
- Save test probabilities (`predict_proba`) and feature importances.

### 3.3 `main.py`
- Use `data/loaders.py` (1.5) instead of inline duplication.
- Save, next to `test_predictions.npy`: `test_probs.npy`, `test_timestamps.npy`, `test_mid_prices.npy`, `test_returns.npy`. The backtester (5.1) needs all four; today the run directory cannot support a backtest at all.
- Save the class distribution of each split into `metrics.json`.
- `--smoke-test` overwrites `y_train[0:3]` in place; harmless but make it operate on a copy.
- Replace `datetime.utcnow()` (deprecated) with timezone-aware `datetime.now(timezone.utc)`.

### 3.4 `run_all.py`
- Add a `--markets`, `--models`, `--seeds`, `--assets` filter so partial reruns are possible.
- Currently swallows subprocess failures and continues; write failures to a `failed_runs.json` and exit non-zero at the end.
- Loop over the multi-asset/exchange/window grid (5.4) and over `feature_set` variants.
- Call the significance runner (4.2) and the plotting script (4.4) after aggregation.

---

## 4. EVALUATION (`eval/`)

### 4.1 `eval/metrics.py`
- Add: per-class precision/recall, Cohen's κ, log-loss and Brier score (needs probabilities), and expected-calibration-error.
- Add a "lift over majority-class" metric (accuracy − majority-class share) so chance level is visible in the table.

### 4.2 `eval/significance.py` — currently never called anywhere
- Add a runner (`eval/run_significance.py`) that, for every market, loads the saved per-seed predictions and produces: (a) paired t-test / Wilcoxon across the 5 seeds for each model pair on Macro-F1, Bal-Acc and MCC; (b) McNemar on pooled test predictions for the top-2 models; (c) 95% bootstrap CIs for each model's metrics. Output a CSV and a LaTeX-ready table with significance markers.
- Seed the bootstrap (`np.random.default_rng(seed)`) — currently unseeded, which contradicts the reproducibility claim.
- Add a **one-sample test against chance** for every model on crypto (is MCC > 0? is Bal-Acc > 1/3?). This is the single most important test given Reviewer 2's comment.

### 4.3 `eval/aggregate.py`
- Emit LaTeX tables directly (`booktabs`), with bold best-per-column, per-market, plus the baseline rows.
- Keep the confusion matrices: aggregate them (sum or mean across seeds) and save as CSV/figure instead of dropping them.
- Aggregate ablation, multi-asset, horizon-sweep, and backtest results into their own tables.
- Use `ddof=1` explicitly and state "std over 5 seeds" in the table caption.

### 4.4 New: `eval/plots.py` (matplotlib is in requirements but there is no plotting code)
- Confusion matrices (per market, best model); horizon × threshold heatmaps (class balance and Macro-F1); ablation bar charts with CIs; equity curves and drawdown plots from the backtest; calibration curves; feature-importance bars; per-asset comparison. Replace the current Figs 1/2/4/5 (which only redraw the tables) with these.

---

## 5. NEW EXPERIMENT MODULES (`experiments/`) — the reviewer-facing additions

### 5.1 `experiments/backtest.py` (Reviewer 2 — mandatory)
- Inputs: saved `test_probs.npy`, `test_mid_prices.npy`, `test_timestamps.npy` per run.
- Signal rules: (a) argmax → long / flat / short; (b) confidence-thresholded: trade only if `max(prob) > τ`, sweep τ ∈ {0.4, 0.5, 0.6, 0.7}.
- Execution model: enter at next observation's best ask/bid (not mid) after a configurable latency of L observations (L ∈ {0, 1, 4, 20}); hold for the label horizon H or until signal flips; position sizing = fixed notional.
- Costs: maker/taker fee in bps (Binance perp taker 4–5 bps, configurable), slippage in bps or as half-spread + volume-impact term.
- Outputs per (model, seed, τ, L, cost): gross and net return, annualized Sharpe and Sortino (state the annualization factor), max drawdown, hit rate, turnover, number of trades, average holding time, PnL per trade in bps.
- Benchmarks: buy-and-hold, random signal (seeded), majority-class, perfect-foresight (upper bound).
- Cost-sensitivity plot: net Sharpe vs fee level.
- FI-2010 backtest using the DecPre-derived mid-price (1.3); note in the paper that FI-2010 has no spread/volume-realistic cost model, so use a bps assumption.

### 5.2 `experiments/ablation_structured_transformer.py` (Reviewer 1 — mandatory rewrite)
- Currently 1 seed, 4 combos, no depth/width/regularization/training-schedule axes, and no aggregation. Rewrite as a factorial grid: `token_mode {flat, grouped, level}` × `pooling {mean, cls, attention}` × `num_layers {2, 4, 6}` × `d_model {64, 128}` × `dropout/weight_decay {off, on}` × `epochs/patience {fixed15, early-stop}`; 5 seeds each; include a parameter-matched standard Transformer at each param budget. Use one-factor-at-a-time from the headline config if the full grid is too expensive, but report which design you used.
- Aggregate into a single ablation table with significance vs the standard Transformer.
- Write temp configs to a scratch directory, not `configs/` (run_all's glob would pick them up).

### 5.3 `experiments/threshold_sweep.py`
- Fix 0.1 (labels on raw prices). Re-run; the whole justification for H=40 / 1 bp must be regenerated.
- Extend the grid to include the smoothed labeling scheme and adaptive thresholds (1.2), and record the class distribution *of the test split*, not just the whole set.
- Save the sweep results somewhere that is **not `.gitignore`d** — `.gitignore` currently excludes `results/` and `*.csv`, so the "supporting evidence kept in the repo" (build.md §7.1) does not exist in the repo.

### 5.4 New: `experiments/multi_asset.py` (Reviewer 2 — mandatory)
- Grid: symbols {BTCUSDT, ETHUSDT, SOLUSDT} × exchanges {Binance, Bybit or OKX} × ≥2 non-overlapping multi-day windows in different volatility regimes. Same pipeline, same configs, 5 seeds.
- Add a data-collection helper (`scripts/collect_lob.py`) using public depth WebSocket/REST snapshots at a fixed interval if Kaggle sets don't exist for the other assets; document exact collection times.
- Output: per-asset table, rank-correlation of model rankings across assets/windows (Kendall's τ) to support or refute "rankings are market-dependent."
- This also satisfies `build.md §7.6` (cross-day robustness), which was specified but never built.

### 5.5 New: `experiments/horizon_sweep.py`
- Run the headline models on crypto at H ∈ {10, 20, 40, 100, 250} and FI-2010 at k ∈ {10, 20, 50, 100}; 3–5 seeds; plot Macro-F1/MCC vs horizon per market. Addresses "one horizon only."

### 5.6 New: `experiments/microstructure_stats.py`
- For each dataset/window: mean/median spread in ticks and bps, depth at level 1/5/10, snapshot update rate, mid-price return volatility at the label horizon, autocorrelation of mid-price returns and of OFI, fraction of snapshots where |return| < threshold. One table comparing FI-2010 vs each crypto asset. This is the evidence for *why* crypto is harder and turns the negative result into a contribution.

### 5.7 New: `experiments/feature_set_comparison.py`
- FI-2010 full144 vs raw40 vs raw40 + engineered (1.1) ; crypto raw40 vs raw40 + engineered. Same models, 5 seeds. Removes the feature-set confound from the cross-market claim.

### 5.8 New: `experiments/imbalance_ablation.py`
- `class_weight` vs `focal_loss` vs `none` for neural models on both markets. Promised in build.md §7.2, absent from the paper.

---

## 6. TESTS (`tests/`) — only metrics are tested; the bug in 0.1 would have been caught by any labeling test

Add:
- `test_labeling.py`: (a) labels computed before and after `to_relative_price` are identical; (b) no NaN/inf in returns; (c) a synthetic monotone up-trend produces all-Up labels; (d) label counts match the saved class distribution.
- `test_splits.py`: chronological order preserved; no index overlap; embargo gap present; scaler statistics computed only from train rows (mutation check).
- `test_fi2010_labels.py`: FI-2010 remapped label direction agrees with sign of DecPre mid-price change on a sample.
- `test_models.py`: forward-pass shape checks for every model (including windowed DeepLOB), parameter-count reproducibility across seeds.
- `test_backtest.py`: perfect-foresight signal yields positive PnL; zero-signal yields zero PnL; costs reduce net PnL monotonically.
- `test_reproducibility.py`: two runs with the same seed give identical `test_predictions.npy`.

---

## 7. REPO HYGIENE / REPRODUCIBILITY (reviewers do open the repo)

- Remove `.idea/` from the repo and add it to `.gitignore`.
- Stop ignoring `results/` wholesale. Commit aggregated CSVs, tuning results, sweep tables, significance tables, and figures (not raw `.npy` predictions) so evidence lives with the code.
- Pin exact versions for every dependency (`huggingface_hub>=1.29.0` is unpinned; `tqdm`, `pyyaml` fine). Add a `environment.yml` or Docker file plus the GPU used.
- README: make dataset source, file names, date ranges, row counts, horizon, threshold, labeling scheme, and feature set exactly match the paper. Add a "Reproduce Table N" command per table.
- Add a top-level `Makefile`/`reproduce.sh`: prepare data → tune → run_all → sweeps → ablations → backtest → significance → plots → tables.
- Delete or update `build.md` — it documents the *previous* plan and contains rules the code violates (§8.8 "Improved" naming, §7.2/7.5/7.6 never run). Reviewers who read it will notice.
- Log Python/OS/GPU in `run_manifest.json`.

---

## 8. PAPER ↔ CODE CONSISTENCY CHECKLIST (things that are currently contradictory)

| Paper says | Code does | Action |
|---|---|---|
| Improved Transformer has a deeper encoder / more capacity | Same depth and width as standard (2 layers, d=64) | Fix text or fix code (0.3) |
| "Relative price representation" | Absolute `price − mid` | Rename or switch to fractional (1.1) |
| Crypto source: S. Raz, Kaggle | Kaggle `martinsn/...` | Reconcile (1.4) |
| Per-class F1 and Weighted-F1 "considered" | Computed but never reported | Add to tables (4.3) |
| Class-balanced metrics emphasized | No chance-level baselines | Add baselines (2.5) |
| 354,825 FI-2010 observations | Unclear which files; standard counts differ | Fix loader, report split sizes (0.4) |
| "Three-day window" | Row-count slicing, not dates | Timestamp slicing (0.6) |
| Seeds capture training variance | Seeds also change tuned hyperparameters | Freeze params (0.5) |
| Horizon/threshold "resolved via sweep" | Sweep ran on broken labels | Re-run sweep (5.3) |
| Standard 7-day/3-day FI split, k=10 | Not stated in the paper at all | Add to paper §III |
| FI-2010 Down/Stationary/Up = classes 0/1/2 | Official labels are Up=1, Down=3 | Remap (0.2) |

---

## 9. Suggested execution order

1. Fix 0.1, 0.2, 0.4, 0.6 and add the labeling/split tests (§6) — verify labels visually against the mid-price series before anything else.
2. Add baselines (2.5), probability/timestamp saving (3.3), freeze hyperparameters (0.5). Re-run the headline grid. Check whether crypto is now above chance.
3. Windowed dataset + real DeepLOB (1.5, 2.1) + temporal Transformers (2.2). Re-run FI-2010; you should get within a few points of published DeepLOB numbers or the reproduction is still wrong.
4. Multi-asset data collection (5.4) in parallel with steps 1–3 — it is the slowest.
5. Ablations (5.2), horizon/threshold/feature-set/imbalance sweeps (5.3, 5.5, 5.7, 5.8).
6. Backtester (5.1), microstructure stats (5.6), significance (4.2), plots (4.4), LaTeX tables (4.3).
7. Rewrite the paper against the new numbers; remove every reference to "the earlier experiment."
