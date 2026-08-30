# Build Specification: Cross-Market LOB Prediction Research Repo

**Read this entire document before writing any code.** This is a research repository, not a demo — every experiment described here needs to run end-to-end from a clean checkout and produce numbers that go directly into a paper. Prioritize reproducibility and traceability over cleverness. Where a decision is genuinely ambiguous, make the most defensible choice, implement it, and **write down what you chose and why** in the README and in code comments — do not silently guess and move on, and do not leave two conflicting implementations lying around.

---

## 1. Project Goal

Build a repository that trains and evaluates 5 model variants on 2 markets, producing a reproducible cross-market comparison:

**Models:** Random Forest, XGBoost, DeepLOB (single-snapshot variant — see §2.3 for why), a standard Transformer, and a structurally-modified Transformer (token-grouped input + attention pooling — do NOT call this "Improved" in code or output; call it `structured_transformer` or similar, since it does not reliably outperform the standard Transformer and that framing caused problems in peer review).

**Markets:**
- **FI-2010** — a public stock-market limit order book benchmark (traditional equities).
- **Crypto LOB** — a Bitcoin perpetual futures order book dataset from Binance (BTCUSDT.P), sourced from Kaggle.

**Task:** 3-class classification (price will go Up / stay Stationary / go Down) at a fixed prediction horizon.

**Evaluation:** Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, per-class F1, Matthews Correlation Coefficient (MCC), and confusion matrices — computed identically for every model/market combination, across multiple random seeds, with significance testing between top models.

This repo is a rebuild of an earlier, disorganized Colab-notebook-based project. That prior notebook only ever fully trained one model (the standard Transformer, on FI-2010) — Random Forest and XGBoost results were loaded from external pre-computed files with no visible training code, and DeepLOB / the modified Transformer / the entire crypto modeling pipeline did not exist in code anywhere. Treat this as building the real pipeline for the first time, not "cleaning up" existing code — you may reference the old notebook's Transformer training loop as a rough guide for that one model, but do not assume anything else from it is correct or complete.

---

## 2. Datasets

### 2.1 FI-2010 (stock market)

FI-2010 is a standard, publicly available academic benchmark: limit order book data from five stocks on the Nasdaq Nordic (Helsinki) exchange, covering 10 consecutive trading days in June 2010, with roughly 4 million raw order-book events. Consecutive events are sampled every 10 non-overlapping updates, giving each sample a **144-dimensional feature vector**, built from three published feature groups (Kercheval & Zhang, 2015 taxonomy):
1. Raw 10-level bid/ask price and volume (the most granular block).
2. Features describing the current LOB state using recent history (spreads, mid-price, price/volume differences, etc.).
3. Time-sensitive derivative features (rates of change in prices, volumes, and order intensity).

**Critical things you must resolve and document, not assume:**
- The official distribution includes **five separate label columns**, one per prediction horizon (commonly k = 10, 20, 30, 50, 100 events ahead), each 3-class (down/stationary/up), typically encoded as integers 1/2/3. **Determine which horizon is being used, and hard-code that choice explicitly in config with a comment citing the source.** Do not silently pick "whatever column happens to be at position 144" the way the old notebook did — that is a real, unresolved gap from the earlier version of this project. Every table you generate must state the horizon used.
- Standard practice (most published FI-2010 papers) uses the **first 7 days for training and the last 3 days for testing**, chronologically, with no shuffling across the boundary — replicate this if the official pre-split files support it.
- The dataset ships in multiple normalization variants (Z-score, Min-Max, Decimal Precision). **Pick one (Z-score is the most common in published work), document it explicitly, and fit any additional scaling only on the training split.**
- Do not fetch this from an ambiguous personal Drive file. Locate the canonical public source (the dataset's original release, or a well-documented mirror/GitHub loader used by published FI-2010 reproductions) and record the exact source and file names used in `scripts/prepare_fi2010.py` and the README, so someone else can get the identical data.
- Label distribution is known to be imbalanced — do not be surprised by this, and do not treat class imbalance mitigation as optional (see §7).

### 2.2 Cryptocurrency LOB (crypto market)

Source: Kaggle dataset "Bitcoin Limit Order Book (LOB) Data" (Binance BTCUSDT.P perpetual contract), 250ms-interval snapshots, ~3.7M rows across 12 consecutive days, 42 columns.

Column layout (verify against the Kaggle dataset's own documentation before trusting this, but this matches what was reverse-engineered from the prior notebook):
- Column `0` and `1`: an index/timestamp pair — confirm which is the usable chronological key.
- Columns `2`–`21`: 10 bid levels, alternating price and volume (`2`=best bid price, `3`=best bid volume, `4`=next level price, etc.).
- Columns `22`–`41`: 10 ask levels, same alternating pattern (`22`=best ask price, `23`=best ask volume, etc.).

**Feature engineering required:**
- Compute mid-price = (best_bid + best_ask) / 2.
- Convert raw prices to a **relative-price representation around the mid-price** (subtract mid-price from each price level) so the model isn't sensitive to BTC's absolute price level; retain volumes as-is (or log-scale if you find raw volume scale causes training instability — document if you do this).
- Fit any standardization/scaling on the training split only.

**Labeling — this is the single most consequential open decision in the whole project, resolve it deliberately:**
The earlier notebook explored several horizon/threshold combinations and left the codebase in an inconsistent state — one late-stage exploratory cell settled on a 500-event horizon (~125 seconds) while other project documentation described a 40-event horizon (~10 seconds), both with a ±1 basis-point (±0.0001 fractional) threshold. **You must not silently pick one.** Build this as a proper experiment (see §7.1), sweep horizon × threshold combinations, log the resulting class balance and downstream model performance for each, and let the results justify the final choice used in the "headline" tables. Whatever is chosen becomes the default in config, with the sweep results kept in the repo as supporting evidence.

**Data scope:** use a substantially larger contiguous window than the ~7 hours (100k rows) used previously if compute allows — you have 12 days of source data; using more of it (ideally 2–3+ full days, or all 12 if feasible) directly strengthens the paper's external validity. At minimum, run the final pipeline on two non-overlapping time windows and confirm model rankings are consistent across them.

---

## 3. Repository Layout

```
lob-cross-market/
├── main.py                          # single entry point; only file a user typically runs directly
├── run_all.py                       # loops over all (model × market × seed) combinations, aggregates results
├── configs/
│   ├── base.yaml                    # shared defaults
│   ├── fi2010_random_forest.yaml
│   ├── fi2010_xgboost.yaml
│   ├── fi2010_deeplob.yaml
│   ├── fi2010_transformer.yaml
│   ├── fi2010_structured_transformer.yaml
│   ├── crypto_random_forest.yaml
│   ├── crypto_xgboost.yaml
│   ├── crypto_deeplob.yaml
│   ├── crypto_transformer.yaml
│   └── crypto_structured_transformer.yaml
├── data/
│   ├── loaders.py                   # FI2010Dataset, CryptoDataset classes
│   ├── labeling.py                  # labeling logic + the horizon/threshold sweep utility
│   └── features.py                  # relative-price transform, scaler fit/transform (train-only fitting)
├── models/
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── deeplob.py
│   ├── transformer.py
│   └── structured_transformer.py    # supports ablation flags: token_mode, pooling_mode (see §7.3)
├── train/
│   ├── train_tree.py                # shared RF/XGBoost training + optional Optuna tuning
│   └── train_neural.py              # shared PyTorch loop for DeepLOB/Transformer/StructuredTransformer
├── eval/
│   ├── metrics.py                   # accuracy, balanced accuracy, macro/weighted F1, per-class F1, MCC, confusion matrix
│   ├── significance.py              # paired bootstrap test, McNemar's test
│   └── aggregate.py                 # multi-seed mean/std tables, final paper-ready CSVs
├── experiments/
│   ├── threshold_sweep.py           # crypto horizon × threshold sensitivity study
│   └── ablation_structured_transformer.py   # 2x2 ablation: token grouping × pooling
├── scripts/
│   ├── prepare_fi2010.py            # raw source -> clean train/val/test files, documents exact source
│   └── prepare_crypto.py            # Kaggle zip -> clean parquet/csv, documents exact source file
├── utils/
│   ├── seeding.py                   # set_seed(seed): torch, numpy, random, cuda — called first in every entry point
│   └── logging.py
├── tests/
│   └── test_metrics.py              # sanity tests for the metrics module (known-input/known-output cases)
├── results/                         # git-ignored; all run outputs land here
├── requirements.txt
└── README.md
```

No file in this repo should contain a hard-coded absolute path (no `/content/drive/...`, no personal directories). Every path is either relative to the repo root or passed in via config.

---

## 4. Libraries

| Purpose | Library | Notes |
|---|---|---|
| Data handling | `pandas`, `numpy` | standard |
| Tree models | `scikit-learn` (RandomForestClassifier), `xgboost` | |
| Neural models | `torch` | keep consistent with the prior Transformer code, which was already correct PyTorch |
| Config files | `pyyaml` | one YAML file per (model, market) run, see §5 |
| Hyperparameter search | `optuna` | for documented, reproducible tuning of RF/XGBoost/neural hyperparameters — see §7.4 |
| Metrics/stats | `scikit-learn.metrics`, `scipy.stats` | MCC and bootstrap/McNemar significance tests |
| Plotting (final reporting only, not inside training code) | `matplotlib` | keep plotting out of training/eval logic — it belongs in a separate reporting script |
| Progress bars | `tqdm` | |
| Testing | `pytest` | |

Pin exact versions in `requirements.txt` once the environment is finalized, and record them in the README's reproduction section.

---

## 5. Config Schema

Every run is defined by one YAML file. Example (`configs/crypto_xgboost.yaml`):

```yaml
market: crypto              # fi2010 | crypto
model: xgboost               # random_forest | xgboost | deeplob | transformer | structured_transformer

data:
  crypto_window_days: [1, 3]        # which days of the 12-day source to use
  horizon_events: 40                # resolved via the threshold sweep in §7.1 — do not leave ambiguous
  threshold: 0.0001                 # ±1 basis point
  standardize: true

model_params:
  n_estimators: null                # filled in by Optuna search, then locked and recorded here
  max_depth: null
  # ... etc, model-specific

imbalance:
  strategy: class_weight            # class_weight | focal_loss | none — see §7.2
  focal_gamma: 2.0                  # only used if strategy == focal_loss

training:
  epochs: 15                        # only relevant for neural models
  batch_size: 256
  learning_rate: 0.001
  seeds: [0, 1, 2, 3, 4]             # multi-seed run, not a single run

output_dir: results/crypto_xgboost/
```

`main.py` takes `--config <path> --seed <int>` (single run) or is invoked by `run_all.py` which loops over every config × every seed automatically.

---

## 6. File-by-File Responsibilities

### `utils/seeding.py`
```python
def set_seed(seed: int) -> None:
    # sets random.seed, np.random.seed, torch.manual_seed,
    # torch.cuda.manual_seed_all, and sets
    # torch.backends.cudnn.deterministic = True (document any speed cost this adds)
```
Must be called as the **first line of execution** in `main.py`, before any data loading or model construction. This did not exist anywhere in the previous codebase and is non-negotiable — every neural training run must be repeatable given the same seed.

### `data/loaders.py`
- `FI2010Dataset`: loads the prepared FI-2010 files, returns `(X_train, y_train, X_val, y_val, X_test, y_test)` as numpy arrays, chronologically split (7 days train / last 3 days test convention, with a validation carve-out from the training days — do not carve validation from the test days). Must expose which label-horizon column was used as an attribute, not just silently select one.
- `CryptoDataset`: loads the prepared crypto parquet/csv, applies the chosen horizon/threshold labeling from `data/labeling.py`, performs a **chronological** (not random) 70/15/15 train/val/test split, matching what the earlier project did correctly for this part.
- Both must raise a clear error rather than silently proceeding if expected columns are missing.

### `data/labeling.py`
- `label_by_threshold(returns, threshold) -> labels`: the core 3-class labeling function used by both the final pipeline and the sweep.
- `run_threshold_sweep(mid_prices, horizons: list, thresholds: list) -> pd.DataFrame`: for every (horizon, threshold) pair, compute resulting class balance (% each class) and save to `results/threshold_sweep.csv`. This is the tool that resolves the horizon ambiguity from §2.2 — run it, inspect it, and only then lock the final horizon/threshold into the model configs.

### `data/features.py`
- `to_relative_price(df) -> df`: mid-price-centered price representation for crypto.
- `TrainOnlyScaler`: thin wrapper ensuring `.fit()` is only ever called on training data and raising an error if `.fit()` is called more than once per pipeline run (this guards against the kind of silent leakage that's easy to introduce by accident).

### `models/*.py`
Each model file exposes a single factory function, e.g. `build_model(config: dict)`, returning a ready-to-train model (sklearn/xgboost estimator, or an `nn.Module` for neural models). Keep architectures close to what was already validated:
- `transformer.py`: scalar-token Transformer (40 or 144 scalar features projected to `d_model`, positional embedding, N encoder layers, mean pooling, classification head) — this is the one architecture from the old notebook that was implemented correctly; preserve its structure.
- `deeplob.py`: three 1D conv blocks → 2-layer bidirectional LSTM → classification head, operating on a single LOB snapshot (not the historical-window version from the original DeepLOB paper — name it accordingly in docstrings so nobody mistakes it for a full reproduction).
- `structured_transformer.py`: must support **independent flags** for (a) token representation — `flat` (scalar tokens) vs `grouped` (level-grouped multi-value tokens), and (b) pooling — `mean`, `cls`, or `attention`. This is required for the ablation in §7.3; do not hard-wire both changes together the way the old "Improved Transformer" did.
- `random_forest.py` / `xgboost_model.py`: must accept a `class_weight` / `sample_weight` argument wired through from config (§7.2) — the old codebase trained trees with no imbalance handling at all.

### `train/train_tree.py`
- Trains RF/XGBoost with actual visible `.fit()` calls in this repo (the old codebase had none — results were loaded from untraceable external files; that must not happen again).
- Supports an Optuna-based hyperparameter search over a validation split when `model_params` fields are `null` in config, then writes the resolved best hyperparameters back into a run manifest so they're recorded, not lost.

### `train/train_neural.py`
- Generic training loop reused by DeepLOB, Transformer, and StructuredTransformer.
- Supports `imbalance.strategy` = `class_weight` (weighted cross-entropy, as before) or `focal_loss` (implement standard focal loss, gamma configurable).
- Selects best checkpoint by validation Macro-F1 (this was already done correctly before — keep it).
- Every run writes: final config used, per-epoch training history, final metrics dict, raw predictions, and a run manifest (timestamp, seed, library versions) to `results/<run_id>/`.

### `eval/metrics.py`
- `compute_all_metrics(y_true, y_pred) -> dict` returning: accuracy, balanced_accuracy, macro_f1, weighted_f1, per_class_f1 (all 3 classes), MCC, and the raw confusion matrix. Every training script calls this same function — no model-specific metric computation duplicated elsewhere.

### `eval/significance.py`
- `paired_bootstrap(y_true, preds_a, preds_b, metric_fn, n_boot=2000) -> (diff, ci_low, ci_high, p_value)`
- `mcnemar(y_true, preds_a, preds_b) -> p_value`
- Used to compare, e.g., Random Forest vs XGBoost on crypto Macro-F1 across seeds — needed because the paper currently claims model rankings with no statistical backing.

### `eval/aggregate.py`
- Reads every `results/<run_id>/metrics.json`, groups by (model, market), computes mean ± std across seeds, and writes final tables in the same shape as the paper's Table II / III / IV, ready to drop into LaTeX or a spreadsheet.

---

## 7. Research Components That Must Be Built as Real Experiments (Not Scratch Cells)

These directly address weaknesses identified in peer review of the earlier draft of this paper. Each needs its own script under `experiments/` with saved, inspectable outputs — not something run once in a notebook and discarded.

### 7.1 Crypto labeling sensitivity sweep (`experiments/threshold_sweep.py`)
Sweep horizon (e.g., events ∈ {10, 40, 100, 250, 500}) × threshold (e.g., ∈ {0.0001, 0.0002, 0.0005, 0.001}), log resulting class balance for every combination, and separately log downstream Macro-F1 for at least the two tree models across a subset of the grid (tree models are cheap enough to run across the full grid; this is what actually justifies the final labeling choice instead of it being asserted).

### 7.2 Class-imbalance mitigation (built into `train_tree.py` / `train_neural.py`, not a separate script)
- Trees: `class_weight="balanced"` for RF, `sample_weight` computed the same way for XGBoost.
- Neural: focal loss as an alternative to weighted cross-entropy, selectable via config.
- Report results both with and without mitigation so the paper can show the effect, not just apply it silently.

### 7.3 Structured-Transformer ablation (`experiments/ablation_structured_transformer.py`)
Run all 4 combinations of `token_mode ∈ {flat, grouped}` × `pooling_mode ∈ {mean, attention}` on both markets. This isolates which design change is actually responsible for the performance difference versus the standard Transformer — the earlier paper only had a hypothesis here, not a controlled result.

### 7.4 Hyperparameter tuning, documented
Trees currently must not be trained on library defaults with no record of tuning. Use Optuna with a small, fixed search space (document it in the config comments), tune on the validation split, and log the winning hyperparameters into the run manifest so the paper can state exactly what was searched and what was chosen.

### 7.5 Multi-seed runs + significance testing
Every model/market config runs across **at least 5 seeds** (`training.seeds` in config). `eval/aggregate.py` reports mean ± std for every metric, and `eval/significance.py` is used to test whether, e.g., XGBoost's Macro-F1 advantage over Random Forest on crypto data is statistically distinguishable from noise.

### 7.6 Cross-day robustness check
Run the final crypto pipeline on two non-overlapping time windows from the 12-day source and confirm (or report honestly if not) that model rankings are stable across them.

---

## 8. Explicit Gotchas — Do Not Repeat These From the Previous Version

1. **Do not** select an FI-2010 label column implicitly. Explicitly name and document the prediction horizon used.
2. **Do not** leave the crypto horizon ambiguous. Resolve the 40-vs-500-event discrepancy found in the prior notebook via the sweep in §7.1, and record the final decision with justification.
3. **Do not** use only a small partial slice of the crypto data (a prior version used only 50,000 of a much larger available dataset while labeling it "for initial experimentation" and then never replaced it). Use the full intended window, and state exactly how many observations and what date/time range were used.
4. **Do not** load model predictions from external files with no training code in the repo. Every reported result must be reproducible via `python main.py --config ... --seed ...` in this repo, full stop.
5. **Do not** leave any Google Drive / Colab-specific paths anywhere.
6. **Do not** skip `set_seed()`. Every entry point calls it first.
7. **Do not** fit any scaler, normalizer, or class-weight computation on anything other than the training split.
8. **Do not** call the modified Transformer "Improved" anywhere in code, configs, or output filenames unless the ablation in §7.3 actually shows it winning — use a neutral name.

---

## 9. Definition of Done

- [ ] `python run_all.py` on a clean checkout, with only the documented data-prep steps run first, reproduces full metric tables for all 5 models × 2 markets × 5 seeds with no manual intervention.
- [ ] `eval/aggregate.py` output matches the shape of the paper's planned Table II/III/IV, with mean ± std per cell.
- [ ] Threshold sweep results exist under `results/threshold_sweep.csv` and the final crypto horizon/threshold in config is justified by reference to them.
- [ ] Ablation results for the structured Transformer exist and are referenced in README.
- [ ] Confusion matrices and MCC are present in every run's saved metrics, not just accuracy/F1.
- [ ] Significance test results exist for at least the top-2 models per market.
- [ ] `README.md` lets a stranger, from a clean clone, install dependencies, fetch/prepare both datasets (with exact source URLs/filenames), and reproduce every reported number with one documented command sequence.
- [ ] No absolute paths, no notebook-only code paths, no orphaned pre-computed result files without matching training code.
- [ ] `pytest tests/` passes, including sanity checks on `eval/metrics.py` against hand-computed known cases.