# Cross-Market LOB Prediction Research Repo

This repository provides a fully reproducible pipeline for evaluating five model architectures across two distinct Limit Order Book (LOB) datasets: traditional equities (FI-2010) and cryptocurrency futures (Binance BTCUSDT.P). It accompanies our paper on cross-market LOB prediction.

**Models:** Random Forest · XGBoost · DeepLOB (single-snapshot variant) · Transformer · StructuredTransformer  
**Markets:** FI-2010 (stock, Helsinki) · Binance BTCUSDT.P perpetual futures (crypto)  
**Task:** 3-class price-direction classification (Up / Stationary / Down)

---

## Reproducibility Guarantees

- `set_seed()` enforces deterministic behavior for PyTorch, NumPy, and Python's random module before any data loading or model construction in every entry point.
- All scalers / class weights are fit **exclusively on the training split**; the validation and test sets are never seen during fitting.
- Every run saves: `metrics.json`, `run_manifest.json` (timestamp + library versions), `config_used.json`, `training_history.json` (neural only), and raw `test_predictions.npy` for post-hoc significance testing.
- Optuna hyperparameter search results are logged to `tuned_config.json` per run.

---

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Pinned library versions** (see `requirements.txt` for exact versions used):

| Library | Version |
|---|---|
| torch | 2.2.1 |
| scikit-learn | 1.4.1.post1 |
| xgboost | 2.0.3 |
| pandas | 2.2.1 |
| numpy | 1.26.4 |
| optuna | 3.5.0 |
| scipy | 1.12.0 |
| pyarrow | 15.0.0 |

---

## Data Preparation

### FI-2010 (stock market)

**Source:** FI-2010 (Kercheval & Zhang, 2015) — Finnish Stock Exchange LOB benchmark  
**Official URL:** https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649  
**Alternative mirror:** https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books (see `data/` folder)  
**Variant used:** `NoAuction_Zscore` (Z-score normalised, no auction events)

Download and place the extracted folder in the project root so the path is:
```
BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
  NoAuction_Zscore_Training/
  NoAuction_Zscore_Testing/
```

Then prepare:
```bash
python3 scripts/prepare_fi2010.py
```

> The `Training/` directory covers the first 7 days chronologically; `Testing/` covers the final 3 days — matching the standard split used in published FI-2010 baselines (Ntakaris et al., 2018; Zhang et al., 2019).

**Prediction horizon:** k=10 events ahead (default; configurable via `data.horizon_k` in each config).

---

### Crypto LOB (Binance BTCUSDT.P futures)

**Source:** "Bitcoin Limit Order Book (LOB) Data" on Kaggle  
**URL:** https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance  
**File:** `bitcoin_lob_data.csv` — place in the **project root** before running

Then prepare:
```bash
python3 scripts/prepare_crypto.py
```

This sorts chronologically and saves a fast-loading parquet at `data/processed/crypto_data.parquet`.

> **Labeling horizon:** The final `horizon_events` and `threshold` in `configs/crypto_*.yaml` are chosen based on the sweep in `experiments/threshold_sweep.py`. Run the sweep first and inspect `results/threshold_sweep_balance.csv` + `results/threshold_sweep_f1.csv` to verify the choice.

---

## Running Experiments

### Single run

```bash
python3 main.py --config configs/fi2010_random_forest.yaml --seed 42
```

Results are written to `results/fi2010_random_forest/seed_42/`.

### Full reproduction (all 5 models × 2 markets × 5 seeds)

```bash
python3 run_all.py
```

This loops over every config in `configs/` (excluding `base.yaml`) × seeds `[0,1,2,3,4]`, then calls `eval/aggregate.py` to produce `results/aggregated_metrics.csv` — the final mean±std tables for the paper.

---

## Special Research Studies

### Crypto labeling sensitivity sweep (§7.1)

Sweeps horizon ∈ {10, 40, 100, 250, 500} × threshold ∈ {0.0001, 0.0002, 0.0005, 0.001}, logging class balance and tree-model Macro-F1 for every combination.

```bash
python3 experiments/threshold_sweep.py
```

Outputs:
- `results/threshold_sweep_balance.csv` — class distribution
- `results/threshold_sweep_f1.csv` — RF + XGBoost Macro-F1 per (horizon, threshold)

Use these to justify the final `horizon_events` / `threshold` locked in the crypto configs.

### Structured Transformer ablation (§7.3)

Runs all 4 combinations of `token_mode ∈ {flat, grouped}` × `pooling_mode ∈ {mean, attention}` on both markets:

```bash
python3 experiments/ablation_structured_transformer.py
```

Results land in `results/ablation_*/`. This isolates which structural change is responsible for performance differences vs the standard Transformer.

---

## Configuration

The `configs/` directory contains one YAML file per (model, market) combination.

Key fields per config:

| Field | Description |
|---|---|
| `data.horizon_k` | FI-2010 prediction horizon k ∈ {10,20,30,50,100} — explicitly documented, never implicit |
| `data.horizon_events` | Crypto horizon in events (from sweep) |
| `data.threshold` | Crypto ±fractional return threshold (from sweep) |
| `model_params.n_estimators` | `null` → triggers Optuna search; set to int to skip tuning |
| `imbalance.strategy` | `class_weight` \| `focal_loss` \| `none` |

---

## Testing

```bash
pytest tests/
```

`tests/test_metrics.py` contains sanity checks for every metric in `eval/metrics.py` against hand-computed known-value cases.

