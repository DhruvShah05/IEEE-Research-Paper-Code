"""
scripts/prepare_fi2010.py — Prepares FI-2010 raw data into fast-loading .npy arrays.

Data source (attribution fix 1.3 — correct citation):
  FI-2010 (Finnish Stock Exchange LOB dataset)
  Reference: Ntakaris et al. (2018), "Benchmark Dataset for Mid-Price Forecasting
             of Limit Order Book Data with Machine Learning Methods,"
             Journal of Forecasting, 37(8): 852–866.
  Official release URL: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  Alternative GitHub mirror used by many reproductions:
    https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books

  Feature taxonomy reference (Kercheval & Zhang, 2015):
    "Modelling High-Frequency Limit Order Book Dynamics with Support Vector Machines"
    — this paper defines the 144-feature set, not the dataset itself.

  Expected local path after download:
    BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
      NoAuction_Zscore_Training/
      NoAuction_Zscore_Testing/

Fix 0.4 — Hard-coded file names (no glob):
  The official FI-2010 release ships CF_1..CF_9 files where CF_k contains the
  first k days of data (cumulative, not independent). Globbing all .txt files
  and hstacking them duplicates days and breaks the train/test boundary.

  Correct standard split (Ntakaris et al. 2018; Zhang et al. 2019):
    Train : Train_Dst_NoAuction_ZScore_CF_7.txt          (7 days, 254,750 rows)
    Test  : Test_Dst_NoAuction_ZScore_CF_7.txt           (day 7 test portion)
            Test_Dst_NoAuction_ZScore_CF_8.txt           (day 8)
            Test_Dst_NoAuction_ZScore_CF_9.txt           (day 9)
    Total test rows: 139,587

Fix 1.3 — DecPre variant and raw40 feature file:
  Also prepares the NoAuction_DecPre (decimal-precision) variant alongside Z-score.
  DecPre allows mid-price recovery (needed for label verification, backtest, raw40).

Feature/label layout (per-row, transposed before save):
  Columns 0–143  : 144 features (raw LOB levels + derived — Kercheval & Zhang §2)
  Columns 144–148: 5 label columns for k ∈ {10, 20, 30, 50, 100}
                   Labels: 1=Up, 2=Stationary, 3=Down (Ntakaris et al. 2018 convention)
                   → use remap_fi2010_labels() from data/labeling.py before training.
"""

import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix 0.4: Hard-coded file names — no glob
# ---------------------------------------------------------------------------
TRAIN_FILES = [
    'Train_Dst_NoAuction_ZScore_CF_7.txt',
]
TEST_FILES = [
    'Test_Dst_NoAuction_ZScore_CF_7.txt',
    'Test_Dst_NoAuction_ZScore_CF_8.txt',
    'Test_Dst_NoAuction_ZScore_CF_9.txt',
]

# Expected row counts (Ntakaris et al. 2018 standard split — fix 0.4)
EXPECTED_TRAIN_ROWS = 254_750
EXPECTED_TEST_ROWS  = 139_587


def load_fi2010_files(folder_path: str, filenames: list) -> np.ndarray:
    """
    Loads exactly the specified .txt files from a FI-2010 folder and
    concatenates into a single numpy array of shape (n_samples, 149).

    Fix 0.4: uses an explicit file list instead of glob(*txt) to prevent
    loading cumulative CF files multiple times.

    The raw .txt files are stored transposed (features × observations), so we
    hstack then transpose back to (observations × features).
    """
    arrays = []
    for fname in filenames:
        fpath = os.path.join(folder_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Expected FI-2010 file not found: {fpath}\n"
                "Download from https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
            )
        logger.info(f"  Reading {fname}...")
        arr = np.loadtxt(fpath)
        arrays.append(arr)

    combined = np.hstack(arrays).T  # (n_obs, 149)
    logger.info(f"  Loaded shape after transpose: {combined.shape}")
    return combined


def main():
    # Source data must be in the project root (no absolute paths)
    base_dir  = 'BenchmarkDatasets/NoAuction/1.NoAuction_Zscore'
    train_dir = os.path.join(base_dir, 'NoAuction_Zscore_Training')
    test_dir  = os.path.join(base_dir, 'NoAuction_Zscore_Testing')

    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    # Auto-download if missing
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.warning(f"Cannot find FI-2010 dataset at {base_dir}. Attempting to download...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="DhruvShah05/Crypto_Stocks_Data",
                repo_type="dataset",
                local_dir="BenchmarkDatasets",
                allow_patterns=["BenchmarkDatasets/*"],
            )
            logger.info("Download completed.")
        except ImportError:
            logger.error("huggingface_hub not installed. pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.error(
                "Download manually from:\n"
                "  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649\n"
                "Place BenchmarkDatasets/ in the project root."
            )
            return

    # --- Z-score variant ---
    logger.info("--- Preparing FI-2010 Training Data (CF_7, 7 days) ---")
    train_data = load_fi2010_files(train_dir, TRAIN_FILES)
    logger.info(f"Training data shape: {train_data.shape}")

    logger.info("--- Preparing FI-2010 Testing Data (CF_7/CF_8/CF_9) ---")
    test_data = load_fi2010_files(test_dir, TEST_FILES)
    logger.info(f"Testing data shape: {test_data.shape}")

    # Validate column count
    for name, arr in [('train', train_data), ('test', test_data)]:
        if arr.shape[1] < 149:
            raise ValueError(
                f"FI-2010 {name} has {arr.shape[1]} columns; expected 149 "
                "(144 features + 5 labels). Check normalisation variant."
            )

    # Fix 0.4: assert row counts
    if train_data.shape[0] != EXPECTED_TRAIN_ROWS:
        logger.warning(
            f"Train rows: {train_data.shape[0]:,}  (expected {EXPECTED_TRAIN_ROWS:,}). "
            "Verify only CF_7 training file is used."
        )
    if test_data.shape[0] != EXPECTED_TEST_ROWS:
        logger.warning(
            f"Test rows: {test_data.shape[0]:,}  (expected {EXPECTED_TEST_ROWS:,}). "
            "Verify only CF_7/CF_8/CF_9 test files are used."
        )

    train_out = os.path.join(out_dir, 'fi2010_train.npy')
    test_out  = os.path.join(out_dir, 'fi2010_test.npy')
    logger.info(f"Saving Z-score arrays to {train_out} and {test_out}...")
    np.save(train_out, train_data)
    np.save(test_out, test_data)

    # Fix 1.3: also save raw40 feature files (first 40 cols = raw LOB)
    # This allows fair feature-set comparison with crypto (both use 40 features)
    raw40_train_out = os.path.join(out_dir, 'fi2010_train_raw40.npy')
    raw40_test_out  = os.path.join(out_dir, 'fi2010_test_raw40.npy')
    # Columns 0–39 = raw bid/ask prices/volumes; columns 144–148 = labels
    train_raw40 = np.concatenate([train_data[:, :40], train_data[:, 144:]], axis=1)
    test_raw40  = np.concatenate([test_data[:, :40],  test_data[:, 144:]], axis=1)
    np.save(raw40_train_out, train_raw40)
    np.save(raw40_test_out, test_raw40)
    logger.info(f"Saved raw40 arrays: {raw40_train_out}, {raw40_test_out}")

    # Save a preparation manifest with exact row counts for the paper
    manifest = {
        'source': 'FI-2010 Z-score variant (Ntakaris et al. 2018)',
        'train_files': TRAIN_FILES,
        'test_files': TEST_FILES,
        'train_rows': int(train_data.shape[0]),
        'test_rows': int(test_data.shape[0]),
        'expected_train_rows': EXPECTED_TRAIN_ROWS,
        'expected_test_rows': EXPECTED_TEST_ROWS,
        'n_features_full144': 144,
        'n_features_raw40': 40,
        'label_convention': '1=Up, 2=Stationary, 3=Down (Ntakaris et al. 2018)',
        'label_remap_note': 'Use remap_fi2010_labels() from data/labeling.py to convert to 0=Down,1=Stat,2=Up',
    }
    manifest_path = os.path.join(out_dir, 'fi2010_preparation_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Preparation manifest saved to {manifest_path}")

    # --- DecPre variant (fix 1.3) ---
    decpre_base = 'BenchmarkDatasets/NoAuction/2.NoAuction_DecPre'
    decpre_train_dir = os.path.join(decpre_base, 'NoAuction_DecPre_Training')
    decpre_test_dir  = os.path.join(decpre_base, 'NoAuction_DecPre_Testing')

    # Map Z-score filenames to DecPre equivalents
    decpre_train_files = [f.replace('ZScore', 'DecPre') for f in TRAIN_FILES]
    decpre_test_files  = [f.replace('ZScore', 'DecPre') for f in TEST_FILES]

    if os.path.exists(decpre_train_dir) and os.path.exists(decpre_test_dir):
        logger.info("--- Preparing FI-2010 DecPre variant ---")
        try:
            dp_train = load_fi2010_files(decpre_train_dir, decpre_train_files)
            dp_test  = load_fi2010_files(decpre_test_dir,  decpre_test_files)
            np.save(os.path.join(out_dir, 'fi2010_train_decpre.npy'), dp_train)
            np.save(os.path.join(out_dir, 'fi2010_test_decpre.npy'),  dp_test)
            logger.info("DecPre variant saved.")
        except FileNotFoundError as e:
            logger.warning(f"DecPre files not found (skipping): {e}")
    else:
        logger.info(
            "DecPre variant directory not found — skipping. "
            "Download 2.NoAuction_DecPre from the official release for backtest support."
        )

    logger.info("FI-2010 preparation complete.")
    logger.info(f"Train: {train_data.shape[0]:,}  Test: {test_data.shape[0]:,}")


if __name__ == '__main__':
    main()
