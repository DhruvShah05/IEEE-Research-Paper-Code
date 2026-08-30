"""
scripts/prepare_fi2010.py — Prepares FI-2010 raw data into fast-loading .npy arrays.

Data source (build.md §2.1 requirement — exact source documented here):
  FI-2010 (Finnish Stock Exchange LOB dataset — Kercheval & Zhang, 2015)
  Official release URL: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
  Alternative GitHub mirror used by many reproductions:
    https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    (see data/ folder — No Auction, Z-score normalisation variant)

  Expected local path after download:
    BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
      NoAuction_Zscore_Training/   ← 7 days (train split)
      NoAuction_Zscore_Testing/    ← 3 days (test split)

Split convention (build.md §2.1):
  The official FI-2010 release provides pre-split Training/Testing directories that
  correspond to the first 7 days of the 10-day recording period for training and
  the final 3 consecutive days for testing, which is the chronological convention
  used by the majority of published FI-2010 baselines (Ntakaris et al., 2018;
  Zhang et al., 2019; Wallbridge, 2020).  No shuffling is applied across this boundary.

Normalisation:
  Z-score variant (NoAuction_Zscore) is used, as it is the most common in published
  FI-2010 work.  Any additional StandardScaler fitting is performed on training data
  only inside main.py (build.md §8 rule 7).

Feature/label layout (per-row in the .txt files, transposed before save):
  Columns 0–143  : 144 features (raw LOB levels + derived features — Kercheval & Zhang §2)
  Columns 144–148: 5 label columns for prediction horizons k ∈ {10, 20, 30, 50, 100}
                   Labels are integers: 1=Down, 2=Stationary, 3=Up.
  Which horizon to use is set in data.horizon_k in the config (default k=10).
"""

import os
import glob
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fi2010_folder(folder_path: str) -> np.ndarray:
    """
    Loads all .txt files in a FI-2010 folder, sorts chronologically by filename,
    and concatenates into a single numpy array of shape (n_samples, 149).

    The raw .txt files are stored transposed (features × observations), so we
    hstack then transpose back to (observations × features).
    """
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder_path}")

    txt_files.sort()  # Alphabetical order == chronological order for FI-2010 naming convention
    logger.info(f"Loading {len(txt_files)} files from {folder_path}...")

    arrays = []
    for f in txt_files:
        logger.info(f"  Reading {os.path.basename(f)}...")
        # Each file is (149 × n_obs) — features-as-rows, observations-as-cols
        arr = np.loadtxt(f)
        arrays.append(arr)

    # hstack horizontally (join observations), then transpose to (n_obs × 149)
    combined = np.hstack(arrays).T
    logger.info(f"  Loaded shape after transpose: {combined.shape}")
    return combined


def main():
    # Source data must be in the project root (no absolute paths — build.md §3)
    base_dir  = 'BenchmarkDatasets/NoAuction/1.NoAuction_Zscore'
    train_dir = os.path.join(base_dir, 'NoAuction_Zscore_Training')
    test_dir  = os.path.join(base_dir, 'NoAuction_Zscore_Testing')

    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.warning(f"Cannot find FI-2010 dataset at {base_dir}. Attempting to download from Hugging Face...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="DhruvShah05/Crypto_Stocks_Data",
                repo_type="dataset",
                allow_patterns="BenchmarkDatasets/*",
                local_dir="."
            )
            logger.info("Download completed.")
        except ImportError:
            logger.error("huggingface_hub is not installed. Please install it to enable automatic downloading: pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            logger.error(f"Cannot find FI-2010 dataset at {base_dir}.")
            logger.error(
                "Download the Z-score variant from:\n"
                "  https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649\n"
                "or the GitHub mirror:\n"
                "  https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books\n"
                "Place the extracted BenchmarkDatasets/ folder in the project root."
            )
            return

    logger.info("--- Preparing FI-2010 Training Data (7 days) ---")
    train_data = load_fi2010_folder(train_dir)
    logger.info(f"Training data shape: {train_data.shape}")

    logger.info("--- Preparing FI-2010 Testing Data (3 days) ---")
    test_data = load_fi2010_folder(test_dir)
    logger.info(f"Testing data shape: {test_data.shape}")

    # Validate column count
    for name, arr in [('train', train_data), ('test', test_data)]:
        if arr.shape[1] < 149:
            raise ValueError(
                f"FI-2010 {name} has {arr.shape[1]} columns; expected 149 (144 features + 5 labels). "
                "Check that you are using the correct normalisation variant."
            )

    train_out = os.path.join(out_dir, 'fi2010_train.npy')
    test_out  = os.path.join(out_dir, 'fi2010_test.npy')

    logger.info(f"Saving to {train_out} and {test_out}...")
    np.save(train_out, train_data)
    np.save(test_out, test_data)

    logger.info("FI-2010 preparation complete.")
    logger.info(f"Train samples: {train_data.shape[0]:,}  Test samples: {test_data.shape[0]:,}")


if __name__ == '__main__':
    main()

