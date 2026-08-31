"""
scripts/prepare_crypto.py — Prepares raw Crypto LOB CSV into a fast-loading parquet.

Data source (build.md §2.2 requirement — exact source documented here):
  Dataset : "Bitcoin Limit Order Book (LOB) Data" (Binance BTCUSDT perpetual futures)
  Platform: Kaggle
  URL     : https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance
  File    : bitcoin_lob_data.csv  (place in the project root before running this script)
  Coverage: 12 consecutive days, 250ms snapshot interval, ~3.7M rows, 42 columns.

Column schema (verified against Kaggle documentation — build.md §2.2):
  Col '0'  : UNIX millisecond timestamp (chronological key)
  Col '1'  : Human-readable datetime string
  Cols '2'–'21'  : 10 bid levels — alternating price and volume
                   '2'=best_bid_price, '3'=best_bid_vol, '4'=next_bid_price, ...
  Cols '22'–'41' : 10 ask levels — alternating price and volume
                   '22'=best_ask_price, '23'=best_ask_vol, '24'=next_ask_price, ...
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLS = 42  # cols 0–41


def main():
    input_file = 'bitcoin_lob_data.csv'
    out_dir    = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, 'crypto_data.parquet')

    if not os.path.exists(input_file):
        logger.warning(f"Cannot find {input_file} locally. Attempting to download from Hugging Face...")
        try:
            from huggingface_hub import download_bucket_files
            download_bucket_files(
                bucket_id="DhruvShah05/Crypto_Stocks_Data",
                files=[(input_file, input_file)]
            )
            logger.info("Download completed.")
        except ImportError:
            logger.error("huggingface_hub is not installed. Please install it to enable automatic downloading: pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {e}")
            logger.error(
                f"Cannot find {input_file} in the project root.\n"
                "Download from Kaggle:\n"
                "  https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance\n"
                "and place bitcoin_lob_data.csv in the project root."
            )
            return

    logger.info(f"Reading {input_file}... (this may take a minute for large files)")
    df = pd.read_csv(input_file)

    # Drop leading unnamed index column if present (artefact of CSV export with row index)
    if df.columns[0].startswith('Unnamed'):
        logger.info(f"Dropping unnamed index column: {df.columns[0]!r}")
        df = df.drop(columns=[df.columns[0]])

    logger.info(f"Loaded crypto data — shape: {df.shape}")

    # Column validation
    if df.shape[1] < EXPECTED_FEATURE_COLS:
        raise ValueError(
            f"Expected at least {EXPECTED_FEATURE_COLS} columns, got {df.shape[1]}. "
            "Verify you are using the correct Kaggle dataset (see source URL in this script's docstring)."
        )

    # Confirm chronological key columns are present
    for required_col in ('0', '1', '2', '22'):
        if required_col not in df.columns:
            raise ValueError(
                f"Required column '{required_col}' not found. "
                "Column names should be '0', '1', ..., '41' per build.md §2.2."
            )

    # Col '1' is a datetime string — convert for potential downstream use
    logger.info("Converting datetime column ('1') to pandas datetime...")
    df['1'] = pd.to_datetime(df['1'], errors='coerce')

    # Sort by UNIX timestamp (col '0') to guarantee chronological order
    logger.info("Sorting by timestamp column ('0')...")
    df = df.sort_values(by='0').reset_index(drop=True)

    logger.info(f"Date range: {df['1'].min()} → {df['1'].max()}")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Saving to {out_file} (pyarrow parquet)...")
    df.to_parquet(out_file, engine='pyarrow', index=False)

    logger.info("Crypto preparation complete.")


if __name__ == '__main__':
    main()

