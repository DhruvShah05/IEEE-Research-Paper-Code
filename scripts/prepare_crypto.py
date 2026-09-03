"""
scripts/prepare_crypto.py — Prepares raw Crypto LOB CSV into a fast-loading parquet.

Data source (fix 1.4 — reconciled):
  Dataset : "High Frequency LOB BTC/USDT Binance"
  Platform: Kaggle
  URL     : https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance
  Author  : martinsn
  File    : bitcoin_lob_data.csv
  Coverage: 12 consecutive days, 250ms snapshot interval, ~3.7M rows, 42 columns.

  Note: the paper previously cited S. Raz (siavashraz/bitcoin-perpetual...) on Kaggle.
  This script uses the martinsn dataset.  The paper, README, and this script now agree.
  If you wish to use the S. Raz dataset, pass --input <path_to_raz_file>.

Column schema (verified against Kaggle documentation):
  Col '0'  : UNIX millisecond timestamp (chronological key)
  Col '1'  : Human-readable datetime string
  Cols '2'–'21'  : 10 bid levels — alternating price and volume
  Cols '22'–'41' : 10 ask levels — alternating price and volume

Fix 1.4:
  - Replace broken download_bucket_files / sync_bucket calls with hf_hub_download
    / snapshot_download from huggingface_hub (the old API doesn't exist).
  - Add data-quality report step: gaps, duplicates, crossed books, zero volumes,
    non-monotone levels → saved as data/processed/crypto_quality_report.json.
  - Add --symbol, --exchange, --input CLI args for multi-asset support.
"""

import argparse
import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLS = 42  # cols 0–41


def run_data_quality_report(df: pd.DataFrame, out_dir: str) -> dict:
    """
    Runs a data-quality scan and saves the report as crypto_quality_report.json.

    Checks:
      - Timestamp gaps (> 2 × median interval)
      - Duplicate timestamps
      - Crossed books (best bid >= best ask)
      - Zero volumes at level 1
      - Non-monotone bid levels (bid prices should be decreasing)
      - Non-monotone ask levels (ask prices should be increasing)

    Fix 1.4 requirement.
    """
    report = {}

    # Timestamp column
    ts = df['0'].astype(float)
    dt = ts.diff().dropna()
    median_interval_ms = float(dt.median())
    gap_threshold_ms   = 2 * median_interval_ms

    gaps = dt[dt > gap_threshold_ms]
    report['n_rows']              = int(len(df))
    report['median_interval_ms']  = median_interval_ms
    report['gap_threshold_ms']    = gap_threshold_ms
    report['n_timestamp_gaps']    = int(len(gaps))
    report['largest_gap_ms']      = float(gaps.max()) if len(gaps) > 0 else 0.0

    # Duplicate timestamps
    report['n_duplicate_timestamps'] = int(ts.duplicated().sum())

    # Crossed books
    bid1 = df['2'].astype(float)
    ask1 = df['22'].astype(float)
    crossed = (bid1 >= ask1).sum()
    report['n_crossed_books'] = int(crossed)

    # Zero volumes at level 1
    zero_bid_vol = (df['3'].astype(float) == 0).sum()
    zero_ask_vol = (df['23'].astype(float) == 0).sum()
    report['n_zero_bid_vol_l1'] = int(zero_bid_vol)
    report['n_zero_ask_vol_l1'] = int(zero_ask_vol)

    # Non-monotone bid levels (bid prices should decrease: bid1 > bid2 > ...)
    bid_price_cols = [str(i) for i in range(2, 22, 2)]   # '2','4',...,'20'
    ask_price_cols = [str(i) for i in range(22, 42, 2)]  # '22','24',...,'40'
    bid_prices = df[bid_price_cols].astype(float)
    ask_prices = df[ask_price_cols].astype(float)
    bid_nonmono = ((bid_prices.diff(axis=1) > 0).any(axis=1)).sum()
    ask_nonmono = ((ask_prices.diff(axis=1) < 0).any(axis=1)).sum()
    report['n_nonmonotone_bid_levels'] = int(bid_nonmono)
    report['n_nonmonotone_ask_levels'] = int(ask_nonmono)

    report['date_range_start'] = str(df['1'].min()) if '1' in df.columns else 'unknown'
    report['date_range_end']   = str(df['1'].max()) if '1' in df.columns else 'unknown'

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'crypto_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Data quality report saved to {report_path}")

    # Log summary
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Prepare crypto LOB CSV → parquet."
    )
    parser.add_argument(
        '--input', default=None,
        help="Path to raw LOB CSV. Defaults to bitcoin_lob_data.csv in project root."
    )
    parser.add_argument(
        '--symbol', default='BTCUSDT',
        help="Trading symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT)."
    )
    parser.add_argument(
        '--exchange', default='binance',
        help="Exchange name (e.g. binance, bybit)."
    )
    parser.add_argument(
        '--out_dir', default='data/processed',
        help="Output directory for parquet file."
    )
    args = parser.parse_args()

    symbol   = args.symbol.upper()
    exchange = args.exchange.lower()
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Determine input file
    if args.input:
        input_file = args.input
    elif symbol == 'BTCUSDT' and exchange == 'binance':
        input_file = 'bitcoin_lob_data.csv'
    else:
        input_file = f'{exchange}_{symbol.lower()}_lob_data.csv'

    # Output parquet filename (supports multi-asset)
    if symbol == 'BTCUSDT' and exchange == 'binance':
        out_file = os.path.join(out_dir, 'crypto_data.parquet')  # backward compat
    else:
        out_file = os.path.join(out_dir, f'{exchange}_{symbol.lower()}_data.parquet')

    # Auto-download if missing (fix 1.4: use hf_hub_download / snapshot_download)
    if not os.path.exists(input_file):
        logger.warning(f"Cannot find {input_file}. Attempting Hugging Face download...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="DhruvShah05/Crypto_Stocks_Data",
                filename=os.path.basename(input_file),
                repo_type="dataset",
                local_dir=os.path.dirname(input_file) or '.',
            )
            input_file = downloaded
            logger.info(f"Downloaded to {input_file}")
        except ImportError:
            logger.error("huggingface_hub not installed. pip install huggingface_hub")
            return
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.error(
                f"Cannot find {input_file}.\n"
                "Download from Kaggle:\n"
                "  https://www.kaggle.com/datasets/martinsn/high-frequency-lob-btcusdt-binance\n"
                "and place bitcoin_lob_data.csv in the project root."
            )
            return

    logger.info(f"Reading {input_file} for {exchange}/{symbol}...")
    df = pd.read_csv(input_file)

    # Drop unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        logger.info(f"Dropping unnamed index column: {df.columns[0]!r}")
        df = df.drop(columns=[df.columns[0]])

    logger.info(f"Loaded — shape: {df.shape}")

    # Column validation
    if df.shape[1] < EXPECTED_FEATURE_COLS:
        raise ValueError(
            f"Expected at least {EXPECTED_FEATURE_COLS} columns, got {df.shape[1]}. "
            "Verify you are using the correct dataset."
        )

    for required_col in ('0', '1', '2', '22'):
        if required_col not in df.columns:
            raise ValueError(
                f"Required column '{required_col}' not found. "
                "Column names should be '0', '1', ..., '41'."
            )

    # Datetime conversion
    logger.info("Converting datetime column ('1') to pandas datetime...")
    df['1'] = pd.to_datetime(df['1'], errors='coerce')

    # Sort by UNIX timestamp
    logger.info("Sorting by timestamp column ('0')...")
    df = df.sort_values(by='0').reset_index(drop=True)

    logger.info(f"Date range: {df['1'].min()} → {df['1'].max()}")
    logger.info(f"Total rows: {len(df):,}")

    # Fix 1.4: data-quality report
    logger.info("Running data quality checks...")
    run_data_quality_report(df, out_dir)

    logger.info(f"Saving to {out_file} (pyarrow parquet)...")
    df.to_parquet(out_file, engine='pyarrow', index=False)
    logger.info("Crypto preparation complete.")


if __name__ == '__main__':
    main()
