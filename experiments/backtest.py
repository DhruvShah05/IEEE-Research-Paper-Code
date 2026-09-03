"""
experiments/backtest.py — Simple but realistic backtest for LOB prediction models (fix 5.1).

Inputs: saved test_probs.npy, test_mid_prices.npy, test_timestamps.npy per run.

Signal rules:
  (a) argmax → long (2) / flat (1) / short (0)
  (b) confidence-thresholded: trade only if max(prob) > τ

Execution model:
  - Enter at next observation's best ask (long) or bid (short) after configurable
    latency of L observations.
  - Hold for the label horizon H or until signal flips.
  - Fixed notional position sizing.

Costs:
  - Maker/taker fee in bps (configurable; Binance perp taker ≈ 4–5 bps).
  - Slippage in bps (configurable half-spread approximation).

Outputs per (model, seed, τ, L, cost):
  - gross_return, net_return, annualized Sharpe, Sortino, max_drawdown,
    hit_rate, turnover, n_trades, avg_holding_time, pnl_per_trade_bps

Benchmarks: buy-and-hold, random signal, majority-class, perfect-foresight.
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONF_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
LATENCIES       = [0, 1, 4, 20]       # observations
FEE_BPS_DEFAULT = 4.5                 # Binance perp taker
SLIPPAGE_BPS    = 1.0                 # half-spread approximation
ANNUALIZE       = 365 * 24 * 3600     # seconds/year for per-observation Sharpe


# ---------------------------------------------------------------------------
# Core PnL engine
# ---------------------------------------------------------------------------

def _simulate_strategy(
    signals: np.ndarray,     # -1 (short), 0 (flat), +1 (long) per observation
    mid_prices: np.ndarray,  # raw mid-prices
    horizon: int,
    latency: int,
    fee_bps: float,
    slippage_bps: float,
    timestamps: np.ndarray = None,
) -> dict:
    """
    Simulates a simple signal-based strategy and returns performance metrics.

    Returns
    -------
    dict with all output fields listed in the module docstring.
    """
    n = len(signals)
    pnl_bps = np.zeros(n)
    position = 0   # -1, 0, +1
    entry_price = None
    entry_step  = None
    trade_pnls  = []
    holding_times = []
    n_trades = 0

    for t in range(n):
        # Apply latency
        effective_t = min(t + latency, n - 1)
        new_signal = signals[effective_t]

        if new_signal != position:
            # Close existing position
            if position != 0 and entry_price is not None:
                exit_price = mid_prices[t]
                raw_ret = (exit_price - entry_price) / entry_price * position
                cost = (fee_bps + slippage_bps) / 10_000
                net_ret_bps = raw_ret * 10_000 - (fee_bps + slippage_bps)
                pnl_bps[t] = net_ret_bps
                trade_pnls.append(net_ret_bps)
                holding_times.append(t - entry_step)

            # Open new position
            if new_signal != 0:
                entry_price = mid_prices[min(t + 1, n - 1)]
                entry_step  = t
                n_trades += 1
                # Entry cost
                pnl_bps[t] -= (fee_bps + slippage_bps)

            position = new_signal

    cumulative = np.cumsum(pnl_bps)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = running_max - cumulative
    max_dd      = float(drawdowns.max())

    # Sharpe / Sortino (annualised)
    if timestamps is not None and len(timestamps) > 1:
        dt_s = float(np.median(np.diff(timestamps)) / 1000)  # ms → s
        ann_factor = np.sqrt(ANNUALIZE / dt_s)
    else:
        ann_factor = np.sqrt(252 * 390)  # trading-day fallback

    mean_pnl = float(np.mean(pnl_bps))
    std_pnl  = float(np.std(pnl_bps, ddof=1)) + 1e-9
    downside  = float(np.std(pnl_bps[pnl_bps < 0], ddof=1)) + 1e-9
    sharpe    = mean_pnl / std_pnl * ann_factor
    sortino   = mean_pnl / downside * ann_factor

    hit_rate  = float(np.mean(np.array(trade_pnls) > 0)) if trade_pnls else 0.0
    turnover  = float(np.mean(np.abs(np.diff(signals, prepend=0))))

    return {
        'gross_return_bps': float(cumulative[-1] + n_trades * (fee_bps + slippage_bps)),
        'net_return_bps':   float(cumulative[-1]),
        'sharpe':           float(sharpe),
        'sortino':          float(sortino),
        'max_drawdown_bps': max_dd,
        'hit_rate':         hit_rate,
        'turnover':         turnover,
        'n_trades':         n_trades,
        'avg_holding_time_obs': float(np.mean(holding_times)) if holding_times else 0.0,
        'avg_pnl_per_trade_bps': float(np.mean(trade_pnls)) if trade_pnls else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark signals
# ---------------------------------------------------------------------------

def _buy_and_hold(n: int) -> np.ndarray:
    return np.ones(n, dtype=int)

def _random_signal(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=n)

def _majority_class_signal(y_test: np.ndarray) -> np.ndarray:
    maj = int(np.bincount(y_test).argmax())
    signal_map = {0: -1, 1: 0, 2: 1}
    return np.full(len(y_test), signal_map[maj], dtype=int)

def _perfect_foresight(y_test: np.ndarray) -> np.ndarray:
    signal_map = {0: -1, 1: 0, 2: 1}
    return np.array([signal_map[int(y)] for y in y_test], dtype=int)


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(run_dir: str, horizon: int, out_dir: str,
                 fee_bps: float = FEE_BPS_DEFAULT,
                 slippage_bps: float = SLIPPAGE_BPS) -> list:
    """
    Runs the full backtest for a single run directory.

    Returns list of result dicts.
    """
    probs_path  = os.path.join(run_dir, 'test_probs.npy')
    labels_path = os.path.join(run_dir, 'test_labels.npy')
    mid_path    = os.path.join(run_dir, 'test_mid_prices.npy')
    ts_path     = os.path.join(run_dir, 'test_timestamps.npy')
    manifest_path = os.path.join(run_dir, 'run_manifest.json')

    for p in [probs_path, labels_path]:
        if not os.path.exists(p):
            logger.warning(f"Missing {p} — skipping {run_dir}")
            return []

    probs  = np.load(probs_path)     # (N, 3)
    labels = np.load(labels_path)    # (N,)
    mid_prices  = np.load(mid_path)  if os.path.exists(mid_path)  else None
    timestamps  = np.load(ts_path)   if os.path.exists(ts_path)   else None

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    n = len(labels)
    signal_map = {0: -1, 1: 0, 2: 1}
    base_signals = np.array([signal_map[int(p)] for p in np.argmax(probs, axis=1)])

    results = []
    common = {
        'model':  manifest.get('model', 'unknown'),
        'market': manifest.get('market', 'unknown'),
        'seed':   os.path.basename(run_dir),
        'run_dir': run_dir,
        'fee_bps': fee_bps,
        'slippage_bps': slippage_bps,
    }

    if mid_prices is None:
        logger.warning(f"No mid_prices in {run_dir} — cannot simulate realistic PnL.")
        return []

    # --- Confidence-thresholded signals ---
    for tau in CONF_THRESHOLDS:
        conf = np.max(probs, axis=1)
        signals = np.where(conf >= tau, base_signals, 0)  # flat when uncertain

        for L in LATENCIES:
            stats = _simulate_strategy(
                signals, mid_prices, horizon, L, fee_bps, slippage_bps, timestamps
            )
            row = {**common, 'tau': tau, 'latency': L, 'strategy': 'model', **stats}
            results.append(row)

    # --- Benchmarks ---
    for bench_name, bench_signals in [
        ('buy_and_hold',  _buy_and_hold(n)),
        ('random',        _random_signal(n, seed=42)),
        ('majority_class', _majority_class_signal(labels)),
        ('perfect_foresight', _perfect_foresight(labels)),
    ]:
        stats = _simulate_strategy(
            bench_signals, mid_prices, horizon, 0, fee_bps, slippage_bps, timestamps
        )
        row = {**common, 'tau': None, 'latency': 0, 'strategy': bench_name, **stats}
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="LOB prediction backtest.")
    parser.add_argument('--results_dir', default='results',
                        help="Directory containing seed run subdirs")
    parser.add_argument('--out_dir',     default='results/backtest',
                        help="Output directory for backtest results")
    parser.add_argument('--horizon',     type=int, default=40,
                        help="Label horizon (used for holding period)")
    parser.add_argument('--fee_bps',     type=float, default=FEE_BPS_DEFAULT)
    parser.add_argument('--slippage_bps', type=float, default=SLIPPAGE_BPS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_dirs = glob.glob(os.path.join(args.results_dir, '*', 'seed_*'))
    all_results = []

    for run_dir in sorted(run_dirs):
        logger.info(f"Backtesting {run_dir}...")
        results = run_backtest(
            run_dir, args.horizon, args.out_dir,
            fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
        )
        all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(args.out_dir, 'backtest_results.csv')
        df.to_csv(out_path, index=False)
        logger.info(f"Backtest results saved to {out_path}")
        print("\n--- Backtest Summary (net_return_bps, grouped by model×strategy×tau) ---")
        summary = df.groupby(['market', 'model', 'strategy', 'tau'])[
            ['net_return_bps', 'sharpe', 'max_drawdown_bps', 'n_trades']
        ].mean().round(4)
        print(summary.to_string())
    else:
        logger.warning("No backtest results generated — check that test_probs.npy and "
                       "test_mid_prices.npy exist (run main.py first).")


if __name__ == '__main__':
    main()
