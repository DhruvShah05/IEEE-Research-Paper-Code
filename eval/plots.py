"""
eval/plots.py — All plotting code for the paper's figures (fix 4.4).

Previously matplotlib was in requirements but there was no plotting code.
This module produces:

  1. Confusion matrices per market × best model
  2. Horizon × threshold heatmaps (class balance and Macro-F1)
  3. Ablation bar charts with CIs
  4. Equity curves and drawdown plots (from backtest results)
  5. Calibration curves
  6. Feature importance bars
  7. Per-asset comparison bar charts
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

CLASS_NAMES = ['Down', 'Stationary', 'Up']


# ---------------------------------------------------------------------------
# 1. Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(cm_csv: str, out_dir: str) -> None:
    """
    Plot confusion matrices per (market, model) from the aggregated CSV.

    Parameters
    ----------
    cm_csv  : path to results/confusion_matrices.csv (from aggregate.py)
    out_dir : output directory for figures
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(cm_csv)

    for (market, model), grp in df.groupby(['market', 'model']):
        cm = np.zeros((3, 3), dtype=int)
        for _, row in grp.iterrows():
            cm[int(row['true']), int(row['pred'])] = int(row['count'])

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{market} — {model}\n(summed over seeds)')

        total = cm.sum()
        for i in range(3):
            for j in range(3):
                pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
                ax.text(j, i, f'{cm[i,j]:,}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color='white' if cm[i, j] > total * 0.15 else 'black',
                        fontsize=9)

        fname = os.path.join(out_dir, f'cm_{market}_{model}.pdf')
        fig.tight_layout(); fig.savefig(fname); plt.close(fig)
        logger.info(f"Confusion matrix saved: {fname}")


# ---------------------------------------------------------------------------
# 2. Horizon × Threshold Heatmaps
# ---------------------------------------------------------------------------

def plot_threshold_heatmaps(balance_csv: str, f1_csv: str, out_dir: str) -> None:
    """
    Plots horizon × threshold heatmaps for class balance and Macro-F1.

    Parameters
    ----------
    balance_csv : threshold_sweep_balance.csv
    f1_csv      : threshold_sweep_f1.csv
    out_dir     : output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(balance_csv):
        df_bal = pd.read_csv(balance_csv)
        horizons   = sorted(df_bal['horizon_events'].unique())
        thresholds = sorted(df_bal['threshold'].unique())

        for col, label in [
            ('pct_stationary', '% Stationary'),
            ('pct_down', '% Down'),
            ('pct_up',   '% Up'),
        ]:
            mat = df_bal.pivot(index='threshold', columns='horizon_events', values=col)
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(mat.values, aspect='auto', cmap='RdYlGn_r',
                           vmin=0, vmax=100)
            fig.colorbar(im, ax=ax, label=label)
            ax.set_xticks(range(len(horizons))); ax.set_xticklabels(horizons)
            ax.set_yticks(range(len(thresholds))); ax.set_yticklabels(
                [f'{t:.4f}' for t in thresholds])
            ax.set_xlabel('Horizon (events)'); ax.set_ylabel('Threshold')
            ax.set_title(f'Label Balance Heatmap — {label}')
            fname = os.path.join(out_dir, f'heatmap_balance_{col}.pdf')
            fig.tight_layout(); fig.savefig(fname); plt.close(fig)
            logger.info(f"Heatmap saved: {fname}")

    if os.path.exists(f1_csv):
        df_f1 = pd.read_csv(f1_csv)
        for model_col in ['rf_val_macro_f1', 'xgb_val_macro_f1']:
            if model_col not in df_f1.columns:
                continue
            mat = df_f1.pivot(index='threshold', columns='horizon_events', values=model_col)
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(mat.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label='Val Macro-F1')
            ax.set_xticks(range(len(mat.columns)));
            ax.set_xticklabels(mat.columns.tolist())
            ax.set_yticks(range(len(mat.index)));
            ax.set_yticklabels([f'{t:.4f}' for t in mat.index])
            ax.set_xlabel('Horizon (events)'); ax.set_ylabel('Threshold')
            ax.set_title(f'{model_col.replace("_", " ").upper()} — Val Macro-F1')
            fname = os.path.join(out_dir, f'heatmap_f1_{model_col}.pdf')
            fig.tight_layout(); fig.savefig(fname); plt.close(fig)
            logger.info(f"Heatmap saved: {fname}")


# ---------------------------------------------------------------------------
# 3. Ablation Bar Charts with CIs
# ---------------------------------------------------------------------------

def plot_ablation_bars(ci_csv: str, out_dir: str, market: str = 'crypto',
                       metric: str = 'macro_f1') -> None:
    """
    Bar chart of model Macro-F1 with 95% bootstrap CIs.

    Parameters
    ----------
    ci_csv  : results/bootstrap_ci.csv (from run_significance.py)
    out_dir : output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(ci_csv):
        logger.warning(f"CI file not found: {ci_csv}")
        return

    df = pd.read_csv(ci_csv)
    df = df[(df['market'] == market) & (df['metric'] == metric)]
    if df.empty:
        return

    df = df.sort_values('point_estimate', ascending=False)
    x  = range(len(df))
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 5))
    ax.bar(x, df['point_estimate'], color='steelblue', alpha=0.85, zorder=2)
    ax.errorbar(
        x, df['point_estimate'],
        yerr=[
            df['point_estimate'] - df['ci_low'],
            df['ci_high'] - df['point_estimate'],
        ],
        fmt='none', color='black', capsize=5, zorder=3
    )
    ax.axhline(1 / 3, color='red', linestyle='--', label='Chance (1/3)', zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(df['model'], rotation=30, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{market.upper()} — {metric.replace("_", " ").title()} with 95% CI')
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    fname = os.path.join(out_dir, f'ablation_bars_{market}_{metric}.pdf')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    logger.info(f"Ablation bar chart saved: {fname}")


# ---------------------------------------------------------------------------
# 4. Equity Curves and Drawdown Plots
# ---------------------------------------------------------------------------

def plot_equity_curve(backtest_csv: str, out_dir: str) -> None:
    """
    Plots equity curve and maximum drawdown from backtest results.

    Parameters
    ----------
    backtest_csv : output of experiments/backtest.py with columns
                   [step, cumulative_pnl, drawdown, model, seed]
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(backtest_csv):
        logger.warning(f"Backtest CSV not found: {backtest_csv}")
        return

    df = pd.read_csv(backtest_csv)

    for (model, seed), grp in df.groupby(['model', 'seed']):
        fig = plt.figure(figsize=(10, 6))
        gs  = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.plot(grp['step'], grp['cumulative_pnl'], label='Cumulative PnL (bps)')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Cumulative PnL (bps)'); ax1.legend()
        ax1.set_title(f'{model} — seed {seed}')
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2.fill_between(grp['step'], grp['drawdown'], 0, alpha=0.4, color='red',
                         label='Drawdown')
        ax2.set_ylabel('Drawdown (bps)'); ax2.set_xlabel('Step'); ax2.legend()

        fname = os.path.join(out_dir, f'equity_{model}_seed{seed}.pdf')
        fig.savefig(fname); plt.close(fig)
        logger.info(f"Equity curve saved: {fname}")


# ---------------------------------------------------------------------------
# 5. Calibration Curves
# ---------------------------------------------------------------------------

def plot_calibration_curves(probs_npy: str, labels_npy: str, model: str,
                             market: str, out_dir: str) -> None:
    """
    Plots reliability diagrams (calibration curves) per class.

    Parameters
    ----------
    probs_npy  : path to test_probs.npy  (shape N×3)
    labels_npy : path to test_labels.npy (shape N,)
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(probs_npy) or not os.path.exists(labels_npy):
        logger.warning("Probs or labels file not found for calibration plot.")
        return

    probs  = np.load(probs_npy)
    labels = np.load(labels_npy)
    n_bins = 10

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for c, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        prob_c  = probs[:, c]
        label_c = (labels == c).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_means, bin_fracs = [], []
        for i in range(n_bins):
            mask = (prob_c >= bins[i]) & (prob_c < bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(prob_c[mask].mean())
                bin_fracs.append(label_c[mask].mean())

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(bin_means, bin_fracs, 'o-', label=f'Class {name}')
        ax.set_xlabel('Mean predicted probability')
        ax.set_title(f'Class: {name}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Fraction of positives')
    fig.suptitle(f'Calibration Curves — {market} {model}')
    fname = os.path.join(out_dir, f'calibration_{market}_{model}.pdf')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    logger.info(f"Calibration curve saved: {fname}")


# ---------------------------------------------------------------------------
# 6. Feature Importance Bars
# ---------------------------------------------------------------------------

def plot_feature_importances(fi_json: str, model: str, market: str,
                              out_dir: str, top_k: int = 20) -> None:
    """
    Plots top-k gain-based feature importances from feature_importances.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(fi_json):
        logger.warning(f"Feature importance file not found: {fi_json}")
        return

    with open(fi_json) as f:
        fi_data = json.load(f)

    gain = fi_data.get('gain', fi_data.get('mdi', {}))
    if not gain:
        return

    items = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names, scores = zip(*items)

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    y = range(len(names))
    ax.barh(y, scores, color='teal', alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (gain)')
    ax.set_title(f'Top-{top_k} Feature Importances — {market} {model}')
    fname = os.path.join(out_dir, f'feature_importance_{market}_{model}.pdf')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    logger.info(f"Feature importance plot saved: {fname}")


# ---------------------------------------------------------------------------
# 7. Per-Asset Comparison
# ---------------------------------------------------------------------------

def plot_per_asset_comparison(ci_csv: str, out_dir: str,
                               metric: str = 'macro_f1') -> None:
    """
    Bar chart comparing model rankings across assets/markets.

    Parameters
    ----------
    ci_csv  : results/bootstrap_ci.csv from run_significance.py
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(ci_csv):
        logger.warning(f"CI file not found: {ci_csv}")
        return

    df = pd.read_csv(ci_csv)
    df = df[df['metric'] == metric]
    if df.empty:
        return

    markets = df['market'].unique()
    models  = df['model'].unique()

    x = np.arange(len(models))
    width = 0.8 / len(markets)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(markets)))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    for i, (market, color) in enumerate(zip(markets, colors)):
        sub = df[df['market'] == market].set_index('model')
        vals = [sub.loc[m, 'point_estimate'] if m in sub.index else 0 for m in models]
        lo   = [sub.loc[m, 'ci_low']         if m in sub.index else 0 for m in models]
        hi   = [sub.loc[m, 'ci_high']        if m in sub.index else 0 for m in models]
        pos  = x + i * width
        ax.bar(pos, vals, width * 0.9, label=market, color=color, alpha=0.85)
        ax.errorbar(pos, vals,
                    yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                    fmt='none', color='black', capsize=3)

    ax.set_xticks(x + width * (len(markets) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Per-Asset Comparison — {metric.replace("_", " ").title()}')
    ax.axhline(1 / 3, color='red', linestyle='--', linewidth=0.8, label='Chance')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fname = os.path.join(out_dir, f'per_asset_{metric}.pdf')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    logger.info(f"Per-asset comparison plot saved: {fname}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--out_dir',     default='results/figures')
    args = parser.parse_args()

    rd  = args.results_dir
    out = args.out_dir

    plot_confusion_matrices(
        os.path.join(rd, 'confusion_matrices.csv'), out
    )
    plot_threshold_heatmaps(
        os.path.join(rd, 'threshold_sweep_balance.csv'),
        os.path.join(rd, 'threshold_sweep_f1.csv'),
        out
    )
    plot_ablation_bars(os.path.join(rd, 'bootstrap_ci.csv'), out, 'crypto')
    plot_ablation_bars(os.path.join(rd, 'bootstrap_ci.csv'), out, 'fi2010')
    plot_per_asset_comparison(os.path.join(rd, 'bootstrap_ci.csv'), out)

    logger.info("All plots generated.")
