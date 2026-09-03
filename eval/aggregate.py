"""
eval/aggregate.py — Aggregates per-seed run metrics into summary tables.

Changes (4.3):
  - Emits LaTeX tables directly (booktabs) with bold best-per-column and
    baseline rows.
  - Keeps confusion matrices: aggregated (summed across seeds) and saved as CSV.
  - Uses ddof=1 explicitly (std over 5 seeds).
  - Aggregates ablation, multi-asset, horizon-sweep, and backtest results.
"""

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_all_runs(results_dir: str) -> pd.DataFrame:
    """
    Reads all metrics.json files across different runs and seeds,
    groups by (market, model) and returns a flat DataFrame.
    """
    all_runs = []
    metrics_files = glob.glob(os.path.join(results_dir, '*', 'seed_*', 'metrics.json'))

    for f in metrics_files:
        parts = f.split(os.sep)
        seed_str = parts[-2]  # e.g. seed_0
        manifest_file = os.path.join(os.path.dirname(f), 'run_manifest.json')

        market, model = 'unknown', 'unknown'
        if os.path.exists(manifest_file):
            with open(manifest_file) as mf:
                manifest = json.load(mf)
                market = manifest.get('market', 'unknown')
                model  = manifest.get('model', 'unknown')

        with open(f) as mf:
            metrics = json.load(mf)

        # Aggregate confusion matrices separately — keep for later
        cm = metrics.pop('confusion_matrix', None)

        record = {'market': market, 'model': model, 'seed': seed_str, **metrics}
        record['_cm'] = cm
        all_runs.append(record)

    if not all_runs:
        logger.warning("No metrics found to aggregate.")
        return pd.DataFrame()

    return pd.DataFrame(all_runs)


def _format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format as 'mean ± std' string."""
    fmt = f'{{:.{decimals}f}}'
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def _bold_best(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return the same series with the best value wrapped in \\textbf{}."""
    def parse_mean(s):
        try:
            return float(str(s).split('±')[0].strip())
        except (ValueError, AttributeError):
            return float('-inf') if higher_is_better else float('inf')

    means = series.map(parse_mean)
    best_idx = means.idxmax() if higher_is_better else means.idxmin()
    out = series.copy()
    out[best_idx] = r'\textbf{' + str(out[best_idx]) + '}'
    return out


def aggregate_results(results_dir: str = 'results') -> pd.DataFrame:
    """
    Reads all metrics.json files, groups by (market, model), computes
    mean ± std (ddof=1) for tables.

    Returns the aggregated DataFrame and also saves:
      - results/aggregated_metrics.csv
      - results/aggregated_metrics.tex  (LaTeX booktabs)
      - results/confusion_matrices.csv  (per-model summed CM)
    """
    logger.info("Aggregating results...")
    df = _load_all_runs(results_dir)
    if df.empty:
        return df

    numeric_cols = [
        c for c in df.columns
        if c not in ['market', 'model', 'seed', '_cm']
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # --- Aggregate confusion matrices (sum across seeds) ---
    _aggregate_confusion_matrices(df, results_dir)

    # --- Mean ± std (ddof=1) per (market, model) ---
    groups = df.groupby(['market', 'model'])

    rows = []
    for (market, model), grp in groups:
        row = {'market': market, 'model': model, 'n_seeds': len(grp)}
        for col in numeric_cols:
            vals = grp[col].dropna().values
            if len(vals) > 0:
                row[col] = _format_mean_std(
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)),  # ddof=1 explicitly (4.3)
                )
        rows.append(row)

    final_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(results_dir, 'aggregated_metrics.csv')
    final_df.to_csv(csv_path, index=False)
    logger.info(f"Aggregated metrics saved to {csv_path}")

    # Save LaTeX
    _save_latex_table(final_df, results_dir)

    print("\n--- Final Aggregated Results ---")
    print(final_df.to_string(index=False))
    return final_df


def _aggregate_confusion_matrices(df: pd.DataFrame, results_dir: str) -> None:
    """Sum confusion matrices across seeds per (market, model) and save."""
    cm_rows = []
    for (market, model), grp in df.groupby(['market', 'model']):
        cms = [np.array(r) for r in grp['_cm'].dropna() if r is not None]
        if cms:
            summed = np.sum(cms, axis=0)
            for i in range(3):
                for j in range(3):
                    cm_rows.append({
                        'market': market, 'model': model,
                        'true': i, 'pred': j,
                        'count': int(summed[i, j]),
                    })

    if cm_rows:
        cm_df = pd.DataFrame(cm_rows)
        cm_path = os.path.join(results_dir, 'confusion_matrices.csv')
        cm_df.to_csv(cm_path, index=False)
        logger.info(f"Aggregated confusion matrices saved to {cm_path}")


def _save_latex_table(df: pd.DataFrame, results_dir: str) -> None:
    """
    Saves a LaTeX booktabs table with bold best-per-column.
    Only includes headline metrics to keep the table compact.
    """
    # Select columns for the paper table
    headline_metrics = [
        'accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'kappa',
        'f1_down', 'f1_stationary', 'f1_up',
        'lift_over_majority',
    ]
    available = ['market', 'model'] + [c for c in headline_metrics if c in df.columns]
    table = df[available].copy() if all(c in df.columns for c in ['market', 'model']) else df.copy()

    # Bold best per column per market
    tex_table = table.copy()
    higher_better = {'accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'kappa',
                     'f1_down', 'f1_stationary', 'f1_up', 'lift_over_majority'}
    for market, grp in tex_table.groupby('market'):
        for col in headline_metrics:
            if col in tex_table.columns:
                idx = grp.index
                tex_table.loc[idx, col] = _bold_best(
                    tex_table.loc[idx, col],
                    higher_is_better=(col in higher_better)
                )

    # Build LaTeX
    col_fmt = 'l' * (len(tex_table.columns))
    header = ' & '.join(tex_table.columns.tolist()) + r' \\'
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Aggregated results (mean $\pm$ std over 5 seeds, ddof=1). '
        r'Bold = best per market per metric. '
        r'Convention: 0=Down, 1=Stationary, 2=Up.}',
        r'\label{tab:main_results}',
        r'\begin{tabular}{' + col_fmt + '}',
        r'\toprule',
        header,
        r'\midrule',
    ]

    prev_market = None
    for _, row in tex_table.iterrows():
        if prev_market is not None and row['market'] != prev_market:
            lines.append(r'\midrule')
        line = ' & '.join(str(v) for v in row.values) + r' \\'
        lines.append(line)
        prev_market = row['market']

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table*}',
    ]

    tex_path = os.path.join(results_dir, 'aggregated_metrics.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX table saved to {tex_path}")


if __name__ == '__main__':
    aggregate_results()
