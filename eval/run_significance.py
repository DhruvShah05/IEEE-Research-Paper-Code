"""
eval/run_significance.py — Runner that loads per-seed predictions and produces
a full significance report (fix 4.2).

For every market, for every pair of models, produces:
  (a) Paired t-test and Wilcoxon across the 5 seeds for Macro-F1, Bal-Acc, MCC.
  (b) McNemar on pooled test predictions for the top-2 models.
  (c) 95 % bootstrap CIs for each model's metrics.
  (d) One-sample test against chance for each model on crypto (is MCC > 0?
      is Bal-Acc > 1/3?).

Outputs:
  results/significance_report.csv    — machine-readable
  results/significance_table.tex     — LaTeX booktabs table with significance markers
  results/chance_tests.json          — one-sample chance results per model × market
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from eval.significance import (
    bootstrap_ci,
    mcnemar,
    one_sample_chance_test,
    paired_bootstrap,
    paired_ttest_across_seeds,
    wilcoxon_across_seeds,
)
from eval.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRICS_OF_INTEREST = ['macro_f1', 'balanced_accuracy', 'mcc']


def _load_run(run_dir: str) -> dict | None:
    """Load predictions, labels, and manifest for one seed run."""
    pred_path  = os.path.join(run_dir, 'test_predictions.npy')
    label_path = os.path.join(run_dir, 'test_labels.npy')
    manifest_path = os.path.join(run_dir, 'run_manifest.json')

    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        return None

    preds  = np.load(pred_path)
    labels = np.load(label_path)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    return {'preds': preds, 'labels': labels, 'manifest': manifest}


def _collect_runs(results_dir: str) -> dict:
    """
    Returns a nested dict:
      runs[market][model] = list of {'preds', 'labels', 'seed', 'manifest'}
    """
    runs = {}
    pattern = os.path.join(results_dir, '*', 'seed_*')
    for run_dir in sorted(glob.glob(pattern)):
        manifest_path = os.path.join(run_dir, 'run_manifest.json')
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)

        market = manifest.get('market', 'unknown')
        model  = manifest.get('model', 'unknown')
        seed   = os.path.basename(run_dir).replace('seed_', '')

        data = _load_run(run_dir)
        if data is None:
            continue
        data['seed'] = seed

        runs.setdefault(market, {}).setdefault(model, []).append(data)

    return runs


def main():
    parser = argparse.ArgumentParser(description="Significance tests across seeds.")
    parser.add_argument('--results_dir', default='results',
                        help="Directory containing run subdirs")
    parser.add_argument('--out_dir', default='results',
                        help="Directory to save significance outputs")
    parser.add_argument('--seed', type=int, default=42,
                        help="Bootstrap/RNG seed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = _collect_runs(args.results_dir)

    if not runs:
        logger.error("No runs found in %s. Run experiments first.", args.results_dir)
        return

    pairwise_rows = []
    chance_results = []
    ci_rows = []

    for market, model_runs in runs.items():
        models = list(model_runs.keys())
        logger.info(f"\n=== Market: {market} | Models: {models} ===")

        # ----- Per-model: seed-level metric vectors + bootstrap CIs -----
        model_seed_metrics = {}   # model → {metric → [seed0_val, seed1_val, ...]}
        model_pooled_preds = {}   # model → pooled predictions across seeds
        model_pooled_labels = {}  # model → pooled labels across seeds

        for model, seed_runs in model_runs.items():
            seed_metrics = {m: [] for m in METRICS_OF_INTEREST}
            all_preds, all_labels = [], []

            for run in seed_runs:
                preds  = run['preds']
                labels = run['labels']
                m_dict = compute_all_metrics(labels, preds)
                for m in METRICS_OF_INTEREST:
                    seed_metrics[m].append(m_dict[m])
                all_preds.append(preds)
                all_labels.append(labels)

            model_seed_metrics[model] = seed_metrics
            model_pooled_preds[model]  = np.concatenate(all_preds)
            model_pooled_labels[model] = np.concatenate(all_labels)

            # Bootstrap CIs on pooled predictions
            pooled_preds  = model_pooled_preds[model]
            pooled_labels = model_pooled_labels[model]
            for m in METRICS_OF_INTEREST:
                metric_fn = lambda yt, yp, _m=m: compute_all_metrics(yt, yp)[_m]
                pt, lo, hi = bootstrap_ci(pooled_labels, pooled_preds, metric_fn,
                                          seed=args.seed)
                ci_rows.append({
                    'market': market, 'model': model, 'metric': m,
                    'point_estimate': pt, 'ci_low': lo, 'ci_high': hi,
                })

            # One-sample chance tests (Reviewer 2 — mandatory for crypto)
            for m in ['balanced_accuracy', 'mcc']:
                res = one_sample_chance_test(
                    pooled_labels, pooled_preds, metric=m, seed=args.seed
                )
                res.update({'market': market, 'model': model})
                chance_results.append(res)
                marker = "✓ above chance" if res['is_above_chance'] else "✗ NOT above chance"
                logger.info(
                    f"  {model} {m}: {res['observed']:.4f} "
                    f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}] "
                    f"p={res['p_value_gt_chance']:.4f}  {marker}"
                )

        # ----- Pairwise tests across seeds -----
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                scores_a = model_seed_metrics[model_a]
                scores_b = model_seed_metrics[model_b]

                for m in METRICS_OF_INTEREST:
                    sa = np.array(scores_a[m])
                    sb = np.array(scores_b[m])

                    if len(sa) >= 2 and len(sb) >= 2:
                        t_stat, p_ttest = paired_ttest_across_seeds(sa, sb)
                        w_stat, p_wilcox = wilcoxon_across_seeds(sa, sb)
                    else:
                        t_stat = p_ttest = w_stat = p_wilcox = float('nan')

                    # Bootstrap on pooled
                    metric_fn = lambda yt, yp, _m=m: compute_all_metrics(yt, yp)[_m]
                    mean_diff, lo, hi, p_boot = paired_bootstrap(
                        model_pooled_labels[model_a],
                        model_pooled_preds[model_a],
                        model_pooled_preds[model_b],
                        metric_fn, seed=args.seed
                    )

                    pairwise_rows.append({
                        'market': market,
                        'model_a': model_a, 'model_b': model_b,
                        'metric': m,
                        'mean_a': float(np.mean(scores_a[m])),
                        'mean_b': float(np.mean(scores_b[m])),
                        'mean_diff_a_minus_b': mean_diff,
                        'boot_ci_low': lo, 'boot_ci_high': hi,
                        'p_bootstrap': p_boot,
                        't_stat': t_stat, 'p_ttest': p_ttest,
                        'w_stat': w_stat, 'p_wilcoxon': p_wilcox,
                        'sig_05': (p_boot < 0.05),
                    })

                # McNemar on pooled predictions for top-2
                p_mn = mcnemar(
                    model_pooled_labels[model_a],
                    model_pooled_preds[model_a],
                    model_pooled_preds[model_b],
                )
                pairwise_rows.append({
                    'market': market,
                    'model_a': model_a, 'model_b': model_b,
                    'metric': 'mcnemar', 'mean_a': None, 'mean_b': None,
                    'mean_diff_a_minus_b': None,
                    'boot_ci_low': None, 'boot_ci_high': None,
                    'p_bootstrap': None,
                    't_stat': None, 'p_ttest': p_mn,
                    'w_stat': None, 'p_wilcoxon': None,
                    'sig_05': (p_mn < 0.05),
                })

    # --- Save outputs ---
    if pairwise_rows:
        sig_df = pd.DataFrame(pairwise_rows)
        csv_path = os.path.join(args.out_dir, 'significance_report.csv')
        sig_df.to_csv(csv_path, index=False)
        logger.info(f"Significance report saved to {csv_path}")

        # LaTeX table (booktabs) with significance markers
        _save_latex_significance(sig_df, args.out_dir)

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_path = os.path.join(args.out_dir, 'bootstrap_ci.csv')
        ci_df.to_csv(ci_path, index=False)
        logger.info(f"Bootstrap CIs saved to {ci_path}")

    if chance_results:
        chance_path = os.path.join(args.out_dir, 'chance_tests.json')
        with open(chance_path, 'w') as f:
            json.dump(chance_results, f, indent=4)
        logger.info(f"Chance tests saved to {chance_path}")


def _save_latex_significance(df: pd.DataFrame, out_dir: str) -> None:
    """Saves a compact LaTeX booktabs significance table."""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Pairwise significance tests (paired bootstrap p-values). '
        r'$^*$ $p<0.05$, $^{**}$ $p<0.01$, $^{***}$ $p<0.001$.}',
        r'\label{tab:significance}',
        r'\begin{tabular}{llllrrr}',
        r'\toprule',
        r'Market & Model A & Model B & Metric & $\Delta$ & 95\% CI & $p$ \\',
        r'\midrule',
    ]

    for _, row in df.iterrows():
        if row['metric'] == 'mcnemar' or row['mean_diff_a_minus_b'] is None:
            continue
        p = row['p_bootstrap']
        sig = (
            r'$^{***}$' if p < 0.001 else
            r'$^{**}$'  if p < 0.01  else
            r'$^*$'     if p < 0.05  else ''
        )
        diff = row['mean_diff_a_minus_b']
        lo   = row['boot_ci_low']
        hi   = row['boot_ci_high']
        line = (
            f"{row['market']} & {row['model_a']} & {row['model_b']} & "
            f"{row['metric']} & "
            f"{diff:+.4f} & [{lo:.4f}, {hi:.4f}] & "
            f"{p:.4f}{sig} \\\\"
        )
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    tex_path = os.path.join(out_dir, 'significance_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX significance table saved to {tex_path}")


if __name__ == '__main__':
    main()
