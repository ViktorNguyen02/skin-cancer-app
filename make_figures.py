"""
make_figures.py — Publication figures for the paper.

Renders four figures straight from the saved models, reusing the EXACT leak-free
split and prediction pipeline from stats_analysis.py (one source of truth), so the
figures agree with the numbers in documentation/statistical_analysis.md:

  Figure 1  ROC curves for all four models, with a bootstrap 95% CI band on the
            from-scratch champion and the val-selected operating point marked.
  Figure 2  Reliability diagram — from-scratch champion vs transfer baseline
            (visualizes the calibration gap: ECE ~0.04 vs ~0.10).
  Figure 3  Ablation bar chart (Runs 0–4), accepted vs rejected, 0.90 target line.
  Figure 4  Per-diagnosis mean P(Concerning) for the champion, Concerning vs Benign.

Loads the four .h5 models and predicts once (~5 min on the M4). Outputs 300-dpi
PNGs to figures/.

Usage:
    python make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless — save to file, no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
import tensorflow as tf

# Reuse the exact split + preprocessing + threshold logic from stats_analysis.py
from stats_analysis import (load_split, fit_metadata_encoder, transform_metadata,
                            predict_probs, val_margin_threshold, CONCERNING_IDX,
                            CLINICAL_GROUPS)

script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(script_dir, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
RNG        = np.random.default_rng(42)

# Models to score (label → file). Missing files are skipped with a warning.
MODELS = {
    'From-scratch, image-only (Run 1)':      'best_model_run1_calib.h5',
    'From-scratch, + metadata (champion)':   'best_model_run4_meta.h5',
    'EfficientNet-B0, image-only (transfer)':'best_model_transfer_effb0.h5',
    'EfficientNet-B0, + metadata (transfer)':'best_model_transfer_effb0_meta.h5',
}
CHAMPION = 'From-scratch, + metadata (champion)'
TRANSFER = 'EfficientNet-B0, image-only (transfer)'

COLORS = {
    'From-scratch, image-only (Run 1)':       '#888888',
    'From-scratch, + metadata (champion)':    '#c0392b',
    'EfficientNet-B0, image-only (transfer)': '#2980b9',
    'EfficientNet-B0, + metadata (transfer)': '#27ae60',
}


# ──────────────────────────────────────────────────────────────────────
# Calibration helpers (10-bin ECE + reliability curve on raw probabilities)
# ──────────────────────────────────────────────────────────────────────
def reliability(probs2, y_int, n_bins=10):
    conf = probs2.max(axis=1); pred = probs2.argmax(axis=1)
    acc  = (pred == y_int).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ece = [], [], 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return np.array(xs), np.array(ys), ece


# ──────────────────────────────────────────────────────────────────────
# Predict all models once
# ──────────────────────────────────────────────────────────────────────
def score_models():
    train_df, val_df, test_df = load_split()
    enc = fit_metadata_encoder(train_df)
    meta_val  = transform_metadata(val_df,  enc)
    meta_test = transform_metadata(test_df, enc)
    val_true  = val_df['group_label'].to_numpy()
    test_true = test_df['group_label'].to_numpy()

    out = {}
    for label, fname in MODELS.items():
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  {fname} not found — skipping '{label}'")
            continue
        print(f"  scoring {label} …")
        model = tf.keras.models.load_model(path, compile=False)
        vp = predict_probs(model, val_df,  meta_val)
        tp = predict_probs(model, test_df, meta_test)
        thr = val_margin_threshold(val_true, vp)
        out[label] = {'test_probs': tp, 'thr': thr,
                      'auc': roc_auc_score(test_true, tp)}
    return test_true, test_df, out


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — ROC curves + champion CI band + operating points
# ──────────────────────────────────────────────────────────────────────
def fig_roc(test_true, scored, n_boot=1000):
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 101)

    for label, d in scored.items():
        fpr, tpr, _ = roc_curve(test_true, d['test_probs'])
        ax.plot(fpr, tpr, color=COLORS[label], lw=2,
                label=f"{label} (AUC {d['auc']:.3f})")
        # operating point at the val-selected threshold
        pred = (d['test_probs'] >= d['thr']).astype(int)
        sens = recall_score(test_true, pred)
        spec = recall_score(1 - test_true, 1 - pred)
        ax.scatter([1 - spec], [sens], color=COLORS[label], s=40, zorder=5,
                   edgecolor='white', linewidth=0.8)

        # bootstrap 95% CI band for the champion only (keeps the plot readable)
        if label == CHAMPION:
            n = len(test_true); tprs = []
            for _ in range(n_boot):
                idx = RNG.integers(0, n, n)
                if test_true[idx].min() == test_true[idx].max():
                    continue
                f, t, _ = roc_curve(test_true[idx], d['test_probs'][idx])
                tprs.append(np.interp(grid, f, t))
            tprs = np.array(tprs)
            lo, hi = np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)
            ax.fill_between(grid, lo, hi, color=COLORS[label], alpha=0.15,
                            label='Champion 95% CI band')

    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC — Concerning-vs-Benign, leak-free test set (n = 2,001)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    fig.tight_layout()
    _save(fig, 'fig1_roc.png')


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — Reliability diagram (champion vs transfer)
# ──────────────────────────────────────────────────────────────────────
def fig_reliability(test_true, scored):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    for label, style in [(CHAMPION, 'o-'), (TRANSFER, 's-')]:
        if label not in scored:
            continue
        p = scored[label]['test_probs']
        probs2 = np.column_stack([1 - p, p])          # binary → 2-class
        xs, ys, ece = reliability(probs2, test_true)
        ax.plot(xs, ys, style, color=COLORS[label], lw=2,
                label=f"{label.split(',')[0]} (ECE {ece:.3f})")
    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title('Reliability Diagram — from-scratch vs transfer (raw probabilities)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, 'fig2_reliability.png')


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — Ablation bar chart (Runs 0–4). Values from the ablation table.
# ──────────────────────────────────────────────────────────────────────
def fig_ablation():
    runs   = ['Run 0\nbaseline', 'Run 1\n+calibration', 'Run 2\n+colour const.',
              'Run 3\n+MixUp', 'Run 4\n+metadata']
    aucs   = [0.883, 0.895, 0.871, 0.8815, 0.9047]
    status = ['kept', 'kept', 'rejected', 'rejected', 'champion']
    cmap   = {'kept': '#7f8c8d', 'rejected': '#c0392b', 'champion': '#27ae60'}
    colors = [cmap[s] for s in status]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(runs, aucs, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0.90, ls='--', color='black', lw=1)
    ax.text(4.4, 0.9015, 'AUC 0.90 target', ha='right', va='bottom', fontsize=8)
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.0015, f'{a:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Test AUC')
    ax.set_ylim(0.85, 0.915)
    ax.set_title('Cumulative Ablation — from-scratch SE-residual CNN')
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ['kept', 'rejected', 'champion']]
    ax.legend(handles, ['kept (AUC improved)', 'rejected (AUC regressed)', 'champion'],
              fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    _save(fig, 'fig3_ablation.png')


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — Per-diagnosis mean P(Concerning) for the champion
# ──────────────────────────────────────────────────────────────────────
def fig_per_diagnosis(test_df, scored):
    if CHAMPION not in scored:
        print("  ⚠️  champion not scored — skipping per-diagnosis figure")
        return
    p = scored[CHAMPION]['test_probs']
    thr = scored[CHAMPION]['thr']
    order = ['mel', 'bcc', 'akiec', 'nv', 'bkl', 'df', 'vasc']   # Concerning first
    dx = test_df['dx'].to_numpy()
    means = [p[dx == d].mean() for d in order]
    colors = ['#c0392b' if CLINICAL_GROUPS[d] == 'Concerning' else '#27ae60' for d in order]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(order, means, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(thr, ls='--', color='black', lw=1)
    ax.text(6.4, thr + 0.005, f'operating threshold ({thr:.3f})', ha='right', va='bottom', fontsize=8)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f'{m:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean predicted P(Concerning)')
    ax.set_title('Per-diagnosis mean P(Concerning) — champion (released model)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['#c0392b', '#27ae60']]
    ax.legend(handles, ['Concerning (should be high)', 'Benign (should be low)'],
              fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, max(means) + 0.08)
    fig.tight_layout()
    _save(fig, 'fig4_per_diagnosis.png')


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ wrote {path}")


def main():
    print("Scoring models (reusing the leak-free split from stats_analysis.py) …")
    test_true, test_df, scored = score_models()
    if not scored:
        print("No models found — nothing to plot.")
        return
    print("\nRendering figures …")
    fig_roc(test_true, scored)
    fig_reliability(test_true, scored)
    fig_ablation()
    fig_per_diagnosis(test_df, scored)
    print(f"\n✅ All figures written to {FIG_DIR}/")


if __name__ == '__main__':
    main()
