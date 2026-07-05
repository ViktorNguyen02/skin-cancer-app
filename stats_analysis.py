"""
stats_analysis.py — Statistical significance for the paper (Tier 1).

Answers the two questions a reviewer will ask first:
  1. "What is the uncertainty on your headline numbers?"  → 95% confidence
     intervals on AUC (analytic DeLong + bootstrap) and on Sensitivity /
     Specificity (bootstrap) at the locked operating point.
  2. "Is the metadata model SIGNIFICANTLY better than image-only?"  → DeLong's
     paired test for two correlated ROC AUCs on the SAME test set, with a p-value.

Runs on the already-trained models in minutes — no retraining. Both models are
scored on the identical leak-free test split (random_state=42), so the paired
DeLong test is valid.

If a `multiseed/` directory of per-seed predictions exists (from
multiseed_train.py), this script also aggregates them into mean ± std — the
multi-seed evidence that a single run cannot provide.

Usage:
    python stats_analysis.py
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1 = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2 = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')

# Models to evaluate. Both must use plain rescale=1./255 (Run 2 colour constancy
# is excluded — different preprocessing + rejected anyway). The two-input Run 4
# model is auto-detected via len(model.inputs).
MODELS = {
    'Run1_image_only': 'best_model_run1_calib.h5',    # from-scratch image-only champion
    'Run4_metadata':   'best_model_run4_meta.h5',      # from-scratch metadata-fusion champion
    'Transfer_effb0':  'best_model_transfer_effb0.h5', # EfficientNet-B0 transfer baseline (image-only)
}
# Paired DeLong significance tests (baseline, candidate). Each runs on the same test set.
#   1. metadata effect:    from-scratch image-only  vs  from-scratch metadata
#   2. pretraining effect: from-scratch image-only  vs  EfficientNet-B0 transfer (fair: same inputs)
# Skipped automatically if either model file is missing.
DELONG_PAIRS = [
    ('Run1_image_only', 'Run4_metadata'),
    ('Run1_image_only', 'Transfer_effb0'),
]

GROUP_ORDER        = ['Benign', 'Concerning']
CONCERNING_IDX     = 1
IMG_SIZE           = 224
BATCH_SIZE         = 32
TARGET_SENS        = 0.90    # clinical sensitivity floor (on TEST)
VAL_SELECTION_SENS = 0.94    # val target with safety margin (matches model.py)
N_BOOT             = 2000    # bootstrap resamples
RNG_SEED           = 42
rng = np.random.default_rng(RNG_SEED)

CLINICAL_GROUPS = {
    'nv': 'Benign', 'bkl': 'Benign', 'df': 'Benign', 'vasc': 'Benign',
    'mel': 'Concerning', 'bcc': 'Concerning', 'akiec': 'Concerning',
}


# ══════════════════════════════════════════════════════════════════════════════
# Fast DeLong (Sun & Xu, 2014) — covariance of correlated AUCs on the same data.
# Implementation follows the widely-used reference (compute_midrank / fastDeLong).
# ══════════════════════════════════════════════════════════════════════════════
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """predictions_sorted_transposed: [k, n] with the m positives sorted first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_variance(y_true, probs):
    """AUC and its variance for a single model (analytic DeLong)."""
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    sorted_t = probs[np.newaxis, order]
    aucs, cov = fastDeLong(sorted_t, label_1_count)
    return float(aucs[0]), float(cov)


def delong_paired_test(y_true, probs_a, probs_b):
    """Two-sided DeLong test for AUC(b) - AUC(a) on the SAME samples."""
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    sorted_t = np.vstack((probs_a, probs_b))[:, order]
    aucs, cov = fastDeLong(sorted_t, label_1_count)
    l = np.array([[1.0, -1.0]])
    var_diff = float(l @ cov @ l.T)
    z = float((aucs[0] - aucs[1]) / np.sqrt(var_diff))
    p = float(2 * scipy.stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


# ══════════════════════════════════════════════════════════════════════════════
# Data — reproduce the exact leak-free split (random_state=42)
# ══════════════════════════════════════════════════════════════════════════════
def load_split():
    df = pd.read_csv(metadata_path)
    df['group'] = df['dx'].map(CLINICAL_GROUPS)
    df['group_label'] = LabelEncoder().fit_transform(df['group'])

    def find_path(image_id):
        for base in (images_path_1, images_path_2):
            p = os.path.join(base, image_id + '.jpg')
            if os.path.exists(p):
                return p
        raise FileNotFoundError(image_id)
    df['path'] = df['image_id'].map(find_path)

    lesion_df = df.drop_duplicates(subset='lesion_id')[['lesion_id', 'group_label']]
    trainval_ids, test_ids = train_test_split(
        lesion_df['lesion_id'], test_size=0.2,
        stratify=lesion_df['group_label'], random_state=42)
    trainval = lesion_df[lesion_df['lesion_id'].isin(trainval_ids)]
    train_ids, val_ids = train_test_split(
        trainval['lesion_id'], test_size=0.15,
        stratify=trainval['group_label'], random_state=42)

    train_df = df[df['lesion_id'].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df['lesion_id'].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df['lesion_id'].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


# Metadata encoder (fit on TRAIN only — identical to model.py)
def fit_metadata_encoder(train_df):
    age_median = float(train_df['age'].median())
    age_filled = train_df['age'].fillna(age_median)
    return {
        'age_median': age_median,
        'age_mean':   float(age_filled.mean()),
        'age_std':    float(age_filled.std() + 1e-6),
        'sex_cats':   sorted(train_df['sex'].dropna().unique().tolist()),
        'loc_cats':   sorted(train_df['localization'].dropna().unique().tolist()),
    }


def transform_metadata(df, enc):
    age   = df['age'].fillna(enc['age_median'])
    age_z = ((age - enc['age_mean']) / enc['age_std']).to_numpy(dtype='float32').reshape(-1, 1)
    sex_oh = np.stack([(df['sex'] == c).to_numpy(dtype='float32') for c in enc['sex_cats']], axis=1)
    loc_oh = np.stack([(df['localization'] == c).to_numpy(dtype='float32') for c in enc['loc_cats']], axis=1)
    return np.concatenate([age_z, sex_oh, loc_oh], axis=1).astype('float32')


def predict_probs(model, df, meta, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Batched P(Concerning). Auto-detects single- vs two-input models."""
    n, paths = len(df), df['path'].tolist()
    two_input = len(model.inputs) == 2
    probs = np.zeros((n, len(GROUP_ORDER)), dtype='float32')
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        X = np.zeros((e - s, img_size, img_size, 3), dtype='float32')
        for j, idx in enumerate(range(s, e)):
            X[j] = img_to_array(load_img(paths[idx], target_size=(img_size, img_size))) / 255.0
        probs[s:e] = model.predict([X, meta[s:e]], verbose=0) if two_input \
            else model.predict(X, verbose=0)
    return probs[:, CONCERNING_IDX]


# ══════════════════════════════════════════════════════════════════════════════
# Metrics + bootstrap
# ══════════════════════════════════════════════════════════════════════════════
def val_margin_threshold(val_true, val_probs):
    """Smallest-FPR val threshold achieving Sens ≥ VAL_SELECTION_SENS (matches model.py)."""
    fpr, tpr, thr = roc_curve(val_true, val_probs)
    elig = np.where(tpr >= VAL_SELECTION_SENS)[0]
    idx = elig[np.argmin(fpr[elig])] if len(elig) else int(np.argmax(tpr))
    return float(thr[idx])


def bootstrap_ci(y_true, probs, threshold, n_boot=N_BOOT):
    """Bootstrap 95% CIs for AUC, Sensitivity, Specificity at a fixed threshold."""
    n = len(y_true)
    pred = (probs >= threshold).astype(int)
    aucs, sens, spec = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():        # need both classes present
            continue
        aucs.append(roc_auc_score(yt, probs[idx]))
        p = pred[idx]
        tp = np.sum((p == 1) & (yt == 1)); pos = np.sum(yt == 1)
        tn = np.sum((p == 0) & (yt == 0)); neg = np.sum(yt == 0)
        sens.append(tp / pos if pos else np.nan)
        spec.append(tn / neg if neg else np.nan)

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return ci(aucs), ci(sens), ci(spec)


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Reproducing leak-free split (random_state=42) …")
    train_df, val_df, test_df = load_split()
    enc = fit_metadata_encoder(train_df)
    meta_val  = transform_metadata(val_df,  enc)
    meta_test = transform_metadata(test_df, enc)
    val_true  = val_df['group_label'].to_numpy()
    test_true = test_df['group_label'].to_numpy()
    print(f"  Val n={len(val_df)} | Test n={len(test_df)} "
          f"(Concerning test n={int(test_true.sum())})")

    results = {}
    lines = []
    lines.append("# Statistical Analysis — Skin Lesion Binary CNN\n")
    lines.append(f"_Test set: n={len(test_df)} ({int(test_true.sum())} Concerning, "
                 f"{int((test_true==0).sum())} Benign). Bootstrap B={N_BOOT}, seed={RNG_SEED}. "
                 f"Operating point: val-margin threshold at val Sens ≥ {VAL_SELECTION_SENS}._\n")
    lines.append("## Per-model performance with 95% confidence intervals\n")
    lines.append("| Model | AUC (DeLong 95% CI) | AUC (bootstrap 95% CI) | thr | Sens (95% CI) | Spec (95% CI) |")
    lines.append("|---|---|---|:---:|---|---|")

    for name, fname in MODELS.items():
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  {fname} not found — skipping {name}")
            continue
        print(f"\nLoading {name} ({fname}) …")
        model = tf.keras.models.load_model(path, compile=False)
        vp = predict_probs(model, val_df,  meta_val)
        tp = predict_probs(model, test_df, meta_test)
        results[name] = {'val_probs': vp, 'test_probs': tp}

        thr = val_margin_threshold(val_true, vp)
        auc, var = delong_variance(test_true, tp)
        half = 1.96 * np.sqrt(var)
        dl_lo, dl_hi = auc - half, auc + half
        (b_auc_lo, b_auc_hi), (sn_lo, sn_hi), (sp_lo, sp_hi) = bootstrap_ci(test_true, tp, thr)

        pred = (tp >= thr).astype(int)
        sens = recall_score(test_true, pred)
        spec = recall_score(1 - test_true, 1 - pred)
        results[name].update(auc=auc, thr=thr, sens=sens, spec=spec)

        print(f"  AUC = {auc:.4f}  (DeLong 95% CI {dl_lo:.4f}–{dl_hi:.4f})")
        print(f"  thr = {thr:.3f} | Sens = {sens:.4f} ({sn_lo:.3f}–{sn_hi:.3f}) | "
              f"Spec = {spec:.4f} ({sp_lo:.3f}–{sp_hi:.3f})")
        lines.append(f"| {name} | {auc:.4f} ({dl_lo:.4f}–{dl_hi:.4f}) | "
                     f"{b_auc_lo:.4f}–{b_auc_hi:.4f} | {thr:.3f} | "
                     f"{sens:.4f} ({sn_lo:.3f}–{sn_hi:.3f}) | "
                     f"{spec:.4f} ({sp_lo:.3f}–{sp_hi:.3f}) |")

    # ── Paired DeLong significance tests ─────────────────────────────────────
    for a, b in DELONG_PAIRS:
        if a not in results or b not in results:
            continue
        auc_a, auc_b, z, p = delong_paired_test(
            test_true, results[a]['test_probs'], results[b]['test_probs'])
        verdict = "SIGNIFICANT (p < 0.05)" if p < 0.05 else "NOT significant (p ≥ 0.05)"
        print("\n" + "=" * 60)
        print(f"DeLong paired test — {b} vs {a}")
        print(f"  AUC {a} = {auc_a:.4f} | AUC {b} = {auc_b:.4f} | Δ = {auc_b-auc_a:+.4f}")
        print(f"  z = {z:.3f}  p = {p:.4f}  →  {verdict}")
        print("=" * 60)
        lines.append(f"\n## Paired DeLong test — {b} vs {a}\n")
        lines.append(f"- AUC {a} = **{auc_a:.4f}**, AUC {b} = **{auc_b:.4f}**, "
                     f"Δ = **{auc_b-auc_a:+.4f}**")
        lines.append(f"- z = {z:.3f}, **p = {p:.4f}** → {b} is **{verdict}** "
                     f"vs {a}, on the same test set.")

    # ── Optional: aggregate multi-seed predictions if present ─────────────────
    seed_files = sorted(glob.glob(os.path.join(script_dir, 'multiseed', 'preds_seed*.npz')))
    if seed_files:
        print(f"\nAggregating {len(seed_files)} multi-seed runs …")
        aucs, senss, specs = [], [], []
        for f in seed_files:
            d = np.load(f)
            vt, vp, tt, tp = d['val_true'], d['val_probs'], d['test_true'], d['test_probs']
            thr = val_margin_threshold(vt, vp)
            pred = (tp >= thr).astype(int)
            aucs.append(roc_auc_score(tt, tp))
            senss.append(recall_score(tt, pred))
            specs.append(recall_score(1 - tt, 1 - pred))
        def ms(x):
            x = np.asarray(x); return x.mean(), x.std(ddof=1)
        (am, asd), (sm, ssd), (pm, psd) = ms(aucs), ms(senss), ms(specs)
        print(f"  AUC  {am:.4f} ± {asd:.4f}")
        print(f"  Sens {sm:.4f} ± {ssd:.4f}")
        print(f"  Spec {pm:.4f} ± {psd:.4f}")
        lines.append(f"\n## Multi-seed champion ({len(seed_files)} seeds) — mean ± std\n")
        lines.append("| Metric | Mean ± SD |")
        lines.append("|---|---|")
        lines.append(f"| Test AUC | {am:.4f} ± {asd:.4f} |")
        lines.append(f"| Test Sensitivity | {sm:.4f} ± {ssd:.4f} |")
        lines.append(f"| Test Specificity | {pm:.4f} ± {psd:.4f} |")
        lines.append(f"\n_Seeds: {[os.path.basename(f) for f in seed_files]}_")
    else:
        lines.append("\n## Multi-seed champion\n")
        lines.append("_No `multiseed/preds_seed*.npz` found yet — run `multiseed_train.py` "
                     "across ≥5 seeds, then re-run this script to fill in mean ± std._")

    out = os.path.join(script_dir, 'documentation', 'statistical_analysis.md')
    with open(out, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n✅ Wrote {out}")


if __name__ == '__main__':
    main()
