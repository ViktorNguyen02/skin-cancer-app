"""
Check 2 — Probability Distribution Plot
Loads best_model_binary.h5 and the held-out test set (same seed/split as
training) and plots how confident the model is that each image is Concerning.
Run standalone — no retraining needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ── Paths ────────────────────────────────────────────────────────────────────
script_dir     = os.path.dirname(os.path.abspath(__file__))
metadata_path  = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1  = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2  = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')
model_path     = os.path.join(script_dir, 'best_model_run4_meta.h5')

# ── Run 4: two-input model ([image, metadata]) on plain rescale=1./255 ───────
# Run 4 fuses patient metadata (age/sex/localization) with the image branch, so
# scoring must feed BOTH inputs. The metadata encoder is re-fit on the SAME train
# split (random_state=42) so its stats/vocabularies match training exactly.

# ── Clinical grouping (must match training) ──────────────────────────────────
CLINICAL_GROUPS = {
    'nv':    'Benign',
    'bkl':   'Benign',
    'df':    'Benign',
    'vasc':  'Benign',
    'mel':   'Concerning',
    'bcc':   'Concerning',
    'akiec': 'Concerning',
}
GROUP_ORDER     = ['Benign', 'Concerning']
CONCERNING_IDX  = 1
IMG_SIZE        = 224
BATCH_SIZE      = 32

# ── Reproduce the exact test split (same random_state=42 as training) ────────
df = pd.read_csv(metadata_path)
df['group'] = df['dx'].map(CLINICAL_GROUPS)
le = LabelEncoder()
df['group_label'] = le.fit_transform(df['group'])

def find_image_path(image_id):
    for base in [images_path_1, images_path_2]:
        p = os.path.join(base, image_id + '.jpg')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(image_id)

df['path'] = df['image_id'].map(find_image_path)

lesion_group_df = df.drop_duplicates(subset='lesion_id')[['lesion_id', 'group_label']]
trainval_ids, test_ids = train_test_split(
    lesion_group_df['lesion_id'], test_size=0.2,
    stratify=lesion_group_df['group_label'], random_state=42
)
trainval_group = lesion_group_df[lesion_group_df['lesion_id'].isin(trainval_ids)]
train_ids, val_ids = train_test_split(
    trainval_group['lesion_id'], test_size=0.15,
    stratify=trainval_group['group_label'], random_state=42
)

train_df = df[df['lesion_id'].isin(train_ids)].reset_index(drop=True)   # needed to fit meta encoder
val_df   = df[df['lesion_id'].isin(val_ids)].reset_index(drop=True)
test_df  = df[df['lesion_id'].isin(test_ids)].reset_index(drop=True)
print(f"Val samples:  {len(val_df)}  |  Test samples: {len(test_df)}")

# ── Metadata encoder (re-fit on TRAIN only — identical to model.py) ───────────
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

meta_encoder = fit_metadata_encoder(train_df)
meta_val  = transform_metadata(val_df,  meta_encoder)
meta_test = transform_metadata(test_df, meta_encoder)

def predict_with_meta(model, df, meta, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Batched two-input predict; rows stay in df order so probs align with df labels."""
    n, paths = len(df), df['path'].tolist()
    probs = np.zeros((n, len(GROUP_ORDER)), dtype='float32')
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_img = np.zeros((end - start, img_size, img_size, 3), dtype='float32')
        for j, idx in enumerate(range(start, end)):
            X_img[j] = img_to_array(load_img(paths[idx], target_size=(img_size, img_size))) / 255.0
        probs[start:end] = model.predict([X_img, meta[start:end]], verbose=0)
    return probs

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model …")
model = tf.keras.models.load_model(model_path, compile=False)
print(f"  Params: {model.count_params():,}  |  Name: {model.name}")

# ── Predict on both val and test ─────────────────────────────────────────────
val_probs  = predict_with_meta(model, val_df,  meta_val)[:,  CONCERNING_IDX]
val_true   = val_df['group_label'].to_numpy()

test_probs = predict_with_meta(model, test_df, meta_test)[:, CONCERNING_IDX]
test_true  = test_df['group_label'].to_numpy()

# ── Find val-selected screening threshold (Sens ≥ 0.90) ─────────────────────
from sklearn.metrics import roc_curve, roc_auc_score
val_fpr, val_tpr, val_thr = roc_curve(
    (val_true == CONCERNING_IDX).astype(int), val_probs
)
elig      = np.where(val_tpr >= 0.90)[0]
hs_idx    = elig[np.argmin(val_fpr[elig])] if len(elig) else int(np.argmax(val_tpr))
screening_thr = val_thr[hs_idx]
print(f"\nVal-selected screening threshold: {screening_thr:.3f}")
print(f"  Val  — Sens: {val_tpr[hs_idx]:.4f}  Spec: {1-val_fpr[hs_idx]:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, probs, true, title in [
    (axes[0], val_probs,  val_true,  'Validation set'),
    (axes[1], test_probs, test_true, 'Test set (blind)'),
]:
    benign_p     = probs[true == 0]
    concerning_p = probs[true == CONCERNING_IDX]

    ax.hist(benign_p,     bins=60, alpha=0.55, color='green',
            density=True, label=f'Actual Benign (n={len(benign_p)})')
    ax.hist(concerning_p, bins=60, alpha=0.55, color='red',
            density=True, label=f'Actual Concerning (n={len(concerning_p)})')
    ax.axvline(0.5,           color='black', linestyle='--', lw=1.5,
               label='Argmax threshold (0.5)')
    ax.axvline(screening_thr, color='blue',  linestyle=':',  lw=1.5,
               label=f'Val-selected thr ({screening_thr:.3f})')

    # Annotate what fraction of Concerning images land above each threshold
    above_argmax   = (concerning_p >= 0.5).mean()
    above_val_thr  = (concerning_p >= screening_thr).mean()
    ax.text(0.97, 0.95,
            f'Concerning caught\n'
            f'@ 0.5:   {above_argmax:.1%}\n'
            f'@ {screening_thr:.3f}: {above_val_thr:.1%}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'Predicted P(Concerning) — {title}')
    ax.set_xlabel('Predicted Probability of being Concerning')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Check 2: Probability Calibration — SE-Residual CNN', fontweight='bold')
plt.tight_layout()
plt.savefig('check2_prob_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved to check2_prob_distribution.png")

# ── Summary numbers ───────────────────────────────────────────────────────────
test_conc_true = (test_true == CONCERNING_IDX).astype(int)
test_auc       = roc_auc_score(test_conc_true, test_probs)
screen_pred    = (test_probs >= screening_thr).astype(int)
from sklearn.metrics import recall_score
test_sens = recall_score(test_conc_true, screen_pred)
test_spec = recall_score(1 - test_conc_true, 1 - screen_pred)

print(f"\n{'─'*45}")
print(f"Test AUC:                        {test_auc:.4f}")
print(f"Test @ val-thr={screening_thr:.3f}:")
print(f"  Sensitivity:                   {test_sens:.4f}")
print(f"  Specificity:                   {test_spec:.4f}")
print(f"Probability range (Concerning):  "
      f"{test_probs[test_true==CONCERNING_IDX].min():.3f} – "
      f"{test_probs[test_true==CONCERNING_IDX].max():.3f}")
print(f"Probability range (Benign):      "
      f"{test_probs[test_true==0].min():.3f} – "
      f"{test_probs[test_true==0].max():.3f}")
print(f"{'─'*45}")
