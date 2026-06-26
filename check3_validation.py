"""
Check 3 — Data integrity + per-diagnosis breakdown
Two things:
  A. Formal lesion-ID isolation: prove no image appears in more than one split.
  B. Per-diagnosis recall within Concerning (mel / bcc / akiec separately).
     AUC of 0.883 is good overall, but if melanoma recall is 50% while BCC
     is 95%, that is a clinical blind spot the aggregate number hides.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1 = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2 = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')
model_path    = os.path.join(script_dir, 'best_model_run2_color.h5')

# ── MUST match the run's training preprocessing ──────────────────────────────
# Run 0/1 → USE_COLOR_CONSTANCY = False ; Run 2+ → True. (Mismatch = garbage scores.)
USE_COLOR_CONSTANCY = True

def shades_of_gray(img, power=6):
    img = img.astype('float32')
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + 1e-6) * np.sqrt(3.0)
    img = img / (illum + 1e-6)
    img = np.clip(img, 0, 255)
    return img / 255.0

CLINICAL_GROUPS = {
    'nv': 'Benign', 'bkl': 'Benign', 'df': 'Benign', 'vasc': 'Benign',
    'mel': 'Concerning', 'bcc': 'Concerning', 'akiec': 'Concerning',
}
GROUP_ORDER    = ['Benign', 'Concerning']
CONCERNING_IDX = 1
IMG_SIZE       = 224
BATCH_SIZE     = 32

# ── Reproduce exact splits ────────────────────────────────────────────────────
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

train_df = df[df['lesion_id'].isin(train_ids)].reset_index(drop=True)
val_df   = df[df['lesion_id'].isin(val_ids)].reset_index(drop=True)
test_df  = df[df['lesion_id'].isin(test_ids)].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3A — Lesion ID isolation
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("CHECK 3A — Lesion ID isolation")
print("=" * 55)

train_lesions = set(train_df['lesion_id'].unique())
val_lesions   = set(val_df['lesion_id'].unique())
test_lesions  = set(test_df['lesion_id'].unique())

tv_overlap  = train_lesions & val_lesions
tt_overlap  = train_lesions & test_lesions
vt_overlap  = val_lesions   & test_lesions

print(f"  Train lesions:              {len(train_lesions):>5}")
print(f"  Val lesions:                {len(val_lesions):>5}")
print(f"  Test lesions:               {len(test_lesions):>5}")
print(f"  Total unique lesions:       {len(train_lesions | val_lesions | test_lesions):>5}")
print()
print(f"  Train ∩ Val   overlap:      {len(tv_overlap):>5}  {'✅ CLEAN' if not tv_overlap else '❌ LEAK DETECTED'}")
print(f"  Train ∩ Test  overlap:      {len(tt_overlap):>5}  {'✅ CLEAN' if not tt_overlap else '❌ LEAK DETECTED'}")
print(f"  Val   ∩ Test  overlap:      {len(vt_overlap):>5}  {'✅ CLEAN' if not vt_overlap else '❌ LEAK DETECTED'}")
print()

# Image-level check: a lesion can have multiple images — ensure no image
# sneaks across via a shared lesion even if lesion_ids don't overlap.
train_imgs = set(train_df['image_id'].unique())
val_imgs   = set(val_df['image_id'].unique())
test_imgs  = set(test_df['image_id'].unique())
img_tv = train_imgs & val_imgs
img_tt = train_imgs & test_imgs
img_vt = val_imgs   & test_imgs
print(f"  Train images:               {len(train_imgs):>5}")
print(f"  Val images:                 {len(val_imgs):>5}")
print(f"  Test images:                {len(test_imgs):>5}")
print(f"  Image-level Train∩Val:      {len(img_tv):>5}  {'✅ CLEAN' if not img_tv else '❌ LEAK'}")
print(f"  Image-level Train∩Test:     {len(img_tt):>5}  {'✅ CLEAN' if not img_tt else '❌ LEAK'}")
print(f"  Image-level Val∩Test:       {len(img_vt):>5}  {'✅ CLEAN' if not img_vt else '❌ LEAK'}")

if any([tv_overlap, tt_overlap, vt_overlap, img_tv, img_tt, img_vt]):
    print("\n❌ DATA LEAK DETECTED — results are not trustworthy.")
else:
    print("\n✅ All splits are fully isolated at both lesion and image level.")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3B — Per-diagnosis recall within Concerning (mel / bcc / akiec)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("CHECK 3B — Per-diagnosis recall within Concerning")
print("=" * 55)

if USE_COLOR_CONSTANCY:
    datagen = ImageDataGenerator(rescale=None, preprocessing_function=shades_of_gray)
else:
    datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_dataframe(
    test_df, x_col='path', y_col='group',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    classes=GROUP_ORDER, class_mode='categorical', shuffle=False
)
val_gen = datagen.flow_from_dataframe(
    val_df, x_col='path', y_col='group',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    classes=GROUP_ORDER, class_mode='categorical', shuffle=False
)

print("\nLoading model …")
model = tf.keras.models.load_model(model_path, compile=False)

val_gen.reset()
val_probs = model.predict(val_gen, verbose=1)[:, CONCERNING_IDX]
val_true  = np.array(val_gen.classes)

test_gen.reset()
test_probs = model.predict(test_gen, verbose=1)[:, CONCERNING_IDX]
test_true  = np.array(test_gen.classes)

# Val-selected screening threshold
val_fpr, val_tpr, val_thr = roc_curve((val_true == CONCERNING_IDX).astype(int), val_probs)
elig       = np.where(val_tpr >= 0.90)[0]
hs_idx     = elig[np.argmin(val_fpr[elig])] if len(elig) else int(np.argmax(val_tpr))
screen_thr = val_thr[hs_idx]

# Attach predicted probs back to test_df (generator preserves order, shuffle=False)
test_df_eval = test_df.copy()
test_df_eval['conc_prob']  = test_probs
test_df_eval['true_label'] = test_true
test_df_eval['pred_screen'] = (test_probs >= screen_thr).astype(int)
test_df_eval['pred_argmax'] = (test_probs >= 0.5).astype(int)

CONCERNING_DX = ['mel', 'bcc', 'akiec']
BENIGN_DX     = ['nv', 'bkl', 'df', 'vasc']

print(f"\n{'Diagnosis':<10} {'n':>5} {'Mean P(Conc)':>13} "
      f"{f'Recall@{screen_thr:.3f}':>13} {'Recall@0.5':>11} {'Median P':>9}")
print("-" * 65)

dx_stats = {}
for dx in CONCERNING_DX + BENIGN_DX:
    sub = test_df_eval[test_df_eval['dx'] == dx]
    if len(sub) == 0:
        continue
    mean_p   = sub['conc_prob'].mean()
    median_p = sub['conc_prob'].median()
    if dx in CONCERNING_DX:
        # Recall = fraction correctly flagged as Concerning
        recall_screen = sub['pred_screen'].mean()
        recall_argmax = sub['pred_argmax'].mean()
        flag = "⚠️ " if recall_screen < 0.80 else "  "
    else:
        # For benign: specificity = fraction correctly cleared (NOT flagged)
        recall_screen = (sub['pred_screen'] == 0).mean()  # correctly cleared
        recall_argmax = (sub['pred_argmax'] == 0).mean()
        flag = "⚠️ " if recall_screen < 0.60 else "  "
    dx_stats[dx] = {'n': len(sub), 'mean_p': mean_p, 'recall_screen': recall_screen,
                    'recall_argmax': recall_argmax, 'median_p': median_p}
    label = f"{dx}{flag}"
    print(f"{label:<12} {len(sub):>4}  {mean_p:>12.3f}  "
          f"{recall_screen:>12.1%}  {recall_argmax:>10.1%}  {median_p:>8.3f}")

print("-" * 65)
print(f"{'(Concerning = recall = % caught)'}")
print(f"{'(Benign     = specificity = % correctly cleared)'}")
print(f"\nScreening threshold: {screen_thr:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Top row: per-dx probability distributions (Concerning only) ---
for i, dx in enumerate(CONCERNING_DX):
    ax  = fig.add_subplot(gs[0, i])
    sub = test_df_eval[test_df_eval['dx'] == dx]['conc_prob']
    ax.hist(sub, bins=30, color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    ax.axvline(screen_thr, color='blue',  linestyle=':', lw=2,
               label=f'Val thr ({screen_thr:.3f})')
    ax.axvline(0.5,        color='black', linestyle='--', lw=1.5, label='Argmax (0.5)')
    caught_screen = (sub >= screen_thr).mean()
    caught_argmax = (sub >= 0.5).mean()
    ax.set_title(f'{dx.upper()}  (n={len(sub)})\n'
                 f'Caught @ thr: {caught_screen:.1%}  |  @ 0.5: {caught_argmax:.1%}',
                 fontsize=10)
    ax.set_xlabel('P(Concerning)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# --- Bottom row: the two most informative Benign classes ---
# nv (dominant benign, the easy case) and bkl (the #1 false-positive driver,
# the bkl↔akiec confusion this run targets). df/vasc are omitted from the plot
# (tiny n=20/31, noisy) but remain in the printed table above.
colors = {'nv': 'forestgreen', 'bkl': 'mediumseagreen', 'df': 'steelblue', 'vasc': 'mediumpurple'}
BOTTOM_BENIGN = ['nv', 'bkl']   # plotted in gs[1,0] and gs[1,1]; ROC takes gs[1,2]
for ax_col, dx in enumerate(BOTTOM_BENIGN):
    ax  = fig.add_subplot(gs[1, ax_col])
    sub = test_df_eval[test_df_eval['dx'] == dx]['conc_prob']
    ax.hist(sub, bins=30, color=colors.get(dx, 'gray'), alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.axvline(screen_thr, color='blue',  linestyle=':', lw=2,
               label=f'Val thr ({screen_thr:.3f})')
    ax.axvline(0.5,        color='black', linestyle='--', lw=1.5, label='Argmax (0.5)')
    fp_screen = (sub >= screen_thr).mean()   # wrongly flagged at the screening threshold
    fp_argmax = (sub >= 0.5).mean()
    ax.set_title(f'{dx.upper()}  (n={len(sub)})\n'
                 f'Wrongly flagged @ thr: {fp_screen:.1%}  |  @ 0.5: {fp_argmax:.1%}',
                 fontsize=10)
    ax.set_xlabel('P(Concerning)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# Fill remaining bottom subplot with ROC curve
ax_roc = fig.add_subplot(gs[1, 2])
test_conc_true = (test_true == CONCERNING_IDX).astype(int)
fpr, tpr, _ = roc_curve(test_conc_true, test_probs)
auc_val      = roc_auc_score(test_conc_true, test_probs)
ax_roc.plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {auc_val:.4f}')
ax_roc.plot([0, 1], [0, 1], 'gray', linestyle='--', lw=1)
# Mark the val-selected operating point on the test curve
op_fpr = 1 - dx_stats.get('nv', {}).get('recall_screen', 0)
screen_pred = (test_probs >= screen_thr).astype(int)
op_sens = recall_score(test_conc_true, screen_pred)
op_spec = recall_score(1 - test_conc_true, 1 - screen_pred)
ax_roc.scatter(1 - op_spec, op_sens, color='blue', marker='D', zorder=5,
               label=f'Val thr={screen_thr:.3f}\nSens={op_sens:.3f} Spec={op_spec:.3f}')
ax_roc.set_xlabel('FPR (1 - Specificity)')
ax_roc.set_ylabel('TPR (Sensitivity)')
ax_roc.set_title('ROC — Test set', fontsize=10)
ax_roc.legend(fontsize=7)
ax_roc.grid(alpha=0.3)

fig.suptitle('Check 3B: Per-Diagnosis Performance Breakdown — SE-Residual CNN',
             fontweight='bold', fontsize=13)
plt.savefig('check3_per_diagnosis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved to check3_per_diagnosis.png")
