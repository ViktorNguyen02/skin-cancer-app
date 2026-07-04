"""
baseline_transfer.py — Transfer-learning baseline (EfficientNet-B0) for the paper.

WHY THIS FILE EXISTS (and why it is NOT model.py):
  Our contribution model (model.py) is trained entirely FROM SCRATCH — no pretrained
  weights. A reviewer will nonetheless ask: "how does a fine-tuned ImageNet backbone do
  on the SAME task?" This script answers that with a fair, apples-to-apples baseline. It
  is a YARDSTICK, not part of our architecture, so it lives in its own file and never
  touches the from-scratch pipeline or the champion weights.

FAIRNESS — everything is identical to model.py EXCEPT the pretraining:
  • same leak-free lesion-level split (random_state=42) → same train/val/test images
  • same binary Concerning-vs-Benign mapping
  • same focal loss (gamma=1.5, label smoothing 0.05, sqrt-balanced alpha)
  • same val_auc model selection + val-margin operating point (VAL_SELECTION_SENS=0.94)
  • same [0,1] image inputs (see preprocessing note below)
  The ONLY difference is: EfficientNet-B0 initialized from ImageNet vs our SE-residual
  net initialized from random. That isolates the effect of transfer learning.

PREPROCESSING GOTCHA (handled):
  Keras EfficientNet models expect inputs in [0,255] and apply ImageNet normalization
  INTERNALLY. To keep this model drop-in compatible with our existing [0,1] pipeline
  (model.py, stats_analysis.py all feed images/255.0), we feed [0,1] images and put a
  Rescaling(255.) layer at the top of the model, which restores [0,255] before the
  backbone's internal normalization. Net effect: correct EfficientNet preprocessing,
  while every downstream scorer can treat this model exactly like the from-scratch ones.

TWO-PHASE FINE-TUNE:
  Phase 1 — freeze the backbone, train only the new head (fast, stabilizes the head).
  Phase 2 — unfreeze the backbone (BatchNorm layers kept frozen, per standard practice)
            and fine-tune at a low LR.

USAGE:
  python baseline_transfer.py                 # image-only baseline (vs from-scratch Run 1)
  python baseline_transfer.py --metadata      # image+metadata (vs champion Run 4)
  python baseline_transfer.py --head-epochs 8 --finetune-epochs 15
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                      Concatenate, Rescaling)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, load_img,
                                                  img_to_array)
from tensorflow.keras.utils import Sequence, to_categorical

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config (mirrors model.py) ─────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1 = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2 = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')

CLINICAL_GROUPS = {
    'nv': 'Benign', 'bkl': 'Benign', 'df': 'Benign', 'vasc': 'Benign',
    'mel': 'Concerning', 'bcc': 'Concerning', 'akiec': 'Concerning',
}
GROUP_ORDER        = ['Benign', 'Concerning']   # alphabetical: Benign=0, Concerning=1
CONCERNING_IDX     = 1
IMG_SIZE           = 224
BATCH_SIZE         = 32
TARGET_SENS        = 0.90
VAL_SELECTION_SENS = 0.94    # same safety margin as model.py
HEAD_LR            = 1e-3    # phase 1: head only
FINETUNE_LR        = 1e-5    # phase 2: unfrozen backbone


# ══════════════════════════════════════════════════════════════════════
# Data — reproduce the EXACT leak-free split (identical to model.py)
# ══════════════════════════════════════════════════════════════════════
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

    # Leak guard (same asserts as model.py)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
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


# ══════════════════════════════════════════════════════════════════════
# Focal loss (identical to model.py)
# ══════════════════════════════════════════════════════════════════════
def make_focal_loss(gamma=1.5, alpha=None, label_smoothing=0.05):
    alpha_t = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        ce = -tf.math.log(p_t)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        if alpha_t is not None:
            a_t = tf.reduce_sum(alpha_t * y_true, axis=-1)
            loss = a_t * focal_weight * ce
        else:
            loss = focal_weight * ce
        return tf.reduce_mean(loss)

    return focal_loss


# ══════════════════════════════════════════════════════════════════════
# Data generator — feeds [0,1] images (Rescaling(255.) sits inside the model)
# ══════════════════════════════════════════════════════════════════════
class LesionSequence(Sequence):
    """Yields (X, y). X is the image array, or [image, meta] when use_meta=True.
    Images are in [0,1] (rescale=1./255), matching model.py; the model's top
    Rescaling(255.) layer restores [0,255] for EfficientNet's internal normalization."""
    def __init__(self, df, meta, datagen, augment, use_meta, shuffle):
        self.df       = df.reset_index(drop=True)
        self.meta     = meta.astype('float32') if meta is not None else None
        self.labels   = self.df['group_label'].to_numpy()
        self.paths    = self.df['path'].tolist()
        self.datagen  = datagen
        self.augment  = augment
        self.use_meta = use_meta
        self.shuffle  = shuffle
        self.indices  = np.arange(len(self.df))
        self.on_epoch_end()

    @property
    def classes(self):
        return self.labels           # val/test never shuffle → predict order matches
    def reset(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.df) / BATCH_SIZE))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, i):
        batch_idx = self.indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        X_img = np.zeros((len(batch_idx), IMG_SIZE, IMG_SIZE, 3), dtype='float32')
        for j, idx in enumerate(batch_idx):
            img = load_img(self.paths[idx], target_size=(IMG_SIZE, IMG_SIZE))
            xi  = img_to_array(img)
            if self.augment:
                xi = self.datagen.random_transform(xi)
            xi = self.datagen.standardize(xi)     # applies rescale=1./255 → [0,1]
            X_img[j] = xi
        y = to_categorical(self.labels[batch_idx], num_classes=len(GROUP_ORDER))
        if self.use_meta:
            return [X_img, self.meta[batch_idx]], y
        return X_img, y


# ══════════════════════════════════════════════════════════════════════
# Model — EfficientNet-B0 backbone + our head (optionally + metadata branch)
# ══════════════════════════════════════════════════════════════════════
def build_model(use_meta, meta_dim):
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    x = Rescaling(255.0)(image_input)             # [0,1] → [0,255] for EfficientNet
    base = EfficientNetB0(include_top=False, weights='imagenet',
                          input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    base._name = 'efficientnetb0_backbone'
    feat = base(x)                                # 1280-d image descriptor

    if use_meta:
        meta_input = Input(shape=(meta_dim,), name='meta_input')
        m = Dense(32, activation='relu')(meta_input)
        m = Dense(16, activation='relu')(m)
        feat = Concatenate()([feat, m])           # matches model.py's fusion design
        inputs = [image_input, meta_input]
    else:
        inputs = image_input

    h = Dense(256, activation='relu')(feat)
    h = BatchNormalization()(h)
    h = Dropout(0.40)(h)
    outputs = Dense(len(GROUP_ORDER), activation='softmax')(h)

    model = Model(inputs=inputs, outputs=outputs,
                  name='EfficientNetB0_Transfer' + ('_Meta' if use_meta else ''))
    return model, base


# ══════════════════════════════════════════════════════════════════════
# val_auc callback (identical role to model.py's ValMetrics)
# ══════════════════════════════════════════════════════════════════════
class ValAUC(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, val_true):
        super().__init__()
        self.val_gen = val_gen
        self.val_true = val_true

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = self.model.predict(self.val_gen, verbose=0)[:, CONCERNING_IDX]
        logs['val_auc'] = roc_auc_score(self.val_true, probs)
        print(f"  — val_auc: {logs['val_auc']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Evaluation helpers (identical logic to model.py / stats_analysis.py)
# ══════════════════════════════════════════════════════════════════════
def val_margin_threshold(val_true, val_probs):
    fpr, tpr, thr = roc_curve(val_true, val_probs)
    elig = np.where(tpr >= VAL_SELECTION_SENS)[0]
    idx = elig[np.argmin(fpr[elig])] if len(elig) else int(np.argmax(tpr))
    return float(thr[idx])


def ece_score(probs2, y_int, n_bins=10):
    conf = probs2.max(axis=1); pred = probs2.argmax(axis=1)
    acc  = (pred == y_int).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return ece


def bootstrap_ci(y_true, probs, threshold, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    pred = (probs >= threshold).astype(int)
    aucs, sens, spec = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, probs[idx]))
        p = pred[idx]
        tp = np.sum((p == 1) & (yt == 1)); pos = np.sum(yt == 1)
        tn = np.sum((p == 0) & (yt == 0)); neg = np.sum(yt == 0)
        sens.append(tp / pos if pos else np.nan)
        spec.append(tn / neg if neg else np.nan)

    def ci(a):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return ci(aucs), ci(sens), ci(spec)


def predict_seq(model, gen):
    return model.predict(gen, verbose=0)[:, CONCERNING_IDX]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', action='store_true',
                    help='fuse age/sex/localization (compare vs champion Run 4)')
    ap.add_argument('--head-epochs', type=int, default=8)
    ap.add_argument('--finetune-epochs', type=int, default=15)
    args = ap.parse_args()
    use_meta = args.metadata
    tag = 'effb0_meta' if use_meta else 'effb0'

    print("--- HARDWARE CHECK ---")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow {tf.__version__} | GPU: {'yes ('+str(len(gpus))+')' if gpus else 'NO (CPU)'}")
    print(f"Baseline: EfficientNet-B0 transfer | metadata={use_meta}\n")

    # ── Data ──
    train_df, val_df, test_df = load_split()
    enc = fit_metadata_encoder(train_df)
    meta_train = transform_metadata(train_df, enc) if use_meta else None
    meta_val   = transform_metadata(val_df,   enc) if use_meta else None
    meta_test  = transform_metadata(test_df,  enc) if use_meta else None
    meta_dim   = meta_train.shape[1] if use_meta else 0
    val_true   = val_df['group_label'].to_numpy()
    test_true  = test_df['group_label'].to_numpy()
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)} "
          f"(Concerning test n={int(test_true.sum())})"
          + (f" | meta_dim={meta_dim}" if use_meta else "") + "\n")

    # ── Focal-loss alpha (sqrt-balanced, identical to model.py) ──
    cw = compute_class_weight('balanced',
                              classes=np.unique(train_df['group_label']),
                              y=train_df['group_label'])
    alpha = np.sqrt(cw).astype(np.float32)
    alpha = (alpha / alpha.sum()).astype(np.float32)
    loss_fn = make_focal_loss(gamma=1.5, alpha=alpha, label_smoothing=0.05)

    # ── Generators (feed [0,1]; augmentation identical to model.py) ──
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=90, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True, vertical_flip=True,
        zoom_range=0.2, brightness_range=[0.9, 1.1], shear_range=0.1,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = LesionSequence(train_df, meta_train, train_datagen,
                               augment=True,  use_meta=use_meta, shuffle=True)
    val_gen   = LesionSequence(val_df, meta_val, test_datagen,
                               augment=False, use_meta=use_meta, shuffle=False)
    test_gen  = LesionSequence(test_df, meta_test, test_datagen,
                               augment=False, use_meta=use_meta, shuffle=False)

    model, base = build_model(use_meta, meta_dim)
    model_path = os.path.join(script_dir, f'best_model_transfer_{tag}.h5')

    callbacks = [
        ValAUC(val_gen, val_true),
        EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
    ]

    # ── Phase 1 — freeze backbone, train head ──
    print("=" * 60)
    print(f"PHASE 1 — frozen backbone, train head ({args.head_epochs} epochs)")
    print("=" * 60)
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(HEAD_LR, clipnorm=1.0),
                  loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen,
              epochs=args.head_epochs, callbacks=callbacks)

    # ── Phase 2 — unfreeze backbone (keep BatchNorm frozen), fine-tune ──
    print("\n" + "=" * 60)
    print(f"PHASE 2 — fine-tune backbone, BN frozen ({args.finetune_epochs} epochs, LR={FINETUNE_LR})")
    print("=" * 60)
    base.trainable = True
    bn_frozen = 0
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False       # keep ImageNet BN running stats stable
            bn_frozen += 1
    print(f"  Unfroze backbone; kept {bn_frozen} BatchNorm layers frozen.")
    model.compile(optimizer=tf.keras.optimizers.Adam(FINETUNE_LR, clipnorm=1.0),
                  loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen,
              epochs=args.finetune_epochs, callbacks=callbacks)

    # ── Evaluate at the val-margin operating point ──
    print("\n" + "=" * 60)
    print("EVALUATION (val-selected threshold, applied once to test)")
    print("=" * 60)
    val_probs  = predict_seq(model, val_gen)
    test_probs = predict_seq(model, test_gen)
    thr = val_margin_threshold(val_true, val_probs)

    auc = roc_auc_score(test_true, test_probs)
    pred = (test_probs >= thr).astype(int)
    sens = recall_score(test_true, pred)
    spec = recall_score(1 - test_true, 1 - pred)
    (a_lo, a_hi), (s_lo, s_hi), (p_lo, p_hi) = bootstrap_ci(test_true, test_probs, thr)
    test_probs2 = np.column_stack([1 - test_probs, test_probs])
    ece = ece_score(test_probs2, test_true)
    floor = "✅ clears" if sens >= TARGET_SENS else "⚠️ below"

    print(f"  Model: {model.name}  ({model.count_params():,} params)")
    print(f"  Test AUC = {auc:.4f}  (95% CI {a_lo:.4f}–{a_hi:.4f})")
    print(f"  thr = {thr:.3f}")
    print(f"  Sensitivity = {sens:.4f}  (95% CI {s_lo:.3f}–{s_hi:.3f})  {floor} the {TARGET_SENS} floor")
    print(f"  Specificity = {spec:.4f}  (95% CI {p_lo:.3f}–{p_hi:.3f})")
    print(f"  ECE (raw) = {ece:.4f}")

    # ── Save model + predictions (drop-in for stats_analysis.py / DeLong) ──
    preds_path = os.path.join(script_dir, f'baseline_transfer_{tag}_preds.npz')
    np.savez(preds_path,
             val_true=val_true, val_probs=val_probs,
             test_true=test_true, test_probs=test_probs, threshold=thr)
    print(f"\n✅ Saved {model_path}")
    print(f"✅ Saved {preds_path}")
    print("\nNext: add this model to stats_analysis.py MODELS to get a paired DeLong test")
    print("vs the from-scratch champion (both feed [0,1] images — preprocessing matches).")


if __name__ == '__main__':
    main()
