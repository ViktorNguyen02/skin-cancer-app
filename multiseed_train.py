"""
multiseed_train.py — Train the CHAMPION config (Run 4: SE-Residual + multi-scale
fusion + metadata) across multiple random seeds, for statistical reporting.

WHY: a single training run is one draw. Two unseeded Run-4 draws already differed
by ~0.007 AUC and flipped the bkl↔mel benefit. To make defensible claims a paper
must report mean ± std (and ideally CIs) over several seeds. This script produces
exactly that, headlessly and resumably.

This MIRRORS model.py's architecture and config EXACTLY (focal γ=1.5 + label
smoothing 0.05, SE-residual stages 64→512, multi-scale fusion, metadata branch
Dense(32)→Dense(16), Adam 3e-4 clipnorm 1.0, val_auc checkpoint/early-stop,
ReduceLROnPlateau, 60 epochs). If you change model.py, update this too.

Usage — one seed per invocation (resumable; re-runs skip finished seeds):
    python multiseed_train.py 1
    python multiseed_train.py 2
    # or a loop (each ~several hours):
    for s in 1 2 3 4 5; do python multiseed_train.py $s; done

Outputs (in ./multiseed/):
    preds_seed<seed>.npz   — val_true, val_probs, test_true, test_probs
    model_seed<seed>.h5    — the trained model
    metrics.csv            — one row per seed (auc, thr, sens, spec, ece, epochs)

Aggregate with: python stats_analysis.py  (auto-detects preds_seed*.npz)
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Input,
                                      BatchNormalization, GlobalAveragePooling2D,
                                      Activation, Add, Multiply, Reshape, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf

# ── Config (must match model.py) ──────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1 = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2 = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')
out_dir       = os.path.join(script_dir, 'multiseed')
os.makedirs(out_dir, exist_ok=True)

CLINICAL_GROUPS = {
    'nv': 'Benign', 'bkl': 'Benign', 'df': 'Benign', 'vasc': 'Benign',
    'mel': 'Concerning', 'bcc': 'Concerning', 'akiec': 'Concerning',
}
GROUP_ORDER        = ['Benign', 'Concerning']
CONCERNING_IDX     = 1
IMG_SIZE           = 224
BATCH_SIZE         = 32
EPOCHS             = 60
TARGET_SENS        = 0.90
VAL_SELECTION_SENS = 0.94


# ── Focal loss (γ=1.5, label smoothing 0.05) — identical to model.py ──────────
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


# ── Architecture (identical to model.py) ──────────────────────────────────────
def se_block(x, ratio=16):
    c = x.shape[-1]
    s = GlobalAveragePooling2D()(x)
    s = Dense(max(c // ratio, 8), activation='relu')(s)
    s = Dense(c, activation='sigmoid')(s)
    s = Reshape((1, 1, c))(s)
    return Multiply()([x, s])


def se_residual_block(x, filters, dropout):
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    f = Conv2D(filters, 3, padding='same')(x)
    f = BatchNormalization()(f); f = Activation('relu')(f)
    f = Conv2D(filters, 3, padding='same')(f)
    f = BatchNormalization()(f)
    f = se_block(f); f = Add()([f, shortcut]); f = Activation('relu')(f)
    f = MaxPooling2D(2, 2)(f); f = Dropout(dropout)(f)
    return f


def build_model(meta_dim, num_classes):
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    x = Conv2D(32, 3, padding='same')(image_input)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    s1 = se_residual_block(x,  64,  0.10)
    s2 = se_residual_block(s1, 128, 0.15)
    s3 = se_residual_block(s2, 256, 0.20)
    s4 = se_residual_block(s3, 384, 0.20)
    s5 = se_residual_block(s4, 512, 0.25)
    ms = Concatenate()([GlobalAveragePooling2D()(s3),
                        GlobalAveragePooling2D()(s4),
                        GlobalAveragePooling2D()(s5)])
    meta_input = Input(shape=(meta_dim,), name='meta_input')
    m = Dense(32, activation='relu')(meta_input)
    m = Dense(16, activation='relu')(m)
    fused = Concatenate()([ms, m])
    h = Dense(256, activation='relu')(fused)
    h = BatchNormalization()(h); h = Dropout(0.40)(h)
    outputs = Dense(num_classes, activation='softmax')(h)
    return Model([image_input, meta_input], outputs, name='SE_Residual_MultiScale_CNN_Meta')


# ── Metadata encoder + two-input Sequence (identical to model.py) ─────────────
def fit_metadata_encoder(train_df):
    age_median = float(train_df['age'].median())
    age_filled = train_df['age'].fillna(age_median)
    return {'age_median': age_median, 'age_mean': float(age_filled.mean()),
            'age_std': float(age_filled.std() + 1e-6),
            'sex_cats': sorted(train_df['sex'].dropna().unique().tolist()),
            'loc_cats': sorted(train_df['localization'].dropna().unique().tolist())}


def transform_metadata(df, enc):
    age   = df['age'].fillna(enc['age_median'])
    age_z = ((age - enc['age_mean']) / enc['age_std']).to_numpy(dtype='float32').reshape(-1, 1)
    sex_oh = np.stack([(df['sex'] == c).to_numpy(dtype='float32') for c in enc['sex_cats']], axis=1)
    loc_oh = np.stack([(df['localization'] == c).to_numpy(dtype='float32') for c in enc['loc_cats']], axis=1)
    return np.concatenate([age_z, sex_oh, loc_oh], axis=1).astype('float32')


class ImageMetaSequence(Sequence):
    def __init__(self, df, meta, datagen, augment, batch_size, img_size, shuffle):
        self.df = df.reset_index(drop=True)
        self.meta = meta.astype('float32')
        self.labels = self.df['group_label'].to_numpy()
        self.paths = self.df['path'].tolist()
        self.datagen = datagen; self.augment = augment
        self.batch_size = batch_size; self.img_size = img_size; self.shuffle = shuffle
        self.class_indices = {g: i for i, g in enumerate(GROUP_ORDER)}
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    @property
    def classes(self):
        return self.labels

    def reset(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, i):
        bi = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.zeros((len(bi), self.img_size, self.img_size, 3), dtype='float32')
        for j, idx in enumerate(bi):
            xi = img_to_array(load_img(self.paths[idx], target_size=(self.img_size, self.img_size)))
            if self.augment:
                xi = self.datagen.random_transform(xi)
            xi = self.datagen.standardize(xi)
            X[j] = xi
        y = to_categorical(self.labels[bi], num_classes=len(GROUP_ORDER))
        return [X, self.meta[bi]], y


class ValAUC(tf.keras.callbacks.Callback):
    def __init__(self, val_gen):
        super().__init__(); self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = self.model.predict(self.val_gen, verbose=0)
        y_true = np.array(self.val_gen.classes)
        logs['val_auc'] = roc_auc_score(y_true, probs[:, CONCERNING_IDX])
        print(f"  — val_auc: {logs['val_auc']:.4f}")


def predict_probs(model, gen):
    return model.predict(gen, verbose=0)[:, CONCERNING_IDX]


def ece_score(probs2, y_int, n_bins=10):
    conf = probs2.max(axis=1); pred = probs2.argmax(axis=1)
    acc = (pred == y_int).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ══════════════════════════════════════════════════════════════════════════════
def main(seed):
    preds_path = os.path.join(out_dir, f'preds_seed{seed}.npz')
    if os.path.exists(preds_path):
        print(f"✅ seed {seed} already done ({preds_path}) — skipping.")
        return

    print(f"\n{'='*60}\nTRAINING CHAMPION — seed {seed}\n{'='*60}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Data (deterministic split — same for every seed; only training stochasticity varies)
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

    # Imbalance: sqrt-softened balanced alpha (single correction) — matches model.py
    cw = compute_class_weight('balanced', classes=np.unique(train_df['group_label']),
                              y=train_df['group_label'])
    alpha_raw = np.sqrt(cw).astype(np.float32)
    alpha = (alpha_raw / alpha_raw.sum()).astype(np.float32)

    enc = fit_metadata_encoder(train_df)
    meta_train = transform_metadata(train_df, enc)
    meta_val   = transform_metadata(val_df,   enc)
    meta_test  = transform_metadata(test_df,  enc)
    meta_dim = meta_train.shape[1]

    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, vertical_flip=True, zoom_range=0.2,
        brightness_range=[0.9, 1.1], shear_range=0.1, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = ImageMetaSequence(train_df, meta_train, train_datagen, True,  BATCH_SIZE, IMG_SIZE, True)
    val_gen   = ImageMetaSequence(val_df,   meta_val,   test_datagen,  False, BATCH_SIZE, IMG_SIZE, False)
    test_gen  = ImageMetaSequence(test_df,  meta_test,  test_datagen,  False, BATCH_SIZE, IMG_SIZE, False)

    model = build_model(meta_dim, len(GROUP_ORDER))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
                  loss=make_focal_loss(1.5, alpha, 0.05), metrics=['accuracy'])

    model_path = os.path.join(out_dir, f'model_seed{seed}.h5')
    callbacks = [
        ValAUC(val_gen),
        EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    epochs_trained = len(history.history['loss'])

    # Predict val + test, save raw predictions (the artifact stats_analysis.py consumes)
    val_probs  = predict_probs(model, val_gen)
    test_probs = predict_probs(model, test_gen)
    val_true   = val_df['group_label'].to_numpy()
    test_true  = test_df['group_label'].to_numpy()
    np.savez(preds_path, val_true=val_true, val_probs=val_probs,
             test_true=test_true, test_probs=test_probs)

    # Per-seed metrics at the val-margin operating point
    fpr, tpr, thr = roc_curve(val_true, val_probs)
    elig = np.where(tpr >= VAL_SELECTION_SENS)[0]
    idx = elig[np.argmin(fpr[elig])] if len(elig) else int(np.argmax(tpr))
    threshold = float(thr[idx])
    pred = (test_probs >= threshold).astype(int)
    test_auc  = roc_auc_score(test_true, test_probs)
    test_sens = recall_score(test_true, pred)
    test_spec = recall_score(1 - test_true, 1 - pred)
    test_probs2 = np.column_stack([1 - test_probs, test_probs])
    ece = ece_score(test_probs2, test_true)

    metrics_csv = os.path.join(out_dir, 'metrics.csv')
    new = not os.path.exists(metrics_csv)
    with open(metrics_csv, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['seed', 'test_auc', 'threshold', 'test_sens', 'test_spec', 'ece', 'epochs'])
        w.writerow([seed, f'{test_auc:.4f}', f'{threshold:.4f}', f'{test_sens:.4f}',
                    f'{test_spec:.4f}', f'{ece:.4f}', epochs_trained])

    print(f"\nseed {seed}: AUC={test_auc:.4f}  thr={threshold:.3f}  "
          f"Sens={test_sens:.4f}  Spec={test_spec:.4f}  ECE={ece:.4f}  (epochs={epochs_trained})")
    print(f"✅ saved {preds_path}, {model_path}, appended {metrics_csv}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python multiseed_train.py <seed>   (e.g. 1, 2, 3, 4, 5)")
        sys.exit(1)
    main(int(sys.argv[1]))
