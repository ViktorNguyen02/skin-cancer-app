# Experiment Tracking — Skin Lesion Binary CNN (Concerning vs Benign)

> Dataset: HAM10000 (7 classes → 2 clinical groups)
> Concerning = mel, bcc, akiec · Benign = nv, bkl, df, vasc
> Constraint: **from-scratch CNN only — NO pretrained models, NO transfer learning.**
> Test set: held-out, leak-free (split by `lesion_id`, `random_state=42`). Fixed across all runs.

---

## 1. Project goal & target metrics

Build a publishable custom CNN for skin-lesion screening. The model is evaluated as a
**screening tool**: the headline metric is **AUC** (threshold-independent), and the operating
point is a **threshold selected on validation** (target Sensitivity ≥ 0.90) then applied to test.

| Metric | Acceptable | Strong / competitive |
|---|---|---|
| AUC | ≥ 0.85 | **≥ 0.90** (SOTA 0.92–0.95) |
| Sensitivity @ operating point | ≥ 0.85 | **≥ 0.90–0.95** |
| Specificity @ that sensitivity | ≥ 0.65 | **≥ 0.80** |
| ECE (calibration error) | < 0.10 | < 0.05 |

Anchor: dermatologist baselines in the literature ≈ 0.86 sens / 0.71 spec.

---

## 2. Methodology fixes (applied before the ablation) — Stages 0/1/3

These fixed how the model is trained/selected/reported. Applied to the original plain 4-block CNN.

### Stage 0 — 3-way split (data leak fix)
- **Problem:** `test_generator` was used as `validation_data` AND for checkpoint/early-stop
  selection AND for final reporting → every reported number was optimistically biased.
- **Fix:** two-stage `train_test_split` by `lesion_id`, stratified by `group_label`:
  train ≈ 68% / val ≈ 12% / test = 20%. Validation drives all model selection + threshold
  tuning; test scored once. Asserts confirm no lesion crosses splits.

### Stage 1 — Stable training & trustworthy checkpoint
- **Monitor `val_auc`** (threshold-independent, smooth) instead of `val_balanced_acc`
  (oscillated 0.50↔0.74, caused "lucky-epoch" checkpoints). `ValMetrics` callback computes
  val_auc / val_balanced_acc / val_macro_f1 each epoch.
- **LR tamed:** 5e-4 → 3e-4, `clipnorm=1.0` on Adam, `ReduceLROnPlateau(monitor='val_auc',
  factor=0.5, patience=4)` replacing fixed CosineDecay. Killed the epoch-8 val_loss spike.
- **Softened augmentation:** rotation 180→90, brightness [0.8,1.2]→[0.9,1.1].

### Stage 3 — Honest threshold policy
- Screening threshold chosen on **validation** (smallest-FPR threshold with Sens ≥ 0.90),
  then applied **fixed** to test. Lead reporting with AUC + this operating point.
- **Do NOT report argmax-0.5 metrics as headline** — they are misleading on imbalanced data.

---

## 3. Architecture — Stage A (the ablation baseline = "Run 0")

Replaced the plain 4-block Sequential CNN with a **from-scratch SE-Residual CNN with
multi-scale fusion** (functional API, `model.py`). This is the paper's architectural contribution.

- **Stem:** Conv2D(32,3) + BN + ReLU.
- **5 SE-residual stages** (`se_residual_block`): filters 64→128→256→384→512, each = two 3×3
  convs + **SE channel attention** (`se_block`) + **residual skip** (1×1 projection) + MaxPool +
  Dropout. Spatial: 112→56→28→14→7.
- **Multi-scale fusion:** GAP of stages 3+4+5 concatenated → 1152-d (`Concatenate`).
- **Head:** Dense(256,relu) → BN → Dropout(0.4) → Dense(2, softmax).
- **Params: 8,221,530.** IMG_SIZE=224, **BATCH_SIZE=32** (lowered from 64 for the 16 GB M4 —
  multi-scale fusion holds 3 feature maps alive at once → higher RAM).
- **Loss:** focal loss, sqrt-softened balanced alpha (single imbalance correction).

### Three mechanisms & why
1. **Residual connections** — keep gradients healthy → enables depth (5 stages) from scratch.
2. **Squeeze-and-Excitation attention** — re-weights channels toward concerning markers.
3. **Multi-scale fusion** — lesion scale varies; classify on fine + coarse features together.

---

## 4. The 4-run cumulative ablation plan

One change per ~8h overnight run, each measured vs the previous (paper-ready ablation table).
Each run saves to its own `.h5`. Config carried forward only from runs that improve AUC.

| Run | Adds | Status | Model file |
|---|---|---|---|
| 0 | Baseline SE-Residual | ✅ done | `best_model_binary.h5` |
| 1 | Calibration (γ=1.5 + label smoothing + temp scaling) | ✅ done | `best_model_run1_calib.h5` |
| 2 | Colour constancy (Shades-of-Gray) | 🔄 code done; training pending | `best_model_run2_color.h5` |
| 3 | MixUp | ⬜ | `best_model_run3_mixup.h5` |
| 4 | Metadata fusion (age/sex/localization) | ⬜ | `best_model_run4_meta.h5` |

---

## 5. Results so far

### Performance trajectory
| Stage | AUC | mel rec | bkl clear | Sens/Spec @ thr | ECE | Notes |
|---|---|---|---|---|---|---|
| Original plain CNN (1st result) | 0.866 | — | — | — | — | pre-methodology |
| Plain CNN + Stages 0/1/3 (honest) | 0.867 | — | — | 0.877/0.699 @0.382 | — | leak removed |
| **Run 0** — Stage A SE-Residual | **0.883** | 85.9% | 32.1% | 0.88/0.71 @0.143 | — | ablation baseline |
| **Run 1** — + calibration | **0.895** | 87.7% | 33.2% | **0.910/0.698** @0.122 | **0.038** | +0.012 AUC |

> ⚠️ At-threshold recall/clearance is **confounded** when the threshold changes between runs
> (0.143 → 0.122). The clean comparison is **AUC** + **mean P(Concerning) per class**.

### Run 1 — full check outputs (model: `best_model_run1_calib.h5`)

**check2_prob_distribution.py**
```
Val-selected screening threshold: 0.122
  Val  — Sens: 0.9032  Spec: 0.6876
Test AUC:                        0.8950
Test @ val-thr=0.122:
  Sensitivity:                   0.9100
  Specificity:                   0.6979
Probability range (Concerning):  0.016 – 0.831
Probability range (Benign):      0.000 – 0.922
```

**check3_validation.py — Check 3A (lesion isolation): ALL CLEAN**
```
Train lesions: 5079 | Val lesions: 897 | Test lesions: 1494 | Total unique: 7470
Train images:  6824 | Val images: 1190 | Test images: 2001
Train∩Val, Train∩Test, Val∩Test overlaps = 0 (lesion AND image level)
```

**check3_validation.py — Check 3B (per-diagnosis @ thr 0.122)**
```
Diagnosis    n     Mean P(Conc)  Recall*   Recall@0.5  Median P
mel          227   0.319         87.7%     17.2%       0.294
bcc          110   0.362         94.5%     20.9%       0.353
akiec         52   0.438         98.1%     42.3%       0.468
nv          1371   0.080         76.3%     99.6%       0.040   (clearance)
bkl ⚠️       190   0.211         33.2%     95.8%       0.177   (clearance)
df  ⚠️        20   0.248         25.0%    100.0%       0.211   (clearance, tiny n)
vasc ⚠️       31   0.206         35.5%     96.8%       0.165   (clearance, tiny n)
```
*Concerning rows = recall (% caught); Benign rows = specificity (% correctly cleared).
NOTE: the script's column header literally prints "Recall@0.143" — STALE LABEL; the actual
threshold used is 0.122. (Bug to fix in the check3 script.)

**Temperature scaling:** T = 0.992 · ECE 0.0376 → 0.0381 (no-op; model already calibrated).

### What Run 1 actually achieved
- **+0.012 AUC** (0.883 → 0.895), genuine and threshold-independent.
- The gain came from **better ranking of the malignant classes** (bcc mean P +0.08, akiec +0.06,
  mel +0.03), **NOT** from resolving bkl (bkl mean P 0.198→0.211, clearance ~unchanged at 33%).
- Excellent val→test generalization (val Sens 0.9032 → test 0.9100).

---

### Run 2 — Code implementation (training not yet complete)

**Model file:** `best_model_run2_color.h5` · **Status:** code complete; overnight training pending.

#### Root problem this run targets

bkl (seborrheic keratosis) and akiec (actinic keratosis) share brown/tan pigmentation and scale.
HAM10000 pools images from multiple imaging sources with different white-balance/illumination
settings. After Run 1 the model scored bkl with a mean P(Concerning) of 0.211 — above the 0.122
screening threshold — causing 67% of bkl images to be wrongly flagged as suspicious. The model
couldn't distinguish *"tan because it's akiec"* from *"tan because of this camera's white-balance."*
Shades-of-Gray colour constancy renders every image under a canonical neutral illuminant before
training, removing the cross-source tint and forcing the model to learn true colour/texture
differences between the classes.

#### Code changes made

**`model.py`**

1. Added `shades_of_gray(img, power=6)` function immediately before the generators cell:

```python
def shades_of_gray(img, power=6):
    img = img.astype('float32')
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + 1e-6) * np.sqrt(3.0)
    img = img / (illum + 1e-6)
    img = np.clip(img, 0, 255)
    return img / 255.0
```

  The Minkowski-6 norm estimates the per-channel illuminant; dividing it out removes the global
  colour cast; the result is already normalised to [0,1] (no double-scaling).

2. `train_datagen` and `test_datagen`: `rescale=1./255` → `rescale=None, preprocessing_function=shades_of_gray`. All augmentation (rotation, brightness, flip, zoom) kept identical to Run 1.

3. `MODEL_PATH = 'best_model_run2_color.h5'`.

Everything else (focal loss γ=1.5, label smoothing ε=0.05, val_auc checkpointing,
ReduceLROnPlateau, clipnorm=1.0, temperature scaling in evaluation) is **carried forward unchanged
from Run 1** — clean single-change ablation.

**`check2_prob_distribution.py`** and **`check3_validation.py`** (both scripts updated together)

- `model_path` → `'best_model_run2_color.h5'`
- Added `USE_COLOR_CONSTANCY = True` flag + the identical `shades_of_gray` function.
- Conditional datagen: `if USE_COLOR_CONSTANCY: datagen = ImageDataGenerator(rescale=None, preprocessing_function=shades_of_gray) else: datagen = ImageDataGenerator(rescale=1./255)`.
- **⚠️ Critical:** set `USE_COLOR_CONSTANCY = False` when re-scoring Run 0/1 — a
  colour-normalised model scored on raw images (or vice-versa) gives garbage probabilities.

Also fixed two long-standing bugs in `check3_validation.py`:

| Bug | Fix |
|---|---|
| Stale column label "Recall@0.143" regardless of actual threshold | Changed to `f"Recall@{screen_thr:.3f}"` (dynamic) |
| Per-dx plot only rendered vasc (divmod indexing bug) | Replaced with explicit `BOTTOM_BENIGN = ['nv', 'bkl']` — now shows nv + bkl (the target class) |

#### Expected outcome (not yet measured)

| Metric | Run 1 | Run 2 target |
|---|---|---|
| Test AUC | 0.895 | ~0.90 |
| bkl clearance | 33.2% | **≥ 45%** |
| mean P(bkl) | 0.211 | ↓ toward 0.12 |
| Specificity @ Sens 0.90 | 0.698 | ~0.75 |
| mel recall | 87.7% | ≥ 87% (must not regress) |

Update this section with the actual check3/check2 outputs after the overnight run completes.

---

## 6. Key learnings & corrections (avoid repeating mistakes)

1. **The "calibration shift" diagnosis was partly WRONG.** The model was already well-calibrated
   (ECE 0.038, T≈0.99). Low argmax-0.5 sensitivity is **correct base-rate behavior** on an 80/20
   dataset, NOT miscalibration. → **Always measure ECE before calling something a calibration bug.**
2. **Label smoothing helped via REGULARIZATION (AUC), not calibration.** Keep it — it works, just
   for a different reason than originally stated.
3. **Temperature scaling is a no-op here** but still worth keeping as a publishable reliability
   figure that *proves* good calibration.
4. **Argmax-0.5 metrics (Macro F1, Balanced Acc, argmax sens) are misleading on imbalanced data.**
   Never report them as headline; use AUC + val-selected-threshold metrics.
5. **At-threshold per-class recall is confounded by threshold changes.** Compare runs via AUC and
   mean P(Concerning) per class.

---

## 7. Remaining issues (no transfer learning)

| # | Issue | Root cause | Priority |
|---|---|---|---|
| 1 | **bkl false positives** — only 33% cleared (mean P 0.211, hovering above threshold) | bkl↔akiec share brown/tan colour + scale; cross-source illumination variance | **HIGH — #1 bottleneck for specificity** |
| 2 | **AUC 0.895 < 0.92 target** | distribution overlap (~0.05–0.45) | HIGH |
| 3 | mel recall 87.7% (face value < 90%) | threshold-confounded + genuine hard cases | MEDIUM |
| 4 | df/vasc poor clearance (25% / 35.5%) | tiny n (20/31) + colour confusion (vasc=red) | LOW (noisy) |
| 5 | Training noise (val_loss spikes) | LR slightly hot for focal loss | LOW (doesn't affect checkpoint) |
| 6 | argmax-0.5 low sensitivity | base rate — correct behavior | NON-ISSUE |

### Known script bugs
- ~~`check3_validation.py`: stale header label "Recall@0.143"~~ **Fixed in Run 2 prep** (dynamic label).
- ~~`check3_validation.py`: per-diagnosis plot only renders VASC~~ **Fixed in Run 2 prep** (`BOTTOM_BENIGN = ['nv', 'bkl']`).

---

## 8. Next steps

### Run 2 — Colour constancy (code done — train overnight)
- `model.py` fully updated. Run with:
  ```bash
  source myenv311/bin/activate
  python model.py   # ~8h; saves best_model_run2_color.h5
  python check3_validation.py        # USE_COLOR_CONSTANCY=True already set
  python check2_prob_distribution.py
  ```
- After scoring, fill the Run 2 row in `ablation_table.md` and update the results table in §5.
- Watch specifically: **bkl clearance** (33% → 45%+) and **mean P(bkl)** (↓ toward 0.12).

### Run 3 — MixUp
- Wrap **train** generator: per batch λ~Beta(0.2,0.2), mix images + soft-mix labels. Focal loss
  already handles soft targets. Expect +0.5–1% AUC, smoother val_loss.

### Run 4 — Metadata fusion (biggest lever)
- Encode age (median-impute, standardize) / sex / localization (one-hot). **Fit encoders on TRAIN
  only.** Replace `flow_from_dataframe` with a custom `tf.keras.utils.Sequence` yielding
  `([image, meta], y)`. Add metadata branch Dense(32)→Dense(16), concat with the 1152-d image
  vector, then existing head. **Check scripts must be updated to feed `[image, meta]`.**
- Expect +1–3% AUC; largest gains in bkl clearance + mel recall.

### Cumulative target
**~0.91–0.92 AUC, ~90% sensitivity at ~0.80–0.82 specificity.**

---

## 9. How to run / score any run

```bash
source myenv311/bin/activate
python model.py                 # train (~8h overnight); saves best_model_runX.h5

# Score: set model_path = 'best_model_runX.h5' at top of EACH check script, then:
python check3_validation.py     # AUC + per-dx mel recall / bkl clearance + leak check
python check2_prob_distribution.py   # calibration / probability distributions
```

**Per-run checklist**
1. Set `ModelCheckpoint` filepath (`MODEL_PATH`) in `model.py` to the new run name.
2. After training, update `model_path` in BOTH check scripts to that file (default still points to
   the previous run — this caused a baseline-vs-Run1 mix-up once; verify the printed AUC matches).
3. Confirm `Params: 8,221,530 | SE_Residual_MultiScale_CNN` loaded.
4. Record AUC · mel recall · bkl clearance · Sens/Spec · ECE into the ablation table (§5).
5. Carry config forward only if AUC improved.

---

## 10. File map

| File | Purpose |
|---|---|
| `model.py` | Training pipeline (split, generators, SE-residual model, focal loss, eval) |
| `check2_prob_distribution.py` | Probability-distribution / calibration plots (val + test) |
| `check3_validation.py` | Lesion-ID isolation check + per-diagnosis recall/clearance breakdown |
| `best_model_binary.h5` | Run 0 baseline (Stage A SE-Residual, AUC 0.883) |
| `best_model_run1_calib.h5` | Run 1 (calibration, AUC 0.895) |
| `best_model_run2_color.h5` | Run 2 (colour constancy) — will be created after overnight training |
| `EXPERIMENT_TRACKING.md` | This file |

_Last updated: after Run 2 code implementation. Next action: `python model.py` overnight to train Run 2._
