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
| 2 | Colour constancy (Shades-of-Gray) | ❌ rejected (AUC regression) | `best_model_run2_color.h5` |
| 3 | MixUp | ❌ rejected (AUC regression + Sens<0.90) | `best_model_run3_mixup.h5` |
| 4 | Metadata fusion (age/sex/localization) | ⬜ next (on Run 1) | `best_model_run4_meta.h5` |

---

## 5. Results so far

### Performance trajectory
| Stage | AUC | mel rec | bkl clear | Sens/Spec @ thr | ECE | Notes |
|---|---|---|---|---|---|---|
| Original plain CNN (1st result) | 0.866 | — | — | — | — | pre-methodology |
| Plain CNN + Stages 0/1/3 (honest) | 0.867 | — | — | 0.877/0.699 @0.382 | — | leak removed |
| **Run 0** — Stage A SE-Residual | **0.883** | 85.9% | 32.1% | 0.88/0.71 @0.143 | — | ablation baseline |
| **Run 1** — + calibration | **0.895** | 87.7% | 33.2% | **0.910/0.698** @0.122 | **0.038** | +0.012 AUC ✅ **champion** |
| ~~Run 2~~ — + colour constancy | ~~0.871~~ | 87.2% | 27.9% | 0.915/0.652 @0.013 | 0.083 (T=2.29) | −0.024 AUC ❌ **rejected** |
| ~~Run 3~~ — + MixUp | ~~0.8815~~ | 85.5% | 36.3% | 0.887/0.718 @0.058 | 0.080 (T=1.55) | −0.0135 AUC, Sens<0.90 ❌ **rejected** |

> ⚠️ At-threshold recall/clearance is **confounded** when the threshold changes between runs.
> The clean comparison is **AUC** + **mean P(Concerning) per class**.
> Run 2's apparent Sens gain (0.910→0.915) is a threshold artefact — measured at thr=0.013 vs 0.122 on a *worse* ROC curve.
> Run 3's bkl-clearance gain (33%→36%) is likewise a threshold artefact (AUC fell), and its test Sens 0.887 **broke the 0.90 floor** — the val threshold did not transfer.

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

### Run 2 — ❌ REJECTED — Colour constancy (Shades-of-Gray)

**Model file:** `best_model_run2_color.h5` (94.5 MB) · **Status:** trained and scored; rejected — AUC regressed.

#### What was changed (single-change ablation on top of Run 1)

- Added `shades_of_gray(img, power=6)` as `preprocessing_function` on all three generators; `rescale=None` (function already returns [0,1]).
- `MODEL_PATH = 'best_model_run2_color.h5'`.
- All other config carried forward from Run 1 unchanged (focal γ=1.5, label smoothing, val_auc checkpointing, etc.).
- Both check scripts updated with `USE_COLOR_CONSTANCY = True` flag + conditional datagen. Two check3 bugs also fixed as bonus (stale label, vasc-only plot).

#### Actual results

**check2_prob_distribution.py**
```
Val-selected screening threshold: 0.013
  Val  — Sens: 0.9032  Spec: 0.6629
Test AUC:                        0.8714
Test @ val-thr=0.013:
  Sensitivity:                   0.9152
  Specificity:                   0.6520
Probability range (Concerning):  0.000 – 0.737
Probability range (Benign):      0.000 – 0.757
```

**Temperature scaling:** T = 2.291 · ECE 0.1495 → 0.0830 (model is now **badly overconfident** — very different from Run 1's T≈1.0).

**check3_validation.py — Check 3A (lesion isolation): ALL CLEAN**

**check3_validation.py — Check 3B (per-diagnosis @ thr 0.013)**
```
Diagnosis    n     Mean P(Conc)  Recall@0.013  Recall@0.5  Median P
mel          227   0.116         87.2%          2.6%        0.052
bcc          110   0.134         97.3%          0.9%        0.088
akiec         52   0.180         98.1%          3.8%        0.123
nv          1371   0.019         71.4%         99.8%        0.006   (clearance)
bkl ⚠️       190   0.047         27.9%        100.0%        0.023   (clearance)
df  ⚠️        20   0.037         40.0%        100.0%        0.026   (clearance)
vasc ⚠️       31   0.041         35.5%        100.0%        0.036   (clearance)
```

#### Run 1 vs Run 2 — head-to-head comparison

| Metric | Run 1 | Run 2 | Δ | Verdict |
|---|:---:|:---:|:---:|:---:|
| **Test AUC** | **0.895** | **0.871** | **−0.024** | ❌ major regression |
| Specificity @ Sens 0.90 | 0.698 | 0.652 | −0.046 | ❌ |
| bkl clearance | 33.2% | 27.9% | −5.3pp | ❌ **(target was +12pp)** |
| mean P(bkl) | 0.211 | 0.047 | −0.164 | ⚠️ dropped but so did everything |
| mel recall @ thr | 87.7% | 87.2% | −0.5pp | ~flat |
| ECE (raw) | 0.038 | 0.149 | +0.111 | ❌ much worse calibration |
| ECE (post-T) | 0.038 | 0.083 | +0.045 | ❌ |
| Temperature T | 0.992 | 2.291 | — | ❌ wildly overconfident |
| Val-selected threshold | 0.122 | 0.013 | — | ⚠️ near-zero (fragile) |
| Apparent Sens | 0.910 | 0.915 | +0.005 | ⚠️ threshold artefact only |

**Mean P(Concerning) per class — the unconfounded signal:**

| Class | Run 1 mean P | Run 2 mean P | Δ | Direction needed |
|---|:---:|:---:|:---:|:---:|
| mel | 0.319 | 0.116 | −0.203 | ↑ higher |
| bcc | 0.362 | 0.134 | −0.228 | ↑ higher |
| akiec | 0.438 | 0.180 | −0.258 | ↑ higher |
| nv | 0.080 | 0.019 | −0.061 | ↓ lower |
| bkl | 0.211 | 0.047 | −0.164 | ↓ lower |

Every class shrank proportionally — the entire distribution collapsed toward zero. The threshold dropped from 0.122 → 0.013 to compensate, but class separation (= AUC) is what suffered.

#### Root cause: Shades-of-Gray violates grey-world on dermoscopy

The grey-world/Minkowski assumption is *"average scene reflectance is achromatic."* A HAM10000 frame is filled edge-to-edge with skin and lesion — it is **not** grey on average; it's pink/brown. The algorithm's illuminant estimate is dominated by genuine diagnostic colour (the brown of bkl/akiec, the pink of bcc, the blue-white of mel). Dividing it out **erases the colour signal the model was ranking on**, not just the cross-source tint. Dermoscopy also uses controlled polarized/immersion lighting, so there was less illuminant variation to fix than the method assumes.

**Verdict: Shades-of-Gray is categorically unsuitable for dermoscopy.** Config NOT carried forward.

#### What to try instead (for Run 3)
Build on **Run 1**, not Run 2. The colour confusion between bkl and akiec is better addressed by MixUp regularization (which implicitly creates soft colour boundaries between classes) or metadata fusion (age/localization fully separates bkl from akiec without touching image colour).

---

### Run 3 — ❌ REJECTED — MixUp (λ~Beta(0.2,0.2))

**Model file:** `best_model_run3_mixup.h5` (94.5 MB) · **Status:** trained and scored; rejected — AUC regressed and test Sens fell below the 0.90 floor.

#### What was changed (single-change ablation on top of Run 1)

- Reverted Run 2's colour constancy (back to `rescale=1./255`).
- Added `mixup_generator(gen, alpha=0.2)` wrapping the **train** generator only; `model.fit` consumes it with `steps_per_epoch=len(train_generator)`. Val/test stay raw.
- No loss change — focal loss handles soft/blended labels natively (`p_t = Σ(y_true·y_pred)`).
- `MODEL_PATH = 'best_model_run3_mixup.h5'`.

#### Actual results

**check2_prob_distribution.py**
```
Val-selected screening threshold: 0.058
  Val  — Sens: 0.9032  Spec: 0.7091
Test AUC:                        0.8815
Test @ val-thr=0.058:
  Sensitivity:                   0.8869      ← below 0.90 floor
  Specificity:                   0.7177
Probability range (Concerning):  0.015 – 0.756
Probability range (Benign):      0.000 – 0.755
```

**Temperature scaling:** T = 1.548 · ECE 0.1124 → 0.0803 (overconfident; worse than Run 1's T≈1.0 / ECE 0.038).

**check3_validation.py — Check 3A (lesion isolation): ALL CLEAN**

**check3_validation.py — Check 3B (per-diagnosis @ thr 0.058)**
```
Diagnosis    n     Mean P(Conc)  Recall@0.058  Recall@0.5  Median P
mel          227   0.166         85.5%          2.6%        0.118
bcc          110   0.214         92.7%          4.5%        0.173
akiec         52   0.273         94.2%          5.8%        0.277
nv          1371   0.040         78.0%         99.8%        0.021   (clearance)
bkl ⚠️       190   0.106         36.3%        100.0%        0.078   (clearance)
df  ⚠️        20   0.112         40.0%        100.0%        0.110   (clearance)
vasc ⚠️       31   0.160         35.5%        100.0%        0.121   (clearance)
```

#### Run 1 vs Run 3 — head-to-head comparison

| Metric | Run 1 (champ) | Run 3 | Δ | Verdict |
|---|:---:|:---:|:---:|:---:|
| **Test AUC** | **0.895** | **0.8815** | **−0.0135** | ❌ regression (≈ Run 0 baseline) |
| Test Sens @ thr | 0.910 | 0.887 | −0.023 | ❌ **below 0.90 floor** |
| Test Spec @ thr | 0.698 | 0.718 | +0.020 | ✅ best of any run (confounded) |
| bkl clearance | 33.2% | 36.3% | +3.1pp | ✅ best of any run (confounded) |
| mel recall | 87.7% | 85.5% | −2.2pp | ❌ |
| ECE (raw) | 0.038 | 0.112 | +0.074 | ❌ |
| ECE (post-T) | 0.038 | 0.080 | +0.042 | ❌ |
| Temperature T | 0.992 | 1.548 | — | ❌ overconfident |
| PPV @ argmax | 0.609 | 0.824 | +0.215 | ✅ (confounded) |

Mean P per class roughly halved vs Run 1 (mel 0.319→0.166, bkl 0.211→0.106) — the same proportional
compression seen in Run 2, milder. The malignant classes did **not** separate further from benign
(AUC fell), so the bkl/specificity gains are threshold artefacts, not real separation.

#### Root cause: over-regularization of an already-regularized model

The model already carries heavy regularization (dropout 0.10–0.40, BatchNorm, label smoothing 0.05,
strong augmentation). MixUp's vicinal smoothing added on top **over-regularized the confident
region** — pulling predictions toward the middle, compressing the distribution, lowering ranking
quality (AUC), and making the val-selected threshold less transferable (val Sens 0.903 → test 0.887).
The train/val loss gap MixUp was meant to close persisted (train→0.055, val climbed to ~0.19).

**Verdict: MixUp does not help here. Config NOT carried forward. Run 1 remains champion.**
A publishable negative result: MixUp does not improve an already well-regularized from-scratch
dermoscopy CNN.

#### What to try instead (for Run 4)
Run 4 (metadata fusion) attacks bkl via a **non-image** signal — patient age and lesion localization
strongly separate bkl (older, trunk) from akiec (sun-exposed face/hands), something neither image-only
run (colour constancy, MixUp) could do. Built on the Run 1 champion config.

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
6. **Shades-of-Gray colour constancy is unsuitable for dermoscopy.** The grey-world assumption
   requires average scene reflectance to be achromatic — dermoscopy images are filled with skin
   and lesion colour, which is not grey. The algorithm erases the diagnostic colour signal rather
   than just the cross-source tint. Run 2 proved this empirically (AUC −0.024, ECE ×4 worse,
   threshold collapsed to 0.013). A negative result but a publishable finding.
7. **Apparent sensitivity gains across runs can be pure threshold artefacts.** Run 2 showed
   Sens 0.915 vs Run 1's 0.910 — but this was measured at thr=0.013 vs 0.122 on a *worse* ROC
   curve. Always verify with AUC, never accept higher sensitivity alone as evidence of improvement.
8. **MixUp over-regularizes an already-regularized model.** Run 3 added MixUp on top of dropout +
   BN + label smoothing + strong augmentation; AUC regressed 0.895→0.8815 and test Sens fell to
   0.887 (below the 0.90 floor). The distribution compressed (mean P halved) without improving
   ranking. → Stack regularizers cautiously; more is not always better. Negative result, publishable.
9. **Watch val→test threshold transfer, not just val Sens.** Run 3's val threshold gave val
   Sens 0.903 but only test Sens 0.887. A run can pass on validation and fail the screening floor
   on test. Always report the test-set sensitivity at the val-selected threshold.

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

### Run 2 — DONE (rejected). See §5 for results and root-cause analysis.
### Run 3 — DONE (rejected). See §5 for results and root-cause analysis.

> **Two consecutive image-only interventions failed** (colour constancy, MixUp). Run 1 is still
> champion at AUC 0.895. The pattern is clear: the image branch is near its ceiling — the remaining
> bkl↔akiec confusion needs a **non-image signal**. That is exactly what Run 4 provides.

### Run 4 — Metadata fusion (NEXT — biggest lever, built on Run 1)
- **Start from the Run 1 champion config.** model.py has been reverted: `rescale=1./255`, no MixUp
  wrapper, `model.fit(train_generator, ...)`. Set `MODEL_PATH = 'best_model_run4_meta.h5'`.
- Encode age (median-impute, standardize) / sex / localization (one-hot). **Fit encoders on TRAIN
  only.** Replace `flow_from_dataframe` with a custom `tf.keras.utils.Sequence` yielding
  `([image, meta], y)`. Add metadata branch Dense(32)→Dense(16), concat with the 1152-d image
  vector, then existing head. **Check scripts must be updated to feed `[image, meta]`.**
- Expect +1–3% AUC; largest gains in bkl clearance + mel recall (age/localization separate bkl from
  akiec without touching image colour — the one lever the two failed image-only runs lacked).

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
| `best_model_run2_color.h5` | Run 2 (colour constancy) — ❌ rejected; AUC 0.871 (−0.024 vs Run 1) |
| `best_model_run3_mixup.h5` | Run 3 (MixUp) — ❌ rejected; AUC 0.8815 (−0.0135 vs Run 1, Sens<0.90) |
| `EXPERIMENT_TRACKING.md` | This file |

_Last updated: after Run 3 training + scoring. Runs 2 & 3 both rejected; Run 1 (`best_model_run1_calib.h5`, AUC 0.895) remains champion. model.py reverted to Run 1 config. Next action: implement Run 4 (metadata fusion)._
