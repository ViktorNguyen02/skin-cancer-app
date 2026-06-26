# Ablation Table — Skin Lesion Binary CNN (Concerning vs Benign)

> From-scratch SE-Residual CNN. **No pretrained models / no transfer learning.**
> Fixed, leak-free test set (split by `lesion_id`, `random_state=42`).
> Cumulative ablation — each run adds ONE change; config carried forward only if AUC improves.
> Headline metric = **AUC** (threshold-independent). Operating point = threshold selected on
> validation (Sens ≥ 0.90), applied to test.

---

## Main ablation table

| Run | Change added | Test AUC | mel recall | bkl clearance | Sens / Spec @ thr | thr | ECE | Status |
|-----|--------------|:--------:|:----------:|:-------------:|:-----------------:|:---:|:---:|:------:|
| 0 | Baseline SE-Residual (+ multi-scale fusion) | 0.883 | 85.9% | 32.1% | 0.88 / 0.71 | 0.143 | — | ✅ done |
| **1** | **+ Calibration** (γ 2.0→1.5, label smoothing 0.05, temp scaling) | **0.895** | 87.7% | 33.2% | **0.910 / 0.698** | 0.122 | 0.038 | ✅ done |
| 2 | + Colour constancy (Shades-of-Gray) | — | — | — | — | — | — | ⬜ next |
| 3 | + MixUp (λ~Beta(0.2,0.2)) | — | — | — | — | — | — | ⬜ |
| 4 | + Metadata fusion (age / sex / localization) | — | — | — | — | — | — | ⬜ |

> ⚠️ **At-threshold recall/clearance is confounded** when the threshold changes between runs
> (e.g. 0.143 → 0.122 catches more Concerning + clears fewer Benign mechanically). The clean
> cross-run comparison is **AUC** and **mean P(Concerning) per class** (table below).

---

## Per-diagnosis breakdown (mean P(Concerning) — the unconfounded signal)

| Diagnosis | n (test) | Run 0 mean P | Run 1 mean P | Δ | Goal direction |
|-----------|:--------:|:------------:|:------------:|:--:|----------------|
| mel (Concerning)   | 227 | 0.286 | 0.319 | +0.033 | ↑ higher |
| bcc (Concerning)   | 110 | 0.280 | 0.362 | +0.082 | ↑ higher |
| akiec (Concerning) |  52 | 0.380 | 0.438 | +0.058 | ↑ higher |
| nv (Benign)        | 1371 | 0.098 | 0.080 | −0.018 | ↓ lower |
| **bkl (Benign)** ⚠️ | 190 | 0.198 | 0.211 | +0.013 | ↓ lower — **stuck (target)** |
| df (Benign)        |  20 | 0.183 | 0.248 | +0.065 | ↓ lower (tiny n) |
| vasc (Benign)      |  31 | 0.182 | 0.206 | +0.024 | ↓ lower (tiny n) |

**Read:** Run 1's AUC gain came from pushing the malignant classes higher (bcc most), **not** from
fixing bkl — bkl barely moved and still sits above the operating threshold (→ false positives).
**bkl is the #1 target for Run 2.**

---

## Per-run change log

### Run 0 — Baseline SE-Residual CNN
- 5 SE-residual stages (64→128→256→384→512) + multi-scale fusion (GAP s3+s4+s5 = 1152-d) + head.
- Focal loss (γ=2.0, sqrt-balanced alpha). 8,221,530 params. IMG_SIZE 224, batch 32.
- Model file: `best_model_binary.h5`.

### Run 1 — Calibration
- Focal **γ 2.0 → 1.5**; **label smoothing ε=0.05** inside focal loss.
- Post-hoc **temperature scaling** + reliability diagram / ECE in evaluation.
- Model file: `best_model_run1_calib.h5`.
- **Outcome: +0.012 AUC (regularization effect). bkl unchanged.** Carry config forward. ✅
- Note: temp scaling was a no-op (T=0.992) — model already well-calibrated (ECE 0.038).

### Run 2 — Colour constancy (planned)
- Shades-of-Gray (Minkowski p=6) `preprocessing_function` on all 3 generators; `rescale=None`.
- Model file: `best_model_run2_color.h5`.
- **Targets:** bkl clearance 33% → **45%+**, mean P(bkl) ↓ toward 0.12, AUC → ~0.90.

### Run 3 — MixUp (planned)
- Wrap train generator: λ~Beta(0.2,0.2), mix images + soft labels. Model file: `best_model_run3_mixup.h5`.

### Run 4 — Metadata fusion (planned)
- age/sex/localization branch fused with image features (encoders fit on TRAIN only); custom
  two-input generator. Model file: `best_model_run4_meta.h5`. Biggest expected lever for bkl + mel.

---

## Cumulative target

**~0.91–0.92 AUC · ~90% sensitivity at ~0.80–0.82 specificity.**

_See `../EXPERIMENT_TRACKING.md` for full methodology, learnings, and raw check outputs._
_Last updated: after Run 1 scoring._
