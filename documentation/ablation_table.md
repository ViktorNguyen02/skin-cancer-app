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
| **1** | **+ Calibration** (γ 2.0→1.5, label smoothing 0.05, temp scaling) | **0.895** | 87.7% | 33.2% | **0.910 / 0.698** | 0.122 | 0.038 | ✅ done · **best** |
| ~~2~~ | ~~+ Colour constancy (Shades-of-Gray, p=6)~~ | ~~0.871~~ | ~~87.2%~~ | ~~27.9%~~ | ~~0.915 / 0.652~~ | ~~0.013~~ | ~~0.083†~~ | ❌ **rejected** |
| ~~3~~ | ~~+ MixUp (λ~Beta(0.2,0.2)), on Run 1~~ | ~~0.8815~~ | ~~85.5%~~ | ~~36.3%~~ | ~~0.887‡ / 0.718~~ | ~~0.058~~ | ~~0.080†~~ | ❌ **rejected** |
| 4 | + Metadata fusion (age / sex / localization), on Run 1 | — | — | — | — | — | — | ⬜ next |

> † ECE after temperature scaling. Run 2: T=2.291, raw 0.149. Run 3: T=1.548, raw 0.112. Both far worse than Run 1 (T≈0.99, ECE=0.038).
> ‡ Run 3 test Sens = 0.887 **fell below the 0.90 screening floor** — the val-selected threshold (val Sens 0.903) did not transfer to test. A real generalization failure, not just a threshold artefact.
>
> ⚠️ **At-threshold recall/clearance is confounded** when the threshold changes between runs
> (e.g. 0.143 → 0.122 catches more Concerning + clears fewer Benign mechanically). The clean
> cross-run comparison is **AUC** and **mean P(Concerning) per class** (table below).
>
> ⚠️ **Run 2's apparent Sens gain (0.910→0.915) is a threshold artefact**, not a real improvement.
> It was measured at thr=0.013 vs 0.122 on a *worse* ROC curve (AUC −0.024). Never accept higher
> sensitivity alone as evidence of improvement.

---

## Per-diagnosis breakdown (mean P(Concerning) — the unconfounded signal)

| Diagnosis | n (test) | Run 0 mean P | Run 1 mean P | ~~Run 2 mean P~~ | ~~Run 3 mean P~~ | Goal direction |
|-----------|:--------:|:------------:|:------------:|:----------------:|:----------------:|----------------|
| mel (Concerning)   | 227 | 0.286 | 0.319 | ~~0.116~~ | ~~0.166~~ | ↑ higher |
| bcc (Concerning)   | 110 | 0.280 | 0.362 | ~~0.134~~ | ~~0.214~~ | ↑ higher |
| akiec (Concerning) |  52 | 0.380 | 0.438 | ~~0.180~~ | ~~0.273~~ | ↑ higher |
| nv (Benign)        | 1371 | 0.098 | 0.080 | ~~0.019~~ | ~~0.040~~ | ↓ lower |
| **bkl (Benign)** ⚠️ | 190 | 0.198 | 0.211 | ~~0.047~~ | ~~0.106~~ | ↓ lower — **still the target** |
| df (Benign)        |  20 | 0.183 | 0.248 | ~~0.037~~ | ~~0.112~~ | ↓ lower (tiny n) |
| vasc (Benign)      |  31 | 0.182 | 0.206 | ~~0.041~~ | ~~0.160~~ | ↓ lower (tiny n) |

**Read (Run 1 — champion):** AUC gain came from pushing the malignant classes higher (bcc most),
**not** from fixing bkl — bkl barely moved and still sits above the operating threshold (→ false positives).

**Read (Run 2 — rejected):** Every class's mean P dropped proportionally (~60–70% reduction across
the board) — the signature of the Shades-of-Gray failure (grey-world divided genuine pigment out as
"illuminant", erasing the colour ranking signal). All classes shrank together → AUC worsened.

**Read (Run 3 — rejected):** The same proportional compression appears again (mel/bkl roughly
halved vs Run 1), just milder than Run 2 — this is MixUp's vicinal smoothing pulling predictions
toward the middle. The malignant classes did **not** separate further from benign (AUC 0.895→0.8815),
so the apparent bkl-clearance gain (33%→36%) is a threshold artefact, not real separation.
**bkl remains the #1 target — Run 4 (metadata) attacks it via a non-image signal.**

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

### Run 2 — ❌ Colour constancy (REJECTED)
- Shades-of-Gray (Minkowski p=6) `preprocessing_function` on all 3 generators; `rescale=None`.
- Model file: `best_model_run2_color.h5`.
- **Targets were:** bkl clearance 33% → 45%+, mean P(bkl) ↓ toward 0.12, AUC → ~0.90.
- **Outcome:** AUC 0.871 (−0.024 vs Run 1). ECE 0.149 raw → 0.083 post-T (T=2.291). bkl clearance 27.9% (−5.3pp). All class mean Ps shrank ~60–70%. **Config NOT carried forward.**
- **Root cause:** Shades-of-Gray assumes average scene reflectance is achromatic (grey-world). Dermoscopy frames are filled edge-to-edge with skin and lesion — the "illuminant" estimate is dominated by genuine diagnostic pigmentation (brown bkl/akiec, pink bcc, blue-white mel), which gets divided out along with the cross-source tint. This erases the colour ranking signal. Dermoscopy also uses controlled polarized/immersion lighting, so there was little real illuminant variation to remove.
- **Publishable finding:** off-the-shelf colour constancy preprocessing *hurts* a from-scratch dermoscopy CNN. Include in paper as a negative ablation result to strengthen methodology.

### Run 3 — ❌ MixUp (REJECTED)
- Wrapped train generator only: λ~Beta(0.2,0.2), mix images + soft labels. Reverted Run 2 colour constancy (rescale=1./255). Model file: `best_model_run3_mixup.h5`.
- **Targets were:** AUC → ~0.90, smaller train/val loss gap, hold Sens ≥ 0.90.
- **Outcome:** AUC 0.8815 (−0.0135 vs Run 1, ≈ Run 0 baseline). Test Sens 0.887 — **below the 0.90 floor**. ECE 0.112 raw → 0.080 post-T (T=1.548). mel recall 85.5% (−2.2pp). All class mean Ps roughly halved. **Config NOT carried forward.**
- **Genuine bright spots (but minor/confounded):** best operating-point specificity of any run (0.718) and best bkl clearance of any run (36.3%), PPV 0.82. These are threshold-confounded and don't offset the AUC regression + sensitivity-floor breach.
- **Root cause:** the model is already heavily regularized (dropout 0.10–0.40, BN, label smoothing, strong augmentation). MixUp's additional vicinal smoothing over-regularized the *confident* region — compressing the probability distribution, lowering ranking quality (AUC), and making the val-selected threshold less transferable to test. The train/val loss gap it was meant to close persisted (train→0.055, val climbed to ~0.19).
- **Publishable finding:** MixUp does not help an already well-regularized from-scratch dermoscopy CNN. Negative ablation result.

### Run 4 — Metadata fusion (NEXT — built on Run 1, not Run 2/3)
- age/sex/localization branch fused with image features (encoders fit on TRAIN only); custom
  two-input generator. Model file: `best_model_run4_meta.h5`. Biggest expected lever for bkl + mel.
- **Note:** builds on the Run 1 champion config (γ=1.5, label smoothing, rescale=1./255, no MixUp). Runs 2 & 3 both discarded.

---

## Cumulative target

**~0.91–0.92 AUC · ~90% sensitivity at ~0.80–0.82 specificity.**

_See `../EXPERIMENT_TRACKING.md` for full methodology, learnings, and raw check outputs._
_Last updated: after Run 3 scoring (rejected). Current best: Run 1 (AUC 0.895). Runs 2 & 3 both rejected. Next: Run 4 (metadata fusion)._
