# Statistical Analysis — Skin Lesion Binary CNN

_Test set: n=2001 (389 Concerning, 1612 Benign). Bootstrap B=2000, seed=42. Operating point: val-margin threshold at val Sens ≥ 0.94._

## Per-model performance with 95% confidence intervals

| Model | AUC (DeLong 95% CI) | AUC (bootstrap 95% CI) | thr | Sens (95% CI) | Spec (95% CI) |
|---|---|---|:---:|---|---|
| Run1_image_only | 0.8950 (0.8798–0.9102) | 0.8795–0.9100 | 0.101 | 0.9512 (0.929–0.972) | 0.6433 (0.621–0.666) |
| Run4_metadata | 0.9047 (0.8906–0.9187) | 0.8907–0.9179 | 0.169 | 0.9229 (0.896–0.948) | 0.7171 (0.695–0.740) |

## Paired DeLong test — Run4_metadata vs Run1_image_only

- AUC Run1_image_only = **0.8950**, AUC Run4_metadata = **0.9047**, Δ = **+0.0097**
- z = -2.030, **p = 0.0423** → metadata fusion is **SIGNIFICANT (p < 0.05)** vs image-only, on the same test set.

## Multi-seed champion

_No `multiseed/preds_seed*.npz` found yet — run `multiseed_train.py` across ≥5 seeds, then re-run this script to fill in mean ± std._
