# Transfer-Learning on HAM10000 — Research Brief for Section 2 (Related Work)

**Purpose:** Your uploaded `Lay_of_the_Land_First_2.md` already surveys the *from-scratch* CNN literature (the minority case your model belongs to). This brief covers the **transfer-learning mainstream** — the body of work your paper positions itself against — so you can write §2 and populate Table 8.

**One framing caution up front (same as your file):** many headline accuracies of 97–99% on HAM10000 come from **image-level splits** that leak near-duplicate lesion images across train/test. When you cite these, cite them as *reported* numbers and note the split protocol — your lesion-level split is a core differentiator, so do not present leakage-inflated figures as legitimate targets you "beat."

---

## 1. Summary by architecture family

### ResNet (residual networks)
- **Typical use:** ResNet50 / ResNet101 / ResNeXt101, ImageNet-pretrained, fine-tuned end-to-end or with a replaced classifier head.
- **Reported HAM10000 metrics:** ResNet50 commonly lands **~87–90% accuracy** (e.g., 87.1%, 90.5% in different studies); ResNeXt101 ~92.8%. In the ITM 2025 paper your file cites, ResNet50 reached only **77.1%** — a reminder that reported ResNet numbers swing widely with preprocessing and split.
- **Role in the field:** the default "strong baseline." Rarely the top performer now, but almost always included for comparison.

### EfficientNet (compound-scaled CNNs)
- **Typical use:** B0–B4 (occasionally B5–B7), ImageNet or ImageNet+extra pretraining, fine-tuned whole-network; frequently combined with test-time augmentation (TTA), multi-crop, or Monte-Carlo dropout.
- **Reported HAM10000 metrics:** **~90–94% accuracy** is the common band (e.g., EfficientNet-B0 ≈ 90.7–91%); with TTA + MC-dropout one study reports up to **97.15%** (treat as split-dependent). EfficientNet-B4 AUC **0.943** is quoted as a strong transfer-learning ceiling in the ITM 2025 counter-narrative paper.
- **Role in the field:** the current "high-accuracy transfer-learning" workhorse and the architecture most often reported as beating custom CNNs.

### DenseNet (densely connected CNNs)
- **Typical use:** DenseNet121 / DenseNet169 / DenseNet201, ImageNet-pretrained, fine-tuned; often paired with global-average-pooling heads or class weighting.
- **Reported HAM10000 metrics:** DenseNet121 ≈ **94.3%**; DenseNet169 ≈ **92.3%** (recall 93.6%, F1 93.3%); DenseNet201 ≈ **94.5%**, with some studies reporting up to **98%** (leakage-suspect). DenseNet201 is repeatedly named the single best-performing backbone in multi-architecture comparisons.
- **Role in the field:** frequent "best single backbone" in DenseNet-vs-others bake-offs.

### Inception / Xception / InceptionResNet
- **Typical use:** InceptionV3, Xception, InceptionResNetV2 as ensemble members or comparison backbones.
- **Reported HAM10000 metrics:** individually mid-pack (~85–90%); most valuable inside ensembles.

### Newer wave — ConvNeXt and Vision Transformers (ViT / Swin)
- **Typical use:** ImageNet-pretrained ConvNeXt or Swin/ViT, fine-tuned; increasingly the top of leaderboards.
- **Reported HAM10000 metrics:** in the even-handed 2025 benchmark below, **ConvNeXtXLarge was the single best of 38 architectures at 87.62% accuracy / 76.15% macro-F1** under a standardized protocol — notably *lower* than the leakage-suspect 94–98% claims, which tells you what an honestly-split transfer-learning number actually looks like. Swin/ViT papers report ~90–94% with larger pretraining.
- **Why you must acknowledge them:** your "against the transfer-learning mainstream" thesis will read as dated if §2 stops at ResNet/DenseNet. One sentence on ConvNeXt/ViT pre-empts the obvious reviewer objection.

### Ensembles and multimodal (image + metadata)
- **Ensembles:** stacking ResNeXt / SeResNeXt / DenseNet / Xception / ResNet with weighted averaging reaches **~94% macro-recall**; some five-fold-CV ensembles report **98.3%** accuracy (split-dependent). Highest raw numbers, least architectural novelty.
- **Multimodal:** fusing patient metadata (age, sex, site) with image features is an established line — directly relevant to your metadata-fusion contribution. Gessert et al. (ISIC 2019 winner) and Pacheco & Krohling are the canonical references; you already cite Pacheco `[18]`, but **Gessert is currently missing and should be added.**

---

## 2. Reported-metrics table (for adapting into Table 8)

| Study / backbone | Pretraining | Fine-tuning | Task | Reported metric | Split protocol / caveat |
|---|---|---|---|---|---|
| Esteva et al. 2017 (Inception-v3) | ImageNet | Full fine-tune | multi-class (photographic + dermoscopic) | Dermatologist-level; AUC ~0.91–0.96 per task | Not HAM10000; landmark context |
| Tschandl et al. 2019 (139 algorithms) | ImageNet (most) | Various | 7-class pigmented lesion | Top algorithms > 27 expert readers | Curated ISIC test; human-vs-machine benchmark |
| Gessert et al. 2020 (EfficientNet B0–B6 ensemble + metadata) | ImageNet | Full fine-tune, multi-crop | ISIC 2019 8-class (HAM10000 subset) | **1st place** ISIC 2019 tasks 1 & 2 | Challenge protocol; multimodal |
| EfficientNet-B0 (representative) | ImageNet | Full fine-tune (+TTA/MC-dropout) | 7-class | ~91% (up to 97.15% w/ TTA) | Split not always lesion-level |
| DenseNet201 (representative) | ImageNet | Full fine-tune | 7-class | ~94.5% (some report ~98%) | High end is leakage-suspect |
| ResNet50 (representative) | ImageNet | Head replace / full | 7-class | ~87–90% (77% in one study) | Wide variance |
| Electronics 2025 benchmark (38 CNNs, 10 families) | ImageNet (frozen) | Feature-extraction | 7-class | **ConvNeXtXLarge 87.62% acc / 76.15% F1**; family range 74–92% | Standardized; ISIC 2019 cross-check — **the honest anchor** |
| Ensemble (ResNeXt/SeResNeXt/DenseNet/Xception/ResNet) | ImageNet | Full fine-tune | 7-class | ~94% macro-recall (some ~98% acc) | Split-dependent |

---

## 3. Recommended baselines for §2 (5 papers, each justified)

Chosen to satisfy your Table 8 inclusion criteria (uses HAM10000/ISIC, reports AUC or acc/sens/spec, states approach) and to span the three sub-lines you must engage: transfer-learning CNNs, ensembles, and multimodal.

**B1 — Esteva et al. 2017, *Nature* (already your ref `[10]`).**
*Why:* the field-defining "dermatologist-level" result; opens §2 and justifies "deep learning is a credible diagnostic tool." Keep it, but note it is not HAM10000-specific.

**B2 — Tschandl et al. 2019, *Lancet Oncology*.** *(new — suggest `[28]`)*
*Why:* the largest human-vs-machine benchmark on pigmented lesions (139 algorithms vs 511 readers); establishes that **ImageNet-pretrained ensembles set the practical ceiling**. It is the strongest external anchor for "transfer learning is the mainstream," and it is from the same group that built HAM10000, so it dovetails with your dataset citation `[1]`.

**B3 — Gessert et al. 2020, *MethodsX*.** *(new — suggest `[29]`)*
*Why:* double relevance — it is (a) the **EfficientNet transfer-learning SOTA** (ISIC 2019 winner) and (b) a **metadata-fusion** method. Citing it simultaneously represents the architecture you argue against *and* the precedent for your second contribution, so your metadata branch looks informed rather than naïve. Currently absent from your references — highest-value add.

**B4 — "Deep Learning Approaches for Skin Lesion Detection," *Electronics* 2025, 14(14):2785.** *(new — suggest `[30]`)*
*Why:* the most **even-handed benchmark** available — 38 architectures across 10 families under one protocol, reporting the realistic **74–92% band** and an honest best of 87.62%. It lets you cite a *fair* transfer-learning number (not a leakage-inflated one) as your comparison point, which strengthens rather than inflates your positioning. Also covers ConvNeXt/newer families in one shot.

**B5 — Pacheco & Krohling (metadata multimodal) — you already cite `[18]`; keep and foreground it.**
*Why:* anchors the "structured clinical context improves classification" claim that motivates your fusion branch. Pair it with B3 so the metadata line has two citations.

*Optional 6th if you want an ensemble exemplar:* an Informatics-in-Medicine-Unlocked-style HAM10000 ensemble paper (ResNeXt/DenseNet/SeNet, ~94% recall) — represents the "stack backbones" trend you implicitly critique as high-cost/low-novelty.

---

## 4. Gaps flagged in your current research

1. **You have only the from-scratch side.** `Lay_of_the_Land_First_2.md` deliberately excludes transfer learning, so §2 currently has *no* mainstream to argue against. This brief fills the missing half; you still need to write the prose.
2. **Gessert 2020 is missing** despite being the single most relevant paper to *both* your transfer-learning framing and your metadata contribution. Add it.
3. **No ConvNeXt / Vision-Transformer coverage.** The newest and now top-performing transfer-learning family is unrepresented; omitting it dates the paper. One or two sentences (via B4) closes this.
4. **Split-protocol column is your differentiator — use it, but don't cite leakage numbers as targets.** Several 97–99% results are image-level-split. Report them as *reported* with the caveat, and make the lesion-level column in Table 8 do the argumentative work.
5. **Metric incomparability is unaddressed.** The literature mixes accuracy, AUC, macro-F1, and different tasks (7-class vs binary vs mel-vs-rest). State explicitly (you partly do in §5.3) that Table 8 positions rather than ranks, and prefer AUC/macro-F1 over raw accuracy when comparing.
6. **Minority-class failure is a shared theme you can exploit.** Nearly all transfer-learning papers still fail on df/vasc (and often bkl) — the same confusion your metadata branch targets. Connecting your bkl/akiec analysis to this documented, cross-paper weakness strengthens your motivation.
7. **External/cross-dataset validation is rarely done in this literature** — which both supports your Limitations (§5.2.1) and is a place you can note you are *no worse* than the mainstream, while flagging it as future work.

---

## 5. Suggested full citations (Vancouver, continuing your numbering)

> Verify each against the primary source before submission — details below are from search-result metadata; the two 2025/ensemble entries in particular need author-list confirmation.

`[28]` Tschandl P, Codella N, Akay BN, Argenziano G, Braun RP, Cabo H, et al. Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study. Lancet Oncol. 2019;20(7):938–47.

`[29]` Gessert N, Nielsen M, Shaikh M, Werner R, Schlaefer A. Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data. MethodsX. 2020;7:100864.

`[30]` [Authors — confirm]. Deep learning approaches for skin lesion detection. Electronics. 2025;14(14):2785.

`[31]` (optional ensemble exemplar) [Authors — confirm]. An approach for multiclass skin lesion classification based on ensemble learning. Inform Med Unlocked. 2021;25:100659.

Already in your reference list and reusable here: Esteva et al. `[10]`, Tschandl HAM10000 `[1]`, Pacheco & Krohling `[18]`, Kawahara et al. `[19]`.

---

## 6. Suggested workflow to finalize

- Pull each recommended paper's PDF and confirm the exact reported metric + split protocol before it enters Table 8 (do not transcribe numbers from this brief without checking the source).
- Use *Papers With Code* → "HAM10000" for a current leaderboard, but read each paper's split description — leaderboard accuracy is meaningless without it.
- Fill Table 8 with 5–6 rows: Esteva (context), Tschandl 2019, Gessert 2020, the Electronics 2025 benchmark, Pacheco & Krohling, and your own bolded row.
- Write §2 in ~4–6 paragraphs: (i) landmark DL dermatology, (ii) the ImageNet transfer-learning mainstream by family with typical metrics, (iii) ConvNeXt/ViT newest wave, (iv) metadata/multimodal, (v) the leakage/split-protocol inconsistency that motivates your lesion-level protocol, closing on the from-scratch gap your file already documents.
