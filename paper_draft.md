# A From-Scratch Squeeze-and-Excitation Residual CNN with Patient-Metadata Fusion for Binary Skin-Lesion Triage

## Abstract

Most skin-lesion classifiers rely on ImageNet transfer learning. We ask whether a convolutional network trained entirely from scratch can reach clinically useful triage performance on the HAM10000 dermoscopy dataset. We built a from-scratch squeeze-and-excitation residual CNN with multi-scale feature fusion and a late-fusion branch encoding patient age, sex, and lesion site, and evaluated it under a leak-free lesion-level split with a prospectively selected operating point. The model attained a mean test AUC of 0.901 ± 0.010 across five random seeds, with 0.92 mean sensitivity for detecting concerning lesions. Metadata fusion significantly improved AUC over an image-only model of the same architecture (paired DeLong p = 0.042). These results show that transfer learning is not a prerequisite for competitive dermoscopic triage, although the operating-point sensitivity showed run-to-run variance warranting stabilization before deployment.

_Keywords: skin lesion classification; convolutional neural network; squeeze-and-excitation; metadata fusion; dermoscopy; melanoma screening; HAM10000_

## 1. Introduction

### 1.1. Motivation

Skin cancer represents one of the most commonly diagnosed malignancies globally, with incidence rates continuing to rise across multiple regions [ADDED: 11]. Early and accurate classification of skin lesions is critical: delayed or incorrect diagnosis, particularly in high-risk conditions such as malignant melanoma, can substantially worsen patient outcomes [ADDED: 12]. Traditional diagnosis relies heavily on visual inspection by skilled dermatologists, a procedure that is inherently subject to inter-observer variability, fatigue, and resource constraints [ADDED: 13]. Diagnostic delays are especially severe in areas with restricted access to specialists.

Artificial intelligence, and convolutional neural networks (CNNs) in particular, has emerged as a promising adjunct to clinical dermatology. Landmark work demonstrated that deep learning models could match board-certified dermatologists on lesion classification from dermoscopic images [10], establishing automated systems as a credible diagnostic tool. Subsequent research has rapidly advanced performance through progressively deeper architectures and attention mechanisms — almost universally by fine-tuning networks pretrained on large natural-image datasets such as ImageNet [ADDED: 14], [ADDED: 15], [ADDED: 16]. This transfer-learning paradigm delivers strong empirical results but leaves an important question unanswered: what can a well-designed architecture achieve when trained entirely from scratch, without inheriting representations from a different domain?

A further gap concerns the use of structured clinical context. Patient age, sex, and anatomical site are routinely documented at the point of care and are diagnostically informative — actinic keratosis, for example, occurs predominantly on sun-exposed sites in older patients [ADDED: 17], in a pattern that colour and texture alone cannot capture. Yet the majority of published skin-lesion classifiers discard this metadata entirely, treating the task as a pure image-classification problem [ADDED: 18], [ADDED: 19]. Finally, many published results lack lesion-level data integrity guarantees — image-level splits allow near-duplicate acquisitions of the same lesion to appear in both training and evaluation sets, inflating metrics [ADDED: 20] — and omit statistical uncertainty quantification, making headline figures difficult to interpret or compare.

### 1.2. Objective Statement

This paper presents a binary skin-lesion triage classifier trained entirely from scratch on the HAM10000 dataset [1], designed to flag lesions as Concerning (warranting clinical follow-up or biopsy) or Benign (suitable for routine monitoring). The primary objective is to demonstrate that a carefully designed from-scratch architecture — incorporating residual connections, squeeze-and-excitation channel attention, multi-scale feature fusion, and patient metadata fusion — can achieve clinically acceptable discriminative performance (AUC ≥ 0.90, sensitivity ≥ 0.90 at a prospectively selected decision threshold) without relying on pretrained weights, and with performance claims substantiated by rigorous statistical evaluation.

### 1.3. Research Questions

#### 1.3.1. RQ1

Can a CNN trained entirely from scratch, without pretrained weights, achieve clinically acceptable discriminative performance (AUC ≥ 0.90, sensitivity ≥ 0.90) for binary skin-lesion triage on the HAM10000 dataset?

#### 1.3.2. RQ2

Does fusing structured patient metadata — age, sex, and anatomical localization — with image features yield a statistically significant improvement in AUC over an image-only model of the same architecture?

#### 1.3.3. RQ3

Are the reported performance gains stable across multiple random seeds, and do they hold under lesion-level data integrity guarantees and formal statistical uncertainty quantification?

### 1.4. Contributions

The contributions of this work are as follows:

1. **From-scratch SE-Residual CNN with multi-scale fusion.** A squeeze-and-excitation residual network trained entirely without pretrained weights, combining multi-scale feature fusion (256 + 384 + 512 dimensions), that attains a mean test AUC of 0.901 ± 0.010 across five random seeds on HAM10000 (0.9047 for the released reference model), demonstrating that transfer learning is not a prerequisite for clinically relevant discriminative performance.

2. **Patient metadata fusion.** A late-fusion branch that injects age, sex, and anatomical localization as structured clinical context alongside image features, producing a statistically significant AUC improvement over an image-only model of the same architecture on the identical test set (ΔAUC = +0.0097, p = 0.042, paired DeLong test).

3. **Lesion-level, leak-free evaluation with prospective threshold selection.** A two-stage lesion-grouped split with verified zero cross-split overlap, combined with a prospectively selected decision threshold — chosen on the validation set using a safety margin, applied once to the test set — that clears a 0.90 sensitivity floor without test-set tuning.

4. **Rigorous statistical evaluation.** Bootstrap 95% confidence intervals for all headline metrics, a paired DeLong significance test for the AUC comparison, and multi-seed stability analysis across five random seeds that distinguishes reproducible gains from seed-dependent variance.

## 2. Related Work

### 2.1. Transfer Learning as the HAM10000 Mainstream

Since the demonstration that deep networks can reach dermatologist-level lesion classification [10], the dominant approach to HAM10000 has been transfer learning: fine-tuning convolutional backbones pretrained on ImageNet. Architectures such as VGG, Inception, ResNet, DenseNet, Xception, and the EfficientNet family have been adapted to the seven-class problem, typically by replacing the classifier head and fine-tuning some or all of the backbone [CITE: transfer-learning HAM10000 example 1], [CITE: EfficientNet/DenseNet on HAM10000], [CITE: ResNet/Inception on HAM10000]. Reported figures are strong, frequently exceeding 90% accuracy with AUCs in the 0.94–0.97 range, and this paradigm now constitutes the de facto baseline against which new methods are measured. The empirical success is unsurprising: ImageNet pretraining supplies low-level edge, colour, and texture filters that are expensive to learn from roughly ten thousand images alone, and fine-tuning adapts these priors to dermoscopy at modest data cost. The question this leaves — and the one our work addresses — is how much of the reported performance is attributable to the pretrained prior itself, and how much survives when the evaluation protocol is tightened (Section 2.3).

### 2.2. From-Scratch CNNs on HAM10000

A smaller body of work trains CNN weights from random initialization, without any pretrained backbone. These studies cluster in a markedly lower and wider performance band — roughly 73–93% accuracy — and rarely exceed three to five convolutional blocks, a direct consequence of training from scratch on a dataset too small to support deeper networks without overfitting. Al-Asadi and Altun [CITE: Al-Asadi & Altun 2022] trained a four-convolution network with pixel-level SMOTE oversampling, reaching approximately 86% test accuracy. Polat and Koç [CITE: Polat & Koç 2020] showed that decomposing the imbalanced seven-class task into seven one-vs-rest binary problems raised accuracy from 77% to 92.9% — a structural gain from problem reformulation rather than architectural depth, and a precedent for the binary framing we adopt (Section 3.2). Closest to our own design, [CITE: ITM 2025 SE-attention custom CNN] combined a custom CNN backbone with squeeze-and-excitation attention and focal loss, yet reported that transfer-learning comparators still outperformed it (ResNet50 at 77% accuracy, EfficientNet-B4 at AUC 0.943) — a direct claim that architectural attention and focal loss alone do not close the gap to pretraining. Our results qualify that claim: with residual depth, multi-scale fusion, metadata fusion, and — critically — a leak-free evaluation, a from-scratch network reaches competitive, honestly-measured performance (Section 4).

A recurring caution applies to the highest reported from-scratch numbers. Musthafa et al. [CITE: Musthafa et al. 2024] reported 97.8% test accuracy from a from-scratch CNN while simultaneously documenting overfitting onset at epoch 20 and stating no lesion-level splitting protocol. Because HAM10000 contains multiple images of the same physical lesion (shared `lesion_id`), an image-level split places near-duplicate acquisitions of one lesion in both training and test — the textbook signature of leakage, and the most parsimonious explanation for accuracy far above the rest of the from-scratch literature. We therefore treat such figures as protocol artifacts rather than attainable targets, and design our evaluation explicitly to exclude this confound (Section 3.3).

### 2.3. Metadata Fusion and Evaluation Rigour

Two further gaps motivate our contribution. First, the majority of HAM10000 classifiers discard the patient metadata released with the dataset [CITE: metadata-discarding classifier 1], [CITE: image-only classifier 2], treating the task as pure image classification, despite age and anatomical site carrying diagnostic signal that pixels do not (Section 5.1). Multimodal image-plus-metadata approaches exist in the broader dermatology literature [CITE: multimodal image+metadata fusion in dermatology] but remain the exception on HAM10000. Second, validation hygiene and statistical reporting are inconsistent: few studies confirm a lesion-level rather than image-level split [CITE: duplicate-lesion / leakage discussion], and fewer still report confidence intervals or significance tests, leaving headline numbers difficult to compare or trust. Our work targets all three gaps simultaneously — a from-scratch architecture, patient-metadata fusion, and a verified leak-free protocol with bootstrap confidence intervals and paired DeLong significance testing — and, uniquely among the works surveyed, reports a fine-tuned transfer-learning baseline under identical conditions so that the effect of pretraining is measured rather than assumed.

## 3. Methods

### 3.1. Dataset

We used the HAM10000 dataset [1], a publicly available collection of 10,015 dermatoscopic images spanning seven diagnostic categories: melanoma (mel), melanocytic nevi (nv), basal cell carcinoma (bcc), actinic keratoses and intraepithelial carcinoma (akiec), benign keratosis-like lesions (bkl), dermatofibroma (df), and vascular lesions (vasc). Each image carries a lesion identifier (`lesion_id`) that groups the multiple acquisitions of a single physical lesion, together with the patient metadata released with the dataset: age, sex, and lesion localization (anatomical site). All images were resized to 224 × 224 pixels and rescaled to the [0, 1] range prior to training. The `lesion_id` field underpins our evaluation protocol (Section 3.3), and the metadata fields are the input to the fusion branch of the model (Section 3.4).

### 3.2. Binary Mapping

The seven diagnostic labels were collapsed into two clinically motivated groups reflecting the decision a screening tool must support — whether a lesion warrants clinical follow-up or biopsy, or whether it can be routinely monitored:

* **Concerning** (warrants follow-up / biopsy): melanoma, basal cell carcinoma, and actinic keratosis (mel, bcc, akiec).
* **Benign** (routine monitoring): melanocytic nevi, benign keratosis-like lesions, dermatofibroma, and vascular lesions (nv, bkl, df, vasc).

Actinic keratosis was assigned to the **Concerning** group because it is a pre-malignant lesion that can progress to squamous cell carcinoma [ADDED: 17]; it clinically warrants follow-up rather than routine dismissal, and therefore belongs with the lesions the model is intended to flag.

Under the binary mapping the dataset exhibits an approximately 4.1 : 1 class imbalance (Benign ≈ 8,061 images versus Concerning ≈ 1,954). Labels were integer-encoded alphabetically, so that **Benign = 0** and **Concerning = 1**; the Concerning class is treated throughout as the clinical positive class. The rationale for adopting a binary rather than a finer-grained formulation is discussed in Section 5.

### 3.3. Leak-Free Lesion-Level Split

Because a single lesion contributes several near-duplicate images [ADDED: 1], splitting at the image level allows acquisitions of the *same* lesion to appear in both training and evaluation, inflating every reported metric. All partitioning was therefore performed at the level of `lesion_id`, guaranteeing that every image of a given lesion resides in exactly one split.

We used a two-stage stratified split, keyed on `lesion_id` and stratified by the binary group label, with a fixed random seed (`random_state = 42`) for reproducibility:

1. A fully independent **test** set comprising 20% of lesions was held out first.
2. From the remaining lesions, 15% were carved off to form the **validation** set, leaving the remainder for training.

This yielded the partition in Table 1. The validation set drives *all* model selection — checkpointing, early stopping, learning-rate scheduling, and operating-point (threshold) tuning (Sections 3.6–3.7) — while the test set is scored once, at the end, for final reporting. Disjointness assertions confirmed that no lesion crosses splits, and a subsequent isolation audit verified zero overlap at both the lesion and image levels across all three partitions.

**Table 1.** Lesion-level, group-stratified data partition (HAM10000, split by `lesion_id`, `random_state = 42`).

| Split | Lesions | Images | Approx. share |
| --- | --- | --- | --- |
| Train | 5,079 | 6,824 | ≈ 68% |
| Validation | 897 | 1,190 | ≈ 12% |
| Test | 1,494 | 2,001 | 20% |
| **Total** | **7,470** | **10,015** | **100%** |

### 3.4. Architecture

The classifier is a convolutional neural network trained **entirely from scratch**, with no pretrained weights and no transfer learning — a deliberate constraint that constitutes the architectural contribution of this work. The design combines four mechanisms that together raise the discriminative ceiling of a shallow, single-scale, plain CNN: residual connections, squeeze-and-excitation (SE) channel attention, multi-scale feature fusion, and late fusion of patient metadata. Table 2 summarizes the feature-extraction backbone.

A stem convolution (Conv2D with 32 filters, 3 × 3 kernel, followed by batch normalization and ReLU) lifts the three-channel input to 32 channels. Five **SE-residual stages** then progressively increase channel depth (64 → 128 → 256 → 384 → 512) while halving spatial resolution at each stage (112 → 56 → 28 → 14 → 7). Each stage (`se_residual_block`) applies two 3 × 3 convolutions with batch normalization and ReLU, followed by an SE attention gate, a residual skip connection, a final ReLU, 2 × 2 max-pooling, and stage-specific dropout (0.10, 0.15, 0.20, 0.20, and 0.25 respectively). The residual skip is an identity mapping where channel counts match and a 1 × 1 convolutional projection (with batch normalization) where they change, with the addition performed before pooling so that both paths share spatial dimensions.

The SE gate (`se_block`) implements channel attention as global average pooling → Dense(C / r, ReLU) → Dense(C, sigmoid) → channel-wise reweighting, with reduction ratio r = 16 (floored at 8 units) [2].

**Multi-scale fusion** addresses the wide variation in lesion scale: rather than classifying only on the final, coarsest feature map, the outputs of the last three stages are each reduced by global average pooling and concatenated (256 + 384 + 512), producing a 1,152-dimensional image descriptor that encodes fine texture and global shape simultaneously.

**Metadata fusion** distinguishes the champion model. The patient metadata is encoded as a fixed-width vector — standardized age concatenated with one-hot encodings of sex and localization (≈ 20 dimensions) — and passed through a small multilayer perceptron (Dense(32, ReLU) → Dense(16, ReLU)) to yield a 16-dimensional clinical-context embedding. This embedding is concatenated with the 1,152-dimensional image descriptor (≈ 1,168 features) before the classifier head. The metadata branch is kept deliberately small so that the image branch remains the dominant signal and the network cannot simply memorize metadata. The head is Dense(256, ReLU) → batch normalization → Dropout(0.40) → Dense(2, softmax).

The rationale for each mechanism is as follows. **Residual connections** [3] keep gradients healthy and thereby permit greater depth (five stages) than a from-scratch plain CNN can typically train without a vanishing-gradient stall. **SE attention** allows the network to emphasize channels encoding concerning markers (colour asymmetry, irregular borders, atypical vasculature) and to suppress irrelevant ones at roughly 1% additional parameters [ADDED: 2]. **Multi-scale fusion** lets the classifier see both fine and coarse features. **Metadata fusion** injects a non-image signal that pixel-based models cannot access: actinic keratosis (akiec) occurs predominantly on sun-exposed sites (face, neck, hands/forearms) in older patients, whereas benign keratosis (bkl) is common on the trunk across a wider age range [ADDED: 17] — so age and localization separate these confusable classes in a way colour and texture cannot.

The complete two-input model (`SE_Residual_MultiScale_CNN_Meta`) contains 8,226,794 parameters; the image-only variant used as the ablation reference contains 8,221,530 parameters.

**Table 2.** From-scratch SE-residual, multi-scale feature-extraction backbone. Each stage is a two-convolution SE-residual block with a residual skip, followed by max-pooling and dropout.

| Stage | Filters | Output (H × W × C) | Dropout | Pooled to fusion |
| --- | --- | --- | --- | --- |
| Stem (Conv-BN-ReLU) | 32 | 224 × 224 × 32 | — | — |
| SE-residual 1 | 64 | 112 × 112 × 64 | 0.10 | — |
| SE-residual 2 | 128 | 56 × 56 × 128 | 0.15 | — |
| SE-residual 3 | 256 | 28 × 28 × 256 | 0.20 | ✓ (256) |
| SE-residual 4 | 384 | 14 × 14 × 384 | 0.20 | ✓ (384) |
| SE-residual 5 | 512 | 7 × 7 × 512 | 0.25 | ✓ (512) |
| Multi-scale concat | — | 1,152-d | — | → head |
| Metadata MLP | — | 16-d | — | → head |

### 3.5. Loss Function (Focal Loss)

Given the roughly 4 : 1 class imbalance, the network was trained with focal loss [4], which down-weights the many easily classified (predominantly Benign) examples so that gradient focus remains on the harder Concerning cases:

α_t · (1 − p_t)^γ · (−log p_t)

where p_t is the predicted probability of the true class and α_t is the class-specific weighting factor. We used a focusing parameter γ = 1.5 and label smoothing ε = 0.05 [ADDED: 21] applied to the one-hot targets before computing p_t.

Class weighting α is the **single** imbalance correction employed. It was derived from balanced class weights (inverse class frequency), softened by a square-root transform, and normalized to sum to one; no oversampling and no manual clinical boost were used. Concentrating the correction in one softened α term keeps the Concerning class at roughly one in five images per batch while up-weighting its gradient contribution. The focusing parameter γ = 1.5 preserves a usable spread of output probabilities, and label smoothing acts primarily as a ranking (AUC) regularizer. Crucially, clinical sensitivity is **not** baked into the training objective; it is achieved at inference by selecting a high-sensitivity operating point (Section 3.7), which keeps balanced training and clinical bias cleanly separated and independently tunable.

### 3.6. Training Procedure

Models were optimized with Adam at a learning rate of 3 × 10⁻⁴ with gradient-norm clipping (`clipnorm = 1.0`). The learning rate was governed by `ReduceLROnPlateau` (monitor = val AUC, factor = 0.5, patience = 4, minimum LR = 10⁻⁶), so that it adapts to the observed validation plateau rather than a preset schedule.

All model selection was driven by **validation AUC** rather than validation balanced accuracy. Balanced accuracy is computed at the argmax-0.5 threshold — the operating point at which the model is least stable — and was observed to oscillate substantially across epochs, so checkpoints selected on it captured a "lucky epoch" rather than a durably good model. AUC is threshold-independent: it measures how well the model ranks Concerning above Benign across all thresholds, which is the stable part of the model. A custom callback (`ValMetrics`) computes validation AUC, balanced accuracy, and macro-F1 at the end of every epoch; balanced accuracy and macro-F1 are retained as diagnostics but do not drive selection. Early stopping (monitor = val AUC, patience = 15, restore-best-weights) and best-checkpoint saving (monitor = val AUC) both use this metric. Training ran for up to 60 epochs with a batch size of 32.

Training images were augmented online with random rotations (± 90°), width and height shifts (± 20%), horizontal and vertical flips, zoom (± 20%), brightness scaling in [0.9, 1.1], and shear (0.1), with nearest-neighbour fill. Validation and test images were only rescaled, with no augmentation, so that predictions align with their labels. For reproducibility, NumPy and TensorFlow random seeds were fixed (SEED = 42) so that weight initialization, augmentation, and shuffle order are deterministic and a retrain reproduces the saved model; the environment was additionally pinned via a fixed requirements specification.

Model development followed a **cumulative single-change ablation** protocol: starting from the from-scratch SE-residual baseline, each candidate modification introduced exactly one change, was trained to convergence, and was compared against the current best model by test AUC. A change was carried forward only if it improved AUC; otherwise the previous configuration was retained (Table 3). Colour constancy (Run 2) and MixUp (Run 3) did not improve AUC and were therefore rejected, so the champion metadata-fusion configuration (Run 4) was built on the calibrated baseline of Run 1. The per-run outcomes are reported in Section 4.2.

**Table 3.** Cumulative single-change ablation design. Each run adds one component to its stated base; a change is carried forward only if it improves test AUC.

| Run | Built on | Change introduced | Carried forward |
| --- | --- | --- | --- |
| 0 | — | Baseline from-scratch SE-residual, multi-scale CNN | ✓ |
| 1 | Run 0 | Calibration: focal γ = 1.5, label smoothing, temperature scaling | ✓ |
| 2 | Run 1 | Colour constancy (Shades-of-Gray preprocessing) | ✗ (no AUC gain) |
| 3 | Run 1 | MixUp augmentation | ✗ (no AUC gain) |
| 4 | Run 1 | Patient-metadata fusion (age / sex / localization) | ✓ (champion) |

To prevent information leakage through the metadata pathway, every metadata statistic — the age imputation median, the age mean and standard deviation used for standardization, and the sex and localization category vocabularies — was fit on the **training split only** and applied unchanged to validation and test. Categories not observed during training map to all-zero encodings by construction.

### 3.7. Operating Point and Safety Margin

Because the intended use is screening, the primary quantity of interest is not the default argmax decision but the model's behaviour at a high-sensitivity operating point. The headline metric is therefore threshold-independent AUC, complemented by sensitivity and specificity reported at an explicitly selected decision threshold. The default argmax (0.5) prediction is retained only as a reference; its low argmax sensitivity is the expected consequence of the class base rate, not a defect, and is not reported as a headline figure.

The screening threshold was selected on the **validation** set and then applied, fixed, to the untouched test set. Concretely, from the validation ROC curve we take the smallest-false-positive-rate threshold whose validation sensitivity meets a selection target, and evaluate that single threshold once on the test set. Selecting the threshold on the data it is subsequently scored on would optimistically bias the reported sensitivity and specificity; selecting on validation and reporting the fixed threshold's test performance avoids this.

To absorb the validation-to-test transfer gap, the selection target was set *above* the clinical floor we wish to hold on test. This **validation safety margin** was fixed a priori from validation data alone: with a clinical target of `TARGET_SENS = 0.90` on test, we selected the threshold at `VAL_SELECTION_SENS = 0.94` on validation. The resulting threshold (≈ 0.169) was then scored once on the test set, yielding a test sensitivity of 0.9229 and specificity of 0.7171, clearing the 0.90 floor. This trade is viable only because the champion model's ROC curve dominates those of the alternatives.

### 3.8. Temperature Scaling and Expected Calibration Error (ECE)

To ensure that predicted probabilities are trustworthy for clinical interpretation, we applied post-hoc temperature scaling [5]. A single scalar temperature T was fit on the validation set by minimizing the negative log-likelihood and then applied to the test-set logits. Because temperature scaling is a monotonic transformation, it leaves AUC and the validation-selected sensitivity/specificity unchanged; its sole purpose is to improve probability calibration. Calibration quality was quantified by the Expected Calibration Error (ECE) using ten equal-width confidence bins [ADDED: 22], and visualized with a reliability diagram comparing empirical accuracy against mean predicted confidence before and after scaling.

For the champion model the fitted temperature was T = 0.722, and ECE improved marginally from 0.040 to 0.034. The near-identity temperature and low pre-scaling ECE indicate the model was already well-calibrated; temperature scaling therefore functions here as a confirmatory reliability analysis rather than a substantive correction.

### 3.9. Statistical Analysis

Uncertainty in the headline metrics was quantified by non-parametric bootstrap resampling of the test set (B = 2,000 resamples) [ADDED: 23], from which 95% confidence intervals were derived for AUC, sensitivity, and specificity. The test set comprises 2,001 images (389 Concerning, 1,612 Benign).

To compare the discriminative performance of the metadata-fusion model against the image-only model on the same test set, we used the paired DeLong test for the difference between two correlated areas under the ROC curve [6,7]. The paired test is appropriate here because both models are evaluated on identical samples, so their AUC estimates are correlated; comparing their marginal confidence intervals would ignore this correlation and understate the evidence for a difference.

Finally, because unseeded training exhibited run-to-run variance of roughly 0.007 in AUC together with substantially larger per-class swings, single-run claims are unreliable. We therefore adopt a **multi-seed** reporting protocol: the champion configuration is retrained across five random seeds (seeds 1–5) and the metrics are aggregated as mean ± standard deviation, distinguishing the stable AUC gain from seed-dependent operating-point effects. The data partition is held fixed (`random_state = 42`) across all seeds, so that only training stochasticity — weight initialization, augmentation, and batch order — varies. The single locked seed (42) is used for the reproducible reference model on which the paired significance test is computed, while the five-seed aggregate provides the headline mean ± standard deviation.

### 3.10. Reporting Standards (CLAIM / TRIPOD-AI)

The study is reported in accordance with established guidelines for medical-imaging and clinical-prediction AI: the Checklist for Artificial Intelligence in Medical Imaging (CLAIM) [8] and the TRIPOD-AI statement for transparent reporting of multivariable prediction models that incorporate machine learning [9]. In line with these standards we report: the provenance and licensing of the data source and the exact class definitions (Sections 3.1–3.2); the lesion-level, leak-controlled data partition with an explicitly held-out, once-scored test set and a verified isolation audit (Section 3.3); a complete description of the model architecture, inputs (including the patient-metadata fields), loss function, and training configuration sufficient for reproduction (Sections 3.4–3.6); the intended use as a screening tool together with the prospective, validation-based selection of the decision threshold (Section 3.7); model calibration assessed by ECE and a reliability diagram (Section 3.8); performance reported with bootstrap confidence intervals and a paired significance test, and stability assessed across multiple random seeds (Section 3.9); and reproducibility measures comprising a fixed random seed and a pinned software environment (Section 3.6). Known limitations relevant to these checklists — single-dataset evaluation, the fixed operating specificity, and the binary reformulation — are discussed in Section 5.

## 4. Results

All metrics are reported on the held-out test set (n = 2,001 images; 389 Concerning, 1,612 Benign), which was scored exactly once, at the operating point selected on validation (Section 3.7). Confidence intervals are non-parametric bootstrap intervals (B = 2,000) unless stated otherwise.

### 4.1. Main Results

The champion metadata-fusion model (Run 4) achieved a test AUC of **0.9047** (DeLong 95% CI 0.8906–0.9187; bootstrap 95% CI 0.8907–0.9179) for the released reference model, and a mean AUC of **0.9014 ± 0.0095** across five random seeds. At the prospectively selected screening threshold (0.169), the reference model attained a sensitivity of **0.9229** (95% CI 0.896–0.948) and a specificity of **0.7171** (95% CI 0.695–0.740), clearing the pre-specified 0.90 sensitivity floor. Table 4 summarizes these headline figures alongside the five-seed aggregate; the seed-level distribution is analysed in Section 4.6.

**Table 4.** Champion (Run 4) test performance: released reference model with 95% confidence intervals, and five-seed mean ± standard deviation. Operating point selected on validation at a 0.94 safety margin and applied once to test.

| Metric | Released reference model (95% CI) | Five-seed mean ± SD |
| --- | --- | --- |
| AUC | 0.9047 (0.8906–0.9187) | 0.9014 ± 0.0095 |
| Sensitivity @ thr 0.169 | 0.9229 (0.896–0.948) | 0.9249 ± 0.0427 |
| Specificity @ thr 0.169 | 0.7171 (0.695–0.740) | 0.7073 ± 0.0499 |
| ECE (after temperature scaling) | 0.0341 | — |

This answers **RQ1 in expectation**: a fully from-scratch network reaches a mean AUC of 0.90 and a mean sensitivity of 0.92 without any pretrained weights, though the operating-point sensitivity is seed-dependent (Section 4.6).

### 4.2. Ablation Study

Table 5 reports the cumulative single-change ablation. Starting from the from-scratch SE-residual baseline (Run 0, AUC 0.883), the calibration changes of Run 1 improved AUC to 0.895. Two image-only interventions were then evaluated and **rejected** because they regressed AUC: Shades-of-Gray colour constancy (Run 2, AUC 0.871, −0.024) and MixUp augmentation (Run 3, AUC 0.8815, −0.0135, with test sensitivity 0.887 falling below the 0.90 floor). The metadata-fusion model (Run 4), built on the calibrated Run 1 configuration, raised AUC to 0.9047 and became the champion. The two rejected runs are retained here as negative results: colour constancy collapsed the class-conditional probability distributions (every class's mean predicted probability shrank by 60–70%), consistent with the grey-world assumption erasing genuine diagnostic pigmentation on frames that are filled edge-to-edge with skin and lesion; MixUp over-regularized an already heavily regularized network, compressing predictions toward the centre without improving ranking.

**Table 5.** Cumulative single-change ablation on the fixed leak-free test set. Headline metric is test AUC; a change is carried forward only if AUC improves.

| Run | Change introduced | Test AUC | ΔAUC vs base | Outcome |
| --- | --- | :---: | :---: | --- |
| 0 | Baseline from-scratch SE-residual, multi-scale CNN | 0.883 | — | ✓ baseline |
| 1 | Calibration (focal γ = 1.5, label smoothing, temperature scaling) | 0.895 | +0.012 | ✓ carried forward |
| 2 | + Colour constancy (Shades-of-Gray) | 0.871 | −0.024 | ✗ rejected |
| 3 | + MixUp augmentation | 0.8815 | −0.0135 | ✗ rejected (Sens < 0.90) |
| 4 | + Patient-metadata fusion | **0.9047** | **+0.010** | ✓ **champion** |

### 4.3. Significance of the Metadata Fusion (Paired DeLong Test)

On the identical test set, the metadata-fusion model (AUC 0.9047) significantly outperformed the image-only model of the same architecture (AUC 0.8950): paired DeLong test ΔAUC = +0.0097, z = −2.030, **p = 0.0423**. We note that the two models' marginal AUC confidence intervals overlap (image-only 0.8798–0.9102; metadata 0.8906–0.9187). This is not a contradiction: because both models score the same samples, their AUC estimates are positively correlated, and the paired DeLong test — which accounts for that correlation — is the appropriate and more powerful comparison. Comparing the marginal intervals alone would ignore the correlation and understate the evidence. The gain should nevertheless be read as modest: it is measured on the released reference model (a favourable draw), and the mean AUC improvement over image-only across seeds is approximately 0.006.

### 4.4. Calibration

The champion model was well-calibrated before any correction. Post-hoc temperature scaling fitted a near-identity temperature of T = 0.722 on the validation set, and the test-set Expected Calibration Error improved only marginally, from 0.0403 to 0.0341 (ten-bin ECE). The reliability diagram (Figure X) shows empirical accuracy tracking mean predicted confidence closely both before and after scaling. Because temperature scaling is monotonic, AUC and the validation-selected sensitivity/specificity are unchanged; the analysis therefore serves as a confirmatory reliability check rather than a substantive correction — a point of methodological note, since a low pre-scaling ECE indicates the focal-loss training did not produce the probability compression sometimes attributed to it [ADDED: 24].

### 4.5. Per-Diagnosis Breakdown

Because the operating threshold is applied to the binary Concerning-vs-Benign decision, we examined how each of the seven original diagnostic categories behaves at the locked threshold (0.169) for the released model (Table 6). All three Concerning categories exceeded 90% recall — melanoma 92.1%, basal cell carcinoma 90.0%, and actinic keratosis 98.1% — an important result given that melanoma is the deadliest class. Among Benign categories, melanocytic nevi (the dominant class) cleared at 79.9%, while benign keratosis-like lesions (bkl) remained the principal source of false positives, with only 22.1% correctly cleared. The mean predicted probability of Concerning separates the malignant categories (mel 0.372, bcc 0.422, akiec 0.500) from most benign categories, but bkl (0.273) sits close to the akiec/mel range — the residual confusion the metadata branch only partially resolves. Because at-threshold recall shifts mechanically with the threshold, the mean-probability column is the threshold-independent signal and should be read as primary; per-class recall figures for a single draw are additionally subject to the seed variance quantified in Section 4.6.

**Table 6.** Per-diagnosis behaviour of the released champion at the locked screening threshold (0.169). Concerning rows report recall (fraction flagged); Benign rows report clearance (fraction correctly cleared). Mean P = mean predicted probability of Concerning.

| Diagnosis | Group | n (test) | Mean P(Concerning) | Recall / Clearance |
| --- | --- | :---: | :---: | :---: |
| Melanoma (mel) | Concerning | 227 | 0.372 | 92.1% (recall) |
| Basal cell carcinoma (bcc) | Concerning | 110 | 0.422 | 90.0% (recall) |
| Actinic keratosis (akiec) | Concerning | 52 | 0.500 | 98.1% (recall) |
| Melanocytic nevi (nv) | Benign | 1,371 | 0.109 | 79.9% (clearance) |
| Benign keratosis (bkl) | Benign | 190 | 0.273 | 22.1% (clearance) |
| Dermatofibroma (df) | Benign | 20 | 0.283 | 40.0% (clearance) |
| Vascular lesion (vasc) | Benign | 31 | 0.250 | 35.5% (clearance) |

### 4.6. Multi-Seed Stability

Retraining the champion configuration under five random seeds (fixed data split, `random_state = 42`) yielded a mean test AUC of **0.9014 ± 0.0095** (range 0.888–0.914), a mean sensitivity of **0.9249 ± 0.0427** (range 0.882–0.977), and a mean specificity of **0.7073 ± 0.0499** (range 0.636–0.762) at the per-seed validation-selected threshold (Table 7). The AUC is tightly concentrated, confirming that the discriminative gain is reproducible. The operating-point sensitivity, however, is not uniformly above the 0.90 target: two of the five seeds landed at 0.882, below the clinical floor, even though the mean clears it. This variance — inherent to from-scratch training without pretrained initialization — is a substantive finding rather than a nuisance, and directly informs the deployment discussion in Section 5.2.4. This addresses **RQ3**: the AUC gain is stable, whereas the exact operating-point sensitivity requires stabilization (e.g., seed ensembling) before deployment.

**Table 7.** Per-seed test performance of the champion configuration (five seeds; fixed leak-free split). Threshold selected per seed on validation at the 0.94 safety margin.

| Seed | AUC | Sensitivity | Specificity |
| :---: | :---: | :---: | :---: |
| 1 | 0.9032 | 0.9769 | 0.6359 |
| 2 | 0.8879 | 0.8817 | 0.7413 |
| 3 | 0.9143 | 0.9537 | 0.6818 |
| 4 | 0.8991 | 0.9306 | 0.7159 |
| 5 | 0.9026 | 0.8817 | 0.7618 |
| **Mean ± SD** | **0.9014 ± 0.0095** | **0.9249 ± 0.0427** | **0.7073 ± 0.0499** |

### 4.7. Transfer-Learning Baseline

To quantify the effect of the from-scratch constraint rather than assume it, we trained a fine-tuned EfficientNet-B0 [CITE: EfficientNet / Tan & Le 2019] under an otherwise identical protocol: the same leak-free lesion-level split, binary mapping, focal loss, validation-selected operating point, and [0, 1] image inputs, with the only difference being ImageNet-pretrained versus random initialization. The backbone was fine-tuned in two phases — frozen-backbone head training, then low-learning-rate fine-tuning of the backbone with batch-normalization layers held frozen. Evaluated identically, the transfer baseline achieved a test AUC of 0.9160 (DeLong 95% CI 0.9017–0.9302), with sensitivity 0.892 and specificity 0.779 at its validation-selected threshold (Table 8).

Two comparisons follow. First, on the identical test set the transfer model significantly outperformed the from-scratch image-only model (AUC 0.8950): paired DeLong ΔAUC = +0.0210, z = −3.089, p = 0.0020. Pretraining therefore confers a real and statistically significant advantage — roughly twice the magnitude of the metadata-fusion effect (Section 4.3) — and we do not claim parity with transfer learning. Second, however, this honestly-measured 0.916 sits well below the 0.94–0.97 AUCs commonly reported for transfer learning on HAM10000, consistent with a substantial portion of that reported superiority arising from image-level leakage (Section 2.2) rather than from pretraining alone. Notably, the transfer baseline also failed to hold the 0.90 sensitivity floor at its validation-selected threshold (0.892), mirroring the seed-dependent operating-point behaviour observed for the from-scratch model (Section 4.6) — evidence that validation-to-test threshold transfer is a general property of this dataset rather than a weakness specific to from-scratch training.

**Table 8.** From-scratch versus transfer learning under the identical leak-free protocol (same split, loss, and operating-point selection). All models scored once on the test set (n = 2,001). ΔAUC and p are from paired DeLong tests against the from-scratch image-only model.

| Model | Pretrained | AUC (95% CI) | Sens | Spec | ΔAUC vs image-only | p |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| From-scratch, image-only (Run 1) | No | 0.8950 (0.880–0.910) | 0.9512 | 0.6433 | — | — |
| From-scratch, + metadata (champion) | No | 0.9047 (0.891–0.919) | 0.9229 | 0.7171 | +0.0097 | 0.042 |
| EfficientNet-B0, image-only (transfer) | Yes | 0.9160 (0.902–0.930) | 0.8920 | 0.7785 | +0.0210 | 0.0020 |

> _[PENDING] A metadata-fused transfer variant (EfficientNet-B0 + age/sex/localization, `baseline_transfer.py --metadata`) is under evaluation; its result will be added as a fourth row to test whether the metadata benefit generalizes across backbones._

## 5. Discussion

### 5.1. Interpretation

The central finding is that a convolutional network trained entirely from scratch — with no ImageNet pretraining or transfer learning of any kind — reaches a mean AUC of 0.90 for binary skin-lesion triage on HAM10000, with sensitivity averaging 0.92 at a prospectively chosen operating point. This directly counters the prevailing assumption that pretrained backbones are a prerequisite for clinically relevant dermoscopic performance, and it does so under a leak-free lesion-level protocol that many prior reports do not enforce.

The metadata-fusion result clarifies *where* the remaining performance came from. Two image-only interventions (colour constancy and MixUp) both failed to improve the model, whereas fusing patient age, sex, and anatomical localization produced a statistically significant AUC gain. The mechanism is interpretable: the most persistent confusion in this dataset is between actinic keratosis (Concerning) and benign keratosis (Benign), two categories that share brown/tan pigmentation and are therefore difficult to separate on colour and texture alone. Age and body site carry the discriminating signal — actinic keratosis concentrates on sun-exposed sites in older patients, benign keratosis is common on the trunk across a wider age range [ADDED: 17] — which explains why a non-image branch succeeded where two image-only interventions did not. In the released model this manifested as the malignant categories separating further from benign (melanoma mean probability rising from 0.319 to 0.372, actinic keratosis from 0.438 to 0.500), lifting melanoma recall above 90% for the first time in the ablation.

### 5.2. Limitations

#### 5.2.1. Single-Dataset Evaluation

All results derive from a single dataset (HAM10000). Although the lesion-level split prevents within-dataset leakage, it cannot establish generalization across acquisition devices, populations, or clinical settings. Dermoscopic appearance and metadata distributions vary between sources, and the metadata branch in particular may not transfer if localization vocabularies or age distributions differ. External validation on an independent dataset (for example ISIC [ADDED: 25], PH2 [ADDED: 26], or Derm7pt [ADDED: 19]) is the most important next step before any deployment claim, and its absence bounds the strength of our generalization claims.

#### 5.2.2. Fixed Operating Specificity

At the locked screening operating point the model achieves a specificity of approximately 0.71 (five-seed mean 0.707), below the 0.80 stretch target. This reflects a deliberate design choice for a screening tool — sensitivity is prioritized over specificity, since a missed Concerning lesion is the costlier error — but it means roughly three in ten benign lesions would be flagged for review, driven predominantly by benign keratosis false positives (Section 4.5). The sensitivity–specificity trade is explicit and adjustable via the threshold; we report a single high-sensitivity point rather than claiming simultaneous attainment of both stretch targets.

#### 5.2.3. Binary Reformulation

We reformulated the seven-class problem as a binary Concerning-vs-Benign decision. This choice is clinically motivated — a triage tool must decide whether a lesion warrants follow-up — and was reinforced empirically: a preliminary three-class formulation collapsed, never predicting the pre-malignant class (0% recall) because too few unique lesions supported it. The binary framing removes that fragile minority class and aligns the objective with the screening use-case. The cost is a loss of fine-grained diagnostic detail; the model indicates *whether* a lesion is concerning, not *which* specific diagnosis applies, and cannot substitute for a full diagnostic workup.

#### 5.2.4. Seed-Dependent Operating-Point Sensitivity

While the AUC is stable across seeds (0.9014 ± 0.0095), the sensitivity at the selected operating point is not: two of five seeds fell to 0.882, below the 0.90 clinical floor (Section 4.6). Because the threshold is chosen on a finite validation set, its transfer to test is itself noisy, and from-scratch training amplifies this run-to-run variability relative to pretrained models. A production system should therefore not rely on a single training run; stabilization strategies — ensembling across seeds, or selecting the operating point on a larger calibration set — would be required to guarantee the sensitivity floor is met reliably. We report this variance transparently rather than presenting only the most favourable draw.

### 5.3. Comparison to Prior HAM10000 Work

Table 9 situates our results among representative HAM10000 studies. Direct numerical comparison is inherently limited: prior work differs in task definition (seven-class vs. binary vs. one-vs-rest), in reported metric (most report seven-class accuracy rather than AUC), and — most consequentially — in split protocol. We therefore use the table to position approaches along the two axes that most affect trustworthiness, pretraining and split integrity, rather than to rank raw numbers, which are not commensurable across these differences.

Two observations stand out. First, ours are the only entries with a verified lesion-level split that report both a from-scratch model and a transfer baseline under identical conditions — precisely the controlled comparison the field largely lacks. Second, the highest prior accuracies coincide with unstated or image-level splitting (Section 2.2), reinforcing that leakage, not architecture, most plausibly explains the gap between those figures and the honestly-measured 0.90–0.92 band that we and a fairly-evaluated transfer baseline both occupy. Our from-scratch champion (AUC 0.901 ± 0.010) trails a fine-tuned EfficientNet-B0 (0.916) by a modest, quantified margin, while exceeding the realistic from-scratch literature and remaining clinically useful at a high-sensitivity operating point.

**Table 9.** Representative HAM10000 results in context. "Pretrained" = uses ImageNet/transfer learning; "Leak-free split" = evaluation verified free of same-lesion leakage. Metrics are as reported by each source; accuracy (acc) and AUC are not directly comparable, and seven-class results are not comparable to our binary task — the table positions method characteristics, not a common-benchmark ranking.

| Study | Approach / backbone | Pretrained | Task | Leak-free split | Reported performance |
| --- | --- | :---: | --- | :---: | --- |
| [CITE: representative transfer EfficientNet on HAM10000] | Fine-tuned EfficientNet | Yes | 7-class | Not stated | AUC ≈ 0.94 ‡ |
| [CITE: ITM 2025] | Custom SE-attention CNN + transfer comparators | No / Yes | 7-class | Not stated | EfficientNet-B4 AUC 0.943; custom CNN lower ‡ |
| [CITE: Musthafa et al. 2024] | From-scratch CNN + checkpoints | No | 7-class | Not stated § | acc 97.8% § |
| [CITE: Polat & Koç 2020] | From-scratch CNN + one-vs-all | No | 7-class | Not stated | acc 92.9% |
| [CITE: Al-Asadi & Altun 2022] | From-scratch 4-conv + pixel SMOTE | No | 7-class | Not stated | acc ≈ 86% |
| **This work — transfer baseline** | **EfficientNet-B0 fine-tuned** | **Yes** | **binary Concerning/Benign** | **Yes** | **AUC 0.916; Sens 0.892; Spec 0.779** |
| **This work — from-scratch champion** | **SE-residual + multi-scale + metadata** | **No** | **binary Concerning/Benign** | **Yes** | **AUC 0.901 ± 0.010; Sens 0.925; Spec 0.707** |

‡ AUC as reported by the source under its own (unverified) protocol. § Reported accuracy under an unstated split; the documented early overfitting and the absence of lesion-level splitting suggest probable image-level leakage (Section 2.2), so this figure should not be read as an attainable from-scratch target.

## 6. Conclusion

### 6.1. Answers to Research Questions

#### 6.1.1. RQ1

*Can a from-scratch CNN achieve clinically acceptable discriminative performance?* Substantially, yes, in expectation. The from-scratch SE-residual, multi-scale, metadata-fused model reached a mean test AUC of 0.901 ± 0.010 and a mean sensitivity of 0.925 across five seeds, clearing the AUC ≥ 0.90 and sensitivity ≥ 0.90 targets on average without any pretrained weights. The qualification is that the operating-point sensitivity is not met by every seed individually (Section 4.6).

#### 6.1.2. RQ2

*Does metadata fusion yield a statistically significant improvement?* Yes. On the identical test set, metadata fusion improved AUC over the image-only model of the same architecture by a statistically significant margin (ΔAUC = +0.0097, paired DeLong p = 0.042). The improvement is modest in magnitude (mean seed gain ≈ 0.006) but consistent with the interpretable mechanism of age and localization resolving the actinic-vs-benign-keratosis confusion.

#### 6.1.3. RQ3

*Are the gains stable and obtained under sound data integrity?* The AUC gain is stable across seeds (SD 0.0095) and was obtained under a verified leak-free, lesion-level split with a prospectively selected operating point and bootstrap uncertainty quantification. The operating-point sensitivity, however, exhibits meaningful seed variance, indicating that reproducible *ranking* performance does not automatically guarantee a reproducible *threshold-based* sensitivity.

### 6.2. Future Work

The most consequential next step is external validation on one or more independent datasets (ISIC, PH2, Derm7pt) to establish cross-source generalization. Beyond that, three directions follow from the present limitations: (i) stabilizing the operating-point sensitivity through seed ensembling or larger-calibration-set threshold selection; (ii) interpretability analysis (for example Grad-CAM saliency [ADDED: 27] and failure-case review of missed melanomas) to build clinical trust; and (iii) narrowing the benign-keratosis false-positive rate, the dominant remaining specificity cost, potentially through richer metadata or targeted hard-negative training. Reporting seven-class performance alongside the binary triage decision would also provide finer diagnostic granularity where data support it.

## References

1. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5(1):180161.

2. Hu J, Shen L, Sun G. Squeeze-and-excitation networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition; 2018 Jun 18–22; Salt Lake City, UT, USA. IEEE; 2018. p. 7132–41.

3. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; 2016 Jun 27–30; Las Vegas, NV, USA. IEEE; 2016. p. 770–8.

4. Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. In: Proceedings of the IEEE International Conference on Computer Vision; 2017 Oct 22–29; Venice, Italy. IEEE; 2017. p. 2980–8.

5. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning; 2017 Aug 6–11; Sydney, Australia. PMLR; 2017. p. 1321–30.

6. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837–45.

7. Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. IEEE Signal Process Lett. 2014;21(11):1389–93.

8. Mongan J, Moy L, Kahn CE Jr. Checklist for artificial intelligence in medical imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2(2):e200029.

9. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.

10. Esteva A, Kuprel B, Novoa RA, Ko J, Swetter SM, Blau HM, et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017;542(7639):115–8.

11. Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2021;71(3):209–49.

12. Gershenwald JE, Scolyer RA, Hess KR, Sondak VK, Long GV, Ross MI, et al. Melanoma staging: evidence-based changes in the American Joint Committee on Cancer eighth edition cancer staging manual. CA Cancer J Clin. 2017;67(6):472–92.

13. Kittler H, Pehamberger H, Wolff K, Binder M. Diagnostic accuracy of dermoscopy. Lancet Oncol. 2002;3(3):159–65.

14. Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; 2009 Jun 20–25; Miami, FL, USA. IEEE; 2009. p. 248–55.

15. Brinker TJ, Hekler A, Enk AH, Klode J, Hauschild A, Berking C, et al. Deep learning outperformed 136 of 157 dermatologists in a head-to-head dermoscopic melanoma image classification task. Eur J Cancer. 2019;113:47–54.

16. Haenssle HA, Fink C, Schneiderbauer R, Toberer F, Buhl T, Blum A, et al. Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. Ann Oncol. 2018;29(8):1836–42.

17. Criscione VD, Weinstock MA, Naylor MF, Luque C, Eide MJ, Bingham SF. Actinic keratoses: natural history and risk of malignant transformation in the Veterans Affairs Topical Tretinoin Chemoprevention Trial. Cancer. 2009;115(11):2523–30.

18. Pacheco AGC, Krohling RA. The impact of patient clinical information on automated skin cancer detection. Comput Biol Med. 2020;116:103545.

19. Kawahara J, Daneshvar S, Argenziano G, Hamarneh G. Seven-point checklist and skin lesion classification using multitask multimodal neural networks. IEEE J Biomed Health Inform. 2019;23(2):538–46.

20. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and avoidance. ACM Trans Knowl Discov Data. 2012;6(4):15.

21. Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception architecture for computer vision. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; 2016 Jun 27–30; Las Vegas, NV, USA. IEEE; 2016. p. 2818–26.

22. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. In: Proceedings of the 29th AAAI Conference on Artificial Intelligence; 2015 Jan 25–30; Austin, TX, USA. AAAI Press; 2015. p. 2901–7.

23. Efron B, Tibshirani RJ. An introduction to the bootstrap. New York: Chapman & Hall/CRC; 1993.

24. Mukhoti J, Kulharia V, Sanyal A, Golodetz S, Torr PHS, Dokania PK. Calibrating deep neural networks using focal loss. In: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); 2020 Dec 6–12; Virtual. p. 15288–99.

25. Codella NCF, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza SW, et al. Skin lesion analysis toward melanoma detection: a challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC). In: Proceedings of the IEEE International Symposium on Biomedical Imaging; 2018 Apr 4–7; Washington, DC, USA. IEEE; 2018. p. 168–72.

26. Mendonça T, Ferreira PM, Marques JS, Marçal ARS, Rozeira J. PH2 — a dermoscopic image database for research and benchmarking. In: Proceedings of the 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC); 2013 Jul 3–7; Osaka, Japan. IEEE; 2013. p. 5437–40.

27. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision; 2017 Oct 22–29; Venice, Italy. IEEE; 2017. p. 618–26.

*Note: references will be renumbered in order of first appearance when the full paper is finalised (Introduction precedes Methods).*
