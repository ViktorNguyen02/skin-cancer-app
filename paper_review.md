# Manuscript Review — From-Scratch SE-Residual CNN for Skin-Lesion Triage

**Reviewed file:** `paper_draft.md`
**Checked against:** `documentation/EXPERIMENT_TRACKING.md`, `documentation/ablation_table.md`, `documentation/statistical_analysis.md`
**Verdict up front:** Do **not** restart. This is a fundamentally sound, above-average draft that is roughly 70% of the way to a submittable paper. The core science is done, honestly reported, and — importantly — the numbers in the draft match the source data almost exactly. The remaining work is finishing (Related Work, comparison table, figures) and tightening a few overstatements, not rebuilding.

---

## 1. Structure & Flow

**What's right**

- The macro-structure is the correct, complete IMRaD arc for a clinical-ML paper: Abstract → Intro → Related Work → Methods → Results → Discussion → Conclusion → References. Nothing is out of order.
- The Introduction is genuinely strong: it states the gap (transfer-learning monoculture), a second gap (discarded metadata), and a third (leakage + missing uncertainty), then converts them into three crisp research questions (RQ1–RQ3) that the Results and Conclusion answer one-for-one. That RQ→Result→Conclusion traceability is exactly what reviewers look for and many drafts lack.
- Methods are unusually complete and reproducible: dataset, binary mapping with clinical justification, leak-free split, architecture, loss, training, operating-point policy, calibration, statistics, and an explicit CLAIM/TRIPOD-AI reporting section. This is a real strength.
- The Discussion separates interpretation from limitations cleanly, and the limitations are honest (single dataset, fixed specificity, binary reformulation, seed-dependent sensitivity).

**What's missing or weak**

- **Related Work (§2) is empty.** It contains only a single subsection heading ("2.1 The From-Scratch Angle Against the Transfer-Learning Mainstream") and no text. This is the single biggest structural hole. A reviewer cannot assess novelty without it, and the paper's whole framing ("against the transfer-learning mainstream") demands that the mainstream actually be surveyed.
- **§5.3 comparison table (Table 8) is an unfilled template** with `[CITE]`/`[value]` placeholders and a visible instruction block. It must be populated or the paper is desk-rejectable on sight.
- **No figures exist.** The draft references a reliability diagram as "Figure X" (§4.4) — a literal placeholder — and has no architecture diagram, no ROC curve, no confusion/threshold visual. A from-scratch-architecture paper with zero architecture figure will read as unfinished.
- Minor: the Abstract has no explicit structured background/methods/results/conclusion segmentation. Fine for many venues, but check the target journal's abstract format.

**Transitions:** Section-to-section flow is good. The one rough seam is Intro→Related Work→Methods, because Related Work is a void the reader falls into. Filling it fixes the flow.

---

## 2. Template & Formatting

- **Current state:** The draft is Markdown with deeply nested decimal headings (1 → 1.1 → 1.1.1) and Vancouver/ICMJE-style numbered references (venues like *Sci Data*, *Radiol Artif Intell*, *BMJ*). It does **not** yet conform to any single publication template.
- **Recommended target/template.** The paper is a clinical-prediction / medical-imaging AI study that already reports to CLAIM and TRIPOD-AI. That points at a medical-AI venue rather than a pure CV conference. Reasonable targets and their templates:
  - *Radiology: Artificial Intelligence*, *npj Digital Medicine*, or *Scientific Reports* — single-column, Vancouver references. Lowest reformatting cost from where you are; matches the CLAIM/TRIPOD framing best. **Recommended primary target.**
  - *IEEE J. Biomedical and Health Informatics (JBHI)* or *Computers in Biology and Medicine* — IEEE double-column, `IEEEtran`. Good fit for the architectural contribution, but more reformatting and a citation-style switch.
  - Avoid framing it as a top-tier CV conference (CVPR/MICCAI main track) paper: single-dataset, no external validation, and modest ΔAUC make that a hard sell. A solid journal is the realistic path.
  - **Why it matters:** picking the venue now determines column format, reference style, figure count, and word limits — reformatting late is wasted effort and a common cause of last-minute errors.
- **Deep decimal headings (1.3.1, 5.2.4, 6.1.1)** are heavier than most journals want. Collapse to at most two levels once you move into the template.
- **Reference ordering is currently wrong for a numbered style.** `[10]` (Esteva) is cited in the Introduction before `[1]` (HAM10000). The draft's own closing note acknowledges renumbering is pending — do it, because citation order by first appearance is mandatory in Vancouver style and reviewers notice immediately.
- **Tables are well-built** (clear headers, aligned, captioned) and will survive review; the only broken one is the placeholder Table 8.
- **No line numbers.** Almost every journal requires line-numbered, double-spaced submission PDFs for review. Add these when you generate the submission version.

---

## 3. Content Accuracy & Consistency

I cross-checked every headline number, CI, p-value, ablation row, per-seed value, and per-diagnosis figure against the three source files. **The draft is remarkably accurate — I found no fabricated or contradicted numbers.** Details:

**Verified consistent (draft ↔ source):**

- Data split (Table 1): Train 5,079 lesions / 6,824 images; Val 897 / 1,190; Test 1,494 / 2,001; Total 7,470 / 10,015 — matches the check3 isolation output exactly, including the ~68/12/20% shares.
- Test composition: n = 2,001 (389 Concerning, 1,612 Benign) — matches, and the per-diagnosis counts sum correctly (227+110+52 = 389; 1,371+190+20+31 = 1,612).
- Champion AUC: released reference 0.9047; five-seed mean 0.9014 ± 0.0095 — matches `statistical_analysis.md` and the tracking headline.
- Operating point: threshold 0.169 at `VAL_SELECTION_SENS = 0.94`; test Sens 0.9229, Spec 0.7171 — matches.
- DeLong: ΔAUC = +0.0097, z = −2.030, p = 0.0423 — matches exactly.
- Confidence intervals (Table 4 / §4.1): DeLong 0.8906–0.9187 and bootstrap 0.8907–0.9179 — match the stats file.
- Ablation (Table 5): Run 0 = 0.883, Run 1 = 0.895 (+0.012), Run 2 = 0.871 (−0.024), Run 3 = 0.8815 (−0.0135, Sens 0.887), Run 4 = 0.9047 — all match `ablation_table.md`.
- Per-diagnosis (Table 6): recalls (mel 92.1%, bcc 90.0%, akiec 98.1%), clearances, and mean-P values all match the Run 4 check3B output.
- Per-seed table (Table 7): all five AUC/Sens pairs match tracking §11b; the specificity column averages to 0.7073 and the "2/5 seeds at 0.882" claim is correct (seeds 2 and 5 = 0.8817).
- ECE 0.0403 → 0.0341 at T = 0.722 — matches.

**Accuracy/consistency issues to fix (none are fatal, but they matter):**

1. **"Significantly improved" is overstated in the Abstract.** The Abstract and Contribution 2 state metadata fusion "significantly improved AUC (p = 0.042)" without qualification. Your own tracking flags this as **borderline** and, crucially, computed on the **single favourable released draw** (0.9047, above the seed mean 0.9014); the mean gain across seeds is only ~0.006, and the image-only model was **not** retrained across seeds, so there is no multi-seed significance test. The body (§4.3, RQ2) does hedge well — but the Abstract should carry the same hedge, e.g. "a modest but statistically significant improvement on the reference model (p = 0.042); the mean cross-seed gain was ≈0.006." Reviewers who read the tracking-level nuance into the borderline p will penalise an unhedged abstract claim.

2. **"Near-identity temperature" is technically wrong (§3.8, §4.4).** T = 0.722 is a ~28% sharpening of the logits, not near-identity — identity is T = 1.0 (that was Run 1's T = 0.992). The conclusion ("model was already well-calibrated") is still supported by the low pre-scaling ECE (0.040), so fix the *reason*, not the claim: say "low pre-scaling ECE (0.040) indicates good calibration; temperature scaling (T = 0.722) produced only a marginal ECE change (→0.034)." Delete "near-identity."

3. **Single-draw per-class figures presented as general (§4.5, §5.1).** The mel 0.319→0.372 / akiec 0.438→0.500 shifts and the per-class recalls are all from Draw B, and your tracking explicitly warns the bkl↔mel benefit *flips* between unseeded draws. §4.5 does caveat this, but §5.1 states the mechanism as settled fact. Add one clause noting these per-class shifts are single-draw and seed-dependent (the AUC gain is the stable part).

4. **"Prospectively selected" threshold (repeated).** The threshold is selected on a held-out validation set before test scoring — that is honest and correct, but "prospective" in clinical-prediction language connotes temporal/future validation. Consider "pre-specified, validation-selected" to avoid a terminology objection.

5. **Minor version-note:** the closing line about references being renumbered is an internal to-do; remove it from the submitted version (do the renumbering instead).

**Method-vs-log consistency:** The described methods (leak-free lesion split at `random_state=42`, val-AUC model selection, focal γ=1.5 + label smoothing ε=0.05, Adam 3e-4 + clipnorm 1.0 + ReduceLROnPlateau, train-only metadata encoders, 4-run cumulative ablation with Runs 2 & 3 rejected, safety-margin threshold, temperature scaling, bootstrap B=2000, 5-seed protocol) all match `EXPERIMENT_TRACKING.md`. One small clarification worth adding: the Run 0 baseline used focal γ=2.0 and Run 1 changed it to 1.5; §3.5 states γ=1.5 flatly, which is correct for the champion but could briefly note the baseline value so Table 3's "calibration" change is self-consistent.

---

## 4. Suggested Improvements (prioritised by impact on acceptance)

**Tier 1 — blocks acceptance; do first**

1. **Write Related Work (§2).** ~4–6 paragraphs surveying: (a) transfer-learning HAM10000 CNNs (EfficientNet/ResNet/DenseNet fine-tuning), (b) attention/ensemble methods, (c) any from-scratch or multimodal (image+metadata) baselines, (d) the data-leakage/lesion-split literature. *Why:* novelty is unassessable without it, and your entire thesis is defined relative to this body of work. Without it the paper is incomplete, full stop.
2. **Populate Table 8 with 5–8 real, cited studies.** *Why:* an empty results-context table is an immediate desk-reject signal and leaves your "not SOTA but honest/leak-free" positioning unsupported. Use the inclusion criteria already written into the template.
3. **Create the figure set.** Minimum: (i) architecture diagram, (ii) ROC curve(s) for champion vs image-only with the operating point marked, (iii) the reliability diagram already referenced as "Figure X," (iv) optionally a per-seed AUC/sensitivity dot plot. *Why:* reviewers skim figures first; a from-scratch-architecture claim with no architecture figure undercuts the core contribution.

**Tier 2 — strongly expected; raises the score**

4. **Hedge the Abstract's significance claim** (see 3.1 above). *Why:* aligns the headline with your own honest multi-seed framing and pre-empts the most likely reviewer attack.
5. **Report a threshold-independent operating summary across seeds** (e.g., specificity-at-fixed-sensitivity or partial AUC), not just the one locked point. *Why:* strengthens RQ3 and shows the seed-sensitivity story is characterised, not just disclosed.
6. **Add a seven-class supplementary result** if data permit, per your own future-work note. *Why:* directly answers the reviewer question "why throw away granularity?" that §5.2.3 anticipates.

**Tier 3 — polish**

7. Fix the "near-identity temperature" wording (3.2 above), the "prospective" terminology (3.4), and the reference ordering.
8. Collapse heading depth to two levels; add line numbers/double-spacing for the submission PDF.
9. Add a short "Data/Code Availability" and "Ethics" statement — required by most medical-AI journals.

**Highest-leverage single move:** finishing Related Work + Table 8 + the figure set. Those three convert an "obviously unfinished" manuscript into a reviewable one; everything else is refinement on top of already-solid science.

---

## 5. Step-by-Step Plan Forward

### Restart? No.

Salvage and finish. The experiments are complete and honestly logged, the writing quality is high, and the numbers are internally consistent and faithful to the data. Restarting would discard a strong Methods/Results core and a well-formed RQ structure for no benefit. The gaps are additive (missing sections/figures/citations) and cosmetic (template, hedging), not structural.

### Order of work (dependency-driven)

Do content that others depend on first: literature → context table → figures → prose polish → template → proofing. Methods and Results need only light edits, so they are *not* the bottleneck — resist re-litigating them.

### Milestones (assumes part-time effort; compress if full-time)

- **Day 1–2 — Choose venue & set template.** Pick the target journal (recommend *Radiology: AI* / *npj Digital Medicine* / *Sci Rep* tier). Set up the manuscript in a reference manager (Zotero/Mendeley) and the venue's template. *Deliverable:* empty templated document with your current text imported.
- **Day 3–7 — Literature & Related Work.** Collect 15–25 references; write §2. *Deliverable:* complete Related Work section.
- **Day 8–10 — Comparison table (Table 8).** Extract AUC/Sens/Spec, task, split protocol, pretrained-or-not from 5–8 studies. *Deliverable:* populated, cited Table 8.
- **Day 11–14 — Figures.** Architecture diagram, ROC, reliability diagram, per-seed plot. *Deliverable:* figure set + captions; remove all "Figure X" placeholders.
- **Day 15–17 — Accuracy/hedging edits.** Fix the abstract significance hedge, the temperature wording, single-draw caveats, terminology; renumber references. *Deliverable:* internally consistent full draft.
- **Day 18–19 — Add statements.** Data/code availability, ethics, author contributions, funding, CLAIM/TRIPOD checklists as supplements. *Deliverable:* submission-complete draft.
- **Day 20–22 — Verification pass & proofing.** Re-run the number cross-check (below), add line numbers/double-spacing, check the template's word/figure limits. *Deliverable:* submission PDF.
- **Day 23–28 — Buffer + co-author/colleague read + revisions.** Realistic 5-day slack for feedback and fixes.
- **~Day 28–30 — First submission.**

That is roughly a **4–5 week** timeline part-time, or **~2 weeks** full-time, from now to first submission. External validation (ISIC/PH2/Derm7pt) is the one thing that would materially raise the venue tier but adds weeks; decide explicitly whether to submit now to a solid journal and add external validation in revision, or hold. Given the manuscript's honesty and completeness, submitting now to a mid-tier medical-AI journal is defensible.

### Recommended tools / workflow

- **Version control:** put the manuscript (and `documentation/`) under git; tag the exact commit that produced each number so a reviewer question maps to a reproducible state.
- **Reference manager:** Zotero or Mendeley with a BibTeX/CSL export in the venue's style — do this before writing Related Work so citations are ordered automatically.
- **Reporting checklists:** fill the actual CLAIM and TRIPOD-AI checklists (you already report to them narratively) and submit as supplements — this is increasingly required and is a fast score-raiser.
- **A statistical-reporting checklist:** for every headline number confirm it carries (metric, point estimate, 95% CI, n, and the exact operating point) — your Results already do this; keep the discipline in the figures/captions.
- **Numbers cross-check as a gate:** before submission, re-verify each reported value against `statistical_analysis.md` / `ablation_table.md` (ideally a tiny script that reads the source tables), so late edits can't silently desync the manuscript from the data.

---

## Bottom line

This is a salvage, not a rebuild. The science is finished and the reporting is honest and accurate — rare strengths. What stands between this draft and a first submission is **three missing pieces (Related Work, the comparison table, figures)** plus a handful of **hedging/wording fixes**, most visibly toning down the Abstract's unqualified "significant" to match your own multi-seed framing. Do the Tier-1 items first; the rest is polish.
