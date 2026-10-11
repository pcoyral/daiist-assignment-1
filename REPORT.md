# Assignment 1 Report

> **DRAFT. Paul: rewrite the prose in your own words before submitting.**
> Per the assignment's AI policy, the ideas and conclusions in this file must be
> yours. The numbers and tables come straight from `artifacts/metrics.json` and
> `artifacts/ablation.csv`. The text was drafted by an AI assistant from the
> pipeline's outputs and from the decisions you made (framing, cost asymmetry,
> duplicate handling). Read every line, check you agree and can explain it, then
> rephrase it. Delete this box when done.

- **Name**:
- **Student ID**:
- **Email**: pcoyral.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

UCI Wine Quality (Cortez et al., 2009): red and white *vinho verde* samples from
northern Portugal. Two semicolon-separated CSVs are committed in `data/`. One row
is one wine sample: 11 physico-chemical lab measurements (acidity, sugar,
chlorides, SO₂, density, pH, sulphates, alcohol) plus a sensory quality score
(0–10, the median of at least 3 expert tasters). Red and white are combined into
one table with an `is_red` flag: **6,497 rows × 13 columns** (1,599 red, 4,898
white), no missing values.

Why this dataset: the lab measurements are cheap and objective, while the quality
score needs a tasting panel. That gap is a real business problem. The features
are also skewed, collinear (density ≈ f(alcohol, sugar)) and behave differently
for red and white, so feature engineering is not trivial.

**Data-quality finding:** 1,177 rows (18%) are exact duplicates. No two rows have
identical chemistry but different scores, so they are probably the same sample
entered twice, or batches that genuinely measured identical.

## Business / real-life framing

**Scenario.** A winery or co-operative produces many batches a season. An expert
tasting panel decides which batches are bottled under the premium label, but
panel time is expensive and limited. The model screens each batch from its
routine lab analysis and decides **whether to send it to the panel as a premium
candidate**.

How this drives the pipeline:

| Decision | Choice | Why (from the framing) |
|---|---|---|
| Target | `premium = quality ≥ 7` (19.0% positive after dedup) | The business decision is binary (panel or not). "≥ 7" is the usual "very good" cut. Predicting the 3–9 score with linear regression would treat an ordinal panel rating as continuous and wouldn't give the panel/no-panel decision directly. |
| Model | Logistic regression | Binary target; outputs a probability that is thresholded on cost. |
| Costs | FP = €50 (ordinary batch wastes panel time), FN = €200 (premium batch sold at the regular price loses margin) | Missing a premium batch costs more than an extra tasting. Assumed figures, and adjustable in the dashboard. |
| Threshold metric | **Minimum expected business cost**, chosen on the validation set | Accuracy would reward "never flag" (81% accurate, €40k cost). With calibrated probabilities the cost-optimal threshold is FP/(FP+FN) = 0.20, not 0.5. |
| Ranking metrics reported | ROC AUC, PR AUC, log loss | PR AUC because positives are the minority. Log loss because the threshold logic relies on calibrated probabilities. |
| Split | **Drop exact duplicates first**, then a stratified 60/20/20 train/val/test split (stratified on premium × colour) | With duplicates, a random split puts identical rows in both train and test, so the test score measures memorisation. There is no date or vintage column, so a time-based split isn't possible. In this scenario every batch is scored on chemistry alone, so a random split is a fair simulation. Stratifying keeps the 19% premium rate and the red/white mix equal across splits. |
| Class imbalance | No class weights or resampling | The cost-based threshold already handles the asymmetry. Re-weighting would distort the probabilities that threshold relies on. |

## Data preparation & feature engineering

All learned preprocessing (clip bounds, spline knots, scaler mean/std) is fit on
the **train split only** (`features.Preprocessor`) and reused unchanged for
val, test and the dashboard.

What was built (`features.py`):

1. **Log transforms** of right-skewed concentrations: residual sugar, chlorides,
   free/total SO₂, sulphates, volatile acidity (skew 1.4–5.3). A linear model then
   treats a change from 1 → 2 g/L sugar more like 10 → 20 than like 10 → 11.
2. **Colour interactions** (`feature × is_red`) for volatile acidity, sulphates,
   residual sugar, citric acid and alcohol. Red and white are different products,
   and a single linear coefficient forces the same effect on both. The fitted
   model confirms it: `sulphates_x_red` is the largest coefficient (+1.34), i.e.
   sulphates matter much more for reds.
3. **Cubic B-splines** (4 quantile knots) on total SO₂, density, volatile acidity
   and free SO₂. These have a "sweet spot" or threshold effect rather than
   "more is better": too little free SO₂ risks spoilage, too much smells of sulphur.
   Splines let a *linear* model fit a curve. The four features were chosen by
   adding a spline to one feature at a time and keeping those that improved CV AUC.
   The others (alcohol, pH, …) didn't improve, so they stay linear.
4. **Winsorizing** at the 0.5% / 99.5% train quantiles, so a few extreme lab
   values (e.g. chlorides at 0.6 g/L) don't pull the fit.
5. **Standardizing**, which is needed so the L2 penalty treats features equally
   and so plain gradient descent in PyTorch converges with one learning rate.

**Ablation** (5-fold stratified CV on train+val, sklearn, C=1; test never touched):

| Features | # | CV ROC AUC (± std) | CV PR AUC | CV log loss | cond(XᵀX) |
|---|---|---|---|---|---|
| raw | 12 | 0.837 ± 0.011 | 0.523 | 0.366 | 177 |
| + log transforms | 12 | 0.841 ± 0.011 | 0.535 | 0.363 | 128 |
| + log + chemistry ratios | 18 | 0.840 ± 0.011 | 0.536 | 0.363 | 221,046 |
| + log + colour interactions | 17 | 0.842 ± 0.012 | 0.541 | 0.361 | 2,191 |
| + log + colour + chemistry | 23 | 0.841 ± 0.012 | 0.539 | 0.362 | 377,150 |
| **+ log + colour + splines (final)** | 33 | **0.846 ± 0.015** | **0.552** | **0.357** | 34,193 |
| final, without winsorizing | 33 | 0.843 ± 0.015 | 0.547 | 0.360 | 880,500 |

**What didn't work, and why:**

- **Domain chemistry ratios** (bound SO₂, free/total SO₂ ratio, *molecular SO₂*
  from free SO₂ and pH, alcohol/density, sugar/alcohol, total acidity) added
  nothing, individually or together. They are almost linear combinations of
  features already in the model, so a linear model gains no new information.
  They also blew up the condition number (128 → 221k), which made the
  PyTorch gradient descent much slower to converge. Kept in the code as the
  `"chem"` ablation group, but **dropped from the final model**.
- **First attempt at splines made XᵀX singular** (condition number ≈ 4×10¹⁶).
  A cubic B-spline basis can already represent a straight line, so keeping the
  raw column *and* its spline basis is perfectly collinear. Fix: the basis
  replaces the raw column, and `include_bias=False` removes the column that is
  collinear with the intercept.
- **One validation split was too noisy to rank variants.** The first ablation
  used the 1,064-row validation set, and the variants differed by ≈0.002 AUC in a
  random order. Switching to 5-fold CV gave consistent results (std ≈ 0.01–0.015).
- **Squared terms** as an alternative to splines: CV AUC 0.843, no better than
  without them.
- Overall, feature engineering bought ≈ +0.01 AUC / −0.009 log loss. That's modest
  but consistent. Most of the signal is in alcohol, volatile acidity, chlorides and
  colour, which a linear model already captures from the raw data.

## Modeling: three implementations, one model

**Model:** L2-regularised logistic regression. All three implementations minimise
the *same* objective. sklearn minimises `C·Σ logloss + ½‖w‖²`. Dividing by
`C·n` gives `mean logloss + ‖w‖²/(2Cn)`, which both PyTorch versions minimise
explicitly (bias not penalised, as in sklearn). I didn't use `SGD(weight_decay=…)`
in the nn.Module version because it would also shrink the bias. Everything is
float64 and full-batch.

| | scikit-learn | Manual PyTorch | nn.Module + optim |
|---|---|---|---|
| Optimiser | L-BFGS (tol 1e-10) | hand-written GD: `w -= lr * w.grad` under `no_grad`, grads zeroed | `torch.optim.SGD`, `nn.Linear`, `BCEWithLogitsLoss` |
| Init | – | zeros | PyTorch default (random, seed 42) |
| Steps | to convergence | 20,000 full-batch | 20,000 full-batch |

**Tuning (sparingly, validation only):** C ∈ {0.01, 0.1, 1, 10} by validation
log loss → **C = 1** (0.3429; C=10 is the same, C=0.01 underfits at 0.357).
Learning rate ∈ {0.1, 0.5, 1, 2} by final training objective → **lr = 1.0**.
lr = 2.0 **diverges** (objective 0.436 vs 0.353): it's above the stability limit
of ≈ 2/L for this problem's curvature.

**Results on the held-out test set** (1,064 batches, 202 premium; threshold
picked on validation):

| Model | ROC AUC | PR AUC | Log loss | Threshold | Precision | Recall | FP | FN | Cost € |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.8105 | 0.5052 | 0.3880 | 0.20 | 0.401 | 0.743 | 224 | 52 | **21,600** |
| Manual PyTorch | 0.8105 | 0.5052 | 0.3880 | 0.20 | 0.401 | 0.743 | 224 | 52 | **21,600** |
| nn.Module + optim | 0.8105 | 0.5052 | 0.3880 | 0.20 | 0.401 | 0.743 | 224 | 52 | **21,600** |
| Baseline: alcohol-only logistic | 0.7791 | 0.4612 | 0.4059 | 0.18 | 0.370 | 0.738 | 254 | 53 | 23,300 |
| Baseline: never flag | 0.5 | 0.190 | 0.486 | – | – | 0 | 0 | 202 | 40,400 |
| Baseline: flag everything | 0.5 | 0.190 | – | – | 0.190 | 1 | 862 | 0 | 43,100 |

The model **halves the cost** of the best trivial policy (€21.6k vs €40.4k,
≈ €17.7 saved per batch). It also beats a one-feature alcohol model by €1,700,
which is the honest measure of what the remaining features and engineering add.

The validation-chosen threshold (0.20) is exactly the theoretical
FP/(FP+FN) = 50/250. That's consistent with logistic regression giving
reasonably calibrated probabilities (see the calibration plot in the dashboard).

**Do the three agree?** Yes, to numerical precision:

| vs sklearn | Train-objective gap | max \|Δ coef\| | Δ intercept | max \|Δ prob\| (test) | Test label disagreements |
|---|---|---|---|---|---|
| Manual PyTorch | 1.6e-11 | 1.3e-4 | 3.6e-7 | 8.4e-6 | 0 |
| nn.Module | 2.2e-11 | 1.7e-4 | 5.7e-7 | 1.1e-5 | 0 |

**They did *not* agree at first, and here's what I found.** With 4,000 epochs, the
PyTorch coefficients differed from sklearn's by up to **0.14** (on `is_red`),
even though predicted probabilities agreed to within 0.01 and the training
objective differed by only 1e-5. The cause is **ill-conditioning**: `is_red` is
strongly correlated with its five interaction columns, so the loss surface is a
long, flat valley in that direction. Gradient descent moves along it very slowly,
and many coefficient combinations give almost the same predictions. A gap
of 1e-5 in the objective can hide a 0.14 gap in the coefficients. Measured
convergence: max |Δ coef| = 0.14 at 4k epochs, 0.005 at 10k, 0.0001 at 20k. So
I raised epochs to 20k instead of hiding it. sklearn's L-BFGS uses curvature
information, so it isn't slowed down this way. The dashboard's "Three-way
agreement" tab plots this convergence curve. Bottom line: **identical
predictions don't guarantee identical coefficients**, and coefficients of
correlated features shouldn't be read one at a time.

## Limitations & next steps

- **Validation → test drop.** Validation AUC 0.862 and CV AUC 0.846 ± 0.015, but
  test AUC 0.811 (cost per batch €15.9 on val vs €20.3 on test). One 1,064-row test
  split is noisy (≈2 CV std below the mean). Validation was also used to pick
  C and the threshold, so it is slightly optimistic. *Next:* nested CV or repeated
  splits to report a distribution rather than one number.
- **Costs are assumptions.** €50/€200 are illustrative. The optimal threshold
  moves with the ratio, which is why the dashboard lets you edit both costs.
  *Next:* get real panel cost per batch and the premium/regular price margin from
  the winery.
- **No time dimension.** There is no vintage or date, so I can't test whether
  the model survives a new harvest (drift in climate, process, panel members).
  *Next:* with dated data, train on past vintages and test on the latest one.
- **Duplicates.** I assumed exact duplicates are double entries and dropped them.
  If they're genuinely separate batches, I'm under-weighting common wine
  profiles. The group-aware split alternative would keep them without leakage.
- **Label noise / subjectivity.** Quality is a median of ≥3 tasters on an integer
  scale, and "≥ 7" is a hard cut. A 6 and a 7 can be nearly identical wines, which
  caps achievable AUC. *Next:* model the score ordinally, or treat 6.5-ish wines
  as uncertain.
- **Linear model ceiling.** Even with splines and interactions, logistic
  regression can't capture complex interactions. A gradient-boosted tree would
  likely do better and is the natural next model (out of scope here by design).
- **Only one region/style** (vinho verde). Not transferable to other regions
  without retraining.

## Generative AI use disclosure

I used Claude Code (Anthropic, Claude Opus 5.5) as a coding assistant. It:
- wrote the code (`train.py`, `features.py`, `models.py`, `app.py`) implementing
  the decisions I made: premium-screening framing, quality ≥ 7 target, missed
  premium costing more than a wasted tasting, dropping duplicates before the split;
- ran the feature experiments (ablation, spline selection, convergence check) and
  reported the numbers above;
- built the Gradio dashboard and its plots;
- drafted this report from the pipeline outputs, which I then reviewed and
  rewrote in my own words. *(Paul: make sure this last part is true before you
  submit.)*
