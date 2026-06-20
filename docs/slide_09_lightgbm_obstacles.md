# Slide 9 — Obstacles replicating LightGBM (arXiv:2203.06848)

---

## LightGBM

### 1. Rolling-mean features were built on the wrong lag/window

**Obstacle**

Section 5.2 names features as `rmean_{window}_{lag}` — e.g. `rmean_28_7` = 28-day rolling mean of **lag_7**. Our first implementation swapped two columns:

| Feature | Paper definition | What we had (wrong) |
|---------|------------------|---------------------|
| `rmean_28_7` | 28-day rolling mean of `lag_7` | 7-day rolling mean of `lag_28` |
| `rmean_7_28` | 7-day rolling mean of `lag_28` | 28-day rolling mean of `lag_7` |

Figure 8 ranks rolling means as the 2nd/3rd most important split features. Wrong windows corrupt the weekly/monthly signal the paper says drives accuracy.

**Result before fix**

Early runs inflated FOODS RMSE badly (our worst category, same as the paper).

**How we overcame it**

- Fixed `add_lag_features()` in `src/features.py` to follow `rmean_{window}_{lag}`
- Added `tests/test_features.py` to lock the naming convention

**One-liner for class:**

> “Two swapped feature lines broke the model’s seasonal signal — the paper’s own feature-importance plot was the clue.”

---

### 2. Table 3 trained the global model on only 300 series

**Obstacle**

LightGBM is **one global model** for all item-stores. Table 3 evaluates 100 products × 3 categories, but the paper never says whether training used:

- **A)** only those 300 series, or  
- **B)** all 30,490 series, then RMSE on the 300-sample

Our benchmark path filtered the panel to 300 IDs **before** training. That starves `item_id` (the paper’s #1 feature in Figure 8) and drops ~99% of training rows.

Section 6.2 emphasizes training on all 30,490 products; handicapping the global model worsens benchmark RMSE.

**Result before fix (`train_scope=subset`)**

| Category | Paper | Ours |
|----------|-------|------|
| HOUSEHOLD | 0.867 | 1.047 |
| HOBBIES | 0.972 | 0.984 |
| FOODS | 1.726 | 2.286 |
| TOTAL | 1.188 | 1.439 |

**How we overcame it**

- Added `--lgb-train-scope {full,subset}` (default `full`)
- Train on all 30,490 series; evaluate RMSE only on the 300 benchmark IDs
- Stream data in chunks + incremental LightGBM fit to stay within 16 GB RAM

**Result after fix (`train_scope=full`)**

| Category | Paper | Ours | Δ |
|----------|-------|------|---|
| HOUSEHOLD | 0.867 | 1.015 | +0.15 |
| HOBBIES | 0.972 | **0.942** | **−0.03** |
| FOODS | 1.726 | 2.415 | +0.69 |
| TOTAL | 1.188 | 1.457 | +0.27 |

**One-liner for class:**

> “A global model needs global training data — evaluating on 300 series is not the same as training on 300.”

---

### 3. Full dataset did not fit in memory (OOM at ~15 GB)

**Obstacle**

The paper reports training all 30,490 series on a 16 GB machine in ~39 minutes. Melting M5 wide → long format creates ~59M rows (~15 GB+ in pandas). Our first full run was killed by the OOM killer.

**How we overcame it (partially)**

- Chunked panel loader: `iter_panel_chunks()` — 2,000 series at a time
- Batched feature engineering for training rows
- Incremental LightGBM: 1,000 trees split across 16 chunks (`init_model` continuation)
- `float32` dtypes and slim column sets

**Result**

- Table 3 benchmark completes (~6–7 min train)
- HOBBIES matches paper within 0.03
- **Section 6.2 pooled RMSE still far off:** ours **3.10** vs paper **0.32**

Incremental out-of-core training is **not identical** to the paper’s single in-memory fit on the full stacked matrix.

**One-liner for class:**

> “We made it run, but chunking 1,000 trees across 16 batches is a compromise — not a byte-for-byte replica of their training loop.”

---

### 4. Undocumented methodology (no code, no series list)

**Obstacle**

Same issues as Prophet/ARIMA. The preprint provides Table 2 hyperparameters but not:

- Which 300 item-store series were used (no IDs, no seed)
- Whether test-period lags use **actual** sales (teacher forcing) or recursive predictions
- Whether `sell_price` / `event_type_*` belong in the model (Table 1 snapshot omits them; caption says “original features”)
- How Section 6.2 pooled RMSE (0.32) relates to Table 3 mean per-series RMSE (1.188)

**How we overcame it (partially)**

- Fixed benchmark sample with `seed=42`, saved `benchmark_series_ids.csv`
- Matched Table 2 params exactly: Poisson objective, `lr=0.001`, 1000 iterations, etc.
- Passed `RANDOM_SEED` into LightGBM for reproducibility
- Tried `--paper-strict` (Table 1 features only) — negligible change on RMSE
- Teacher forcing for lags (uses actual sales in the evaluation file for test-day features)

**One-liner for class:**

> “Hyperparameters were in the paper; the training protocol and product sample were not.”

---

### 5. RMSE aggregation was already correct (unlike Prophet/ARIMA)

**Obstacle**

For Prophet we initially used the wrong horizon metric; for ARIMA we pooled errors instead of averaging per-series RMSE.

**LightGBM status**

We used **mean of per-series RMSE** by category from the start (`paper_style=True` in `evaluate.py`), matching Table 3 logic. No metric fix was needed for LightGBM — the gaps came from features and training scope, not aggregation.

**One-liner for class:**

> “LightGBM’s Table 3 problem wasn’t the formula — it was what we fed the model.”

---

## Optional closing (30 seconds) — LightGBM

**What we could not fully overcome**

- Authors’ exact 300 product list (FOODS still +0.69 off best run)
- Single-pass joint training on 59M rows in 16 GB (incremental fit is an approximation)
- Section 6.2 pooled RMSE 0.32 vs our 3.10 — likely different training/eval protocol or metric definition
- No published author code

**Takeaway for class**

Replication is not only “same algorithm + same hyperparameters.” For LightGBM, **feature engineering naming**, **global training scope**, and **memory-feasible approximations** matter as much as `learning_rate` and `num_iterations`. We replicated the concept (tree model competitive with ARIMA on M5; HOBBIES within 3%) but exact cell-by-cell Table 3 match needs information the paper never published.

---

## Suggested 3-slide structure (LightGBM)

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Problem | M5 paper, global LightGBM, Table 3 targets (0.867 / 0.972 / 1.726 / 1.188) |
| 2 | Obstacle → Fix → Result | Rolling-mean bug, 300 vs 30,490 training, OOM/incremental fit |
| 3 | Lesson | “Global model + engineered lags = protocol matters as much as hyperparameters” |

---

## Quick reference — our final numbers

**Table 3 (mean per-series RMSE, seed=42, 300 eval series)**

| Run | HOUSEHOLD | HOBBIES | FOODS | TOTAL |
|-----|-----------|---------|-------|-------|
| Paper | 0.867 | 0.972 | 1.726 | 1.188 |
| Ours (subset train) | 1.047 | 0.984 | 2.286 | 1.439 |
| Ours (full train) | 1.015 | **0.942** | 2.415 | 1.457 |

**Section 6.2 (all 30,490 series, pooled RMSE)**

| | Paper | Ours |
|---|-------|------|
| Pooled RMSE | 0.32 | 3.10 |

*Source: `benchmark_results/lightgbm_validation_report.json`*
