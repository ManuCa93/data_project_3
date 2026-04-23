# Notebook Long-Memory Notes

This README is a compact memory aid for the full notebook

## 1) Project goal

The notebook explores a large scientific-paper dataset and builds features for link prediction and citation analysis. The workflow includes:
- data loading from DuckDB / S3 parquet
- exploratory data analysis
- visualization
- missing-data analysis
- entity-resolution and normalization checks
- feature extraction and feature selection
- graph-based modeling and tabular pairwise baselines
- export of engineered datasets

## 2) Dataset characteristics

- Source: DBLP scientific papers dataset
- Scale: over 6.7 million papers
- Format: JSONL originally, then used as a parquet source in the notebook pipeline
- Main entities: papers, authors, keywords, references, venues, years, document types, languages
- Important fields used in the notebook: id, title, abstract, year, authors, keywords, references, venue, lang, doc_type, doi, page_start, page_end, n_citation

## 3) Core technical requirements

The workflow must satisfy these requirements:
- perform data exploration and visualization
- perform feature extraction and feature selection
- improve data quality by addressing missing data, including missing IDs and missing organizations
- handle entity-resolution issues, such as the same author having an organization in one year but missing it in another
- account for spelling variations in author surnames, such as Mitrovic vs Mitrović
- correct inconsistencies between venue names and document types, such as an international conference being incorrectly labeled as a journal

## 4) Notebook structure and memory map

### Data loading and stable table creation

The notebook opens a DuckDB connection and loads the parquet source from S3. It creates a stable local table called `papers` with derived fields:
- n_authors
- n_keywords
- n_references
- title_len
- abstract_len
- page_count

This table is the main working base for almost everything else.

### Exploration and visualization

The notebook includes visual checks for:
- publication counts by year
- document type distribution
- language distribution
- citation distribution
- citations by document type
- citations over time
- authors per paper over time
- prolific authors
- top keywords
- venue volume
- missingness by feature

### Missingness and data quality

The notebook explicitly measures missingness for:
- id
- title
- abstract
- keywords
- year
- authors
- references
- lang
- venue
- doc_type
- doi
- page_start
- page_end
- n_citation

It also checks missingness for nested author attributes:
- author.id
- author.org

Important reminder:
- missing author.org is much more serious than a simple null column issue because it affects entity resolution and affiliation features
- author.id should be treated as the primary identity key whenever available

### Author normalization and entity resolution

The notebook explores:
- spelling variation in author names
- normalization of names to detect likely aliases
- temporal inconsistency in author organizations

Key idea:
- normalize names by lowercasing, removing punctuation, and collapsing whitespace
- keep a small manual alias map for recurring variants
- use author.id as the stable identity when possible

### Venue and document-type consistency

The notebook also needs a cleanup mindset for metadata inconsistencies such as:
- conference incorrectly labeled as journal
- venue string variants that should map to a canonical venue name

This is important for feature engineering because venue-based signals are strong predictors.

## 5) Feature engineering memory map

The notebook builds several feature groups and then merges them into a master table called `papers_with_features`.

### Text features

From title and abstract:
- title_word_count
- abstract_word_count
- title_abstract_ratio
- title_complexity
- abstract_complexity
- has_numbers_in_title
- has_numbers_in_abstract

### Author features

- n_authors
- avg_author_citations
- max_author_citations
- multi_authored

### Network / reference features

- n_references
- references_author_ratio
- citation_richness
- has_high_ref_count

### Temporal features

- years_since_publication
- publication_era
- decade
- is_recent

### Quality / impact features

- citation_per_year
- is_highly_cited_for_year
- impact_score

### Venue / domain features

- venue
- venue_median_citations
- venue_volume
- is_top_venue
- is_english
- venue_tier

## 6) Analysis memory map

The notebook computes correlations between the target `n_citation` and engineered features, then studies interactions such as:
- venue tier vs citations
- author reputation tiers vs citations
- era vs citations
- venue tier × author tier interactions

The target variable used for modeling is:
- `n_citation`

The master feature table is:
- `papers_with_features`

## 7) Modeling memory map

### Model 1: Graph Neural Network

The notebook builds a citation graph and trains a GCN-style link predictor.
Purpose:
- predict whether one paper cites another

Inputs:
- node features from paper metadata
- citation edges from references

Evaluation:
- ROC-AUC
- Average Precision

### Model 3: Random Forest baseline for pair-level prediction

The notebook also builds a pairwise supervised dataset for citation-link prediction.

Important rule:
- avoid leakage from citation-count features like `n_citation`, `impact_score`, or other future-derived popularity signals when predicting links

Pair features include:
- year_gap
- same_year
- same_decade
- same_venue
- author_overlap
- author_jaccard
- keyword_jaccard
- title_jaccard
- abstract_jaccard
- bibliographic_coupling
- reference_jaccard
- length and count features for each paper in the pair

Evaluation and interpretation:
- ROC-AUC
- PR-AUC
- Brier score
- confusion matrix
- permutation importance
- SHAP when available

## 8) Practical reminders for future work

- Always prefer the stable `papers` table for exploratory work.
- Use `papers_with_features` for modeling and correlation analysis.
- Missing organizations and name variants are not minor issues; they affect downstream identity and collaboration features.
- Venue normalization is critical because venue prestige is one of the strongest signals.
- When predicting citation links, keep the problem pair-based and avoid leakage from global popularity metrics.
- The notebook is both an EDA report and a feature-engineering pipeline, so changes should preserve both roles.

## 9) Short memory checklist

Before continuing the notebook, remember:
1. Load data into DuckDB.
2. Build and reuse `papers`.
3. Inspect missingness and quality.
4. Normalize authors and venues.
5. Engineer text, author, network, temporal, quality, and venue features.
6. Merge everything into `papers_with_features`.
7. Correlate features with `n_citation`.
8. Train graph and pairwise models.
9. Export outputs.

## 10) Current Issues — Forensic Audit (last updated 2026-04-24)

This section is an evidence-based audit of every issue currently present in the notebook. Each finding cites the actual cell number and code path.

### Raw Dataset Schema (for reference)

The original parquet contains these columns (cell 5 `DESCRIBE` output):

| Column | Type | Meaning |
|---|---|---|
| `id` | VARCHAR | Paper ID |
| `references` | VARCHAR[] | **List of paper IDs this paper cites** (outgoing edges) |
| `n_citation` | BIGINT | **Number of times this paper has been cited by others** (incoming count) |

These are fundamentally different: `references` = who I cite, `n_citation` = how popular I am.

---

### Issue 1: Feature-Level Data Leakage in Node Features (MITIGATED but FRAGILE)

**What the notebook does right:** Cell 85 defines a `leaky_features` list that correctly identifies 11 features derived from `n_citation` (the target):

```python
leaky_features = [
    "citation_per_year",      # n_citation / (2024 - year)
    "impact_score",           # n_citation-derived
    "citation_richness",      # n_citation-derived
    "is_highly_cited_for_year", # threshold on n_citation
    "avg_author_citations",   # mean of n_citation across author's papers
    "max_author_citations",   # max of n_citation across author's papers
    "venue_median_citations", # median n_citation for the venue
    "years_since_publication",# not leaky per se, but listed
    "n_references",           # count of outgoing references
    "references_author_ratio",# derived from n_references
    "has_high_ref_count",     # threshold on n_references
]
```

Cell 86 then **correctly drops** these before building the train/val/test splits used by LGBM and XGBoost.

**What is still problematic:**
- `n_references` (outgoing reference count) is NOT the same as `n_citation` (incoming citation count). `n_references` is a legitimate feature — it measures how many papers the author chose to cite, which is knowable at publication time. Including it in `leaky_features` is **overly conservative** and throws away a valid signal.
- `years_since_publication` is also not leaky — it is a deterministic function of the publication year and the current date. Including it in the leaky list is incorrect.
- The real leaky features are the ones computed from `n_citation`: `citation_per_year`, `impact_score`, `citation_richness`, `is_highly_cited_for_year`, `avg_author_citations`, `max_author_citations`, `venue_median_citations`. These ARE correctly dropped.

**Verdict:** The LGBM and XGBoost models do NOT have target leakage from their feature set. However, two legitimate features (`n_references`, `years_since_publication`) are unnecessarily dropped.

---

### Issue 2: Global Statistics Computed Before Train/Test Split (MODERATE LEAKAGE)

**Where:** Cells 60, 66, 68 compute aggregate statistics across the **entire dataset** before any temporal split:

- Cell 60: `avg_author_citations` and `max_author_citations` — computed over ALL papers for each author
- Cell 66: `year_citation_stats` — median/mean `n_citation` per year, computed over ALL papers
- Cell 68: `venue_stats` — median/mean `n_citation` per venue, computed over ALL papers

These aggregates include papers from 2021-2027 (validation and test sets). When these statistics are used as node features for papers in the training set (≤2020), they leak future information.

**Impact:** These features are in the `leaky_features` list and ARE dropped before LGBM/XGBoost training. So the leakage does NOT affect model performance. However:
- The EDA correlation analysis (cells 72-77) uses these pre-split aggregates, which means the "top predictive features" ranking shown in the notebook is **misleading**. Features like `impact_score` (r=0.63) appear highly predictive partly because they contain future information.
- Any conclusions drawn from the EDA about feature importance are unreliable.

---

### Issue 3: Random Forest Model Uses a Completely Independent Pipeline (DESIGN ISSUE)

**Where:** Cells 108-123 build an entirely separate pair-level dataset and feature set for the Random Forest, bypassing the LGBM/XGBoost pipeline entirely.

**What it does differently:**
- Constructs its own positive/negative pairs from the raw parquet (cell 108-112)
- Builds pair-level features using raw Python functions (`build_pair_features` in cell 114), not the cleaned SQL tables
- Uses a simple `train_test_split` (cell 117) instead of the temporal split used by LGBM/XGBoost (cell 86)

**Problems:**
1. **No temporal split:** The RF uses random 80/20 splitting, meaning future papers can appear in the training set. This is a form of temporal leakage — in real-world citation prediction, you can only train on past papers and predict future citations.
2. **Ignores data cleaning:** The author canonicalization work (cells 40-47) is never used by the RF. It re-parses raw author names from scratch.
3. **Small sample size:** Only 5,000 positive + 5,000 negative pairs (cell 110), vs. ~1.6M per class for LGBM/XGBoost. This makes results unreliable.
4. **Different features:** The RF uses handcrafted pair features (Jaccard, overlap counts), while LGBM/XGBoost concatenate node features. This makes cross-model comparison unfair.

---

### Issue 4: Unfair Model Comparison

**Where:** The three models operate on different data, different features, and different splits:

| Aspect | LGBM (cell 96) | XGBoost (cell 100) | Random Forest (cell 118) |
|---|---|---|---|
| Task | Link prediction (binary) | Link prediction (binary) | Link prediction (binary) |
| Feature type | Concatenated node features (25 + 25 = 50) | Same as LGBM | Handcrafted pair features (28) |
| Split method | Temporal (≤2020 / 2021-22 / 2023+) | Same as LGBM | Random 80/20 |
| Sample size | ~3.2M train rows | Same as LGBM | ~8K train rows |
| Test set | ~2M rows | Same as LGBM | ~2K rows |

LGBM and XGBoost are fairly compared against each other (cell 100 prints a "FAIR COMPARISON" table). But the Random Forest cannot be compared to them because it uses a different split, different features, and a vastly smaller dataset.

**The checklist requires:** "at least 3 different types of models" with fair comparison. This is **not satisfied** in its current form.

---

### Issue 5: Disconnected Data Cleaning

**Where:** Cells 40-47 build `author_name_canonical` and `author_canonical_name` tables in DuckDB via SQL. Cells 46-47 create cleaned reference/author join tables.

**Problem:** None of the three models use these cleaned tables:
- LGBM/XGBoost (cell 91) build their feature matrix from the `model_splits/*.parquet` files, which were derived from `papers_with_features_with_refs` — which uses the raw `features_author` table (cell 60), not the canonicalized authors.
- The Random Forest (cell 114) re-parses raw parquet data in Python.

**Result:** The data cleaning section (cells 39-47) has no downstream impact on any model. It is effectively dead code.

---

### Issue 6: GCN / Neural Network Model Is Commented Out (CHECKLIST GAP)

**Where:** Cells 80, 93-95 contain commented-out PyTorch/GCN code. Cell 82 is also commented out.

**Problem:** The checklist asks for "more complex models." The notebook acknowledges this by including GCN code, but it is entirely commented out and never runs. The only working models are three tree-based methods (LGBM, XGBoost, Random Forest).

**Impact:** If "more complex models" is a graded requirement, this is not met.

---

### Issue 7: EDA is Node-Focused, Target is Pair-Focused (COHERENCE ISSUE)

**Where:** Cells 9-22 (EDA) and cells 72-77 (correlation analysis).

**Problem:** The entire EDA section analyzes single-paper properties: publication counts, citation distributions, top keywords, author prolificacy, etc. The correlation analysis (cells 72-77) ranks features by their correlation with `n_citation` (a node-level metric).

But the actual modeling target is **pair-level**: "does paper A cite paper B?" The EDA never explores pair-level phenomena such as:
- Distribution of year gaps between citing and cited papers
- How often citing papers share a venue or keywords with cited papers
- Author overlap distributions in positive vs. negative pairs

This creates a narrative disconnect — the EDA tells a story about what makes a paper popular, but the models solve a different problem (what makes one paper cite another).

---

### Issue 8: Code Quality and Consistency (MINOR)

1. **Mixed languages:** Cells 57, 59, 61, 63, 65, 67, 78 have markdown explanations in Italian. Other cells use English. The code comments in cell 86, 91, 96, 100 are also in Italian.
2. **Dead/commented-out code:** Cells 74, 80, 82, 87, 93-95, 98-99, 125 contain commented-out code blocks. Cell 125 has a comment "FORSE DA TOGLIERE" (maybe remove).
3. **Redundant cell 101:** This cell was used for debugging (`n_references` vs `target_variable` correlation check) but remains in the final notebook with its output displayed. It should be removed or moved to an appendix.

---

### Summary Table

| Issue | Severity | Impact on Results |
|---|---|---|
| 1. Feature leakage in LGBM/XGB | ✅ Mitigated | Leaky features correctly dropped. Two non-leaky features unnecessarily dropped. |
| 2. Global stats pre-split | ⚠️ Moderate | EDA correlation analysis is misleading. No impact on final models (features dropped). |
| 3. RF uses separate pipeline | 🔴 Serious | RF has temporal leakage (random split), tiny sample, ignores data cleaning. |
| 4. Unfair model comparison | 🔴 Serious | Three models cannot be compared — different data, features, splits, and scales. |
| 5. Data cleaning is dead code | ⚠️ Moderate | Author canonicalization (cells 40-47) is never used by any model. |
| 6. GCN model commented out | ⚠️ Moderate | "More complex model" checklist item not satisfied. |
| 7. EDA/target mismatch | ⚠️ Moderate | Node-level EDA doesn't support the pair-level prediction task. |
| 8. Code quality/consistency | 🟡 Minor | Mixed languages, dead code, debugging cells left in. |

### Recommended Fixes (Priority Order)

1. **Fix RF temporal leakage:** Replace `train_test_split` with the same temporal split used by LGBM/XGBoost (train ≤2020, val 2021-22, test ≥2023).
2. **Unify the model comparison:** train all 3 models on the same features, first by using the flat features computed in the notebook, and then by computing pair-level features (year gap, Jaccard similarity, etc.) to make the comparison fairer
3. **Connect data cleaning:** Use the `author_name_canonical` tables when computing author overlap and Jaccard features.
4. **Add pair-level EDA:** Before modeling, explore pair-level statistics (year gap, venue overlap, keyword Jaccard) to build a coherent narrative.
5. **Un-drop `n_references` and `years_since_publication`:** These are not leaky — they are knowable at publication time.
6. **Uncomment or replace the GCN model** to satisfy the "more complex model" requirement.
7. **Clean up code:** Remove debugging cells, standardize language to English, delete commented-out code blocks.

