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

## 10) Current Issues — Forensic Audit (Updated)

This section contains an evidence-based audit of the structural, conceptual, and data leakage issues currently present in the notebook.

### Issue 1: Data Leakage via Global Statistics

**Problem:** Several features in the notebook are computed as aggregates over the *entire* dataset (including papers up to 2024). While some numeric features (`venue_median_citations`, `avg_author_citations`) are correctly dropped via the `leaky_features` list before training LGBM/XGB, the `is_top_venue` and `venue_tier` features are **not dropped**. 
**Impact:** Because these features rely on future prestige (median citations across all time), the 2020 training set has access to information about which venues will be popular in 2024. This leaks future knowledge directly into the training data.

### Issue 2: Temporal Leakage in Negative Sampling (LGBM / XGBoost)

**Problem:** The `sample_negatives_fast` function generates fake citation pairs by picking a random source node and a random destination node within the temporal split. However, it **does not enforce that the source paper is published after or in the same year as the destination paper (`src_year >= dst_year`)**. 
**Impact:** A real paper cannot cite a paper published after it. By generating temporally impossible pairs (e.g., a 2010 paper citing a 2018 paper) and keeping features like `decade` and `is_recent`, the LGBM/XGB models can trivially identify fake pairs simply by noting that `src_year < dst_year`. This makes the classification task artificially easy for half the dataset and falsely inflates ROC-AUC and PR-AUC.

### Issue 3: Unnecessary Dropping of Valid Features

**Problem:** The `leaky_features` list drops `years_since_publication` and `n_references`. 
**Impact:** `n_references` is the count of *outgoing* references from the source paper. This is strictly known at the exact moment of publication and is perfectly valid for prediction. Dropping it discards a highly predictive and completely legitimate feature.

### Issue 4: Conceptual Disconnect (Concatenated Nodes vs. Pair Features)

**Problem:** For the Link Prediction task (does A cite B?):
- **LGBM and XGBoost** are trained by simply concatenating 25 flat features from Paper A and 25 flat features from Paper B. Tree-based models are historically very poor at inferring pairwise interactions (like Jaccard similarity or exact venue matches) from flat concatenated arrays.
- **Random Forest** is trained using explicitly engineered pair-level features (`year_gap`, `author_jaccard`, `same_venue`).
**Impact:** LGBM/XGB are fundamentally handicapped conceptually compared to the Random Forest, making any comparison between them structurally flawed. A proper tabular baseline for link prediction should compute explicit pairwise similarities for *all* tree models.

### Issue 5: Unfair Model Comparison and Inconsistent Splits

**Problem:** The project checklist strictly requires a "fair comparison" between models. Currently:
- **LGBM/XGBoost** are trained on ~3.2M pairs using a strict and correct **temporal split** (Train $\le$ 2020, Test $\ge$ 2023).
- **Random Forest** is trained on exactly 10,000 pairs using a **random split** (`train_test_split` with 80/20 proportion). 
**Impact:** The Random Forest suffers from severe temporal leakage because a random split allows future pairs (e.g., from 2023) to leak into the training set, giving the model access to future citation behavior. Because the datasets, features, and split strategies are entirely different, it is impossible to compare these models fairly.

### Proposed Experimental Design (Flat vs. Pair Features)

To completely resolve the conceptual disconnect (Issue 4) and provide a bulletproof "fair comparison" (Issue 5), we propose a two-phase modeling approach:
1. **Phase 1 (Flat Features):** Train all three models (LGBM, XGBoost, Random Forest) using only the concatenated flat node features.
2. **Phase 2 (Pair Features):** Train all three models using exclusively explicitly engineered pair-level features (e.g., Jaccard similarities, year gaps).

**Why this works:** This design explicitly tests and proves a major machine learning concept—that tree-based models struggle to infer relationships from flat arrays but excel when given explicit relationship features. For this to be valid, all models in both phases must use the exact same strict temporal split, the exact same subset of data, and the exact same time-respecting negative sampling.

### Recommended Fixes (Priority Order)

1. **Fix Negative Sampling & Filtering:** Update `sample_negatives_fast` to strictly enforce `src_year >= dst_year` for all generated fake pairs. Furthermore, abandon purely random sampling across the entire 6.7M dataset to prevent trivially easy fake pairs. Use one of these strategies instead:
    *   **Domain/Venue Filtering:** Filter the dataset to a specific domain (e.g., top CS venues) *before* creating edges or sampling.
    *   **Snowball / Subgraph Sampling:** Pick a seed set of related papers and extract all papers 1-hop away in the citation graph to create a dense, connected neighborhood.
    *   **Hard Negative Sampling:** Force the negative generator to pick a paper from the *same year* or *same venue* as the true citation, but which wasn't actually cited, to force the model to learn subtle boundaries.
2. **Implement the 2-Phase Experimental Design:** Train all 3 models on the flat features, then train all 3 on the explicit pair-level dataset.
3. **Unify Splits (Temporal for All):** Remove the random `train_test_split` from the Random Forest pipeline. Enforce the `Train <= 2020`, `Val 2021-2022`, `Test >= 2023` temporal split strictly across all three models.
4. **Fix Global Statistics Leakage:** Explicitly drop `is_top_venue` and `venue_tier` from the training features, or calculate them *only* using citations accrued up to the year 2020.
5. **Restore Valid Features:** Remove `n_references` and `years_since_publication` from the `leaky_features` drop list.
